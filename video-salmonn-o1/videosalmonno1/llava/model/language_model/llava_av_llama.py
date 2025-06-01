#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Modified by Tang


from typing import List, Optional, Tuple, Union
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import torch
import torch.nn as nn
import math
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
import ast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from llava.mm_utils import get_anyres_image_grid_shape
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, unpad_image
from llava.model.multimodal_encoder.modeling_whisper import WhisperModel
from llava.model.multimodal_encoder.beats.BEATs import BEATsConfig, BEATs
from llava.model.multimodal_projector.Qformer import BertConfig, BertLMHeadModel
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

class LlavaAVConfig(LlamaConfig):
    model_type = "llava_av_llama"


class LlavaAVLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaAVConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaAVLlamaModel, self).__init__(config)

def init_speech_Qformer(num_query_token, speech_width, num_hidden_layers=2):
    encoder_config = BertConfig.from_pretrained("bert-base-uncased")
    encoder_config.num_hidden_layers = num_hidden_layers
    encoder_config.encoder_width = speech_width
    # insert cross-attention layer every other block
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = 1
    encoder_config.query_length = num_query_token
    Qformer = BertLMHeadModel(config=encoder_config)
    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size)
    )
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return Qformer, query_tokens


class LlavaAVLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaAVConfig

    def __init__(self, config, **audio_config):
        
        LlamaForCausalLM.__init__(self, config)
        self.model = LlavaAVLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.whisper_path = audio_config.get("whisper_path", "")
        self.freeze_whisper = audio_config.get("freeze_whisper", True)
        self.beats_path = audio_config.get("beats_path", "")
        self.use_speech_Qformer = audio_config.get("use_speech_Qformer", True)
        self.num_speech_query_token = audio_config.get("num_speech_query_token", 1)
        self.freeze_speech_QFormer = audio_config.get("freeze_speech_QFormer", False)
        self.window_level_Qformer = audio_config.get("window_level_Qformer", True)
        self.second_per_window = audio_config.get("second_per_window", 0.333333)
        self.second_stride = audio_config.get("second_stride", 0.333333)
        self.salmonn_path = audio_config.get("salmonn_path", "")

        self.speech_encoder = WhisperModel.from_pretrained(self.whisper_path).encoder
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
        if self.freeze_whisper:
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            self.speech_encoder.eval()

        ext_total_dim = 0
        if self.beats_path:
            beats_ckpt = torch.load(self.beats_path, map_location='cpu')
            beats_cfg = BEATsConfig(beats_ckpt['cfg'])
            # import pdb; pdb.set_trace()
            self.beats = BEATs(beats_cfg)
            self.beats.load_state_dict(beats_ckpt['model'])
            self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
            if self.freeze_beats:
                for name, param in self.beats.named_parameters():
                    param.requires_grad = False
                self.beats.eval()

        if self.use_speech_Qformer:
            self.speech_Qformer, self.speech_query_tokens = init_speech_Qformer(
                num_query_token=self.num_speech_query_token, speech_width=self.speech_encoder.config.d_model + ext_total_dim
            )
            self.speech_Qformer.bert.embeddings.word_embeddings = None
            self.speech_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.speech_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.speech_Qformer.cls = None
            if self.freeze_speech_QFormer:
                for name, param in self.speech_Qformer.named_parameters():
                    param.requires_grad = False
                self.speech_Qformer.eval()
                self.speech_query_tokens.requires_grad = False

            self.speech_llama_proj = nn.Linear(
                self.speech_Qformer.config.hidden_size, config.hidden_size
            )

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def _encode_auditory_feature(self, speech_embeds, audio_embeds=None):
        # with self.maybe_autocast():
        if self.use_speech_Qformer:
            speech_embeds = self.ln_speech(speech_embeds)
            if audio_embeds is not None:
                audio_embeds = self.ln_audio(audio_embeds)
                if audio_embeds.size(1) < speech_embeds.size(1):
                    audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
                elif audio_embeds.size(1) > speech_embeds.size(1):
                    speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
                speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

            if self.window_level_Qformer:
                B, T, C = speech_embeds.shape
                kernel = round(1500 * self.second_per_window / 30.0)
                stride = round(1500 * self.second_stride / 30.0)
                kernel = (1, kernel)
                stride = (1, stride)
                speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
                speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
                _, _, L = speech_embeds_overlap.shape
                speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
                speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
                speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

            query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
            query_output = self.speech_Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=speech_embeds,
                encoder_attention_mask=speech_atts,
                return_dict=True,
            )
            speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)

            if self.window_level_Qformer:
                speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()

            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
        else:
            raise NotImplementedError

        return speech_embeds, speech_atts
    
    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        # with self.maybe_autocast():
        speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state

        if self.beats_path and raw_wav is not None:
            audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)
        else:
            audio_embeds = None

        return self._encode_auditory_feature(speech_embeds, audio_embeds=audio_embeds)
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, modalities, image_sizes=None, prompts=None, raw_wav=None, spectrogram=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if torch.cuda.current_device()==0:
            print('>>> [RANK0 PRINT] | modalities in batch:', modalities)

        if isinstance(modalities, str):
            modalities = [modalities]

        # Audio
        audio_padding_mask = None
        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)
        frame_per_second = 1

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] in ["video"]:
                    video_idx_in_batch.append(_)


            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            image_features = self.encode_images(concat_images, video_idx_in_batch, split_sizes) # [v_ts1, v_ts2, ...]
            
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            new_image_features = []

            # breakpoint()
            if mm_patch_merge_type == 'flat':
                for image_idx, image_feature in enumerate(image_features):
                    new_image_features.append(image_feature.flatten(0, 1))
                image_features = new_image_features
                
            elif mm_patch_merge_type.startswith('spatial'):
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)

                    if image_feature.shape[0] > 1:
                        if image_idx in video_idx_in_batch:
                            # breakpoint()
                            if self.config.mm_newline_position == "grid":
                                # Grid-wise
                                resize_h = int(math.sqrt(image_feature.shape[1]))
                                num_frames = image_feature.shape[0]
                                image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)

                                i = image_idx
                                image_feature_i = image_feature
                                image_feature_i = image_feature_i.view(image_features[i].size(0), -1, image_feature_i.size(-1))
                                if image_feature_i.size(0) > 30:
                                    image_feature_i = image_feature_i[:30, :, :]

                                speech_embeds_i = speech_embeds[i]
                                if (ip_delta := speech_embeds_i.size(0) % round(30 * frame_per_second)) != 0:
                                    after_interpolate_num = speech_embeds_i.size(0) + round(30 * frame_per_second) - ip_delta
                                speech_embeds_i = speech_embeds_i.unsqueeze(0)
                                speech_embeds_i = F.interpolate(speech_embeds_i.transpose(1, 2), size=after_interpolate_num, mode='nearest').transpose(1, 2).squeeze()
                                speech_embeds_i_alignv = speech_embeds_i.view(round(30 * frame_per_second), -1, speech_embeds_i.size(-1))

                                speech_embeds_i_alignv_front = speech_embeds_i_alignv[:image_feature_i.size(0), :, :]
                                speech_embeds_i_alignv_back = speech_embeds_i_alignv[image_feature_i.size(0):, :, :].view(-1, speech_embeds_i_alignv.size(-1))

                                av_embeds_i_alignv = torch.cat([image_feature_i, speech_embeds_i_alignv_front], dim=1)
                                av_embeds_i_alignv = av_embeds_i_alignv.view(-1, av_embeds_i_alignv.size(-1))
                                av_embeds_i = torch.cat([av_embeds_i_alignv, speech_embeds_i_alignv_back], dim=0)

                                new_image_features.append(av_embeds_i)

                            elif self.config.mm_newline_position == "frame":
                                # Frame-wise
                                image_feature = image_feature.permute(2, 0, 1).contiguous()
                                image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                                image_feature = image_feature.permute(1, 2, 0).contiguous()
                                new_image_features.append(image_feature.flatten(0, 1))
                            elif self.config.mm_newline_position == "one_token":
                                # one-token
                                image_feature = image_feature.flatten(0, 1)
                                if 'unpad' in mm_patch_merge_type:
                                    image_feature = torch.cat((
                                        image_feature,
                                        self.model.image_newline[None].to(image_feature.device)
                                    ), dim=0)
                                new_image_features.append(image_feature)      
                            elif self.config.mm_newline_position == "no_token":
                                new_image_features.append(image_feature.flatten(0, 1))
                            else:
                                raise ValueError(f"Unexpected mm_newline_position: {self.config.mm_newline_position}")

                            continue

                        # breakpoint()
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            except:
                                breakpoint() 
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)
                        if 'maxpool2x2' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if 'nobase' in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)

                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError
        
        # av_interleave_features = []
        # frame_per_second = 1
        # for i in len(image_features):
        #     image_feature_i = image_features[i]
        #     image_feature_i = image_feature_i.view(-1, 144, image_feature_i.size(-1))
        #     speech_embeds_i = speech_embeds[i]
        #     if (ip_delta := speech_embeds_i.size(0) % round(30 * frame_per_second)) != 0:
        #         after_interpolate_num = speech_embeds_i.size(0) + round(30 * frame_per_second) - ip_delta
        #     speech_embeds_i = F.interpolate(speech_embeds_i.transpose(0, 1), size=after_interpolate_num, mode='nearest').transpose(0, 1)
        #     speech_embeds_i_alignv = speech_embeds_i.view(round(30 * frame_per_second), -1, speech_embeds_i_alignv.size(-1))

        #     speech_embeds_i_alignv_front = speech_embeds_i_alignv[:image_feature_i.size(0), :, :]
        #     speech_embeds_i_alignv_back = speech_embeds_i_alignv[image_feature_i.size(0):, :, :].view(-1, speech_embeds_i_alignv.size(-1))

        #     av_embeds_i_alignv = torch.cat([image_feature_i, speech_embeds_i_alignv_front], dim=0)
        #     av_embeds_i_alignv = av_embeds_i_alignv.view(-1, av_embeds_i_alignv.size(-1))
        #     av_embeds_i = torch.cat([av_embeds_i_alignv, speech_embeds_i_alignv_back], dim=0)
        #     av_interleave_features.append(av_embeds_i)

        # image_features = av_interleave_features

        # breakpoint()
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # breakpoint()
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        modality_max_length = getattr(self.config, 'modality_max_length', None)

        
        if modality_max_length is None or modality_max_length == "None":
            if tokenizer_model_max_length is not None:
                new_input_embeds =[x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
                new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        else:
            # breakpoint()
            modality_max_length = ast.literal_eval(modality_max_length)
            modality_max_length_dict = {"image": modality_max_length[0], "text": modality_max_length[1], "video": modality_max_length[2]}
            new_input_embeds =[x[: modality_max_length_dict[modality]] for x, modality in zip(new_input_embeds, modalities)]
            new_labels = [x[: modality_max_length_dict[modality]] for x, modality in zip(new_labels, modalities)]

        # TODO: Hard code for control loss spike                  
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # breakpoint()
        if torch.cuda.current_device()==0:
            print('>>> [RANK0 PRINT] | batch new_input_embeds\' shape:', new_input_embeds.shape)
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        prompts: Optional[List[str]] = None,
        modalities: Optional[List[str]] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[bool] = None,
        raw_wav=None, 
        spectrogram=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                modalities,
                image_sizes,
                prompts,
                raw_wav=raw_wav,
                spectrogram=spectrogram,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        raw_wav=None, 
        spectrogram=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        modalities = kwargs.pop("modalities", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                modalities,
                image_sizes=image_sizes,
                raw_wav=raw_wav, 
                spectrogram=spectrogram,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

# if LlavaConfig.model_type == "llava":
#     LlavaConfig.model_type = "llava_av_llama" # directly set to llava_dev to avoid conflict with HF's llava
    
AutoConfig.register("llava_av_llama", LlavaAVConfig)
AutoModelForCausalLM.register(LlavaAVConfig, LlavaAVLlamaForCausalLM)
