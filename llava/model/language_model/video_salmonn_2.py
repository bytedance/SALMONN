# Copyright (2025) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/llava/model/language_model/llava_qwen.py. Copyright 2024 Hao Zhang. The original license is located at 'third-party-license/llava_next.txt'.
# Adapted from https://github.com/bytedance/SALMONN. The original license is located at 'third-party-license/salmonn.txt'.

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import math
from llava.model.multimodal_encoder.modeling_whisper import WhisperModel
import ast
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from llava.mm_utils import get_anyres_image_grid_shape
from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, unpad_image
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.model.multimodal_projector.Qformer import BertConfig, BertLMHeadModel

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

class VideoSALMONN2Config(Qwen2Config):
    model_type = "video_salmonn_2"

class VideoSALMONN2Model(LlavaMetaModel, Qwen2Model):
    config_class = VideoSALMONN2Config

    def __init__(self, config: Qwen2Config):
        super(VideoSALMONN2Model, self).__init__(config)

        self.has_init_vm = False

    def initialize_vision_modules(self, model_args, fsdp=None):
        if self.has_init_vm:
            return
        self.has_init_vm = True
        vision_tower = model_args.vision_tower
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            # vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            ## Get the mm_spatial_pool_mode and  mm_spatial_pool_stride
            # for k, v in vision_resampler.config.items():
            #     setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        # self.config.mm_hidden_size = getattr(vision_resampler, 'hidden_size', vision_tower.hidden_size)
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            print(self.mm_projector.load_state_dict(mm_projector_weights))

class VideoSALMONN2ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = VideoSALMONN2Config

    def __init__(self, config, **audio_config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "video_salmonn_2"
        config.rope_scaling = None
        self.vocab_size = config.vocab_size

        self.add_time_token = config.add_time_token

        self.model = VideoSALMONN2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.audio_visual = audio_config.get("audio_visual", False)
        self.video_fps = audio_config.get("video_fps", 1)

        if self.audio_visual:
            self.whisper_path = audio_config.get("whisper_path", "")
            self.num_speech_query_token = audio_config.get("num_speech_query_token", 1)
            self.window_level_Qformer = audio_config.get("window_level_Qformer", True)
            self.second_per_window = audio_config.get("second_per_window", 0.333333)
            self.second_stride = audio_config.get("second_stride", 0.333333)
            self.use_final_linear = audio_config.get("use_final_linear", False)

            self.speech_encoder = WhisperModel.from_pretrained(self.whisper_path).encoder
            self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)

            self.speech_Qformer, self.speech_query_tokens = init_speech_Qformer(num_query_token=self.num_speech_query_token, speech_width=self.speech_encoder.config.d_model)
            self.speech_Qformer.bert.embeddings.word_embeddings = None
            self.speech_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.speech_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.speech_Qformer.cls = None

            self.speech_llama_proj = nn.Linear(
                self.speech_Qformer.config.hidden_size, config.hidden_size
            )

            if self.use_final_linear:
                self.final_linear = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size)
                )

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def encode_images(self, images, video_idx_in_batch=[], split_sizes=None):
        image_features = self.get_model().get_vision_tower()(images)

        if split_sizes is None:
            split_sizes = [1 for image in images]
        per_image_features = torch.split(image_features, split_sizes, dim=0) # tuple, (dim_1, 576, 4096)
        all_image_features = []

        for idx, img_feat in enumerate(per_image_features):

            if self.config.mm_pooling_position == "before":
                if idx in video_idx_in_batch and self.config.mm_spatial_pool_stride > 1:
                    img_feat = self.get_2dPool(img_feat) # (num_vid*num_frames, 576, 4096) -> (num_vid*num_frames, 144, 4096)

            img_feat = self.get_model().mm_projector(img_feat) # (dim_1_sum, 576, 1024) -> (dim_1_sum, 576, 4096)

            if self.config.mm_pooling_position == "after":
                if idx in video_idx_in_batch and self.config.mm_spatial_pool_stride > 1:
                    img_feat = self.get_2dPool(img_feat) # (num_vid*num_frames, 576, 4096) -> (num_vid*num_frames, 144, 4096)

            all_image_features.append(img_feat)
        return all_image_features

    def _encode_auditory_feature(self, speech_embeds):
        # with self.maybe_autocast():
        if True:
            speech_embeds = self.ln_speech(speech_embeds)
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
            if self.use_final_linear:
                speech_embeds = self.final_linear(speech_embeds)

            if self.window_level_Qformer:
                speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()

            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

        return speech_embeds, speech_atts
    
    def encode_speech(self, spectrogram):
        speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state
        return self._encode_auditory_feature(speech_embeds)

    def sinusoidal_position(self, max_len, d_model):
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pos_enc = torch.zeros((max_len + 1, d_model))   
        pos_enc[1:, 0::2] = torch.sin(position * div_term)
        pos_enc[1:, 1::2] = torch.cos(position * div_term)
        return pos_enc
   
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, modalities, spectrogram=None, org_groups=None, real_time=None,
    ):
        vision_tower = self.get_vision_tower()
        # if vision_tower is None or images is None or input_ids.shape[1] == 1:
        if vision_tower is None or (not isinstance(input_ids, list) and input_ids.shape[1] == 1):
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if torch.cuda.current_device() == 0:
            print('>>> [RANK0 PRINT] | modalities in batch:', modalities)

        if isinstance(modalities, str):
            modalities = [modalities]

        # Audio
        if spectrogram is not None and "audio" in modalities[0]:
            speech_embeds, speech_atts = self.encode_speech(spectrogram)
            tmp_idx = 0
            new_speech_embeds = []

            seg_position = self.sinusoidal_position(200, speech_embeds.shape[-1]).to(speech_embeds.device).to(speech_embeds.dtype)
            if org_groups is not None:
                for speech_len in org_groups:
                    speech_groups = speech_embeds[tmp_idx: tmp_idx + speech_len]
                    speech_groups = speech_groups + seg_position[:speech_len].unsqueeze(1)
                    new_speech_embeds.append(speech_groups.view(-1, speech_embeds.size(-1)))
                    tmp_idx = tmp_idx + speech_len

            speech_embeds = new_speech_embeds

        if images is not None and (type(images) is list or images.ndim == 5) and not all([m == 'audio' for m in modalities]) :
            if type(images) is list:
                # images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
                images_w_none = []
                temp_cnt_idx = 0
                video_idx_in_batch_wo_none = []
                for k, x in enumerate(images):
                    if x is not None:
                        images_w_none.append(x.unsqueeze(0) if x.ndim == 3 else x)
                        if modalities[k] in ["video", "audio-video"]:
                            video_idx_in_batch_wo_none.append(temp_cnt_idx)
                        temp_cnt_idx += 1
                    else:
                        images_w_none.append(None)

                images = [item for item in images_w_none if item is not None]

            else:
                print(type(images), images.shape, modalities)

            video_idx_in_batch = []
            for modality in range(len(modalities)):
                if modalities[modality] in ["video", "audio-video"]:
                    video_idx_in_batch.append(modality)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            image_features = self.encode_images(concat_images, video_idx_in_batch_wo_none, split_sizes)

            image_features_w_none = []
            temp_cnt_idx = 0
            for ori_image in images_w_none:
                if ori_image is not None:
                    image_features_w_none.append(image_features[temp_cnt_idx])
                    temp_cnt_idx += 1
                else:
                    image_features_w_none.append(None)
            image_features = image_features_w_none

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            new_image_features = []

            if mm_patch_merge_type.startswith('spatial'):
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)

                    if image_feature is None:
                        new_image_features.append(speech_embeds[image_idx])
                        continue

                    if image_feature.shape[0] > 1:
                        if image_idx in video_idx_in_batch:
                            if self.config.mm_newline_position == "grid":
                                # Grid-wise
                                i = image_idx

                                resize_h = int(math.sqrt(image_feature.shape[1]))
                                num_frames = image_feature.shape[0]
                                image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                                image_feature_i = image_feature
                                image_feature_i = image_feature_i.view(image_features[i].size(0), -1, image_feature_i.size(-1))
                                    
                                if self.add_time_token:
                                    frame_per_time = real_time[image_idx] / num_frames
                                    time_idx = [str(round(frame_per_time * f_idx, 1)) for f_idx in range(1, num_frames + 1)]
                                    time_tokens = [self.tokenizer(t_idx, return_tensors='pt')["input_ids"].to(self.device) for t_idx in time_idx]
                                    time_embeds = [self.get_model().embed_tokens(t_tok).squeeze() for t_tok in time_tokens]
                                    padded_time_embeds = pad_sequence(time_embeds, batch_first=True)
                                    image_feature = image_feature.view(num_frames, -1, image_feature.size(-1))
                                    image_feature = torch.cat((image_feature, padded_time_embeds), dim=1)
                                    image_feature = image_feature.view(-1, image_feature.size(-1))

                                if spectrogram is not None and "audio" in modalities[i]:
                                    if real_time is None:
                                        sample_time = image_feature_i.size(0) * self.video_fps
                                    else:
                                        sample_time = real_time[image_idx]

                                    speech_embeds_i = speech_embeds[i]
                                    speech_embeds_i_front_num = round(sample_time * 50)
                                    speech_embeds_i_front = speech_embeds_i[:speech_embeds_i_front_num, :]
                                    speech_embeds_i_back = speech_embeds_i[speech_embeds_i_front_num:, :]

                                    if (ip_delta := speech_embeds_i_front.size(0) % image_feature_i.size(0)) != 0:
                                        after_interpolate_num = speech_embeds_i_front.size(0) + image_feature_i.size(0) - ip_delta
                                        speech_embeds_i_front = speech_embeds_i_front.unsqueeze(0)
                                        speech_embeds_i_front = F.interpolate(speech_embeds_i_front.transpose(1, 2), size=after_interpolate_num, mode='nearest').transpose(1, 2).squeeze()
                                    
                                    speech_embeds_i_alignv_front = speech_embeds_i_front.view(image_feature_i.size(0), -1, speech_embeds_i_front.size(-1))
                                    speech_embeds_i_alignv_back = speech_embeds_i_back

                                    av_embeds_i_alignv = torch.cat([image_feature_i, speech_embeds_i_alignv_front], dim=1)
                                    av_embeds_i_alignv = av_embeds_i_alignv.view(-1, av_embeds_i_alignv.size(-1))
                                    av_embeds_i = torch.cat([av_embeds_i_alignv, speech_embeds_i_alignv_back], dim=0)

                                    new_image_features.append(av_embeds_i)
                                else:
                                    new_image_features.append(image_feature)
                            else:
                                raise ValueError(f"Unexpected mm_newline_position: {self.config.mm_newline_position}")
                            
                    else:
                        raise NotImplementedError

                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            
        elif images is None or all([m == 'audio' for m in modalities]):
            image_features = speech_embeds
        else:
            image_features = self.encode_images(images)

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
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.bool()
        # if position_ids is None:
        #     position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

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
                new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
                new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        else:
            modality_max_length = ast.literal_eval(modality_max_length)
            modality_max_length_dict = {"image": modality_max_length[0], "text": modality_max_length[1], "video": modality_max_length[2]}
            new_input_embeds =[x[: modality_max_length_dict[modality]] for x, modality in zip(new_input_embeds, modalities)]
            new_labels = [x[: modality_max_length_dict[modality]] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=new_labels_padded.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=attention_mask.device)

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
            attention_mask = attention_mask.to(dtype=new_input_embeds.dtype)

        if _position_ids is None:
            position_ids = None

        if torch.cuda.current_device() == 0:
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
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        spectrogram=None,
        org_groups=None,
        real_time=None,
        reject_input_ids=None,
        reject_labels=None,
        reject_attention_mask=None,
        gt_input_ids=None,
        gt_labels=None,
        gt_attention_mask=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if dpo_forward:
            (input_ids, position_ids_v1, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, spectrogram=spectrogram, org_groups=org_groups, real_time=real_time)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids_v1,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,   
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)

            (reject_input_ids, position_ids_v2, reject_attention_mask, reject_past_key_values, reject_inputs_embeds, reject_labels) = self.prepare_inputs_labels_for_multimodal(reject_input_ids, position_ids, reject_attention_mask, past_key_values, reject_labels, images, modalities, spectrogram=spectrogram, org_groups=org_groups, real_time=real_time)

            reject_outputs = self.model(
                input_ids=reject_input_ids,
                attention_mask=reject_attention_mask,
                position_ids=position_ids_v2,
                past_key_values=reject_past_key_values,
                inputs_embeds=reject_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,   
            )

            reject_hidden_states = reject_outputs[0]
            reject_logits = self.lm_head(reject_hidden_states)

            return logits, labels, reject_logits, reject_labels, self.vocab_size

        else:
            if gt_input_ids is None or gt_labels is None or gt_attention_mask is None:
                gt_input_ids = input_ids
                gt_labels = labels
                gt_attention_mask = attention_mask
            if inputs_embeds is None:
                (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(gt_input_ids, position_ids, gt_attention_mask, past_key_values, gt_labels, images, modalities, spectrogram=spectrogram, org_groups=org_groups, real_time=real_time)
            
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
            )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        spectrogram=None,
        org_groups=None,
        real_time=None,
        max_new_tokens=1024,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or spectrogram is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, None, None, images, modalities, spectrogram=spectrogram, org_groups=org_groups, real_time=real_time)
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens, **kwargs)
    
AutoConfig.register("video_salmonn_2", VideoSALMONN2Config)
AutoModelForCausalLM.register(VideoSALMONN2Config, VideoSALMONN2ForCausalLM)
