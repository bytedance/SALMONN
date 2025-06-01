#    Copyright 2024 Hao Zhang
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


import copy
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM, AutoModel, AutoTokenizer, AutoProcessor

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, unpad_image
# from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from llava.model.modeling_qwen2 import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from llava.mm_utils import get_anyres_image_grid_shape

from llava.model.language_model.llava_av_llama import init_speech_Qformer
# from llava.model.multimodal_encoder.delta_embedding_predictor import DeltaPredictor

import math

from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_resampler.builder import build_vision_resampler
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.model.multimodal_projector.mfcnn import MFCNN
from llava.model.multimodal_projector.mftrans import MFTrans
from llava.model.multimodal_projector.transposelinear import SpatialTemporalProj

import random
import re

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is"
        "The correct option is",
        "Best answer:"
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
        "The final answer is:\n"
    ]
    for answer_prefix in answer_prefixes:
        s = s.split(answer_prefix)[-1]
        # s = s.replace(answer_prefix, "")
    if s == "":
        return s
    if s[0].lower() == s[0]:
        s = s[0].upper() + s[1:]
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ""
    return matches[0]


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        # if hasattr(config, "mm_vision_tower"):
        #     del config.mm_vision_tower
        super(LlavaQwenModel, self).__init__(config)
        self.multi_frame_projector = hasattr(config, "multi_frame_projector") and config.multi_frame_projector
        self.spt_projector = hasattr(config, "spt_projector") and config.spt_projector
        assert not (self.multi_frame_projector and self.spt_projector)

        self.multi_frame_num = hasattr(config, "multi_frame_num") and config.multi_frame_num
        self.mf_split_init = hasattr(config, "mf_split_init") and config.mf_split_init
        self.use_mfcnn = hasattr(config, "use_mfcnn") and config.use_mfcnn

        self.has_init_vm = False

        # from dataclasses import make_dataclass
        # TempData = make_dataclass("TempData", config.model_args.keys())
        # model_args = TempData(**config.model_args)
        # self.initialize_vision_modules(model_args)
        

    def initialize_vision_modules(self, model_args, fsdp=None):
        if self.has_init_vm:
            return
        self.has_init_vm = True
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            ## Get the mm_spatial_pool_mode and  mm_spatial_pool_stride
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            if self.vision_resampler is not None:
                for p in self.vision_resampler.parameters():
                    p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = getattr(vision_resampler, 'hidden_size', vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None or self.multi_frame_projector or self.spt_projector:
            if False:
                pass
            elif self.multi_frame_projector or self.spt_projector:
                if getattr(self, 'mm_projector', None) is not None and self.mf_split_init:
                    if self.multi_frame_projector:
                        mf_projector = nn.Sequential(
                            nn.Linear(self.config.mm_hidden_size * self.multi_frame_num, self.config.hidden_size * self.multi_frame_num),
                            nn.GELU(),
                            nn.Linear(self.config.hidden_size * self.multi_frame_num, self.config.hidden_size)
                        )
                        mm_projector = self.mm_projector

                        for i in range(self.multi_frame_num):
                            mf_projector[0].weight.data[i * self.config.hidden_size: (i + 1) * self.config.hidden_size, i * self.config.mm_hidden_size: (i + 1) * self.config.mm_hidden_size] = mm_projector[0].weight.data
                            mf_projector[0].bias.data[i * self.config.hidden_size: (i + 1) * self.config.hidden_size] = mm_projector[0].bias.data / self.multi_frame_num

                        for i in range(self.multi_frame_num):
                            mf_projector[2].weight.data[:, i * self.config.hidden_size: (i + 1) * self.config.hidden_size] = mm_projector[2].weight.data / self.multi_frame_num
                        mf_projector[2].bias.data = mm_projector[2].bias.data
                        self.mm_projector = mf_projector

                    elif self.spt_projector:
                        mf_projector = SpatialTemporalProj(self.config)
                        mm_projector = self.mm_projector
                        for i in range(self.multi_frame_num):
                            mf_projector.linear2.weight.data[i * self.config.hidden_size: (i + 1) * self.config.hidden_size, i * self.config.mm_hidden_size: (i + 1) * self.config.mm_hidden_size] = mm_projector[0].weight.data
                            mf_projector.linear2.bias.data[i * self.config.hidden_size: (i + 1) * self.config.hidden_size] = mm_projector[0].bias.data / self.multi_frame_num

                        for i in range(self.multi_frame_num):
                            mf_projector.linear3.weight.data[:, i * self.config.hidden_size: (i + 1) * self.config.hidden_size] = mm_projector[2].weight.data / self.multi_frame_num
                        mf_projector.linear3.bias.data = mm_projector[2].bias.data
                        self.mm_projector = mf_projector

                    print("Matrix Split Initialize")

                else:
                    if self.multi_frame_projector:
                        self.mm_projector = nn.Sequential(
                            nn.Linear(self.config.mm_hidden_size * self.multi_frame_num, self.config.hidden_size * self.multi_frame_num),
                            nn.GELU(),
                            nn.Linear(self.config.hidden_size * self.multi_frame_num, self.config.hidden_size)
                        )
                    elif self.spt_projector:
                        self.mm_projector = SpatialTemporalProj(self.config)
            else:
                self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if 'unpad' in mm_patch_merge_type and not self.use_mfcnn:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            ## Lagacy: at the very beginging, image_newline is not initialized in the stage_1_5
            # if model_args.from_stage_1_5:
            #     if 'unpad' in mm_patch_merge_type:
            #         embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
            #         self.image_newline = nn.Parameter(
            #             torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
            #         )
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            print(self.mm_projector.load_state_dict(mm_projector_weights))



class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config, **audio_config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)

        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.add_time_token = config.add_time_token
        self.use_mfcnn = config.use_mfcnn
        self.use_mftrans = config.use_mftrans
        if self.use_mfcnn:
            self.mfcnn = MFCNN()
        elif self.use_mftrans:
            self.mftrans = MFTrans()

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.segmentation = -1
        self.do_rag = False
        if "do_rag" in config.model_args:
            self.do_rag = config.model_args["do_rag"]
            self.rag_type = config.model_args["rag_type"]
        if "segmentation" in config.model_args:
            self.segmentation = config.model_args["segmentation"]
            self.rag_input_frames = config.model_args["rag_input_frames"]
        if self.do_rag:
            self.rag_dropout = nn.Dropout(0.1)
            if "text" in self.rag_type:
                if "video" not in self.rag_type or "both" in self.rag_type:
                    self.text_encoder = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-1.5B-instruct')
                    self.text_tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-1.5B-instruct')
                if "video" in self.rag_type:
                    self.text_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", trust_remote_code=True)
                    if "train" in self.rag_type:
                        self.video_encoder = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").train()
                self.rag_topk = config.model_args["rag_topk"]
            else:
                # keydim = 128
                self.segmentTFM = nn.MultiheadAttention(config.hidden_size, 1, batch_first=True)
                self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
                # self.key_proj = nn.Linear(config.hidden_size, keydim)
                # self.do_rag_layer = nn.Linear(config.hidden_size, 1, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def encode_images(self, images, video_idx_in_batch=[], split_sizes=None):
        if self.get_model().multi_frame_projector or self.get_model().spt_projector or self.use_mfcnn or self.use_mftrans:
            step = 100
            image_features = []
            for i in range(0, len(images), step):
                image_features_i = self.get_model().get_vision_tower()(images[i: i + step])
                image_features.append(image_features_i)
            image_features = torch.cat(image_features, dim=0)

            if self.use_mfcnn:
                img_feat = self.mfcnn(image_features)
                img_feat = self.get_model().mm_projector(img_feat)
            
            else:
                if self.use_mftrans:
                    img_feat = image_features
                    if self.config.mm_pooling_position == "before":
                        img_feat = self.get_2dPool(img_feat)
                    
                    img_feat = self.mftrans(img_feat)
                    img_feat = self.get_model().mm_projector(img_feat)

                    if self.config.mm_pooling_position == "after":
                        img_feat = self.get_2dPool(img_feat)

                    img_feat = img_feat.reshape(-1, 169, img_feat.size(-1))
                    
                else:
                    tail_mf_num = self.get_model().multi_frame_num - image_features.size(0) % self.get_model().multi_frame_num
                    if tail_mf_num > 0:
                        image_features_tail = torch.zeros([tail_mf_num, image_features.size(1), image_features.size(2)], device=image_features.device, dtype=image_features.dtype)
                        image_features = torch.cat([image_features, image_features_tail], dim=0)

                
                    image_features = image_features.reshape(-1, self.get_model().multi_frame_num, image_features.size(-2), image_features.size(-1))
                    image_features = image_features.transpose(1, 2)
                    image_features = image_features.reshape(image_features.size(0), image_features.size(1), -1)

                    img_feat = image_features

                    if self.config.mm_pooling_position == "before":
                        img_feat = self.get_2dPool(img_feat) # (num_vid*num_frames, 576, 4096) -> (num_vid*num_frames, 144, 4096)
                    
                    img_feat = self.get_model().mm_projector(img_feat) # (dim_1_sum, 576, 1024) -> (dim_1_sum, 576, 4096)

                    if self.config.mm_pooling_position == "after":
                        img_feat = self.get_2dPool(img_feat) # (num_vid*num_frames, 576, 4096) -> (num_vid*num_frames, 144, 4096)

            return [img_feat]
        else:
            if images.size(0) > 200:
                image_features = []
                split = math.ceil(images.size(0) / 200)
                for i in range(split):
                    image_features.append(self.get_model().get_vision_tower()(images[i*200:(i+1)*200]))
                image_features = torch.cat(image_features, dim=0)
            else:
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

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, modalities, image_sizes=None, prompts=None, real_time=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None

        if isinstance(modalities, str):
            modalities = [modalities]

        # if torch.cuda.current_device()==0:
        #     print(f'[RANK0 PRINT] | Modality Check: {modalities}')

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] in ["video", "audio-video"]:
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            image_features = []
            stepsize = 500
            index = 0
            while index < len(concat_images):
                split_sizes = [len(concat_images[index:index+stepsize])]
                sub_features = self.encode_images(concat_images[index:index+stepsize], video_idx_in_batch, split_sizes)
                index = index + stepsize
                image_features.append(sub_features[0])
            # image_features = self.encode_images(concat_images, video_idx_in_batch, split_sizes) # list: [F * N * C]
            image_features = [torch.cat(image_features, dim=0)]
            if self.use_mfcnn:
                if self.add_time_token:
                    image_feature = image_features[0]
                    num_frames = round(image_feature.size(0) / 183)
                    frame_per_time = real_time[0] / num_frames
                    time_idx = [str(round(frame_per_time * f_idx, 1)) for f_idx in range(1, num_frames + 1)]
                    time_tokens = [self.tokenizer(t_idx, return_tensors='pt')["input_ids"].to(self.device) for t_idx in time_idx]
                    time_embeds = [self.get_model().embed_tokens(t_tok).squeeze() for t_tok in time_tokens]
                    padded_time_embeds = pad_sequence(time_embeds, batch_first=True)
                    image_feature = image_feature.view(num_frames, -1, image_feature.size(-1))
                    image_feature = torch.cat((image_feature, padded_time_embeds), dim=1)
                    image_feature = image_feature.view(-1, image_feature.size(-1))
                    image_features = [image_feature]
            else:
                # image_features = torch.split(image_features, split_sizes, dim=0)
                mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
                image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
                new_image_features = []

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
                                if self.config.mm_newline_position == "grid": # here
                                    # Grid-wise
                                    resize_h = int(math.sqrt(image_feature.shape[1]))
                                    num_frames = image_feature.shape[0]
                                    image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
                                    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                    image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                    image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                                    image_feature = image_feature.flatten(1, 2).transpose(0, 1) # N * C

                                    if self.add_time_token:
                                        frame_per_time = real_time[image_idx] / num_frames
                                        time_idx = [str(round(frame_per_time * f_idx, 1)) for f_idx in range(1, num_frames + 1)]
                                        time_tokens = [self.tokenizer(t_idx, return_tensors='pt')["input_ids"].to(self.device) for t_idx in time_idx]
                                        time_embeds = [self.get_model().embed_tokens(t_tok).squeeze() for t_tok in time_tokens]
                                        padded_time_embeds = pad_sequence(time_embeds, batch_first=True)
                                        image_feature = image_feature.view(num_frames, -1, image_feature.size(-1))
                                        image_feature = torch.cat((image_feature, padded_time_embeds), dim=1)
                                        image_feature = image_feature.view(-1, image_feature.size(-1))

                                    new_image_features.append(image_feature)
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

                                # no-token
                                continue

                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            height = width = self.get_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]
                            if image_aspect_ratio == 'anyres':
                                try:
                                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                                except:
                                    import pdb; pdb.set_trace() 
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

        all_image_features = []
        all_segment_keys = []

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        image_feature_sizes = 0
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

            # gs534 - segment process
            if self.segmentation > 0: # and not self.config.model_args["streamdecoder"]:
                image_feature_list = []
                start = 0
                num_frames = images[0].shape[0]
                per_frame_feature_len = image_features[0].size(0) // num_frames
                while start < image_features[0].size(0):
                    endpoint = start + self.segmentation * per_frame_feature_len
                    if endpoint > image_features[0].size(0):
                        start = image_features[0].size(0) - self.segmentation * per_frame_feature_len
                    im_feature = image_features[0][start: endpoint]
                    if self.do_rag:
                        all_image_features.append(im_feature)
                        if "llm" not in self.rag_type and "text" not in self.rag_type:
                            im_mid_key = self.encode_video_shortclip(im_feature, keeplen=1)
                            all_segment_keys.append(im_mid_key)
                    im_mid_feature = im_feature[:per_frame_feature_len]
                    image_feature_list.append(im_mid_feature)
                    start = endpoint
                if len(image_feature_list) > self.rag_input_frames:
                    step = (len(image_feature_list) - 1) / (self.rag_input_frames - 1)
                    indices = [int(round(i*step)) for i in range(self.rag_input_frames)]
                    image_feature_list = [image_feature_list[idx] for idx in indices]
                image_features = [torch.cat(image_feature_list, dim=0)]
                image_feature_sizes = [image_features[0].size(0)]
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
            modality_max_length = ast.literal_eval(modality_max_length)
            modality_max_length_dict = {"image": modality_max_length[0], "text": modality_max_length[1], "video": modality_max_length[2]}
            new_input_embeds =[x[: modality_max_length_dict[modality]] for x, modality in zip(new_input_embeds, modalities)]
            new_labels = [x[: modality_max_length_dict[modality]] for x, modality in zip(new_labels, modalities)]

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

        if torch.cuda.current_device() == 0:
            print(f'[RANK0 PRINT] | new_input_embeds\'s shape: {new_input_embeds.shape}')

        # all_image_features = torch.stack(all_image_features, dim=0) if all_image_features != [] else None
        all_segment_keys = torch.cat(all_segment_keys, dim=0) if all_segment_keys != [] else None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, all_image_features, all_segment_keys, image_feature_sizes

    def encode_video_shortclip(self, inputs_embeds, keeplen=1):
        causal_mask = torch.tril(torch.ones(inputs_embeds.size(0), inputs_embeds.size(0)))
        query = inputs_embeds.unsqueeze(0)
        key = inputs_embeds.unsqueeze(0)
        value = inputs_embeds.unsqueeze(0)
        causal_mask = causal_mask.to(query.device).to(query.dtype)
        output, output_weight = self.segmentTFM(query, key, value, is_causal=True, attn_mask=causal_mask)
        # else:
            # output = self.segmentTFM(inputs_embeds.unsqueeze(0), is_causal=True, src_mask=causal_mask)
        return output[0, -keeplen:]
    
    def encode_images_no_projector(self, images, video_idx_in_batch=[], split_sizes=None):
        image_features = self.get_model().get_vision_tower()(images)
        if split_sizes is None:
            split_sizes = [1 for image in images]
        per_image_features = torch.split(image_features, split_sizes, dim=0) # tuple, (dim_1, 576, 4096)
        all_image_features = []

        for idx, img_feat in enumerate(per_image_features):
            if self.config.mm_pooling_position == "before":
                if idx in video_idx_in_batch and self.config.mm_spatial_pool_stride > 1:
                    img_feat = self.get_2dPool(img_feat) # (num_vid*num_frames, 576, 4096) -> (num_vid*num_frames, 144, 4096)
            # img_feat = self.get_model().mm_projector(img_feat) # (dim_1_sum, 576, 1024) -> (dim_1_sum, 576, 4096)

            if self.config.mm_pooling_position == "after":
                if idx in video_idx_in_batch and self.config.mm_spatial_pool_stride > 1:
                    img_feat = self.get_2dPool(img_feat) # (num_vid*num_frames, 576, 4096) -> (num_vid*num_frames, 144, 4096)

            all_image_features.append(img_feat)
        return all_image_features

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
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        real_time=None,
        cache_position=None,
        raw_wav=None,
        spectrogram=None,
        org_groups=None,
        do_test=False,
        duration=None,
        captions=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        do_rag = True
        if inputs_embeds is not None or images is None:
            do_rag = False
        elif duration is not None and duration[0] == 0:
            do_rag = False

        if inputs_embeds is None:
            orig_input_ids = input_ids
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, all_image_features, all_segment_keys, image_feature_sizes) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes, real_time=real_time)

        #     import pdb; pdb.set_trace()
        if self.do_rag and do_rag:
            # RAG prefixes and suffixes
            rag_prefix_ids = self.tokenizer("Retrieved Video Clip:\n", return_tensors='pt')["input_ids"].to(self.device)
            rag_exec_ids = self.tokenizer("<clip>\n", return_tensors='pt')["input_ids"].to(self.device)
            rag_suffix_token = self.tokenizer("\nThe answer is:\n", return_tensors='pt')["input_ids"].to(self.device)
            rag_prefix = self.get_model().embed_tokens(rag_prefix_ids)
            rag_exec = self.get_model().embed_tokens(rag_exec_ids)
            rag_suffix = self.get_model().embed_tokens(rag_suffix_token)
            rag_pos = torch.where(labels==151644)[1][2] + 3
            end_pos = torch.where(labels==151645)[1][2] + 1
            query_embeds = inputs_embeds[:, :rag_pos]
            answer_embeds = inputs_embeds[:, rag_pos:end_pos+1]
            if "front" in self.rag_type:
                image_token_pos = torch.where(orig_input_ids==-200)[1][0]
                query_start_pos = image_token_pos + image_feature_sizes[0]

            query_text = self.tokenizer.decode(orig_input_ids[0, 17:])

            # Then locate the clip
            rag_loss = 0
            if "llm" in self.rag_type:
                all_segment_keys = []
                for im_feature in all_image_features:
                    with torch.no_grad():
                        clip_output = self.model(attention_mask=attention_mask.new_ones(1, im_feature.size(1)), inputs_embeds=im_feature.unsqueeze(0))
                    im_mid_key = self.encode_video_shortclip(clip_output[0][0], keeplen=1)
                    all_segment_keys.append(im_mid_key)
                all_segment_keys = torch.cat(all_segment_keys, dim=0)

            if "direct" in self.rag_type:
                selected_doc = None
                top_doc_scores = []
                if "text" in self.rag_type:
                    top_doc_ids = []
                    if len(all_image_features) > (2 * self.rag_topk) or "train" in self.rag_type:
                        if "video" in self.rag_type:
                            video_features = images[0].to(torch.float) if "train" not in self.rag_type else images[0]
                            query_text = query_text.split("<|im_end")[0]
                            if "A. " in query_text and "B. " in query_text:
                                query_text = query_text.split("A. ")[0]
                            mem_inputs = self.text_processor(text=[query_text], padding="max_length", return_tensors="pt")
                            mem_inputs["input_ids"] = torch.cat([mem_inputs.input_ids[:, :63], mem_inputs.input_ids[:, -1:]], dim=-1).to(self.device)
                            logits_per_image = []
                            stepsize = 100
                            with torch.no_grad():
                                index = 0
                                while index < video_features.size(0):
                                    mem_inputs["pixel_values"] = video_features[index:index+stepsize]
                                    outputs = self.video_encoder(**mem_inputs)
                                    logits_per_image.append(outputs.logits_per_image)
                                    index += stepsize
                                logits_per_image = torch.cat(logits_per_image, dim=0)
                                clip_level_scores = []
                                for i in range(0, video_features.size(0), self.segmentation):
                                    clip_level_scores.append(logits_per_image[i:i+self.segmentation, 0].max())
                                scores = torch.tensor(clip_level_scores).to(self.device)
                                videoscores = scores
                                top_doc_ids = scores.topk(min(scores.size(0), self.rag_topk), dim=0)[1].tolist()
                                top_doc_ids = sorted(top_doc_ids)
                            if "train" in self.rag_type and "both" not in self.rag_type:
                                top_doc_ids_reward = scores.topk(min(scores.size(0), 2 * self.rag_topk), dim=0)[1].tolist()
                                for idx in top_doc_ids_reward:
                                    mem_inputs["pixel_values"] = video_features[idx*self.segmentation:(idx+1)*self.segmentation]
                                    outputs = self.video_encoder(**mem_inputs)
                                    score = outputs.logits_per_image[:, 0].max()
                                    top_doc_scores.append(score)
                                top_doc_scores = torch.stack(top_doc_scores, dim=-1)
                        if captions is not None and len(all_image_features) == len(captions[0]) and ("video" not in self.rag_type or "both" in self.rag_type):
                            query_text = f'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query_text}'
                            input_texts = [query_text.split("<|im_end")[0]] + captions[0]
                            with torch.no_grad():
                                all_embeddings = []
                                stepsize = 30
                                count = 0
                                while count < len(input_texts):
                                    batch_dict = self.text_tokenizer(input_texts[count:count+stepsize], max_length=4096, padding=True, truncation=True, return_tensors='pt').to(self.device)
                                    outputs = self.text_encoder(**batch_dict, use_cache=False)
                                    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                                    all_embeddings.append(embeddings)
                                    count += stepsize
                                embeddings = torch.cat(all_embeddings, dim=0)
                                # normalize embeddings
                                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                                scores = (embeddings[:1] @ embeddings[1:].T) * 100
                                if "both" in self.rag_type:
                                    scores = scores + videoscores.unsqueeze(0)
                                top_doc_ids = scores.topk(self.rag_topk, dim=1)[1].tolist()[0]
                                top_doc_ids = sorted(top_doc_ids)
                        selected_doc = []
                        selected_labels = []
                        for doc_id in top_doc_ids:
                            selected_doc.append(rag_exec[0])
                            selected_doc.append(all_image_features[doc_id])
                            selected_labels.append(rag_exec_ids*0-100)
                            selected_labels.append(labels.new_ones(1, all_image_features[doc_id].size(0)) * -100)
                        if len(selected_doc) > 0:
                            selected_doc = torch.cat(selected_doc, dim=0)
                            selected_labels = torch.cat(selected_labels, dim=-1)
                else:
                    with torch.no_grad():
                        outputs2 = self.model(attention_mask=attention_mask, inputs_embeds=torch.cat([query_embeds, rag_prefix], dim=1))
                    query = self.rag_dropout(self.query_proj(outputs2[0][0, -1]))
                    # vector_norms = torch.norm(query, dim=-1) * torch.norm(all_segment_keys, dim=-1)
                    scores = torch.einsum("k,jk->j", query, all_segment_keys) / (query.size(-1))
                    selected_doc = torch.einsum("i,ijk->jk", torch.softmax(scores, dim=-1), torch.stack(all_image_features, dim=0))
                    selected_labels = labels.new_ones(1, selected_doc.size(0)) * -100

            # Forward LLM with RAG
            if selected_doc is not None and len(selected_doc) > 0:
                if "front" in self.rag_type:
                    selected_input_embeds = torch.cat([query_embeds[:, :query_start_pos], rag_prefix, selected_doc.unsqueeze(0), rag_suffix, query_embeds[:, query_start_pos:], answer_embeds], dim=1)
                    selected_labels = torch.cat([labels[:, :query_start_pos], rag_prefix_ids*0-100, selected_labels, rag_suffix_token*0-100, labels[:, query_start_pos:end_pos+1]], dim=1)
                else:
                    selected_input_embeds = torch.cat([query_embeds, rag_prefix, selected_doc.unsqueeze(0), rag_suffix, answer_embeds], dim=1)
                    selected_labels = torch.cat([labels[:, :rag_pos], rag_prefix_ids*0-100, selected_labels, rag_suffix_token*0-100, labels[:, rag_pos:end_pos+1]], dim=1)
            else:
                selected_input_embeds = inputs_embeds[:, :end_pos+1]
                selected_labels = labels[:, :end_pos+1]
            outputs = super().forward(
                attention_mask=attention_mask.new_ones(1, selected_input_embeds.size(1)),
                inputs_embeds=selected_input_embeds,
                labels=selected_labels,
            )

            # Compute RAG loss
            rag_loss = 0
            if "train" in self.rag_type and self.training: # and len(top_doc_scores) == 2 * self.rag_topk:
                if len(top_doc_ids_reward) < 2 * self.rag_topk:
                    rag_loss = top_doc_scores.mean() * 0
                else:
                    top_doc_sets = [sorted(top_doc_ids_reward[:self.rag_topk]), sorted(top_doc_ids_reward[self.rag_topk:])]
                    mean_scores = [top_doc_scores[:self.rag_topk].mean(), top_doc_scores[self.rag_topk:].mean()]
                    rewards = []
                    for doc_ids in top_doc_sets:
                        set_selected_doc = []
                        set_selected_labels = []
                        for doc_id in doc_ids:
                            set_selected_doc.append(rag_exec[0])
                            set_selected_doc.append(all_image_features[doc_id])
                            set_selected_labels.append(rag_exec_ids*0-100)
                            set_selected_labels.append(labels.new_ones(1, all_image_features[doc_id].size(0)) * -100)
                        set_selected_doc = torch.cat(set_selected_doc, dim=0)
                        set_selected_labels = torch.cat(set_selected_labels, dim=-1)
                        with torch.no_grad():
                            if "front" in self.rag_type:
                                set_selected_input_embeds = torch.cat([query_embeds[:, :query_start_pos], rag_prefix, set_selected_doc.unsqueeze(0), rag_suffix, query_embeds[:, query_start_pos:], answer_embeds], dim=1)
                                set_selected_labels = torch.cat([labels[:, :query_start_pos], rag_prefix_ids*0-100, set_selected_labels, rag_suffix_token*0-100, labels[:, query_start_pos:end_pos+1]], dim=1)
                            else:
                                set_selected_input_embeds = torch.cat([query_embeds, rag_prefix, set_selected_doc.unsqueeze(0), rag_suffix, answer_embeds], dim=1)
                                set_selected_labels = torch.cat([labels[:, :rag_pos], rag_prefix_ids*0-100, set_selected_labels, rag_suffix_token*0-100, labels[:, rag_pos:end_pos+1]], dim=1)
                            tmp_outputs = super().forward(
                                attention_mask=attention_mask.new_ones(1, set_selected_input_embeds.size(1)),
                                inputs_embeds=set_selected_input_embeds,
                                labels=set_selected_labels,
                            )
                            rewards.append(tmp_outputs.loss)
                    reward_factor = 1.0 if rewards[1] > rewards[0] else 0
                    p_s0_s1 = torch.log(torch.sigmoid(mean_scores[0] - mean_scores[1]))
                    p_s1_s0 = torch.log(torch.sigmoid(mean_scores[1] - mean_scores[0]))
                    rag_loss = - (reward_factor * p_s0_s1 + (1 - reward_factor) * p_s1_s0)
                    if torch.cuda.current_device() == 0:
                        print("RAG LOSS: {:.3f}".format(rag_loss))
                outputs.loss = outputs.loss + rag_loss
            return outputs
        else:
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
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        real_time=None,
        captions=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        inputs = input_ids
        orig_input_ids = input_ids

        if images is not None:
            (_, position_ids, attention_mask, _, inputs_embeds, labels, all_image_features, all_segment_keys, image_feature_sizes) = self.prepare_inputs_labels_for_multimodal(
                inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, real_time=real_time)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if self.do_rag:
            # RAG prefixes and suffixes
            rag_prefix_ids = self.tokenizer("Retrieved Video Clip:\n", return_tensors='pt')["input_ids"].to(self.device)
            rag_exec_ids = self.tokenizer("<clip>\n", return_tensors='pt')["input_ids"].to(self.device)
            rag_suffix_token = self.tokenizer("\nThe answer is:\n", return_tensors='pt')["input_ids"].to(self.device)
            rag_prefix = self.get_model().embed_tokens(rag_prefix_ids)
            rag_exec = self.get_model().embed_tokens(rag_exec_ids)
            rag_suffix = self.get_model().embed_tokens(rag_suffix_token)
            # rag_pos = torch.where(orig_input_ids==151644)[1][2] + 2 + image_feature_sizes[0]
            # query_embeds = inputs_embeds[:, :rag_pos]
            if "front" in self.rag_type:
                image_token_pos = torch.where(orig_input_ids==-200)[1][0]
                query_start_pos = image_token_pos + image_feature_sizes[0]
                pre_image_embs = inputs_embeds[:, :query_start_pos]
                post_image_embs = inputs_embeds[:, query_start_pos:]
            query_text = self.tokenizer.decode(orig_input_ids[0, 17:])
            query_text = query_text.replace("Select the best answer to the following multiple-choice question based on the video. \nRespond with only the letter (A, B, C, or D) of the correct option.", "")

            # Then locate the clip
            rag_loss = 0
            if "llm" in self.rag_type:
                all_segment_keys = []
                for im_feature in all_image_features:
                    with torch.no_grad():
                        clip_output = self.model(attention_mask=attention_mask, inputs_embeds=im_feature.unsqueeze(0))
                    im_mid_key = self.encode_video_shortclip(clip_output[0][0], keeplen=1)
                    all_segment_keys.append(im_mid_key)
                all_segment_keys = torch.cat(all_segment_keys, dim=0)

            if "direct" in self.rag_type:
                selected_doc = None
                if kwargs['duration'] is not None and "ref" in self.rag_type:
                    duration = kwargs['duration']
                    start = int(duration[0][0] / real_time[0] * len(all_image_features))
                    end = min(int(duration[0][1] / real_time[0] * len(all_image_features)) + 1, len(all_image_features))
                    end = min(end, start+3)
                    selected_doc = []
                    query_pos = [inputs_embeds.size(1)+rag_prefix.size(1) - 1]
                    for im_feature in all_image_features[start:end]:
                        selected_doc.append(rag_exec[0])
                        selected_doc.append(im_feature)
                        query_pos.append(query_pos[-1] + rag_exec[0].size(0) + im_feature.size(0))
                    selected_doc = torch.cat(selected_doc, dim=0)
                elif "text" in self.rag_type:
                    top_doc_ids = []
                    if len(all_image_features) > (2 * self.rag_topk):
                        if "video" in self.rag_type:
                            video_features = images[0].to(torch.float)
                            mem_inputs = self.text_processor(text=[query_text.split("<|im_end")[0]], padding="max_length", return_tensors="pt")
                            mem_inputs["input_ids"] = torch.cat([mem_inputs.input_ids[:, :63], mem_inputs.input_ids[:, -1:]], dim=-1).to(self.device)
                            logits_per_image = []
                            stepsize = 100
                            index = 0
                            while index < video_features.size(0):
                                mem_inputs["pixel_values"] = video_features[index:index+stepsize]
                                outputs = self.video_encoder(**mem_inputs)
                                logits_per_image.append(outputs.logits_per_image)
                                index += stepsize
                            logits_per_image = torch.cat(logits_per_image, dim=0)
                            clip_level_scores = []
                            for i in range(0, video_features.size(0), self.segmentation):
                                clip_level_scores.append(logits_per_image[i:i+self.segmentation, 0].max())
                            scores = torch.tensor(clip_level_scores).to(self.device)
                            if "both" in self.rag_type:
                                videoscores = scores
                            else:
                                top_doc_ids = scores.topk(self.rag_topk, dim=0)[1].tolist()
                                top_doc_ids = sorted(top_doc_ids)
                        if captions is not None and len(all_image_features) == len(captions[0]) and ("video" not in self.rag_type or "both" in self.rag_type):
                            query_text = f'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {query_text}'
                            input_texts = [query_text.split("<|im_end")[0]] + captions[0]
                            all_embeddings = []
                            stepsize = 30
                            count = 0
                            while count < len(input_texts):
                                batch_dict = self.text_tokenizer(input_texts[count:count+stepsize], max_length=4096, padding=True, truncation=True, return_tensors='pt').to(self.device)
                                outputs = self.text_encoder(**batch_dict, use_cache=False)
                                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                                all_embeddings.append(embeddings)
                                count += stepsize
                            embeddings = torch.cat(all_embeddings, dim=0)
                            # normalize embeddings
                            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                            scores = (embeddings[:1] @ embeddings[1:].T) * 100
                            if "both" in self.rag_type:
                                scores = scores + videoscores.unsqueeze(0)
                            top_doc_ids = scores.topk(self.rag_topk, dim=1)[1].tolist()[0]
                            top_doc_ids = sorted(top_doc_ids)
                    selected_doc = []
                    for doc_id in top_doc_ids:
                        selected_doc.append(rag_exec[0])
                        selected_doc.append(all_image_features[doc_id])
                    if len(selected_doc) > 0:
                        selected_doc = torch.cat(selected_doc, dim=0)
                else:
                    outputs2 = self.model(attention_mask=attention_mask, inputs_embeds=torch.cat([inputs_embeds, rag_prefix], dim=1))
                    query = self.query_proj(outputs2[0][0, -1])
                    scores = torch.einsum("k,jk->j", query, all_segment_keys) / (query.size(-1))
                    selected_doc = torch.einsum("i,ijk->jk", torch.softmax(scores, dim=-1), torch.stack(all_image_features, dim=0))
                if selected_doc is not None and len(selected_doc) > 0:
                    if "front" in self.rag_type:
                        inputs_embeds = torch.cat([pre_image_embs, rag_prefix, selected_doc.unsqueeze(0), rag_suffix, post_image_embs], dim=1)
                    else:
                        inputs_embeds = torch.cat([inputs_embeds, rag_prefix, selected_doc.unsqueeze(0), rag_suffix], dim=1)
            else:
                generated_list = []
                # scores = torch.log_softmax(scores, dim=-1)
                for selected_doc_id, selected_doc in enumerate(all_image_features):
                    segment_length = real_time[0] / (len(all_segment_keys) - 1)
                    # print("Use RAG clip from {:.2f}s to {:.2f}s".format(segment_length*selected_doc_id, segment_length*(selected_doc_id+1)))
                    inputs_embeds = torch.cat([pre_image_embs, selected_doc.unsqueeze(0), post_image_embs], dim=1)
                    attention_mask = attention_mask.new_ones(1, inputs_embeds.size(1)) if attention_mask is not None else None
                    kwargs["max_new_tokens"] = 256
                    output = super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
                    description = self.tokenizer.decode(output["sequences"][0]).replace("<|im_end|>", "")
                    generated_list.append((description, [segment_length*selected_doc_id, segment_length*(selected_doc_id+1)]))
                return generated_list

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
