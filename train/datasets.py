# Copyright (2026) Tsinghua University, Bytedance Ltd. and/or its affiliates
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

# Adapted from https://github.com/baaivision/UniVLA. The original license is located at 'third-party-license/UniVLA.txt'.

import json
import os.path as osp
import os
import random
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union
from torch.utils.data import Dataset
from PIL import Image
import sys
ELLSA_BASE_PATH = os.environ.get("ELLSA_BASE_PATH")
ELLSA_DATA_PATH = os.environ.get("ELLSA_DATA_PATH")
COSY_CKPT_PATH = os.environ.get("COSY_CKPT_PATH")
LLAMA_CKPT_PATH = os.environ.get("LLAMA_CKPT_PATH")

sys.path.append(ELLSA_BASE_PATH)
from models.tokenizer.action_tokenizer import ActionTokenizer
from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor
import kaldifeat, torchaudio, math
from lhotse import Fbank, FbankConfig
from torch.nn.utils.rnn import pad_sequence

try:
    sys.path.append("reference/")
    from cosyvoice.cli.cosyvoice import CosyVoice2
except:
    print("failed to load CosyVoice2")
    
class Emu3SFTDataset(Dataset):

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__()

        self.args = args
        # data args
        self.random_frame_sampling = args.random_frame_sampling
        self.raw_image = args.raw_image
        self.data_path = args.data_path
        
        with open(args.data_path,'rb') as f:
            self.data = pickle.load(f)
        
        if not self.random_frame_sampling:
            self.data = list(self.sliding_window_sampling(self.data, interval=args.action_frames*args.frames))
        
        if "libero" in args.data_path:
            action_high = np.array([
                0.93712500009996,
                0.86775000009256,
                0.93712500009996,
                0.13175314309916836,
                0.19275000005139997,
                0.3353504997073735,
                0.9996000000999599
            ])
            action_low = np.array([
                -0.7046250000751599,
                -0.80100000008544,
                -0.9375000001,
                -0.11467779149968735,
                -0.16395000004372,
                -0.2240490058320433,
                -1.0000000001
            ])
            normalized = 2 * (np.array([0, 0, 0, 0, 0, 0, -1]) - action_low) / (action_high - action_low + 1e-8) - 1
            self.dummy_action = np.clip(normalized, -1, 1).tolist()   
        elif "calvin" in args.data_path:
            action_high = np.array([
                0.67640000006764,
                0.5560000000555998,
                0.5944000000594398,
                0.42640000004264,
                0.41200000004119985,
                0.9996000000999599,
                0.9996000000999599
            ])
            
            action_low = np.array([
                -0.69960000006996,
                -0.57760000005776,
                -0.4336000000433601,
                -0.42320000004232006,
                -0.46520000004652007,
                -1.0000000001,
                -1.0000000001
            ])
            """
            action_high = np.array([
                0.6840,
                0.5388,
                0.6012,
                0.3996,
                0.3840,
                0.9996,
                0.9996
            ])

            action_low = np.array([
                -0.6652,
                -0.5620,
                -0.4344,
                -0.4056,
                -0.4264,
                -1.0000,
                -1.0000
            ])
            """
            normalized = 2 * (np.array([0, 0, 0, 0, 0, 0, 1]) - action_low) / (action_high - action_low) - 1
            self.dummy_action = np.clip(normalized, -1, 1).tolist()
            normalized = 2 * (np.array([0, 0, 0, 0, 0, 0, -1]) - action_low) / (action_high - action_low) - 1
            self.dummy_action_minus = np.clip(normalized, -1, 1).tolist()
        
        self.tokenizer = tokenizer
        self.bov = tokenizer.encode(args.visual_token_pattern.format(token_id=0))[0]
        self.eov = tokenizer.encode(args.visual_token_pattern.format(token_id=args.codebook_size - 1))[0]
        self.chat_template="You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:"
        self.gen_template="You are a powerful painter. USER: {text} ASSISTANT:{image}"
        self.act_template="Action: {action_prompt}"
        self.VL = args.VL
        self.cfg = False
        self.post_training = args.post_training

        # pretrain use
        if self.post_training:
            # v2
            # self.dataset_fps = {'rt1':3, 'bridgev2':5, 'droid':15, '1x':1, 'kuka':3, 'calvin':5, 'libero':5} 
            # v3
            self.dataset_fps = {'1x':1, 'SSv2':1,'rt1':3, 'kuka':3, \
                                'bridgev2':5, 'taco_play':5, \
                                'calvin':10, 'libero':10,'maniskill':10,'cmu_play_fusion':10,'utaustin_mutex':10, \
                                'droid':15, 'viola':15, \
                                'toto':20} # calvin:10
        else:
            self.dataset_fps = {}
        self.T = args.frames
        self.action_frames = args.action_frames
        
        self.actions = args.actions
        self.actions_format = args.actions_format

        self.use_gripper = args.use_gripper  

        self.video_format = args.video_format

        if self.raw_image:
            self.vision_hub = os.path.join(ELLSA_BASE_PATH,"ckpt/Emu3-VisionVQ")
            self.image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)
            self.image_tokenizer = AutoModel.from_pretrained(self.vision_hub, trust_remote_code=True)
            self.image_processor.min_pixels = 80 * 80
        if self.actions_format == "openvla":
            self.action_tokenizer = ActionTokenizer(tokenizer, bins=256, min_action=-1.0, max_action=1.0)
        elif self.actions_format == "fast":
            self.fast_path = args.action_tokenizer_path
            sys.path.append(self.fast_path)
            from processing_action_tokenizer import UniversalActionProcessor
            self.action_tokenizer = UniversalActionProcessor.from_pretrained(self.fast_path, trust_remote_code=True)

    def __len__(self):
        return len(self.data)
    
    def sliding_window_sampling(self, data, interval=5):
        """
        Implement sliding window sampling using a generator.
        """
        for item in data:
            T = len(item['image'])
            if T <= interval:
                raise ValueError("Length of 'image', 'action', and 'gripper' must be greater than 'interval'.")
            for start_idx in range(0, T - interval + 1, 1):
                yield {
                    'text': item['text'],
                    'image': item['image'][start_idx:start_idx+interval],
                    'action': item['action'][start_idx:start_idx+interval],
                    'gripper_image': item['gripper_image'][start_idx:start_idx+interval],
                }

    def random_frames_to_tensor(self, img_list, T, action_prompt=None, gripper=None, return_start=False, start_idx=-1):
        
        if start_idx == -1:
            start_idx = random.randint(0, len(img_list) - T)

        if hasattr(self, 'raw_image') and self.raw_image:
            self.image_tokenizer.eval()
            # Process raw images with VQ encoding
            selected_frames = [Image.open(img_path) for img_path in img_list[start_idx:start_idx + T]]
            selected_frames = [self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0) for img in selected_frames]

            tensor_frames = torch.stack(selected_frames, dim=0)
            with torch.no_grad():
                image_code = self.image_tokenizer.encode(tensor_frames)
            
            if gripper is not None and action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                selected_gripper = [Image.open(img_path) for img_path in gripper[start_idx:start_idx + T]]
                selected_gripper = [self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0) for img in selected_gripper]
                tensor_gripper = torch.stack(selected_gripper, dim=0)
                with torch.no_grad():
                    gripper_code = self.image_tokenizer.encode(tensor_gripper)
                return image_code, selected_actions, gripper_code
            elif action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                return image_code, selected_actions
        else:
            selected_frames = [np.load(img_path) for img_path in img_list[start_idx:start_idx + T]]
            tensor_frames = [torch.from_numpy(frame) for frame in selected_frames]
            tensor = torch.stack(tensor_frames, dim=1)

            if gripper is not None and action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                selected_gripper = [np.load(img_path) for img_path in gripper[start_idx:start_idx + T]]
                tensor_gripper = [torch.from_numpy(frame) for frame in selected_gripper]
                if return_start:
                    return tensor.squeeze(0), selected_actions, torch.stack(tensor_gripper, dim=1).squeeze(0), start_idx
                else:
                    return tensor.squeeze(0), selected_actions, torch.stack(tensor_gripper, dim=1).squeeze(0)
            elif action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                if return_start:
                    return tensor.squeeze(0), selected_actions, start_idx
                else:
                    return tensor.squeeze(0), selected_actions
            elif gripper is not None:
                selected_gripper = [np.load(img_path) for img_path in gripper[start_idx:start_idx + T]]
                tensor_gripper = [torch.from_numpy(frame) for frame in selected_gripper]
                if return_start:
                    return tensor.squeeze(0), torch.stack(tensor_gripper, dim=1).squeeze(0), start_idx
                else:
                    return tensor.squeeze(0), torch.stack(tensor_gripper, dim=1).squeeze(0)
        return tensor.squeeze(0)
    
    def get_fps_for_path(self, image_tokens_path):
        for key in self.dataset_fps.keys():
            if key in image_tokens_path[0]:
                return self.dataset_fps[key]
        # Default return value if no key matches
        return None  # or some default FPS value
    
    def pad_tensor(self, tensor, max_length, pad_value):
        """Pads a tensor to a specified maximum length."""
        current_length = tensor.shape[-1]
        if current_length < max_length:
            pad_length = max_length - current_length
            padding = torch.full((pad_length,), fill_value=pad_value, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=-1)
        return tensor

    def __getitem__(self, index: int, start_idx=-1):

        scene = self.data[index]

        if self.cfg:
            p_prob = random.random()
            if p_prob < self.args.null_prompt_prob:
                prompt = ""
            else:
                prompt = scene["text"]
        else:
            prompt = scene["text"]

        image_tokens_path = scene["image"]

        # handle different dataset fps for post training
        fps = self.get_fps_for_path(image_tokens_path)
        if fps is not None:
            self.action_frames = fps
        
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        
        # use action information
        if self.actions:
            action = scene["action"] 
            if self.use_gripper:
                gripper = scene["gripper_image"]
                image_tokens, action_tokens, gripper_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, gripper=gripper, return_start=True, start_idx=start_idx)
            else:
                image_tokens, action_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, return_start=True, start_idx=start_idx)
            
            if self.video_format == "interleave":
                if self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                    action_ids = self.action_tokenizer(action_tokens)
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
            else:
                if self.actions_format == "openvla":
                    action_tokens = action_tokens.flatten()
                    action_ids = self.action_tokenizer(action_tokens)

                    # Debugging
                    # action_debug = self.action_tokenizer.decode_token_ids_to_actions(action_ids)
                    # error = action_tokens - action_debug
                elif self.actions_format == "text":
                    action_str = "\n".join(",".join(f"{num:.2f}" for num in row) for row in action_tokens)
                    action_prompt = self.act_template.format(action_prompt=action_str)
                elif self.actions_format == "continuous":
                    action_continuous = action_tokens
                elif self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_ids = self.action_tokenizer(action_tokens)[0]
                    # action_decode = self.action_tokenizer.decode([action_ids])
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - id for id in action_ids]
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
        else:
            if self.use_gripper:
                gripper = scene["gripper_image"]
                image_tokens, gripper_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, gripper=gripper)
            else:
                image_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num) 
        # video VLA
        if self.video_format == "interleave":
            text_prompt = self.tokenizer.bos_token + prompt
            image_tokens = image_tokens[0::self.action_frames,...]
            if self.use_gripper:
                gripper_tokens = gripper_tokens[0::self.action_frames,...]
            
            sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                if self.use_gripper:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                    image_prompt += gripper_prompt
                sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                if self.actions:
                    if self.actions_format == "fast":
                        action_sample = self.wrap_action_sequence(action_ids[i].tolist()) 
                        sample_input_ids = torch.cat([sample_input_ids, image_input_ids, action_sample], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1) 
                        action_start = len(sample_input_ids) - len(action_sample)
                        action_end = len(sample_input_ids)
                        if self.args.apply_loss_on_only_action:  
                            labels[action_start:action_end] = action_sample
                        else:  # Otherwise, fill both vision and action parts in the labels
                            labels[action_start-len(image_input_ids):action_start] = image_input_ids  
                            labels[action_start:action_end] = action_sample 
                else:
                    sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                    sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                    labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
            
            sample = self.tokenizer.pad(
                    {
                        "input_ids": sample_input_ids,
                        "attention_mask": sample_attention_mask,
                        "labels": labels
                    },
                    padding="max_length",
                    return_tensors="pt"
                )
            for k, v in sample.items():
                sample[k] = v.squeeze(0)
            # sample["path"] = image_tokens_path[0][:image_tokens_path[0].rfind("/")]
            # sample["frames_num"] = len(image_tokens_path) - frames_num
        # VLA Baseline (Img)
        else:
            image_tokens = image_tokens[0:self.T,...]
            image_prompt = self.format_video_prompt(image_tokens)

            if self.use_gripper:
                gripper_tokens = gripper_tokens[0:self.T,...]
                gripper_prompt = self.format_video_prompt(gripper_tokens)
                image_prompt = image_prompt + gripper_prompt  

            if self.VL:
                p_prob_order = random.random()
                if p_prob_order < 0.5:
                    input = self.tokenizer.bos_token + prompt + image_prompt + self.tokenizer.eos_token
                else:
                    # input = self.tokenizer.bos_token + image_prompt + prompt
                    input = self.tokenizer.bos_token + self.chat_template.format(image_prompt=image_prompt, text_prompt=prompt) + self.tokenizer.eos_token
            else:
                input = self.tokenizer.bos_token + prompt + image_prompt 
            # 先不进行padding，后面统一padding
            sample = self.tokenizer(
                input,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            labels = sample["input_ids"]

            # only use vision loss
            if self.args.apply_loss_on_only_vision:
                labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)

            sample["labels"] = labels
            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            # based on the actions_format, append the action information to the sample
            if self.actions:
                if self.actions_format == "openvla":
                    action_sample = self.wrap_action_sequence(action_ids)
                    sample["input_ids"] = torch.cat([sample["input_ids"], action_sample], dim=-1)

                    # Update attention_mask
                    action_mask = torch.ones_like(action_sample, dtype=torch.long)
                    sample["attention_mask"] = torch.cat([sample["attention_mask"], action_mask], dim=-1)

                    action_labels = action_sample.clone()  # Clone action_sample for labels
                    sample["labels"] = torch.cat([sample["labels"], action_labels], dim=-1)
                
                # FAST
                elif self.actions_format == "fast":
                    if self.args.apply_loss_on_only_action:
                        sample['labels'] = torch.full_like(sample['labels'], self.args.ignore_index)
                    sample = self.append_action_to_sample(sample, action_ids)
                
                # Flow Matching
                elif self.actions_format == "continuous":
                    boa_token_id = self.tokenizer.encode(self.tokenizer.boa_token)[0]
                    sample = self.append_boa_to_sample(sample, [boa_token_id])
                    sample["action"] = action_continuous
            
            # finally, do padding
            sample = self.tokenizer.pad(
                sample,
                padding="max_length",
                return_tensors="pt"
            )

            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            if "labels" in sample:
                sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)
        return sample

    def append_action_to_sample(self, sample, action_ids):
        """
        将 action_ids 处理后，追加到 sample 中，包括 input_ids, attention_mask 和 labels。
        """
        action_sample = self.wrap_action_sequence(action_ids)
        action_mask = torch.ones_like(action_sample, dtype=torch.long)

        for key, value in zip(["input_ids", "attention_mask", "labels"], [action_sample, action_mask, action_sample.clone()]):
            sample[key] = torch.cat([sample[key], value], dim=-1)

        return sample
    
    def append_boa_to_sample(self, sample, action_ids):

        action_sample = torch.tensor(action_ids, dtype=torch.long)
        action_mask = torch.ones_like(action_sample, dtype=torch.long)

        for key, value in zip(["input_ids", "attention_mask", "labels"], [action_sample, action_mask, action_sample.clone()]):
            sample[key] = torch.cat([sample[key], value], dim=-1)

        return sample

    def wrap_action_sequence(self, action_ids: List[int]) -> torch.Tensor:
        """
        Wraps a sequence of action token IDs with special tokens (beginning and end).

        Args:
            action_ids (List[int]): The sequence of action token IDs.

        Returns:
            torch.Tensor: A tensor containing the wrapped sequence.
        """
        # Encode the beginning and end action tokens
        action_begin = self.tokenizer.encode(self.tokenizer.boa_token)[0]
        action_end = self.tokenizer.encode(self.tokenizer.eoa_token)[0]
        eos = self.tokenizer.encode(self.tokenizer.eos_token)[0]

        # Wrap the action sequence
        # wrapped_action = [action_begin] + action_ids + [action_end] + [eos]
        wrapped_action = [action_begin] + action_ids + [action_end]
        
        # Convert to a PyTorch tensor
        return torch.tensor(wrapped_action, dtype=torch.long)

    def format_video_prompt(self, video_tokens):
        # 假设video_tokens是一个形状为[frames, height, width]的张量
        frames, h, w = video_tokens.shape
        videostr = self.to_videostr(video_tokens)

        video_prompt = (
            self.tokenizer.boi_token +
            f"{frames}*{h}*{w}" +  # 视频的帧数、高度和宽度
            self.tokenizer.img_token +  # 视频开始标记
            videostr +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )

        return video_prompt

    def to_videostr(self, video_tokens):
        frame_str_list = []
        for frame in video_tokens:
            frame_token_str = [
                self.args.visual_token_pattern.format(token_id=token_id)
                for token_id in frame.flatten()
            ]
            frame_str = "".join(frame_token_str)
            frame_str_list.append(frame_str)
        videostr = self.tokenizer.eof_token.join(frame_str_list)
        return videostr

    def format_image_prompt(self, image_tokens):
        h, w = image_tokens.shape
        imgstr = self.to_imgstr(image_tokens)

        image_prompt = (
            self.tokenizer.boi_token +
            f"{h}*{w}" +
            self.tokenizer.img_token +
            imgstr +
            self.tokenizer.eol_token +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )

        return image_prompt

    def to_imgstr(self, image_tokens):
        image_token_str = [
            [
                self.args.visual_token_pattern.format(token_id=token_id)
                for token_id in token_row
            ]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr

class Emu3SpeechDataset(Emu3SFTDataset):

    def __init__(self, args, tokenizer, moe, encoder_type="mamba"):
        super().__init__(args, tokenizer=tokenizer)

        self.encoder_type = encoder_type
        self.time_block = args.time_block

        if self.encoder_type == "mamba":
            opts = kaldifeat.FbankOptions()
            opts.frame_opts.dither = 0
            opts.frame_opts.snip_edges = False
            opts.frame_opts.samp_freq = 16000 # only support 16k audio
            opts.mel_opts.num_bins = 80 # 80-bin
            opts.mel_opts.high_freq = -400

            self.fbank = kaldifeat.Fbank(opts)
        elif self.encoder_type == "zipformer2":
            self.fbank = Fbank(FbankConfig(num_mel_bins=128))
        self.moe = moe
    
    def random_frames_to_tensor(self, img_list, T, action_prompt=None, gripper=None, return_start=False, select_start=0, select_end=-1):
        
        if select_end == -1:
            start_idx = random.randint(select_start, len(img_list) - T)
        else:
            start_idx = random.randint(select_start, min(select_end,len(img_list)) - T)

        if hasattr(self, 'raw_image') and self.raw_image:
            self.image_tokenizer.eval()
            # Process raw images with VQ encoding
            selected_frames = [Image.open(img_path) for img_path in img_list[start_idx:start_idx + T]]
            selected_frames = [self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0) for img in selected_frames]

            tensor_frames = torch.stack(selected_frames, dim=0)
            with torch.no_grad():
                image_code = self.image_tokenizer.encode(tensor_frames)
            
            if gripper is not None and action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                selected_gripper = [Image.open(img_path) for img_path in gripper[start_idx:start_idx + T]]
                selected_gripper = [self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0) for img in selected_gripper]
                tensor_gripper = torch.stack(selected_gripper, dim=0)
                with torch.no_grad():
                    gripper_code = self.image_tokenizer.encode(tensor_gripper)
                return image_code, selected_actions, gripper_code
            elif action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                return image_code, selected_actions
        else:
            selected_frames = [np.load(img_path) for img_path in img_list[start_idx:start_idx + T]]
            tensor_frames = [torch.from_numpy(frame) for frame in selected_frames]
            tensor = torch.stack(tensor_frames, dim=1)

            if gripper is not None and action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                selected_gripper = [np.load(img_path) for img_path in gripper[start_idx:start_idx + T]]
                tensor_gripper = [torch.from_numpy(frame) for frame in selected_gripper]
                if return_start:
                    return tensor.squeeze(0), selected_actions, torch.stack(tensor_gripper, dim=1).squeeze(0), start_idx
                else:
                    return tensor.squeeze(0), selected_actions, torch.stack(tensor_gripper, dim=1).squeeze(0)
            elif action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                if return_start:
                    return tensor.squeeze(0), selected_actions, start_idx
                else:
                    return tensor.squeeze(0), selected_actions
            elif gripper is not None:
                selected_gripper = [np.load(img_path) for img_path in gripper[start_idx:start_idx + T]]
                tensor_gripper = [torch.from_numpy(frame) for frame in selected_gripper]
                if return_start:
                    return tensor.squeeze(0), torch.stack(tensor_gripper, dim=1).squeeze(0), start_idx
                else:
                    return tensor.squeeze(0), torch.stack(tensor_gripper, dim=1).squeeze(0)
        return tensor.squeeze(0)

    def __getitem__(self, index: int):
        scene = self.data[index]

        if self.cfg:
            p_prob = random.random()
            if p_prob < self.args.null_prompt_prob:
                prompt = ""
            else:
                prompt = scene["speech"]
        else:
            prompt = scene["speech"]
        
        audio4stream, fs = torchaudio.load(scene["speech"])

        speech_len = math.ceil(audio4stream.shape[1] / fs * self.action_frames)

        # extend data
        image_tokens_path = scene["image"]
        image_tokens_path = [image_tokens_path[0]] * speech_len + image_tokens_path

        # handle different dataset fps for post training
        # fps = self.get_fps_for_path(image_tokens_path)
        # if fps is not None:
        #     self.action_frames = fps
        
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        
        # use action information
        if self.actions:
            action = scene["action"]
            if "calvin" in self.data_path:
                if action[0][-1] > 0:
                    dummy_action = self.dummy_action
                elif action[0][-1] < 0:
                    dummy_action = self.dummy_action_minus
            else:
                dummy_action = self.dummy_action
            action = np.concatenate((np.array([dummy_action] * speech_len),action),axis=0)
            if self.use_gripper:
                gripper = scene["gripper_image"]
                gripper = [gripper[0]] * speech_len + gripper
                image_tokens, action_tokens, gripper_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, gripper=gripper, return_start=True)
            else:
                image_tokens, action_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, return_start=True)
            
            if self.video_format == "interleave":
                if self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                    action_ids = self.action_tokenizer(action_tokens)
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
            else:
                if self.actions_format == "openvla":
                    action_tokens = action_tokens.flatten()
                    action_ids = self.action_tokenizer(action_tokens)

                    # Debugging
                    # action_debug = self.action_tokenizer.decode_token_ids_to_actions(action_ids)
                    # error = action_tokens - action_debug
                elif self.actions_format == "text":
                    action_str = "\n".join(",".join(f"{num:.2f}" for num in row) for row in action_tokens)
                    action_prompt = self.act_template.format(action_prompt=action_str)
                elif self.actions_format == "continuous":
                    action_continuous = action_tokens
                elif self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_ids = self.action_tokenizer(action_tokens)[0]
                    # action_decode = self.action_tokenizer.decode([action_ids])
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - id for id in action_ids]
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
        else:
            if self.use_gripper:
                gripper = scene["gripper_image"]
                image_tokens, gripper_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, gripper=gripper)
            else:
                image_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num) 
        # video VLA
        if self.video_format == "interleave":
            text_prompt = ""
            for i in range(math.ceil(start_idx/self.action_frames)):
                text_prompt += self.tokenizer.bos_token + self.tokenizer.eos_token
            image_tokens = image_tokens[0::self.action_frames,...]
            if self.use_gripper:
                gripper_tokens = gripper_tokens[0::self.action_frames,...]
            
            sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                if self.use_gripper:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                    image_prompt += gripper_prompt
                sample_speech = self.tokenizer(self.tokenizer.bos_token + self.tokenizer.eos_token, padding=False, return_token_type_ids=False, return_tensors="pt")
                speech_input_ids = sample_speech["input_ids"][0]
                speech_attention_mask = sample_speech["attention_mask"][0]
                sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                if self.actions:
                    if self.actions_format == "fast":
                        action_sample = self.wrap_action_sequence(action_ids[i].tolist()) 
                        sample_input_ids = torch.cat([sample_input_ids, speech_input_ids, image_input_ids, action_sample], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, speech_attention_mask, image_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1) 
                        action_start = len(sample_input_ids) - len(action_sample)
                        action_end = len(sample_input_ids)
                        if self.args.apply_loss_on_only_action:  
                            labels[action_start:action_end] = action_sample
                        else:  # Otherwise, fill both vision and action parts in the labels
                            labels[action_start-len(image_input_ids):action_start] = image_input_ids  
                            labels[action_start:action_end] = action_sample 
                else:
                    sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                    sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                    labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
            
            if int((start_idx + frames_num)/self.action_frames * fs) > audio4stream.shape[1]:
                pad_audio = torch.zeros(
                    (1, int((start_idx + frames_num)/self.action_frames * fs) - audio4stream.shape[1])
                )
                audio4stream = torch.cat(
                    (audio4stream, pad_audio), dim=1
                )
            if audio4stream.shape[1] % fs:
                pad_audio = torch.zeros(
                    (1, fs - audio4stream.shape[1] % fs)
                )
                audio4stream = torch.cat(
                    (pad_audio, audio4stream), dim=1
                )
            audio4stream = [audio4stream.squeeze()]
            if self.encoder_type == "mamba":
                fbank_feature = self.fbank(audio4stream)[0]
            elif self.encoder_type == "zipformer2":
                fbank_feature = self.fbank.extract(audio4stream[0], sampling_rate=fs)
            fbank_feature_len = fbank_feature.size(0)
            
            sample = {}
            sample["input_ids"] = sample_input_ids,
            sample["attention_mask"] = sample_attention_mask,
            sample["labels"] = labels
            sample["fbank_feature"] = fbank_feature,
            sample["fbank_feature_len"] = fbank_feature_len
            sample["sent_lens"] = []
            sample["codecs"] = []
            sample["codec_lens"] = []
        # VLA Baseline (Img)
        else:        
            image_tokens = image_tokens[0:self.T,...]
            image_prompt = self.format_video_prompt(image_tokens)

            if self.use_gripper:
                gripper_tokens = gripper_tokens[0:self.T,...]
                gripper_prompt = self.format_video_prompt(gripper_tokens)
                image_prompt = image_prompt + gripper_prompt  

            if self.VL:
                p_prob_order = random.random()
                if p_prob_order < 0.5:
                    input = self.tokenizer.bos_token + prompt + image_prompt + self.tokenizer.eos_token
                else:
                    # input = self.tokenizer.bos_token + image_prompt + prompt
                    input = self.tokenizer.bos_token + self.chat_template.format(image_prompt=image_prompt, text_prompt=prompt) + self.tokenizer.eos_token
            else:
                input = self.tokenizer.bos_token + prompt + image_prompt 
            # 先不进行padding，后面统一padding
            sample = self.tokenizer(
                input,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            labels = sample["input_ids"]

            # only use vision loss
            if self.args.apply_loss_on_only_vision:
                labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)

            sample["labels"] = labels
            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            # based on the actions_format, append the action information to the sample
            if self.actions:
                if self.actions_format == "openvla":
                    action_sample = self.wrap_action_sequence(action_ids)
                    sample["input_ids"] = torch.cat([sample["input_ids"], action_sample], dim=-1)

                    # Update attention_mask
                    action_mask = torch.ones_like(action_sample, dtype=torch.long)
                    sample["attention_mask"] = torch.cat([sample["attention_mask"], action_mask], dim=-1)

                    action_labels = action_sample.clone()  # Clone action_sample for labels
                    sample["labels"] = torch.cat([sample["labels"], action_labels], dim=-1)
                
                # FAST
                elif self.actions_format == "fast":
                    if self.args.apply_loss_on_only_action:
                        sample['labels'] = torch.full_like(sample['labels'], self.args.ignore_index)
                    sample = self.append_action_to_sample(sample, action_ids)
                
                # Flow Matching
                elif self.actions_format == "continuous":
                    boa_token_id = self.tokenizer.encode(self.tokenizer.boa_token)[0]
                    sample = self.append_boa_to_sample(sample, [boa_token_id])
                    sample["action"] = action_continuous
            
            # finally, do padding
            sample = self.tokenizer.pad(
                sample,
                padding="max_length",
                return_tensors="pt"
            )

            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            if "labels" in sample:
                sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)
        return sample

class Emu3SpeechOnlyDataset(Dataset): 

    def __init__(self, args, tokenizer, generate=False, encoder_type="mamba"):
        super().__init__()

        self.data = json.load(open(args.data_path, "r"))["annotation"]
        self.tokenizer = tokenizer
        self.encoder_type = encoder_type
        self.time_block = args.time_block

        if self.encoder_type == "mamba":
            opts = kaldifeat.FbankOptions()
            opts.frame_opts.dither = 0
            opts.frame_opts.snip_edges = False
            opts.frame_opts.samp_freq = 16000 # only support 16k audio
            opts.mel_opts.num_bins = 80 # 80-bin
            opts.mel_opts.high_freq = -400

            self.fbank = kaldifeat.Fbank(opts)
        elif self.encoder_type == "zipformer2":
            self.fbank = Fbank(FbankConfig(num_mel_bins=128))
        
        self.token_per_second = args.token_per_second
        self.generate = generate
        self.prompts = {"dia_qa":"Please answer the question.","dia_asr":"Generate a transcript of the speech."}

        if self.generate:
            self.cosy_frontend = CosyVoice2(COSY_CKPT_PATH, load_jit=False, load_trt=False, fp16=False).frontend

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        while True:
            try:
                sent_lens = []
                codecs = []
                codec_lens = []

                sample = self.data[index]

                audio4stream, fs = torchaudio.load(sample["path"][0])
                assert fs == 16000
                frames = int(self.time_block * fs)
                speech_len = math.ceil(audio4stream.shape[1] / frames)

                prompt = self.prompts[sample["task"]]

                labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)

                text_prompt = self.tokenizer.bop_token + prompt + self.tokenizer.eop_token
                silence_output = self.tokenizer(self.tokenizer.silence_token + self.tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                for i in range(speech_len):
                    text_prompt += self.tokenizer.bos_token + self.tokenizer.eos_token + self.tokenizer.bot_token + self.tokenizer.silence_token + self.tokenizer.eot_token
                    
                sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                sample_input_ids = sample_text["input_ids"][0]
                sample_attention_mask = sample_text["attention_mask"][0]

                if len(self.tokenizer) > 150000:
                    bot_id = 151842
                else:
                    bot_id = 128259
                bot_idx = (sample_input_ids == bot_id).nonzero(as_tuple=True)[0]
                for idx in bot_idx:
                    labels[idx+1:idx+3] = silence_output

                if self.generate and sample["task"] != "dia_asr":
                    sample_answer_input_ids = []
                    if "sentences" in sample:
                        for sent in sample["sentences"]:
                            sent_input_ids = self.tokenizer(sent, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                            sample_answer_input_ids.append(sent_input_ids)
                            sent_lens.append(len(sent_input_ids))
                        sample_answer_input_ids = torch.cat(sample_answer_input_ids, dim=-1)
                    else:
                        sample_answer_input_ids = self.tokenizer(sample["text"], padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                        sent_lens.append(len(sample_answer_input_ids))
                    sample_answer_attention_mask = torch.ones_like(sample_answer_input_ids)

                    for path in sample["path_a"]:
                        audio4stream_a, _ = torchaudio.load(path)
                        audio4stream_a = audio4stream_a[:1].contiguous()
                        dp = self.cosy_frontend._extract_speech_token(audio4stream_a, cpu=True)
                        codecs.append(dp["speech_token"])
                        codec_lens.append(dp["speech_token_len"])

                    assert len(sent_lens) == len(codecs)                
                else:
                    answer_text = sample["text"]
                    sample_answer_text = self.tokenizer(answer_text, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                    sample_answer_input_ids = sample_answer_text["input_ids"][0]
                    sample_answer_attention_mask = sample_answer_text["attention_mask"][0]

                token_per_second = self.token_per_second

                t = 0
                answer_len = len(sample_answer_input_ids)
                while t < answer_len:
                    sample_speech = self.tokenizer(self.tokenizer.bos_token + self.tokenizer.eos_token + self.tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                    speech_input_ids = sample_speech["input_ids"][0]
                    speech_attention_mask = sample_speech["attention_mask"][0]
                    answer_input_ids = sample_answer_input_ids[t:t+token_per_second]
                    answer_attention_mask = sample_answer_attention_mask[t:t+token_per_second]
                    if t + token_per_second > answer_len:
                        sample_eot = self.tokenizer(self.tokenizer.silence_token + self.tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                        eot_input_ids = sample_eot["input_ids"][0]
                        eot_attention_mask = sample_eot["attention_mask"][0]
                    else:
                        sample_eot = self.tokenizer(self.tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                        eot_input_ids = sample_eot["input_ids"][0]
                        eot_attention_mask = sample_eot["attention_mask"][0]

                    sample_input_ids = torch.cat([sample_input_ids, speech_input_ids, answer_input_ids, eot_input_ids], dim=-1)  
                    sample_attention_mask = torch.cat([sample_attention_mask, speech_attention_mask, answer_attention_mask, eot_attention_mask], dim=-1)

                    action_start = len(sample_input_ids) - len(answer_input_ids) - len(eot_input_ids)
                    action_end = len(sample_input_ids)
                    labels[action_start:action_end] = torch.cat([answer_input_ids, eot_input_ids], dim=-1) 

                    t += token_per_second
                
                sample_speech = self.tokenizer(self.tokenizer.bos_token + self.tokenizer.eos_token + self.tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                speech_input_ids = sample_speech["input_ids"][0]
                speech_attention_mask = sample_speech["attention_mask"][0]
                sample_eot = self.tokenizer(self.tokenizer.silence_token + self.tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                eot_input_ids = sample_eot["input_ids"][0]
                eot_attention_mask = sample_eot["attention_mask"][0]
                
                sample_input_ids = torch.cat([sample_input_ids, speech_input_ids, eot_input_ids], dim=-1)  
                sample_attention_mask = torch.cat([sample_attention_mask, speech_attention_mask, eot_attention_mask], dim=-1)

                action_start = len(sample_input_ids) - len(eot_input_ids)
                action_end = len(sample_input_ids)
                labels[action_start:action_end] = eot_input_ids 
                    
                pad_audio = torch.zeros(
                    (1, (math.ceil(answer_len/token_per_second) + 2) * frames)
                )
                audio4stream = torch.cat(
                    (audio4stream, pad_audio), dim=1
                )
                if audio4stream.shape[1] % frames:
                    pad_audio = torch.zeros(
                        (1, frames - audio4stream.shape[1] % frames)
                    )
                    audio4stream = torch.cat(
                        (audio4stream, pad_audio), dim=1
                    )
                audio4stream = [audio4stream.squeeze()]
                if self.encoder_type == "mamba":
                    fbank_feature = self.fbank(audio4stream)[0]
                elif self.encoder_type == "zipformer2":
                    fbank_feature = self.fbank.extract(audio4stream[0], sampling_rate=fs)
                fbank_feature_len = fbank_feature.size(0)
                
                task = sample["task"]
                wav_path = sample["path"][0]
                sample = {}
                if fbank_feature_len > 180 * 100:
                    succ = False
                    index += 1234
                    index = index % len(self.data)
                else:
                    succ = True
                sample["input_ids"] = sample_input_ids,
                sample["attention_mask"] = sample_attention_mask,
                sample["labels"] = labels
                sample["fbank_feature"] = fbank_feature,
                sample["fbank_feature_len"] = fbank_feature_len
                sample["sent_lens"] = sent_lens
                sample["codecs"] = codecs
                sample["codec_lens"] = codec_lens
                # distribution
                sample["task"] = task
                sample["path"] = wav_path
            
            except Exception as e:
                succ = False
                print(e)
                import traceback
                traceback.print_exc()
                index += 1
                index = index % len(self.data)
                continue

            if succ:
                break

        return sample

class Emu3MixDataset(Emu3SpeechDataset):

    def __init__(self, args, tokenizer, moe, contemporary=False, stop=False, stop_ratio=0.1, vqa=False, context_vqa=False, generate=False, encoder_type="mamba"):
        super().__init__(args, tokenizer=tokenizer[0], moe=moe, encoder_type=encoder_type)

        self.data_speech = json.load(open(args.data_speech_path, "r"))["annotation"]
        self.moe = moe
        self.contemporary = contemporary
        self.vqa = vqa
        self.context_vqa = context_vqa
        self.stop = stop
        self.stop_ratio = stop_ratio
        self.refuse = args.refuse
        self.refuse_ratio = args.refuse_ratio
        self.mix_init_data = args.mix_init_data
        if stop:
            self.stop_base_dir = os.path.join(ELLSA_DATA_PATH,"interrupt/train")
            self.stop_speech = os.listdir(self.stop_base_dir)
        if self.vqa:
            self.vqa_data = json.load(open("LLaVA-Instruct/vqa_speech.json", "r"))["annotation"]
            self.vision_hub = os.path.join(ELLSA_BASE_PATH,"ckpt/Emu3-VisionVQ")
            self.image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)
            self.image_tokenizer = AutoModel.from_pretrained(self.vision_hub, trust_remote_code=True)
            self.image_processor.min_pixels = 80 * 80
        if self.context_vqa:
            self.context_questions = json.load(open(os.path.join(ELLSA_DATA_PATH,"json/split.json"), "r"))
            self.context_questions_speech = json.load(open(os.path.join(ELLSA_DATA_PATH,"json/10_vqa_questions_speech.json"), "r"))
        self.speech_tokenizer = tokenizer[0]
        self.vision_tokenizer = tokenizer[1]
        self.tokenizer = tokenizer[1]

        self.token_per_second = args.token_per_second
        # self.prompts = {"dia_qa":"Please answer the question.","dia_asr":"Generate a transcript of the speech.","vla":"Please act following the speech instruction.","contemporary":"Please answer by action or text following the speech instruction."}
        self.prompts = {"dia_qa":"Please answer the question.","dia_asr":"Generate a transcript of the speech.","vla":"Please answer by action or text following the speech instruction.","contemporary":"Please answer by action or text following the speech instruction.","vqa":"Please answer the question based on the image."}
        self.bos_id = self.speech_tokenizer(self.speech_tokenizer.bos_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        self.eos_id = self.speech_tokenizer(self.speech_tokenizer.eos_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        self.bot_id = self.speech_tokenizer(self.speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        self.eot_id = self.speech_tokenizer(self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        self.silence_id = self.speech_tokenizer(self.speech_tokenizer.silence_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        self.boi_id = self.vision_tokenizer(self.vision_tokenizer.boi_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        self.eoi_id = self.vision_tokenizer(self.vision_tokenizer.eoi_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        self.boa_id = self.vision_tokenizer(self.vision_tokenizer.boa_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        self.eoa_id = self.vision_tokenizer(self.vision_tokenizer.eoa_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        self.generate = generate

        if self.generate:
            self.cosy_frontend = CosyVoice2(COSY_CKPT_PATH, load_jit=False, load_trt=False, fp16=False).frontend
    
    def __len__(self):
        if self.contemporary:
            if self.vqa:
                return len(self.data_speech) + len(self.data_speech) + len(self.data_speech) + len(self.data_speech)
            else:
                return len(self.data_speech) + len(self.data_speech) + len(self.data_speech)
        else:
            return len(self.data_speech) + len(self.data_speech)
        
    def __getitem__(self, index: int):
        if self.contemporary:
            if self.vqa:
                if index < int(0.1 * len(self.data_speech)):
                    return self.get_speech_item(index)
                elif index < int(0.2 * len(self.data_speech)):  # 1.8 is a hyperparameter
                    return self.get_vla_item(index % len(self.data))
                elif index < int(5 * len(self.data_speech)):
                    return self.get_mix_item(index % len(self.data), random.randint(0, len(self.data_speech) - 1))
                else:
                    return self.get_vqa_item(index % len(self.vqa_data))
            elif self.generate:
                return self.get_speech_item(index % len(self.data_speech))
                """
                if index < int(2 * len(self.data_speech)):
                    return self.get_speech_item(index % len(self.data_speech))
                else:
                    return self.get_mix_item(index % len(self.data), random.randint(0, len(self.data_speech) - 1))
                """
            elif self.mix_init_data:
                if self.refuse and self.stop:
                    if index < int(0.9 * len(self.data_speech)): # 0.9
                        return self.get_speech_item(random.randint(0, len(self.data_speech) - 1))
                    elif index < int(1.0 * len(self.data_speech)):
                        return self.get_speech_only_item(random.randint(0, len(self.data_speech) - 1))
                    elif index < int(1.72 * len(self.data_speech)): # 1.72
                        return self.get_vla_item(index % len(self.data))
                    elif index < int(1.8 * len(self.data_speech)):  # 1.8 is a hyperparameter
                        return self.get_action_only_item(index % len(self.data))
                    else:
                        return self.get_mix_item(index % len(self.data), random.randint(0, len(self.data_speech) - 1))
            else:
                if self.refuse and self.stop:
                    if index < int(1 * len(self.data_speech)):
                        return self.get_speech_item(index)
                    elif index < int(1.8 * len(self.data_speech)):  # 1.8 is a hyperparameter
                        return self.get_vla_item(index % len(self.data))
                    else:
                        return self.get_mix_item(index % len(self.data), random.randint(0, len(self.data_speech) - 1))
                else:
                    if index < int(1 * len(self.data_speech)):
                        return self.get_speech_item(index)
                    elif index < int(1.9 * len(self.data_speech)):  # 1.8 is a hyperparameter
                        return self.get_vla_item(index % len(self.data))
                    else:
                        return self.get_mix_item(index % len(self.data), random.randint(0, len(self.data_speech) - 1))
        else:
            if index < len(self.data_speech):
                return self.get_speech_item(index)
            else:
                return self.get_vla_item((index - len(self.data_speech)) % len(self.data))
    
    def get_vla_item(self, index: int):
        sent_lens = []
        codecs = []
        codec_lens = []

        scene = self.data[index]

        if self.cfg:
            p_prob = random.random()
            if p_prob < self.args.null_prompt_prob:
                prompt = ""
            else:
                prompt = scene["speech"]
        else:
            prompt = scene["speech"]
        
        audio4stream, fs = torchaudio.load(scene["speech"])
        one_time_block_frames = int(fs * self.time_block)
        speech_len = math.ceil(audio4stream.shape[1] / one_time_block_frames * self.action_frames)

        # extend data
        image_tokens_path = scene["image"]
        image_tokens_path = [image_tokens_path[0]] * speech_len + image_tokens_path

        # handle different dataset fps for post training
        # fps = self.get_fps_for_path(image_tokens_path)
        # if fps is not None:
        #     self.action_frames = fps
        
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        
        # use action information
        if self.actions:
            action = scene["action"]
            if "calvin" in self.data_path:
                if action[0][-1] > 0:
                    dummy_action = self.dummy_action
                elif action[0][-1] < 0:
                    dummy_action = self.dummy_action_minus
            else:
                dummy_action = self.dummy_action
            action = np.concatenate((np.array([dummy_action] * speech_len),action),axis=0)
            if self.use_gripper:
                gripper = scene["gripper_image"]
                gripper = [gripper[0]] * speech_len + gripper
                image_tokens, action_tokens, gripper_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, gripper=gripper, return_start=True, select_end=600)
            else:
                image_tokens, action_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, return_start=True, select_end=600)
            
            if self.video_format == "interleave":
                if self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                    action_ids = self.action_tokenizer(action_tokens)
                    self.last_vocab_idx = self.vision_tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
        
        if self.video_format == "interleave":
            text_prompt = self.speech_tokenizer.bop_token + self.prompts["vla"] + self.speech_tokenizer.eop_token
            for i in range(math.ceil(start_idx/self.action_frames)):
                text_prompt += self.speech_tokenizer.bos_token + self.speech_tokenizer.eos_token + self.speech_tokenizer.bot_token + self.speech_tokenizer.silence_token + self.speech_tokenizer.eot_token
            image_tokens = image_tokens[0::self.action_frames,...]
            if self.use_gripper:
                gripper_tokens = gripper_tokens[0::self.action_frames,...]
            
            sample_text = self.speech_tokenizer(text_prompt, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.vision_tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                if self.use_gripper:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                    image_prompt += gripper_prompt
                sample_speech = self.speech_tokenizer(self.speech_tokenizer.bos_token + self.speech_tokenizer.eos_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                speech_input_ids = sample_speech["input_ids"][0]
                speech_attention_mask = sample_speech["attention_mask"][0]
                sample_img = self.vision_tokenizer(image_prompt, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                if self.actions:
                    if self.actions_format == "fast":
                        sample_answer = self.speech_tokenizer(self.speech_tokenizer.bot_token + self.speech_tokenizer.silence_token + self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                        answer_input_ids = sample_answer["input_ids"][0]
                        answer_attention_mask = sample_answer["attention_mask"][0]
                        sample_input_ids = torch.cat([sample_input_ids, speech_input_ids, image_input_ids, answer_input_ids], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, speech_attention_mask, image_attention_mask, answer_attention_mask], dim=-1) 
                        answer_start = len(sample_input_ids) - len(answer_input_ids) + 1
                        answer_end = len(sample_input_ids)
                        labels[answer_start:answer_end] = answer_input_ids[1:]

                        action_sample = self.wrap_action_sequence(action_ids[i].tolist()) 
                        sample_input_ids = torch.cat([sample_input_ids, action_sample], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1) 
                        action_start = len(sample_input_ids) - len(action_sample)
                        action_end = len(sample_input_ids)
                        if self.args.apply_loss_on_only_action:  
                            labels[action_start:action_end] = action_sample
                        else:  # Otherwise, fill both vision and action parts in the labels
                            labels[action_start-len(image_input_ids):action_start] = image_input_ids  
                            labels[action_start:action_end] = action_sample 
                else:
                    sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                    sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                    labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids

                    sample_input_ids = torch.cat([sample_input_ids, speech_input_ids, answer_input_ids, eot_input_ids], dim=-1)  
                    sample_attention_mask = torch.cat([sample_attention_mask, speech_attention_mask, answer_attention_mask, eot_attention_mask], dim=-1)

                    action_start = len(sample_input_ids) - len(answer_input_ids) - len(eot_input_ids)
                    action_end = len(sample_input_ids)
                    labels[action_start:action_end] = torch.cat([answer_input_ids, eot_input_ids], dim=-1)
            
            if start_idx % self.action_frames != 0:
                pad_audio = torch.zeros(
                    (1, int((self.action_frames - start_idx % self.action_frames)/self.action_frames * one_time_block_frames))
                )
                audio4stream = torch.cat(
                    (pad_audio, audio4stream), dim=1
                )
            if math.ceil((start_idx + frames_num)/self.action_frames) * one_time_block_frames > audio4stream.shape[1]:
                pad_audio = torch.zeros(
                    (1, math.ceil((start_idx + frames_num)/self.action_frames) * one_time_block_frames - audio4stream.shape[1])
                )
                audio4stream = torch.cat(
                    (audio4stream, pad_audio), dim=1
                )
            audio4stream = [audio4stream.squeeze()]
            if self.encoder_type == "mamba":
                fbank_feature = self.fbank(audio4stream)[0]
            elif self.encoder_type == "zipformer2":
                fbank_feature = self.fbank.extract(audio4stream[0], sampling_rate=fs)
            fbank_feature_len = fbank_feature.size(0)
            
            sample = {}
            sample["input_ids"] = sample_input_ids,
            sample["attention_mask"] = sample_attention_mask,
            sample["labels"] = labels
            sample["fbank_feature"] = fbank_feature,
            sample["fbank_feature_len"] = fbank_feature_len
            sample["sent_lens"] = sent_lens
            sample["codecs"] = codecs
            sample["codec_lens"] = codec_lens

        return sample
    
    def get_speech_item(self, index: int):
        while True:
            try:
                sent_lens = []
                codecs = []
                codec_lens = []

                sample = self.data_speech[index]

                audio4stream, fs = torchaudio.load(sample["path"][0])
                wav_path = sample["path"][0]
                frames = int(self.time_block * fs)
                speech_len = math.ceil(audio4stream.shape[1] / frames)

                prompt = self.prompts[sample["task"]]

                labels = torch.full((self.vision_tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)

                text_prompt = self.speech_tokenizer.bop_token + prompt + self.speech_tokenizer.eop_token
                silence_output = self.speech_tokenizer(self.speech_tokenizer.silence_token + self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                eoa_id = self.vision_tokenizer(self.vision_tokenizer.eoa_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

                action_tokens = np.array([self.dummy_action] * speech_len * self.action_frames)
                if isinstance(action_tokens, list):
                    tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                    # Concatenate tensors along the first dimension
                    action_tokens = torch.cat(tensor_list, dim=0)
                action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                action_ids = self.action_tokenizer(action_tokens)
                self.last_vocab_idx = self.vision_tokenizer.pad_token_id - 1
                action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                    
                sample_text = self.speech_tokenizer(text_prompt, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                sample_input_ids = sample_text["input_ids"][0]
                for i in range(speech_len):
                    sample_input_ids = torch.cat([sample_input_ids, self.bos_id, self.eos_id, self.boi_id, self.eoi_id, self.bot_id, self.silence_id, self.eot_id, self.boa_id, self.eoa_id], dim=-1)
                sample_attention_mask = torch.ones_like(sample_input_ids, dtype=torch.long)
                
                boa_id = 151844
                boa_idx = (sample_input_ids == boa_id).nonzero(as_tuple=True)[0]
                step = 0
                for idx in boa_idx:
                    sample_input_ids = torch.cat((sample_input_ids[:idx+1],action_ids[step],sample_input_ids[idx+1:]),dim=-1)
                    labels[idx+1:idx+1+len(action_ids[step])] = action_ids[step]
                    labels[idx+1+len(action_ids[step])] = eoa_id
                    step += 1
                    boa_idx += len(action_ids[step-1])

                bot_id = 128259
                bot_idx = (sample_input_ids == bot_id).nonzero(as_tuple=True)[0]
                for idx in bot_idx:
                    labels[idx+1:idx+3] = silence_output
                
                if self.generate and sample["task"] != "dia_asr":
                    sample_answer_input_ids = []
                    if "sentences" in sample:
                        for sent in sample["sentences"]:
                            sent_input_ids = self.speech_tokenizer(sent, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                            sample_answer_input_ids.append(sent_input_ids)
                            sent_lens.append(len(sent_input_ids))
                        sample_answer_input_ids = torch.cat(sample_answer_input_ids, dim=-1)
                    else:
                        sample_answer_input_ids = self.speech_tokenizer(sample["text"], padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                        sent_lens.append(len(sample_answer_input_ids))
                    sample_answer_attention_mask = torch.ones_like(sample_answer_input_ids)

                    for path in sample["path_a"]:
                        audio4stream_a, _ = torchaudio.load(path)
                        audio4stream_a = audio4stream_a[:1].contiguous()
                        dp = self.cosy_frontend._extract_speech_token(audio4stream_a, cpu=True)
                        codecs.append(dp["speech_token"])
                        codec_lens.append(dp["speech_token_len"])

                    assert len(sent_lens) == len(codecs)
                else:
                    answer_text = sample["text"]
                    sample_answer_text = self.speech_tokenizer(answer_text, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                    sample_answer_input_ids = sample_answer_text["input_ids"][0]
                    sample_answer_attention_mask = sample_answer_text["attention_mask"][0]

                token_per_second = self.token_per_second

                t = 0
                answer_len = len(sample_answer_input_ids)

                action_tokens = np.array([self.dummy_action] * (int(answer_len / token_per_second)+2) * self.action_frames)
                if isinstance(action_tokens, list):
                    tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                    # Concatenate tensors along the first dimension
                    action_tokens = torch.cat(tensor_list, dim=0)
                action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                action_ids = self.action_tokenizer(action_tokens)
                self.last_vocab_idx = self.vision_tokenizer.pad_token_id - 1
                action_ids_2 = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                i = 0

                while t < answer_len:
                    sample_speech = self.speech_tokenizer(self.speech_tokenizer.bos_token + self.speech_tokenizer.eos_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                    speech_input_ids = sample_speech["input_ids"][0]
                    speech_attention_mask = sample_speech["attention_mask"][0]
                    sample_speech_2 = self.vision_tokenizer(self.vision_tokenizer.boi_token + self.vision_tokenizer.eoi_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                    speech_input_ids_2 = sample_speech_2["input_ids"][0]
                    speech_attention_mask_2 = sample_speech_2["attention_mask"][0]
                    sample_speech_3 = self.speech_tokenizer(self.speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                    speech_input_ids_3 = sample_speech_3["input_ids"][0]
                    speech_attention_mask_3 = sample_speech_3["attention_mask"][0]
                    answer_input_ids = sample_answer_input_ids[t:t+token_per_second]
                    answer_attention_mask = sample_answer_attention_mask[t:t+token_per_second]
                    if t + token_per_second > answer_len:
                        sample_eot = self.speech_tokenizer(self.speech_tokenizer.silence_token + self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                        eot_input_ids = sample_eot["input_ids"][0]
                        eot_attention_mask = sample_eot["attention_mask"][0]
                    else:
                        sample_eot = self.speech_tokenizer(self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                        eot_input_ids = sample_eot["input_ids"][0]
                        eot_attention_mask = sample_eot["attention_mask"][0]

                    sample_input_ids = torch.cat([sample_input_ids, speech_input_ids, speech_input_ids_2, speech_input_ids_3, answer_input_ids, eot_input_ids], dim=-1)  
                    sample_attention_mask = torch.cat([sample_attention_mask, speech_attention_mask, speech_attention_mask_2, speech_attention_mask_3, answer_attention_mask, eot_attention_mask], dim=-1)

                    action_start = len(sample_input_ids) - len(answer_input_ids) - len(eot_input_ids)
                    action_end = len(sample_input_ids)
                    labels[action_start:action_end] = torch.cat([answer_input_ids, eot_input_ids], dim=-1)

                    action_sample = self.wrap_action_sequence(action_ids_2[i].tolist()) 
                    sample_input_ids = torch.cat([sample_input_ids, action_sample], dim=-1)  
                    sample_attention_mask = torch.cat([sample_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1)

                    action_start = len(sample_input_ids) - len(action_sample)
                    action_end = len(sample_input_ids)
                    labels[action_start:action_end] = action_sample  

                    t += token_per_second
                    i += 1
                
                # add silence as end of decoding
                sample_speech = self.speech_tokenizer(self.speech_tokenizer.bos_token + self.speech_tokenizer.eos_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                speech_input_ids = sample_speech["input_ids"][0]
                speech_attention_mask = sample_speech["attention_mask"][0]
                sample_speech_2 = self.vision_tokenizer(self.vision_tokenizer.boi_token + self.vision_tokenizer.eoi_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                speech_input_ids_2 = sample_speech_2["input_ids"][0]
                speech_attention_mask_2 = sample_speech_2["attention_mask"][0]
                sample_speech_3 = self.speech_tokenizer(self.speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                speech_input_ids_3 = sample_speech_3["input_ids"][0]
                speech_attention_mask_3 = sample_speech_3["attention_mask"][0]
                sample_answer = self.speech_tokenizer(self.speech_tokenizer.silence_token + self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                answer_input_ids = sample_answer["input_ids"][0]
                answer_attention_mask = sample_answer["attention_mask"][0]

                sample_input_ids = torch.cat([sample_input_ids, speech_input_ids, speech_input_ids_2, speech_input_ids_3, answer_input_ids], dim=-1)  
                sample_attention_mask = torch.cat([sample_attention_mask, speech_attention_mask, speech_attention_mask_2, speech_attention_mask_3, answer_attention_mask], dim=-1)

                action_start = len(sample_input_ids) - len(answer_input_ids)
                action_end = len(sample_input_ids)
                labels[action_start:action_end] = answer_input_ids

                action_sample = self.wrap_action_sequence(action_ids_2[i].tolist()) 
                sample_input_ids = torch.cat([sample_input_ids, action_sample], dim=-1)  
                sample_attention_mask = torch.cat([sample_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1)

                action_start = len(sample_input_ids) - len(action_sample)
                action_end = len(sample_input_ids)
                labels[action_start:action_end] = action_sample 

                # pad audio
                pad_audio = torch.zeros(
                    (1, (math.ceil(answer_len/token_per_second) + 2) * frames)
                )
                audio4stream = torch.cat(
                    (audio4stream, pad_audio), dim=1
                )
                if audio4stream.shape[1] % frames:
                    pad_audio = torch.zeros(
                        (1, frames - audio4stream.shape[1] % frames)
                    )
                    audio4stream = torch.cat(
                        (audio4stream, pad_audio), dim=1
                    )
                audio4stream = [audio4stream.squeeze()]
                if self.encoder_type == "mamba":
                    fbank_feature = self.fbank(audio4stream)[0]
                elif self.encoder_type == "zipformer2":
                    fbank_feature = self.fbank.extract(audio4stream[0], sampling_rate=fs)
                fbank_feature_len = fbank_feature.size(0)
                
                sample = {}
                if fbank_feature_len > 120 * 100:
                    succ = False
                    index += 1234
                    index = index % len(self.data)
                else:
                    succ = True
                sample["input_ids"] = sample_input_ids,
                sample["attention_mask"] = sample_attention_mask,
                sample["labels"] = labels
                sample["fbank_feature"] = fbank_feature,
                sample["fbank_feature_len"] = fbank_feature_len
                sample["sent_lens"] = sent_lens
                sample["codecs"] = codecs
                sample["codec_lens"] = codec_lens
            
            except Exception as e:
                succ = False
                print(e)
                import traceback
                traceback.print_exc()
                index += 1
                index = index % len(self.data_speech)
                continue

            if succ:
                break

        return sample

    def get_mix_item(self, index_vla: int, index_speech: int):
        scene = self.data[index_vla]

        if self.cfg:
            p_prob = random.random()
            if p_prob < self.args.null_prompt_prob:
                prompt = ""
            else:
                prompt = scene["speech"]
        else:
            prompt = scene["speech"]
        
        random_num = random.random()
        use_stop = False
        use_refuse = False
        if random_num < self.stop_ratio and self.stop:
            use_stop = True
        elif random_num < self.stop_ratio + self.refuse_ratio and random_num > self.stop_ratio and self.refuse and "defective" in scene and scene["defective"]["succ"]:
            use_refuse = True

        if use_refuse:
            audio4stream, fs = torchaudio.load(scene["defective"]["path_i"])
        else:
            audio4stream, fs = torchaudio.load(scene["speech"])
        one_time_block_frames = int(fs * self.time_block)
        speech_len_vla = math.ceil(audio4stream.shape[1] / one_time_block_frames * self.action_frames)

        token_per_second = self.token_per_second

        while True:
            sent_lens = []
            codecs = []
            codec_lens = []

            try:
                if use_stop:
                    sample = os.path.join(self.stop_base_dir,random.choice(self.stop_speech))
                    audio4stream_speech, fs = torchaudio.load(sample)
                    if fs != 16000:
                        resample_transform = torchaudio.transforms.Resample(fs, 16000)
                        fs = 16000
                        audio4stream_speech = resample_transform(audio4stream_speech)
                    speech_len_speech = math.ceil(audio4stream_speech.shape[1] / one_time_block_frames * self.action_frames)

                    sample_answer_text = self.speech_tokenizer("Action cancelled." + self.speech_tokenizer.silence_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                    sample_answer_input_ids = sample_answer_text["input_ids"][0]
                    sample_answer_attention_mask = sample_answer_text["attention_mask"][0]
                    answer_len = len(sample_answer_input_ids)
                    text_len_speech = math.ceil(answer_len/token_per_second) + 1
                    use_context_qa = False

                    """ # used stop
                    answer_len = 0
                    text_len_speech = 2
                    use_context_qa = False
                    """
                elif use_refuse:
                    speech_len_speech = 0
                    audio4stream_speech = torch.zeros((1, one_time_block_frames))
                    sample_answer_text = self.speech_tokenizer("Action cancelled. " + scene["defective"]["Response"] + self.speech_tokenizer.silence_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                    sample_answer_input_ids = sample_answer_text["input_ids"][0]
                    sample_answer_attention_mask = sample_answer_text["attention_mask"][0]
                    answer_len = len(sample_answer_input_ids)
                    text_len_speech = math.ceil(answer_len/token_per_second) + 1
                    use_context_qa = False
                else:
                    if self.context_vqa:
                        sample_name = scene["speech"].split("/")[-2]
                        if sample_name in self.context_questions.keys():
                            use_context_qa = random.random() < 0.8
                            # use_context_qa = True
                        else:
                            use_context_qa = False
                    else:
                        use_context_qa = False
                    if use_context_qa:
                        context_qa = self.context_questions[sample_name]
                        idx = random.randint(0,len(context_qa)-1)
                        qa_text = list(context_qa[idx].keys())[0]
                        qa_speech = self.context_questions_speech[qa_text]

                        audio4stream_speech, fs = torchaudio.load(qa_speech)
                        speech_len_speech = math.ceil(audio4stream_speech.shape[1] / one_time_block_frames * self.action_frames)

                        qa_answer = random.choice(self.context_questions[sample_name][idx][qa_text])
                        sample_answer_text = self.speech_tokenizer(qa_answer[2] + self.speech_tokenizer.silence_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                        sample_answer_input_ids = sample_answer_text["input_ids"][0]
                        sample_answer_attention_mask = sample_answer_text["attention_mask"][0]
                        answer_len = len(sample_answer_input_ids)
                        text_len_speech = math.ceil(answer_len/token_per_second) + 1
                        
                    else:
                        sample = self.data_speech[index_speech]
                        if sample["task"] == "dia_asr":
                            index_speech += random.randint(100000,500000)
                            index_speech = index_speech % len(self.data_speech)
                            continue

                        audio4stream_speech, fs = torchaudio.load(sample["path"][0])
                        speech_len_speech = math.ceil(audio4stream_speech.shape[1] / one_time_block_frames * self.action_frames)

                        if self.generate and sample["task"] != "dia_asr":
                            sample_answer_input_ids = []
                            if "sentences" in sample:
                                for sent in sample["sentences"]:
                                    sent_input_ids = self.speech_tokenizer(sent, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                                    sample_answer_input_ids.append(sent_input_ids)
                                    sent_lens.append(len(sent_input_ids))
                                sample_answer_input_ids = torch.cat(sample_answer_input_ids, dim=-1)
                            else:
                                sample_answer_input_ids = self.speech_tokenizer(sample["text"], padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                                sent_lens.append(len(sample_answer_input_ids))
                            sample_answer_attention_mask = torch.ones_like(sample_answer_input_ids)

                            for path in sample["path_a"]:
                                audio4stream_a, _ = torchaudio.load(path)
                                audio4stream_a = audio4stream_a[:1].contiguous()
                                dp = self.cosy_frontend._extract_speech_token(audio4stream_a, cpu=True)
                                codecs.append(dp["speech_token"])
                                codec_lens.append(dp["speech_token_len"])

                            assert len(sent_lens) == len(codecs)
                        else:
                            sample_answer_text = self.speech_tokenizer(sample["text"] + self.speech_tokenizer.silence_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                            sample_answer_input_ids = sample_answer_text["input_ids"][0]
                            sample_answer_attention_mask = sample_answer_text["attention_mask"][0]

                        answer_len = len(sample_answer_input_ids)
                        text_len_speech = math.ceil(answer_len/token_per_second) + 1

                if use_context_qa:
                    middle_break = random.randint(max(speech_len_speech+5,qa_answer[0]),qa_answer[1]-5)
                elif use_refuse:
                    middle_break = 0
                else:
                    if "calvin" in self.data_path:
                        middle_break = random.randint(self.action_frames*1,self.action_frames*4)
                    else:
                        middle_break = random.randint(self.action_frames*2,self.action_frames*8) # used to be (20,50)
                pad_audio = torch.zeros(
                    (1, int(middle_break * one_time_block_frames / self.action_frames))
                )
                audio4stream = torch.cat(
                    (audio4stream, pad_audio, audio4stream_speech), dim=1
                )

                if middle_break + speech_len_speech + text_len_speech * self.action_frames > 120 * 10:
                    succ = False
                    index_speech += 1
                    index_speech = index_speech % len(self.data_speech)
                else:
                    succ = True

            except Exception as e:
                succ = False
                print(e)
                import traceback
                traceback.print_exc()
                index_speech += 1
                index_speech = index_speech % len(self.data_speech)
                continue

            if succ:
                break

        # extend data
        image_tokens_path = scene["image"]
        if use_refuse:
            image_tokens_path = [image_tokens_path[0]]
        image_len = len(image_tokens_path)
        speech_qa_len = middle_break + speech_len_speech + text_len_speech * self.action_frames
        image_tokens_path = [image_tokens_path[0]] * speech_len_vla + image_tokens_path
        if speech_qa_len > image_len:
            image_tokens_path = image_tokens_path + [image_tokens_path[-1]] * (speech_qa_len - image_len)
        """ # used stop
        if use_stop:
            end_frame = speech_len_vla + middle_break + speech_len_speech
            image_tokens_path = image_tokens_path[:end_frame] + [image_tokens_path[end_frame]] * (text_len_speech * 10)
        """

        # handle different dataset fps for post training
        # fps = self.get_fps_for_path(image_tokens_path)
        # if fps is not None:
        #     self.action_frames = fps
        
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        
        # use action information
        if self.actions:
            if use_refuse:
                action = np.array([self.dummy_action])
            else:
                action = scene["action"]
            if "calvin" in self.data_path:
                if action[0][-1] > 0:
                    dummy_action = self.dummy_action
                elif action[0][-1] < 0:
                    dummy_action = self.dummy_action_minus
            else:
                dummy_action = self.dummy_action
            action = np.concatenate((np.array([dummy_action] * speech_len_vla),action),axis=0)
            if speech_qa_len > image_len:
                action = np.concatenate((action,np.array([self.dummy_action] * (speech_qa_len - image_len))),axis=0)
            """ # used stop
            if use_stop:
                action = np.concatenate((action[:end_frame],np.array([self.dummy_action] * (text_len_speech * 10))),axis=0)
            """
            if self.use_gripper:
                gripper = scene["gripper_image"]
                if use_refuse:
                    gripper = [gripper[0]]
                gripper = [gripper[0]] * speech_len_vla + gripper
                if speech_qa_len > image_len:
                    gripper = gripper + [gripper[-1]] * (speech_qa_len - image_len)
                """ # used stop
                if use_stop:
                    gripper = gripper[:end_frame] + [gripper[end_frame]] * (text_len_speech * 10)
                    image_tokens, action_tokens, gripper_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, gripper=gripper, return_start=True, select_start=end_frame-20)
                    # image_tokens, action_tokens, gripper_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, gripper=gripper, return_start=True, select_start=end_frame-30)
                elif use_context_qa:
                    image_tokens, action_tokens, gripper_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, gripper=gripper, return_start=True, select_start=speech_len_vla+speech_qa_len-20, select_end=speech_len_vla+speech_qa_len)
                """
                if use_stop:
                    image_tokens, action_tokens, gripper_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, gripper=gripper, return_start=True, select_start=speech_len_vla, select_end=speech_len_vla+speech_qa_len)
                elif use_context_qa:
                    image_tokens, action_tokens, gripper_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, gripper=gripper, return_start=True, select_start=speech_len_vla+middle_break-15, select_end=speech_len_vla+speech_qa_len)
                    # image_tokens, action_tokens, gripper_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, gripper=gripper, return_start=True, select_start=speech_len_vla+middle_break-10, select_end=speech_len_vla+speech_qa_len)
                else:
                    image_tokens, action_tokens, gripper_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, gripper=gripper, return_start=True)
            else:
                image_tokens, action_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, return_start=True, select_start=end_frame-40)
            
            if self.video_format == "interleave":
                if self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                    action_ids = self.action_tokenizer(action_tokens)
                    self.last_vocab_idx = self.vision_tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
        
        if self.video_format == "interleave":
            text_prompt = self.speech_tokenizer.bop_token + self.prompts["contemporary"] + self.speech_tokenizer.eop_token
            qa_start_idx = speech_len_vla + middle_break + speech_len_speech
            text_idx = 0
            if start_idx % self.action_frames:
                qa_start_idx += self.action_frames - start_idx % self.action_frames
            """ # used stop
            if use_stop:
                pre_silence = math.ceil(start_idx/10)
            else:
                pre_silence = min(math.ceil(start_idx/10),math.ceil(qa_start_idx/10))
            """
            pre_silence = min(math.ceil(start_idx/self.action_frames),math.ceil(qa_start_idx/self.action_frames))
            for i in range(pre_silence):
                text_prompt += self.speech_tokenizer.bos_token + self.speech_tokenizer.eos_token + self.speech_tokenizer.bot_token + self.speech_tokenizer.silence_token + self.speech_tokenizer.eot_token
            image_tokens = image_tokens[0::self.action_frames,...]
            if self.use_gripper:
                gripper_tokens = gripper_tokens[0::self.action_frames,...]
            
            sample_text = self.speech_tokenizer(text_prompt, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.vision_tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            if pre_silence < math.ceil(start_idx/self.action_frames):
                for i in range(math.ceil(start_idx/self.action_frames) - pre_silence):
                    sample_speech = self.speech_tokenizer(self.speech_tokenizer.bos_token + self.speech_tokenizer.eos_token + self.speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                    speech_input_ids = sample_speech["input_ids"][0]
                    speech_attention_mask = sample_speech["attention_mask"][0]
                    answer_input_ids = sample_answer_input_ids[text_idx:text_idx+token_per_second]
                    answer_attention_mask = sample_answer_attention_mask[text_idx:text_idx+token_per_second]
                    sample_eot = self.speech_tokenizer(self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                    eot_input_ids = sample_eot["input_ids"][0]
                    eot_attention_mask = sample_eot["attention_mask"][0]

                    sample_input_ids = torch.cat([sample_input_ids, speech_input_ids, answer_input_ids, eot_input_ids], dim=-1)  
                    sample_attention_mask = torch.cat([sample_attention_mask, speech_attention_mask, answer_attention_mask, eot_attention_mask], dim=-1)

                    text_idx += token_per_second

                    action_start = len(sample_input_ids) - len(answer_input_ids) - len(eot_input_ids)
                    action_end = len(sample_input_ids)
                    if self.generate:
                        labels[action_start:action_end] = torch.cat([answer_input_ids, eot_input_ids], dim=-1)

            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                if self.use_gripper:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                    image_prompt += gripper_prompt
                sample_speech = self.speech_tokenizer(self.speech_tokenizer.bos_token + self.speech_tokenizer.eos_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                speech_input_ids = sample_speech["input_ids"][0]
                speech_attention_mask = sample_speech["attention_mask"][0]
                sample_img = self.vision_tokenizer(image_prompt, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                if self.actions:
                    if self.actions_format == "fast":
                        if (pre_silence < math.ceil(qa_start_idx/self.action_frames) and text_idx == 0) or text_idx >= answer_len: # or use_stop # used stop
                            sample_answer = self.speech_tokenizer(self.speech_tokenizer.bot_token + self.speech_tokenizer.silence_token + self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                            answer_input_ids = sample_answer["input_ids"][0]
                            answer_attention_mask = sample_answer["attention_mask"][0]
                            pre_silence += 1
                        elif text_idx < answer_len:
                            sample_bot = self.speech_tokenizer(self.speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                            bot_input_ids = sample_bot["input_ids"][0]
                            bot_attention_mask = sample_bot["attention_mask"][0]
                            answer_input_ids = sample_answer_input_ids[text_idx:text_idx+token_per_second]
                            answer_attention_mask = sample_answer_attention_mask[text_idx:text_idx+token_per_second]
                            sample_eot = self.speech_tokenizer(self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                            eot_input_ids = sample_eot["input_ids"][0]
                            eot_attention_mask = sample_eot["attention_mask"][0]
                            answer_input_ids = torch.cat([bot_input_ids, answer_input_ids, eot_input_ids], dim=-1)  
                            answer_attention_mask = torch.cat([bot_attention_mask, answer_attention_mask, eot_attention_mask], dim=-1)

                            text_idx += token_per_second

                        sample_input_ids = torch.cat([sample_input_ids, speech_input_ids, image_input_ids, answer_input_ids], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, speech_attention_mask, image_attention_mask, answer_attention_mask], dim=-1) 
                        answer_start = len(sample_input_ids) - len(answer_input_ids) + 1
                        answer_end = len(sample_input_ids)
                        labels[answer_start:answer_end] = answer_input_ids[1:]

                        action_sample = self.wrap_action_sequence(action_ids[i].tolist()) 
                        sample_input_ids = torch.cat([sample_input_ids, action_sample], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1) 
                        action_start = len(sample_input_ids) - len(action_sample)
                        action_end = len(sample_input_ids)
                        if self.args.apply_loss_on_only_action:  
                            labels[action_start:action_end] = action_sample
                        else:  # Otherwise, fill both vision and action parts in the labels
                            labels[action_start-len(image_input_ids):action_start] = image_input_ids  
                            labels[action_start:action_end] = action_sample 
                else:
                    sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                    sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                    labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids

                    sample_input_ids = torch.cat([sample_input_ids, speech_input_ids, answer_input_ids, eot_input_ids], dim=-1)  
                    sample_attention_mask = torch.cat([sample_attention_mask, speech_attention_mask, answer_attention_mask, eot_attention_mask], dim=-1)

                    action_start = len(sample_input_ids) - len(answer_input_ids) - len(eot_input_ids)
                    action_end = len(sample_input_ids)
                    labels[action_start:action_end] = torch.cat([answer_input_ids, eot_input_ids], dim=-1)
            
            if start_idx % self.action_frames != 0:
                pad_audio = torch.zeros(
                    (1, int((self.action_frames - start_idx % self.action_frames)/self.action_frames * one_time_block_frames))
                )
                audio4stream = torch.cat(
                    (pad_audio, audio4stream), dim=1
                )
            if math.ceil((start_idx + frames_num)/self.action_frames) * one_time_block_frames > audio4stream.shape[1]:
                pad_audio = torch.zeros(
                    (1, math.ceil((start_idx + frames_num)/self.action_frames) * one_time_block_frames - audio4stream.shape[1])
                )
                audio4stream = torch.cat(
                    (audio4stream, pad_audio), dim=1
                )
            audio4stream = [audio4stream.squeeze()]
            if self.encoder_type == "mamba":
                fbank_feature = self.fbank(audio4stream)[0]
            elif self.encoder_type == "zipformer2":
                fbank_feature = self.fbank.extract(audio4stream[0], sampling_rate=fs)
            fbank_feature_len = fbank_feature.size(0)
            
            sample = {}
            sample["input_ids"] = sample_input_ids,
            sample["attention_mask"] = sample_attention_mask,
            sample["labels"] = labels
            sample["fbank_feature"] = fbank_feature,
            sample["fbank_feature_len"] = fbank_feature_len
            sample["sent_lens"] = sent_lens
            sample["codecs"] = codecs
            sample["codec_lens"] = codec_lens
            sample["context_qa"] = use_context_qa
        
        return sample

    def get_vqa_item(self, index: int):
        sample = self.vqa_data[index]

        audio4stream, fs = torchaudio.load(sample["speech_path"])
        speech_len = math.ceil(audio4stream.shape[1] / fs)

        prompt = self.prompts[sample["task"]]

        labels = torch.full((self.vision_tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)

        text_prompt = self.speech_tokenizer.bop_token + prompt + self.speech_tokenizer.eop_token
        silence_output = self.speech_tokenizer(self.speech_tokenizer.silence_token + self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        eoa_id = self.vision_tokenizer(self.vision_tokenizer.eoa_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

        action_tokens = np.array([self.dummy_action] * speech_len * self.action_frames)
        if isinstance(action_tokens, list):
            tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
            # Concatenate tensors along the first dimension
            action_tokens = torch.cat(tensor_list, dim=0)
        action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
        action_ids = self.action_tokenizer(action_tokens)
        self.last_vocab_idx = self.vision_tokenizer.pad_token_id - 1
        action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]

        self.image_tokenizer.eval()
        img_list = [sample["image_path"]]
        selected_frames = [Image.open(img_path).resize((200, 200)) for img_path in img_list]
        selected_frames = [self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0) for img in selected_frames]

        tensor_frames = torch.stack(selected_frames, dim=0)
        with torch.no_grad():
            image_code = self.image_tokenizer.encode(tensor_frames)
        image_prompt = self.format_video_prompt(image_code)
        sample_img = self.vision_tokenizer(image_prompt, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
        image_input_ids = sample_img["input_ids"][0]
        image_attention_mask = sample_img["attention_mask"][0]
                    
        sample_text = self.speech_tokenizer(text_prompt, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
        sample_input_ids = sample_text["input_ids"][0]
        sample_input_ids = torch.cat([sample_input_ids, self.bos_id, self.eos_id, image_input_ids, self.bot_id, self.silence_id, self.eot_id, self.boa_id, self.eoa_id], dim=-1)
        for i in range(speech_len - 1):
            sample_input_ids = torch.cat([sample_input_ids, self.bos_id, self.eos_id, self.boi_id, self.eoi_id, self.bot_id, self.silence_id, self.eot_id, self.boa_id, self.eoa_id], dim=-1)
        sample_attention_mask = torch.ones_like(sample_input_ids, dtype=torch.long)
                
        boa_id = 151844
        boa_idx = (sample_input_ids == boa_id).nonzero(as_tuple=True)[0]
        step = 0
        for idx in boa_idx:
            sample_input_ids = torch.cat((sample_input_ids[:idx+1],action_ids[step],sample_input_ids[idx+1:]),dim=-1)
            labels[idx+1:idx+1+len(action_ids[step])] = action_ids[step]
            labels[idx+1+len(action_ids[step])] = eoa_id
            step += 1
            boa_idx += len(action_ids[step-1])

        bot_id = 128259
        bot_idx = (sample_input_ids == bot_id).nonzero(as_tuple=True)[0]
        for idx in bot_idx:
            labels[idx+1:idx+3] = silence_output

        answer_text = sample["text"]
        sample_answer_text = self.speech_tokenizer(answer_text, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
        sample_answer_input_ids = sample_answer_text["input_ids"][0]
        sample_answer_attention_mask = sample_answer_text["attention_mask"][0]

        token_per_second = self.token_per_second

        t = 0
        answer_len = len(sample_answer_input_ids)

        action_tokens = np.array([self.dummy_action] * (int(answer_len / token_per_second)+2) * self.action_frames)
        if isinstance(action_tokens, list):
            tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
            # Concatenate tensors along the first dimension
            action_tokens = torch.cat(tensor_list, dim=0)
        action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
        action_ids = self.action_tokenizer(action_tokens)
        self.last_vocab_idx = self.vision_tokenizer.pad_token_id - 1
        action_ids_2 = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
        i = 0

        while t < answer_len:
            sample_speech = self.speech_tokenizer(self.speech_tokenizer.bos_token + self.speech_tokenizer.eos_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
            speech_input_ids = sample_speech["input_ids"][0]
            speech_attention_mask = sample_speech["attention_mask"][0]
            sample_speech_2 = self.vision_tokenizer(self.vision_tokenizer.boi_token + self.vision_tokenizer.eoi_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
            speech_input_ids_2 = sample_speech_2["input_ids"][0]
            speech_attention_mask_2 = sample_speech_2["attention_mask"][0]
            sample_speech_3 = self.speech_tokenizer(self.speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
            speech_input_ids_3 = sample_speech_3["input_ids"][0]
            speech_attention_mask_3 = sample_speech_3["attention_mask"][0]
            answer_input_ids = sample_answer_input_ids[t:t+token_per_second]
            answer_attention_mask = sample_answer_attention_mask[t:t+token_per_second]
            if t + token_per_second > answer_len:
                sample_eot = self.speech_tokenizer(self.speech_tokenizer.silence_token + self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                eot_input_ids = sample_eot["input_ids"][0]
                eot_attention_mask = sample_eot["attention_mask"][0]
            else:
                sample_eot = self.speech_tokenizer(self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                eot_input_ids = sample_eot["input_ids"][0]
                eot_attention_mask = sample_eot["attention_mask"][0]

            sample_input_ids = torch.cat([sample_input_ids, speech_input_ids, speech_input_ids_2, speech_input_ids_3, answer_input_ids, eot_input_ids], dim=-1)  
            sample_attention_mask = torch.cat([sample_attention_mask, speech_attention_mask, speech_attention_mask_2, speech_attention_mask_3, answer_attention_mask, eot_attention_mask], dim=-1)

            action_start = len(sample_input_ids) - len(answer_input_ids) - len(eot_input_ids)
            action_end = len(sample_input_ids)
            labels[action_start:action_end] = torch.cat([answer_input_ids, eot_input_ids], dim=-1)

            action_sample = self.wrap_action_sequence(action_ids_2[i].tolist()) 
            sample_input_ids = torch.cat([sample_input_ids, action_sample], dim=-1)  
            sample_attention_mask = torch.cat([sample_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1)

            action_start = len(sample_input_ids) - len(action_sample)
            action_end = len(sample_input_ids)
            labels[action_start:action_end] = action_sample  

            t += token_per_second
            i += 1
                
        # add silence as end of decoding
        sample_speech = self.speech_tokenizer(self.speech_tokenizer.bos_token + self.speech_tokenizer.eos_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
        speech_input_ids = sample_speech["input_ids"][0]
        speech_attention_mask = sample_speech["attention_mask"][0]
        sample_speech_2 = self.vision_tokenizer(self.vision_tokenizer.boi_token + self.vision_tokenizer.eoi_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
        speech_input_ids_2 = sample_speech_2["input_ids"][0]
        speech_attention_mask_2 = sample_speech_2["attention_mask"][0]
        sample_speech_3 = self.speech_tokenizer(self.speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
        speech_input_ids_3 = sample_speech_3["input_ids"][0]
        speech_attention_mask_3 = sample_speech_3["attention_mask"][0]
        sample_answer = self.speech_tokenizer(self.speech_tokenizer.silence_token + self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
        answer_input_ids = sample_answer["input_ids"][0]
        answer_attention_mask = sample_answer["attention_mask"][0]

        sample_input_ids = torch.cat([sample_input_ids, speech_input_ids, speech_input_ids_2, speech_input_ids_3, answer_input_ids], dim=-1)  
        sample_attention_mask = torch.cat([sample_attention_mask, speech_attention_mask, speech_attention_mask_2, speech_attention_mask_3, answer_attention_mask], dim=-1)

        action_start = len(sample_input_ids) - len(answer_input_ids)
        action_end = len(sample_input_ids)
        labels[action_start:action_end] = answer_input_ids

        action_sample = self.wrap_action_sequence(action_ids_2[i].tolist()) 
        sample_input_ids = torch.cat([sample_input_ids, action_sample], dim=-1)  
        sample_attention_mask = torch.cat([sample_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1)

        action_start = len(sample_input_ids) - len(action_sample)
        action_end = len(sample_input_ids)
        labels[action_start:action_end] = action_sample 

        # pad audio
        pad_audio = torch.zeros(
            (1, (math.ceil(answer_len/token_per_second) + 2) * fs)
        )
        audio4stream = torch.cat(
            (audio4stream, pad_audio), dim=1
        )
        if audio4stream.shape[1] % fs:
            pad_audio = torch.zeros(
                (1, fs - audio4stream.shape[1] % fs)
            )
            audio4stream = torch.cat(
                (audio4stream, pad_audio), dim=1
            )
        audio4stream = [audio4stream.squeeze()]
        fbank_feature = self.fbank(audio4stream)[0]
        fbank_feature_len = fbank_feature.size(0)
                
        sample = {}
        succ = True
        sample["input_ids"] = sample_input_ids,
        sample["attention_mask"] = sample_attention_mask,
        sample["labels"] = labels
        sample["fbank_feature"] = fbank_feature,
        sample["fbank_feature_len"] = fbank_feature_len

        return sample

    def get_speech_only_item(self, index: int):
        while True:
            try:
                sent_lens = []
                codecs = []
                codec_lens = []

                sample = self.data_speech[index]

                audio4stream, fs = torchaudio.load(sample["path"][0])
                assert fs == 16000
                frames = int(self.time_block * fs)
                speech_len = math.ceil(audio4stream.shape[1] / frames)

                prompt = self.prompts[sample["task"]]

                labels = torch.full((self.speech_tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)

                text_prompt = self.speech_tokenizer.bop_token + prompt + self.speech_tokenizer.eop_token
                silence_output = self.speech_tokenizer(self.speech_tokenizer.silence_token + self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                for i in range(speech_len):
                    text_prompt += self.speech_tokenizer.bos_token + self.speech_tokenizer.eos_token + self.speech_tokenizer.bot_token + self.speech_tokenizer.silence_token + self.speech_tokenizer.eot_token
                    
                sample_text = self.speech_tokenizer(text_prompt, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                sample_input_ids = sample_text["input_ids"][0]
                sample_attention_mask = sample_text["attention_mask"][0]

                bot_id = 128259
                bot_idx = (sample_input_ids == bot_id).nonzero(as_tuple=True)[0]
                for idx in bot_idx:
                    labels[idx+1:idx+3] = silence_output

                if self.generate and sample["task"] != "dia_asr":
                    sample_answer_input_ids = []
                    if "sentences" in sample:
                        for sent in sample["sentences"]:
                            sent_input_ids = self.speech_tokenizer(sent, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                            sample_answer_input_ids.append(sent_input_ids)
                            sent_lens.append(len(sent_input_ids))
                        sample_answer_input_ids = torch.cat(sample_answer_input_ids, dim=-1)
                    else:
                        sample_answer_input_ids = self.speech_tokenizer(sample["text"], padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                        sent_lens.append(len(sample_answer_input_ids))
                    sample_answer_attention_mask = torch.ones_like(sample_answer_input_ids)

                    for path in sample["path_a"]:
                        audio4stream_a, _ = torchaudio.load(path)
                        audio4stream_a = audio4stream_a[:1].contiguous()
                        dp = self.cosy_frontend._extract_speech_token(audio4stream_a, cpu=True)
                        codecs.append(dp["speech_token"])
                        codec_lens.append(dp["speech_token_len"])

                    assert len(sent_lens) == len(codecs)                
                else:
                    answer_text = sample["text"]
                    sample_answer_text = self.speech_tokenizer(answer_text, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                    sample_answer_input_ids = sample_answer_text["input_ids"][0]
                    sample_answer_attention_mask = sample_answer_text["attention_mask"][0]

                token_per_second = self.token_per_second

                t = 0
                answer_len = len(sample_answer_input_ids)
                while t < answer_len:
                    sample_speech = self.speech_tokenizer(self.speech_tokenizer.bos_token + self.speech_tokenizer.eos_token + self.speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                    speech_input_ids = sample_speech["input_ids"][0]
                    speech_attention_mask = sample_speech["attention_mask"][0]
                    answer_input_ids = sample_answer_input_ids[t:t+token_per_second]
                    answer_attention_mask = sample_answer_attention_mask[t:t+token_per_second]
                    if t + token_per_second > answer_len:
                        sample_eot = self.speech_tokenizer(self.speech_tokenizer.silence_token + self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                        eot_input_ids = sample_eot["input_ids"][0]
                        eot_attention_mask = sample_eot["attention_mask"][0]
                    else:
                        sample_eot = self.speech_tokenizer(self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                        eot_input_ids = sample_eot["input_ids"][0]
                        eot_attention_mask = sample_eot["attention_mask"][0]

                    sample_input_ids = torch.cat([sample_input_ids, speech_input_ids, answer_input_ids, eot_input_ids], dim=-1)  
                    sample_attention_mask = torch.cat([sample_attention_mask, speech_attention_mask, answer_attention_mask, eot_attention_mask], dim=-1)

                    action_start = len(sample_input_ids) - len(answer_input_ids) - len(eot_input_ids)
                    action_end = len(sample_input_ids)
                    labels[action_start:action_end] = torch.cat([answer_input_ids, eot_input_ids], dim=-1) 

                    t += token_per_second
                
                sample_speech = self.speech_tokenizer(self.speech_tokenizer.bos_token + self.speech_tokenizer.eos_token + self.speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                speech_input_ids = sample_speech["input_ids"][0]
                speech_attention_mask = sample_speech["attention_mask"][0]
                sample_eot = self.speech_tokenizer(self.speech_tokenizer.silence_token + self.speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                eot_input_ids = sample_eot["input_ids"][0]
                eot_attention_mask = sample_eot["attention_mask"][0]
                
                sample_input_ids = torch.cat([sample_input_ids, speech_input_ids, eot_input_ids], dim=-1)  
                sample_attention_mask = torch.cat([sample_attention_mask, speech_attention_mask, eot_attention_mask], dim=-1)

                action_start = len(sample_input_ids) - len(eot_input_ids)
                action_end = len(sample_input_ids)
                labels[action_start:action_end] = eot_input_ids 
                    
                pad_audio = torch.zeros(
                    (1, (math.ceil(answer_len/token_per_second) + 2) * frames)
                )
                audio4stream = torch.cat(
                    (audio4stream, pad_audio), dim=1
                )
                if audio4stream.shape[1] % frames:
                    pad_audio = torch.zeros(
                        (1, frames - audio4stream.shape[1] % frames)
                    )
                    audio4stream = torch.cat(
                        (audio4stream, pad_audio), dim=1
                    )
                audio4stream = [audio4stream.squeeze()]
                if self.encoder_type == "mamba":
                    fbank_feature = self.fbank(audio4stream)[0]
                elif self.encoder_type == "zipformer2":
                    fbank_feature = self.fbank.extract(audio4stream[0], sampling_rate=fs)
                fbank_feature_len = fbank_feature.size(0)
                
                task = sample["task"]
                wav_path = sample["path"][0]
                sample = {}
                if fbank_feature_len > 150 * 100:
                    succ = False
                    index += 1234
                    index = index % len(self.data)
                else:
                    succ = True
                sample["input_ids"] = sample_input_ids,
                sample["attention_mask"] = sample_attention_mask,
                sample["labels"] = labels
                sample["fbank_feature"] = fbank_feature,
                sample["fbank_feature_len"] = fbank_feature_len
                sample["sent_lens"] = sent_lens
                sample["codecs"] = codecs
                sample["codec_lens"] = codec_lens
                sample["type"] = "speech_only"
            
            except Exception as e:
                succ = False
                print(e)
                import traceback
                traceback.print_exc()
                index += 1
                index = index % len(self.data)
                continue

            if succ:
                break

        return sample

    def get_action_only_item(self, index: int, start_idx=-1):

        scene = self.data[index]

        if self.cfg:
            p_prob = random.random()
            if p_prob < self.args.null_prompt_prob:
                prompt = ""
            else:
                prompt = scene["text"]
        else:
            prompt = scene["text"]

        image_tokens_path = scene["image"]

        # handle different dataset fps for post training
        fps = self.get_fps_for_path(image_tokens_path)
        if fps is not None:
            self.action_frames = fps
        
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        
        # use action information
        if self.actions:
            action = scene["action"] 
            if self.use_gripper:
                gripper = scene["gripper_image"]
                image_tokens, action_tokens, gripper_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, gripper=gripper, return_start=True)
            else:
                image_tokens, action_tokens, start_idx = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, return_start=True)
            
            if self.video_format == "interleave":
                if self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                    action_ids = self.action_tokenizer(action_tokens)
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
            else:
                if self.actions_format == "openvla":
                    action_tokens = action_tokens.flatten()
                    action_ids = self.action_tokenizer(action_tokens)

                    # Debugging
                    # action_debug = self.action_tokenizer.decode_token_ids_to_actions(action_ids)
                    # error = action_tokens - action_debug
                elif self.actions_format == "text":
                    action_str = "\n".join(",".join(f"{num:.2f}" for num in row) for row in action_tokens)
                    action_prompt = self.act_template.format(action_prompt=action_str)
                elif self.actions_format == "continuous":
                    action_continuous = action_tokens
                elif self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_ids = self.action_tokenizer(action_tokens)[0]
                    # action_decode = self.action_tokenizer.decode([action_ids])
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - id for id in action_ids]
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
        # video VLA
        if self.video_format == "interleave":
            text_prompt = self.tokenizer.bos_token + prompt
            image_tokens = image_tokens[0::self.action_frames,...]
            if self.use_gripper:
                gripper_tokens = gripper_tokens[0::self.action_frames,...]
            
            sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                if self.use_gripper:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                    image_prompt += gripper_prompt
                sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                if self.actions:
                    if self.actions_format == "fast":
                        action_sample = self.wrap_action_sequence(action_ids[i].tolist()) 
                        sample_input_ids = torch.cat([sample_input_ids, image_input_ids, action_sample], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1) 
                        action_start = len(sample_input_ids) - len(action_sample)
                        action_end = len(sample_input_ids)
                        if self.args.apply_loss_on_only_action:  
                            labels[action_start:action_end] = action_sample
                        else:  # Otherwise, fill both vision and action parts in the labels
                            labels[action_start-len(image_input_ids):action_start] = image_input_ids  
                            labels[action_start:action_end] = action_sample 
                else:
                    sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                    sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                    labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
            
            sample = self.tokenizer.pad(
                    {
                        "input_ids": sample_input_ids,
                        "attention_mask": sample_attention_mask,
                        "labels": labels
                    },
                    padding="max_length",
                    return_tensors="pt"
                )
            for k, v in sample.items():
                sample[k] = v
            sample["type"] = "action_only"
            # sample["path"] = image_tokens_path[0][:image_tokens_path[0].rfind("/")]
            # sample["frames_num"] = len(image_tokens_path) - frames_num

        return sample    

class Emu3WorldModelDataset(Emu3SFTDataset):    

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__(args, tokenizer=tokenizer)
        # weights
        dataset_weights = {
            'rt1': 0.3,
            'droid_fast': 0.2,
            'oxembodiment/bridge': 1.0,
            'oxembodiment/toto': 1.0,
            'oxembodiment/taco_play': 1.0,
            'oxembodiment/fmb': 1.0,
            'oxembodiment/maniskill': 0.5,
            'oxembodiment/kuka': 0.1,
            'oxembodiment/berkeley_autolab_ur5': 1.0,
            'calvin': 0.8,
            'libero': 1.0,
        }
        self.datasets_weight = args.datasets_weight
        if self.datasets_weight:
            self.sample_weights = [dataset_weights.get(d["dataset"], 1.0) for d in self.data]
        self.without_text = args.without_text

    def __getitem__(self, index: int):

        scene = self.data[index]

        if self.without_text:
            prompt = ""
        else:
            prompt = scene["text"]

        image_tokens_path = scene["image"]

        # handle different dataset fps for post training
        fps = self.get_fps_for_path(image_tokens_path)
        if fps is not None:
            self.action_frames = fps
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        
        if self.use_gripper and "gripper_image" in scene:
            gripper = scene["gripper_image"]
            image_tokens, gripper_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, gripper=gripper)
        else:
            image_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num) 
        
        # video VLA
        if self.video_format == "interleave":
            text_prompt = self.tokenizer.bos_token + prompt
            image_tokens = image_tokens[0::self.action_frames,...]
            if self.use_gripper and "gripper_image" in scene:
                gripper_tokens = gripper_tokens[0::self.action_frames,...]
            
            sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                if self.use_gripper and "gripper_image" in scene:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                    image_prompt += gripper_prompt
                sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                
                sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
            
            if self.args.apply_loss_on_only_vision:
                labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)
            
            sample = self.tokenizer.pad(
                    {
                        "input_ids": sample_input_ids,
                        "attention_mask": sample_attention_mask,
                        "labels": labels
                    },
                    padding="max_length",
                    return_tensors="pt"
                )
            for k, v in sample.items():
                sample[k] = v.squeeze(0)
        
        else:
            raise NotImplementedError("Only interleave video format is supported for world model dataset.")
        return sample
    
class Emu3RealRobotDataset(Emu3SFTDataset):    

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__(args, tokenizer=tokenizer)
        self.use_views = ['cam_high','cam_left_wrist','cam_right_wrist']
    
    def random_frames_to_tensor(self, img_list, T, action_prompt=None, wrist=None):
        
        start_idx = random.randint(0, len(img_list) - T)

        selected_frames = [np.load(img_path) for img_path in img_list[start_idx:start_idx + T]]
        tensor_frames = [torch.from_numpy(frame) for frame in selected_frames]
        tensor = torch.stack(tensor_frames, dim=1)

        wrist_left = wrist["cam_left_wrist"]
        wrist_right = wrist["cam_right_wrist"]

        select_wrist_left = [torch.from_numpy(np.load(img_path)) for img_path in wrist_left[start_idx:start_idx + T]]
        select_wrist_right = [torch.from_numpy(np.load(img_path)) for img_path in wrist_right[start_idx:start_idx + T]]

        tensor_wrist_left = torch.stack(select_wrist_left, dim=1)
        tensor_wrist_right = torch.stack(select_wrist_right, dim=1)

        if action_prompt is None:
            return tensor.squeeze(0), tensor_wrist_left.squeeze(0), tensor_wrist_right.squeeze(0)

        selected_actions = action_prompt[start_idx:start_idx + T]
        return tensor.squeeze(0), tensor_wrist_left.squeeze(0), tensor_wrist_right.squeeze(0), selected_actions
    
    def __getitem__(self, index: int):

        scene = self.data[index]

        prompt = scene["text"]

        image_tokens_path = scene["cam_high"]
        
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        
        # use action information
        if self.actions:
            action = scene["action"] 
            image_tokens, wrist_left_token, wrist_right_token, action_tokens= self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, wrist=scene)
            
            if self.video_format == "interleave":
                if self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                    action_ids = self.action_tokenizer(action_tokens)
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                    
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
            else:
                if self.actions_format == "openvla":
                    action_tokens = action_tokens.flatten()
                    action_ids = self.action_tokenizer(action_tokens)
                elif self.actions_format == "text":
                    action_str = "\n".join(",".join(f"{num:.2f}" for num in row) for row in action_tokens)
                    action_prompt = self.act_template.format(action_prompt=action_str)
                elif self.actions_format == "continuous":
                    action_continuous = action_tokens
                elif self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_ids = self.action_tokenizer(action_tokens)[0]
                    # action_decode = self.action_tokenizer.decode([action_ids])
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - id for id in action_ids]
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
        else:
            image_tokens, wrist_left_token, wrist_right_token = self.random_frames_to_tensor(image_tokens_path, frames_num, wrist=scene)
        
        # video VLA
        if self.video_format == "interleave":
            text_prompt = self.tokenizer.bos_token + prompt
            image_tokens = image_tokens[0::self.action_frames,...]
            wrist_left_token = wrist_left_token[0::self.action_frames,...]
            wrist_right_token = wrist_right_token[0::self.action_frames,...]
            
            sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                wrist_left_prompt = self.format_video_prompt(wrist_left_token[i:i+1])
                wrist_right_prompt = self.format_video_prompt(wrist_right_token[i:i+1])
                image_prompt += wrist_left_prompt + wrist_right_prompt
                sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                if self.actions:
                    if self.actions_format == "fast":
                        action_sample = self.wrap_action_sequence(action_ids[i].tolist()) 
                        sample_input_ids = torch.cat([sample_input_ids, image_input_ids, action_sample], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1) 
                        action_start = len(sample_input_ids) - len(action_sample)
                        action_end = len(sample_input_ids)
                        if self.args.apply_loss_on_only_action:  
                            labels[action_start:action_end] = action_sample
                        else:  # Otherwise, fill both vision and action parts in the labels
                            labels[action_start-len(image_input_ids):action_start] = image_input_ids  
                            labels[action_start:action_end] = action_sample 
                else:
                    sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                    sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                    labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
            sample = self.tokenizer.pad(
                    {
                        "input_ids": sample_input_ids,
                        "attention_mask": sample_attention_mask,
                        "labels": labels
                    },
                    padding="max_length",
                    return_tensors="pt"
                )
            for k, v in sample.items():
                sample[k] = v.squeeze(0)
        # VLA Baseline (Img)
        else:
            image_tokens = image_tokens[0:self.T,...]
            image_prompt = self.format_video_prompt(image_tokens)

            wrist_left_tokens = wrist_left_token[0:self.T,...]
            wrist_right_tokens = wrist_right_token[0:self.T,...]
            wrist_left_prompt = self.format_video_prompt(wrist_left_tokens)
            wrist_right_prompt = self.format_video_prompt(wrist_right_tokens)
            image_prompt = image_prompt + wrist_left_prompt + wrist_right_prompt
            
            input = self.tokenizer.bos_token + prompt + image_prompt 

            sample = self.tokenizer(
                input,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            labels = sample["input_ids"]

            # only use vision loss
            if self.args.apply_loss_on_only_vision:
                labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)

            sample["labels"] = labels
            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            # based on the actions_format, append the action information to the sample
            if self.actions:
                if self.actions_format == "openvla":
                    action_sample = self.wrap_action_sequence(action_ids)
                    sample["input_ids"] = torch.cat([sample["input_ids"], action_sample], dim=-1)

                    # Update attention_mask
                    action_mask = torch.ones_like(action_sample, dtype=torch.long)
                    sample["attention_mask"] = torch.cat([sample["attention_mask"], action_mask], dim=-1)

                    action_labels = action_sample.clone()  # Clone action_sample for labels
                    sample["labels"] = torch.cat([sample["labels"], action_labels], dim=-1)
                
                # FAST
                elif self.actions_format == "fast":
                    if 'state' in scene.keys():
                        state = scene['state'].reshape(1, 1, -1)
                        state_tokens = self.action_tokenizer(state)[0]
                        state_ids = [self.last_vocab_idx - id for id in state_tokens]
                        state_tensor = torch.tensor(state_ids, dtype=sample["input_ids"].dtype, device=sample["input_ids"].device)

                        sample["input_ids"] = torch.cat([sample["input_ids"], state_tensor], dim=-1)

                        state_label_tensor = torch.full_like(state_tensor, fill_value=-100)  # -100 means ignored in loss
                        sample["labels"] = torch.cat([sample["labels"], state_label_tensor], dim=-1)

                        state_mask = torch.ones_like(state_tensor)
                        sample["attention_mask"] = torch.cat([sample["attention_mask"], state_mask], dim=-1)
                    
                    if self.args.apply_loss_on_only_action:
                        sample['labels'] = torch.full_like(sample['labels'], self.args.ignore_index)
                    sample = self.append_action_to_sample(sample, action_ids)
                # Flow Matching
                elif self.actions_format == "continuous":
                    boa_token_id = self.tokenizer.encode(self.tokenizer.boa_token)[0]
                    sample = self.append_boa_to_sample(sample, [boa_token_id])
                    sample["action"] = action_continuous
            
            # finally, do padding
            sample = self.tokenizer.pad(
                sample,
                padding="max_length",
                return_tensors="pt"
            )

            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            if "labels" in sample:
                sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)
        return sample

class Emu3CoTDataset(Emu3SFTDataset):    

    def __init__(self, args: "DataArguments", tokenizer):
        super().__init__(args, tokenizer=tokenizer)
    
    def random_frames_to_tensor(self, img_list, T, action_prompt=None, reason_prompt=None):
        start_idx = random.randint(0, len(img_list) - T)

        selected_frames = [np.load(img_path) for img_path in img_list[start_idx:start_idx + T]]
        tensor_frames = [torch.from_numpy(frame) for frame in selected_frames]
        tensor = torch.stack(tensor_frames, dim=1)

        selected_actions = action_prompt[start_idx:start_idx + T]
        selected_reason = reason_prompt[start_idx:start_idx + T]
        return tensor.squeeze(0), selected_actions, selected_reason
    
    def __getitem__(self, index: int):

        scene = self.data[index]
        prompt = scene["text"]
        image_tokens_path = scene["image"]
        
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        

        action = scene["action"] 
        image_tokens, action_tokens, reason_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, reason_prompt=scene["reasoning"])
        
        if self.video_format == "interleave":
            if self.actions_format == "fast":
                if isinstance(action_tokens, list):
                    tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                    # Concatenate tensors along the first dimension
                    action_tokens = torch.cat(tensor_list, dim=0)
                action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                action_ids = self.action_tokenizer(action_tokens)
                self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                
            else:
                raise ValueError(f"Invalid actions_format: {self.actions_format}")
        else:
            if self.actions_format == "openvla":
                action_tokens = action_tokens.flatten()
                action_ids = self.action_tokenizer(action_tokens)

                # Debugging
                # action_debug = self.action_tokenizer.decode_token_ids_to_actions(action_ids)
                # error = action_tokens - action_debug
            elif self.actions_format == "text":
                action_str = "\n".join(",".join(f"{num:.2f}" for num in row) for row in action_tokens)
                action_prompt = self.act_template.format(action_prompt=action_str)
            elif self.actions_format == "continuous":
                action_continuous = action_tokens
            elif self.actions_format == "fast":
                if isinstance(action_tokens, list):
                    tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                    # Concatenate tensors along the first dimension
                    action_tokens = torch.cat(tensor_list, dim=0)
                action_ids = self.action_tokenizer(action_tokens)[0]
                # action_decode = self.action_tokenizer.decode([action_ids])
                self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                action_ids = [self.last_vocab_idx - id for id in action_ids]
            else:
                raise ValueError(f"Invalid actions_format: {self.actions_format}")
        
        # video VLA
        if self.video_format == "interleave":
            text_prompt = self.tokenizer.bos_token + prompt
            image_tokens = image_tokens[0::self.action_frames,...]
            if self.use_gripper:
                gripper_tokens = gripper_tokens[0::self.action_frames,...]
            
            sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                if self.use_gripper:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                    image_prompt += gripper_prompt
                sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                if self.actions:
                    if self.actions_format == "fast":
                        action_sample = self.wrap_action_sequence(action_ids[i].tolist()) 
                        sample_input_ids = torch.cat([sample_input_ids, image_input_ids, action_sample], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1) 
                        action_start = len(sample_input_ids) - len(action_sample)
                        action_end = len(sample_input_ids)
                        if self.args.apply_loss_on_only_action:  
                            labels[action_start:action_end] = action_sample
                        else:  # Otherwise, fill both vision and action parts in the labels
                            labels[action_start-len(image_input_ids):action_start] = image_input_ids  
                            labels[action_start:action_end] = action_sample 
                else:
                    sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                    sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                    labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
            
            sample = self.tokenizer.pad(
                    {
                        "input_ids": sample_input_ids,
                        "attention_mask": sample_attention_mask,
                        "labels": labels
                    },
                    padding="max_length",
                    return_tensors="pt"
                )
            for k, v in sample.items():
                sample[k] = v.squeeze(0)
                
        # VLA Baseline (Img)
        else:
            image_tokens = image_tokens[0:self.T,...]
            image_prompt = self.format_video_prompt(image_tokens)

            reason_tokens = reason_tokens[0:self.T]

            input = self.tokenizer.bos_token + prompt + image_prompt + self.tokenizer.bot_token + reason_tokens[0]['reasoning'] + self.tokenizer.eot_token

            sample = self.tokenizer(
                input,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            labels = sample["input_ids"]

            # not use vision loss
            labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), self.args.ignore_index, labels)

            sample["labels"] = labels
            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            # based on the actions_format, append the action information to the sample
            if self.actions:
                if self.args.apply_loss_on_only_action:
                    sample['labels'] = torch.full_like(sample['labels'], self.args.ignore_index)
                sample = self.append_action_to_sample(sample, action_ids)
            
            # finally, do padding
            sample = self.tokenizer.pad(
                sample,
                padding="max_length",
                return_tensors="pt"
            )

            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            if "labels" in sample:
                sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)
        return sample
    