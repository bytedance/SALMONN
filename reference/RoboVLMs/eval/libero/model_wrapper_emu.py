#!/bin/bash

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
import torch
import numpy as np
from queue import Queue
from PIL import Image

from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor, AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import sys
import os
ELLSA_BASE_PATH = os.environ.get("ELLSA_BASE_PATH")
COSY_CKPT_PATH = os.environ.get("COSY_CKPT_PATH")
LLAMA_CKPT_PATH = os.environ.get("LLAMA_CKPT_PATH")
UNIVLA_CKPT_PATH = os.environ.get("UNIVLA_CKPT_PATH")

sys.path.append(os.path.join(ELLSA_BASE_PATH,"reference/Emu3"))
from emu3.mllm import Emu3Tokenizer, Emu3ForCausalLM, Emu3Processor, Emu3MoEConfig
from emu3.mllm import Emu3MoE, LlamaWithSpeech, Emu3ForMix
from transformers import LogitsProcessor

class ActionIDConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        """
        :param allowed_token_ids: 允许的token ID列表
        """
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        # 创建掩码：允许的token位置为True，其他为False
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if mask.ndim == 1:
            mask[self.allowed_token_ids] = True
        else:
            mask[:, self.allowed_token_ids] = True
        
        # 将不允许的token概率设为负无穷
        scores[~mask] = -float("inf")
        return scores

class EmuVLAModel:
    # model option
    def __init__(
        self,
        emu_hub,
        vq_hub,
        vision_hub,
        device,
        speech,
        moe=False,
        mix=False,
        attn_adapter=False,
        attn_adapter_type="None",
        merge_speech_lora=False,
        lora_modules="default",
        generate=False,
        encoder_type="mamba",
        four_expert=False,
        four_expert_text=False,
        time_block=1.0,
        predict_action_frames=10
    ):

        self.emu_hub = emu_hub
        self.vq_hub = vq_hub
        self.vision_hub = vision_hub
        self.device = device
        self.speech = speech
        self.moe = moe
        self.mix = mix
        self.attn_adapter = attn_adapter
        self.attn_adapter_type = attn_adapter_type
        self.merge_speech_lora = merge_speech_lora
        self.lora_modules = lora_modules
        self.generation = generate
        self.encoder_type = encoder_type
        self.four_expert = four_expert
        self.four_expert_text = four_expert_text
        self.time_block = time_block

        ## hard code here
        self.window_size = 2
        self.predict_action_frames = predict_action_frames
        self.context_frames = 1
        self.predict_frames = 1
        self.action_dim = 7
        self.use_gripper = True
        self.use_fast = True
        self.use_one_step = False
        self.eoa_token_id = 151845
        self.eot_token_id = 128260
        self.use_cot = False  # always disable CoT

        self.video_mode = True
    
        # load model and tokenizer
        self.init_config(device=device)
        self.image_processor.min_pixels = 80 * 80
        self.prompts = {"dia_qa":"Please answer the question.","dia_asr":"Generate a transcript of the speech.","vqa":"Please answer the question based on the image."}
        # self.dataset_stat = self.load_dataset_stat()

        self.kwargs = dict(
            mode='VLA',
            padding="longest",
        )
        if self.use_fast:
            self.GENERATION_CONFIG = GenerationConfig(
                    pad_token_id=self.model.config.pad_token_id,
                    bos_token_id=self.model.config.bos_token_id,
                    eos_token_id=self.eoa_token_id,
                    do_sample=False,
                )
        else:
            self.GENERATION_CONFIG = GenerationConfig(
                use_cache=True,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
                max_new_tokens=800,
                do_sample=True,
                top_k=2048,
                temperature=0.8,
            )
        
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

    def init_config(self, device):
        
        if self.moe:
            speech_tokenizer = AutoTokenizer.from_pretrained(self.emu_hub)
            speech_tokenizer.bosp_token = "<bosp>" # modified
            speech_tokenizer.eosp_token = "<eosp>" # modified
            speech_tokenizer.bot_token = "<bot>"
            speech_tokenizer.eot_token = "<eot>"
            speech_tokenizer.silence_token = "<silence>"
            speech_tokenizer.bop_token = "<bop>"
            speech_tokenizer.eop_token = "<eop>"
            speech_tokenizer.padding_side = "right"
            vision_tokenizer = Emu3Tokenizer.from_pretrained(
                UNIVLA_CKPT_PATH,
                model_max_length=6400,
                padding_side="right",
                use_fast=False,
            )
            self.tokenizer = [speech_tokenizer, vision_tokenizer]
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.emu_hub)
            tokenizer.bosp_token = "<bosp>"
            tokenizer.eosp_token = "<eosp>"
            tokenizer.bot_token = "<bot>"
            tokenizer.eot_token = "<eot>"
            tokenizer.silence_token = "<silence>"
            tokenizer.bop_token = "<bop>"
            tokenizer.eop_token = "<eop>"
            tokenizer.padding_side = "right"
            if self.mix:
                vision_tokenizer = Emu3Tokenizer.from_pretrained(
                    UNIVLA_CKPT_PATH,
                    model_max_length=6400,
                    padding_side="right",
                    use_fast=False,
                )
                self.tokenizer = [tokenizer, vision_tokenizer]
            else:
                self.tokenizer = tokenizer

        if self.moe:
            config_vision = Emu3MoEConfig.from_pretrained(os.path.join(UNIVLA_CKPT_PATH,"config.json"))
            config_vision._attn_implementation = "flash_attention_2"
            self.model = Emu3ForMix.from_pretrained(
                self.emu_hub,
                config_vision=config_vision,
                tokenizer=self.tokenizer,
                speech_encoder_path="",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                peft=True,
                freeze=True,
                debug=False,
                test=True,
                attn_adapter=self.attn_adapter,
                attn_adapter_type=self.attn_adapter_type,
                merge_speech_lora=self.merge_speech_lora,
                lora_modules=self.lora_modules,
                generate=self.generation,
                encoder_type=self.encoder_type,
                time_block=self.time_block
            )
            self.model._use_sdpa = False
            self.model._use_flash_attention_2 = True
        else:
            self.model = LlamaWithSpeech.from_pretrained(
                self.emu_hub,
                tokenizer=self.tokenizer,
                llama_path=LLAMA_CKPT_PATH,
                speech_encoder_path="",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                peft=True,
                freeze=False,
                debug=False,
                mix=self.mix,
                time_block=self.time_block,
                encoder_type=self.encoder_type
            )

        self.model.to(device).eval()

        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(self.vision_hub, trust_remote_code=True).to(device).eval()
        if self.moe or self.mix:
            self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer[1])
        else:
            pass

        # fast tokenization
        fast_path = os.path.join(ELLSA_BASE_PATH,"pretrain/fast")
        self.action_tokenizer = AutoProcessor.from_pretrained(fast_path, trust_remote_code=True)

        self.rgb_list = []
        self.hand_rgb_list = []
        self.action_hist_list = []
        self.rollout_step_counter = 0

        self.vision_queue = Queue(maxsize=self.window_size)
        self.text_queue = Queue(maxsize=500)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)

    def wrap_action_sequence(self, action_ids):
        """
        Wraps a sequence of action token IDs with special tokens (beginning and end).

        Args:
            action_ids (List[int]): The sequence of action token IDs.

        Returns:
            torch.Tensor: A tensor containing the wrapped sequence.
        """
        # Encode the beginning and end action tokens
        if self.moe or (self.mix):
            vision_tokenizer = self.tokenizer[1]
        else:
            vision_tokenizer = self.tokenizer

        action_begin = vision_tokenizer.encode(vision_tokenizer.boa_token)[0]
        action_end = vision_tokenizer.encode(vision_tokenizer.eoa_token)[0]
        eos = vision_tokenizer.encode(vision_tokenizer.eos_token)[0]

        # Wrap the action sequence
        # wrapped_action = [action_begin] + action_ids + [action_end] + [eos]
        wrapped_action = [action_begin] + action_ids + [action_end]
        
        # Convert to a PyTorch tensor
        return torch.tensor(wrapped_action, dtype=torch.long)

    @staticmethod
    def load_dataset_stat():
        stat = {}
        with open(
            "/share/project/yuqi.wang/OmniSim/reference/RoboVLMs-main/configs/data/libero_dataset_stats/dataset_libero_10.json", "r"
        ) as f:
            libero_10_info = json.load(f)
        stat["libero_10"] = libero_10_info

        return stat
    
    def add_image(self, image):
        if self.vision_queue.full():
            self.vision_queue.get()
        self.vision_queue.put(image)
    
    def get_history(self):
        return list(self.vision_queue.queue) 

    def add_action(self, action):
        if self.action_queue.full():
            self.action_queue.get()
        self.action_queue.put(action)
    
    def get_action_history(self):
        return list(self.action_queue.queue)
    
    def get_text_history(self):
        return list(self.text_queue.queue)
    
    def add_text(self, text):
        if self.text_queue.full():
            self.text_queue.get()
        self.text_queue.put(text)

    def reset(self):

        self.rgb_list = []
        self.hand_rgb_list = []
        self.rollout_step_counter = 0
        self.action_hist_list = []
        self.tts_features = []

        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()
        while not self.text_queue.empty():
            self.text_queue.get()

    def preprocess(self, image):
        # preprocess image
        agent_view = image['full_image']
        agent_view = Image.fromarray(agent_view)
        agent_view = agent_view.resize((200, 200))
        image_x = self.image_processor(agent_view, return_tensors="pt")["pixel_values"].cuda()
        image_code = self.image_tokenizer.encode(image_x)

        gripper_code = None
        if "wrist_image" in image:
            gripper_view = image['wrist_image']
            gripper_view = Image.fromarray(gripper_view)
            gripper_view = gripper_view.resize((200, 200))
            gripper_x = self.image_processor(gripper_view, return_tensors="pt")["pixel_values"].cuda()  
            gripper_code = self.image_tokenizer.encode(gripper_x)

        return (
            image_code,
            gripper_code,
        )

    def step(self, image, goal):
        input_dict = dict()
        
        image_code, gripper_code = self.preprocess(image)

        prompt,neg_prompt = goal,""

        video_code = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None

        text_prompt = [self.tokenizer.bos_token + prompt]
        text_tokens = self.processor.tokenizer(text_prompt)
        
        text_tokens = BatchFeature(data={**text_tokens}, tensor_type='pt')

        if self.video_mode:
            kwargs = dict(
                    mode='VLA_Video',
                    padding="longest",
                )
            pos_inputs = self.processor.video_process(text=prompt, video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt", **kwargs)
        else:
            pos_inputs = self.processor.video_process(text=prompt, video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt", **self.kwargs)
        
        if self.video_mode:
            self.add_image(pos_inputs)
            
            # 获取历史图像和动作
            history = self.get_history()
            action_history = self.get_action_history()

            # 初始化输入ID、token类型ID和attention mask
            all_input_ids = []
            all_token_type_ids = []
            all_attention_mask = []

            # Add text
            all_input_ids.append(text_tokens['input_ids'])
            all_token_type_ids.append(text_tokens['token_type_ids'])
            all_attention_mask.append(text_tokens['attention_mask'])

            # 遍历历史图像
            for i in range(len(history)):
                img_input_ids = history[i]['input_ids']
                img_token_type_ids = history[i]['token_type_ids']
                img_attention_mask = history[i]['attention_mask']
                
                # 对应的动作
                if i < len(action_history):
                    act_input_ids = action_history[i]
                    
                    # 动作的token_type_ids和attention_mask分别填充为全0和全1
                    act_token_type_ids = torch.zeros_like(act_input_ids)
                    act_attention_mask = torch.ones_like(act_input_ids)
                    
                    # 交替添加图像和动作数据
                    all_input_ids.extend([img_input_ids, act_input_ids])
                    all_token_type_ids.extend([img_token_type_ids, act_token_type_ids])
                    all_attention_mask.extend([img_attention_mask, act_attention_mask])
                else:
                    # 若没有对应的动作，添加图像数据
                    all_input_ids.append(img_input_ids)
                    all_token_type_ids.append(img_token_type_ids)
                    all_attention_mask.append(img_attention_mask)
            # 拼接所有的input_ids、token_type_ids和attention_mask
            concatenated_input_ids = torch.cat(all_input_ids, dim=1)
            concatenated_token_type_ids = torch.cat(all_token_type_ids, dim=1)
            concatenated_attention_mask = torch.cat(all_attention_mask, dim=1)
            
            # 更新pos_inputs
            final_inputs = pos_inputs.copy()
            final_inputs['input_ids'] = concatenated_input_ids
            final_inputs['token_type_ids'] = concatenated_token_type_ids
            final_inputs['attention_mask'] = concatenated_attention_mask
        else:
            final_inputs = pos_inputs

        if self.use_fast: 
            last_token_id = self.tokenizer.pad_token_id - 1
            allowed_token_ids = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    final_inputs.input_ids.to(self.device),
                    self.GENERATION_CONFIG,
                    max_new_tokens=80,
                    logits_processor=[action_id_processor],
                    attention_mask=final_inputs.attention_mask.to(self.device),
                )
            # omit the eoa token
            orig_outputs = outputs[:, final_inputs.input_ids.shape[-1]:]
            outputs = outputs[:, final_inputs.input_ids.shape[-1]:-1]
            last_token_id_tensor = torch.tensor(last_token_id, dtype=outputs.dtype, device=outputs.device)
            processed_outputs = last_token_id_tensor - outputs
            action_outputs = self.action_tokenizer.decode(
                processed_outputs, time_horizon=self.predict_action_frames, action_dim=self.action_dim
            )
            action = action_outputs[0]
            if self.video_mode:
                self.add_action(orig_outputs.detach().cpu())

        else:
            pass
        
        # unnormalize action
        action = self.unormalize_action(action)

        # NOTE(zbzhu): Flip the gripper action here
        # refer to https://github.com/openvla/openvla/blob/1b024f242eda833dc8e321953f25cfd5f3d2f76d/experiments/robot/libero/run_libero_eval.py#L225
        action[..., -1] = np.where(action[..., -1] > 0, 1, -1)

        
        if self.use_one_step:
            # only one step
            action_pred = action[0:1]
        else:
            # action chunk
            action_pred = action
        
        if self.use_cot:
            return action_pred, thought
        else:
            return action_pred
    
    def step_withspeech(self, image, fbank_feature, fbank_feature_len, t, generate_text=False):
        input_dict = dict()
        
        image_code, gripper_code = self.preprocess(image)
        if self.moe or (self.mix):
            speech_tokenizer = self.tokenizer[0]
            vision_tokenizer = self.tokenizer[1]
        else:
            speech_tokenizer = self.tokenizer
            vision_tokenizer = self.tokenizer

        video_code = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None

        if self.moe or self.mix:
            if generate_text:
                text_prompt = speech_tokenizer.bop_token + "Please answer by action or text following the speech instruction." + speech_tokenizer.eop_token
                text_idx = 0
                text_history = self.get_text_history()
            else:
                # text_prompt = speech_tokenizer.bop_token + "Please act following the speech instruction." + speech_tokenizer.eop_token
                text_prompt = speech_tokenizer.bop_token + "Please answer by action or text following the speech instruction." + speech_tokenizer.eop_token
        else:
            text_prompt = ""
        
        all_input_ids = []
        all_attention_mask = []

        if generate_text:
            sample_text = speech_tokenizer(text_prompt, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
            all_input_ids.append(sample_text["input_ids"])
            all_attention_mask.append(sample_text["attention_mask"])
            for _ in range(max(0,len(text_history)-(self.window_size - 1))):
                sample_speech = speech_tokenizer(speech_tokenizer.bos_token + speech_tokenizer.eos_token + speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                speech_input_ids = sample_speech["input_ids"]
                speech_attention_mask = sample_speech["attention_mask"]
                        
                text_input_ids = text_history[text_idx]
                text_attention_mask = torch.ones_like(text_input_ids)
                text_idx += 1

                all_input_ids.extend([speech_input_ids, text_input_ids])
                all_attention_mask.extend([speech_attention_mask, text_attention_mask])
        else:
            for i in range(max(0,int(t/self.predict_action_frames)-self.window_size)):
                if self.moe or self.mix:
                    text_prompt += speech_tokenizer.bos_token + speech_tokenizer.eos_token + speech_tokenizer.bot_token + speech_tokenizer.silence_token + speech_tokenizer.eot_token
                else:
                    text_prompt += speech_tokenizer.bos_token + speech_tokenizer.eos_token
            sample_text = speech_tokenizer(text_prompt, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")

            all_input_ids.append(sample_text["input_ids"])
            all_attention_mask.append(sample_text["attention_mask"])

        if self.video_mode:
            if self.moe or self.mix:
                kwargs = dict(
                        mode='VLA_Video_moe',
                        padding="longest",
                    )
            else:
                kwargs = dict(
                        mode='VLA_Video',
                        padding="longest",
                    )
            pos_inputs = self.processor.video_process(text="", video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt", **kwargs)
        else:
            pos_inputs = self.processor.video_process(text="", video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt", **self.kwargs)
        
        if self.video_mode:
            self.add_image(pos_inputs)
            
            # 获取历史图像和动作
            history = self.get_history()
            action_history = self.get_action_history()

            # 遍历历史图像
            for i in range(len(history)):
                sample_speech = speech_tokenizer(speech_tokenizer.bos_token + speech_tokenizer.eos_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                speech_input_ids = sample_speech["input_ids"]
                speech_attention_mask = sample_speech["attention_mask"]

                img_input_ids = history[i]['input_ids']
                img_attention_mask = history[i]['attention_mask']
                
                # 对应的动作
                if i < len(action_history):
                    act_input_ids = action_history[i]
                    
                    # 动作的token_type_ids和attention_mask分别填充为全0和全1
                    act_attention_mask = torch.ones_like(act_input_ids)
                    
                    # 交替添加图像和动作数据
                    if self.moe or self.mix:
                        if generate_text:
                            answer_input_ids = torch.cat([speech_tokenizer(speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"],text_history[text_idx]],dim=1)
                            answer_attention_mask = torch.ones_like(answer_input_ids)
                            text_idx += 1
                        else:
                            sample_answer = speech_tokenizer(speech_tokenizer.bot_token + speech_tokenizer.silence_token + speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                            answer_input_ids = sample_answer["input_ids"]
                            answer_attention_mask = sample_answer["attention_mask"]
                        all_input_ids.extend([speech_input_ids, img_input_ids, answer_input_ids, act_input_ids])
                        all_attention_mask.extend([speech_attention_mask, img_attention_mask, answer_attention_mask, act_attention_mask])
                    else:
                        all_input_ids.extend([speech_input_ids, img_input_ids, act_input_ids])
                        all_attention_mask.extend([speech_attention_mask, img_attention_mask, act_attention_mask])
                else:
                    # 若没有对应的动作，添加图像数据
                    all_input_ids.extend([speech_input_ids, img_input_ids])
                    all_attention_mask.extend([speech_attention_mask, img_attention_mask])
            # 拼接所有的input_ids、token_type_ids和attention_mask
            concatenated_input_ids = torch.cat(all_input_ids, dim=1)
            concatenated_attention_mask = torch.cat(all_attention_mask, dim=1)
            
        else:
            final_inputs = pos_inputs
        
        if self.moe:
            bot_sample = speech_tokenizer(speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
            bot_id = bot_sample["input_ids"]
            bot_attention_mask = bot_sample["attention_mask"]
            concatenated_input_ids = torch.cat([concatenated_input_ids, bot_id], dim=1)
            concatenated_attention_mask = torch.cat([concatenated_attention_mask, bot_attention_mask], dim=1)        
        elif self.mix:
            bot_sample = speech_tokenizer(speech_tokenizer.bot_token + speech_tokenizer.silence_token + speech_tokenizer.eot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
            bot_id = bot_sample["input_ids"]
            bot_attention_mask = bot_sample["attention_mask"]
            boa_sample = vision_tokenizer(vision_tokenizer.boa_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
            boa_id = boa_sample["input_ids"]
            boa_attention_mask = boa_sample["attention_mask"]
            concatenated_input_ids = torch.cat([concatenated_input_ids, bot_id, boa_id], dim=1)
            concatenated_attention_mask = torch.cat([concatenated_attention_mask, bot_attention_mask, boa_attention_mask], dim=1)

        if self.use_fast: 
            last_token_id = vision_tokenizer.pad_token_id - 1
            allowed_token_ids = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
            
            with torch.no_grad():
                if self.moe:
                    outputs = self.model.generate(
                        input_ids=[concatenated_input_ids.to(self.device)],
                        generation_config=self.GENERATION_CONFIG,
                        fbank_feature=fbank_feature.unsqueeze(0).to(self.device),
                        fbank_feature_len=torch.tensor([fbank_feature_len]).to(self.device),
                        max_new_tokens=40,
                        logits_processor=[action_id_processor],
                        vla=True
                    )
                else:
                    outputs = self.model.generate(
                        input_ids=[concatenated_input_ids.to(self.device)],
                        generation_config=self.GENERATION_CONFIG,
                        fbank_feature=fbank_feature.unsqueeze(0).to(self.device),
                        fbank_feature_len=torch.tensor([fbank_feature_len]).to(self.device),
                        max_new_tokens=40,
                        logits_processor=[action_id_processor]
                    )
            
            if self.moe:
                valid_hidden_tensor = outputs[2]
                if generate_text:
                    text_outputs = outputs[0]
                    orig_text_outputs = text_outputs
                    if text_outputs[0][-1] != self.eot_token_id:
                        text_outputs[0][-1] = self.eot_token_id
                    text_pred = speech_tokenizer.decode(text_outputs[0])
                    self.add_text(orig_text_outputs.detach().cpu())
                outputs = outputs[1]
            
            # omit the eoa token
            orig_outputs = outputs
            if self.moe:
                outputs = outputs[:, 1:-1]
            else:
                outputs = outputs[:, :-1]
            last_token_id_tensor = torch.tensor(last_token_id, dtype=outputs.dtype, device=outputs.device)
            processed_outputs = last_token_id_tensor - outputs
            action_outputs = self.action_tokenizer.decode(
                processed_outputs, time_horizon=self.predict_action_frames, action_dim=self.action_dim
            )
            action = action_outputs[0]
            if self.video_mode:
                self.add_action(orig_outputs.detach().cpu())

        else:
            pass
        
        # unnormalize action
        action = self.unormalize_action(action)

        # NOTE(zbzhu): Flip the gripper action here
        # refer to https://github.com/openvla/openvla/blob/1b024f242eda833dc8e321953f25cfd5f3d2f76d/experiments/robot/libero/run_libero_eval.py#L225
        action[..., -1] = np.where(action[..., -1] > 0, 1, -1)

        if self.moe and self.generation and len(valid_hidden_tensor) > 0:
            dots = speech_tokenizer(
                ['。','！','？','.','!','?','\n','*\n','**\n','\n\n','.\n'], return_tensors="pt", add_special_tokens=False
            ).input_ids.squeeze()

            valid_out = outputs[(outputs != 128261) & (outputs != 128260)].detach().cpu()
            valid_hidden_tensor = torch.cat(valid_hidden_tensor, dim=1)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                valid_hidden_tensor = self.model.speech_expert.generator_proj(valid_hidden_tensor)
            valid_hidden_tensor = valid_hidden_tensor.detach().cpu()

            split_idx = -1
            for d in dots:
                positions = torch.where(valid_out == d)[0]
                if len(positions) > 0 and positions[-1].item() > split_idx:
                    split_idx = positions[-1].item()

            if len(self.tts_features) == 0:
                self.tts_features.append(valid_hidden_tensor)
            else:
                if self.tts_features[-1].shape[1] + valid_hidden_tensor.shape[1] > 20 and split_idx >= 0:
                    self.tts_features[-1] = torch.cat(
                        [self.tts_features[-1], valid_hidden_tensor[:, :split_idx+1]], dim=1
                    )
                    self.tts_features.append(valid_hidden_tensor[:, split_idx+1:])
                else:
                    self.tts_features[-1] = torch.cat(
                        [self.tts_features[-1], valid_hidden_tensor], dim=1
                    )

        
        if self.use_one_step:
            # only one step
            action_pred = action[0:1]
        else:
            # action chunk
            action_pred = action
        
        if self.use_cot:
            return action_pred, thought
        else:
            if generate_text:
                return text_pred, action_pred
            else:
                return action_pred
    
    def step_onlyspeech(self, fbank_feature, fbank_feature_len, t, task, image=None):

        self.GENERATION_CONFIG = GenerationConfig(
            pad_token_id=self.model.config.pad_token_id,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.eot_token_id,
            do_sample=False,
        )

        if self.moe or (self.mix):
            speech_tokenizer = self.tokenizer[0]
            vision_tokenizer = self.tokenizer[1]
        else:
            speech_tokenizer = self.tokenizer
            vision_tokenizer = self.tokenizer

        text_history = self.get_text_history()
        all_input_ids = []
        all_attention_mask = []
        sample_prompt = speech_tokenizer(speech_tokenizer.bop_token + self.prompts[task] +speech_tokenizer.eop_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
        all_input_ids.append(sample_prompt["input_ids"])
        all_attention_mask.append(sample_prompt["attention_mask"])

        for i in range(len(text_history)):
            if self.moe or self.mix:
                sample_speech = speech_tokenizer(speech_tokenizer.bos_token + speech_tokenizer.eos_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                speech_input_ids = sample_speech["input_ids"]
                speech_attention_mask = sample_speech["attention_mask"]

                if i == 0 and image is not None:
                    question_image = Image.open(image).resize((200, 200))
                    image_x = [self.image_processor(question_image, return_tensors="pt")["pixel_values"].squeeze(0).cuda()]
                    tensor_frames = torch.stack(image_x, dim=0)
                    image_code = self.image_tokenizer.encode(tensor_frames)

                    image_prompt = self.processor.format_video_prompt(image_code)
                    sample_img = vision_tokenizer(image_prompt, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                    image_input_ids = sample_img["input_ids"]
                    image_attention_mask = sample_img["attention_mask"]
                else:
                    sample_image = vision_tokenizer(vision_tokenizer.boi_token + vision_tokenizer.eoi_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                    image_input_ids = sample_image["input_ids"]
                    image_attention_mask = sample_image["attention_mask"]
                    
                text_input_ids = torch.cat([speech_tokenizer(speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")["input_ids"],text_history[i]],dim=1)
                text_attention_mask = torch.ones_like(text_input_ids)

                action_tokens = np.array([self.dummy_action] * self.predict_action_frames)
                if isinstance(action_tokens, list):
                    tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                    # Concatenate tensors along the first dimension
                    action_tokens = torch.cat(tensor_list, dim=0)
                action_tokens = action_tokens.reshape(-1, self.predict_action_frames, action_tokens.shape[-1])
                action_ids = self.action_tokenizer(action_tokens)
                last_vocab_idx = vision_tokenizer.pad_token_id - 1
                action_ids = [last_vocab_idx - torch.tensor(id) for id in action_ids]
                action_sample = self.wrap_action_sequence(action_ids[0].tolist()).unsqueeze(0) 
                        
                all_input_ids.extend([speech_input_ids, image_input_ids, text_input_ids, action_sample])
                all_attention_mask.extend([speech_attention_mask, image_attention_mask, text_attention_mask, torch.ones_like(action_sample, dtype=torch.long)])
            else:
                sample_speech = speech_tokenizer(speech_tokenizer.bos_token + speech_tokenizer.eos_token + speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                speech_input_ids = sample_speech["input_ids"]
                speech_attention_mask = sample_speech["attention_mask"]
                    
                text_input_ids = text_history[i]
                text_attention_mask = torch.ones_like(text_input_ids)
                        
                all_input_ids.extend([speech_input_ids, text_input_ids])
                all_attention_mask.extend([speech_attention_mask, text_attention_mask])
        
        if self.moe or self.mix:
            sample_speech = speech_tokenizer(speech_tokenizer.bos_token + speech_tokenizer.eos_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
            speech_input_ids = sample_speech["input_ids"]
            speech_attention_mask = sample_speech["attention_mask"]

            if len(text_history) == 0 and image is not None:
                question_image = Image.open(image).resize((200, 200))
                image_x = [self.image_processor(question_image, return_tensors="pt")["pixel_values"].squeeze(0).cuda()]
                tensor_frames = torch.stack(image_x, dim=0)
                image_code = self.image_tokenizer.encode(tensor_frames)

                image_prompt = self.processor.format_video_prompt(image_code)
                sample_img = vision_tokenizer(image_prompt, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"]
                image_attention_mask = sample_img["attention_mask"]
            else:
                sample_image = vision_tokenizer(vision_tokenizer.boi_token + vision_tokenizer.eoi_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
                image_input_ids = sample_image["input_ids"]
                image_attention_mask = sample_image["attention_mask"]

            sample_bot = speech_tokenizer(speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
            bot_input_ids = sample_bot["input_ids"]
            bot_attention_mask = sample_bot["attention_mask"]

            all_input_ids.extend([speech_input_ids, image_input_ids, bot_input_ids])
            all_attention_mask.extend([speech_attention_mask, image_attention_mask, bot_attention_mask])
        else:
            sample_speech = speech_tokenizer(speech_tokenizer.bos_token + speech_tokenizer.eos_token + speech_tokenizer.bot_token, padding=False, return_token_type_ids=False, add_special_tokens=False, return_tensors="pt")
            speech_input_ids = sample_speech["input_ids"]
            speech_attention_mask = sample_speech["attention_mask"]
            all_input_ids.extend([speech_input_ids])
            all_attention_mask.extend([speech_attention_mask])
        # 拼接所有的input_ids、token_type_ids和attention_mask
        concatenated_input_ids = torch.cat(all_input_ids, dim=1)
        concatenated_attention_mask = torch.cat(all_attention_mask, dim=1)

        with torch.no_grad():
            if self.moe:
                last_token_id = vision_tokenizer.pad_token_id - 1
                allowed_token_ids = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
                action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)

                outputs = self.model.generate(
                    input_ids=[concatenated_input_ids.to(self.device)],
                    generation_config=self.GENERATION_CONFIG,
                    fbank_feature=fbank_feature.unsqueeze(0).to(self.device),
                    fbank_feature_len=torch.tensor([fbank_feature_len]).to(self.device),
                    max_new_tokens=9,
                    logits_processor=[action_id_processor],
                    vla=False
                )

                valid_hidden_tensor = outputs[2]

            else:
                outputs = self.model.generate(
                    input_ids=[concatenated_input_ids.to(self.device)],
                    generation_config=self.GENERATION_CONFIG,
                    fbank_feature=fbank_feature.unsqueeze(0).to(self.device),
                    fbank_feature_len=torch.tensor([fbank_feature_len]).to(self.device),
                    max_new_tokens=9,
                )
        
        if self.moe:
            outputs = outputs[0]

        # omit the eoa token
        orig_outputs = outputs
        if outputs[0][-1] != self.eot_token_id:
            outputs[0][-1] = self.eot_token_id
        text_pred = speech_tokenizer.decode(outputs[0])
        self.add_text(orig_outputs.detach().cpu())

        if self.moe and self.generation and len(valid_hidden_tensor) > 0:
            dots = speech_tokenizer(
                ['。','！','？','.','!','?','\n','*\n','**\n','\n\n','.\n'], return_tensors="pt", add_special_tokens=False
            ).input_ids.squeeze()

            valid_out = outputs[(outputs != 128261) & (outputs != 128260)].detach().cpu()
            valid_hidden_tensor = torch.cat(valid_hidden_tensor, dim=1)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                valid_hidden_tensor = self.model.speech_expert.generator_proj(valid_hidden_tensor)
            valid_hidden_tensor = valid_hidden_tensor.detach().cpu()

            split_idx = -1
            for d in dots:
                positions = torch.where(valid_out == d)[0]
                if len(positions) > 0 and positions[-1].item() > split_idx:
                    split_idx = positions[-1].item()

            if len(self.tts_features) == 0:
                self.tts_features.append(valid_hidden_tensor)
            else:
                if self.tts_features[-1].shape[1] + valid_hidden_tensor.shape[1] > 20 and split_idx >= 0:
                    self.tts_features[-1] = torch.cat(
                        [self.tts_features[-1], valid_hidden_tensor[:, :split_idx+1]], dim=1
                    )
                    self.tts_features.append(valid_hidden_tensor[:, split_idx+1:])
                else:
                    self.tts_features[-1] = torch.cat(
                        [self.tts_features[-1], valid_hidden_tensor], dim=1
                    )

        return text_pred

    def unormalize_action(self, action):
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
        action = 0.5 * (action + 1) * (action_high - action_low) + action_low
        return action
