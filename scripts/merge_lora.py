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

import sys
import os
sys.path.append("/mnt/bn/audio-visual-llm-data5/wangsiyin/models/UniVLA/reference/Emu3")
from emu3.mllm import Emu3Config, Emu3MoEConfig, Emu3Tokenizer, Emu3MoEWithSpeech, Emu3MoE, LlamaWithSpeech
from transformers import AutoTokenizer, AutoConfig
import torch

llama = True
model_name_or_path = "/mnt/bn/audio-visual-llm-data5/wangsiyin/models/UniVLA/output/speechonly_node32_bs256_step40000_peftnewlora256_asrqa_tokenpersecond8_llama_zipformer/checkpoint-20000"
speech_encoder_path = ""
if llama:
    model_config = AutoConfig.from_pretrained(os.path.join(model_name_or_path,"config.json"))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = LlamaWithSpeech.from_pretrained(
        model_name_or_path,
        config=model_config,
        tokenizer=tokenizer,
        llama_path="/mnt/bn/audio-visual-llm-data5/wangsiyin/models/Llama-3.1-8B-Instruct",
        speech_encoder_path=speech_encoder_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        peft=True,
        freeze=True,
        encoder_type="zipformer2"
    )
else:
    model_config = Emu3MoEConfig.from_pretrained("/mnt/bn/audio-visual-llm-data5/wangsiyin/models/UniVLA/configs/moe_fast_video.json")

    tokenizer = Emu3Tokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=6400,
        padding_side="right",
        use_fast=False,
    )

    model = Emu3MoEWithSpeech.from_pretrained(
        model_name_or_path,
        config=model_config,
        tokenizer=tokenizer,
        speech_encoder_path=speech_encoder_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        peft=True,
        freeze=True
    )

model.merge_lora()

model.save_pretrained(model_name_or_path+"-merged")
tokenizer.save_pretrained(model_name_or_path+"-merged")


"""
model_name_or_path = "/mnt/bn/audio-visual-llm-data5/wangsiyin/models/UniVLA/ckpt/WORLD_MODEL_POSTTRAIN"
model_config = Emu3MoEConfig.from_pretrained("/mnt/bn/audio-visual-llm-data5/wangsiyin/models/UniVLA/configs/moe_fast_video.json")

model = Emu3MoE.from_pretrained(
    model_name_or_path,
    config=model_config,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

torch.save(model.lm_head,"/mnt/bn/audio-visual-llm-data5/wangsiyin/models/UniVLA/ckpt/lm_head_worldmodel.pt")
"""