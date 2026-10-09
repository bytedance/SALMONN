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

import argparse
import os
import sys

sys.path.append(os.path.join(os.environ["ELLSA_BASE_PATH"], "reference/Emu3"))
from emu3.mllm import LlamaWithSpeech
from transformers import AutoTokenizer, AutoConfig
import torch

parser = argparse.ArgumentParser(description="Merge the speech-stage LoRA into its Llama weights")
parser.add_argument("checkpoint")
parser.add_argument("--output")
args = parser.parse_args()
model_config = AutoConfig.from_pretrained(args.checkpoint)
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
model = LlamaWithSpeech.from_pretrained(
    args.checkpoint,
    config=model_config,
    tokenizer=tokenizer,
    llama_path=os.environ["LLAMA_CKPT_PATH"],
    speech_encoder_path="",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    peft=True,
    freeze=True,
    encoder_type="zipformer2",
)
model.merge_lora()
output = args.output or args.checkpoint + "-merged"
model.save_pretrained(output)
tokenizer.save_pretrained(output)
