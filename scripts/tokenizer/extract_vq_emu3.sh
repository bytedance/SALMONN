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

CUDA_VISIBLE_DEVICES=0 python3 models/tokenizer/emu3_tokenizer.py 0 & 
CUDA_VISIBLE_DEVICES=1 python3 models/tokenizer/emu3_tokenizer.py 1 & 
CUDA_VISIBLE_DEVICES=2 python3 models/tokenizer/emu3_tokenizer.py 2 & 
CUDA_VISIBLE_DEVICES=3 python3 models/tokenizer/emu3_tokenizer.py 3 & 
CUDA_VISIBLE_DEVICES=4 python3 models/tokenizer/emu3_tokenizer.py 4 & 
CUDA_VISIBLE_DEVICES=5 python3 models/tokenizer/emu3_tokenizer.py 5 & 
CUDA_VISIBLE_DEVICES=6 python3 models/tokenizer/emu3_tokenizer.py 6 & 
CUDA_VISIBLE_DEVICES=7 python3 models/tokenizer/emu3_tokenizer.py 7

