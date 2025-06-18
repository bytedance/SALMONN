#!/bin/bash

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

PROJECT_ROOT=$(cd $(dirname $0); pwd)
cd $PROJECT_ROOT

bash run.sh \
    --merge_and_new_lora --max_time 110 --fps 1 \
    --add_time_token --use_lora --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
    --load_from_lora \
    --model /opt/tiger/sft_ytb0-62kFinvideo_av110fLora128_ctn12kWin2_250428/checkpoint-15475.bin \
    --model_base /mnt/bn/tiktok-mm-4/aiic/public/model/OV-Qwen2-7B-AM9 \
    --winqf_second 0.5 --mm_pooling_position after --save_steps 200 --epochs 3 \
    --training_data /mnt/bn/tiktok-mm-5/aiic/users/tangchangli/preprocess_dataset/nips_dpor0_ytb13k.json --save_model_name debug \
    --ce_loss_weight 0.1 --dpo_train --with_ce_loss --audio_visual