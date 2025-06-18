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

# Note: 
# To evaluate a PyTorch model in ".bin" format, please add the "--load_from_lora" parameter.
# For Hugginface format models, this parameter is not required.
# To use this script for dpo data generation, you can add "--dpo_train".

#   --use_lora --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 --load_from_lora \

bash run.sh \
    --do_test \
    --test_data /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/preprocess_dataset/videomme_audioVisual_short_test.json \
    --test_id debug \
    --max_time 110 \
    --fps 1 \
    --model /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/video-SALMONN2/output/video_SALMONN_2 \
    --model_base /mnt/bn/tiktok-mm-4/aiic/public/model/OV-Qwen2-7B-AM9 \
    --add_time_token --mm_pooling_position after \
    --audio_visual --winqf_second 0.5

