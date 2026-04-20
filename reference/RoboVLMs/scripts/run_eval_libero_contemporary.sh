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


export OMP_NUM_THREADS=16
export PYTHONWARNINGS=ignore
export ELLSA_BASE_PATH="/mnt/bn/audio-visual-llm-data5/wangsiyin/models/VLA_speech" # set this path to your ELLSA code base
export ELLSA_DATA_PATH="/mnt/bn/audio-visual-llm-data5/wangsiyin/datasets/ELLSA_test_data" # set this path to your ELLSA test data path
export COSY_CKPT_PATH="/mnt/bn/audio-visual-llm-data6/ckpts/CosyVoice2-0.5B" # set this path to your COSYVOICE ckpt path
export LLAMA_CKPT_PATH="/mnt/bn/audio-visual-llm-data5/wangsiyin/models/Llama-3.1-8B-Instruct" # set this path to your LLama ckpt path
export UNIVLA_CKPT_PATH="/mnt/bn/audio-visual-llm-data5/wangsiyin/models/UniVLA/ckpt/UNIVLA_LIBERO_VIDEO_BS192_8K" # set this path to your UniVLA ckpt path
export VISION_VQ_PATH="/mnt/bn/audio-visual-llm-data5/wangsiyin/models/UniVLA/ckpt/Emu3-VisionVQ" # set this path to your Emu3-VisionVQ ckpt path

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"

ckpt_dir=$1
GPUS_PER_NODE=$ARNOLD_WORKER_GPU

python eval/libero/evaluate_libero_emu_contemporary.py \
--emu_hub $ckpt_dir \
--no_nccl \
--no_action_ensemble \
--task_suite_name libero_object \
--cache_root /mnt/bn/audio-visual-llm-data5/wangsiyin/models/VLA_speech/cache \
--speech True \
--moe True \
--merge_speech_lora True \
--lora_modules qkv \
--encoder_type zipformer2 \
--generate True \
--silence True \

# --task_suite_name \ # libero_test_set (available: libero_object, libero_goal, libero_10, libero_spatial)

### Choose one setting to test
# --speech_task_suite_name llama_questions \ # speaking while acting
# --context_vqa True \ # context vqa
# --silence True \ # speech-conditioned robot manipulation
# --stop True \ # action barge-in

# --generate True \ # whether to generate speech

# --cache_root \ # path to save results
