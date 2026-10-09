#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
source "$REPO_ROOT/scripts/training_env.sh"

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

DATAPATH="${DATAPATH:-$DATA_SPEECH_PATH}"
EXP_NAME="${EXP_NAME:-ellsa_speech}"

exec torchrun \
    --nproc_per_node=${GPU_NUM} \
    "${DISTRIBUTED_ARGS[@]}" \
    train/train_moe.py \
    --model_name_or_path "$LLAMA_CKPT_PATH" \
    --model_config_path "$LLAMA_CKPT_PATH/config.json" \
    --deepspeed scripts/sft/zero2.json \
    --output_dir "output/"${EXP_NAME} \
    --learning_rate 2e-4 \
    --null_prompt_prob 0 \
    --weight_decay 0.1 \
    --min_learning_rate 1e-5 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path "$DATAPATH" \
    --max_steps "${MAX_STEPS:-20000}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-12}" \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps "${WARMUP_STEPS:-200}" \
    --per_device_train_batch_size "${BATCH_SIZE:-4}" \
    --frames 2 \
    --action_frames 10 \
    --max_position_embeddings 6400 \
    --seed 42 \
    --logging_steps "${LOGGING_STEPS:-10}" \
    --gradient_checkpointing True \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS:-2000}" \
    --eval_strategy no \
    --apply_loss_on_only_vision False \
    --apply_loss_on_only_action True \
    --actions True \
    --actions_format "fast" \
    --use_gripper True \
    --video_format "interleave" \
    --action_tokenizer_path "$ACTION_TOKENIZER_PATH" \
    --report_to "${REPORT_TO:-none}" \
    --run_name ${EXP_NAME} \
    --speech True \
    --speech_only True \
    --peft True \
    --freeze False \
    --llama True \
    --debug_mode False \
    --time_block 1.0 \
    --token_per_second 8 \
    --encoder_type "zipformer2" \
    --speech_encoder_path "${SPEECH_ENCODER_PATH:?Set SPEECH_ENCODER_PATH to the pretrained SPEAR encoder checkpoint}" "$@"
