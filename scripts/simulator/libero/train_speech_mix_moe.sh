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

DATAPATH="${DATAPATH:-$ELLSA_DATA_PATH/libero/metadata.json}"
EXP_NAME="${EXP_NAME:-ellsa_libero_mix}"

: "${SPEECH_EXPERT_PATH:?Set SPEECH_EXPERT_PATH to the merged speech-stage checkpoint}"

exec torchrun \
    --nproc_per_node=${GPU_NUM} \
    "${DISTRIBUTED_ARGS[@]}" \
    train/train_moe.py \
    --model_name_or_path "$SPEECH_EXPERT_PATH" \
    --speech_expert_path "$SPEECH_EXPERT_PATH" \
    --speech_encoder_path "" \
    --vision_expert_path "$UNIVLA_CKPT_PATH" \
    --config_speech_path "$LLAMA_CKPT_PATH/config.json" \
    --config_vision_path configs/moe_fast_video.json \
    --deepspeed scripts/sft/zero2.json \
    --output_dir "output/"${EXP_NAME} \
    --learning_rate 1e-4 \
    --null_prompt_prob 0 \
    --weight_decay 0.1 \
    --min_learning_rate 0.5e-5 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path "$DATAPATH" \
    --data_speech_path "$DATA_SPEECH_PATH" \
    --max_steps "${MAX_STEPS:-50000}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps "${WARMUP_STEPS:-500}" \
    --per_device_train_batch_size "${BATCH_SIZE:-1}" \
    --frames 2 \
    --action_frames 10 \
    --max_position_embeddings 6400 \
    --seed 42 \
    --logging_steps "${LOGGING_STEPS:-10}" \
    --gradient_checkpointing True \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS:-5000}" \
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
    --mix True \
    --moe True \
    --peft True \
    --llama True \
    --freeze False \
    --debug_mode False \
    --contemporary True \
    --stop True \
    --stop_ratio 0.12 \
    --refuse True \
    --refuse_ratio 0.08 \
    --context_vqa True \
    --token_per_second 8 \
    --merge_speech_lora True \
    --lora_modules qkv \
    --encoder_type zipformer2 \
    "$@"
