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

echo "All parameters: $@"

export TOKENIZERS_PARALLELISM=false

PROJECT_ROOT=$(cd $(dirname $0); pwd)
cd $PROJECT_ROOT
cd ..

pip install -r requirements.txt

TRAINING_DATA=/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/preprocess_dataset/ytb0-62kCapHumanDimQa.json
MODEL_ID=/mnt/bn/tiktok-mm-4/aiic/public/model/OV-Qwen2-7B-AM9
MODEL_BASE=/mnt/bn/tiktok-mm-4/aiic/public/model/OV-Qwen2-7B-AM9
SAVE_MODEL_NAME=debug # 
LOAD_FROM_LORA=False # True # 
EPOCHS=4 # 5
TRAIN_BS=1 # 1 # 2
ACCUM_STEPS=1
SAVE_STEPS=1000 # 2000

LR=2e-5
MAX_TIME=30 # 30
FPS=1

UNFREEZE_MM_VISION_TOWER=False
FREEZE_BACKBONE=True
DEEPSPEED_TYPE=zero2
SEED=2024
LORA_ENABLE=False
ADD_TIME_TOKEN=False
LORA_R=128
LORA_ALPHA=256
LORA_DROPOUT=0.05
WINQF_SECOND=0.5
PRETRAIN_WEIGHT=None
MM_POOLING_POSITION=after
LOAD_FULL=False
MERGE_AND_NEW_LORA=False
AUDIO_VISUAL=False

DO_DEMO=False
DO_TEST=False
TEST_DATA_PATH=/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/preprocess_dataset/videomme_audioVisual_test.json
TEST_ID=debug

DPO_TRAIN=False
CE_LOSS_WEIGHT=0.1
WITH_CE_LOSS=False

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --use_lora) LORA_ENABLE=True; ;;
        --training_data) TRAINING_DATA="$2"; shift ;;
        --model) MODEL_ID="$2"; shift ;;
        --save_model_name) SAVE_MODEL_NAME="$2"; shift ;;
        --load_from_lora) LOAD_FROM_LORA=True; ;;
        --epochs) EPOCHS="$2"; shift ;;
        --train_bs) TRAIN_BS="$2"; shift ;;
        --accum_steps) ACCUM_STEPS="$2"; shift ;;
        --save_steps) SAVE_STEPS="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --max_time) MAX_TIME="$2"; shift ;;
        --fps) FPS="$2"; shift ;;
        --unfreeze_mm_vision_tower) UNFREEZE_MM_VISION_TOWER=True; ;;
        --no_freeze_backbone) FREEZE_BACKBONE=False; ;;
        --deepspeed_type) DEEPSPEED_TYPE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --add_time_token) ADD_TIME_TOKEN=True; ;;
        --lora_r) LORA_R="$2"; shift ;;
        --lora_alpha) LORA_ALPHA="$2"; shift ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift ;;
        --model_base) MODEL_BASE="$2"; shift ;;
        --winqf_second) WINQF_SECOND="$2"; shift ;;
        --pretrain_weight) PRETRAIN_WEIGHT="$2"; shift ;;
        --mm_pooling_position) MM_POOLING_POSITION="$2"; shift ;;
        --load_full) LOAD_FULL=True; ;;
        --merge_and_new_lora) MERGE_AND_NEW_LORA=True; ;;
        --audio_visual) AUDIO_VISUAL=True; ;;
        --do_demo) DO_DEMO=True; ;;
        --do_test) DO_TEST=True; ;;
        --test_data) TEST_DATA_PATH="$2"; shift ;;
        --test_id) TEST_ID="$2"; shift ;;
        --dpo_train) DPO_TRAIN=True; ;;
        --ce_loss_weight) CE_LOSS_WEIGHT="$2"; shift ;;
        --with_ce_loss) WITH_CE_LOSS=True; ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

SAVE_DIR=output
TEST_OUTPUT_DIR=output/test/$TEST_ID

MODEL=$MODEL_ID
echo "BASE CKPT: $MODEL"

MODEL_MAX_LENGTH=32768
VISION_ENCODER=google/siglip-so400m-patch14-384

FRAMES_UPBOUND=30 # 32
POOLING_STYLE=max
POOLING_STRIDE=2
NEW_LINE_POSITION=grid
IMAGE_ASPECT_RATIO=anyres

WHISPER_PATH=openai/whisper-large-v3
FREEZE_SPEECH_QFORMER=False
FREEZE_FINAL_LINEAR=False
USE_FINAL_LINEAR=True

export HF_HOME="/mnt/bn/tiktok-mm-2/aiic/public/model/huggingface"

if [[ "$DO_DEMO" = "True" ]]; then
    GPU_NUM=1
else
    GPU_NUM=${ARNOLD_WORKER_GPU}
fi

# ${ARNOLD_WORKER_GPU}
torchrun --nproc_per_node=$GPU_NUM --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port=13396 \
    llava/train/train_mem.py \
        --ckpt $MODEL \
        --deepspeed ./scripts/$DEEPSPEED_TYPE.json \
        --max_time $MAX_TIME \
        --load_from_lora $LOAD_FROM_LORA \
        --lora_enable $LORA_ENABLE --lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT --mm_projector_lr 2e-5 \
        --audio_visual $AUDIO_VISUAL \
        --whisper_path $WHISPER_PATH \
        --audio_processor $WHISPER_PATH \
        --freeze_speech_QFormer $FREEZE_SPEECH_QFORMER \
        --use_final_linear $USE_FINAL_LINEAR \
        --freeze_final_linear $FREEZE_FINAL_LINEAR \
        --second_per_window $WINQF_SECOND \
        --second_stride $WINQF_SECOND \
        --version qwen_1_5 \
        --data_path $TRAINING_DATA \
        --vision_tower ${VISION_ENCODER} \
        --image_processor ${VISION_ENCODER} \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --video_fps $FPS \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir ${SAVE_DIR}/${SAVE_MODEL_NAME} \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $TRAIN_BS \
        --per_device_eval_batch_size $TRAIN_BS \
        --gradient_accumulation_steps $ACCUM_STEPS \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps $SAVE_STEPS \
        --save_total_limit 100 \
        --learning_rate $LR \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length ${MODEL_MAX_LENGTH} \
        --gradient_checkpointing True \
        --mm_spatial_pool_mode ${POOLING_STYLE} \
        --mm_spatial_pool_stride ${POOLING_STRIDE} \
        --mm_spatial_pool_out_channels 1152 \
        --image_aspect_ratio ${IMAGE_ASPECT_RATIO} \
        --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
        --mm_patch_merge_type spatial_unpad \
        --mm_newline_position ${NEW_LINE_POSITION} \
        --freeze_backbone $FREEZE_BACKBONE \
        --dataloader_num_workers 8 \
        --disable_tqdm False \
        --unfreeze_mm_vision_tower $UNFREEZE_MM_VISION_TOWER \
        --seed $SEED \
        --add_time_token $ADD_TIME_TOKEN \
        --model_base $MODEL_BASE \
        --pretrain_weight $PRETRAIN_WEIGHT \
        --mm_pooling_position $MM_POOLING_POSITION \
        --load_full $LOAD_FULL \
        --merge_and_new_lora $MERGE_AND_NEW_LORA \
        --do_demo $DO_DEMO \
        --do_test $DO_TEST \
        --test_data_path $TEST_DATA_PATH \
        --test_output_dir $TEST_OUTPUT_DIR \
        --dpo_train $DPO_TRAIN \
        --ce_loss_weight $CE_LOSS_WEIGHT \
        --with_ce_loss $WITH_CE_LOSS;