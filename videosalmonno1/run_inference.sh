#!/bin/bash

# Some specific versions
# pip install ffmpeg-python
# pip install transformers==4.39.2
# pip install accelerate==0.28.0

export HF_HOME="/mnt/bn/tiktok-mm-2/aiic/public/model/huggingface" # Change to your own path
TEST_OUTPUT=output/PRMDPO_with_caption_qa_prev_newdata_MCQsubset_withceloss_videomme
MODEL_PATH=/mnt/bn/tiktok-mm-4/aiic/users/guangzhisun/llava-video/output/models/SoftPRMDPO_with_caption_qa_prev_newdata_MCQsubset_withceloss/checkpoint_11000_qwen
TEST_DATA=/mnt/bn/tiktok-mm-2/aiic/users/guangzhisun/dataprep/preprocessed_data/videomme_audiovisual_test_reasoning.json
VISION_ENCODER=google/siglip-so400m-patch14-384
WHISPER_PATH=/mnt/bn/tiktok-mm-2/aiic/public/model/whisper-large-v3

python3 inference.py \
    --test_output_dir $TEST_OUTPUT \
    --model_name_or_path $MODEL_PATH \
    --output_dir $TEST_OUTPUT \
    --test_data_path $TEST_DATA \
    --vision_tower $VISION_ENCODER \
    --image_processor $VISION_ENCODER \
    --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
    --whisper_path $WHISPER_PATH \
    --audio_processor $WHISPER_PATH \
    --bf16 True \
    --tf32 True \
    --dataloader_num_workers 16 \