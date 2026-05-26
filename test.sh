cd "$(cd $(dirname $0); pwd)"

pip3 install line-profiler

DATASET=path_to_your_dataset  # formatting examples given in example.json
MODEL=path_to_your_model # Path to pre-trained video-SALMONN 2+ ckpt
MODEL_BASE=path_to_your_model_base # Path to Qwen3VL model base
LR=2e-5
BS=1
ACCUM_STEPS=1
RUN_NAME="qwen2vl-baseline"
DEEPSPEED=2
TRAIN_LLM=False
TRAIN_PROJ=False
TRAIN_ENC=False
EPOCH=1
MEMGROUPSIZE=0
WORKINGMEMSIZE=0
SEARCH_TYPE="none"
RETAIN_FACTOR="diversity"
LAMBDAS="0.0"
DIV_FACTOR=0.0
SLOT_TYPE="none"
EMA_FACTOR=0.1
LAG_DISTANCES="0"
ALL_TTT_LAYERS="0"
TOTAL_TIMESTAMPS=5000
SHRINK_CONFIG="none"

MAX_PIXELS=176400
MIN_PIXELS=784

SAVE_STEPS=1000

MIN_FRAMES=4
MAX_FRAMES=128
INTERVAL=0.2

USE_LORA=False
LORA_R=128
LORA_ALPHA=256
LORA_DROPOUT=0.05
LORA_CKPT=No
NUM_SAMPLE=1
DO_SAMPLE=False
NO_AUDIO=False

MODEL_TYPE=moe

NUM_WORKER=8
FIXED_MEMORY_SIZE=0
FIXED_MEMORY_SIZE_AUDIO=0
STEPSIZE=0
TTT_TYPE=sim
CG_max_iter=0
TTT_HIDDEN_SIZE=4
TTT_NUM_HEADS=8
GLOBAL_FACTOR=0.0
LOSS_FACTOR=0.0
SAVEMODE_STOPLAYER=0
SHRINK_SIZE=0

mkdir -p output
mkdir -p dataset

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --model_base) MODEL_BASE="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --run_name) RUN_NAME="$2"; shift ;;
        --bs) BS="$2"; shift ;;
        --accum_steps) ACCUM_STEPS="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --deepspeed) DEEPSPEED="$2"; shift ;;
        --train_llm) TRAIN_LLM=True ;;
        --train_proj) TRAIN_PROJ=True ;;
        --train_enc) TRAIN_ENC=True ;;
        --max_pixels) MAX_PIXELS="$2"; shift ;;
        --min_pixels) MIN_PIXELS="$2"; shift ;;
        --epoch) EPOCH="$2"; shift ;;
        --save_steps) SAVE_STEPS="$2"; shift ;;
        --min_frames) MIN_FRAMES="$2"; shift ;;
        --max_frames) MAX_FRAMES="$2"; shift ;;
        --interval) INTERVAL="$2"; shift ;;
        --use_lora) USE_LORA=True ;;
        --lora_r) LORA_R="$2"; shift ;;
        --lora_alpha) LORA_ALPHA="$2"; shift ;;
        --lora_dropout) LORA_DROPOUT="$2"; shift ;;
        --lora_ckpt) LORA_CKPT="$2"; shift ;;
        --do_sample) DO_SAMPLE=True ;;
        --num_sample) NUM_SAMPLE="$2"; shift ;;
        --no_audio) NO_AUDIO=True ;;
        --num_worker) NUM_WORKER="$2"; shift ;;
        --fixed_memory_size) FIXED_MEMORY_SIZE="$2"; shift ;;
        --stepsize) STEPSIZE="$2"; shift ;;
        --ttt_type) TTT_TYPE="$2"; shift ;;
        --cg_max_iter) CG_max_iter="$2"; shift ;;
        --ttt_hidden_size) TTT_HIDDEN_SIZE="$2"; shift ;;
        --ttt_num_heads) TTT_NUM_HEADS="$2"; shift ;;
        --model_type) MODEL_TYPE="$2"; shift ;;
        --fixed_memory_size_audio) FIXED_MEMORY_SIZE_AUDIO="$2"; shift ;;
        --memgroupsize) MEMGROUPSIZE="$2"; shift ;;
        --workingmemsize) WORKINGMEMSIZE="$2"; shift ;;
        --search_type) SEARCH_TYPE="$2"; shift ;;
        --retain_factor) RETAIN_FACTOR="$2"; shift ;;
        --lambdas) LAMBDAS="$2"; shift ;;
        --div_factor) DIV_FACTOR="$2"; shift ;;
        --slot_type) SLOT_TYPE="$2"; shift ;;
        --ema_factor) EMA_FACTOR="$2"; shift ;;
        --lag_distances) LAG_DISTANCES="$2"; shift ;;
        --all_ttt_layers) ALL_TTT_LAYERS="$2"; shift ;;
        --total_timestamps) TOTAL_TIMESTAMPS="$2"; shift ;;
        --global_factor) GLOBAL_FACTOR="$2"; shift ;;
        --loss_factor) LOSS_FACTOR="$2"; shift ;;
        --savemode_stoplayer) SAVEMODE_STOPLAYER="$2"; shift ;;
        --shrinksize) SHRINK_SIZE="$2"; shift ;;
        --shrink_config) SHRINK_CONFIG="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done



i=0
export CUDA_VISIBLE_DEVICES=1

torchrun --nproc_per_node=1 --master_port=12349 \
    qwenvl/train/train_qwen.py \
        --model_base $MODEL_BASE \
        --run_test True \
        --pred_rank 0 \
        --deepspeed /opt/tiger/thu_qwenvl/scripts/zero${DEEPSPEED}.json \
        --model_name_or_path "$MODEL" \
        --dataset_use $DATASET \
        --tune_mm_vision $TRAIN_ENC \
        --tune_mm_mlp $TRAIN_PROJ \
        --tune_mm_llm $TRAIN_LLM \
        --bf16 \
        --output_dir output/test \
        --num_train_epochs $EPOCH \
        --per_device_train_batch_size $BS \
        --gradient_accumulation_steps $ACCUM_STEPS \
        --max_pixels $MAX_PIXELS \
        --min_pixels $MIN_PIXELS \
        --video_max_frame_pixels $MAX_PIXELS \
        --video_min_frame_pixels $MIN_PIXELS \
        --eval_strategy "no" \
        --save_strategy "steps" \
        --save_steps $SAVE_STEPS \
        --save_total_limit 5 \
        --learning_rate $LR \
        --weight_decay 0 \
        --warmup_ratio 0.03 \
        --max_grad_norm 1 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 1310720 \
        --gradient_checkpointing True \
        --dataloader_num_workers $NUM_WORKER \
        --run_name $RUN_NAME \
        --report_to wandb \
        --video_min_frames $MIN_FRAMES \
        --video_max_frames $MAX_FRAMES \
        --base_interval $INTERVAL \
        --use_lora $USE_LORA \
        --lora_r $LORA_R \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_ckpt $LORA_CKPT \
        --num_sample $NUM_SAMPLE \
        --do_sample $DO_SAMPLE \
        --no_audio $NO_AUDIO \
        --fixed_memory_size $FIXED_MEMORY_SIZE \
        --stepsize $STEPSIZE \
        --ttt_type $TTT_TYPE \
        --cg_max_iter $CG_max_iter \
        --ttt_hidden_size $TTT_HIDDEN_SIZE \
        --ttt_num_heads $TTT_NUM_HEADS \
        --fixed_memory_size_audio $FIXED_MEMORY_SIZE_AUDIO \
        --memgroupsize $MEMGROUPSIZE \
        --workingmemsize $WORKINGMEMSIZE \
        --search_type $SEARCH_TYPE \
        --retain_factor $RETAIN_FACTOR \
        --lambdas $LAMBDAS \
        --div_factor $DIV_FACTOR \
        --ema_factor $EMA_FACTOR \
        --lag_distances $LAG_DISTANCES \
        --slot_type $SLOT_TYPE \
        --all_ttt_layers $ALL_TTT_LAYERS \
        --total_timestamps $TOTAL_TIMESTAMPS \
        --global_factor $GLOBAL_FACTOR \
        --loss_factor $LOSS_FACTOR \
        --savemode_stoplayer $SAVEMODE_STOPLAYER \
        --shrinksize $SHRINK_SIZE \
        --shrinksize_config $SHRINK_CONFIG \
        --model_type $MODEL_TYPE