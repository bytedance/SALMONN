# Shared paths and torchrun settings. Source after setting REPO_ROOT.
export ELLSA_BASE_PATH="$REPO_ROOT"
export ELLSA_DATA_PATH="${ELLSA_DATA_PATH:-$REPO_ROOT/training_data}"
export LLAMA_CKPT_PATH="${LLAMA_CKPT_PATH:-$REPO_ROOT/ckpt/Llama-3.1-8B-Instruct}"
export COSY_CKPT_PATH="${COSY_CKPT_PATH:-$REPO_ROOT/ckpt/CosyVoice2-0.5B}"
export UNIVLA_CKPT_PATH="${UNIVLA_CKPT_PATH:-$REPO_ROOT/ckpt/UNIVLA_LIBERO_VIDEO_BS192_8K}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/reference:$REPO_ROOT/reference/Emu3:$REPO_ROOT/reference/cosyvoice/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"
GPU_NUM=${GPU_NUM:-$(nvidia-smi -L | wc -l)}
NODE_NUM=${NODE_NUM:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
DISTRIBUTED_ARGS=(--nnodes "$NODE_NUM" --node-rank "$NODE_RANK" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")
if [[ "$NODE_NUM" == 1 ]]; then
    DISTRIBUTED_ARGS=(--standalone)
fi
ACTION_TOKENIZER_PATH=${ACTION_TOKENIZER_PATH:-$REPO_ROOT/pretrain/fast}
# Speech audio is external to the text-only release; provide a manifest with real WAV paths.
: "${DATA_SPEECH_PATH:?Set DATA_SPEECH_PATH to a speech annotation JSON with input WAV paths}"
cd "$REPO_ROOT"
