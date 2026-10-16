#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train.py "$@"
