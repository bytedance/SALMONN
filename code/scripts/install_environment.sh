#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -c 'import platform, sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"; assert sys.platform == "linux" and platform.machine() == "x86_64", "Use Linux x86_64"'

# Build tools must precede DeepSpeed metadata generation (--no-build-isolation).
python -m pip install --index-url https://pypi.org/simple -c requirements-lock.txt \
  setuptools==80.9.0 wheel==0.45.1 packaging==25.0 ninja==1.13.0
# CUDA wheels come from PyTorch; their ordinary dependencies come from PyPI.
python -m pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu126 -c requirements-lock.txt \
  torch==2.9.0+cu126 torchaudio==2.9.0+cu126
python -m pip install --index-url https://pypi.org/simple --no-build-isolation -r requirements.txt
python -m pip check
# Verify binary extensions and public entry-point imports, beyond pip metadata.
python - <<'PYTHON'
import importlib.metadata
import torch, torchaudio, transformers, deepspeed
from dataset import CoSpeechGestureDataset
from modeling_ellsa2 import CoSpeechGestureLM
from streaming_transformer_encoder_v2.audio_encoder import CausalAudioEncoder
if torch.cuda.is_available():
    import flash_attn  # Check native ABI on the GPU execution host.
else:
    print("CPU setup: CUDA extension checks must run on the GPU host (glibc >= 2.32).")
print(f"Environment ready: torch={torch.__version__}, transformers={transformers.__version__}, flash_attn={importlib.metadata.version('flash-attn')}")
PYTHON
