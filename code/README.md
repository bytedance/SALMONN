# ELLSA 2 — Standalone Gesture and Expression Generator

This repository contains the **Standalone generator** of **ELLSA 2: Full-Duplex Interaction with Speech, Expressions and Gestures**. The paper studies coordinated speech, gestures and facial expressions during both speaking and listening. The gesture-expression synthesizer provides the starting point for the integrated dialogue model.

[Installation](#installation) · [Inference](#inference) · [Demo](#interactive-demo) · [Training](#training) · [Evaluation](#evaluation) · [Repository structure](#repository-structure)

## Model overview

The generator conditions on the target person's audio, the partner's audio and a separate motion-style clip. It processes **0.8-second blocks** with 10 audio embeddings per stream and predicts six motion latent steps. Each step combines **18 RVQ indices**: six upper-body, six hand, four facial-expression and two mouth codebooks. Decoders recover 30 fps SMPL-H and FLAME motion.

## Packages and prerequisites

Download the code, data, and checkpoint packages into this layout:

```text
ELLSA2/
├── code/                  # This repository
├── data/                  # Test set and preprocessing
└── checkpoints/           # Generator and RVQ weights
    └── external_assets/   # Local body/face and optional KaoLRM assets
```

Alternatively set `ELLSA2_DATA_ROOT` and `ELLSA2_CHECKPOINT_ROOT` to the respective download directories. All commands below run from `code/`.

The [data guide](../data/README.md) documents training media, selection and preprocessing. The [model guide](../checkpoints/README.md) describes every checkpoint. KaoLRM mono weights are included in the checkpoint package. Obtain SMPL-H, SMPL-X, and FLAME using the [complete asset download and conversion guide](../checkpoints/README.md#external-assets-sources-and-download-guide).

## Installation

Here we provide a conda environment setup for the project.

```bash
conda create -n ellsa2 python=3.12 pip -y
conda activate ellsa2
bash scripts/install_environment.sh
```

The base setup covers training, inference, and evaluation. Install `demo/requirements.txt` for the web demo; KaoLRM uses the separate extraction environment in the [data guide](../data/README.md#extract-raw-facial-labels).

For `--render true` and the web demo, install FFmpeg/ffprobe with libx264 and AAC encoding:

```bash
conda install -c conda-forge 'ffmpeg>=6=gpl_*' --freeze-installed -y
```

Rendering also needs EGL/OpenGL. On Ubuntu, the system libraries can be installed with:

```bash
sudo apt-get install -y libegl1-mesa libgl1-mesa-dri libosmesa6 mesa-utils
```

## Inference

### Included conversation

```bash
python infer.py --model-path ../checkpoints/standalone_generator --test-id fro_00038 --output-dir outputs/example --save-motion-tokens true
```

This generates the target person's body, hands and face from both audio streams. Outputs contain 30 fps SMPL-H poses, 108D FLAME `motioncode` and discrete motion codes.

### Your own audio

Supply aligned 16 kHz mono streams and a 30 fps SMPL-H style NPZ containing at least 612 frames:

```bash
python infer.py --model-path ../checkpoints/standalone_generator \
  --audio-path target.wav --audio-listen-path partner.wav \
  --motion-style-pose-path style.npz --motion-style-start 0 --motion-style-length 612 \
  --output-path outputs/example.npz
```

Add `--render true` to test-ID, direct-audio, or batch-JSON inference. Rendering requires the separately obtained SMPL-X asset, FFmpeg with libx264/AAC and EGL/OpenGL.

```bash
python infer.py --model-path ../checkpoints/standalone_generator \
  --audio-path target.wav --audio-listen-path partner.wav \
  --motion-style-pose-path ../data/testset/style/fro_00038.npz \
  --output-path outputs/custom.npz --render true \
  --smplx-model ../checkpoints/external_assets/smplx/SMPLX_NEUTRAL_2020.npz
```

Release decoding is greedy: `temperature=0`, `top_k=0`, `top_p=1`.

Long audio continues automatically with a **12-second context** (`--max-history-sec 12`, including the current block). When the next block would exceed that limit, the last **three completed audio/motion blocks** (`--reset-history-chunks 3`, 2.4 seconds) are retained, then the new 0.8-second block is appended. `--reset-history-chunks` must be smaller than the context capacity in blocks.

Run `python infer.py --help` for other options. For batch `--audio-json` jobs, an explicit `output_path` in each row determines its filename.

## Interactive demo

```bash
# Install the GPL build to provide libx264 and AAC encoding.
conda install -c conda-forge 'ffmpeg>=6=gpl_*' --freeze-installed -y
python -m pip install -r demo/requirements.txt
python demo/app.py \
  --smplx-model ../checkpoints/external_assets/smplx/SMPLX_NEUTRAL_2020.npz
```

Open **http://127.0.0.1:7860** and upload aligned target/partner audio. More details are shown in the [demo guide](demo/README.md).

## Training

The selected inventory contains **1,318 recordings from 81 speakers**. Download the source data, extract raw KaoLRM facial labels, and create the links in the [data guide](../data/README.md#prepare-the-selected-training-inventory).

`configs/training.json` contains shared settings and two stages. The ASR initializer is already trained:

```bash
conda activate ellsa2
# Stage 1: gesture only, 5,000 updates from the ASR base.
NPROC_PER_NODE=8 bash scripts/train.sh --stage gesture

# Stage 2: joint expression/gesture, 50,000 updates from runs/expression.
NPROC_PER_NODE=8 bash scripts/train.sh --stage gesture_expression
```

The recipe uses BF16 and DeepSpeed ZeRO-2. Adjust `NPROC_PER_NODE`, `--per_device_train_batch_size`, and `--gradient_accumulation_steps` for your GPUs; the published defaults give an effective batch of 128 on eight GPUs. CLI values override the selected stage. If stage 1 uses another output directory, set stage 2's `--model_name_or_path` to it. Existing `checkpoint-*` directories in an output directory resume automatically; choose a new `--output_dir` for a new run.

To fine-tune the released joint model:

```bash
NPROC_PER_NODE=8 bash scripts/train.sh --stage gesture_expression \
  --model_name_or_path ../checkpoints/standalone_generator --output_dir runs/finetune
```

## Evaluation

Generate the full **138-clip benchmark: 67 duplex + 71 semantic clips**:

```bash
python scripts/prepare_testset.py --output-dir outputs/test
python infer.py --model-path ../checkpoints/standalone_generator \
  --audio-json outputs/test/inference_jobs.json --output-dir outputs/test
```

Choose `--metrics {motion,face,mllm,all}`; the default is `motion`. Objective gesture metrics and face metrics run independently:

```bash
python evaluation/evaluate.py --metrics motion \
  --prediction-dir outputs/test --output-dir outputs/metrics_motion \
  --smplx-model ../checkpoints/external_assets/smplx/SMPLX_NEUTRAL_2020.npz

python evaluation/evaluate.py --metrics face \
  --prediction-dir outputs/test --output-dir outputs/metrics_face \
  --flame-asset-dir ../checkpoints/external_assets/flame --device cuda:0
```

`mllm` evaluates gestures using two Gemini and two Seed judgments per clip. Configure separate keys through `--gemini-api-key` / `--seed-api-key` or `GEMINI_API_KEY` / `SEED_API_KEY`; endpoints, model names, and key-variable names have explicit options too. `all` combines motion, face, and MLLM metrics. Gemini and Seed requests run concurrently with separate `--gemini-max-pending` and `--seed-max-pending` limits; `--max-pending` supplies their defaults. Use `--mllm-score-dir` to aggregate existing judge records offline in either MLLM mode. The [evaluation guide](evaluation/README.md#choose-a-metric-group) provides complete commands and explains the grouped output files.

Render all generated motions with the evaluation geometry:

```bash
python evaluation/render.py \
  --jobs-json outputs/test/render_jobs.json \
  --smplh-model-dir ../checkpoints/external_assets --concurrent 2 \
  --zero-lower --zero-global-orient
```

For body, hands, and face in one mesh, use the [SMPL-X rendering mode](evaluation/README.md#whole-person-rendering-with-facial-expression).

## Repository structure

| Path | Purpose |
| --- | --- |
| `modeling_ellsa2.py` | Autoregressive Standalone generator |
| `dataset.py` | Training dataset, validity masks, style conditioning, and token-cache loading |
| `models/vq/` | Upper-body, hand, and expression RVQ-VAE implementations |
| `train.py`, `train_ellsa2.py`, `configs/` | Published recipe and training implementation |
| `infer.py`, `scripts/` | Public inference, shared helpers, and launchers |
| `evaluation/` | Metrics, rendering, and machine/human judge aggregation |
| `demo/` | Interactive standalone audio-to-motion demo |
| `assets/` | Model statistics and paper illustrations |
| `tests/` | Regression tests |

Data preparation lives in `../data/preprocess/`; model weights and local licensed geometry live in `../checkpoints/`.

## Package checks

Verify the packaged files and checksums after downloading:

```bash
python scripts/validate_release.py --package all --checksums
```
