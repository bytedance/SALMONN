# ELLSA 2 — Evaluation Pipeline

This package provides the evaluation pipeline used in *ELLSA 2: Full-Duplex Interaction with Speech, Expressions and Gestures*. It evaluates dialogue-conditioned gestures and facial expressions. The benchmark contains **138 human-verified, 20-second clips: 67 duplex and 71 semantic**. Each has aligned audio, ground-truth motion, a separate nonoverlapping 612-frame style reference, facial labels, reference video and frozen evaluation context. See the [data guide](../../data/README.md) for the packaged files.

All commands below run from `code/`.

## Generate predictions

```bash
python scripts/prepare_testset.py --output-dir outputs/test
python infer.py --model-path ../checkpoints/standalone_generator \
  --audio-json outputs/test/inference_jobs.json --output-dir outputs/test
```

## Choose a metric group

Use `--metrics {motion,face,mllm,all}`. The default is `motion`.

| Group | Metrics | Required inputs |
| --- | --- | --- |
| `motion` | Speaking/listening activity and prosodic alignment | Prediction NPZs and SMPL-X skeleton asset |
| `face` | Speaking/listening facial geometry and dynamics: LVE, MHD, FDD, JDD | Prediction NPZs and FLAME asset |
| `mllm` | Duplex/semantic gesture scores from two Gemini and two Seed judgments per clip | Rendered videos and API configuration, or existing judge records |
| `all` | All three groups above | The union of their inputs |

```bash
python evaluation/evaluate.py --metrics motion \
  --prediction-dir outputs/test --output-dir outputs/metrics_motion \
  --smplx-model ../checkpoints/external_assets/smplx/SMPLX_NEUTRAL_2020.npz

python evaluation/evaluate.py --metrics face \
  --prediction-dir outputs/test --output-dir outputs/metrics_face \
  --flame-asset-dir ../checkpoints/external_assets/flame --device cuda:0
```

For online MLLM scoring, first render the predictions, 

```bash
python evaluation/render.py \
  --jobs-json outputs/test/render_jobs.json \
  --smplh-model-dir ../checkpoints/external_assets --concurrent 2 \
  --zero-lower --zero-global-orient
```

then configure both providers. Replace the endpoint and Seed model placeholders with your service settings:

```bash
export GEMINI_API_KEY='your-gemini-key'
export SEED_API_KEY='your-seed-key'
export GEMINI_BASE_URL='https://your-gemini-endpoint/v1'
export SEED_BASE_URL='https://your-seed-endpoint/v1'

python evaluation/evaluate.py --metrics mllm \
  --video-dir outputs/test/videos --output-dir outputs/metrics_mllm \
  --gemini-model gemini-2.5-pro --gemini-base-url "$GEMINI_BASE_URL" \
  --gemini-api-key "$GEMINI_API_KEY" \
  --seed-model seed-2.0-lite --seed-base-url "$SEED_BASE_URL" \
  --seed-api-key "$SEED_API_KEY" \
  --gemini-max-pending 2 --seed-max-pending 4
```

Run all metrics together with the same provider configuration:

```bash
python evaluation/evaluate.py --metrics all \
  --prediction-dir outputs/test --video-dir outputs/test/videos \
  --output-dir outputs/metrics_all \
  --smplx-model ../checkpoints/external_assets/smplx/SMPLX_NEUTRAL_2020.npz \
  --flame-asset-dir ../checkpoints/external_assets/flame --device cuda:0 \
  --gemini-model gemini-2.5-pro --gemini-base-url "$GEMINI_BASE_URL" \
  --gemini-api-key "$GEMINI_API_KEY" \
  --seed-model seed-2.0-lite --seed-base-url "$SEED_BASE_URL" \
  --seed-api-key "$SEED_API_KEY" \
  --gemini-max-pending 2 --seed-max-pending 4
```

Aggregate existing four-judge records without API requests:

```bash
python evaluation/evaluate.py --metrics mllm \
  --mllm-score-dir outputs/metrics_mllm/mllm/judges \
  --output-dir outputs/metrics_mllm_offline
```

Objective metrics use the central 4–16 seconds. MLLM scoring uses the frozen per-clip context in `data/testset/judge_context.json`, including the original candidate focus intervals. Online scores may vary with provider/model versions. Each metric group writes its summary and per-clip results under the chosen output directory.

## Render predicted motion

The preparation command above creates `outputs/test/render_jobs.json`. Render the complete prediction set with SMPL-H:

```bash
python evaluation/render.py \
  --jobs-json outputs/test/render_jobs.json \
  --smplh-model-dir ../checkpoints/external_assets --concurrent 2 \
  --zero-lower --zero-global-orient
```

For one generated file:

```bash
python evaluation/render.py \
  --npz-path outputs/test/fro_00038.npz \
  --wav-path ../data/testset/audio/fro_00038_target.wav \
  --output-path outputs/test/fro_00038.mp4 \
  --smplh-model-dir ../checkpoints/external_assets --concurrent 2 \
  --zero-lower --zero-global-orient
```

## Whole-person rendering with facial expression

To generate and visualize in one command, add `--render true` to `python infer.py`. This renders each prediction with the same SMPL-X expression mapping and writes an MP4 beside its NPZ. See the [inference options](../README.md#inference). The commands below render predictions that already exist.

Use `--smplx-model` to animate body, hands, and face in one SMPL-X mesh:

```bash
python evaluation/render.py \
  --npz-path outputs/test/fro_00038.npz \
  --wav-path ../data/testset/audio/fro_00038_target.wav \
  --output-path outputs/test/fro_00038_expression.mp4 \
  --smplx-model ../checkpoints/external_assets/smplx/SMPLX_NEUTRAL_2020.npz \
  --device cuda:0 --concurrent 2 --zero-lower --zero-global-orient
```