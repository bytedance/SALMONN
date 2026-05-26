# OmniMem

Audio-visual large language models (LLMs) hold strong promise for long-form video understanding, yet their long-video inference is fundamentally limited by the linear growth of video tokens and key-value (KV) caches. We present OmniMem, a memory-efficient streaming framework designed specifically for audio-visual LLMs.
Unlike existing compression methods that treat all tokens uniformly, OmniMem introduces a modality-aware memory allocation strategy that separately manages visual and audio contexts, addressing the severe token imbalance between the two modalities. OmniMem further preserves informative and non-redundant KV states through perturbation-aware memory selection, enabling compact memory without sacrificing long-range understanding. To strengthen compression under realistic deployment constraints, we also explore budget-aware fine-tuning, which encourages the model to consolidate useful information into retained memory. Experiments on VideoMME Long, LVBench, and LVOmniBench with video-SALMONN 2+ and Qwen-2.5-Omni show that OmniMem consistently improves over strong training-free compression baselines by 2–4\% absolute accuracy under the same memory budgets, with an additional 1–2\% gain after fine-tuning.

## Requirements
```
transformers=4.57.0
torch==2.7.1+cu126
line-profiler
```
Our implementation also depends on Liger-kernel. Please find the official installation of [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) or do `pip install liger-kernel`

## Preparation
- Setup your data in the format shown in example.json
- Download model checkpoints

## Inference with OmniMem

To run test with the best setting in the paper (lambda=0.02, T=0.2)

```
bash test.sh \
    --interval 1 \
    --run_name <your_own_run_name> \
    --dataset <path_to_dataset> \
    --max_frames 100000 \
    --model path_to_vs2_checkpoint \
    --model_base path_to_model_base \
    --model_type dense \
    --fixed_memory_size 20480 \
    --workingmemsize 20480 \
    --memgroupsize 16384 \
    --model_type dense \
    --search_type sim_modmask \
    --num_worker 5 \
    --total_timestamps 256 \
    --shrink_config scripts/per_layer_budget_norm_audiovisual_temperature_0.2_audiolarge_8k_avratio5.json \
    --div_factor 0.02
```
