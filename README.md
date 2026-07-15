# video-SALMONN S

## Abstract
Long-duration streaming video understanding is fundamental for future AI agents, yet remains limited by ineffective long-term memory. We introduce video-SALMONN S, a memory-enhanced streaming audio-visual large language model that processes over 3-hour videos at 1 FPS and 360p resolution, outperforming strong non-streaming models under the same memory budget.
In addition to token merging or downsampling, video-SALMONN S is the first to employ test-time training (TTT) as a streaming memory mechanism for video understanding. TTT continuously transforms short-term multimodal representations into long-term memory embedded in model parameters. To improve long-range dependency modeling and memory capacity, we propose (i) a TTT_MEM layer with an additional long-span prediction objective, (ii) a two-stage training scheme, and (iii) a modality-aware memory reader.
We further introduce the **E**pisodic **L**earning from **Vi**deo **M**emory (ELViM) benchmark, simulating agent-like scenarios where models must learn from videos observed hours earlier. video-SALMONN S consistently outperforms both streaming and non-streaming baselines by 3-7% on long video benchmarks. Notably, video-SALMONN S achieves a 15% absolute accuracy improvement over strong non-streaming models on ELViM, demonstrating strong learning abilities from video memory.

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2510.11129'><img src='https://img.shields.io/badge/arXiv-PDF-red'></a>
<a href='https://huggingface.co/datasets/tsinghua-ee/ELViM'><img src='https://img.shields.io/badge/dataset-ELViM-blue'></a> 
<a href='https://huggingface.co/tsinghua-ee/video_SALMONN_S'><img src='https://img.shields.io/badge/checkpoints-videoSALMONNS-orange'></a> 
</div>


## Requirements
```
transformers=4.57.0
torch==2.7.1+cu126
line-profiler
```
Our implementation also depends on Liger-kernel. Please find the official installation of [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) or do `pip install liger-kernel`

## Inference
First, download model checkpoints from <a href='https://huggingface.co/datasets/tsinghua-ee/video_SALMONN_S'><img src='https://img.shields.io/badge/checkpoints-videoSALMONNS-orange'></a>  \
Put the model checkpoints under `models/` directory

The model checkpoints contains one base model and one lora weight with TTT layer:
- The base model is `models/video_SALMONN_S/base`
- The lora and TTT layer is `models/video_SALMONN_S/video_salmonn_s_ttt_lora`

The logic is:
- Load base model with standard video-SALMONN 2 structure
- Then load LoRA and TTT parameters
- This process is all included in `qwenvl/train/train_qwen.py` for details

To run inference directly on your data (replace the example.json with your data following the same formatting):
```
bash test.sh \
    --interval 1 \
    --run_name video_SALMONN_S \
    --dataset data/example.json \
    --max_frames 100000 \
    --model models/video_SALMONN_S/base \
    --model_base models/video_SALMONN_S/base \
    --model_type dense \
    --lora_ckpt models/video_SALMONN_S/video_salmonn_s_ttt_lora \
    --fixed_memory_size 16384 \
    --stepsize 4096 \
    --ttt_type ttt_simsample \
    --ttt_hidden_size 1 \
    --ttt_num_heads 8 \
    --fixed_memory_size_audio 4096 \
    --num_worker 5 \
    --lag_distances 1024 \
    --slot_type forward_usekey_carryover
```
