# video-SALMONN S

## Abstract
Long-duration streaming video understanding is fundamental for future AI agents, yet remains limited by ineffective long-term memory. We introduce video-SALMONN S, a memory-enhanced streaming audio-visual large language model that processes over 3-hour videos at 1 FPS and 360p resolution, outperforming strong non-streaming models under the same memory budget.
In addition to token merging or downsampling, video-SALMONN S is the first to employ test-time training (TTT) as a streaming memory mechanism for video understanding. TTT continuously transforms short-term multimodal representations into long-term memory embedded in model parameters. To improve long-range dependency modeling and memory capacity, we propose (i) a TTT_MEM layer with an additional long-span prediction objective, (ii) a two-stage training scheme, and (iii) a modality-aware memory reader.
We further introduce the **E**pisodic **L**earning from **Vi**deo **M**emory (ELViM) benchmark, simulating agent-like scenarios where models must learn from videos observed hours earlier. video-SALMONN S consistently outperforms both streaming and non-streaming baselines by 3-7% on long video benchmarks. Notably, video-SALMONN S achieves a 15% absolute accuracy improvement over strong non-streaming models on ELViM, demonstrating strong learning abilities from video memory.

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2510.11129'><img src='https://img.shields.io/badge/arXiv-PDF-red'></a>
<a href='https://huggingface.co/datasets/tsinghua-ee/ELViM'><img src='https://img.shields.io/badge/dataset-ELViM-blue'></a> 
</div>


## Inference
Coming Soon
