# video-SALMONN 2: Captioning-Enhanced Audio-Visual Large Language Models

🚀🚀 Welcome to the repo of **video-SALMONN 2**!

video-SALMONN 2 is a powerful audio-visual large language model (LLM) that **generates high-quality audio-visual video captions**, which is developed by the Department of Electronic Engineering at Tsinghua University and ByteDance. 

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2410.06682'><img src='https://img.shields.io/badge/video_SALMONN_2_paper-PDF-green'></a>
<a href='https://video-salmonn-2.github.io'><img src='https://img.shields.io/badge/demo-green'></a>
</div>

## 🔥 News

- **2025-06-18**: We release the code of video-SALMONN 2.

## ⚡️ Future Plans

- ~~Release the code.~~
- Release the visual base model and audio weight. 
- Release final video-SALMONN 2.

## 🌈 How to Use

### How to train a model

1. Prepare the dataset following `scripts/example_sft.json` and `scripts/example_dpo.json`.
2. Download LLaVA-OneVision Model from [huggingface](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov).
3. Modify the parameters in `scripts/train_sft.sh` and `scripts/train_dpo.sh`.
4. Run `bash scripts/train_sft.sh` or `bash scripts/train_dpo.sh`.

### How to evaluate a checkpoint

1. Prepare the dataset following `scripts/example_sft.json`.
2. Modify the parameters in `scripts/eval.sh`.
3. Run `bash scripts/eval.sh`.

## 👀 Team

**Team Tsinghua**: Changli Tang, Yixuan Li, Yudong Yang, Jimin Zhuang, Guangzhi Sun, Chao Zhang

**Team ByteDance**: Wei Li, Zejun Ma

## ✨ Citation
If you find video-SALMONN 2 useful, please cite the paper:

```
@article{tang2024enhancing,
  title={Enhancing Multimodal LLM for Detailed and Accurate Video Captioning using Multi-Round Preference Optimization},
  author={Tang, Changli and Li, Yixuan and Yang, Yudong and Zhuang, Jimin and Sun, Guangzhi and Li, Wei and Ma, Zujun and Zhang, Chao},
  journal={arXiv preprint arXiv:2410.06682},
  year={2024}
}
```
