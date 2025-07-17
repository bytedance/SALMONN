# video-SALMONN 2: Captioning-Enhanced Audio-Visual Large Language Models

🚀🚀 Welcome to the repo of **video-SALMONN 2**!

video-SALMONN 2 is a powerful audio-visual large language model (LLM) that **generates high-quality audio-visual video captions**, which is developed by the Department of Electronic Engineering at Tsinghua University and ByteDance. 

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2506.15220'><img src='https://img.shields.io/badge/video_SALMONN_2_paper-PDF-green'></a>
<a href='https://video-salmonn-2.github.io'><img src='https://img.shields.io/badge/demo-green'></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN-2'><img src='https://img.shields.io/badge/video_SALMONN_2_7B-checkpoint-yellow'></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_7B'><img src='https://img.shields.io/badge/video_SALMONN_2+_7B-checkpoint-yellow'></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_72B'><img src='https://img.shields.io/badge/video_SALMONN_2+_72B-checkpoint-yellow'></a>
</div>

## 🔥 News

- **2025-07-17**: We release the code and checkpoint of video-SALMONN 2+ at [video-SALMONN 2+](https://github.com/bytedance/video-SALMONN-2/tree/main/video_SALMONN2_plus). video-SALMONN 2+ achieves SOTA results on [Video-MME](https://video-mme.github.io/home_page.html) benchmark.
- **2025-07-08**: We release the 7B version of video-SALMONN 2.
- **2025-06-18**: We release the code of video-SALMONN 2.

## Results

### Video-MME (w/o sub / w/ sub)

| **7B Model**         | **Short**         | **Medium**        | **Long**      | **Avg**           |
| -------------------- | ----------------- | ----------------- | ------------- | ----------------- |
| LinVT 7B             | 79.0 / 71.7         | 71.6 / 68.7         | **63.2** / 63.3     | 70.3 / 71.7         |
| VideoLLaMA3 7B       | **80.1** / **80.2** | 63.7 / 69.6         | 54.9 / 61.0     | 66.2 / 70.3         |
| Qwen 2.5-VL 7B       | -                 | -                 | -             | 65.1 / 71.6         |
| video-SALMONN 2 7B   | 79.8 / -            | 65.0 / -            | 57.3 / -        | 67.4 / -            |
| video-SALMONN 2+ 7B  | 79.0 / 79.4         | **72.1** / **73.1** | 62.3 / **63.9** | **71.1** / **72.1** |
| **Larger Model**     |                   |                   |               |                   |
| GPT-4o               | 80.0 / 82.8         | 70.3 / 76.6         | 65.3 / 72.1     | 71.9 / 77.2         |
| Gemini-1.5-pro       | 81.7 / 84.5         | 74.3 / **81.0**     | 67.4 / **77.4** | 75.0 / **81.3**     |
| Qwen 2.5-VL 72B      | -                 | -                 | -             | 73.3 / 79.1         |
| video-SALMONN 2+ 72B | **84.3** / **85.1** | **79.4** / 79.7     | **71.2** / 72.0 | **78.3** / 78.9     |

### Other Audio-Visual Video Benchmarks

| **Model**            | **MLVU** | **LongVideoBench** | **DailyOmni** | **VideoHolmes** |
| -------------------- | -------- | ------------------ | ------------- | --------------- |
| GPT-4o               | 64.6     | 66.7               | 56.47         | 42.0            |
| Gemini-1.5-pro       | -        | 64.0               | -             | 41.2            |
| Qwen 2.5-VL 72B      | 75.1     | **67.4**           | 61.82         | 50.2            |
| video-SALMONN 2+ 72B | **77.8** | 66.4               | **69.84**     | **55.6**        |

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
@article{tang2025video,
    title={{video-SALMONN 2: Captioning-Enhanced Audio-Visual Large Language Models}}, 
    author={Changli Tang and Yixuan Li and Yudong Yang and Jimin Zhuang and Guangzhi Sun and Wei Li and Zejun Ma and Chao Zhang},
    journal={arXiv preprint arXiv:2506.15220},
    year={2025},
}
```
