# video-SALMONN 2+ (Qwen 2.5-VL Based video-SALMONN 2)

video-SALMONN 2+ is built on Qwen 2.5-VL. Based on a better baseline and some other optimizations, video-SALMONN 2+ achieves SOTA on [Video-MME](https://video-mme.github.io/home_page.html) benchmark.

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2506.15220'><img src='https://img.shields.io/badge/paper-PDF-green'></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_7B'><img src='https://img.shields.io/badge/video_SALMONN_2+_7B-checkpoint-yellow'></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_72B'><img src='https://img.shields.io/badge/video_SALMONN_2+_72B-checkpoint-yellow'></a>
</div>

## Results

### Video-MME (w/o sub / w/ sub)

| **7B Model**         | **Short**         | **Medium**        | **Long**      | **Avg**           |
| -------------------- | ----------------- | ----------------- | ------------- | ----------------- |
| LinVT 7B             | 79.0 / 71.7         | 71.6 / 68.7         | 63.2 / 63.3     | 70.3 / 71.7         |
| VideoLLaMA3 7B       | **80.1** / **80.2** | 63.7 / 69.6         | 54.9 / 61.0     | 66.2 / 70.3         |
| Qwen 2.5-VL 7B       | -                 | -                 | -             | 65.1 / 71.6         |
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

## How to Use
1. Prepare the dataset following `scripts/example_av.json`, `scripts/example_v.json`, `scripts/example_dpo.json`, and `scripts/example_a.json`
2. Prepare base audio model through modifying the path in `gen_audio_model.py`
3. To conduct audio alignment, use the following script:
   ```bash
   bash scripts/train.sh --interval 0.1 --run_name audio_alignment --dataset path_to_dataset --lr 2e-5 --train_qformer --max_frames 768 --max_pixels 61250 --model path_to_audio_model --model_base path_to_audio_model --bs 16 --epoch 5 --save_steps 5000
   ```
4. To conduct audio-visual SFT, use the following script:
    ```bash
    bash scripts/train.sh --interval 0.1 --run_name av_sft --dataset path_to_dataset --lr 2e-5 --train_qformer --train_proj --max_frames 768 --max_pixels 61250 --model audio_align_model --model_base path_to_audio_model --epoch 5 --save_steps 2000 --use_lora --lora_r 128 --lora_alpha 256
    ```
5. To conduct DPO, use the following script:
    ```bash
    bash scripts/train.sh --interval 0.1 --run_name dpo --dataset path_to_dataset --max_frames 768 --max_pixels 61250 --model audio_visual_base --model_base audio_align_model --lora_ckpt audio_visual_checkpoint --train_type gdpo --use_lora --lora_r 128 --lora_alpha 256 --lr 5e-6 --epoch 1 --save_steps 200 --train_qformer --train_proj
    ```
6. To evaluate 7B model, use the following script:
   ```bash
   bash scripts/test.sh --interval 0.1 --run_name eval --dataset path_to_dataset --max_frames 768 --max_pixels 61250 --model path_to_audio_model --model_base path_to_audio_model --lora_ckpt model_ckpt
   ```
7. To evaluate 72B model, use the following script:
   ```bash
   bash scripts/test_8.sh --interval 0.1 --run_name eval --dataset path_to_dataset --max_frames 768 --max_pixels 61250 --model path_to_audio_model --model_base path_to_audio_model --lora_ckpt model_ckpt
   ```
