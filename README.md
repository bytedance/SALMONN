# video-SALMONN: Speech-Enhanced Audio-Visual Large Language Models

<div style='display:flex; gap: 0.25rem; '>
<a href='https://openreview.net/pdf?id=nYsh5GFIqX'><img src='https://img.shields.io/badge/video_SALMONN_paper-PDF-green'></a>
<a href='https://huggingface.co/tsinghua-ee/Video-SALMONN/tree/main'><img src='https://img.shields.io/badge/video--SALMONN-checkpoint-yellow'></a> 
</div>

## Inference

### Preparation
The python version is 3.9.20, and other required packages can be installed with the following command: 
```
pip install -r requirements.txt
```
Create directory to store checkpoints (If modify the structure/rename directories, need to change config files and model files accordingly)
```
mkdir -p ckpt/MultiResQFormer
mkdir -p ckpt/pretrained_ckpt
```
Then download the following model checkpoints:

1. Main video-SALMONN model [checkpoint](https://huggingface.co/tsinghua-ee/Video-SALMONN/tree/main), then put it under `ckpt/MultiResQFormer`
2. InstructBLIP [checkpoint](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna13b_trimmed.pth) for Vicuna-13B model, then put it under `ckpt/pretrained_ckpt`
3. EVA_VIT model [checkpoint](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth) for InstructBLIP, then put it under `ckpt/pretrained_ckpt`
4. BEATs encoder [checkpoint](https://huggingface.co/spaces/fffiloni/SALMONN-7B-gradio/blob/677c0125de736ab92751385e1e8664cd03c2ce0d/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt), then put it under `ckpt/pretrained_ckpt`


### Run inference
```
python inference.py --cfg-path config/test.yaml 
```

### Check the result
The result is saved in the following path:
```
./ckpt/MultiResQFormer/<DateTime>/eval_result.json
```

Expecting the following result:
```
[
    {
        "id": "./dummy/4405327307.mp4_Describe the video and audio in detail",
        "conversation": [
            {
                "from": "human",
                "value": "Describe the video and audio in detail"
            },
            {
                "from": "gpt",
                "value": "None"
            }
        ],
        "task": "audiovisual_video_input",
        "ref_answer": "None",
        "gen_answer": "The video shows a group of musicians performing on stage, with a man singing into a microphone and playing the piano. There is also a drum set and a saxophone on stage. The audience is not visible in the video. The music is upbeat and energetic, and the performers seem to be enjoying themselves.</s>"
    }
]
```

## License & CODE_OF_CONDUCT
Please refer to [salmonn branch](https://huggingface.co/tsinghua-ee/Video-SALMONN/tree/main) for more details.

## ✨ Citation
If you find video-SALMONN useful, please cite the paper:
```
@inproceedings{
  sun2024videosalmonn,
  title={video-{SALMONN}: Speech-Enhanced Audio-Visual Large Language Models},
  author={Guangzhi Sun and Wenyi Yu and Changli Tang and Xianzhao Chen and Tian Tan and Wei Li and Lu Lu and Zejun MA and Yuxuan Wang and Chao Zhang},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=nYsh5GFIqX}
}
```