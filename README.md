# SALMONN: Speech Audio Language Music Open Neural Network

<div align=center><img src="resource/salmon.png" height="256px" width="256px"/></div>

🚀🚀 Welcome to the repo of **SALMONN**!

SALMONN is a large language model (LLM) enabling **speech, audio event, and music inputs**, which is developed by the Department of Electronic Engineering at Tsinghua University and ByteDance. Instead of speech-only input or audio-event-only input, SALMONN can perceive and understand all kinds of audio inputs and therefore obtains emerging capabilities such as multilingual speech recognition & translation and audio-speech co-reasoning. This can be regarded as giving the LLM "ears" and cognitive hearing abilities, which makes SALMONN a step towards hearing-enabled artificial general intelligence.

<div style='display:flex; gap: 0.25rem; '>
<a href='https://bytedance.github.io/SALMONN/'><img src='https://img.shields.io/badge/gradio-Demo-blue'></a>
<a href='https://arxiv.org/pdf/2310.13289.pdf'><img src='https://img.shields.io/badge/paper-PDF-green'></a>
<a href='https://huggingface.co/MSIIP/SALMONN'><img src='https://img.shields.io/badge/huggingface-checkpoint-yellow'></a> 
</div>

## 🔥 News

- [10-08] ✨ We release **the model checkpoint** and **the inference code** for SALMONN!

## 🌟 Structure

The model architecture of SALMONN is shown below. A window-level Q-Former is used as the connection module to fuse the outputs from a Whisper speech encoder and a BEATs audio encoder as augmented audio tokens, which are aligned with the LLM input space. The LoRA adaptor aligns the augmented LLM input space with its output space. The text prompt is used to instruct SALMONN to answer open-ended questions about the general audio inputs and the answers are in the LLM text responses. 

<div align=center><img src="resource/structure.png" height="100%" width="75%"/></div>

## ⚡️ Demos

Compared with traditional speech and audio processing tasks such as speech recognition and audio caption, SALMONN leverages the general knowledge and cognitive abilities of the LLM to achieve a cognitively oriented audio perception, which dramatically improves the versatility of the model and the richness of the task. In addition, SALMONN is able to follow textual commands, and even spoken commands, with a relatively high degree of accuracy. Since SALMONN only uses training data based on textual commands, listening to spoken commands is also a cross-modal emergent ability.

Here are some examples of SALMONN.

| Audio                                                  | Response                                     |
| ------------------------------------------------------ | -------------------------------------------- |
| [gunshots.wav](./resource/audio_demo/gunshots.wav)     | ![sac](resource/response_demo/sac.png)       |
| [duck.wav](./resource/audio_demo/duck.wav)             | ![story](resource/response_demo/story.png)   |
| [music.wav](./resource/audio_demo/music.wav)           | ![mc](resource/response_demo/mc.png)         |


## 🌈 How to inference in CLI

1. Our environment: python3 version is 3.9.17 and ```pip3 install soundfile librosa torch==2.0.1 transformers==4.28.0 peft==0.3.0```.
2. Download [whisper large v2](https://huggingface.co/openai/whisper-large-v2/tree/main) to ```whisper_path```.
3. Download [Fine-tuned BEATs_iter3+ (AS2M) (cpt2)](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) to `beats_path`.
4. Download [vicuna 13B v1.1](https://huggingface.co/lmsys/vicuna-13b-v1.1/tree/main) to ```vicuna_path```.
5. Download [salmonn v1](https://huggingface.co/MSIIP/SALMONN/blob/main/salmonn_v1.pth) to ```ckpt_path```.
6. Running with ```python3 cli_inference.py --ckpt_path xxx --whisper_path xxx --beats_path xxx --vicuna_path xxx``` in A100-SXM-80GB. Now you can input ```wav_path``` and ```prompt```. Enjoy yourself !

## 🌈 How to launch a web demo

1. Same as **How to inference in CLI: 1-5**.
2. Running with ```python3 web_demo.py --ckpt_path xxx --whisper_path xxx --beats_path xxx --vicuna_path xxx``` in A100-SXM-80GB.

## 👀 Team

**Team Tsinghua**: Wenyi Yu, Changli Tang, Guangzhi Sun, Chao Zhang

**Team ByteDance**: Xianzhao Chen, Wei Li, Tian Tan, Lu Lu, Zejun Ma

## ✨ Citation
If you find SALMONN useful, please cite our paper:
```
@article{tang2023salmonn,
      title={{SALMONN}: Towards Generic Hearing Abilities for Large Language Models}, 
      author={Changli, Tang and Wenyi, Yu and Guangzhi, Sun and Xianzhao, Chen and Tian, Tan and Wei, Li and Lu, Lu and Zejun, Ma and Chao, Zhang},
      journal={arXiv:2310.13289},
      year={2023}
}
```
