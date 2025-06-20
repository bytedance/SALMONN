# SALMONN family: A suite of advanced multi-modal LLMs

<div align=center><img src="resource/salmon.png" height="256px" width="256px"/></div>

<h1 align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.herokuapp.com/?lines=Hello,+There!+👋;Welcome+to+SALMONN+family!;&center=true&size=20">
  </a>
</h1>

🚀🚀 Welcome to the repo of **SALMONN**!

The SALMONN model family consists of a series of advanced multi-modal large language models. For more details, please refer to the corresponding branches.

- [SALMONN](https://github.com/bytedance/SALMONN/tree/salmonn)
- [video-SALMONN](https://github.com/bytedance/SALMONN/tree/videosalmonn)
- [SALMONN for speech quality assessment](https://github.com/bytedance/SALMONN/tree/speech_quality_assessment)

## 🔥 News
- [2025-06-01] We have opensourced **QualiSpeech** dataset - A speech quality assessment dataset with natural language reasoning. You can use QualiSpeech to develop your own audio LLM for speech quality assessment or to evaluate the low-level speech perception capabilities of existing audio LLMs. Feel free to download it [here](https://huggingface.co/datasets/tsinghua-ee/QualiSpeech)!
- [2025-03-03] We have released the data processing scripts and finetuned model checkpoints for **SALMONN** for speech quality assessment! See [here](https://github.com/bytedance/SALMONN/tree/speech_quality_assessment)!
- [2024-09-04] We have released the model and inference code for **video-SALMONN**! See [here](https://github.com/bytedance/SALMONN/tree/videosalmonn)!
- [2024-05-28] 🧳 We have released all the annotations (including 600k SQA/AQA data and 50k audio-based storytelling data) for the 3-stage training of SALMONN! Feel free to download them [here](https://drive.google.com/file/d/15cQO--rtMM9JD22y-A5oXXvT3DujgE2e/view?usp=sharing)!
- [2024-04-07] 🤖 We have released all the codes you need to train your own SALMONN! Try some cool things!
- [2024-01-16] 💖 Our paper was accepted by ICLR 2024!
- [2023-11-13] 🎁 We have released a **7B version of SALMONN** at [tsinghua-ee/SALMONN-7B](https://huggingface.co/tsinghua-ee/SALMONN-7B) and built the 7B demo [here](https://huggingface.co/spaces/tsinghua-ee/SALMONN-7B-gradio)!
- [2023-10-08] ✨ We have released [**the model checkpoint**](https://huggingface.co/tsinghua-ee/SALMONN) and **the inference code** for SALMONN-13B!

## 📖 Paper List
```
@inproceedings{wang2024enabling,
  title={Enabling Auditory Large Language Models for Automatic Speech Quality Evaluation},
  author={Wang, Siyin and Yu, Wenyi and Yang, Yudong and Tang, Changli and Li, Yixuan and Zhuang, Jimin and Chen, Xianzhao and Tian, Xiaohai and Zhang, Jun and Sun, Guangzhi and others},
  booktitle={Proc. ICASSP},
  address={Hyderabad},
  year={2025}
}

@inproceedings{wang2024enabling,
  title={QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning and Descriptions},
  author={Wang, Siyin and Yu, Wenyi and Chen, Xianzhao and Tian, Xiaohai and Zhang, Jun and Sun, Guangzhi and others},
  booktitle={Proc. ACL},
  address={Vienna},
  year={2025}
}

@inproceedings{
  sun2024videosalmonn,
  title={video-{SALMONN}: Speech-Enhanced Audio-Visual Large Language Models},
  author={Guangzhi Sun and Wenyi Yu and Changli Tang and Xianzhao Chen and Tian Tan and Wei Li and Lu Lu and Zejun MA and Yuxuan Wang and Chao Zhang},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024},
  url={https://openreview.net/forum?id=nYsh5GFIqX}
}

@inproceedings{
  tang2024salmonn,
  title={SALMONN: Towards Generic Hearing Abilities for Large Language Models},
  author={Changli Tang and Wenyi Yu and Guangzhi Sun and Xianzhao Chen and Tian Tan and Wei Li and Lu Lu and Zejun MA and Chao Zhang},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=14rn7HpKVk}
}
```
