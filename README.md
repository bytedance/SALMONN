# SALMONN: Speech Audio Language Music Open Neural Network
<div align=center><img src="resource/salmon.png" height="256px" width="256px"/></div>

Welcome to the repo of **SALMONN**!

SALMONN is the first large language model with universal auditory ability, created in collaboration with the Department of the Electronic Engineering of Tsinghua University and ByteDance. Compared with other auditory models that only support speech input or non-speech audio input, SALMONN has the ability to perceive and understand **all kinds of audio inputs** such as speech, audio events, music, etc., which gives the LLM "ears", so as to emerge advanced capabilities such as multi-language and cross-modal reasoning.

<div style='display:flex; gap: 0.25rem; '>
<a href='https://cf8f64dc98536f6102.gradio.live/'><img src='https://img.shields.io/badge/gradio-Demo-blue'></a>
<a href=''><img src='https://img.shields.io/badge/paper-PDF-green'></a>
</div>

## Demos

Compared with traditional speech and audio processing tasks such as speech recognition and audio caption, SALMONN leverages the general knowledge and cognitive abilities of the LLM to achieve a cognitively oriented audio perception, which dramatically improves the versatility of the model and the richness of the task. In addition, SALMONN is able to follow textual commands, and even spoken commands, with a relatively high degree of accuracy. Since SALMONN only uses training data based on textual commands, listening to spoken commands is also a cross-modal emergent ability.

Here are some demos of SALMONN.

| Audio                                                        | Response                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [asr.wav](./resource/audio_demo/asr.wav)                     | ![asr](./resource/response_demo/asr.png)                     |
| [audiocaption.wav](./resource/audio_demo/audiocaption.wav)   | ![audiocaption](./resource/response_demo/audiocaption.png)   |
| [music.wav](./resource/audio_demo/music.wav)                 | ![music](./resource/response_demo/music.png)                 |
| [emotion.wav](./resource/audio_demo/emotion.wav)             | ![emotion](./resource/response_demo/emotion.png)             |
| [asr_en2de.wav](./resource/audio_demo/asr_en2de.wav)         | ![asr_en2de](./resource/response_demo/asr_en2de.png)         |
| [keywords.flac](./resource/audio_demo/keywords.flac)         | ![keywords](./resource/response_demo/keywords.png)           |
| [spoken_query.wav](./resource/audio_demo/spoken_query.wav)   | ![spoken_query](./resource/response_demo/spoken_query.png)   |
| [audio_story_telling.wav](./resource/audio_demo/audio_story_telling.wav) | ![audio_story_telling](./resource/response_demo/audio_story_telling.png) |
| [spoken_audio_query.wav](./resource/audio_demo/spoken_audio_query.wav) | ![spoken_audio_query](./resource/response_demo/spoken_audio_query.png) |



## Team

**Team Tsinghua**: Wenyi Yu, Changli Tang, Guangzhi Sun, Chao Zhang

**Team ByteDance**: Xianzhao Chen, Wei Li, Tian Tan, Lu Lu, Zejun Ma
