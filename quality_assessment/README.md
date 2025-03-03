# SALMONN for speech quality assessment

## Data preperation

Please download the following datasets first.

Speech quality assessment datasets:

[BVCC](https://zenodo.org/records/10691660): How do Voices from Past Speech Synthesis Challenges Compare Today  
[NISQA](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus): A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets  
[SOMOS](https://zenodo.org/records/7378801): The Samsung Open MOS Dataset for the Evaluation of Neural Text-to-Speech Synthesis
Creators  
(Note: you need to resample audios to 16kHz since the input sampling rate of SALMONN is 16kHz)

Speaker similarity dataset:

[VoxSim](https://mm.kaist.ac.kr/projects/voxsim/): A perceptual voice similarity dataset

Then run data processing scripts to generate annotations, you can also refer to our [annotations](https://huggingface.co/tsinghua-ee/Speech_Quality_Assessment).

## Finetuned SALMONN-7B checkpoint

Our finetuned SALMONN-7B checkpoint can also be downloaded [here](https://huggingface.co/tsinghua-ee/Speech_Quality_Assessment).

## Run and inference

Just follow the repo of SALMONN.

## Citation
```
@inproceedings{wang2024enabling,
  title={Enabling Auditory Large Language Models for Automatic Speech Quality Evaluation},
  author={Wang, Siyin and Yu, Wenyi and Yang, Yudong and Tang, Changli and Li, Yixuan and Zhuang, Jimin and Chen, Xianzhao and Tian, Xiaohai and Zhang, Jun and Sun, Guangzhi and others},
  booktitle={Proc. ICASSP},
  address={Hyderabad},
  year={2025}
}
```
