#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import os
import json
import csv
from tqdm import tqdm
import random
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence
from fractions import Fraction
import soundfile as sf

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler, UniformClipSampler
from pytorchvideo.data.encoded_video import EncodedVideo
from transformers import WhisperFeatureExtractor


AUDIO_EXISTANCE = ["Is there any sound?", "Can you hear anything?", "Is there audio with this video?"]
AUDIO_VIDEO_MATCHING = [
    "Is the audio compatible with the video?",
    "Does the audio come from the same source as the video?",
    "Is the audio related to the video?"
]
video_specaug_params = {
    "mask_rate": 0.0,
}

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, image_root_path: str):
        super(SupervisedDataset, self).__init__()

        with open(data_path, 'r') as f:
            json_data = json.load(f)

        self.image_path_list, self.caption_list = [], []
        for item in json_data:
            one_image_name, one_caption = item["image_name"], item["conversation"]
            # TODO: stage 2 dataset format is invalid
            if not one_image_name.endswith('.jpg'):
                one_image_name += '.jpg'
            one_image_path = image_root_path + '/{}'.format(one_image_name)
            self.image_path_list.append(one_image_path)
            self.caption_list.append(one_caption)
        print(f'[!] collect {len(self.image_path_list)} samples for training')

    def __len__(self): # number of instances
        return len(self.image_path_list)

    #def __getitem__(self, i) -> Dict[str, torch.Tensor]: # how to get item, 取一个样本
    def __getitem__(self, i):
        print(i)
        return dict(image_paths=self.image_path_list[i], output_texts=self.caption_list[i])

    def collate(self, instances):
        image_paths, output_texts = tuple([instance[key] for instance in instances] for key in ("image_paths", "output_texts"))
        return dict(
            image_paths=image_paths,
            output_texts=output_texts
        )


class SupervisedAudioVisualDataset(Dataset):
    """Dataset for supervised fine-tuning with audio captioning."""

    def __init__(self,
        data_type: str,
        audio_data_path: str = "",
        audio_root_path: str = "",
        video_data_path: str = "",
        image_data_path: str = "",
        video_root_path: str = "",
        image_root_path: str = "",
        sample_rate: int = 16000,
        sample_per_clip: int = 2,
        clip_duration: int = 1,
        use_whisper: str = "",
        use_blip: str = "",
        training: bool = True,
        # [Yu]
        sin_pos: bool = False,
        return_raw: bool = False,
        audio_only: bool = False,
        video_only: bool = False,
        use_nemo: bool = False,
        # [npy]
        use_npy: bool = False
    ):
        super(SupervisedAudioVisualDataset, self).__init__()
        if audio_data_path == "" and video_data_path == "" and image_data_path == "":
            raise
        self.modality_range = []
        self.audiofiles = []
        self.spokencocofiles = []
        self.training = training
        # [Yu]
        self.sin_pos = sin_pos
        self.return_raw = return_raw
        self.audio_path_list, self.audio_caption_list = [], []
        self.audio_only = audio_only
        self.video_only = video_only
        self.use_nemo = use_nemo
        # [npy]
        self.use_npy = use_npy
        if audio_data_path != "" and "audio" in data_type and audio_data_path is not None:
            self.audio_path_list, self.audio_caption_list = self.get_data_json(
                audio_data_path, audio_root_path, modality="audio",
            )
            self.modality_range.append("audio")
        self.image_path_list, self.image_caption_list = [], []
        if image_data_path != "" and "image" in data_type and image_data_path is not None:
            self.image_path_list, self.image_caption_list = self.get_data_json(
                image_data_path, image_root_path, modality="image",
            )
            self.modality_range.append("image")
        self.video_path_list, self.video_caption_list = [], []
        if video_data_path != "" and "video" in data_type and video_data_path is not None:
            self.video_path_list, self.video_caption_list = self.get_data_json(
                video_data_path, video_root_path, modality="video",
            )
            if data_type != "audiovideoimage":
                self.modality_range.append("video")
            self.frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=sample_per_clip)
            self.clip_sampler = UniformClipSampler(
                clip_duration=clip_duration, backpad_last=True
            )
            self.sample_per_clip = sample_per_clip
            self.clip_duration = clip_duration
        self.use_whisper = use_whisper
        self.use_blip = use_blip
        self.sample_rate = sample_rate
        self.data_type = data_type
        if self.data_type == "audiovideoimage" and self.training:
            if audio_only:
                self.modality_range = ["audiovideoimage", "audioimage", "audio"]
            elif video_only:
                self.modality_range = ["audiovideoimage"]
            else:
                self.modality_range = ["audiovideoimage", "audioimage"]
            print(self.modality_range)
        elif self.data_type == "audiovideoimage":
            self.modality_range = ["audiovideoimage"]
        self.modality = random.choice(self.modality_range)
        if self.use_whisper == "true":
            # whispermodel = "/mnt/bn/audio-visual-llm-data/yuwenyi/ckpt/whisper/whisper-large-v3"
            whispermodel = "/mnt/bn/audio-visual-llm-data/yuwenyi/ckpt/whisper/whisper_large_v2"
            self.transform = WhisperFeatureExtractor.from_pretrained(whispermodel)
            self.use_whisper = True

    def get_data_json(self, data_path, root_path, modality='image'):
        with open(data_path, 'r') as f:
            json_data = json.load(f)
            # if not self.training:
            #     json_data = json_data[:2000]
            if self.video_only and not self.training:
                json_data = json_data[:100]
            elif self.video_only and self.training:
                json_data = json_data[:1000]

        path_list, caption_list = [], []
        for item in json_data:
            one_image_name, one_caption = item["image_name"], item["conversation"]
            if isinstance(one_image_name, list) and "SpokenCOCO" in one_image_name[1]:
                self.spokencocofiles.append(one_image_name[1])
            elif "audiocaps" in one_image_name:
                self.audiofiles.append(one_image_name)
            if modality in ["image", "video", "audio"]:
                one_path = one_image_name
            else:
                one_path = root_path + '/{}'.format(one_image_name)
            # if modality == "image" or os.path.exists(one_path):
            path_list.append(one_path)
            caption_list.append(one_caption)
        print(f'[!] collect {len(path_list)} {modality} samples for {"train" if self.training else "valid"}')
        return path_list, caption_list

    def __len__(self): # number of instances
        return len(self.audio_path_list) + len(self.image_path_list) + len(self.video_path_list)

    def get_audio(self, i, audiopath=None):
        i = i % max(len(self.audio_path_list), 1)
        if audiopath is None:
            audiopath = self.audio_path_list[i]
        if self.use_whisper:
            audio, _ = sf.read(audiopath)
            if len(audio.shape) == 2:
                audio = audio[:, 0]
            if audio.shape[0] < 3 * self.sample_rate:
                audio = np.concatenate((audio, np.zeros((3 * self.sample_rate - audio.shape[0]), dtype=float)), axis=0)
            if len(audio) > 30 * self.sample_rate and self.sin_pos:
                audio_list = [audio[i: i + 30 * self.sample_rate] for i in range(0, len(audio), 30 * self.sample_rate)]
                spectrogram_list = []
                for audio_piece in audio_list:
                    spectrogram_piece = self.transform(
                        audio_piece,
                        sampling_rate=self.sample_rate,
                        return_tensors="pt",
                        max_length=30 * self.sample_rate,
                    )
                    spectrogram_list.append(spectrogram_piece["input_features"].squeeze())
                spectrogram = torch.stack(spectrogram_list, dim=0)
                return dict(
                    image_paths=spectrogram,
                    output_texts=copy.deepcopy(self.audio_caption_list[i]) if self.audio_caption_list != [] else None,
                    modality="audio",
                    orig_paths=audiopath,
                    raw_audio=audio_list if self.return_raw else None,
                )
            else:
                spectrogram = self.transform(
                    audio,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    max_length=30 * self.sample_rate,
                )
                spectrogram = spectrogram["input_features"].squeeze()
                return dict(
                    image_paths=spectrogram,
                    output_texts=copy.deepcopy(self.audio_caption_list[i]) if self.audio_caption_list != [] else None,
                    modality="audio",
                    orig_paths=audiopath,
                    raw_audio=[audio[:30 * self.sample_rate]] if self.return_raw else None,
                )
        elif self.use_nemo:
            pass
        else:
            return dict(
                image_paths=audiopath,
                output_texts=copy.deepcopy(self.audio_caption_list[i]) if self.audio_caption_list != [] else None,
                modality="audio",
            )

    def get_image(self, i):
        if i >= len(self.image_path_list):
            i = i % len(self.image_path_list)
        if not self.training and isinstance(self.image_path_list[i], list):
            imagepath = self.image_path_list[i][0]
        else:
            imagepath = self.image_path_list[i]
        return dict(image_paths=imagepath, output_texts=copy.deepcopy(self.image_caption_list[i]), modality="image")

    def get_video(self, i, videopath=None):
        if videopath is None:
            if i >= len(self.video_path_list):
                i = i % len(self.video_path_list)
            videopath = self.video_path_list[i]
        if isinstance(videopath, list):
            videopath = videopath[0]

        if self.training:
            if self.use_npy: # npy training
                return dict(image_paths=videopath, output_texts=copy.deepcopy(self.video_caption_list[i]), modality="video")
        video = EncodedVideo.from_path(
            videopath,
            decoder="decord",
            decode_audio=False,
            **{"sample_rate": self.sample_rate},
        )
        if "egovideos" in videopath or "how2videos" in videopath:
            durations = videopath[:-4].split("_")[-2:]
            if durations[-1] == "sum":
                duration = 30
            else:
                duration = float(durations[1]) - float(durations[0])
        else:
            duration = video.duration
        
        all_clips_timepoints = self.get_clip_timepoints(
            self.clip_sampler, duration)
        all_video = []
        for clip_timepoints in all_clips_timepoints:
            # Read the clip, get frames
            try:
                clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
                video_clip = self.frame_sampler(clip["video"])
                if "mask_rate" in video_specaug_params and random.random() < video_specaug_params["mask_rate"]:
                    video_clip = video_clip * 0  # mask specific video frame
                video_clip = video_clip / 255.0  # since this is float, need 0-1
                all_video.append(video_clip)
            except:
                print("skipped frame {}".format(clip_timepoints))
                print(videopath)
                pass
        return dict(image_paths=all_video, output_texts=copy.deepcopy(self.video_caption_list[i]), modality="video")

    def get_audioimage(self, i):
        image_data = self.get_image(i)
        if isinstance(image_data["image_paths"], list):
            audiopath = image_data["image_paths"][1]
            image_data["image_paths"] = image_data["image_paths"][0]
            prompt = image_data["output_texts"]
            avmask = 1
            if image_data["output_texts"][1]["value"] == "audio_text_matching":
                if random.random() > 0.5:
                    audio_data = self.get_audio(i, random.choice(self.spokencocofiles))
                    prompt[1]["value"] = "No"
                else:
                    audio_data = self.get_audio(i, audiopath)
                    prompt[1]["value"] = "Yes"
            else:
                audio_data = self.get_audio(i, audiopath)
        else:
            if self.audio_only:
                audio_data = self.get_audio(i)
                prompt = image_data["output_texts"]
                avmask = 0
            else:
                audio_data = self.get_audio(i)
                image_data["output_texts"][0]['value'] = "In the image, " + image_data["output_texts"][0]['value'].lower()
                promptlist = [audio_data["output_texts"], image_data["output_texts"]]
                random.shuffle(promptlist)
                avmask = 1
                if random.random() < 0.5:
                    prompt = promptlist[0]
                else:
                    userprompt = promptlist[0][0]['value'] + ", and " + promptlist[1][0]['value'].lower()
                    gptresponse = promptlist[0][1]['value'] + ", and " + promptlist[1][1]['value'].lower()
                    prompt = [{'from': 'human', 'value': userprompt}, {'from': 'gpt', 'value': gptresponse}]
        return dict(
            image_paths=[audio_data["image_paths"], image_data["image_paths"]],
            output_texts=prompt,
            modality="audioimage",
            mask_audio=avmask,
            orig_paths=audio_data["orig_paths"],
            raw_audio=audio_data["raw_audio"]
        )

    def get_videoaudioimage(self, i):
        if i >= len(self.video_path_list):
            i = i % len(self.video_path_list)
        videopath = self.video_path_list[i]
        # print(videopath)
        if isinstance(videopath, list):
            videopath, audiopath = videopath
            video_data = self.get_video(i, videopath)
            audio_data = self.get_audio(i, audiopath)
            if "only_need_video" in videopath:
                avmask = [1, 0]
            # elif "egovideos" in videopath[0]:
            #     avmask = [1, 1] if random.random() > 0.2 else [0, 1]
            # elif "how2videos" in videopath[0]:
            #     avmask = [1, 1] if random.random() > 0.2 else [1, 0]
            else:
                avmask = [1, 1]
            output_texts = video_data["output_texts"]
            if self.use_npy and self.training: # npy training
                pass
            elif random.random() > 0.9 and len(self.audiofiles) != 0 and "yuwenyi" not in audiopath and self.training:
                output_texts[0]["value"] = random.choice(AUDIO_VIDEO_MATCHING)
                if random.random() > 0.5:
                    audio_data = self.get_audio(i, random.choice(self.audiofiles))
                    output_texts[1]["value"] = "No."
                else:
                    output_texts[1]["value"] = "Yes."
            return dict(
                image_paths=[audio_data["image_paths"], video_data["image_paths"]],
                output_texts=output_texts,
                modality=self.data_type,
                mask_audio=avmask,
                orig_paths=audio_data["orig_paths"],
                raw_audio=audio_data["raw_audio"]
            )
        else:
            video_data = self.get_video(i, videopath)
            audio_data = self.get_audio(i)
            if random.random() < 0.8:
                output_texts = video_data["output_texts"]
                mask_audio = [1, 0]
                # mask_audio = [1, 1]
            else:
                video_data["output_texts"][0]['value'] = "In the video, " + video_data["output_texts"][0]['value'].lower()
                promptlist = [audio_data["output_texts"], video_data["output_texts"]]
                random.shuffle(promptlist)
                userprompt = promptlist[0][0]['value'] + ", and, " + promptlist[1][0]['value'].lower()
                gptresponse = promptlist[0][1]['value'] + ", and, " + promptlist[1][1]['value'].lower()
                output_texts = [{'from': 'human', 'value': userprompt}, {'from': 'gpt', 'value': gptresponse}]
                mask_audio = [1, 1]
            # promptlist = [audio_data["output_texts"], video_data["output_texts"]]
            return dict(
                image_paths=[audio_data["image_paths"], video_data["image_paths"]],
                output_texts=output_texts,
                modality=self.data_type,
                mask_audio=mask_audio,
                orig_paths=audio_data["orig_paths"],
                raw_audio=audio_data["raw_audio"]
            )

    def __getitem__(self, i):
        if self.data_type == "audioimage" or self.modality == "audioimage":
            return self.get_audioimage(i)
        elif self.modality == "audiovideoimage":
            return self.get_videoaudioimage(i)
        elif self.modality == "audio":
            return self.get_audio(i)
        elif self.modality == "image":
            return self.get_image(i)
        elif self.modality == "video":
            return self.get_video(i)

    def sample_modality(self):
        self.modality = random.choice(self.modality_range)

    def collate(self, instances):
        image_paths = []
        output_texts = []
        first_modality = instances[0]["modality"]
        audiomasks = []
        orig_paths = []
        raw_audios = []
        trigger_reduce = 0
        if "video" in first_modality:
            length_thred = int(30 / self.clip_duration * self.sample_per_clip)
        for instance in instances:
            assert instance["modality"] == first_modality # should have the same modality in one minibatch
            if instance["modality"] == "video":
                if len(instance["image_paths"]) < length_thred:
                    image_paths.append(instance["image_paths"])
                    output_texts.append(instance["output_texts"])
            elif instance["modality"] in ["image", "audio", "audioimage"]:
                image_paths.append(instance["image_paths"])
                output_texts.append(instance["output_texts"])
                if "mask_audio" in instance:
                    if instance["mask_audio"] == 1:
                        instance["mask_audio"] = [1, 1]
                    else:
                        instance["mask_audio"] = [1, 0]
                audiomasks.append(instance["mask_audio"] if "mask_audio" in instance else [1, 1])
                orig_paths.append(instance["orig_paths"] if "orig_paths" in instance else "")
                raw_audios.append(instance["raw_audio"] if "raw_audio" in instance else None)
            elif instance["modality"] == "audiovideoimage":
                if self.use_npy or len(instance["image_paths"][1]) < length_thred:
                    image_paths.append(instance["image_paths"])
                    output_texts.append(instance["output_texts"])
                    audiomasks.append(instance["mask_audio"] if "mask_audio" in instance else [1, 1])
                    orig_paths.append(instance["orig_paths"] if "orig_paths" in instance else "")
                    raw_audios.append(instance["raw_audio"] if "raw_audio" in instance else None)
            # reduce if long
            # if len(instance["output_texts"][1]["value"].split()) > 80:
            #     trigger_reduce = max(trigger_reduce, len(instance["output_texts"][1]["value"].split()) // 80)
            if len(instance["output_texts"][1]["value"].split()) > 500:
                trigger_reduce = max(trigger_reduce, len(instance["output_texts"][1]["value"].split()) // 500)
            elif len(instance["output_texts"]) > 2:
                trigger_reduce = 3
        
            if "/AMI/BeamformIt/" in instance["orig_paths"]:
                image_paths = [instance["image_paths"]]
                output_texts = [instance["output_texts"]]
                audiomasks = [instance["mask_audio"] if "mask_audio" in instance else 0]
                orig_paths = [instance["orig_paths"] if "orig_paths" in instance else ""]
                raw_audios = [instance["raw_audio"] if "raw_audio" in instance else None]
                break
        
        if image_paths == []:
            if first_modality == "audiovideoimage":
                image_paths.append(
                    [instances[0]["image_paths"][0], instances[0]["image_paths"][1][:length_thred]]
                )
                audiomasks.append(instances[0]["mask_audio"])
            else:
                image_paths.append(instances[0]["image_paths"][:length_thred])
            output_texts.append(instances[0]["output_texts"])
            orig_paths.append(instances[0]["orig_paths"])
            raw_audios.append(instances[0]["raw_audio"] if "raw_audio" in instances[0] else None)
        elif len(image_paths) >= 2 and trigger_reduce > 0 and self.training:
            cut_len = len(image_paths) // (trigger_reduce + 1) + 1
            # print("reducing batchsize to {}".format(cut_len))
            image_paths = image_paths[:cut_len]
            output_texts = output_texts[:cut_len]
            audiomasks = audiomasks[:cut_len]
            orig_paths = orig_paths[:cut_len]
            raw_audios = raw_audios[:cut_len]

        self.sample_modality()
        return dict(
            image_paths=image_paths,
            output_texts=output_texts,
            modality=first_modality,
            audiomasks=torch.tensor(audiomasks) if audiomasks != [] else None,
            orig_paths=orig_paths,
            raw_audios=None if None in raw_audios else raw_audios,
        )

    def get_clip_timepoints(self, clip_sampler, duration):
        # Read out all clips in this video
        all_clips_timepoints = []
        is_last_clip = False
        end = 0.0
        while not is_last_clip:
            start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
            all_clips_timepoints.append((start, end))
        return all_clips_timepoints