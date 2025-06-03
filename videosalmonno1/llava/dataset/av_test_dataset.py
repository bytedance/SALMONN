# Copyright (2025) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/llava/model/language_model/llava_qwen.py. Copyright 2024 Hao Zhang. The original license is located at 'third-party-license/llava_next.txt'.
# Adapted from https://github.com/bytedance/SALMONN. The original license is located at 'third-party-license/salmonn.txt'.

from llava.dataset.preprocess_utils import preprocess_multimodal, preprocess, preprocess_multimodal_movie
import json
from torch.utils.data import Dataset
import torch.distributed as dist
import math
import pickle
from decord import VideoReader, cpu
import re
from moviepy.editor import VideoFileClip
from dataclasses import dataclass, field
from PIL import Image
import time
import copy
import cv2
import os
from typing import Dict, Optional, Sequence, List
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
import numpy as np
import transformers
import torch 
from transformers import WhisperFeatureExtractor
import random
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence
import signal
import io

TT_AVAILABLE = False

def handler(signum, frame):
    raise Exception("Out of time!")


class LazyAVTestDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazyAVTestDataset, self).__init__()
        if isinstance(data_path, list):
            list_data_dict = data_path
        else:
            list_data_dict = json.load(open(data_path, "r"))
        print('Training Data Size:', len(list_data_dict))

        self.list_caption_data_dict = {}
        if os.path.exists(getattr(data_args, "caption_data", "")):
            with open(getattr(data_args, "caption_data", "")) as fin:
                list_caption_data_dict = json.load(fin)
            for datapiece in list_caption_data_dict:
                self.list_caption_data_dict[datapiece['video']] = list(datapiece["captions"].values())

        print("Formatting inputs...Skip in lazy mode. Audio visual dataset")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        whisper_path = self.data_args.audio_processor
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.max_time = data_args.max_time

        self.max_frame_num = round(self.max_time * self.data_args.video_fps)
        print("Max frame num: ", self.max_frame_num)

        self.insert_time_precision = data_args.insert_time_precision
        self.demo = False

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            # cur_len = cur_len if 'image' in sample else -cur_len
            cur_len = cur_len if ('image' in sample) or ('video' in sample) or ("audio" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def process_image(self, image_file):
        
        image_mode = "image" # modality determine
        
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        if not isinstance(image_file, np.ndarray):
            image = Image.open(image_file).convert('RGB')
        else:
            image = Image.fromarray(image_file).convert('RGB')
        image_size = image.size
        if self.data_args.image_aspect_ratio == 'highres':
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif self.data_args.image_aspect_ratio == 'anyres':
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif self.data_args.image_aspect_ratio == 'crop_split':
            image = process_highres_image_crop_split(image, self.data_args)
        elif self.data_args.image_aspect_ratio == 'pad':
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        elif self.data_args.image_aspect_ratio == 'fake_video':
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            # copy n images to form a fake video
            random_frames = random.randint(5, 16)
            image = torch.stack([image]*random_frames, dim=0)
            image_mode = "video"
            # breakpoint()
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image, image_size, image_mode # image_mode is 'image' for all `image_aspect_ratio` except fake_video

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        sample = self._get_item(i)
        return sample

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if "textonly" in sources:
            sources['video'] = "/mnt/bn/tiktok-mm-4/aiic/users/guangzhisun/dataprep/NEXTQA/videos/7555434398.mp4"
            sources['audio'] = "/mnt/bn/tiktok-mm-4/aiic/users/guangzhisun/dataprep/NEXTQA/audios/7555434398.wav"
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # assert len(sources[0]["conversations"]) == 2 or len(sources[0]["conversations"]) == 3 # TODO: Support multi-turn dialog

        prompt = [sources[0]["conversations"][k] for k in range(len(sources[0]["conversations"]) - 1)]
        label = sources[0]["conversations"][-1]["value"]
        if "train" not in getattr(self.data_args, "train_orm", "") and getattr(self.data_args, "rag_type", "") != "replace":
            sources[0]["conversations"][-1]["value"] = ""

        # if self.random_video:
        #     if 'image' not in sources[0] and 'video' not in sources[0] and 'audio' in sources[0]:
        #         sources[0] = copy.deepcopy(self.list_data_dict[i])
        #         sources[0]['video'] = random.choice(self.all_video)

        suffix = None
        real_time = 0
        # breakpoint()
        if 'image' in sources[0]:
            # breakpoint()
            image_file = sources[0]['image']
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
            else:
                image = [self.process_image(image_file)]
            num_frames = 0
            process_sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args,
                num_frames
            )
        elif 'video' in sources[0]:
            if sources[0]["video"] in self.list_caption_data_dict:
                videocaptions = self.list_caption_data_dict[sources[0]["video"]]
                sources[0]["captions"] = videocaptions
            else:
                videocaptions = []

            if "tt_id" in sources[0]:
                video_file = sources[0]['tt_id']
                suffic = None
            else:
                video_file = sources[0]['video']
                if not isinstance(video_file, list):
                    suffix = video_file.split('.')[-1]
                    if not os.path.exists(video_file):
                        print('File {} not exist!'.format(video_file))
                
            if suffix == 'pkl':
                video_info = pickle.load(open(video_file, 'rb'))
                image = torch.from_numpy(video_info['feats'][:, 1:])
                input_prompt = video_info['inputs'].replace('...', '')
                # replace the default image token with multiple tokens
                input_prompt = input_prompt.replace(DEFAULT_IMAGE_TOKEN, 
                                                    DEFAULT_IMAGE_TOKEN * self.data_args.video_token)
                process_sources, query_prompt = preprocess_multimodal_movie(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args, input_prompt)
            else:
                try:
                    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)

                    total_frame_num = len(vr)
                    ori_fps = vr.get_avg_fps()
                    avg_fps = max(round(ori_fps / self.data_args.video_fps), 1)
                    real_time = total_frame_num / vr.get_avg_fps()
                    
                    max_frames = self.max_frame_num
                    frame_idx = [k for k in range(0, total_frame_num, round(avg_fps))]
                    if len(frame_idx) > max_frames:
                        frame_idx = np.linspace(0, total_frame_num - 1, max_frames, dtype=int).tolist()

                    video = vr.get_batch(frame_idx).asnumpy() # video: (F, H, W, C)
                    video = np.array(video)

                    processor = self.data_args.image_processor

                    image = processor.preprocess(video, return_tensors='pt')['pixel_values']
                    image = [(image, video[0].size, "video")]
                    process_sources = preprocess_multimodal(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args,
                        num_frames=image[0][0].shape[0]
                    )

                except Exception as e:
                    print(f"Failed to read video file: {video_file}. Line: {e.__traceback__.tb_lineno}, Exception:", e)
                    if self.demo:
                        raise e
                    else:
                        return self._get_item(0)
        
        else:
            # sources = copy.deepcopy([e["conversations"] for e in sources])
            process_sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args,
                num_frames=0, # image[0][0].shape[0]
            )

        if 'audio' in sources[0]:
            audio_file = sources[0]['audio']
            audio, sr = sf.read(audio_file)
            if len(audio.shape) == 2: # stereo to mono
                audio = audio[:, 0]
            if len(audio) < sr: # pad audio to at least 1s
                sil = np.zeros(sr - len(audio), dtype=float)
                audio = np.concatenate((audio, sil), axis=0)

            if 'video' in sources[0]:
                audio = audio[:round(sr * real_time)] 
            else:
                audio = audio[:round(sr * self.max_time)] # truncate audio to at most 30s
            audio_lst = [audio[k: k + 30 * sr] for k in range(0, len(audio), 30 * sr)]
            spectrogram_lst = [self.wav_processor(a, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze() for a in audio_lst]
        
        else:
            audio_file = None
            sr = 16000
            if 'video' in sources[0]:
                audio = np.zeros(round(sr * real_time))
            else:
                audio = np.zeros(sr * 10)

            audio_lst = [audio[k: k + 30 * sr] for k in range(0, len(audio), 30 * sr)]
            spectrogram_lst = [self.wav_processor(a, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze() for a in audio_lst]


        has_image = ('image' in sources[0]) or ('video' in sources[0]) or ('audio' in sources[0])
        if "video" in sources[0]:
            data_id = "['{}', '{}']".format(sources[0]["video"], audio_file)
        else:
            data_id = "['{}', '{}']".format(None, audio_file)
        data_dict = preprocess(
            process_sources,
            self.tokenizer,
            has_image=has_image,
            prompt=self.data_args.input_prompt,
            refine_prompt=self.data_args.refine_prompt
        )

        # if 'prompt' in data_dict:
        #     prompt = data_dict['prompt']
        # else:
        #     prompt = None
        
        # if suffix == 'pkl':
        #     prompt = [query_prompt]

        if isinstance(i, int):
            if "train" not in getattr(self.data_args, "train_orm", "") and getattr(self.data_args, "rag_type", "") != "replace":
                data_dict["input_ids"] = data_dict["input_ids"][:, :-2]
                data_dict["labels"] = data_dict["labels"][:, :-2]
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
            if "prefix_steps" in sources[0]:
                data_dict["prefix_steps"] = sources[0]["prefix_steps"]

        # breakpoint()
        # image exist in the data
        if 'image' in sources[0]:
            data_dict['image'] = image
            data_dict['modality'] = "image"
        elif 'video' in sources[0]:
            # breakpoint()
            data_dict['image'] = image
            if 'audio' not in sources[0]:
                data_dict['modality'] = "video"
            else:
                data_dict['modality'] = "audio-video"
        else:
            data_dict['modality'] = "audio"
            data_dict['image'] = None

        data_dict["raw_wav"] = audio_lst
        data_dict["spectrogram"] = torch.stack(spectrogram_lst, dim=0)

        # prompt exist in the data
        # breakpoint()
        # if prompt is not None:
        #     data_dict['prompt'] = prompt

        if data_dict['modality'] != "audio":
            data_dict['id'] = data_id # "['{}', '{}']".format(sources[0]['video'], audio_file)
        else:
            data_dict['id'] = audio_file

        if "captions" in sources[0]:
            data_dict["captions"] = sources[0]["captions"]
        data_dict['prompt'] = prompt
        data_dict['text'] = label

        data_dict["real_time"] = real_time

        if "duration" in sources[0]:
            data_dict["duration"] = sources[0]["duration"]
        
        return data_dict


@dataclass
class DataCollatorForAVTestDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids] 
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=batch_first,
            padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels",'id'))

        captions = [instances[i]["captions"] for i in range(len(instances))] if "captions" in instances[0] else None
        input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[:self.tokenizer.model_max_length] for _labels in labels]
        # gs534 - duration
        duration = [instances[i]["duration"] for i in range(len(instances))] if "duration" in instances[0] else None
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                self.tokenizer.pad_token_id = 151643
            else:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
        input_ids = self.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels,
                                   batch_first=True,
                                   padding_value=IGNORE_INDEX)
        # import pdb;pdb.set_trace()
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            ids=ids
        )

        batch['modalities'] = [im['modality'] for im in instances]

        if 'image' in instances[0]:
            # instances[1]['image'][0][0].shape
            # torch.Size([5, 3, 224, 224])
            images = [instance['image'] for instance in instances]
        
            batch['image_sizes'] = [] # [im[1] for im_list in images for im in im_list]
            for im_list in images:
                if im_list is not None:
                    for im in im_list:
                        batch['image_sizes'].append(im[1])
                else:
                    batch['image_sizes'].append(None)

            true_images = [] # [im[0] for im_list in images for im in im_list]
            for im_list in images:
                if im_list is not None:
                    for im in im_list:
                        true_images.append(im[0])
                else:
                    true_images.append(None)
            images = true_images

            if False: # all(x is not None and x.shape == images[0].shape for x in images):
                # Image: (N, P, C, H, W)
                # Video: (N, F, C, H, W)
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        # import pdb;pdb.set_trace()
        if 'prompt' in instances[0]:
            batch['prompts'] = [instance['prompt'] for instance in instances]
        
        if "text" in instances[0]:
            batch['texts'] = [instance['text'] for instance in instances]

        if "prefix_steps" in instances[0]:
            batch["prefix_steps"] = [instance["prefix_steps"] for instance in instances]

        # samples_spectrogram = [s["spectrogram"] for s in instances]
        # cat_spectrogram = torch.stack(samples_spectrogram, dim=0)
        # raw_wav = [torch.from_numpy(s["raw_wav"]) for s in instances]
        # raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)

        samples_spectrogram = [s["spectrogram"] for s in instances]
        cat_spectrogram = torch.cat(samples_spectrogram, dim=0)
        org_groups = [s.size(0) for s in samples_spectrogram]

        raw_wav = []
        for s in instances:
            raw_wav.extend(s["raw_wav"])
        raw_wav = [torch.from_numpy(it) for it in raw_wav]
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)

        batch["spectrogram"] = cat_spectrogram
        batch["raw_wav"] = raw_wav
        batch['org_groups'] = org_groups

        batch["captions"] = captions
        batch['real_time'] = [s["real_time"] for s in instances]
        batch['duration'] = duration

        return batch

