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

from llava.dataset.preprocess_utils import preprocess_multimodal, preprocess, preprocess_multimodal_movie, preprocess_qwen
import json
import torch.distributed as dist
from torch.utils.data import Dataset
import math
import pickle
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip
from dataclasses import dataclass, field
from PIL import Image
import time
import re
import copy
import os
from typing import Dict, Optional, Sequence, List
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
import numpy as np
import ffmpeg
import transformers
import torch
from transformers import WhisperFeatureExtractor
import random
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
import soundfile as sf
from torch.nn.utils.rnn import pad_sequence
import signal
import io
from llava import conversation as conversation_lib
import cv2
# import nltk
# from nltk.corpus import words
import random

TT_AVAILABLE = False

# nltk.download('words')

def handler(signum, frame):
    raise Exception("Out of time!")

class LazyAVSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 test_split=None):
        super(LazyAVSupervisedDataset, self).__init__()
        if data_args.val_path is None:
            if test_split is not None:
                list_data_dict = test_split
                print('Testing Data Size:', len(list_data_dict))
            else:
                list_data_dict = json.load(open(data_path, "r"))
                
                if data_args.val_ratio != 0.0:
                    random.shuffle(list_data_dict)
                    self.test_data = list_data_dict[0 : int(data_args.val_ratio * len(list_data_dict))]
                    list_data_dict = list_data_dict[int(data_args.val_ratio * len(list_data_dict)) + 1:]
                else:
                    random.shuffle(list_data_dict)
                    print("Not Shuffle")
                    pass

                print('Training Data Size:', len(list_data_dict))
        else:
            list_data_dict = json.load(open(data_path, "r"))
            # gs534 - text-only data
            self.list_text_data_dict = [datapiece for datapiece in list_data_dict if 'textonly' in datapiece]
            if len(self.list_text_data_dict) > 0:
                list_data_dict = [datapiece for datapiece in list_data_dict if 'textonly' not in datapiece]
                print('Text-only Data Size:', len(self.list_text_data_dict))
                random.shuffle(list_data_dict)

        self.list_caption_data_dict = {}
        if os.path.exists(getattr(data_args, "caption_data", "")):
            with open(getattr(data_args, "caption_data", "")) as fin:
                list_caption_data_dict = json.load(fin)
            for datapiece in list_caption_data_dict:
                self.list_caption_data_dict[datapiece['video']] = list(datapiece["captions"].values())

        self.random_video = data_args.random_video
        if test_split is not None:
            self.random_video = False
        if self.random_video:
            self.all_video = [item['video'] for item in list_data_dict if 'video' in item]
            random.seed(2024)
    
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

    def get_test_set(self):
        return self.test_data

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
            # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
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
        else:
            image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image, image_size, image_mode # image_mode is 'image' for all `image_aspect_ratio` except fake_video

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[try #{attempt_idx}] Failed to fetch sample {i}. Line: {e.__traceback__.tb_lineno}, Exception:', e)
                time.sleep(0.1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                sample_idx = random.choice(range(len(self)))
                sample = self._get_item(sample_idx)
                return sample
            except Exception as e:
                # no need to sleep
                print(f'[try other #{attempt_idx}] Failed to fetch sample {sample_idx}. Line: {e.__traceback__.tb_lineno}, Exception:', e)
                pass

        # still fail, most likely to be path issue or cloud disk issue, retry the same sample for longer
        for attempt_idx in range(num_final_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f'[final try #{attempt_idx}] Failed to fetch sample {i}. Line: {e.__traceback__.tb_lineno}, Exception:', e)
                time.sleep(0.05)

        # Finally raise exception on failing.
        assert False, "Failed to fetch sample."

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]
            # gs534 - text-only data
            if len(self.list_text_data_dict) > 0 and random.random() < self.data_args.text_only_ratio:
                sources = copy.deepcopy(random.choice(self.list_text_data_dict))
                # sources["conversations"][0]["value"] = "<image>\n" + sources["conversations"][0]["value"]
                sources['video'] = "/mnt/bn/tiktok-mm-4/aiic/users/guangzhisun/dataprep/NEXTQA/videos/7555434398.mp4"
                sources['audio'] = "/mnt/bn/tiktok-mm-4/aiic/users/guangzhisun/dataprep/NEXTQA/audios/7555434398.wav"
            # print(sources['video'])
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

            if self.random_video:
                if 'image' not in sources[0] and 'video' not in sources[0] and 'audio' in sources[0]:
                    sources[0] = copy.deepcopy(self.list_data_dict[i])
                    sources[0]['video'] = random.choice(self.all_video)


            suffix = None
            if 'image' in sources[0]:
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

                        if "timestamps" not in sources[0]:
                            frame_idx = [k for k in range(0, total_frame_num, round(avg_fps))]
                            if len(videocaptions) > 0:
                                max_clip_frames = len(videocaptions) * 5 * self.data_args.video_fps
                                max_frames = max(max_frames, max_clip_frames)
                            if len(frame_idx) > max_frames:
                                frame_idx = np.linspace(0, total_frame_num - 1, max_frames, dtype=int).tolist()
                        else:
                            start = round(sources[0]["timestamps"][0] * vr.get_avg_fps())
                            end = round(sources[0]["timestamps"][1] * vr.get_avg_fps())
                            end = min(end, total_frame_num - 1)
                            frame_idx = [k for k in range(start, end + 1, round(avg_fps))]
                            if len(frame_idx) > max_frames:
                                frame_idx = np.linspace(start, end, max_frames, dtype=int).tolist()
                            real_time = sources[0]["timestamps"][1] - sources[0]["timestamps"][0]

                        video = vr.get_batch(frame_idx).asnumpy() # video: (F, H, W, C)
                        video = np.array(video)
                        processor = self.data_args.image_processor
                        image = processor.preprocess(video, return_tensors='pt')['pixel_values']

                        assert len(image) > 1

                        image = [(image, video[0].size, "video")]
                        process_sources = preprocess_multimodal(
                            copy.deepcopy([e["conversations"] for e in sources]),
                            self.data_args,
                            num_frames=image[0][0].shape[0]
                        )
                    except Exception as e:
                        print(f"Failed to read video file: {video_file}. Line: {e.__traceback__.tb_lineno}, Exception:", e)
                        return self._get_item(random.choice(range(len(self))))
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

                if "timestamps" in sources[0]:
                    audio = audio[sources[0]["timestamps"][0] * sr: sources[0]["timestamps"][1] * sr]

                org_audio = audio
                if True:
                    org_audio = org_audio[:, np.newaxis]

                if len(audio) < sr: # pad audio to at least 1s
                    sil = np.zeros(sr - len(audio), dtype=float)
                    audio = np.concatenate((audio, sil), axis=0)
                
                # audio = audio[: 30]
                # spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()

                if 'video' in sources[0]:
                    audio = audio[:round(sr * real_time)] 
                    org_audio = org_audio[:round(sr * real_time), :] 
                else:
                    audio = audio[:round(sr * self.max_time)] # truncate audio to at most 30s
                    org_audio = org_audio[:round(sr * self.max_time), :]
                audio_lst = [audio[k: k + 30 * sr] for k in range(0, len(audio), 30 * sr)]
                spectrogram_lst = [self.wav_processor(a, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze() for a in audio_lst]
                org_audio_lst = [org_audio[k: k + 30 * sr, :] for k in range(0, len(audio), 30 * sr)]

            else:
                audio_file = None
                sr = 16000
                if 'video' in sources[0]:
                    audio = np.zeros(round(sr * real_time))
                else:
                    audio = np.zeros(sr * 10)
                org_audio = audio
                if True:
                    org_audio = org_audio[:, np.newaxis]

                audio_lst = [audio[k: k + 30 * sr] for k in range(0, len(audio), 30 * sr)]
                spectrogram_lst = [self.wav_processor(a, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze() for a in audio_lst]
                org_audio_lst = [org_audio[k: k + 30 * sr, :] for k in range(0, len(org_audio), 30 * sr)]

            has_image = ('image' in sources[0]) or ('video' in sources[0]) or ('audio' in sources[0])
            if "video" in sources[0]:
                data_id = "['{}', '{}']".format(sources[0]["video"], audio_file)
                if "extend_video" in sources[0] and len(sources[0]["extend_video"]) != 0:
                    data_id = "['{}', '{}', '{}']".format(sources[0]["video"], audio_file, str(sources[0]["extend_video"]))
            else:
                data_id = "['{}', '{}']".format(None, audio_file)

            # gs534 - get process values
            if "process_rewards" in sources[0]:
                for n, source in enumerate(sources):
                    process_sources[n].append({"from": "process", "value": source["process_rewards"]})
            elif "prm" in getattr(self.data_args, "train_orm", "") and "unfinished" in sources[0]:
                for n, source in enumerate(sources):
                    process_sources[n].append({"from": "process", "value": ""})

            data_dict = preprocess(
                process_sources,
                self.tokenizer,
                has_image=has_image,
                prompt=self.data_args.input_prompt,
                refine_prompt=self.data_args.refine_prompt
            )

            prompt = [sources[0]["conversations"][k] for k in range(len(sources[0]["conversations"]) - 1)]

            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0], reject_input_ids=data_dict["reject_input_ids"][0], reject_labels=data_dict["reject_labels"][0], gt_input_ids=data_dict["gt_input_ids"][0], gt_labels=data_dict["gt_labels"][0])

            if "metric" in sources[0]["conversations"][1]:
                data_dict['metric_chosen'] = sources[0]["conversations"][1]['metric']
                data_dict['metric_reject'] = sources[0]["conversations"][2]['metric']
            else:
                data_dict['metric_chosen'] = None
                data_dict['metric_reject'] = None

            
            # image exist in the data
            if 'image' in sources[0]:
                data_dict['image'] = image
                data_dict['modality'] = "image"
            elif 'video' in sources[0]:
                data_dict['image'] = image
                if 'audio' not in sources[0] and not ('tos_audio' in sources[0] and sources[0]['tos_audio']):
                    data_dict['modality'] = "video"
                else:
                    data_dict['modality'] = "audio-video"
            elif "textonly" in sources[0]:
                data_dict['modality'] = "audio-video"
                data_dict['image'] = None
            else:
                data_dict['modality'] = "audio"
                data_dict['image'] = None

            data_dict["raw_wav"] = audio_lst
            data_dict["spectrogram"] = torch.stack(spectrogram_lst, dim=0)
            data_dict["multi_channel_wav"] = org_audio_lst

            if data_dict['modality'] != "audio":
                if "textonly" not in sources[0]:
                    data_dict["real_time"] = real_time
                else:
                    data_dict["real_time"] = 0
            else:
                data_dict["real_time"] = 30 * len(audio_lst)

            if "duration" in sources[0]:
                data_dict["duration"] = sources[0]["duration"]

            if "captions" in sources[0]:
                data_dict["captions"] = sources[0]["captions"]

            # prompt exist in the data
            if prompt is not None:
                data_dict['prompt'] = prompt

            data_dict['id'] = data_id # "['{}', '{}']".format(sources[0]['video'], audio_file)

            if self.data_args.online_self_dpo or "online" in getattr(self.data_args, "train_orm", ""):
                prompt_sentence = sources[0]["conversations"][0]["value"]
                if self.data_args.online_self_dpo:
                    prompt_online_loss = sources[0]["prompt_for_online_loss"] # sources[0]["conversations"][0]["value"] # 

                replace_token = DEFAULT_IMAGE_TOKEN
                if self.data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt_sentence = prompt_sentence.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                if self.data_args.online_self_dpo:
                    prompt_online_loss = prompt_online_loss.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                if "unfinished" in sources[0] and sources[0]["unfinished"]:
                    unfinished_response = "<end_of_step>\n".join(sources[0]["conversations"][1]["value"].split("<end_of_step>\n")[:-2])
                    fake_conversations = [{"from": "human", "value": prompt_sentence}, {"from": "gpt", "value": unfinished_response}]
                    data_dict["unfinished"] = True
                else:
                    fake_conversations = [{"from": "human", "value": prompt_sentence}, {"from": "gpt", "value": ""}]
                if self.data_args.online_self_dpo:
                    fake_conversations_online_loss = [{"from": "human", "value": prompt_online_loss}, {"from": "gpt", "value": ""}]

                if conversation_lib.default_conversation.version == "qwen":
                    prompt_sentence_dict = preprocess_qwen([fake_conversations], self.tokenizer, has_image=has_image)
                    prompt_input_ids = prompt_sentence_dict["input_ids"][:, :-2]
                    prompt_labels = prompt_sentence_dict["labels"][:, :-2]

                    if self.data_args.online_self_dpo:
                        prompt_online_loss_dict = preprocess_qwen([fake_conversations_online_loss], self.tokenizer, has_image=has_image)
                        prompt_online_loss_input_ids = prompt_online_loss_dict["input_ids"][:, :-2]
                        prompt_online_loss_labels = prompt_online_loss_dict["labels"][:, :-2]
                else:
                    raise Exception(f"Conversation version {conversation_lib.default_conversation.version} is not supported")
                
                data_dict['online_inference_input_ids'] = prompt_input_ids[0]
                data_dict['online_inference_labels'] = prompt_labels[0]
                if self.data_args.online_self_dpo:
                    data_dict['online_loss_input_ids'] = prompt_online_loss_input_ids[0]
                    data_dict['online_loss_labels'] = prompt_online_loss_labels[0]
                data_dict['online_loss_input_ids'] = None
                data_dict['online_loss_labels'] = None
                if "online" in getattr(self.data_args, "train_orm", "") or (getattr(self.data_args, "do_rag", False) and "ref_answer" in sources[0]):
                    data_dict["ref_answer"] = sources[0]["ref_answer"]
            elif "grpo" in getattr(self.data_args, "train_orm", "") and "preds_pos" in sources[0]:
                data_dict["advantages"] = []
                data_dict['preds_inputs'] = []
                data_dict['preds_labels'] = []
                for group in ["preds_pos", "preds_neg"]:
                    for pred_pos in sources[0][group]:
                        prompt_sentence = sources[0]["conversations"][0]["value"]
                        replace_token = DEFAULT_IMAGE_TOKEN
                        if self.data_args.mm_use_im_start_end:
                            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                        prompt_sentence = prompt_sentence.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                        fake_conversations = [{"from": "human", "value": prompt_sentence}, {"from": "gpt", "value": pred_pos}]
                        prompt_sentence_dict = preprocess_qwen([fake_conversations], self.tokenizer, has_image=has_image)
                        data_dict['preds_inputs'].append(prompt_sentence_dict["input_ids"])
                        data_dict['preds_labels'].append(prompt_sentence_dict["labels"])
                        data_dict["advantages"].append(1 if group == "preds_pos" else 0)
                data_dict['online_inference_input_ids'] = None
                data_dict['online_inference_labels'] = None
                data_dict['online_loss_input_ids'] = None
                data_dict['online_loss_labels'] = None
            else:
                data_dict['online_inference_input_ids'] = None
                data_dict['online_inference_labels'] = None
                data_dict['online_loss_input_ids'] = None
                data_dict['online_loss_labels'] = None
            if "soft" in getattr(self.data_args, "train_orm", ""):
                data_dict['orm_loss_labels'] = [sources[0]["pos_score"], sources[0]["neg_score"]]

            return data_dict
        
        except Exception as e:
            # sleep 1s in case it is a cloud disk issue
            print(f'GGGG {i}. Line: {e.__traceback__.tb_lineno}, Exception:', e)
            return self._get_item(random.choice(range(len(self))))
        

@dataclass
class DataCollatorForAVSupervisedDataset(object):
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
        input_ids, labels, ids, reject_input_ids, reject_labels, gt_input_ids, gt_labels, online_inference_input_ids, online_inference_labels, online_loss_input_ids, online_loss_labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", 'id', "reject_input_ids", "reject_labels", "gt_input_ids", "gt_labels", 'online_inference_input_ids', 'online_inference_labels', "online_loss_input_ids", "online_loss_labels"))
        input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[:self.tokenizer.model_max_length] for _labels in labels]
        # gs534 - online dpo
        ref_answer = [instances[i]["ref_answer"] for i in range(len(instances))] if "ref_answer" in instances[0] else None
        unfinished = [instances[i]["unfinished"] for i in range(len(instances))] if "unfinished" in instances[0] else None
        preds_inputs = [instances[i]["preds_inputs"] for i in range(len(instances))] if "preds_inputs" in instances[0] else None
        preds_labels = [instances[i]["preds_labels"] for i in range(len(instances))] if "preds_labels" in instances[0] else None
        advantages = [instances[i]["advantages"] for i in range(len(instances))] if "advantages" in instances[0] else None
        duration = [instances[i]["duration"] for i in range(len(instances))] if "duration" in instances[0] else None
        captions = [instances[i]["captions"] for i in range(len(instances))] if "captions" in instances[0] else None
        # raw_video = [instances[i]["raw_video"] for i in range(len(instances))] if "raw_video" in instances[0] else None

        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                # print("Setting pad token to bos token for qwen model.")
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

        for input in reject_input_ids:
            if input is None:
                reject_input_ids = None
                reject_labels = None
                reject_attention_mask=None
                dpo_forward=False
                gt_input_ids = None
                gt_labels = None
                gt_attention_mask = None
                break
        else:
            reject_input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in reject_input_ids]
            reject_labels = [_labels[:self.tokenizer.model_max_length] for _labels in reject_labels]
            reject_input_ids = self.pad_sequence(
                reject_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            reject_labels = self.pad_sequence(reject_labels,
                                    batch_first=True,
                                    padding_value=IGNORE_INDEX)
            reject_attention_mask=reject_input_ids.ne(self.tokenizer.pad_token_id)
            dpo_forward=True
            gt_input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in gt_input_ids]
            gt_labels = [_labels[:self.tokenizer.model_max_length] for _labels in gt_labels]
            gt_input_ids = self.pad_sequence(
                gt_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            gt_labels = self.pad_sequence(gt_labels,
                                    batch_first=True,
                                    padding_value=IGNORE_INDEX)
            gt_attention_mask=gt_input_ids.ne(self.tokenizer.pad_token_id)


        if online_inference_input_ids[0] is None:
            online_inference_input_ids = None
            online_inference_labels = None
            online_loss_input_ids = None
            online_loss_labels = None
        else:
            online_inference_input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in online_inference_input_ids]
            online_inference_labels = [_labels[:self.tokenizer.model_max_length] for _labels in online_inference_labels]
            if online_loss_input_ids[0] is not None:
                online_loss_input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in online_loss_input_ids]
                online_loss_labels = [_labels[:self.tokenizer.model_max_length] for _labels in online_loss_labels]
            online_inference_input_ids = self.pad_sequence(
                online_inference_input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            )
            online_inference_labels = self.pad_sequence(
                online_inference_labels,
                batch_first=True,
                padding_value=IGNORE_INDEX
            )
            if online_loss_input_ids[0] is not None:
                online_loss_input_ids = self.pad_sequence(
                    online_loss_input_ids,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id
                )
                online_loss_labels = self.pad_sequence(
                    online_loss_labels,
                    batch_first=True,
                    padding_value=IGNORE_INDEX
                )
            dpo_forward = True

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            reject_input_ids=reject_input_ids,
            reject_labels=reject_labels,
            gt_input_ids=gt_input_ids,
            gt_labels=gt_labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            ids=ids,
            reject_attention_mask=reject_attention_mask,
            gt_attention_mask=gt_attention_mask,
            dpo_forward=dpo_forward,
            online_inference_input_ids=online_inference_input_ids,
            online_inference_labels=online_inference_labels,
            online_loss_input_ids=online_loss_input_ids,
            online_loss_labels=online_loss_labels,
            ref_answer=ref_answer,
            unfinished=unfinished,
            preds_inputs=preds_inputs,
            preds_labels=preds_labels,
            advantages=advantages,
            duration=duration,
            captions=captions,
            # raw_video=raw_video,
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
                
        if 'prompt' in instances[0]:
            batch['prompts'] = [instance['prompt'] for instance in instances]

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

        multi_channel_wav = []
        for s in instances:
            multi_channel_wav.extend(s["multi_channel_wav"])
        multi_channel_wav = [torch.from_numpy(it) for it in multi_channel_wav]
        multi_channel_wav = pad_sequence(multi_channel_wav, batch_first=True, padding_value=0).transpose(1, 2)

        batch["spectrogram"] = cat_spectrogram
        batch["raw_wav"] = raw_wav
        batch["multi_channel_wav"] = multi_channel_wav
        batch['org_groups'] = org_groups

        if "orm_loss_labels" in instances[0]:
            batch["orm_loss_labels"] = [instance["orm_loss_labels"] for instance in instances]

        batch['real_time'] = [s["real_time"] for s in instances]
        batch['metric_chosen'] = [s["metric_chosen"] for s in instances]
        batch['metric_reject'] = [s["metric_reject"] for s in instances]

        return batch

@dataclass
class DataCollatorForAVSupervisedDatasetFullFrame(object):
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
        input_ids = None
        input_ids = None
        labels = None
        ids = None

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=None,
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
        
        batch['prompts'] = None


        batch["spectrogram"] = None
        batch["raw_wav"] = None
        batch['org_groups'] = None

        batch['real_time'] = None

        return batch