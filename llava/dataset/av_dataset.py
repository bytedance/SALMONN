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

# Adapted from https://github.com/LLaVA-VL/LLaVA-NeXT. The original license is located at 'third-party-license/llava_next.txt'.

from llava.dataset.preprocess_utils import preprocess_multimodal, preprocess
import json
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from dataclasses import dataclass
import copy
from typing import Dict, Sequence
from llava.constants import IGNORE_INDEX
import numpy as np
import transformers
import torch
from transformers import WhisperFeatureExtractor
import random
import soundfile as sf
import random

class LazyAVSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args, is_test=False):
        super(LazyAVSupervisedDataset, self).__init__()

        list_data_dict = json.load(open(data_path, "r"))
    
        print("Formatting inputs...Skip in lazy mode. Audio visual dataset")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        whisper_path = self.data_args.audio_processor
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.max_time = data_args.max_time

        self.is_test = is_test

        self.max_frame_num = round(self.max_time * self.data_args.video_fps)
        print("Max frame num: ", self.max_frame_num)

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
            cur_len = cur_len if ('image' in sample) or ('video' in sample) or ("audio" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self._get_item(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1  # "Don't know why it is wrapped to a list"

            ori_item = copy.deepcopy(sources[0])
            prompt = [sources[0]["conversations"][k] for k in range(len(sources[0]["conversations"]) - 1)]

            text = sources[0]["conversations"][-1]["value"]
            if self.is_test:
                if "prefix" not in sources[0]["conversations"][-1]:
                    sources[0]["conversations"][-1]["value"] = ""
                else:
                    sources[0]["conversations"][-1]["value"] = sources[0]["conversations"][-1]["prefix"]

            if 'video' in sources[0]:
                video_file = sources[0]['video']
                vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)

                total_frame_num = len(vr)
                ori_fps = vr.get_avg_fps()
                avg_fps = max(round(ori_fps / self.data_args.video_fps), 1)
                real_time = total_frame_num / vr.get_avg_fps()
                
                max_frames = self.max_frame_num

                if "timestamps" not in sources[0]:
                    frame_idx = [k for k in range(0, total_frame_num, round(avg_fps))]
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
                )

            else:
                # sources = copy.deepcopy([e["conversations"] for e in sources])
                process_sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args,
                )
            
            if 'audio' in sources[0]:
                audio_file = sources[0]['audio']
                audio, sr = sf.read(audio_file)
                assert sr == 16000  # only support 16kHz audio
                if len(audio.shape) == 2: # stereo to mono
                    audio = audio[:, 0]

                if "timestamps" in sources[0]:
                    audio = audio[sources[0]["timestamps"][0] * sr: sources[0]["timestamps"][1] * sr]

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

            has_image = ('image' in sources[0]) or ('video' in sources[0]) or ('audio' in sources[0])
            if "video" in sources[0]:
                data_id = "['{}', '{}']".format(sources[0]["video"], audio_file)
            else:
                data_id = "['{}', '{}']".format(None, audio_file)

            data_dict = preprocess(process_sources, self.tokenizer, has_image=has_image)

            if self.is_test:
                data_dict["input_ids"] = data_dict["input_ids"][:, :-2]
                data_dict["labels"] = data_dict["labels"][:, :-2]
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0], reject_input_ids=data_dict["reject_input_ids"][0], reject_labels=data_dict["reject_labels"][0], gt_input_ids=data_dict["gt_input_ids"][0], gt_labels=data_dict["gt_labels"][0])

            if 'video' in sources[0]:
                data_dict['image'] = image
                if 'audio' not in sources[0]:
                    data_dict['modality'] = "video"
                else:
                    data_dict['modality'] = "audio-video"
            elif "audio" in sources[0]:
                data_dict['modality'] = "audio"
                data_dict['image'] = None
            else:
                data_dict['modality'] = "text"
                data_dict['image'] = None

            if audio_file is not None:
                data_dict["spectrogram"] = torch.stack(spectrogram_lst, dim=0)
            else:
                data_dict["spectrogram"] = None

            if data_dict['modality'] != "audio" and data_dict['modality'] != "text":
                data_dict["real_time"] = real_time
            else:
                data_dict["real_time"] = 30 * len(audio_lst)

            data_dict['prompt'] = prompt

            data_dict['id'] = data_id

            data_dict["ori_item"] = ori_item
            data_dict["ce_only"] = sources[0].get("ce_only", False)
            data_dict["text"] = text

            return data_dict
        
        except Exception as e:
            print(f'GGGG {i}. Line: {e.__traceback__.tb_lineno}, Exception:', e)
            if self.is_test:
                raise e
            else:
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
        input_ids, labels, ids, reject_input_ids, reject_labels, gt_input_ids, gt_labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", 'id', "reject_input_ids", "reject_labels", "gt_input_ids", "gt_labels"))
        input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[:self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                self.tokenizer.pad_token_id = 151643
            else:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME
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
        )

        batch['modalities'] = [im['modality'] for im in instances]

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]

            true_images = []
            for im_list in images:
                if im_list is not None:
                    for im in im_list:
                        true_images.append(im[0])
                else:
                    true_images.append(None)
            images = true_images
            batch['images'] = images
                
        if 'prompt' in instances[0]:
            batch['prompts'] = [instance['prompt'] for instance in instances]

        if instances[0]["spectrogram"] is not None: # assert batchsize = 1
            samples_spectrogram = [s["spectrogram"] for s in instances]
            cat_spectrogram = torch.cat(samples_spectrogram, dim=0)
            org_groups = [s.size(0) for s in samples_spectrogram]

            
            batch["spectrogram"] = cat_spectrogram
            batch['org_groups'] = org_groups

        batch['real_time'] = [s["real_time"] for s in instances]
        batch["ori_item"] = [s["ori_item"] for s in instances]
        batch["ce_only"] = [s["ce_only"] for s in instances]
        batch["texts"] = [s["text"] for s in instances]

        return batch