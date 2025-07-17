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

# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl.txt'.

import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchcodec.decoders import VideoDecoder, AudioDecoder
import transformers

from .rope2d import get_rope_index_25, get_rope_index_2
from decord import VideoReader, cpu

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_AUDIO_TOKEN = "<audio>"

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def split_into_groups(counts, groups):
    result = []
    for count, g in zip(counts, groups):
        base = count // g
        remainder = count % g
        group_list = [base + 1] * remainder + [base] * (g - remainder)
        result.append(group_list)
    return result

def generate_id_target(
    source,
    grid_thw_image, 
    grid_thw_video, 
    audio_lengths, 
    tokenizer, 
    target_role,
    merge_size: int = 2,
):
    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    roles = {"human": "user", "gpt": "assistant", "chosen": "assistant", "reject": "assistant"}
    system_message = "You are a helpful assistant."
    input_id, target = [], []

    input_id += tokenizer.apply_chat_template(
        [{"role": "system", "content": system_message}]
    )
    target += [IGNORE_INDEX] * len(input_id)
    for conv in source:
        try:
            role = conv["role"]
            content = conv["content"]
        except:
            role = conv["from"]
            content = conv["value"]
        if role not in ["human", target_role]:
            continue

        role = roles.get(role, role)
        if role == "user":
            if "<image>" in content:
                parts = content.split("<image>")
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    replacement = (
                        "<|vision_start|>"
                        + f"<|image_pad|>"
                        * grid_thw_image[i]
                        + "<|vision_end|>"
                    )
                    new_parts.append(replacement)
                new_parts.append(parts[-1])
                content = "".join(new_parts)

            if "<video>" in content:
                parts = content.split("<video>")
                new_parts = []
                if audio_lengths is None:
                    grid_thw_video = [
                        merged_thw.prod() // merge_size**2
                        for merged_thw in grid_thw_video
                    ]
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|video_pad|>"
                            * grid_thw_video[i]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)
                else:
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        per_timestep_audio_len = split_into_groups(audio_lengths, [grid_thw_video[i][0] for i in range(len(grid_thw_video))])
                        replacement = "<|vision_start|>"
                        for timestep in range(grid_thw_video[i][0]):
                            replacement += (
                                f"<|video_pad|>" 
                                * (grid_thw_video[i][1] * grid_thw_video[i][2] // merge_size**2)
                                + f"<|audio_pad|>"
                                * per_timestep_audio_len[i][timestep]
                            )
                        replacement += "<|vision_end|>"
                        new_parts.append(replacement)
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)
                            

            if "<audio>" in content:
                parts = content.split("<audio>")
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    replacement = (
                        "<|vision_start|>" # no need to train more start token
                        + f"<|audio_pad|>"
                        * audio_lengths[i]
                        + "<|vision_end|>"
                    )
                    new_parts.append(replacement)
                new_parts.append(parts[-1])
                content = "".join(new_parts)
        conv = [{"role": role, "content": content}]
        encode_id = tokenizer.apply_chat_template(conv)
        input_id += encode_id
        if role in ["user", "system"]:
            target += [IGNORE_INDEX] * len(encode_id)
        else:
            target_mask = encode_id.copy()
            target_mask[:3] = [IGNORE_INDEX] * 3
            target += target_mask
    return input_id, target


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
    audio_lengths = None,
    merge_size=2,
) -> Dict:
    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    input_ids, targets, chosen_ids, chosen_targets, reject_ids, reject_targets = [], [], [], [], [], []

    is_dpo_data = False
    for i, source in enumerate(sources):
        try:
            if source[0]["from"] != "human":
                source = source[1:]
        except:
            print(sources)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]
            if role in ["chosen", "reject"]:
                is_dpo_data = True
                break
        
        input_id, target = generate_id_target(source, grid_thw_image, grid_thw_video, audio_lengths, tokenizer, "gpt", merge_size)
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

        if is_dpo_data:
            chosen_id, chosen_target = generate_id_target(source, grid_thw_image, grid_thw_video, audio_lengths, tokenizer, "chosen", merge_size)
            reject_id, reject_target = generate_id_target(source, grid_thw_image, grid_thw_video, audio_lengths, tokenizer, "reject", merge_size)

            assert len(chosen_id) == len(chosen_target), f"{len(chosen_id)}!= {len(chosen_target)}"
            assert len(reject_id) == len(reject_target), f"{len(reject_id)}!= {len(reject_target)}"
            chosen_ids.append(chosen_id)
            chosen_targets.append(chosen_target)
            reject_ids.append(reject_id)
            reject_targets.append(reject_target)


    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    if is_dpo_data:
        chosen_ids = torch.tensor(chosen_ids, dtype=torch.long)
        chosen_targets = torch.tensor(chosen_targets, dtype=torch.long)
        reject_ids = torch.tensor(reject_ids, dtype=torch.long)
        reject_targets = torch.tensor(reject_targets, dtype=torch.long)
    else:
        chosen_ids = None
        chosen_targets = None
        reject_ids = None
        reject_targets = None

    
    return dict(
        input_ids=input_ids,
        labels=targets,
        chosen_ids=chosen_ids,
        chosen_labels=chosen_targets,
        reject_ids=reject_ids,
        reject_labels=reject_targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = dataset
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data.split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data)
            else:
                annotations = json.load(open(data, "r"))
            list_data_dict += annotations

        for d in list_data_dict:
            if "<image>" in d["conversations"][0]["value"] and not "image" in d and "video" in d:
                d["conversations"][0]["value"] = d["conversations"][0]["value"].replace(
                    "<image>", "<video>"
                )
            if "<image>" in d["conversations"][0]["value"] and not "image" in d and not "video" in d and "audio" in d:
                d["conversations"][0]["value"] = d["conversations"][0]["value"].replace(
                    "<image>", "<audio>"
                )

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_audio(self, audio_file):
        try:
            audio_kwargs = {
                "sampling_rate": 16000,
                "padding": "max_length",
                "return_attention_mask": False,
            }
            processor = copy.deepcopy(self.data_args.audio_processor)
            if isinstance(audio_file, list):
                audio_data = []
                for file in audio_file:
                    decoder = AudioDecoder(
                        file,
                        sample_rate=audio_kwargs["sampling_rate"],
                        num_channels=1,
                    )
                    audio = decoder.get_all_samples()
                    audio_data.append(audio.data.numpy().squeeze(0))
            else:
                decoder = AudioDecoder(
                    audio_file,
                    sample_rate=audio_kwargs["sampling_rate"],
                    num_channels=1,
                )
                audio = decoder.get_all_samples()
                audio_data = [audio.data.numpy().squeeze(0)]
            audio_inputs = []
            audio_lengths = []
            for idx in range(len(audio_data)):
                if audio_data[idx].shape[0] < audio_kwargs["sampling_rate"]:
                    padding = audio_kwargs["sampling_rate"] - audio_data[idx].shape[0]
                    audio_data[idx] = np.pad(audio_data[idx], (0, padding), mode="constant", constant_values=0)
                audio_lst = [audio_data[idx][k: k + 30 * audio_kwargs["sampling_rate"]] for k in range(0, len(audio_data[idx]), 30 * audio_kwargs["sampling_rate"])]
                spectrogram_lst = [processor(a, sampling_rate=audio_kwargs["sampling_rate"], return_tensors="pt")["input_features"].squeeze() for a in audio_lst]
                audio_inputs.append(torch.stack(spectrogram_lst, dim=0))
                audio_lengths.append(math.ceil(len(audio_data[idx]) / (30 * audio_kwargs["sampling_rate"])) * 60)
            return audio_inputs, audio_lengths
        except:
            return None, None


    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        torchcodec_video = None
        try:
            torchcodec_video = self.video_torchcodec(video_file)
            return torchcodec_video
        except:
            try:
                decord_video = self.read_video_decord(video_file)
                return decord_video
            except Exception as e:
                print(f"torchcodec attempt failed: {e}")

    def read_video_decord(self, video_file):
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        ori_fps = vr.get_avg_fps()
        interval = getattr(self.data_args, "base_interval", 4)
        avg_fps = max(round(ori_fps * interval), 1)
        video_length = total_frame_num / ori_fps
        
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)
        frame_idx = [k for k in range(0, total_frame_num, round(avg_fps))]
        if len(frame_idx) > video_max_frames:
            frame_idx = np.linspace(0, total_frame_num - 1, video_max_frames, dtype=int).tolist()
        video = vr.get_batch(frame_idx).asnumpy().transpose(0, 3, 1, 2)
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)
        new_pixel = self.data_args.video_max_frame_pixels
        if len(frame_idx) < video_max_frames:
            new_pixel = 0.95 * video_max_frames / len(frame_idx) * new_pixel
        processor.max_pixels = new_pixel
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        if self.data_args.run_test:
            item_to_return = self.__getitem__(random.randint(0, len(self) - 1))
            item_to_return["should_use"] = False
            return item_to_return
        else:
            print(f"Failed to fetch sample {i}. Try another sample.")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

            # define some variables
            grid_thw_merged = None
            video_grid_thw_merged = None
            grid_thw = None
            video_grid_thw = None
            second_per_grid_ts = None
            audio = None
            audio_lengths = None

            if "image" in sources[0]:
                image_file = self.list_data_dict[i]["image"]
                if isinstance(image_file, List):
                    if len(image_file) > 1:
                        image_file = [
                            file for file in image_file
                        ]
                        results = [self.process_image_unified(file) for file in image_file]
                        image, grid_thw = zip(*results)
                    else:
                        image_file = image_file[0]
                        image, grid_thw = self.process_image_unified(image_file)
                        image = [image]
                else:
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
                grid_thw_merged = copy.deepcopy(grid_thw)
                if not isinstance(grid_thw, Sequence):
                    grid_thw_merged = [grid_thw_merged]
                    grid_thw = [grid_thw]
                grid_thw_merged = [
                    merged_thw.prod() // self.data_args.image_processor.merge_size**2
                    for merged_thw in grid_thw_merged
                ]
            if "video" in sources[0]:
                video_file = sources[0]["video"]
                if isinstance(video_file, List):
                    if len(video_file) > 1:
                        video_file = [
                            file for file in video_file
                        ]
                        results = [self.process_video(file) for file in video_file]
                        video, video_grid_thw, second_per_grid_ts = zip(*results)
                    else:
                        video_file = video_file[0]
                        video, video_grid_thw, second_per_grid_ts = self.process_video(
                            video_file
                        )
                        video = [video]
                else:
                    video, video_grid_thw, second_per_grid_ts = self.process_video(
                        video_file
                    )
                    video = [video]
                if "use_audio" in sources[0] and sources[0]["use_audio"]:
                    audio, audio_lengths = self.process_audio(
                        video_file
                    )
                else:
                    audio, audio_lengths = None, None
                video_grid_thw_merged = copy.deepcopy(video_grid_thw)
                if not isinstance(video_grid_thw, Sequence):
                    video_grid_thw_merged = [video_grid_thw_merged]
                    video_grid_thw = [video_grid_thw]
                # video_grid_thw_merged = [
                #     merged_thw.prod() // self.data_args.image_processor.merge_size**2
                #     for merged_thw in video_grid_thw_merged
                # ]
            if "audio" in sources[0]:
                audio_file = sources[0]["audio"]
                audio, audio_lengths = self.process_audio(
                    audio_file
                )
            chat_sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                chat_sources,
                self.tokenizer,
                grid_thw_image=grid_thw_merged if grid_thw_merged else None,
                grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
                audio_lengths=audio_lengths if audio_lengths else None,
                merge_size=self.data_args.image_processor.merge_size,
            )
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
                video_grid_thw=(
                    torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
                ),
                second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
                audio_lengths=audio_lengths if audio_lengths else None,
            )
            if data_dict["chosen_ids"] is not None:
                chosen_position_ids, _ = self.get_rope_index(
                    self.data_args.image_processor.merge_size,
                    data_dict["chosen_ids"],
                    image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
                    video_grid_thw=(
                        torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
                    ),
                    second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
                    audio_lengths=audio_lengths if audio_lengths else None,
                )
            else:
                chosen_position_ids = None
            if data_dict["reject_ids"] is not None:
                reject_position_ids, _ = self.get_rope_index(
                    self.data_args.image_processor.merge_size,
                    data_dict["reject_ids"],
                    image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
                    video_grid_thw=(
                        torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
                    ),
                    second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
                    audio_lengths=audio_lengths if audio_lengths else None,
                )
            else:
                reject_position_ids = None
            if "image" not in sources[0] and "video" not in sources[0] and "audio" not in sources[0]:
                grid_thw_merged = None
                sources = copy.deepcopy([e["conversations"] for e in sources])
                data_dict = preprocess_qwen_2_visual(
                    sources, self.tokenizer, None, None
                )
                position_ids = (
                    torch.arange(0, data_dict["input_ids"].size(1))
                    .view(1, -1)
                    .unsqueeze(0)
                    .expand(3, -1, -1)
                )

            data_dict["position_ids"] = position_ids
            data_dict["chosen_position_ids"] = chosen_position_ids
            data_dict["reject_position_ids"] = reject_position_ids
            data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]
            if data_dict["chosen_ids"] is not None:
                data_dict["chosen_attention_mask"] = [
                    data_dict["chosen_ids"][0].size(0)
                ]
            if data_dict["reject_ids"] is not None:
                data_dict["reject_attention_mask"] = [
                    data_dict["reject_ids"][0].size(0)
                ]

            if "image" in self.list_data_dict[i]:
                data_dict["pixel_values"] = torch.cat(image, dim=0)
                data_dict["image_grid_thw"] = torch.cat(
                    [thw.unsqueeze(0) for thw in grid_thw], dim=0
                )
            # video exist in the data
            elif "video" in self.list_data_dict[i]:
                data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
                data_dict["video_grid_thw"] = torch.cat(
                    [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
                )
            if audio is not None:
                audio = torch.cat(audio, dim=0)
            data_dict["audio_feature"] = audio
            data_dict["audio_lengths"] = audio_lengths
            if data_dict["chosen_ids"] is None and self.data_args.train_type != "grpo":
                data_dict["train_type"] = "sft"
            else:
                data_dict["train_type"] = self.data_args.train_type
            
            if self.data_args.run_test:
                labels = data_dict.pop("labels", None)
                len_input = sum(labels[0] == IGNORE_INDEX)
                data_dict["input_ids"] = data_dict["input_ids"][:, :len_input]
                data_dict["position_ids"] = data_dict["position_ids"][:, :, :len_input]
                data_dict["attention_mask"] = torch.ones_like(data_dict["input_ids"])

                data_dict["video"] = sources[0].get("video", None)
                data_dict["image"] = sources[0].get("image", None)
                data_dict["audio"] = sources[0].get("audio", None)
                data_dict["use_audio"] = sources[0].get("use_audio", False)

                data_dict["prompt"] = sources[0]["conversations"][0]
                data_dict["ref"] = sources[0]["conversations"][1]["value"]
                data_dict["should_use"] = sources[0].get("should_use", True)
                data_dict.pop("chosen_ids", None)
                data_dict.pop("reject_ids", None)
                data_dict.pop("chosen_position_ids", None)
                data_dict.pop("reject_position_ids", None)
                data_dict.pop("chosen_labels", None)
                data_dict.pop("reject_labels", None)
                data_dict.pop("audio_lengths", None)

            return data_dict
        except Exception as e:
            print(f"Error: {e}, line: {e.__traceback__.tb_lineno}")
            raise e



def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def process_ids(self, input_ids, labels, position_ids):
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        return input_ids, labels, position_ids, attention_mask

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids, labels, position_ids, attention_mask = self.process_ids(
            input_ids, labels, position_ids
        )
        chosen_ids, chosen_labels, chosen_position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("chosen_ids", "chosen_labels", "chosen_position_ids")
        )
        if chosen_ids[0] is not None:
            chosen_ids, chosen_labels, chosen_position_ids, chosen_attention_mask = self.process_ids(
                chosen_ids, chosen_labels, chosen_position_ids
            )
        else:
            chosen_ids, chosen_labels, chosen_position_ids, chosen_attention_mask = None, None, None, None
        reject_ids, reject_labels, reject_position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("reject_ids", "reject_labels", "reject_position_ids")
        )
        if reject_ids[0] is not None:
            reject_ids, reject_labels, reject_position_ids, reject_attention_mask = self.process_ids(
                reject_ids, reject_labels, reject_position_ids
            )
        else:
            reject_ids, reject_labels, reject_position_ids, reject_attention_mask = None, None, None, None
        train_type = [instance["train_type"] for instance in instances][0]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            position_ids=position_ids,
            chosen_ids=chosen_ids,
            chosen_labels=chosen_labels,
            chosen_position_ids=chosen_position_ids,
            reject_ids=reject_ids,
            reject_labels=reject_labels,
            reject_position_ids=reject_position_ids,
            attention_mask=attention_mask,
            chosen_attention_mask=chosen_attention_mask,
            reject_attention_mask=reject_attention_mask,
            train_type=train_type,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        audios = list(
            instance["audio_feature"]
            for instance in instances
            if instance["audio_feature"] is not None
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        if len(audios)!= 0:
            concat_audios = torch.cat([audio for audio in audios], dim=0)
            audio_lengths = [
                instance["audio_lengths"]
                for instance in instances
                if "audio_lengths" in instance
            ]
            audio_lengths = [l for length in audio_lengths for l in length]
        else:
            concat_audios = None
            audio_lengths = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["audio_feature"] = concat_audios
        batch["audio_lengths"] = audio_lengths
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass