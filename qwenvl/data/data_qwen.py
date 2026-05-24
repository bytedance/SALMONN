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
from typing import Dict, Optional, Sequence, List, Tuple, Union
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchcodec.decoders import VideoDecoder, AudioDecoder
import transformers

from decord import VideoReader, cpu

import bytedtos

from joblib import Parallel, delayed, cpu_count

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

def split_into_groups(counts, groups, second_per_grid_ts=None):
    result = []
    if second_per_grid_ts is None:
        for count, g in zip(counts, groups):
            g = g.item()
            base = count // g
            remainder = count % g
            if remainder == 0:
                group_list = [base] * g
            else:
                group_list = [base] * g
                step = g / remainder
                for i in range(1, remainder + 1):
                    position = i * step
                    index = math.floor(position) - 1
                    if index >= g:
                        index = g - 1
                    group_list[index] += 1
            result.append(group_list)
    else:
        for count, g, second in zip(counts, groups, second_per_grid_ts):
            g = g.item()
            frame_idx = (torch.arange(g) * second * 2).long()
            per_grid_t = torch.diff(frame_idx)
            group_list = per_grid_t.tolist()
            group_list.append(count - sum(group_list))
            assert sum(group_list) == count, f"Strage count under count={count}, g={g}, second={second}"
            result.append(group_list)
    return result

def _calculate_timestamps(indices: Union[list[int], np.ndarray], video_fps: float, merge_size: int = 2):
    if not isinstance(indices, list):
        indices = indices.tolist()
    if len(indices) % merge_size != 0:
        indices.extend(indices[-1] for _ in range(merge_size - len(indices) % merge_size))
    timestamps = [idx / video_fps for idx in indices]
    # @JJJYmmm frames are merged by self.merge_size, \
    # so we need to average the timestamps between the first/last frame within the temporal patch
    timestamps = [
        (timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)
    ]
    return timestamps

def split_indices(indices: List[int], num_chunks: int) -> List[List[int]]:
    """Split a list of indices into approximately equal chunks."""
    chunk_size = len(indices) // num_chunks
    chunks = []

    for i in range(num_chunks - 1):
        chunks.append(indices[i * chunk_size:(i + 1) * chunk_size])

    # Last chunk may be slightly larger
    chunks.append(indices[(num_chunks - 1) * chunk_size:])
    return chunks

def decode_sequentially(indices: List[int], video_path):
    """Decode frames sequentially using a single decoder instance."""
    decoder = VideoDecoder(video_path)
    return decoder.get_frames_at(indices)

def decode_with_multithreading(
    indices: List[int],
    num_threads: int,
    video_path
):
    """Decode frames using multiple threads with joblib."""
    chunks = split_indices(indices, num_chunks=num_threads)

    results = Parallel(n_jobs=num_threads, prefer="threads", verbose=0)(
        delayed(decode_sequentially)(chunk, video_path) for chunk in chunks
    )

    # Concatenate results from all threads
    return torch.cat([frame_batch.data for frame_batch in results], dim=0)

def generate_id_target(
    source,
    grid_thw_image, 
    grid_thw_video, 
    audio_lengths, 
    tokenizer, 
    target_role,
    merge_size: int = 2,
    second_per_grid_ts: List = [],
    fps: List = [],
    frame_idx: List = []
):
    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    roles = {"human": "user", "gpt": "assistant", "chosen": "assistant", "reject": "assistant"}
    system_message = "You are a helpful assistant."
    input_id, target = [], []

    # input_id += tokenizer.apply_chat_template(
    #     [{"role": "system", "content": system_message}]
    # )
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
                frame_seq_length = [
                    merged_thw[1:].prod() // merge_size**2
                    for merged_thw in grid_thw_video
                ]
                if audio_lengths is None:
                    for i in range(len(parts) - 1):
                        curr_timestamp = _calculate_timestamps(
                            frame_idx[i],
                            fps[i],
                            merge_size,
                        )
                        new_parts.append(parts[i])
                        replacement = ""
                        for idx in range(grid_thw_video[i][0]):
                            curr_time = curr_timestamp[idx]
                            replacement += f"<{curr_time:.1f} seconds>"
                            replacement += "<|vision_start|>"
                            replacement += "<|video_pad|>" * frame_seq_length[i]
                            replacement += "<|vision_end|>"
                        new_parts.append(replacement)
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)
                else:
                    for i in range(len(parts) - 1):
                        curr_timestamp = _calculate_timestamps(
                            frame_idx[i],
                            fps[i],
                            merge_size,
                        )
                        new_parts.append(parts[i])
                        if second_per_grid_ts is None:
                            per_timestep_audio_len = split_into_groups(audio_lengths, [grid_thw_video[i][0] for i in range(len(grid_thw_video))])
                        else:
                            per_timestep_audio_len = split_into_groups(audio_lengths, [grid_thw_video[i][0] for i in range(len(grid_thw_video))], [ts[0] for ts in second_per_grid_ts])
                        replacement = ""
                        # print(len(curr_timestamp), grid_thw_video)
                        total_timestamps = 6000
                        in_idx = None
                        if grid_thw_video[i][0] > total_timestamps:
                            in_idx = np.unique(np.linspace(0, grid_thw_video[i][0]-1, total_timestamps, dtype=int))

                        for idx in range(grid_thw_video[i][0]):
                            if in_idx is None or idx in in_idx:
                                curr_time = curr_timestamp[idx]
                                replacement += f"<{curr_time:.1f} seconds>"
                                replacement += "<|vision_start|>"
                            replacement += "<|video_pad|>" * frame_seq_length[i]
                            replacement += f"<|audio_pad|>" * per_timestep_audio_len[i][idx]
                            if in_idx is None or idx in in_idx:
                                replacement += "<|vision_end|>"
                        new_parts.append(replacement)
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)
                            

            if "<audio>" in content:
                parts = content.split("<audio>")
                new_parts = []
                for i in range(len(parts) - 1):
                    new_parts.append(parts[i])
                    replacement = f"<|audio_pad|>" * audio_lengths[i] # remove vision_start for minimum change on rope index
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


def preprocess_qwen_3_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
    audio_lengths = None,
    merge_size=2,
    second_per_grid_ts: List = [],
    fps: List = [],
    frame_idx: List = [],
) -> Dict:
    if second_per_grid_ts is not None and isinstance(second_per_grid_ts, list) and not isinstance(second_per_grid_ts[0], list):
        second_per_grid_ts = [second_per_grid_ts]
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
        
        input_id, target = generate_id_target(source, grid_thw_image, grid_thw_video, audio_lengths, tokenizer, "gpt", merge_size, second_per_grid_ts, fps, frame_idx)
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

        if is_dpo_data:
            chosen_id, chosen_target = generate_id_target(source, grid_thw_image, grid_thw_video, audio_lengths, tokenizer, "chosen", merge_size, second_per_grid_ts, fps, frame_idx)
            reject_id, reject_target = generate_id_target(source, grid_thw_image, grid_thw_video, audio_lengths, tokenizer, "reject", merge_size, second_per_grid_ts, fps, frame_idx)

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

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args, validation=False):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",") if not validation else [data_args.validation_data]
        dataset_list = dataset
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type

        list_data_dict = []

        for data in dataset_list:
            file_format = data.split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data)
            else:
                annotations = json.load(open(data, "r"))
            list_data_dict += annotations

        for d in list_data_dict:
            if "<image>" in d["conversations"][0]["value"] and not "image" in d and ("video" in d or "tos_key" in d):
                d["conversations"][0]["value"] = d["conversations"][0]["value"].replace(
                    "<image>", "<video>"
                )
            if "<image>" in d["conversations"][0]["value"] and not "image" in d and not "video" in d and ("audio" in d or "tos_audio" in d):
                d["conversations"][0]["value"] = d["conversations"][0]["value"].replace(
                    "<image>", "<audio>"
                )
            if ("video" in d or "tos_key" in d) and (not "<image>" in d["conversations"][0]["value"]) and (not "<video>" in d["conversations"][0]["value"]):
                d["conversations"][0]["value"] = "<video>\n" + d["conversations"][0]["value"]
            if "batch_audio" in d:
                for idx in range(len(d["batch_audio"])):
                    if "<image>" in d["batch_audio"][idx]["conversations"][0]["value"]:
                        d["batch_audio"][idx]["conversations"][0]["value"] = d["batch_audio"][idx]["conversations"][0]["value"].replace(
                            "<image>", "<audio>"
                        )
            if "timestamps" in d:
                d["conversations"][0]["timestamps"] = d["timestamps"]

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        # random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.video_processor.max_pixels = data_args.video_max_frame_pixels
        self.data_args.video_processor.min_pixels = data_args.video_min_frame_pixels
        self.data_args.video_processor.size["longest_edge"] = data_args.video_max_frame_pixels
        self.data_args.video_processor.size["shortest_edge"] = data_args.video_min_frame_pixels
        self.data_args.video_processor.do_sample_frames = False

        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        self.type_list = ["av" if (("video" in sample or "tos_key" in sample) and "use_audio" in sample and sample["use_audio"]) else "v" if ("video" in sample or "tos_key" in sample) else "a" if ("audio" in sample or "tos_audio" in sample or "batch_audio" in sample) else "t" for sample in self.list_data_dict]

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
                cur_len if ("image" in sample) or ("video" in sample) or ("tos_key" in sample) else -cur_len
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

    def process_audio(self, audio_file, timestamps=None):
        try:
            audio_kwargs = {
                "sampling_rate": 16000,
                "padding": "max_length",
                "return_attention_mask": False,
            }
            processor = copy.deepcopy(self.data_args.audio_processor)
            if isinstance(audio_file, list):
                audio_data = []
                if timestamps is None:
                    for file in audio_file:
                        decoder = AudioDecoder(
                            file,
                            sample_rate=audio_kwargs["sampling_rate"],
                            num_channels=1,
                        )
                        audio = decoder.get_all_samples()
                        audio_data.append(audio.data.numpy().squeeze(0))
                else:
                    for file, ts in zip(audio_file, timestamps):
                        decoder = AudioDecoder(
                            file,
                            sample_rate=audio_kwargs["sampling_rate"],
                            num_channels=1,
                        )
                        audio = decoder.get_samples_played_in_range(ts[0], ts[1])
                        audio_data.append(audio.data.numpy().squeeze(0))
            else:
                decoder = AudioDecoder(
                    audio_file,
                    sample_rate=audio_kwargs["sampling_rate"],
                    num_channels=1,
                )
                if timestamps is None:
                    audio = decoder.get_all_samples()
                else:
                    audio = decoder.get_samples_played_in_range(timestamps[0], timestamps[1])
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
                assert audio_lengths[-1] % 60 == 0, "Wierd audio lengths"
            return audio_inputs, audio_lengths
        except:
            raise ValueError(f"Error processing audio file {audio_file}")
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

    def process_video(self, video_file, timestamps=None, video_max_frames=0):
        torchcodec_video = None
        try:
            torchcodec_video = self.video_torchcodec(video_file, timestamps, video_max_frames=video_max_frames)
            return torchcodec_video
        except:
            try:
                decord_video = self.read_video_decord(video_file, timestamps, video_max_frames=video_max_frames)
                return decord_video
            except Exception as e:
                print(f"torchcodec attempt failed: {e}")
                raise e

    def read_video_decord(self, video_file, timestamps, video_max_frames=0):
        vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        avg_fps = vr.get_avg_fps()
        interval = getattr(self.data_args, "base_interval", 4)
        video_length = total_frame_num / avg_fps
        start_idx = 0
        end_idx = total_frame_num - 1
        if timestamps is not None:
            timestamps[0] = min(max(timestamps[0], 0), video_length)
            timestamps[1] = min(max(timestamps[1], 0), video_length)
            start_idx = round(timestamps[0] * avg_fps)
            end_idx = round(timestamps[1] * avg_fps)
            start_idx = min(max(start_idx, 0), total_frame_num - 1)
            end_idx = min(max(end_idx, 0), total_frame_num - 1)
            video_length = timestamps[1] - timestamps[0]

        num_frames_to_sample = round(video_length / interval)
        
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        if video_max_frames == 0:
            video_max_frames = getattr(self.data_args, "video_max_frames", 8)
        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(start_idx, end_idx, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy().transpose(0, 3, 1, 2)
        return *self.process_video_frames(video, frame_idx, video_length), avg_fps, frame_idx

    def video_torchcodec(self, video_file, timestamps, video_max_frames=0):
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frame_num = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frame_num / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)
        start_idx = 0
        end_idx = total_frame_num - 1
        if timestamps is not None:
            timestamps[0] = min(max(timestamps[0], 0), video_length)
            timestamps[1] = min(max(timestamps[1], 0), video_length)
            start_idx = round(timestamps[0] * avg_fps)
            end_idx = round(timestamps[1] * avg_fps)
            start_idx = min(max(start_idx, 0), total_frame_num - 1)
            end_idx = min(max(end_idx, 0), total_frame_num - 1)
            video_length = timestamps[1] - timestamps[0]

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        if video_max_frames == 0:
            video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(start_idx, end_idx, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        # frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        frame_batch = decode_with_multithreading(indices=frame_idx.tolist(), num_threads=8, video_path=video_file)
        video = frame_batch.data.cpu().numpy()
        return *self.process_video_frames(video, frame_idx, video_length), avg_fps, frame_idx

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.video_processor)
        # max_pixels = getattr(self.data_args, "video_max_frames", 8) * getattr(self.data_args, "video_max_frame_pixels", 8)
        max_pixels = len(frame_idx) * getattr(self.data_args, "video_max_frame_pixels", 8)
        processor.max_pixels = max_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor(videos=video)
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
                sources = self.list_data_dict[i]
                if isinstance(i, int):
                    sources = [sources]
                assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
                sample = self._get_item(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                # raise e
                time.sleep(1)

        item_to_return = self.__getitem__(random.randint(0, len(self) - 1))
        item_to_return["should_use"] = False
        return item_to_return

    def downsample(self, video, video_grid_thw, second_per_grid_ts, fps, frame_idx):
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)
        video = torch.cat(video, dim=0)
        new_video_grid_thw = video_grid_thw[0]
        new_video_grid_thw[0] = torch.stack(video_grid_thw).sum(dim=0)[0]
        second_per_grid_ts = second_per_grid_ts[0]
        fps = [2 / second_per_grid_ts[0]]
        frame_idx = np.array([i for i in range(new_video_grid_thw[0]*2)])
        frame_idx = frame_idx.astype(int)
        video = video.view(new_video_grid_thw.tolist()+[-1])
        if len(frame_idx) > video_max_frames:
            downsample_frame_ids = np.unique(np.linspace(0, len(frame_idx)-1, video_max_frames, dtype=int))
            downsample_video_ids = np.unique(np.linspace(0, new_video_grid_thw[0]-1, video_max_frames//2, dtype=int))
            downsample_ratio = len(frame_idx) / video_max_frames
            video = video[downsample_video_ids]
            new_video_grid_thw[0] = video_max_frames // 2
            second_per_grid_ts = [s * downsample_ratio for s in second_per_grid_ts]
            frame_idx = frame_idx[downsample_frame_ids]
        video = video.view(-1, video.size(-1))
        return video, new_video_grid_thw, second_per_grid_ts, fps, frame_idx

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        try:
            if "batch_audio" in sources[0]:
                # return [
                #     self._get_item([source]) for source in sources[0]["batch_audio"]
                # ]
                return Parallel(n_jobs=8, prefer="threads", verbose=0)(
                    delayed(self._get_item)([chunk]) for chunk in sources[0]["batch_audio"]
                )
            # define some variables
            grid_thw_merged = None
            video_grid_thw_merged = None
            grid_thw = None
            video_grid_thw = None
            second_per_grid_ts = None
            audio = None
            audio_lengths = None
            fps=None
            frame_idx = None

            if "image" in sources[0]:
                image_file = sources[0]["image"]
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
            if "video" in sources[0] or "tos_key" in sources[0]:
                timestamps = sources[0]["conversations"][0].get("timestamps", None)
                if timestamps is None:
                    timestamps = sources[0].get("timestamps", None)
                if "tos_key" in sources[0]:
                    video_file = sources[0]["tos_key"]
                    if isinstance(video_file, List):
                        if len(video_file) > 1:
                            video_file = [
                                file for file in video_file
                            ]
                            if timestamps is not None:
                                results = [self.process_video_tos(file, ts) for file, ts in zip(video_file, timestamps)]
                            else:
                                results = [self.process_video_tos(file, video_max_frames=100000) for file in video_file]
                            video, video_grid_thw, second_per_grid_ts, fps, frame_idx = zip(*results)
                            video, video_grid_thw, second_per_grid_ts, fps, frame_idx = self.downsample(
                                video, video_grid_thw, second_per_grid_ts, fps, frame_idx
                            )
                            video = [video]
                            frame_idx = [frame_idx]
                        else:
                            video_file = video_file[0]
                            video, video_grid_thw, second_per_grid_ts, fps, frame_idx = self.process_video_tos(
                                video_file, timestamps
                            )
                            video = [video]
                            fps = [fps]
                            frame_idx = [frame_idx]
                    else:
                        video, video_grid_thw, second_per_grid_ts, fps, frame_idx = self.process_video_tos(
                            video_file, timestamps
                        )
                        video = [video]
                        fps = [fps]
                        frame_idx = [frame_idx]
                    if "use_audio" in sources[0] and sources[0]["use_audio"]:
                        audio, audio_lengths = self.process_audio_tos(
                            video_file, timestamps
                        )
                        if len(audio) > 1:
                            audio = [torch.cat(audio, dim=0)]
                            audio_lengths = [sum(audio_lengths)]
                    else:
                        audio, audio_lengths = None, None
                elif "video" in sources[0]:
                    video_file = sources[0]["video"]
                    if isinstance(video_file, List):
                        if len(video_file) > 1:
                            video_file = [
                                file for file in video_file
                            ]
                            if timestamps is not None:
                                results = [self.process_video(file, ts) for file, ts in zip(video_file, timestamps)]
                            else:
                                results = [self.process_video(file, video_max_frames=10000) for file in video_file]
                            video, video_grid_thw, second_per_grid_ts, fps, frame_idx = zip(*results)
                            video, video_grid_thw, second_per_grid_ts, fps, frame_idx = self.downsample(
                                video, video_grid_thw, second_per_grid_ts, fps, frame_idx
                            )
                            video = [video]
                            frame_idx = [frame_idx]
                        else:
                            video_file = video_file[0]
                            video, video_grid_thw, second_per_grid_ts, fps, frame_idx = self.process_video(
                                video_file, timestamps
                            )
                            video = [video]
                            fps = [fps]
                            frame_idx = [frame_idx]
                    else:
                        video, video_grid_thw, second_per_grid_ts, fps, frame_idx = self.process_video(
                            video_file, timestamps
                        )
                        video = [video]
                        fps = [fps]
                        frame_idx = [frame_idx]
                    if "use_audio" in sources[0] and sources[0]["use_audio"]:
                        audio, audio_lengths = self.process_audio(
                            video_file, timestamps
                        )
                        if len(audio) > 1:
                            audio = [torch.cat(audio, dim=0)]
                            audio_lengths = [sum(audio_lengths)]
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
            elif "audio" in sources[0]:
                audio_file = sources[0]["audio"]
                audio, audio_lengths = self.process_audio(
                    audio_file
                )
            elif "tos_audio" in sources[0]:
                audio_file = sources[0]["tos_audio"]
                audio, audio_lengths = self.process_audio_tos(
                    audio_file
                )
            chat_sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_3_visual(
                chat_sources,
                self.tokenizer,
                grid_thw_image=grid_thw_merged if grid_thw_merged else None,
                grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
                audio_lengths=audio_lengths if audio_lengths else None,
                merge_size=self.data_args.image_processor.merge_size,
                second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
                fps=fps if fps else None,
                frame_idx=frame_idx if frame_idx else None,
            )
            debug_input_ids = data_dict["input_ids"]
            audio_token_number = (debug_input_ids == 151669).sum()
            assert audio_token_number % 60 == 0, f"Weird data: {sources}"
            position_ids = None
            chosen_position_ids = None
            reject_position_ids = None
            if "image" not in sources[0] and "video" not in sources[0] and "tos_key" not in sources[0] and "audio" not in sources[0] and "tos_audio" not in sources[0]:
                grid_thw_merged = None
                sources = copy.deepcopy([e["conversations"] for e in sources])
                data_dict = preprocess_qwen_3_visual(
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

            if "image" in sources[0]:
                data_dict["pixel_values"] = torch.cat(image, dim=0)
                data_dict["image_grid_thw"] = torch.cat(
                    [thw.unsqueeze(0) for thw in grid_thw], dim=0
                )
            # video exist in the data
            elif "video" in sources[0] or "tos_key" in sources[0]:
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

            data_dict["prompt"] = sources[0]["conversations"][0]
            data_dict["ref_answer"] = sources[0]["ref_answer"] if "ref_answer" in sources[0] else sources[0]["conversations"][1]["value"]
            labels = data_dict.pop("labels", None)
            len_input = sum(labels[0] == IGNORE_INDEX)
            data_dict["input_ids"] = data_dict["input_ids"][:, :len_input]
            # data_dict["position_ids"] = data_dict["position_ids"][:, :, :len_input]
            data_dict["attention_mask"] = torch.ones_like(data_dict["input_ids"])

            data_dict["tos_key"] = sources[0].get("tos_key", None)
            data_dict["video"] = sources[0].get("video", None)
            data_dict["image"] = sources[0].get("image", None)
            data_dict["tos_audio"] = sources[0].get("tos_audio", None)
            data_dict["audio"] = sources[0].get("audio", None)
            data_dict["use_audio"] = sources[0].get("use_audio", False)

            # data_dict["prompt"] = sources[0]["conversations"][0]
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



# def pad_and_cat(tensor_list):
#     max_length = max(tensor.shape[2] for tensor in tensor_list)

#     padded_tensors = []
#     for tensor in tensor_list:
#         pad_length = max_length - tensor.shape[2]
#         padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
#         padded_tensors.append(padded_tensor)

#     stacked_tensor = torch.cat(padded_tensors, dim=1)

#     return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def process_ids(self, input_ids, labels):
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        # input_ids = input_ids[:, : self.tokenizer.model_max_length]
        # labels = labels[:, : self.tokenizer.model_max_length]
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        return input_ids, labels, attention_mask

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list) and len(instances) == 1:
            instances = instances[0]
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids, labels, attention_mask = self.process_ids(
            input_ids, labels
        )
        # gs534 - prompt and ref_answer for generation
        prompts = [instance["prompt"] for instance in instances]
        ref_answer = [instance["ref_answer"] for instance in instances]
        chosen_ids, chosen_labels, chosen_position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("chosen_ids", "chosen_labels", "chosen_position_ids")
        )
        if chosen_ids[0] is not None:
            chosen_ids, chosen_labels, chosen_attention_mask = self.process_ids(
                chosen_ids, chosen_labels
            )
        else:
            chosen_ids, chosen_labels, chosen_attention_mask = None, None, None
        reject_ids, reject_labels, reject_position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("reject_ids", "reject_labels", "reject_position_ids")
        )
        if reject_ids[0] is not None:
            reject_ids, reject_labels, reject_attention_mask = self.process_ids(
                reject_ids, reject_labels
            )
        else:
            reject_ids, reject_labels, reject_attention_mask = None, None, None
        train_type = [instance["train_type"] for instance in instances][0]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            chosen_ids=chosen_ids,
            chosen_labels=chosen_labels,
            reject_ids=reject_ids,
            reject_labels=reject_labels,
            attention_mask=attention_mask,
            chosen_attention_mask=chosen_attention_mask,
            reject_attention_mask=reject_attention_mask,
            train_type=train_type,
            prompts=prompts,
            ref_answer=ref_answer,
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
    if data_args.validation_data != "":
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args, validation=True)
    else:
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator
    )


if __name__ == "__main__":
    pass