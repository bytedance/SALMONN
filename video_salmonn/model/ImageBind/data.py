#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import logging

from .models.multimodal_preprocessors import SimpleTokenizer
from PIL import Image
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler, UniformClipSampler
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo
from torchvision.transforms.functional import InterpolationMode

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds

BPE_PATH = "bpe/bpe_simple_vocab_16e6.txt.gz"


def waveform2melspec(waveform, sample_rate, num_mel_bins, target_length):
    # Based on https://github.com/YuanGongND/ast/blob/d7d8b4b8e06cdaeb6c843cdb38794c1c7692234c/src/dataloader.py#L102
    waveform -= waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length=25,
        frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
    )
    # Convert to [mel_bins, num_frames] shape
    fbank = fbank.transpose(0, 1)
    # Pad to target_length
    n_frames = fbank.size(1)
    p = target_length - n_frames
    # if p is too large (say >20%), flash a warning
    if abs(p) / n_frames > 0.2:
        logging.warning(
            "Large gap between audio n_frames(%d) and "
            "target_length (%d). Is the audio_target_length "
            "setting correct?",
            n_frames,
            target_length,
        )
    # cut and pad
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, p), mode="constant", value=0)
    elif p < 0:
        fbank = fbank[:, 0:target_length]
    # Convert to [1, mel_bins, num_frames] shape, essentially like a 1
    # channel image
    fbank = fbank.unsqueeze(0)
    return fbank


def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


def load_and_transform_vision_data(image_paths, device):
    if image_paths is None:
        return None

    image_ouputs = []
    for image_path in image_paths:
        data_transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        image = data_transform(image).to(device)
        image_ouputs.append(image)
    return torch.stack(image_ouputs, dim=0)


def load_and_transform_vision_data_blip(image_paths, device, training=False, hi_rs=False, hi_rs_cfg=None):
    if image_paths is None:
        return None

    if training and not hi_rs:
        data_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.5, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        data_transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    image_ouputs = []
    for image_path in image_paths:
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")

        if hi_rs:
            if hi_rs_cfg is None:
                hi_rs_cfg = (4, 1)
            n_split, dup = hi_rs_cfg
            width, height = image.size
            image_blocks = []
            for _ in range(dup):
                image_blocks.append(
                    data_transform(image).to(device).unsqueeze(0)
                )

            dx = width // n_split * 2
            dy = height // n_split * 2
            for ny in range(n_split - 1):
                for nx in range(n_split - 1):
                    x = width // n_split * nx
                    y = height // n_split * ny
                    box = (x, y, x + dx, y + dy)
                    for _ in range(dup):
                        image_blocks.append(
                            data_transform(
                                image.crop(box)
                            ).to(device).unsqueeze(0)
                        )
                
            # for y in range(0, height, height // 4):
            #     for x in range(0, width, width // 4):
            #         box = (x, y, x + width // 4 * 2, y + height // 4 * 2)
            #         image_blocks.append(
            #             data_transform(
            #                 image.crop(box)
            #             ).to(device).unsqueeze(0)
            #         )
            image_blocks = torch.cat(image_blocks, dim=0)
            image_ouputs.append(image_blocks)
            
        else:
            image = data_transform(image).to(device)
            image_ouputs.append(image)

    if hi_rs:
        image_lens = [img.shape[0] for img in image_ouputs]
        max_image_len = max(image_lens)
        img_mask = torch.arange(max_image_len).unsqueeze(0) < torch.tensor(image_lens).unsqueeze(1)
        return pad_sequence(image_ouputs, batch_first=True), img_mask.to(device)
    else:
        return torch.stack(image_ouputs, dim=0)


def load_and_transform_thermal_data(thermal_paths, device):
    if thermal_paths is None:
        return None

    thermal_ouputs = []
    for thermal_path in thermal_paths:
        data_transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        with open(thermal_path, "rb") as fopen:
            thermal = Image.open(fopen).convert("L")
        thermal = data_transform(thermal).to(device)
        thermal_ouputs.append(thermal)
    return torch.stack(thermal_ouputs, dim=0)


def load_and_transform_text(text, device):
    if text is None:
        return None
    tokenizer = SimpleTokenizer(bpe_path=BPE_PATH)
    tokens = [tokenizer(t).unsqueeze(0).to(device) for t in text]
    tokens = torch.cat(tokens, dim=0)
    return tokens

def load_and_transform_audio_data_fulllen(
    audio_paths,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    mean=-4.268,
    std=9.138,
    maxlen=30,
):
    if audio_paths is None:
        return None

    audio_outputs = []
    # clip_sampler = ConstantClipsPerVideoSampler(
    #     clip_duration=clip_duration, clips_per_video=clips_per_video
    # )

    for audio_path in audio_paths:
        waveform, sr = torchaudio.load(audio_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        full_lengths = waveform.size(1)
        if full_lengths < maxlen * sample_rate:
            diffsize = maxlen * sample_rate - full_lengths - 1
            waveform = torch.cat(
                    [waveform, waveform.new_zeros(waveform.size(0), diffsize)], dim=-1)
        full_lengths = min(waveform.size(1), maxlen * sample_rate)
        all_clips = []
        start = 0
        stepsize = clip_duration * sample_rate
        while start < full_lengths:
            end = min(start + stepsize, full_lengths)
            waveform_clip = waveform[
                :,
                int(start) : int(end),
            ]
            if int(end) - int(start) < stepsize:
                diffsize = stepsize - int(end) + int(start)
                waveform_clip = torch.cat(
                    [waveform_clip, waveform_clip.new_zeros(waveform_clip.size(0), diffsize)], dim=-1)
            waveform_melspec = waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)
            start = start + stepsize

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)
        # for audio in audio_outputs:
        #     if audio.size(0) > 5:
        #         import pdb; pdb.set_trace()

    return torch.stack(audio_outputs, dim=0)


def load_and_transform_audio_data(
    audio_paths,
    device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    if audio_paths is None:
        return None

    audio_outputs = []
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )

    for audio_path in audio_paths:
        waveform, sr = torchaudio.load(audio_path)
        if sample_rate != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr, new_freq=sample_rate
            )
        all_clips_timepoints = get_clip_timepoints(
            clip_sampler, waveform.size(1) / sample_rate
        )
        all_clips = []
        for clip_timepoints in all_clips_timepoints:
            waveform_clip = waveform[
                :,
                int(clip_timepoints[0] * sample_rate) : int(
                    clip_timepoints[1] * sample_rate
                ),
            ]
            waveform_melspec = waveform2melspec(
                waveform_clip, sample_rate, num_mel_bins, target_length
            )
            all_clips.append(waveform_melspec)

        normalize = transforms.Normalize(mean=mean, std=std)
        all_clips = [normalize(ac).to(device) for ac in all_clips]

        all_clips = torch.stack(all_clips, dim=0)
        audio_outputs.append(all_clips)

    return torch.stack(audio_outputs, dim=0)


def get_clip_timepoints(clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = clip_sampler(end, duration, annotation=None)
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


def crop_boxes(boxes, x_offset, y_offset):
    """
    Peform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to peform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
        scale_size (int): optinal. If not None, resize the images to scale_size before
            performing any crop.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    assert spatial_idx in [0, 1, 2]
    ndim = len(images.shape)
    if ndim == 3:
        images = images.unsqueeze(0)
    height = images.shape[2]
    width = images.shape[3]

    if scale_size is not None:
        if width <= height:
            width, height = scale_size, int(height / width * scale_size)
        else:
            width, height = int(width / height * scale_size), scale_size
        images = torch.nn.functional.interpolate(
            images,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    cropped_boxes = crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    if ndim == 3:
        cropped = cropped.squeeze(0)
    return cropped, cropped_boxes


class SpatialCrop(nn.Module):
    """
    Convert the video into 3 smaller clips spatially. Must be used after the
        temporal crops to get spatial crops, and should be used with
        -2 in the spatial crop at the slowfast augmentation stage (so full
        frames are passed in here). Will return a larger list with the
        3x spatial crops as well.
    """

    def __init__(self, crop_size: int = 224, num_crops: int = 3):
        super().__init__()
        self.crop_size = crop_size
        if num_crops == 3:
            self.crops_to_ext = [0, 1, 2]
            self.flipped_crops_to_ext = []
        elif num_crops == 1:
            self.crops_to_ext = [1]
            self.flipped_crops_to_ext = []
        else:
            raise NotImplementedError("Nothing else supported yet")

    def forward(self, videos):
        """
        Args:
            videos: A list of C, T, H, W videos.
        Returns:
            videos: A list with 3x the number of elements. Each video converted
                to C, T, H', W' by spatial cropping.
        """
        assert isinstance(videos, list), "Must be a list of videos after temporal crops"
        assert all([video.ndim == 4 for video in videos]), "Must be (C,T,H,W)"
        res = []
        for video in videos:
            for spatial_idx in self.crops_to_ext:
                res.append(uniform_crop(video, self.crop_size, spatial_idx)[0])
            if not self.flipped_crops_to_ext:
                continue
            flipped_video = transforms.functional.hflip(video)
            for spatial_idx in self.flipped_crops_to_ext:
                res.append(uniform_crop(flipped_video, self.crop_size, spatial_idx)[0])
        return res


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__

def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(
            f"target size should be tuple (height, width), instead got {target_size}"
        )
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode, align_corners=False
    )

class ResizeVideo(object):
    def __init__(self, target_size, interpolation_mode="bilinear"):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(resize_size={0})".format(self.target_size)

def load_and_transform_video_data_full(
    video_paths,
    device,
    clip_duration=1,
    sample_per_clip=2,
    sample_rate=16000,
):
    if video_paths is None:
        return None

    video_outputs = []
    video_transform = transforms.Compose(
        [
            ResizeVideo((224, 224), interpolation_mode="bicubic"),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    clip_sampler = UniformClipSampler(
        clip_duration=clip_duration, backpad_last=True
    )
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=sample_per_clip)

    maxlen = 0
    for video_path in video_paths:
        if not isinstance(video_path, list):
            video = EncodedVideo.from_path(
                video_path,
                decoder="decord",
                decode_audio=False,
                **{"sample_rate": sample_rate},
            )

            all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)

            all_video = []
            for clip_timepoints in all_clips_timepoints:
                # Read the clip, get frames
                clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
                if clip is None:
                    raise ValueError("No clip found")
                video_clip = frame_sampler(clip["video"])
                video_clip = video_clip / 255.0  # since this is float, need 0-1

                all_video.append(video_clip)
        else:
            all_video = video_path

        all_video = [video_transform(clip) for clip in all_video]
        # all_video = SpatialCrop(224, num_crops=3)(all_video)
        if len(all_video) > maxlen:
            maxlen = len(all_video)
        all_video = torch.stack(all_video, dim=0)
        video_outputs.append(all_video)

    padded_video_outputs = []
    padded_video_mask = []
    for video in video_outputs:
        if video.size(0) < maxlen:
            diffsize = maxlen - video.size(0)
            padded_video_mask.append([1] * video.size(0) + [0] * diffsize)
            video = torch.cat([video, video.new_zeros(
                diffsize, video.size(1), video.size(2), video.size(3), video.size(4))], dim=0)
        else:
            padded_video_mask.append([1] * video.size(0))
        padded_video_outputs.append(video)

    return torch.stack(padded_video_outputs, dim=0).to(device), torch.tensor(padded_video_mask).to(device)

def load_and_transform_video_data_blip(
    video_paths,
    device,
    clip_duration=1,
    sample_per_clip=2,
    sample_rate=16000,
):
    if video_paths is None:
        return None

    video_outputs = []
    video_transform = transforms.Compose(
        [
            ResizeVideo((224, 224), interpolation_mode="bicubic"),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    clip_sampler = UniformClipSampler(
        clip_duration=clip_duration, backpad_last=True
    )
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=sample_per_clip)

    maxlen = 0
    for all_video in video_paths:
        if not isinstance(all_video, list):
            video = EncodedVideo.from_path(
                all_video,
                decoder="pyav",
                decode_audio=False,
                # **{"sample_rate": sample_rate},
            )

            all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)

            all_video = []
            for clip_timepoints in all_clips_timepoints:
                # Read the clip, get frames
                clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
                if clip is None:
                    raise ValueError("No clip found")
                video_clip = frame_sampler(clip["video"])
                video_clip = video_clip / 255.0  # since this is float, need 0-1

                all_video.append(video_clip)
            # Hard set here to be less than 60 seconds
            if len(all_video) > 60:
                all_video = all_video[:60]

        all_video = torch.cat(all_video, dim=1)
        all_video = video_transform(all_video).transpose(0, 1)  # C, T, H, W -> T, C, H, W
        if all_video.size(0) > maxlen:
            maxlen = all_video.size(0)
        video_outputs.append(all_video)

    padded_video_outputs = []
    padded_video_mask = []
    for video in video_outputs:
        if video.size(0) < maxlen:
            diffsize = maxlen - video.size(0)
            padded_video_mask.append([1] * video.size(0) + [0] * diffsize)
            video = torch.cat([video, video.new_zeros(
                diffsize, video.size(1), video.size(2), video.size(3))], dim=0)
        else:
            padded_video_mask.append([1] * video.size(0))
        padded_video_outputs.append(video)

    return torch.stack(padded_video_outputs, dim=0).to(device), torch.tensor(padded_video_mask).to(device)

def load_and_transform_video_data(
    video_paths,
    device,
    clip_duration=2,
    clips_per_video=5,
    sample_rate=16000,
):
    if video_paths is None:
        return None

    video_outputs = []
    video_transform = transforms.Compose(
        [
            pv_transforms.ShortSideScale(224),
            NormalizeVideo(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )
    frame_sampler = pv_transforms.UniformTemporalSubsample(num_samples=clip_duration)

    for video_path in video_paths:
        video = EncodedVideo.from_path(
            video_path,
            decoder="decord",
            decode_audio=False,
            **{"sample_rate": sample_rate},
        )

        all_clips_timepoints = get_clip_timepoints(clip_sampler, video.duration)

        all_video = []
        for clip_timepoints in all_clips_timepoints:
            # Read the clip, get frames
            clip = video.get_clip(clip_timepoints[0], clip_timepoints[1])
            if clip is None:
                raise ValueError("No clip found")
            video_clip = frame_sampler(clip["video"])
            video_clip = video_clip / 255.0  # since this is float, need 0-1

            all_video.append(video_clip)

        all_video = [video_transform(clip) for clip in all_video]
        all_video = SpatialCrop(224, num_crops=3)(all_video)

        all_video = torch.stack(all_video, dim=0)
        video_outputs.append(all_video)

    return torch.stack(video_outputs, dim=0).to(device)
