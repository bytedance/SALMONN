# coding=utf-8
# Copyright (2026) Tsinghua University, Bytedance Ltd. and/or its affiliates
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

# Adapted from https://github.com/baaivision/UniVLA. The original license is located at 'third-party-license/UniVLA.txt'.
# Adapted from https://github.com/baaivision/Emu3. The original license is located at 'third-party-license/Emu.txt'.

""" Processor class for Emu3. """

from math import ceil
import re
from typing import List, Optional, Sequence, Union
from functools import partial

from PIL import Image
import torch
from torch.nn import functional as F
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput
from transformers.utils import logging

from .utils_emu3 import Emu3PrefixConstrainedLogitsHelper,Emu3PrefixConstrainedVideoLogitsHelper


logger = logging.get_logger(__name__)


class Emu3Processor(ProcessorMixin):
    r"""
    Constructs an Emu3 processor which wraps an Emu3 image processor and an Emu3 vision vq model and an Emu3 tokenizer into a single processor.

    [`Emu3Processor`] offers all the functionalities of [`Emu3VisionVQModel`] and [`Emu3Tokenizer`]. See the
    [`~Emu3Processor.__call__`], [`~Emu3Processor.decode`], [`~Emu3Processor.vision_encode`], [`~Emu3Processor.vision_decode`]
    for more information.

    Args:
        image_processor ([`Emu3VisionVQImageProcessor`]):
            The image processor is a required input.
        vision_tokenizer ([`Emu3VisionVQModel`]):
            The vision tokenizer is a required input.
        tokenizer ([`Emu3Tokenizer`]):
            The tokenizer is a required input.
        prefix_template(`str`, *optional*):
            The prefix template for image tokens
        visual_template(`Tuple[str, ...]`, *optional*):
            The visual token template for image tokens
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["vision_tokenizer", "prefix_template", "visual_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        vision_tokenizer=None,
        tokenizer=None,
        chat_template="You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:",
        prefix_template="{H}*{W}",
        visual_template=("<|visual token {token_id:0>6d}|>", r"<\|visual token (\d+)\|>"),
        **kwargs,
    ):
        assert vision_tokenizer is not None, "image tokenizer can not be None"

        self.vision_tokenizer = vision_tokenizer
        self.prefix_template = prefix_template
        self.visual_template = visual_template
        if hasattr(self.vision_tokenizer, "config"):
            self.vis_tok_spatial_factor = 2 ** (len(self.vision_tokenizer.config.ch_mult) - 1)

        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        if hasattr(self.vision_tokenizer, "config"):
            self.const_helper = self.build_const_helper_video()

    @torch.no_grad()
    def video_process(
        self,
        text: Optional[TextInput or PreTokenizedInput] = None,
        image: Optional[Image.Image or List[Image.Image]] = None,
        video_tokens: Optional[torch.Tensor] = None,
        gripper_tokens: Optional[torch.Tensor] = None,
        frames: int = 0,
        *,
        mode: str = "G",
        ratio: str or List[str] = "1:1",
        image_area: int = 518400,
        padding_image: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Emu3Tokenizer's [`~Emu3Tokenizer.__call__`] to encode the text.
        To prepare the image(s), this method forwards the `image` argument to
        Emu3VisionVQImageProcessor's [`~Emu3VisionVQImageProcessor.__call__`] and Emu3VisionVQModel's [`~EmuVideoVQModel.encode`]
        if `image` is not `None`. Please refer to the doctsring of the above two methods for more information.

        Args:
            text (`str` or `List[str]`):
                The sequence or a batch of sequence to be encoded. A sequence is a string.
            image (`PIL.Image.Image` or `List[PIL.Image.Image]`, *optional*):
                The image or a batch of images to be prepared. An image is a PIL image.
            mode (`str`, *optional*, in `G` or `U`):
                task mode, `G` for generation and `U` for understanding
            ratio (`str`, *optional*):
                the image width-height ratio for generation
            image_area (`int`, *optional*):
                image area used to calcualte the generated image height and width
            padding_image (`bool`, *optional*):
                whether pad images to same size for fast preprocessing if they have different sizes
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **image_size** -- List of image size of input images or generated images.
        """
        assert mode in ('G', 'U', "IG", "VLA", "VLA_Video", "VLA2", "VLA2_Video", "VLA_Video_moe", "INTERLEAVE", "VLA_ALOHA", "VLA_COT", "VLA_VIDEO_ALOHA"), "mode must be 'G' or 'U'."
        if isinstance(text, str):
            text = [text]

        if isinstance(image, Image.Image):
            image = [image]

        if not isinstance(text[0], str):
            raise ValueError("`text` must be string or list of string")

        image_tokens = None
        if mode == 'G':
            if image is not None:
                raise ValueError("You have to specify only `text` in generation mode")

            if isinstance(ratio, str):
                ratio = [ratio] * len(text)

            if len(ratio) != len(text):
                raise ValueError("ratio number must match text number")
        else:
            if image is None and video_tokens is None:
                raise ValueError("Invalid input image. Please provide exactly one PIL.Image.Image per text.")

        prompt_list, size_list = [], []
        for idx, text_prompt in enumerate(text):
            prompt = self.tokenizer.bos_token
            if mode == 'U':
                t, h, w = video_tokens[idx].shape
                image_prompt = self.format_video_prompt(video_tokens[idx])
                
                prompt += self.chat_template.format(image_prompt=image_prompt, text_prompt=text_prompt)
                # prompt += image_prompt
            elif mode == 'G':
                _, h, w = video_tokens[idx].shape
                t = frames
                video_prompt = (
                    self.tokenizer.boi_token +
                    f"{t}*{h}*{w}" +
                    self.tokenizer.img_token
                )
                prompt += (text_prompt + video_prompt)
            elif mode == 'IG':
                _, h, w = video_tokens[idx].shape
                t = frames
                context_frames = kwargs['context_frames']
                video_prompt = self.format_video_prompt(video_tokens[idx], context_frames=context_frames)
                prompt += (text_prompt + video_prompt)
            elif mode == 'INTERLEAVE':
                _, h, w = video_tokens[idx].shape
                t = frames
                context_frames = kwargs['context_frames']
                video_prompt = self.format_video_prompt(video_tokens[idx], context_frames=context_frames)
                prompt += (text_prompt + video_prompt) 
                # interleave
                prompt +=self.format_video_null_prompt(t, h, w)
            elif mode == 'VLA':
                _, h, w = video_tokens[idx].shape
                t = frames
                context_frames = kwargs['context_frames']
                video_prompt = self.format_video_prompt(video_tokens[idx], context_frames=context_frames)
                prompt += (text_prompt + video_prompt) 
                if gripper_tokens is not None:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[idx], context_frames=context_frames)
                    prompt += gripper_prompt
                # action 
                prompt +=self.tokenizer.boa_token
            elif mode == 'VLA_COT':
                _, h, w = video_tokens[idx].shape
                t = frames
                context_frames = kwargs['context_frames']
                video_prompt = self.format_video_prompt(video_tokens[idx], context_frames=context_frames)
                prompt += (text_prompt + video_prompt) 
                if gripper_tokens is not None:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[idx], context_frames=context_frames)
                    prompt += gripper_prompt
                # reasoning 
                prompt +=self.tokenizer.bot_token
            elif mode == 'VLA_ALOHA':
                _, h, w = video_tokens[idx].shape
                t = frames
                context_frames = kwargs['context_frames']
                video_prompt = self.format_video_prompt(video_tokens[idx], context_frames=context_frames)
                prompt += (text_prompt + video_prompt) 
                if gripper_tokens is not None:
                    gripper_prompt_left = self.format_video_prompt(gripper_tokens[0][idx], context_frames=context_frames)
                    prompt += gripper_prompt_left
                    gripper_prompt_right = self.format_video_prompt(gripper_tokens[1][idx], context_frames=context_frames)
                    prompt += gripper_prompt_right
                # action 
                prompt +=self.tokenizer.boa_token
            elif mode == 'VLA_VIDEO_ALOHA':
                _, h, w = video_tokens[idx].shape
                t = frames
                context_frames = kwargs['context_frames']
                video_prompt = self.format_video_prompt(video_tokens[idx], context_frames=context_frames)
                prompt = video_prompt
                if gripper_tokens is not None:
                    gripper_prompt_left = self.format_video_prompt(gripper_tokens[0][idx], context_frames=context_frames)
                    prompt += gripper_prompt_left
                    gripper_prompt_right = self.format_video_prompt(gripper_tokens[1][idx], context_frames=context_frames)
                    prompt += gripper_prompt_right
                # action 
                prompt +=self.tokenizer.boa_token
            elif mode == 'VLA_Video':
                _, h, w = video_tokens[idx].shape
                t = frames
                context_frames = kwargs['context_frames']
                video_prompt = self.format_video_prompt(video_tokens[idx], context_frames=context_frames)
                prompt = video_prompt 
                if gripper_tokens is not None:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[idx], context_frames=context_frames)
                    prompt += gripper_prompt
                # action 
                prompt +=self.tokenizer.boa_token
            elif mode == 'VLA_Video_moe':
                _, h, w = video_tokens[idx].shape
                t = frames
                context_frames = kwargs['context_frames']
                video_prompt = self.format_video_prompt(video_tokens[idx], context_frames=context_frames)
                prompt = video_prompt 
                if gripper_tokens is not None:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[idx], context_frames=context_frames)
                    prompt += gripper_prompt
            elif mode == 'VLA2_Video':
                _, h, w = video_tokens.shape
                t = frames
                context_frames = kwargs['context_frames']
                video_prompt = self.format_video_prompt(video_tokens, context_frames=context_frames)
                prompt = video_prompt 
                if gripper_tokens is not None:
                    gripper_prompt = self.format_video_prompt(gripper_tokens, context_frames=context_frames)
                    prompt += gripper_prompt
                # action 
                prompt +=self.tokenizer.boa_token
            elif mode == 'VLA2':
                _, h, w = video_tokens.shape
                t = frames
                context_frames = kwargs['context_frames']
                video_prompt = self.format_video_prompt(video_tokens, context_frames=context_frames)
                prompt += (text_prompt + video_prompt) 
                if gripper_tokens is not None:
                    gripper_prompt = self.format_video_prompt(gripper_tokens, context_frames=context_frames)
                    prompt += gripper_prompt
                # action 
                prompt +=self.tokenizer.boa_token

            prompt_list.append(prompt)
            size_list.append([h, w, t])

        text_inputs = self.tokenizer(prompt_list, **kwargs)
        return BatchFeature(data={**text_inputs, "video_size": size_list}, tensor_type=kwargs.get("return_tensors"))

    @torch.no_grad()
    def __call__(
        self,
        text: Optional[TextInput or PreTokenizedInput] = None,
        image: Optional[Image.Image or List[Image.Image]] = None,
        *,
        mode: str = "G",
        ratio: str or List[str] = "1:1",
        image_area: int = 518400,
        padding_image: bool = False,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Emu3Tokenizer's [`~Emu3Tokenizer.__call__`] to encode the text.
        To prepare the image(s), this method forwards the `image` argument to
        Emu3VisionVQImageProcessor's [`~Emu3VisionVQImageProcessor.__call__`] and Emu3VisionVQModel's [`~EmuVideoVQModel.encode`]
        if `image` is not `None`. Please refer to the doctsring of the above two methods for more information.

        Args:
            text (`str` or `List[str]`):
                The sequence or a batch of sequence to be encoded. A sequence is a string.
            image (`PIL.Image.Image` or `List[PIL.Image.Image]`, *optional*):
                The image or a batch of images to be prepared. An image is a PIL image.
            mode (`str`, *optional*, in `G` or `U`):
                task mode, `G` for generation and `U` for understanding
            ratio (`str`, *optional*):
                the image width-height ratio for generation
            image_area (`int`, *optional*):
                image area used to calcualte the generated image height and width
            padding_image (`bool`, *optional*):
                whether pad images to same size for fast preprocessing if they have different sizes
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model.
            - **image_size** -- List of image size of input images or generated images.
        """
        assert mode in ('G', 'U'), "mode must be 'G' or 'U'."
        if isinstance(text, str):
            text = [text]

        if isinstance(image, Image.Image):
            image = [image]

        if not isinstance(text[0], str):
            raise ValueError("`text` must be string or list of string")

        image_tokens = None
        if mode == 'G':
            if image is not None:
                raise ValueError("You have to specify only `text` in generation mode")

            if isinstance(ratio, str):
                ratio = [ratio] * len(text)

            if len(ratio) != len(text):
                raise ValueError("ratio number must match text number")
        else:
            if image is None:
                raise ValueError("Invalid input image. Please provide exactly one PIL.Image.Image per text.")

            if not isinstance(image, Sequence) and not isinstance(image, Image.Image):
                raise ValueError("Invalid input image. Please provide PIL.Image.Image or List[PIL.Image.Image].")

            if isinstance(image, Sequence) and not isinstance(image[0], Image.Image):
                raise ValueError("Invalid input image. Please provide PIL.Image.Image or List[PIL.Image.Image].")

            image_tokens = self.tokenize_image(image, padding_image=padding_image)
            if len(text) != len(image_tokens):
                raise ValueError("number of image must match number of text prompt")

        prompt_list, size_list = [], []
        for idx, text_prompt in enumerate(text):
            prompt = self.tokenizer.bos_token
            if mode == 'U':
                h, w = image_tokens[idx].shape
                imgstr = self.to_imgstr(image_tokens[idx])
                image_prompt = (
                    self.tokenizer.boi_token +
                    self.prefix_template.format(H=h, W=w) +
                    self.tokenizer.img_token + 
                    imgstr +
                    self.tokenizer.eol_token +
                    self.tokenizer.eof_token +
                    self.tokenizer.eoi_token
                )
                prompt += self.chat_template.format(image_prompt=image_prompt, text_prompt=text_prompt)
            else:
                h, w = self.calculate_generate_size(ratio[idx], image_area, self.vision_tokenizer.spatial_scale_factor)
                image_prompt = (
                    self.tokenizer.boi_token +
                    self.prefix_template.format(H=h, W=w) +
                    self.tokenizer.img_token
                )
                prompt += (text_prompt + image_prompt)

            prompt_list.append(prompt)
            size_list.append([h, w])

        text_inputs = self.tokenizer(prompt_list, **kwargs)
        return BatchFeature(data={**text_inputs, "image_size": size_list}, tensor_type=kwargs.get("return_tensors"))

    @torch.no_grad()
    def batch_decode(self, *args, **kwargs):
        docs = self.tokenizer.batch_decode(*args, **kwargs)
        return [self.multimodal_decode(d) for d in docs]

    @torch.no_grad()
    def decode(self, *args, **kwargs):
        doc = self.tokenizer.decode(*args, **kwargs)
        return self.multimodal_decode(doc)
        
    @torch.no_grad()
    def decode_video(self, *args, **kwargs):
        doc = self.tokenizer.decode(*args, **kwargs)
        return self.multimodal_decode_video(doc)


    @torch.no_grad()
    def vision_encode(self, *args, **kwargs):
        return self.vision_tokenizer.encode(*args, **kwargs)

    @torch.no_grad()
    def vision_decode(self, *args, **kwargs):
        return self.vision_tokenizer.decode(*args, **kwargs)
    
    @torch.no_grad()
    def multimodal_decode_video(self, doc):
        multimodal_output = []
        pattern = rf'({re.escape(self.tokenizer.boi_token)}.*?{re.escape(self.tokenizer.eoi_token)})'
        chunks = re.split(pattern, doc)
        tokens = []
        for c in chunks:
            if len(c) == 0:
                continue
            if self.tokenizer.boi_token in c:
                video = []
                frame_chunks = re.split(re.escape(self.tokenizer.eof_token), c)  # 按帧分割
                for frame in frame_chunks:
                    token_ids = re.findall(self.visual_template[1], frame)  # 提取所有视觉 Token
                    if len(token_ids) > 0:
                        image = torch.tensor([int(m) for m in token_ids], dtype=torch.long,
                                            device=self.vision_tokenizer.device)  # 构建张量
                        if len(image) == 576: # DROID
                            image = image.reshape(18,32)
                        elif len(image) == 320: # RT1
                            image = image.reshape(16,20)
                        elif len(image) == 256: # 1X
                            image = image.reshape(16,16)
                        elif len(image) == 625: # calvin
                            image = image.reshape(25,25)
                        tokens.append(image)
                        image = self.vision_tokenizer.decode(image[None]).float()
                        image = self.image_processor.postprocess(image)["pixel_values"][0]
                        video.append(image)
                multimodal_output.append(video)
                multimodal_output.append(tokens)
        else:
            multimodal_output.append(c)

        return multimodal_output if len(multimodal_output) > 1 else multimodal_output[0]

    @torch.no_grad()
    def multimodal_decode(self, doc):
        multimodal_output = []
        pattern = rf'({re.escape(self.tokenizer.boi_token)}.*?{re.escape(self.tokenizer.eoi_token)})'
        chunks = re.split(pattern, doc)
        for c in chunks:
            if len(c) == 0:
                continue

            if self.tokenizer.boi_token in c:
                image = []
                image_rows = re.split(re.escape(self.tokenizer.eol_token), c)
                for r in image_rows:
                    token_ids = re.findall(self.visual_template[1], r)
                    if len(token_ids) > 0:
                        row_token = [int(m) for m in token_ids]
                        image.append(row_token)
                image = torch.tensor(image, dtype=torch.long, device=self.vision_tokenizer.device)
                image = self.vision_tokenizer.decode(image[None]).float()
                image = self.image_processor.postprocess(image)["pixel_values"][0]
                multimodal_output.append(image)
            else:
                multimodal_output.append(c)

        return multimodal_output if len(multimodal_output) > 1 else multimodal_output[0]

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    # def to_imgstr(self, image_tokens):
    #     image_tokens = image_tokens.cpu().numpy().tolist()
    #     image_token_str = [
    #         [
    #             self.visual_template[0].format(token_id=token_id)
    #             for token_id in token_row
    #         ]
    #         for token_row in image_tokens
    #     ]
    #     image_row_str = ["".join(token_row) for token_row in image_token_str]
    #     imgstr = self.tokenizer.eol_token.join(image_row_str)
    #     return imgstr
    
    @torch.no_grad()
    def format_image_prompt(self, image_tokens):
        h, w = image_tokens.shape
        imgstr = self.to_imgstr(image_tokens)

        image_prompt = (
            self.tokenizer.boi_token +
            f"{h}*{w}" +
            self.tokenizer.img_token +
            imgstr +
            self.tokenizer.eol_token +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )

        return image_prompt

    def to_imgstr(self, image_tokens):
        image_token_str = [
            [
                self.visual_template[0].format(token_id=token_id)
                for token_id in token_row
            ]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr

    def calculate_generate_size(self, ratio, image_area, spatial_scale_factor):
        w, h = map(int, ratio.split(":"))
        current_area = h * w
        target_ratio = (image_area / current_area) ** 0.5

        th = int(round(h * target_ratio / spatial_scale_factor))
        tw = int(round(w * target_ratio / spatial_scale_factor))
        return th, tw
    
    @torch.no_grad()
    def add_prefix_template(self, frames, h, w):
        video_prompt = (
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token + 
            self.tokenizer.boi_token +
            f"{frames}*{h}*{w}" +  # 视频的帧数、高度和宽度
            self.tokenizer.img_token  # 视频开始标记
        )
        return self.tokenizer(video_prompt)['input_ids']

    def format_video_null_prompt(self, frames, h, w):
        video_prompt = (
            self.tokenizer.boi_token +
            f"{frames}*{h}*{w}" +  # 视频的帧数、高度和宽度
            self.tokenizer.img_token  # 视频开始标记
        )

        return video_prompt

    def format_video_prompt(self, video_tokens, context_frames=0):
        # 假设video_tokens是一个形状为[frames, height, width]的张量
        frames, h, w = video_tokens.shape
        if context_frames > 0:
            video_tokens = video_tokens[:context_frames]
            videostr = self.to_videostr(video_tokens)

            video_prompt = (
                self.tokenizer.boi_token +
                f"{frames}*{h}*{w}" +  # 视频的帧数、高度和宽度
                self.tokenizer.img_token +  # 视频开始标记
                videostr + 
                self.tokenizer.eof_token + 
                self.tokenizer.eoi_token
            )

        else:
            videostr = self.to_videostr(video_tokens)

            video_prompt = (
                self.tokenizer.boi_token +
                f"{frames}*{h}*{w}" +  # 视频的帧数、高度和宽度
                self.tokenizer.img_token +  # 视频开始标记
                videostr +
                self.tokenizer.eof_token +
                self.tokenizer.eoi_token
            )

        return video_prompt

    def to_videostr(self, video_tokens):
        frame_str_list = []
        for frame in video_tokens:
            frame_token_str = [
                self.visual_template[0].format(token_id=token_id)
                for token_id in frame.flatten()
            ]
            frame_str = "".join(frame_token_str)
            frame_str_list.append(frame_str)
        videostr = self.tokenizer.eof_token.join(frame_str_list)
        return videostr


    def tokenize_image(self, image: List[Image.Image], *, padding_image: bool = False):
        is_all_same_size, prev_size = True, None
        for im in image:
            if prev_size is not None:
                is_all_same_size &= (prev_size == im.size)
            prev_size = im.size

        if is_all_same_size:
            image_inputs = self.image_processor(image, return_tensors="pt")["pixel_values"]
            image_inputs = image_inputs.to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
            image_tokens = self.vision_tokenizer.encode(image_inputs)
        elif padding_image:
            image_inputs = [self.image_processor(im, return_tensors="pt")["pixel_values"] for im in image]
            image_shapes = [im.shape[2:] for im in image_inputs]
            max_shape = (
                max([im_shape[0] for im_shape in image_shapes]),
                max([im_shape[1] for im_shape in image_shapes]),
            )
            image_inputs = [
                F.pad(im_inp, (0, max_shape[1] - im_shape[1], 0, max_shape[0] - im_shape[0]))
                for im_inp, im_shape in zip(image_inputs, image_shapes)
            ]
            image_inputs = torch.cat(image_inputs, dim=0).to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
            image_tokens = self.vision_tokenizer.encode(image_inputs)
            image_tokens = [
                im_tok[:ceil(im_shape[0] / self.vis_tok_spatial_factor), :ceil(im_shape[1] / self.vis_tok_spatial_factor)]
                for im_tok, im_shape in zip(image_tokens, image_shapes)
            ]
        else:
            image_tokens = []
            for im in image:
                image_input = self.image_processor(im, return_tensors="pt")["pixel_values"]
                image_input = image_input.to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
                image_tokens.append(self.vision_tokenizer.encode(image_input).squeeze(0))

        return image_tokens

    def build_const_helper(self):
        (
            img_token,
            eoi_token,
            eos_token,
            eol_token,
            eof_token,
            pad_token,
            vis_start,
            vis_end,
        ) = self.tokenizer.encode([
            self.tokenizer.img_token,
            self.tokenizer.eoi_token,
            self.tokenizer.eos_token,
            self.tokenizer.eol_token,
            self.tokenizer.eof_token,
            self.tokenizer.pad_token,
            self.visual_template[0].format(token_id=0),
            self.visual_template[0].format(token_id=self.vision_tokenizer.config.codebook_size - 1),
        ])

        const_helper = partial(
            Emu3PrefixConstrainedLogitsHelper,
            img_token=img_token,
            eoi_token=eoi_token,
            eos_token=eos_token,
            eol_token=eol_token,
            eof_token=eof_token,
            pad_token=pad_token,
            visual_tokens=list(range(vis_start, vis_end + 1)),
        )
        return const_helper

    def build_prefix_constrained_fn(self, height, width):
        helper = self.const_helper(height=height, width=width)
        return helper
    
    def build_const_helper_video(self):
        (
            img_token,
            eoi_token,
            eos_token,
            eol_token,
            eof_token,
            pad_token,
            vis_start,
            vis_end,
        ) = self.tokenizer.encode([
            self.tokenizer.img_token,
            self.tokenizer.eoi_token,
            self.tokenizer.eos_token,
            self.tokenizer.eol_token,
            self.tokenizer.eof_token,
            self.tokenizer.pad_token,
            self.visual_template[0].format(token_id=0),
            self.visual_template[0].format(token_id=self.vision_tokenizer.config.codebook_size - 1),
        ])

        const_helper = partial(
            Emu3PrefixConstrainedVideoLogitsHelper,
            img_token=img_token,
            eoi_token=eoi_token,
            eos_token=eos_token,
            eol_token=eol_token,
            eof_token=eof_token,
            pad_token=pad_token,
            visual_tokens=list(range(vis_start, vis_end + 1)),
        )
        return const_helper

    def build_prefix_constrained_fn_video(self, height, width, frames):
        helper = self.const_helper(height=height, width=width, frames=frames)
        return helper
