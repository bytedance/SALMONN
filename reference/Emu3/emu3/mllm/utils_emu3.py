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

""" Logits Processor Helper class for Emu3. """

import torch

class Emu3PrefixConstrainedLogitsHelper:

    def __init__(
        self,
        height,
        width,
        img_token,
        eoi_token,
        eos_token,
        eol_token,
        eof_token,
        pad_token,
        visual_tokens,
    ):
        self.height = height
        self.width = width
        self.img_token = img_token
        self.eoi_token = eoi_token
        self.eos_token = eos_token
        self.eol_token = eol_token
        self.eof_token = eof_token
        self.pad_token = pad_token
        self.visual_tokens = visual_tokens

        self.offset_cache = {}

    def __call__(self, batch_id, input_ids):
        if batch_id not in self.offset_cache:
            position = torch.nonzero(input_ids == self.img_token, as_tuple=True)[0][0]
            self.offset_cache[batch_id] = position

        height = self.height[batch_id] if self.height.shape[0] > 1 else self.height[0]
        width = self.width[batch_id] if self.width.shape[0] > 1 else self.width[0]

        offset = input_ids.shape[0] - self.offset_cache[batch_id]
        height = height.to(offset.device)
        width = width.to(offset.device)

        if offset % (width + 1) == 0:
            return (self.eol_token, )
        elif offset == (width + 1) * height + 1:
            return (self.eof_token, )
        elif offset == (width + 1) * height + 2:
            return (self.eoi_token, )
        elif offset == (width + 1) * height + 3:
            return (self.eos_token, )
        elif offset > (width + 1) * height + 3:
            return (self.pad_token, )
        else:
            return self.visual_tokens

class Emu3PrefixConstrainedVideoLogitsHelper:

    def __init__(
        self,
        height,
        width,
        frames,
        img_token,
        eoi_token,
        eos_token,
        eol_token,
        eof_token,
        pad_token,
        visual_tokens,
    ):
        self.height = height
        self.width = width
        self.frames = frames
        self.img_token = img_token
        self.eoi_token = eoi_token
        self.eos_token = eos_token
        self.eol_token = eol_token
        self.eof_token = eof_token
        self.pad_token = pad_token
        self.visual_tokens = visual_tokens

        self.offset_cache = {}

    def __call__(self, batch_id, input_ids):
        if batch_id not in self.offset_cache:
            position = torch.nonzero(input_ids == self.img_token, as_tuple=True)[0][0]
            self.offset_cache[batch_id] = position

        height = self.height[batch_id] if self.height.shape[0] > 1 else self.height[0]
        width = self.width[batch_id] if self.width.shape[0] > 1 else self.width[0]
        frames = self.frames[batch_id] if self.frames.shape[0] > 1 else self.frames[0]

        offset = input_ids.shape[0] - self.offset_cache[batch_id]
        height = height.to(offset.device)
        width = width.to(offset.device)

        if offset == (width * height + 1) * frames + 1:
            return (self.eoi_token, )
        elif offset % (width * height + 1) == 0:
            return (self.eof_token, )
        elif offset == (width * height + 1) * frames + 2:
            return (self.eos_token, )
        elif offset > (width * height + 1) * frames + 2:
            return (self.pad_token, )
            # return (self.eos_token, )
        else:
            return self.visual_tokens

        # 计算帧和帧内偏移
        # frame_offset = offset // (width * height+1)
        # in_frame_offset = offset % (width * height+1)

        # if frame_offset == frames:  # 全部帧处理完成，返回 EOI
        #     return (self.eoi_token,)
        # elif in_frame_offset == 0:  # 当前帧结束，返回 EOF
        #     return (self.eof_token,)
        # elif offset == (width * height + 1) * frames + 1:  # 序列结束，返回 EOS
        #     return (self.eos_token,)
        # elif offset > (width * height + 1) * frames + 1:  # 超出范围，返回 PAD
        #     return (self.pad_token,)
        # else:  # 返回视觉 token
        #     return self.visual_tokens