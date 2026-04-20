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

# Adapted from https://github.com/Robot-VLAs/RoboVLMs. The original license is located at 'third-party-license/RoboVLMs.txt'.

import os
import re
from PIL import Image
from typing import List


def sort_ckpt(ckpt_dir):
    if isinstance(ckpt_dir, str):
        # get sorted ckpt list
        ckpt_files = os.listdir(ckpt_dir)
        ckpt_files = [f for f in ckpt_files if f.endswith(".ckpt")]
        ckpt_dirs = [ckpt_dir] * len(ckpt_files)

    else:
        # sometimes trials may fail, and the ckpt_dir will be a list of dirs including
        # ckpts from both the original trial and resumed trials
        assert isinstance(ckpt_dir, list)
        ckpt_files = []
        ckpt_dirs = []
        for d in ckpt_dir:
            _ckpt_files = os.listdir(d)
            _ckpt_files = [f for f in _ckpt_files if f.endswith(".ckpt")]
            _ckpt_dirs = [d] * len(_ckpt_files)
            ckpt_files.extend(_ckpt_files)
            ckpt_dirs.extend(_ckpt_dirs)

    ckpt_steps = [re.search(r"step=\d+", f).group() for f in ckpt_files]
    ckpt_steps = [int(s[5:]) for s in ckpt_steps]
    ckpts = list(zip(ckpt_steps, ckpt_dirs, ckpt_files))
    ckpts = sorted(ckpts, key=lambda x: x[0])
    ckpt_files = [os.path.join(x[1], x[2]) for x in ckpts]
    ckpt_steps = [x[0] for x in ckpts]
    return ckpt_files, ckpt_steps


def save_gif(images: List[Image.Image], save_path: str, duration: int = 300):
    images[0].save(
        save_path, save_all=True, loop=True, append_images=images[1:], duration=duration
    )
