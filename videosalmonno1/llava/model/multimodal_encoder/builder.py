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

import os

from .siglip_encoder_flash import SigLipVisionTowerFlash


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    return SigLipVisionTowerFlash(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
