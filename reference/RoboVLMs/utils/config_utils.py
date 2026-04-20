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

import math
import json
import os
import re

from utils.common import list_files


def deep_update(d1, d2):
    for k, v in d2.items():
        if isinstance(v, dict):
            if v.get("__override__", False):
                d1[k] = v
                d1[k].pop("__override__", None)
            elif k in d1 and isinstance(d1[k], dict):
                deep_update(d1[k], v)
            else:
                d1[k] = v
        else:
            d1[k] = v

    return d1


def load_config(config_file):
    print(config_file)
    _config = json.load(open(config_file))
    config = {}
    if _config.get("parent", None):
        deep_update(config, load_config(_config["parent"]))
    deep_update(config, _config)
    return config


def get_single_gpu_bsz(exp_config):
    if isinstance(exp_config["batch_size"], int):
        if isinstance(exp_config["train_dataset"], list):
            return exp_config["batch_size"] * len(exp_config["train_dataset"])
        else:
            assert isinstance(exp_config["train_dataset"], dict)
            return exp_config["batch_size"]
    else:
        assert isinstance(exp_config["batch_size"], list)
        return sum(exp_config["batch_size"])


def get_exp_name(exp, mode="pretrain"):
    if mode == "pretrain":
        return exp
    else:
        return f"{exp}_{mode}"
