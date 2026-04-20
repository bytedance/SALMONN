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
from typing import List
import sys
from io import BytesIO
import base64
from PIL import Image
from tqdm import tqdm
import json
import csv
import numpy as np

from torch.utils.data import default_collate


def collate_with_none(batch):
    assert isinstance(batch[0], dict)

    delete_keys = set()
    data_type = None
    for k in batch[0]:
        if batch[0][k] is None:
            delete_keys.add(k)
        elif "data_type" in batch[0]:
            data_type = batch[0]["data_type"]

    delete_keys.add("data_type")
    for k in delete_keys:
        for d in batch:
            d.pop(k, None)

    collated = default_collate(batch)
    for k in delete_keys:
        collated[k] = None
    collated["data_type"] = data_type

    return collated


def list_files(folders: List[str]) -> List[str]:
    files = []
    for folder in folders:
        if os.path.isdir(folder):
            files.extend([os.path.join(folder, d) for d in os.listdir(folder)])
        elif os.path.isfile(folder):
            files.append(folder)
        else:
            print("Path {} is invalid".format(folder))
            sys.stdout.flush()
    return files


def list_all_files(dirs, verbose=False):
    sub_dirs = list_files(dirs)
    all_files = []
    all_dirs = []

    if verbose:
        _iter = tqdm(sub_dirs)
    else:
        _iter = sub_dirs

    for d in _iter:
        if os.path.isdir(d):
            all_dirs.append(d)
        else:
            all_files.append(d)

    if all_dirs:
        all_files.extend(list_all_files(all_dirs))
    return all_files


def list_dir_with_cache(data_dir, cache_dir=None, verbose=True):
    from robovlms.utils.dist_train import get_rank

    data_dir = data_dir.rstrip("/")

    if cache_dir is None:
        _parent_dir = os.path.dirname(data_dir)
        _base_name = os.path.basename(data_dir)
        _cache_file = os.path.join(_parent_dir, _base_name + f"_filelist.json")
    else:
        max_name_length = os.pathconf("/", "PC_NAME_MAX")
        _cache_name = data_dir.strip("/").replace("/", "_") + ".json"
        _cache_name = _cache_name[-max_name_length:]
        os.makedirs(cache_dir, exist_ok=True)
        _cache_file = os.path.join(cache_dir, _cache_name)

    if os.path.exists(_cache_file):
        if get_rank() == 0 and verbose:
            print(f"Loading data list from {_cache_file}...")

        with open(_cache_file) as f:
            return json.load(f)

    else:
        verbose = get_rank() == 0 and verbose
        data_list = list_all_files([data_dir], verbose=verbose)
        _temp_cache = _cache_file + f".rank{str(get_rank())}"
        max_name_length = os.pathconf("/", "PC_NAME_MAX")
        _temp_cache = _temp_cache[-max_name_length:]
        with open(_temp_cache, "w") as f:
            json.dump(data_list, f)
        if not os.path.exists(_cache_file):
            import shutil

            shutil.move(_temp_cache, _cache_file)

    return data_list


def grouping(data_list, num_group):
    groups = [[] for _ in range(num_group)]
    for i, d in enumerate(data_list):
        groups[i % num_group].append(d)
    return groups


def b64_2_img(data):
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)


def read_csv(rpath, encoding=None, **kwargs):
    if rpath.startswith("hdfs"):
        raise NotImplementedError
    cfg_args = dict(delimiter=",")
    cfg_args.update(kwargs)
    try:
        data = []
        with open(rpath, encoding=encoding) as csv_file:
            csv_reader = csv.reader(csv_file, **cfg_args)
            columns = next(csv_reader)
            for row in csv_reader:
                data.append(dict(zip(columns, row)))
        return data
    except:
        return []


def deep_update(d1, d2):
    # use d2 to update d1
    if d2.get("__override__", False):
        # override
        d1.clear()
        d1.update(d2)
        d1.pop("__override__", None)
        return d1

    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            deep_update(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    return d1


def load_config(config_file):
    _config = json.load(open(config_file))
    config = {}
    if _config.get("parent", None):
        deep_update(config, load_config(_config["parent"]))
    deep_update(config, _config)
    return config


def alpha2rotm(a):
    """Alpha euler angle to rotation matrix."""
    if isinstance(a, float):
        rotm = np.array(
            [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
        )
        return rotm
    elif isinstance(a, list) or isinstance(a, np.ndarray):
        rotm = np.zeros((a.shape[0], 3, 3))
        rotm[:, 0, 0] = 1
        rotm[:, 1, 1] = np.cos(a)
        rotm[:, 1, 2] = -np.sin(a)
        rotm[:, 2, 1] = np.sin(a)
        rotm[:, 2, 2] = np.cos(a)
        return rotm
    else:
        raise TypeError("a must be float, list or ndarray")


def beta2rotm(b):
    """Beta euler angle to rotation matrix."""
    if isinstance(b, float):
        rotm = np.array(
            [[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]]
        )
        return rotm
    elif isinstance(b, list) or isinstance(b, np.ndarray):
        b = np.array(b)
        rotm = np.zeros((b.shape[0], 3, 3))
        rotm[:, 0, 0] = np.cos(b)
        rotm[:, 0, 2] = np.sin(b)
        rotm[:, 1, 1] = 1
        rotm[:, 2, 0] = -np.sin(b)
        rotm[:, 2, 2] = np.cos(b)
        return rotm
    else:
        raise TypeError("b must be float, list or ndarray")


def gamma2rotm(c):
    """Gamma euler angle to rotation matrix."""
    if isinstance(c, float):
        rotm = np.array(
            [[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]]
        )
        return rotm
    elif isinstance(c, list) or isinstance(c, np.ndarray):
        rotm = np.zeros((c.shape[0], 3, 3))
        rotm[:, 0, 0] = np.cos(c)
        rotm[:, 0, 1] = -np.sin(c)
        rotm[:, 1, 0] = np.sin(c)
        rotm[:, 1, 1] = np.cos(c)
        rotm[:, 2, 2] = 1
        return rotm
    else:
        raise TypeError("c must be float, list or ndarray")


def euler2rotm(euler_angles):
    """Euler angle (ZYX) to rotation matrix."""
    alpha = euler_angles[..., 0]
    beta = euler_angles[..., 1]
    gamma = euler_angles[..., 2]

    rotm_a = alpha2rotm(alpha)
    rotm_b = beta2rotm(beta)
    rotm_c = gamma2rotm(gamma)

    rotm = rotm_c @ rotm_b @ rotm_a

    return rotm


def rotm2euler(Rs):
    assert Rs.shape[-2:] == (3, 3), "Input must be of shape (b, 3, 3)"

    sy = np.sqrt(Rs[..., 0, 0] ** 2 + Rs[..., 1, 0] ** 2)
    singular = sy < 1e-6

    x = np.zeros_like(sy)
    y = np.zeros_like(sy)
    z = np.zeros_like(sy)

    x[~singular] = np.arctan2(Rs[~singular, 2, 1], Rs[~singular, 2, 2])
    y[~singular] = np.arctan2(-Rs[~singular, 2, 0], sy[~singular])
    z[~singular] = np.arctan2(Rs[~singular, 1, 0], Rs[~singular, 0, 0])

    x[singular] = np.arctan2(-Rs[singular, 1, 2], Rs[singular, 1, 1])
    y[singular] = np.arctan2(-Rs[singular, 2, 0], sy[singular])

    # Normalize angles to (-pi, pi]
    x = np.mod(x + np.pi, 2 * np.pi) - np.pi
    y = np.mod(y + np.pi, 2 * np.pi) - np.pi
    z = np.mod(z + np.pi, 2 * np.pi) - np.pi

    return np.stack([x, y, z], axis=-1)
