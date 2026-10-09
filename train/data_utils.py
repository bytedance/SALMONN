"""Read original robot data and the portable training_data release."""
import json
import os
import pickle
from functools import lru_cache

import numpy as np


class RobotDataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported data object: {module}.{name}")
        return allowed[module, name]


PATH_FIELDS = {"image", "gripper_image", "speech", "path", "path_a", "path_i", "path_r", "speech_path", "image_path"}


def resolve_data_paths(data, root, field=None):
    if isinstance(data, dict):
        for k, value in data.items():
            data[k] = resolve_data_paths(value, root, k)
    elif isinstance(data, list):
        for i, value in enumerate(data):
            data[i] = resolve_data_paths(value, root, field)
    elif isinstance(data, tuple):
        return tuple(resolve_data_paths(v, root, field) for v in data)
    elif isinstance(data, str) and field in PATH_FIELDS:
        return os.path.join(root, data)
    return data


def load_robot_data(path, root):
    with open(path, "rb") as f:
        data = json.load(f) if path.endswith(".json") else RobotDataUnpickler(f).load()
    for scene in data:
        scene["action"] = np.asarray(scene["action"])
    return resolve_data_paths(data, root)


@lru_cache(maxsize=16)
def _trajectory_tokens(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def load_visual_tokens(path):
    if "#" in path:
        trajectory, frame = path.rsplit("#", 1)
        return _trajectory_tokens(trajectory)[int(frame)].copy()
    return np.load(path, allow_pickle=False)
