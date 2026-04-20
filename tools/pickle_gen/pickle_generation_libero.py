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

import os
import os.path as osp
import pickle
from tqdm import tqdm
import numpy as np
import sys
import argparse

# Project-specific paths
sys.path.append("/mnt/bn/audio-visual-llm-data5/wangsiyin/models/UniVLA")

# Import normalization utility
from train.dataset.normalize_pi0 import RunningStats, save

def sort_by_int(filename):
    return int(os.path.splitext(filename)[0])

def main(dataset_path, output_path, normalizer_path, output_filename):
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(normalizer_path, exist_ok=True)

    language_dir = osp.join(dataset_path, "libero_all")
    vq_dir = osp.join(dataset_path, "libero_all_codes_200_augshift")
    gripper_vq_dir = osp.join(dataset_path, "libero_all_gripper_codes_200_augshift")

    min_frames = 8
    result_file = []

    print("Loading scenes from:", language_dir)
    for scene in tqdm(os.listdir(language_dir)):
        instr_file = osp.join(language_dir, scene, "instruction.txt")
        if not osp.exists(instr_file):
            continue
        with open(instr_file, "r") as f:
            text = f.read()
        speech = osp.join(language_dir, scene, "instruction2.wav")

        # Load action sequences
        action_folder = osp.join(language_dir, scene, "actions")
        if not osp.exists(action_folder):
            continue
        action_files = [osp.join(action_folder, file) for file in sorted(os.listdir(action_folder), key=sort_by_int)]
        if len(action_files) < min_frames:
            continue
        action = [np.load(a) for a in action_files]

        # Load image tokens
        img_dir = osp.join(vq_dir, scene)
        if not osp.exists(img_dir):
            continue
        img_files = [osp.join(img_dir, file) for file in sorted(os.listdir(img_dir), key=sort_by_int)]

        # Load gripper image tokens
        gripper_img_dir = osp.join(gripper_vq_dir, scene)
        if not osp.exists(gripper_img_dir):
            continue
        gripper_img_files = [osp.join(gripper_img_dir, file) for file in sorted(os.listdir(gripper_img_dir), key=sort_by_int)]

        # Filter out short clips
        if len(img_files) < min_frames or len(gripper_img_files) < min_frames:
            continue

        result_file.append({
            "text": text,
            "image": img_files,
            "action": action,
            "gripper_image": gripper_img_files,
            "speech": speech
        })

    print(f"Total number of valid scenes: {len(result_file)}")
    if not result_file:
        raise ValueError("No valid scenes found. Check your dataset path.")

    # === Normalize actions ===
    normalizer = RunningStats()
    action_data = np.concatenate([scene["action"] for scene in result_file])
    normalizer.update(action_data)
    stats = normalizer.get_statistics()

    print("Mean:", stats.mean)
    print("Std:", stats.std)
    print("Q01:", stats.q01)
    print("Q99:", stats.q99)

    for scene in result_file:
        action = scene["action"]
        # Normalize to [-1, 1] using Q01 and Q99 as bounds
        normalized = 2 * (action - stats.q01) / (stats.q99 - stats.q01 + 1e-8) - 1
        scene["action"] = np.clip(normalized, -1, 1)

    # === Save normalized dataset ===
    output_file = osp.join(output_path, output_filename)
    with open(output_file, "wb") as f:
        pickle.dump(result_file, f)
    print(f"Saved normalized data to {output_file}")

    # === Save normalization statistics ===
    save(normalizer_path, {"libero": stats})
    print(f"Saved normalizer statistics to {normalizer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize Libero dataset action values.")
    parser.add_argument("--dataset_path", type=str, default="/mnt/bn/audio-visual-llm-data5/datasets/modified_libero_rlds", help="Root path to dataset.")
    parser.add_argument("--output_path", type=str, default="/mnt/bn/audio-visual-llm-data5/datasets/modified_libero_rlds", help="Path to save normalized data.")
    parser.add_argument("--normalizer_path", type=str, default="/mnt/bn/audio-visual-llm-data5/wangsiyin/models/UniVLA/configs/normalizer_libero", help="Path to save normalization stats.")
    parser.add_argument("--output_filename", type=str, default="libero_all_norm_augshift_speech2.pkl", help="Filename for normalized pickle output.")
    args = parser.parse_args()

    main(args.dataset_path, args.output_path, args.normalizer_path, args.output_filename)
