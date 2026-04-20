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
from omegaconf import OmegaConf
import json
import random
# ====================== Config ======================

# Project-specific paths
PROJECT_ROOT = "/mnt/bn/audio-visual-llm-data5/datasets"
sys.path.append(f"/mnt/bn/audio-visual-llm-data5/wangsiyin/models/UniVLA")

# Import normalization utility
from train.dataset.normalize_pi0 import RunningStats, save

# Input paths
DATASET_NAME = "calvin"  # Options: calvin, calvin_d, calvin_abc
dataset_path = f"/mnt/bn/audio-visual-llm-data6/datasets/calvin/processed_data"
language_dir = f"{dataset_path}/{DATASET_NAME}"
vq_dir = f"{dataset_path}/{DATASET_NAME}_codes_augshift"
gripper_vq_dir = f"{dataset_path}/{DATASET_NAME}_gripper_codes_augshift"

# Output paths
output_path = f"{dataset_path}/meta"
normalizer_path = f"{PROJECT_ROOT}/configs/normalizer_calvin"
output_pkl_file = osp.join(output_path, f"{DATASET_NAME}_norm_multispeech.pkl")

# Settings
interval = 1           # Not currently used but may be useful
min_frame_count = 8    # Minimum frame count per scene
use_raw_images = False # If True, use raw RGB images instead of VQ codes

# ====================================================

# Ensure output dirs exist
os.makedirs(normalizer_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

with open("/mnt/bn/audio-visual-llm-data6/datasets/multitask_json/calvin_text_multi.json",'r') as f:
    val_annotations = json.load(f)

# Load and process dataset
result_file = []
for scene in tqdm(os.listdir(language_dir), desc="Processing scenes"):
    instr_file = osp.join(language_dir, scene, "instruction.txt")
    if not osp.exists(instr_file):
        print(f"Warning: Missing instruction file in {scene}")
        continue

    with open(instr_file, "r") as f:
        text = f.read().strip()
    # speech = osp.join(language_dir, scene, "instruction.wav")
    speech = random.choice(val_annotations[text])

    # Load action data
    action_folder = osp.join(language_dir, scene, "actions")
    if not osp.exists(action_folder):
        print(f"Warning: Missing action folder in {scene}")
        continue

    action_files = sorted([
        osp.join(action_folder, fname)
        for fname in os.listdir(action_folder)
        if fname.endswith(".npz")
    ])

    try:
        actions = [np.load(f)["rel_actions"] for f in action_files]
    except Exception as e:
        print(f"Error loading actions for {scene}: {e}")
        continue

    # Load image paths
    if use_raw_images:
        img_dir = osp.join(language_dir, scene, "rgb_static")
        gripper_dir = osp.join(language_dir, scene, "rgb_gripper")
    else:
        img_dir = osp.join(vq_dir, scene)
        gripper_dir = osp.join(gripper_vq_dir, scene)

    try:
        img_files = sorted([
            osp.join(img_dir, fname) for fname in os.listdir(img_dir)
        ])
        gripper_img_files = sorted([
            osp.join(gripper_dir, fname) for fname in os.listdir(gripper_dir)
        ])
    except FileNotFoundError:
        print(f"Warning: Missing VQ images in {scene}")
        continue

    # Skip scenes with too few frames
    if len(img_files) < min_frame_count:
        continue

    result_file.append({
        "text": text,
        "image": img_files,
        "action": actions,
        "gripper_image": gripper_img_files,
        "speech": speech
    })

print(f"\nTotal valid scenes: {len(result_file)}")

# Initialize and compute normalization statistics
if not result_file:
    raise RuntimeError("No valid scenes found. Exiting.")

normalizer = RunningStats()
action_data = np.concatenate([scene["action"] for scene in result_file])
normalizer.update(action_data)
stats = normalizer.get_statistics()

# Print stats
print("Normalization statistics:")
print("  Mean:", stats.mean)
print("  Std:", stats.std)
print("  Q01:", stats.q01)
print("  Q99:", stats.q99)

# Save normalization parameters
# save(normalizer_path, {DATASET_NAME: stats})

# Normalize all actions
for scene in result_file:
    action = scene["action"]
    normalized = 2 * (action - stats.q01) / (stats.q99 - stats.q01 + 1e-8) - 1
    scene["action"] = np.clip(normalized, -1, 1)

# Save result
with open(output_pkl_file, "wb") as f:
    pickle.dump(result_file, f)

print(f"\nProcessed dataset saved to: {output_pkl_file}")
