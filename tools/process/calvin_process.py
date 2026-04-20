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
import numpy as np
from PIL import Image
from tqdm import tqdm

# Data paths
# DATA_ROOT = '/share/project/yuqi.wang/datasets/robotics/task_ABCD_D'
# SAVE_ROOT = '/share/project/yuqi.wang/datasets/processed_data/calvin'

# DATA_ROOT = '/share/project/yuqi.wang/datasets/robotics/task_ABC_D'
# SAVE_ROOT = '/share/project/yuqi.wang/datasets/processed_data/calvin_abc'

DATA_ROOT = '/share/project/yuqi.wang/datasets/robotics/task_D_D'
SAVE_ROOT = '/share/project/yuqi.wang/datasets/processed_data/calvin_d'

split = 'training'

lang_file = os.path.join(DATA_ROOT, split, 'lang_annotations', 'auto_lang_ann.npy')
ep_start_end_ids = np.load(os.path.join(DATA_ROOT, split, "ep_start_end_ids.npy"))

data = np.load(lang_file, allow_pickle=True).item()

episodes = data['info']['episodes']
epi_idx = data['info']['indx']

lang = data['language']['ann']
task = data['language']['task']

assert len(epi_idx) == len(lang)
for lan, idx in tqdm(zip(lang, epi_idx), total=len(lang), desc="Processing episodes"):

    video_name = 'video_' + str(idx[0]) + '_' + str(idx[1])
    video_dir = os.path.join(SAVE_ROOT, video_name)
    os.makedirs(video_dir, exist_ok=True)

    # Save language instruction
    lang_file = os.path.join(video_dir, 'instruction.txt')
    with open(lang_file, 'w') as f:
        f.write(lan)
    
    # Save observations
    for i in range(idx[0], idx[1] + 1):
        # Construct file name, ensuring it's 7 digits long
        npz_name = f'episode_{i:07d}.npz'
        npz_file = os.path.join(DATA_ROOT, split, npz_name)
        
        # Check that the file exists
        assert os.path.exists(npz_file), f"{npz_file} does not exist"
        
        # Load the .npz file
        obs_i = np.load(npz_file)

        # Uncomment this to see keys inside the file
        # print(f"Keys in {npz_name}: {obs_i.files}")

        # Extract data
        rgb_static = obs_i['rgb_static']
        rgb_gripper = obs_i['rgb_gripper']

        actions = obs_i['actions']
        rel_actions = obs_i['rel_actions']
        robot_obs = obs_i['robot_obs']
        scene_obs = obs_i['scene_obs']

        rgb_static_dir = os.path.join(video_dir, 'rgb_static')
        rgb_gripper_dir = os.path.join(video_dir, 'rgb_gripper')

        os.makedirs(rgb_static_dir, exist_ok=True)
        os.makedirs(rgb_gripper_dir, exist_ok=True)

        # Save rgb_static and rgb_gripper as images
        rgb_static_path = os.path.join(rgb_static_dir, f'{i:07d}.jpg')
        rgb_gripper_path = os.path.join(rgb_gripper_dir, f'{i:07d}.jpg')

        # Assume rgb_static and rgb_gripper are 3D arrays (H, W, C), ensure they are image-compatible
        Image.fromarray(rgb_static).save(rgb_static_path)
        Image.fromarray(rgb_gripper).save(rgb_gripper_path)

        # Create and save .npz file containing actions and observations
        actions_data = {
            'actions': actions,
            'rel_actions': rel_actions,
            'robot_obs': robot_obs,
            'scene_obs': scene_obs
        }

        # Save the dictionary as .npz format
        npz_save_path = os.path.join(video_dir, 'actions', f'{i:07d}.npz')
        os.makedirs(os.path.dirname(npz_save_path), exist_ok=True)

        np.savez(npz_save_path, **actions_data)
