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
sys.path.append("/share/project/yuqi.wang/UniVLA")
from train.dataset.normalize_pi0 import RunningStats, save, load

def normalize_action(action):
    normalized = action.copy()
    for i in range(14):
        if i not in [6, 13]:
            normalized[:, i] = action[:, i] - action[0, i]
    return normalized

def normalize_action_pose(action):
    normalized = action.copy()
    state = action[0].copy()
    mask = np.ones_like(state, dtype=bool)
    mask[6] = False
    mask[13] = False
    dims = mask.shape[0]
    delta = np.where(mask, state, 0)
    normalized[..., :dims] -= np.expand_dims(delta, axis=-2)
    normalized[..., 3:6] -= 2 * np.pi * np.round(normalized[..., 3:6] / (2 * np.pi))
    normalized[..., 10:13] -= 2 * np.pi * np.round(normalized[..., 10:13] / (2 * np.pi))
    return normalized

# ===== 路径设置 =====
output_path = ""
normalizer_path = ""
os.makedirs(normalizer_path, exist_ok=True)

language_root = ""
vq_root = ""

tasks = {
    'clean_blackboard_processed_end_6dpose_adaptive_merged_5e-3_max10':'clean_blackboard_codes_128',
    'fold_clothes_new_end100_6dpose_adaptive_merged_1e-3_max10_filtered': 'aloha_fold_codes_128',
    'insert_plug_processed_end_6dpose_adaptive_merged_5e-3_max10':'insert_plug_codes_128',
    'put_glasses_processed_end_6dpose_adaptive_merged_5e-3_max10':'put_glasses_codes_128',
    'clean_desk_processed_end_6dpose_adaptive_merged_5e-3_max10':'clean_desk_codes_128',
    'food_pack_processed_end_6dpose_adaptive_merged_5e-3_max10': 'food_pack_codes_128',
    'pour_water_processed_end_6dpose_action_interval_5': 'aloha_pour_final_inter5_eepose_codes_128',
    'unplug_processed_end_6dpose_adaptive_merged_5e-3_max10':'unplug_codes_128',
}

task_name_translations = {
    'clean_blackboard_processed_end_6dpose_adaptive_merged_5e-3_max10': 'clean the blackboard',
    'fold_clothes_new_end100_6dpose_adaptive_merged_1e-3_max10_filtered': 'fold the clothes',
    'insert_plug_processed_end_6dpose_adaptive_merged_5e-3_max10': 'insert the plug',
    'put_glasses_processed_end_6dpose_adaptive_merged_5e-3_max10': 'put on the glasses',
    'clean_desk_processed_end_6dpose_adaptive_merged_5e-3_max10': 'clean the desk',
    'food_pack_processed_end_6dpose_adaptive_merged_5e-3_max10': 'pack the food',
    'pour_water_processed_end_6dpose_action_interval_5': 'pour the water',
    'unplug_processed_end_6dpose_adaptive_merged_5e-3_max10': 'unplug the device'
}

views = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
use_chunk = True
use_pose = True
chunk_size = 20

result_file = []
norm_stats_save = {}  # 存储每个任务的统计量

# ===== 遍历每个任务 =====
for task_name, task_vq_name in tqdm(tasks.items(), desc="Processing Tasks"):

    task_result_file = []

    lang_dir = osp.join(language_root, task_name)
    vq_dir = osp.join(vq_root, task_vq_name)
    if not osp.exists(lang_dir) or not osp.exists(vq_dir):
        print(f"[Warning] Missing data for {task_name}")
        continue

    for scene in os.listdir(lang_dir):
        action_path = osp.join(lang_dir, scene, "action.npy")
        if not osp.exists(action_path):
            continue

        action = np.load(action_path)

        # 读取每个视角下的图像路径
        view_imgs = {}
        valid = True
        for view in views:
            img_dir = osp.join(vq_dir, scene, view)
            if not osp.exists(img_dir):
                valid = False
                break
            img_files = sorted(os.listdir(img_dir))
            view_imgs[view] = [osp.join(img_dir, file) for file in img_files]
        if not valid:
            continue

        if use_chunk:
            num_chunks = len(action) - chunk_size + 1
            for start in range(1, num_chunks):
                end = start + chunk_size
                item = {view: view_imgs[view][start:end] for view in views}
                item["text"] = task_name_translations.get(task_name, task_name)
                chunk_action = action[start:end].copy()
                item["action"] = normalize_action_pose(chunk_action) if use_pose else normalize_action(chunk_action)
                task_result_file.append(item)
        else:
            item = {view: view_imgs[view] for view in views}
            item["text"] = task_name_translations.get(task_name, task_name)
            item["action"] = normalize_action(action)
            task_result_file.append(item)

    if not task_result_file:
        continue

    # ==== 统计归一化参数 ====
    normalizer = RunningStats()
    action_data = np.concatenate([scene["action"] for scene in task_result_file])
    normalizer.update(action_data)
    stats = normalizer.get_statistics()
    norm_stats_save[task_name] = stats

    # ==== 对每个动作做归一化 ====
    for scene in task_result_file:
        action = scene["action"].copy()
        normalized = 2 * (action - stats.q01) / (stats.q99 - stats.q01 + 1e-8) - 1
        scene["action"] = np.clip(normalized, -1, 1)

    # 加入总数据集
    result_file.extend(task_result_file)

# ===== 保存数据集 =====
os.makedirs(output_path, exist_ok=True)
output_file = osp.join(output_path, "aloha_joint.pkl")
with open(output_file, "wb") as f:
    pickle.dump(result_file, f)

save(normalizer_path, norm_stats_save)

print(f"✅ Total samples: {len(result_file)}")
print(f"✅ Saved dataset to: {output_file}")
