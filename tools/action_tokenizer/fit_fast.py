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

import numpy as np
from transformers import AutoProcessor
import pickle
from tqdm import tqdm

# def split_action_into_subsegments(action, T):
#     N = len(action)
#     num_segments = N // T  # 忽略末尾不足T长度的部分
#     trimmed = action[:num_segments * T]  # 裁剪掉多余的帧
#     subsegments = trimmed.reshape(num_segments, T, -1)
#     return subsegments

def split_action_into_subsegments(action, T):
    """
    将action矩阵按长度T分解成连续的子段，每个子段的形状是 (T, 7)
    滑动窗口为1
    """
    N = len(action)
    subsegments = []
    
    # 滑动窗口，生成子段
    for start in range(N - T + 1):
        subsegment = action[start:start+T]  # 取连续的T行
        subsegments.append(subsegment)
    
    return np.array(subsegments)


# Load the tokenizer from the Hugging Face hub
tokenizer = AutoProcessor.from_pretrained("/share/project/yuqi.wang/UniVLA/pretrain/fast", trust_remote_code=True)

bridge_pickle = '/share/project/yuqi.wang/datasets/processed_data/meta/simplerenv_bridge_trainval.pkl'
with open(bridge_pickle, 'rb') as f:
    data = pickle.load(f)
##### configure the parameters
T = 10
scale = 50
save_path = '/share/project/yuqi.wang/UniVLA/pretrain/fast_bridge_t5_s50'
#####
all_subsegments = []
for value in tqdm(data):
    action = value["action"]
    subsegments = split_action_into_subsegments(action, T)
    all_subsegments.append(subsegments)

all_subsegments = np.concatenate(all_subsegments, axis=0)

print(all_subsegments.shape)  # 输出形状

# test original the tokenizer
tokens = tokenizer(all_subsegments)
decoded_actions = tokenizer.decode(tokens)

# compute the difference between the original and the new decoded actions
diff = np.abs(all_subsegments - decoded_actions)
# mean difference
mean_diff = np.mean(diff)
print("mean diff:", mean_diff)

# train the tokenizer
tokenizer = tokenizer.fit(all_subsegments, scale=scale)
# save the tokenizer
tokenizer.save_pretrained(save_path)

# compute the difference between the original and the new decoded actions
tokens = tokenizer(all_subsegments)
# print average length of the tokens
print(np.mean([len(token) for token in tokens]))
print(np.max([len(token) for token in tokens]))
print(np.min([len(token) for token in tokens]))
decoded_actions = tokenizer.decode(tokens)
mean_diff = np.mean(np.abs(all_subsegments - decoded_actions))
print(mean_diff)




