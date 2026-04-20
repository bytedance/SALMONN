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
import json
import numpy as np

# corresponding to the evaluation results saved
exp_name = 'univla_calvin_abcd_video'
path = f'/share/project/yuqi.wang/UniVLA/logs/calvin_exp_main/{exp_name}/eval'

def compute_average_scores(path, num_files=8):
    # Initialize accumulators
    total_avg_seq_len = 0
    total_chain_sr = {str(i): 0 for i in range(1, 6)}
    
    for i in range(num_files):
        json_path = os.path.join(path, f'results_calvin_rand-{i}.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Accumulate scores
        total_avg_seq_len += data['null']['avg_seq_len']
        
        # Accumulate chain success rates
        for length in range(1, 6):
            total_chain_sr[str(length)] += data['null']['chain_sr'][str(length)]
    
    # Calculate averages
    avg_seq_len = total_avg_seq_len / num_files
    avg_chain_sr = {length: value / num_files for length, value in total_chain_sr.items()}
    
    # Print results
    print(f"Average sequence length across {num_files} runs: {avg_seq_len:.4f}")
    print("Average chain success rates:")
    for length, rate in avg_chain_sr.items():
        print(f"  Length {length}: {rate:.4f}")
    
    return {
        'avg_seq_len': avg_seq_len,
        'chain_sr': avg_chain_sr
    }

results = compute_average_scores(path)