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

import transformers
from .av_dataset import LazyAVSupervisedDataset, DataCollatorForAVSupervisedDataset
from typing import Dict

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazyAVSupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    test_dataset = train_dataset
    data_collator = DataCollatorForAVSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=test_dataset, data_collator=data_collator)


def make_test_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for testing"""
    test_dataset = make_test_data(tokenizer, data_args)
    data_collator = DataCollatorForAVSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=test_dataset, eval_dataset=test_dataset, data_collator=data_collator)

def make_test_data(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    test_dataset = LazyAVSupervisedDataset(tokenizer=tokenizer, data_path=data_args.test_data_path, data_args=data_args, is_test=True)
    return test_dataset
