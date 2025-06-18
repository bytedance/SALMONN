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

# Adapted from https://github.com/LLaVA-VL/LLaVA-NeXT. The original license is located at 'third-party-license/llava_next.txt'.

from typing import Dict, Optional, Sequence, List
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava import conversation as conversation_lib
import transformers
import torch
from packaging import version
import tokenizers
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


def preprocess_multimodal(
    sources: Sequence[str],
    data_args,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value'] and not sentence['value'].startswith(DEFAULT_IMAGE_TOKEN):
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            
    return sources

def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant", "reject_gpt": "<|im_start|>assistant", "gt": "<|im_start|>assistant", "history_gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets, reject_input_ids, reject_targets, gt_input_ids, gt_targets = [], [], [], [], [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        dpo_ver = False
        for sentence in source:
            if sentence["from"] == "reject_gpt":
                dpo_ver = True
                break

        if dpo_ver:
            input_id, target = [], []
            system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
            input_id += system
            target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
            for j, sentence in enumerate(source):
                if sentence["from"] == "gpt" or sentence["from"] == "gt":
                    continue
                role = roles[sentence["from"]]
                if has_image and "<image>" in sentence["value"]:
                    assert sentence["value"].startswith("<image>"), print(sentence["value"])

                    _input_id = tokenizer(role).input_ids + nl_tokens + [IMAGE_TOKEN_INDEX] + nl_tokens + tokenizer(sentence["value"][len("<image>") :]).input_ids + [im_end] + nl_tokens
                else:
                    _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
                input_id += _input_id
                if role == "<|im_start|>user":
                    _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
                elif role == "<|im_start|>assistant":
                    if sentence["from"] == "history_gpt":
                        _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
                    else:
                        _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
                else:
                    raise NotImplementedError
                target += _target
            assert len(input_id) == len(target), f"input id len {len(input_id)} != target len {len(target)}"
            # input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
            # target += [IGNORE_INDEX] * (max_len - len(target))
            reject_input_ids.append(input_id)
            reject_targets.append(target)

            input_id, target = [], []
            system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
            input_id += system
            target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
            for j, sentence in enumerate(source):
                if sentence["from"] == "gpt" or sentence["from"] == "reject_gpt":
                    continue
                role = roles[sentence["from"]]
                if has_image and "<image>" in sentence["value"]:
                    assert sentence["value"].startswith("<image>"), print(sentence["value"])

                    _input_id = tokenizer(role).input_ids + nl_tokens + [IMAGE_TOKEN_INDEX] + nl_tokens + tokenizer(sentence["value"][len("<image>") :]).input_ids + [im_end] + nl_tokens
                else:
                    _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
                input_id += _input_id
                if role == "<|im_start|>user":
                    _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
                elif role == "<|im_start|>assistant":
                    _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
                else:
                    raise NotImplementedError
                target += _target
            assert len(input_id) == len(target), f"input id len {len(input_id)} != target len {len(target)}"
            # input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
            # target += [IGNORE_INDEX] * (max_len - len(target))
            gt_input_ids.append(input_id)
            gt_targets.append(target)

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
        for j, sentence in enumerate(source):
            if sentence["from"] == "reject_gpt" or sentence["from"] == "gt":
                continue
            role = roles[sentence["from"]]
            if has_image and "<image>" in sentence["value"]:
                assert sentence["value"].startswith("<image>"), print(sentence["value"])

                _input_id = tokenizer(role).input_ids + nl_tokens + [IMAGE_TOKEN_INDEX] + nl_tokens + tokenizer(sentence["value"][len("<image>") :]).input_ids + [im_end] + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == "<|im_start|>user":
                _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == "<|im_start|>assistant":
                _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target), f"input id len {len(input_id)} != target len {len(target)}"
        # input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        # target += [IGNORE_INDEX] * (max_len - len(target))
        input_ids.append(input_id)
        targets.append(target)

    reject_input_ids_out = [None]
    reject_targets_out = [None]
    if len(input_ids) == len(reject_input_ids):
        reject_input_ids_out = torch.tensor(reject_input_ids, dtype=torch.long)
        reject_targets_out = torch.tensor(reject_targets, dtype=torch.long)
    gt_input_ids_out = [None]
    gt_targets_out = [None]
    if len(input_ids) == len(gt_input_ids):
        gt_input_ids_out = torch.tensor(gt_input_ids, dtype=torch.long)
        gt_targets_out = torch.tensor(gt_targets, dtype=torch.long)
    else:
        gt_input_ids_out = torch.tensor(input_ids, dtype=torch.long)
        gt_targets_out = torch.tensor(targets, dtype=torch.long)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
        reject_input_ids=reject_input_ids_out,
        reject_labels=reject_targets_out,
        gt_input_ids=gt_input_ids_out,
        gt_labels=gt_targets_out,
        # attention_mask=input_ids.ne(tokenizer.pad_token_id), # tensor(bs x seq_len)
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # print(conversation_lib.default_conversation.version)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    else:
        raise NotImplementedError
