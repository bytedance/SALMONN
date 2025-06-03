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

# Adapted from https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/llava/model/language_model/llava_qwen.py. Copyright 2024 Hao Zhang. The original license is located at 'third-party-license/llava_next.txt'.
# Adapted from https://github.com/bytedance/SALMONN. The original license is located at 'third-party-license/salmonn.txt'.

import torch.distributed as dist
import importlib.metadata
import math
import os
import shutil
import sys
import re
import time
from multiprocessing import Lock
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from accelerate import __version__ as accelerate_version
from accelerate import skip_first_batches
from accelerate.data_loader import (
    _PYTORCH_DATALOADER_KWARGS,
    BatchSamplerShard,
    DataLoaderShard,
    SeedableRandomSampler,
)
from accelerate.utils import DistributedType
from contextlib import contextmanager, nullcontext
try:
    from transformers.trainer_callback import ExportableState
    TRAINER_SAVE = True
except:
    print("Failed to import ExportableState")
    TRAINER_SAVE = False

try:
    from cruise.utilities.distributed import DIST_ENV
    from cruise.utilities.hdfs_io import hcopy, hmkdir, hrm, hlist_files
except:
    print("Failed to import cruise")

from torch.utils.data import Dataset, Sampler, RandomSampler
from tqdm import tqdm

import json
from copy import deepcopy

import torch.distributed as dist

from multiprocessing import Lock

from typing import Dict, Union, Any, Tuple

# Create a lock
lock = Lock()

from transformers import Trainer, set_seed, AutoModel, AutoProcessor
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)

from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf
from typing import List, Optional

from transformers.integrations.tpu import tpu_spmd_dataloader

from transformers.debug_utils import DebugOption, DebugUnderflowOverflow

from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available

from transformers.trainer_callback import (
    TrainerState,
)

from transformers.trainer_pt_utils import (
    get_dataloader_sampler,
    get_model_param_count,
    get_parameter_names,
    nested_detach,
)

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_nested_concat, smp_forward_only, smp_forward_backward

from transformers.training_args import ParallelMode
from transformers.utils import (
    is_peft_available,
    is_accelerate_available,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
    is_apex_available
)

if is_apex_available():
    from apex import amp

from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
)

from transformers.integrations import (
    hp_params,
)

from transformers.utils.quantization_config import QuantizationMethod

from accelerate import __version__ as accelerate_version

from accelerate.data_loader import SeedableRandomSampler

from accelerate import Accelerator, skip_first_batches
from collections import OrderedDict
from accelerate.utils import (
    DistributedType,
)

import math

import sys

from packaging import version

from peft import PeftModel

import time

from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import KeywordsStoppingCriteria
from llava.vcd_utils.vcd_add_noise import add_diffusion_noise


TIME_STAMP = os.environ.get('TIME_STAMP', 'default_value')
BYTENAS = os.environ.get('BYTENAS', 'vl-research')
HDFS_BASE_PATH = None
# from apex import amp

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult = 8, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i : i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        variable_length: bool = False,
        group_by_modality: bool = False,
        group_by_modality_auto: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.variable_length = variable_length
        self.group_by_modality = group_by_modality
        self.group_by_modality_auto = group_by_modality_auto

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.variable_length:
            assert not self.group_by_modality, "Variable length grouping is not supported with modality grouping."
            indices = get_variable_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            if self.group_by_modality:
                indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            elif self.group_by_modality_auto:
                indices = get_modality_length_grouped_indices_auto(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            else:
                indices = get_length_grouped_indices_auto_single(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)



def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


TRAINER_STATE_NAME = "trainer_state.json"


class LLaVATrainer(Trainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__first_save = True
        self.ref_adapter_name = None
        if "text" in getattr(self.model, "rag_type", "") and "video" in getattr(self.model, "rag_type", "") and "train" not in getattr(self.model, "rag_type", ""):
            self.text_encoder = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=torch.float).eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
            )
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality=True,
            )
        elif self.args.group_by_modality_length_auto:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality_auto=True,
            )
        elif self.args.group_by_varlen:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                # self.args.train_batch_size, # TODO: seems that we should have gradient_accumulation_steps
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                variable_length=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            if self.args.mm_projector_lr is not None:
                lr_mapper['mm_projector'] = self.args.mm_projector_lr
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper['vision_tower'] = self.args.mm_vision_tower_lr
            if len(lr_mapper) > 0:
                special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
                for module_keyword, lr in lr_mapper.items():
                    module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
                    optimizer_grouped_parameters.extend([
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": lr,
                        },
                        {
                            "params": [
                                p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": lr,
                        },
                    ])
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.args.save_torch_ckpt:
            if DIST_ENV.rank != 0:
                return
            
            if self.args.save_torch_ckpt: # isinstance(model.module, PeftModel)
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                # checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                # output_dir = os.path.join(run_dir, checkpoint_folder)
                output_dir = run_dir
                param_grad_dic = {
                    k: v.requires_grad for (k, v) in model.named_parameters()
                }
                
                if True: # self.args.save_lora_only: #  and not self.__first_save:
                    state_dict = model.state_dict().copy()
                    keys = list(state_dict.keys())
                    for k in keys:
                        if k in param_grad_dic.keys() and not param_grad_dic[k]:
                            # delete parameters that do not require gradient
                            del state_dict[k]

                    if True:
                        output_path = os.path.join(HDFS_BASE_PATH, output_dir, f"checkpoint-{self.state.global_step}.bin")
                        if self.__first_save:
                            hmkdir(os.path.join(HDFS_BASE_PATH, output_dir))
                        torch.save(state_dict, os.path.join(output_dir, f"checkpoint-{self.state.global_step}.bin"))
                        hcopy(os.path.join(output_dir, f"checkpoint-{self.state.global_step}.bin"), output_path, chunk_thread_num=1)
                    else:
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, f"checkpoint-{self.state.global_step}.bin")
                        torch.save(state_dict, output_path)
                
                if self.__first_save: # or not self.args.save_lora_only: #  else:
                    state_dict = model.state_dict().copy()
                    os.makedirs(output_dir, exist_ok=True)
                    self.__first_save = False
                    try:
                        model.module.config.save_pretrained(output_dir)
                        hcopy(os.path.join(output_dir, "config.json"), os.path.join(HDFS_BASE_PATH, output_dir, "config.json"))
                    except Exception as e:
                        print(e)

                    if True:
                        output_path = os.path.join(HDFS_BASE_PATH, output_dir, f"all_parameters.bin")
                        torch.save(state_dict, os.path.join(output_dir, "all_parameters.bin"))
                        hcopy(os.path.join(output_dir, "all_parameters.bin"), output_path, chunk_thread_num=1)
                    else:
                        output_path = os.path.join(output_dir, f"all_parameters.bin")
                        torch.save(state_dict, output_path)

                print(f"Save model at {output_path}")
                
        else:
            # Save model checkpoint
            checkpoint_folder = f"checkpoint-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None: # self.hp_search_backend is None
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            self.save_model(output_dir, _internal_call=True)
            if self.args.lora_enable:
                if dist.get_rank() == 0:
                    new_ckpt = OrderedDict()
                    for name, params in model.named_parameters():
                        if params.requires_grad and 'lora' not in name:
                            new_ckpt[name[len('module.'):]] = params.data.cpu()
                    torch.save(new_ckpt, os.path.join(output_dir, 'nonLora.bin'))
                dist.barrier()

            if not self.args.save_only_model: # self.args.save_only_model = False
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Determine the new best metric / best model checkpoint
            if metrics is not None and self.args.metric_for_best_model is not None: # metris is None
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                try:
                    metric_value = metrics[metric_to_check]
                except KeyError as exc:
                    raise KeyError(
                        f"The `metric_for_best_model` training argument is set to '{metric_to_check}', "
                        f"which is not found in the evaluation metrics. "
                        f"The available evaluation metrics are: {list(metrics.keys())}. "
                        f"Please ensure that the `compute_metrics` function returns a dictionary that includes '{metric_to_check}' or "
                        f"consider changing the `metric_for_best_model` via the TrainingArguments."
                    ) from exc

                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            # Save the Trainer state
            if self.args.should_save and TRAINER_SAVE: # True
                # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
                for cb in [
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if cb_name in self.state.stateful_callbacks:
                        if isinstance(self.state.stateful_callbacks[cb_name], list):
                            self.state.stateful_callbacks[cb_name].append(cb_state)
                        else:
                            self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub: # False
                self._push_from_checkpoint(output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save: # False
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

            
            if dist.get_rank() == 0:
                output_path = os.path.join(HDFS_BASE_PATH, output_dir)
                base_path = os.path.join(HDFS_BASE_PATH, run_dir)
                hmkdir(base_path)
                
                files = hlist_files([base_path])
                steps_all = []
                pattern = r"(?<=checkpoint-)\d+"
                if len(files) >= 5:
                    for file in files:
                        match = re.search(pattern, file)
                        if match:
                            extracted_number = int(match.group())
                            steps_all.append(extracted_number)
                    steps_all.sort()
                    for step in steps_all[-5::-1]:
                        print(os.path.join(base_path, f"checkpoint-{step}"))
                        hrm(os.path.join(base_path, f"checkpoint-{step}"))
                hcopy(output_dir, output_path, chunk_thread_num=1)
            
            dist.barrier()
            output_path_i = os.path.join(HDFS_BASE_PATH, output_dir)
            rng_state_path = os.path.join(output_dir, f"rng_state_{dist.get_rank()}.pth")
            hcopy(rng_state_path, output_path_i, chunk_thread_num=1)

            output_path_i = os.path.join(output_path_i, f"global_step{self.state.global_step}")
            optim_state_path = os.path.join(output_dir, f"global_step{self.state.global_step}", f"bf16_zero_pp_rank_{dist.get_rank()}_mp_rank_00_optim_states.pt")
            hcopy(optim_state_path, output_path_i, chunk_thread_num=1)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)


    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.


        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            cur_idx = {}
            for step, inputs in enumerate(epoch_iterator):                                                                                                    
                total_batched_samples += 1

                # ids = inputs.pop('ids', None)
                ids = inputs.get('ids', None)
                # print(dist.get_rank(), ids)
                
                cur_batch_idx = "###".join([str(cur_ids) for cur_ids in ids])
                # cur_idx[cur_batch_idx] = 0


                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)
                    # import pdb;pdb.set_trace()

                cur_idx[cur_batch_idx] = tr_loss_step.item()

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm
                    
                    # Optimizer step
                    # if dist.get_rank() == 0:
                    #     print(f"step {step} starts update param")
                    self.optimizer.step()
                    # if dist.get_rank() == 0:
                        # print(f"step {step} ends update param")
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    # TODO: Check spike data:
                    # if grad_norm >= 50 and self.state.global_step > 100:
                    # if tr_loss >= 5 and self.state.global_step > 100: 
                    #     print("Abnoral in loss")
                    #     print(cur_idx)
                    #     with lock:
                    #         with open(f"/mnt/bn/{BYTENAS}/workspace/yhzhang/llava-video-old/spike_{TIME_STAMP}.json", "a+") as f:
                    #             f.write(json.dumps(cur_idx) + "\n")
                    #             f.write(f"Current global step: {self.state.global_step}\n")

                    
                    # import pdb;pdb.set_trace()
                    cur_idx = {}
                    if self.state.global_step % self.state.eval_steps == 0:
                        val_loss = self.predict(self.eval_dataset, prediction_loss_only=True)
                        torch.distributed.reduce(val_loss, 0)
                        if DIST_ENV.rank == 0:
                            print("Validation Loss: {:.5f}".format(val_loss.item()/dist.get_world_size()))
                    
                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    # def store_flos(self):
    #     return

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            # tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss_scalar = tr_loss.item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def do_rollout(self, model, inputs, frames, stop_str, subtree, prev_strs):
        step = 0
        new_rollouts = []
        prev_tokens = []
        if len(prev_strs) > 0:
            prev_tokens = torch.tensor(self.tokenizer.encode("".join(prev_strs))).to(model.device)
        preds = ""
        try_times = 0
        while not preds.endswith(stop_str) and try_times < 1:
            new_inputs = deepcopy(inputs)
            if len(prev_strs) > 0:
                new_inputs["input_ids"] = torch.cat([inputs["input_ids"][0], prev_tokens], dim=0).unsqueeze(0)
            else:
                new_inputs["input_ids"] = inputs["input_ids"]
            stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, inputs['input_ids'])
            generated_tokens, _ = self.prediction_generate(model, new_inputs, frames, True, stopping_criteria)
            preds = self.tokenizer.decode(generated_tokens[0])
            try_times += 1
        # unfinished rollouts we return nothing
        if not preds.endswith(stop_str):
            # return [], []
            preds += stop_str
        split_steps = preds.split("<end_of_step>\n")
        steps = []
        for k, step in enumerate(split_steps):
            if k < len(split_steps) - 1:
                steps.append(step + "<end_of_step>\n")
        if steps == []:
            steps = split_steps
        else:
            steps[-1] += split_steps[-1]
        subtree["rollouts"] = subtree["rollouts"] + [steps]
        if prev_strs != []:
            new_rollouts.append((prev_strs, steps))
        state = subtree
        # Find how many steps should we rollout
        topk_step = 2
        rollout_contrast = []
        for k, step in enumerate(steps[:-1]):
            if len(steps) > topk_step and prev_strs == [] and self.args.contrastive_adaptive == "kldiv":
                contrast_distance = self.contrast_probs(inputs, model, steps[:k+1], steps[k+1])
                rollout_contrast.append(contrast_distance)
            if step not in state:
                state[step] = {"rollouts": [steps[k+1:]], "N": 1}
            else:
                state[step]["rollouts"].append(steps[k+1:])
                state[step]["N"] += 1
            new_rollouts.append((prev_strs + steps[:k+1], steps[k+1:]))
            state = state[step]

        multiplier = 2
        rollout_beam = [1] * len(rollout_contrast)
        if rollout_contrast != []:
            rollout_contrast = torch.tensor(rollout_contrast)
            indices = torch.topk(rollout_contrast, topk_step, dim=0)[1].tolist()
            rollout_beam = [multiplier if i in indices else 1 for i in range(len(rollout_contrast))]
        else:
            rollout_beam = [1 for i in range(len(new_rollouts))]
        return new_rollouts, rollout_beam

    def contrastive_dfs(self, model, inputs, frames, onebest_token, onebestscore, beam=1, maxsteps=20):
        stop_str = conv_templates['qwen_1_5'].sep if conv_templates['qwen_1_5'].sep_style!= SeparatorStyle.TWO else conv_templates['qwen_1_5'].sep2
        onebest = self.tokenizer.decode(onebest_token)
        if not onebest.endswith(stop_str):
            onebest += stop_str
            onebestscore -= 100
        saved_alternatives = {onebest: onebestscore}
        if beam == 1:
            return onebest, saved_alternatives
        onebest_steps = onebest.split("<end_of_step>\n")
        steps = [step + "<end_of_step>\n" for step in onebest_steps[:-1]]
        steps.append(onebest_steps[-1])
        cumstep = []
        index = 0
        while (cumstep == [] or not cumstep[-1].endswith(stop_str)) and len(cumstep) < maxsteps:
            stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, inputs['input_ids'])
            rollouts = []
            contrastive_steps = []
            for i in range(beam-1):
                contrast_step = self.contrast_gen(inputs, model, cumstep, stop_str, frames)
                contrastive_steps.append(contrast_step)
            for contrast_step in contrastive_steps:
                if contrast_step.endswith("<end_of_step>"):
                    contrast_step = contrast_step + stop_str
                prefix_steps = cumstep + [contrast_step]
                if not contrast_step.endswith(stop_str):
                    prev_tokens = torch.tensor(self.tokenizer.encode("".join(prefix_steps))).to(model.device)
                    new_inputs = deepcopy(inputs)
                    new_inputs["input_ids"] = torch.cat([inputs["input_ids"][0], prev_tokens], dim=0).unsqueeze(0)
                    generated_tokens, _ = self.prediction_generate(model, new_inputs, frames, False, stopping_criteria)
                    all_tokens = torch.cat([prev_tokens, generated_tokens[0]], dim=-1)
                else:
                    all_tokens = torch.tensor(self.tokenizer.encode("".join(prefix_steps))).to(model.device)
                score = self.compute_score(inputs, model, all_tokens)
                full_solution = self.tokenizer.decode(all_tokens)
                if full_solution.endswith(stop_str) and len(full_solution.split("<end_of_step>\n")) < maxsteps:
                    rollouts.append((full_solution, score))
            rollouts.append(("".join(steps), onebestscore))
            rollouts = sorted(rollouts, key=lambda x: x[1], reverse=True)
            full_solution, score = rollouts[0]
            saved_alternatives[full_solution] = score
            onebestscore = score
            onebest_steps = full_solution.split("<end_of_step>\n")
            steps = [step + "<end_of_step>\n" for step in onebest_steps[:-1]]
            steps.append(onebest_steps[-1])
            cumstep.append(steps[index])
            index += 1
        return "".join(cumstep), saved_alternatives

    def prediction_rollout_contrast(self, model, inputs, frames, beam=1, rollout="contrastive_rollout"):
        stop_str = conv_templates['qwen_1_5'].sep if conv_templates['qwen_1_5'].sep_style!= SeparatorStyle.TWO else conv_templates['qwen_1_5'].sep2
        rollouts = []
        tree = {"rollouts": [], "N": 1}
        all_prefixes = {}
        for prefix in inputs["prefix_steps"][0]:
            if rollout == "contrastive_expand_rollout":
                contrast_step = prefix[-1]
                n_trial = 0
                while contrast_step == prefix[-1] and n_trial < 3:
                    contrast_step = self.contrast_gen(inputs, model, prefix[:-1], stop_str, frames)
                    n_trial += 1
                all_prefixes[tuple(prefix[:-1]+[contrast_step])] = []
            all_prefixes[tuple(prefix)] = []
        for prefix, rolloutlist in all_prefixes.items():
            for i in range(beam):
                new_rollouts, _ = self.do_rollout(model, inputs, frames, stop_str, tree, list(prefix))
                if new_rollouts == []:
                    rolloutlist.append({"full_solution": list(prefix)})
                else:
                    rolloutlist.append({"full_solution": new_rollouts[0][0] + new_rollouts[0][1]})
        return all_prefixes

    def generate_with_context(self, model, inputs, frames, rollout="prmtree", stopping_criteria=None, buildtree=True):
        if rollout == "prmdpotree":
            generated_tokens, score = self.prediction_generate(model, inputs, frames, True, stopping_criteria, buildtree=buildtree)
        else:
            with self.null_ref_context():
                generated_tokens, score = self.prediction_generate(model, inputs, frames, True, stopping_criteria, buildtree=buildtree)
        return generated_tokens, score

    def prediction_buildtree(self, model, inputs, frames, beam=1, max_depth=100, interp_factor=0.0, rollout="prmtree"):
        search_trees = {}
        stop_str = conv_templates['qwen_1_5'].sep if conv_templates['qwen_1_5'].sep_style != SeparatorStyle.TWO else conv_templates['qwen_1_5'].sep2
        unfinished_states = []
        finished_sequences = []
        covered_sequence = []
        for i in range(beam):
            stopping_criteria = KeywordsStoppingCriteria(["<end_of_step>\n", stop_str], self.tokenizer, inputs['input_ids'])
            generated_tokens, score = self.generate_with_context(model, inputs, frames, rollout, stopping_criteria)
            preds = self.tokenizer.decode(generated_tokens[0])
            if preds not in covered_sequence:
                covered_sequence.append(preds)
                # Update node
                score_prm = (1 - interp_factor) * score + interp_factor * self.compute_score(inputs, model, generated_tokens[0])
                search_trees[preds] = {"state": generated_tokens[0].tolist(), "depth": 1, "state_str": [preds], "score": [score_prm]}
                if preds.endswith(stop_str) or "<end_of_step>" not in preds:
                    search_trees[preds]["end"] = True if preds.endswith(stop_str) else False
                    if preds.endswith(stop_str):
                        finished_sequences.append(("".join(search_trees[preds]["state_str"]), search_trees[preds]["score"], len(search_trees[preds]["state"])))
                else:
                    unfinished_states.append(search_trees[preds])
        while len(unfinished_states) > 0:
            # print("\n***Start expanding with {} nodes\n".format(len(unfinished_states)))
            new_unfinished_states = []
            for unfinished_state in unfinished_states:
                if unfinished_state["depth"] >= max_depth:
                    # Force it to finish
                    new_inputs = deepcopy(inputs)
                    prev_tokens = torch.tensor(unfinished_state["state"]).to(inputs["input_ids"].device)
                    new_inputs["input_ids"] = torch.cat([inputs["input_ids"][0], prev_tokens], dim=0).unsqueeze(0)
                    stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, new_inputs['input_ids'])
                    generated_tokens, score = self.generate_with_context(model, new_inputs, frames, rollout, stopping_criteria, buildtree=False)
                    preds = self.tokenizer.decode(generated_tokens[0])
                    if preds.endswith(stop_str):
                        state = unfinished_state["state"] + generated_tokens[0].tolist()
                        score_prm = (1 - interp_factor) * score + interp_factor * self.compute_score(new_inputs, model, generated_tokens[0])
                        score = unfinished_state["score"][:] + [score_prm]
                        finished_sequences.append(("".join(unfinished_state["state_str"] + [preds]), score, len(state)))
                    continue
                covered_sequence = []
                for i in range(beam):
                    new_inputs = deepcopy(inputs)
                    prev_tokens = torch.tensor(unfinished_state["state"]).to(inputs["input_ids"].device)
                    new_inputs["input_ids"] = torch.cat([inputs["input_ids"][0], prev_tokens], dim=0).unsqueeze(0)
                    stopping_criteria = KeywordsStoppingCriteria(["<end_of_step>\n", stop_str], self.tokenizer, new_inputs['input_ids'])
                    generated_tokens, score = self.generate_with_context(model, new_inputs, frames, rollout, stopping_criteria)
                    preds = self.tokenizer.decode(generated_tokens[0])
                    if preds not in covered_sequence:
                        covered_sequence.append(preds)
                        # Update node
                        score_prm = (1 - interp_factor) * score + interp_factor * self.compute_score(new_inputs, model, generated_tokens[0])
                        unfinished_state[preds] = {
                            "state": unfinished_state["state"] + generated_tokens[0].tolist(),
                            "depth": unfinished_state["depth"]+1,
                            "state_str": unfinished_state["state_str"] + [preds],
                            "score": unfinished_state["score"] + [score_prm],
                        }
                        if preds.endswith(stop_str) or "<end_of_step>" not in preds:
                            unfinished_state[preds]["end"] = True if preds.endswith(stop_str) else False
                            if preds.endswith(stop_str):
                                finished_sequences.append(("".join(unfinished_state[preds]["state_str"]), unfinished_state[preds]["score"], len(unfinished_state[preds]["state"])))
                        else:
                            new_unfinished_states.append(unfinished_state[preds])
            state_tuples = [(state, sum(state["score"]) / len(state["state"])) for state in new_unfinished_states]
            state_tuples = sorted(state_tuples, key=lambda x: x[1], reverse=True)
            unfinished_states = [state[0] for state in state_tuples[:beam]]
        # if len(finished_sequences) == 0:
        #     import pdb; pdb.set_trace()
        if finished_sequences == []:
            finished_sequences = unfinished_states
        return search_trees, finished_sequences

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ):
        if logits.shape[:-1] != labels.shape:
            seq_len = labels.size(-1)
            logits = logits[:, -seq_len:, :]

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return per_token_logps * loss_mask

    def bare_compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        logits = model(**inputs)
        if "single" in self.args.train_orm:
            labels[:, :18] = labels[:, :18] * 0 - 100
            if "analysis" in self.args.train_orm:
                all_probs = self.get_batch_logps(logits.logits, labels)
                pos_inds = torch.where(labels[0, :]==397)[0]
                pos_inds = [0] + pos_inds.tolist() + [labels.size(1)]
                pos_prob = [all_probs[0, pos_inds[k]:pos_inds[k+1]].sum().item() for k, step in enumerate(pos_inds[:-1])]
            else:
                pos_prob = self.get_batch_logps(logits.logits, labels, average_log_prob=True)
        else:
            labels[:, :18] = labels[:, :18] * 0 - 100
            labels[labels == -1] = -100
            pos_seq_len = labels.size(-1)
            logits = logits[:, -pos_seq_len:]
            pos_log_prob = torch.log_softmax(logits, dim=-1)[:, :, 0]
            pos_loss_mask = labels != -100
            if "min" in self.args.train_orm:
                pos_prob = (pos_log_prob * pos_loss_mask).min(dim=-1).values
            else:
                pos_prob = (pos_log_prob * pos_loss_mask).sum(dim=-1) # / pos_loss_mask.sum(dim=-1)
        return pos_prob

    def compute_score(self, inputs, model, gen_tokens):
        with torch.no_grad():
            new_inputs = deepcopy(inputs)
            newlabels = torch.cat([inputs["input_ids"][0] * 0 - 100, gen_tokens], dim=0).unsqueeze(0)
            new_inputs["input_ids"] = torch.cat([inputs["input_ids"][0], gen_tokens], dim=0).unsqueeze(0)
            new_inputs["attention_mask"] = torch.ones_like(new_inputs["input_ids"]).bool()
            logits = model(**new_inputs)
            logp = self.get_batch_logps(logits.logits, newlabels).sum()
        return logp

    def contrast_probs(self, inputs, model, prev_strs, current_str):
        with torch.no_grad():
            new_inputs = deepcopy(inputs)
            prev_tokens = torch.tensor(self.tokenizer.encode("".join(prev_strs)), dtype=torch.long).to(model.device)
            current_tokens = torch.tensor(self.tokenizer.encode(current_str)).to(model.device)
            new_inputs["input_ids"] = torch.cat([inputs["input_ids"][0], prev_tokens, current_tokens], dim=0).unsqueeze(0)
            new_inputs["attention_mask"] = torch.ones_like(new_inputs["input_ids"]).bool()
            logits = model(**new_inputs).logits[:, -len(current_tokens)-1:-1, :]
            # new_inputs["spectrogram"] = new_inputs["spectrogram"] * 0
            # new_inputs["raw_wav"] = new_inputs["raw_wav"] * 0
            new_inputs["images"] = [add_diffusion_noise(new_inputs["images"][0], 100)]
            contrast_logits = model(**new_inputs).logits[:, -len(current_tokens)-1:-1, :]
            logp = torch.log_softmax(logits, dim=-1)
            contrast_logp = torch.log_softmax(contrast_logits, dim=-1)
            kl_div = (torch.exp(logp) * (logp - contrast_logp)).sum(dim=-1)
        return kl_div.mean().item()

    def contrast_gen(self, inputs, model, prev_strs, stop_str, frames):
        new_inputs = deepcopy(inputs)
        prev_tokens = torch.tensor(self.tokenizer.encode("".join(prev_strs)), dtype=torch.long).to(model.device)
        if len(prev_strs) > 0:
            new_inputs["input_ids"] = torch.cat([inputs["input_ids"][0], prev_tokens], dim=0).unsqueeze(0)
        else:
            new_inputs["input_ids"] = inputs["input_ids"]
        # new_inputs["images"] = [add_diffusion_noise(new_inputs["images"][0], 100)]
        stopping_criteria = KeywordsStoppingCriteria(["<end_of_step>", stop_str], self.tokenizer, new_inputs['input_ids'])
        # preds = []
        generated_tokens, _ = self.prediction_generate(model, new_inputs, frames, True, stopping_criteria, buildtree=True)
        preds = self.tokenizer.decode(generated_tokens[0])
        return preds
    
    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with self.accelerator.unwrap_model(
            self.model
        ).disable_adapter():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only=False,
        ignore_keys=None,
        do_sample=False,
        do_reasoning=False,
        rollout="none",
        beam=1,
        interp_factor=0,
    ):
        if prediction_loss_only:
            ids = inputs.pop('ids', None)
                
            cur_batch_idx = "###".join([str(cur_ids) for cur_ids in ids])
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            if has_labels or loss_without_labels:
                labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None

            with torch.no_grad():
                # if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                return (loss, None, None)

        elif do_reasoning:
            new_prompt2 = "\n<|im_start|>user\nWhat is the final answer to the question based on your thinking steps?<|im_end|>\n<|im_start|>assistant\n"
            new_input_tokens2 = self.tokenizer(new_prompt2, return_tensors="pt").input_ids[0].to(model.device)
            inputs = self._prepare_inputs(inputs)
            ids = inputs.pop("ids", None)
            prompts = inputs.pop("prompts", None)
            labels = inputs.pop("labels", None)
            texts = inputs.pop("texts", None)

            inputs['images'] = [it.to(torch.bfloat16) for it in inputs['images']]
            inputs["raw_wav"] = inputs["raw_wav"].to(torch.bfloat16)
            inputs["spectrogram"] = inputs["spectrogram"].to(torch.bfloat16)

            conv = conv_templates['qwen_1_5']
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, inputs['input_ids'])

            frames = inputs.get("frames")
            if frames:
                frames = [frame.to(torch.bfloat16) for frame in frames]

            if beam > 1 and do_sample:
                preds = []
                for i in range(beam):
                    do_sample = True if i != 0 else False
                    generated_tokens, _ = self.prediction_generate(model, inputs, frames, do_sample, stopping_criteria)
                    reasoning_pred = self.tokenizer.decode(generated_tokens[0])

                    # Contrastive estimation
                    if self.args.contrastive_adaptive in ["kldiv", "sample"]:
                        split_steps = reasoning_pred.split("<end_of_step>\n")
                        split_steps = [step for step in split_steps if len(step.split()) > 2]
                        step_contrasts = []
                        prev_strs = []
                        for k, step in enumerate(split_steps):
                            if self.args.contrastive_adaptive == "kldiv":
                                contrast = self.contrast_probs(inputs, model, prev_strs, step)
                            else:
                                contrast = self.contrast_gen(inputs, model, prev_strs, stop_str, frames)
                            if k < len(split_steps) - 1:
                                prev_strs.append(step + "<end_of_step>\n")
                            else:
                                prev_strs.append(step)
                            step_contrasts.append((prev_strs[-1], contrast))

                    new_inputs = deepcopy(inputs)
                    new_inputs["input_ids"] = torch.cat([inputs["input_ids"][0], generated_tokens[0], new_input_tokens2], dim=0).unsqueeze(0)
                    generated_tokens, _ = self.prediction_generate(model, new_inputs, frames, do_sample, stopping_criteria)
                    if self.args.contrastive_adaptive in ["kldiv", "sample"]:
                        preds_str = "Reasoning: {}\nAnswer: {}".format(reasoning_pred, self.tokenizer.decode(generated_tokens[0]))
                        preds.append({"text": preds_str, "contrast": step_contrasts})
                    else:
                        preds.append("Reasoning: {}\nAnswer: {}".format(reasoning_pred, self.tokenizer.decode(generated_tokens[0])))
                preds = [preds]
            elif beam > 1 and rollout in ["prmtree", "prmdpotree"]:
                tree, preds = self.prediction_buildtree(model, inputs, frames, beam=beam, max_depth=10, interp_factor=interp_factor, rollout=rollout)
                pred_tuples = [(state[0], sum(state[1])/state[2]) for state in preds]
                pred_tuples = sorted(pred_tuples, key=lambda x: x[1], reverse=True)
                print("\n***Found {} hypothesis***\n".format(len(preds)))
                predlist = []
                for pred in pred_tuples:
                    pred = pred[0]
                    generated_tok = self.tokenizer(pred, return_tensors="pt").input_ids[0].to(model.device)
                    new_inputs = deepcopy(inputs)
                    new_inputs["input_ids"] = torch.cat([inputs["input_ids"][0], generated_tok, new_input_tokens2], dim=0).unsqueeze(0)
                    generated_tokens, score = self.generate_with_context(model, new_inputs, frames, rollout, stopping_criteria)
                    answer = self.tokenizer.decode(generated_tokens[0])
                    predlist.append("Reasoning: {}\nAnswer: {}".format(pred, answer))
                preds = [predlist]
            elif rollout == "contrastive_rollout" or rollout == "contrastive_expand_rollout":
                rollouts = self.prediction_rollout_contrast(model, inputs, frames, beam=beam, rollout=rollout)
                count = 0
                list_of_rollouts = []
                for prefix, rolloutlist in rollouts.items():
                    for rollout in rolloutlist:
                        full_solution = "".join(rollout["full_solution"])
                        generated_tok = self.tokenizer(full_solution, return_tensors="pt").input_ids[0].to(model.device)
                        new_inputs = deepcopy(inputs)
                        new_inputs["input_ids"] = torch.cat([inputs["input_ids"][0], generated_tok, new_input_tokens2], dim=0).unsqueeze(0)
                        generated_tokens, _ = self.prediction_generate(model, new_inputs, frames, do_sample, stopping_criteria)
                        rollout["answer"] = self.tokenizer.decode(generated_tokens[0])
                        count += 1
                    rollout_new_item = {"prefix": list(prefix), "rolloutlist": rolloutlist}
                    list_of_rollouts.append(rollout_new_item)
                print("Rollout {} solutions".format(count))
                preds = [list_of_rollouts]
            elif rollout == "contrastive_dfs":
                generated_tokens, score = self.prediction_generate(model, inputs, frames, False, stopping_criteria)
                onebest, alternatives = self.contrastive_dfs(model, inputs, frames, generated_tokens[0], score, beam=beam)
                rollouts = [(solution, score) for solution, score in alternatives.items()]
                rollouts = sorted(rollouts, key=lambda x: x[1], reverse=True)
                predlist = []
                for solution, score in rollouts:
                    newtokens = torch.tensor(self.tokenizer.encode(onebest)).to(model.device)
                    new_inputs = deepcopy(inputs)
                    new_inputs["input_ids"] = torch.cat([inputs["input_ids"][0], newtokens, new_input_tokens2], dim=0).unsqueeze(0)
                    generated_tokens, _ = self.prediction_generate(model, new_inputs, frames, do_sample, stopping_criteria)
                    predlist.append("Reasoning: {}\nAnswer: {}".format(solution, self.tokenizer.decode(generated_tokens[0])))
                preds = [predlist]
            else:
                generated_tokens, _ = self.prediction_generate(model, inputs, frames, do_sample, stopping_criteria)
                preds = [self.tokenizer.decode(t) for t in generated_tokens]
                # question = "A synthetised video usually contains unusual motions or distorted objects. Is the given video synthesized? Answer YES if it is, and NO if it is not. Pay attention to the moves and objects, and reason step by step. Clearly output your reasons. Mark the end of each step with <end_of_step> token.<|im_end|>\n<|im_start|>assistant\n"
                # question = "An AI generated video contains unnatural distorted things, such as distorted hands, faces or color. Is the given video AI generated? Answer YES or NO. Answer step by step and output each step clearly.<|im_end|>\n<|im_start|>assistant\n" # visual only antispoofing
                # question = "Question: Watch the video together with the audio carefully. Are there any places in the video where the audio does not match the video? Answer YES if it is modified or generated, and NO if it is not.\nAnswer this question step by step. Clearly output each step. Mark the end of each step with <end_of_step> token.<|im_end|>\n<|im_start|>assistant\n"
                # preds, generated_tokens = self.change_prompt(question, inputs, model, frames, do_sample, stopping_criteria)
                # preds = [preds]
                inputs["input_ids"] = torch.cat([inputs["input_ids"][0], generated_tokens[0], new_input_tokens2], dim=0).unsqueeze(0)
                generated_tokens, _ = self.prediction_generate(model, inputs, frames, do_sample, stopping_criteria)
                preds = ["Reasoning: {}\nAnswer: {}".format(preds[k], self.tokenizer.decode(t)) for k, t in enumerate(generated_tokens)]
            return ([None], preds, texts)
        else:
            inputs = self._prepare_inputs(inputs)
            inputs['images'] = [it.to(torch.bfloat16) for it in inputs['images']]
            inputs["raw_wav"] = inputs["raw_wav"].to(torch.bfloat16)
            inputs["spectrogram"] = inputs["spectrogram"].to(torch.bfloat16)
            
            ids = inputs.pop("ids", None)
            prompts = inputs.pop("prompts", None)
            labels = inputs.pop("labels", None)
            texts = inputs.pop("texts", None)

            if getattr(model, "do_rag", False) and model.rag_type == "replace":
                question = "Generate a short description for the video within 3 sentences.<|im_end|>\n<|im_start|>assistant\n"
                input_id = torch.cat([inputs["input_ids"][:, :17], self.tokenizer(question, return_tensors="pt")["input_ids"].to(inputs["input_ids"].device)], dim=-1)
                inputs['input_ids'] = input_id

            conv = conv_templates['qwen_1_5']
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, inputs['input_ids'])

            frames = inputs.get("frames")
            if frames:
                frames = [frame.to(torch.bfloat16) for frame in frames]

            generated_tokens, _ = self.prediction_generate(model, inputs, frames, do_sample, stopping_criteria)
            if getattr(model, "do_rag", False) and model.rag_type == "replace":
                preds = [generated_tokens]
                print("Total {} segments".format(len(generated_tokens)))
            else:
                preds = [self.tokenizer.decode(t) for t in generated_tokens]

            return ([None], preds, texts)

    def prediction_generate(self, model, inputs, frames, do_sample, stopping_criteria, buildtree=False):
        with torch.no_grad():
            if not do_sample:
                set_seed(2024)
            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                images=inputs["images"],
                image_sizes=inputs["image_sizes"],
                modalities=inputs["modalities"],
                raw_wav=inputs["raw_wav"],
                spectrogram=inputs["spectrogram"],
                org_groups=inputs["org_groups"],
                frames=frames,
                frame_sizes=inputs.get("frame_sizes"),
                max_new_tokens=self.args.max_new_tokens if not buildtree else 64,
                num_beams=1,
                do_sample=do_sample,
                stopping_criteria=[stopping_criteria],
                real_time=inputs['real_time'],
                top_p=0.9,
                output_scores=True,
                return_dict_in_generate=True,
                duration=inputs['duration'],
                captions=inputs["captions"],
            )
        if isinstance(generated_tokens, list):
            return generated_tokens, None
        all_scores = torch.log_softmax(torch.cat(generated_tokens["scores"], dim=0), dim=-1)
        indices = generated_tokens["sequences"][0]
        scores = all_scores[torch.arange(indices.size(0)), indices].sum().item()
        return generated_tokens["sequences"], scores

    def change_prompt(self, question, inputs, model, frames, do_sample, stopping_criteria):
        input_id = torch.cat([inputs["input_ids"][:, :17], self.tokenizer(question, return_tensors="pt")["input_ids"].to(inputs["input_ids"].device)], dim=-1)
        inputs["input_ids"] = input_id
        generated_tokens, _ = self.prediction_generate(model, inputs, frames, do_sample, stopping_criteria)
        preds = self.tokenizer.decode(generated_tokens[0])
        return preds, generated_tokens

    def predict(
        self,
        test_dataset,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        do_sample=False,
        do_reasoning=False,
        prediction_loss_only=False,
        rollout="none",
        beam=1,
        interp_factor=0,
    ):
        if "video" in getattr(self.model.base_model, "rag_type", "direct") and "train" not in self.model.base_model.rag_type:
            self.model.base_model.model.video_encoder = self.text_encoder.to(self.model.device)
        test_dataloader = self.get_test_dataloader(test_dataset)
        return self.prediction_loop(
            test_dataloader,
            description="Prediction",
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            do_sample=do_sample,
            do_reasoning=do_reasoning,
            predloss=prediction_loss_only,
            beam=beam,
            rollout=rollout,
            interp_factor=interp_factor,
        )

    def prediction_loop(
        self,
        dataloader,
        description,
        ignore_keys=None,
        metric_key_prefix="eval",
        do_sample=False,
        do_reasoning=False,
        predloss=False,
        rollout="none",
        beam=1,
        interp_factor=0,
    ):
        model = self.model
        batch_size = dataloader.batch_size

        if not predloss:
            self.model.eval()
        results = []
        total_loss = 0
        total_tokens = 0
        default_ids = ["/mnt/bn/tiktok-mm-4/aiic/users/guangzhisun/dataprep/NEXTQA/videos/7555434398.mp4", "/mnt/bn/tiktok-mm-4/aiic/users/guangzhisun/dataprep/NEXTQA/audios/7555434398.wav"]
        if dist.get_rank() == 0:
            for inputs in tqdm(dataloader):
                ids = inputs['ids'] if inputs["ids"][0] is not None else default_ids
                # print(ids)
                # ids = [eval(it) for it in ids]
                loss, preds, refs = self.prediction_step(
                    model,
                    inputs,
                    do_sample=do_sample,
                    prediction_loss_only=predloss,
                    do_reasoning=do_reasoning,
                    beam=beam,
                    rollout=rollout,
                    interp_factor=interp_factor,
                )
                if predloss:
                    total_loss += (loss * len(inputs['input_ids']))
                    total_tokens += len(inputs['input_ids'])
                else:
                    prompts = inputs['prompts']
                    prompts[0].append({"from": "gpt", "value": inputs["texts"][0]})
                    results.append((ids, prompts, preds, refs, loss))
        else:
            for inputs in dataloader:
                ids = inputs['ids'] if inputs["ids"][0] is not None else default_ids
                # ids = [eval(it) for it in ids]
                loss, preds, refs = self.prediction_step(
                    model,
                    inputs,
                    do_sample=do_sample,
                    prediction_loss_only=predloss,
                    do_reasoning=do_reasoning,
                    beam=beam,
                    rollout=rollout,
                    interp_factor=interp_factor,
                )
                if predloss:
                    total_loss += (loss * len(inputs['input_ids']))
                    total_tokens += len(inputs['input_ids'])
                else:
                    prompts = inputs['prompts']
                    prompts[0].append({"from": "gpt", "value": inputs["texts"][0]})
                    results.append((ids, prompts, preds, refs, loss))

        if predloss:
            return total_loss / total_tokens
        output_data = []
        for batch_item in results:
            for i in range(len(batch_item[0])):
                id, prp, pred, ref, loss = batch_item[0][i], batch_item[1][i], batch_item[2][i], batch_item[3][i], batch_item[4][i]
                output_data.append({
                    "id": id,
                    'prompt': prp,
                    "pred": pred,
                    "ref": ref,
                })
        return output_data

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        
        # if dist.get_rank() == 0:
        #     print(f"step {self.state.global_step} starts forward")
        if "video" in getattr(self.model.base_model, "rag_type", "direct") and "train" not in self.model.base_model.rag_type:
            self.model.base_model.model.video_encoder = self.text_encoder.to(self.model.device)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        # if dist.get_rank() == 0:
        #     print(f"step {self.state.global_step} ends forward")

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # if dist.get_rank() == 0:
        #     print(f"step {self.state.global_step} starts backward")
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        # if dist.get_rank() == 0:
        #     print(f"step {self.state.global_step} ends backward")

        return loss.detach() / self.args.gradient_accumulation_steps
