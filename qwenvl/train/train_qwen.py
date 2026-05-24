# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
import random
import time

from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# from qwenvl.model.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
from qwenvl.model.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.processing_qwen3_vl import Qwen3VLProcessor
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, WhisperFeatureExtractor
from qwenvl.train.trainer import QwenVLTrainer

from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

from tqdm import tqdm
import torch.distributed as dist

try:
    from cruise.utilities.distributed import DIST_ENV
    from cruise.utilities.hdfs_io import hcopy, hmkdir
except:
    pass

import qwenvl.train.deepspeed_patch

local_rank = None

def collate_fn(batch):
    return batch[0]

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def apply_fused_kernel_to_moe():
    print("Applying fused kernel to MoE")
    from qwenvl.model.qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedSparseMoeBlock
    from qwenvl.model import modeling_qwen3_vl_moe

    modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock = Qwen3MoeFusedSparseMoeBlock

def apply_liger_kernel_to_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2.5-VL models.
    NOTE: Qwen2.5-VL is not available in transformers<4.48.2

    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """

    print("Applying Liger kernels to Qwen3-VL model...")

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    # from qwenvl.model import modeling_qwen3_vl_moe
    from qwenvl.model import modeling_qwen3_vl

    if rms_norm:
        # modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm = LigerRMSNorm
        modeling_qwen3_vl.Qwen3VLTextRMSNorm = LigerRMSNorm
    if swiglu:
        # modeling_qwen3_vl_moe.Qwen3VLMoeTextMLP = LigesrSwiGLUMLP
        modeling_qwen3_vl.Qwen3VLTextMLP = LigerSwiGLUMLP


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        model.model.visual.requires_grad_(True)
    else:
        model.model.visual.requires_grad_(False)

    if model_args.tune_mm_mlp:
        model.model.visual.merger.requires_grad_(True)
    else:
        model.model.visual.merger.requires_grad_(False)

    if model_args.tune_mm_audio:
        model.model.audio.requires_grad_(True)
    else:
        model.model.audio.requires_grad_(False)

    if model_args.tune_mm_qformer:
        model.model.audio.qformer.requires_grad_(True)
        model.model.audio.q_tokens.requires_grad_(True)
        model.model.audio.audio_proj.requires_grad_(True)
    else:
        model.model.audio.qformer.requires_grad_(False)
        model.model.audio.q_tokens.requires_grad_(False)
        model.model.audio.audio_proj.requires_grad_(False)

    if model_args.tune_mm_llm:
        if model_args.use_lora:
            raise Exception("tune_mm_llm is not supported when use_lora is True")
        model.model.language_model.requires_grad_(True)
        model.lm_head.requires_grad_(True)
    else:
        model.model.language_model.requires_grad_(False)
        model.lm_head.requires_grad_(False)


def prediction_step(
    model,
    inputs,
    processor,
):
    preds = []
    with torch.no_grad():
        generated_tokens = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        preds = processor.decode(generated_tokens[0][inputs["input_ids"].size(1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return preds


def train(attn_implementation="flash_attention_3"):
    print("Start")

    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False
    training_args.video_max_frames = data_args.video_max_frames
    apply_liger_kernel_to_qwen2_5_vl()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    data_args.video_processor = Qwen3VLProcessor.from_pretrained(
        model_args.model_base,
    ).video_processor
    data_args.image_processor = Qwen3VLProcessor.from_pretrained(
        model_args.model_base,
    ).image_processor
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size, 
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )
    data_args.model_type = "qwen3vl"
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_base,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    os.makedirs(os.path.join(training_args.output_dir, training_args.run_name), exist_ok=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        device_map="cpu"
    )
    # Stream
    model.model.fixed_memory_size = model_args.fixed_memory_size
    model.model.fixed_memory_size_audio = model_args.fixed_memory_size_audio
    model.model.stepsize = model_args.stepsize
    model.model.ttt_type = model_args.ttt_type
    if "ttt" in model_args.ttt_type:
        model.model.init_ttt_layers(
            num_heads=model_args.ttt_num_heads,
            ttt_hidden_size=model_args.ttt_hidden_size,
            CG_max_iter=model_args.cg_max_iter,
            slot_type=model_args.slot_type,
            ema_factor=model_args.ema_factor,
            lag_distances=model_args.lag_distances,
        )
    model.model.init_mem_search(
        workingmemsize=model_args.workingmemsize,
        memgroupsize=model_args.memgroupsize,
        retain_factor=model_args.retain_factor,
        lambdas=model_args.lambdas,
        div_factor=model_args.div_factor,
    )

    if model_args.lora_ckpt != "No":
        if torch.cuda.current_device() == 0:
            from peft import PeftModel
            if not training_args.no_audio:
                audio_layers = model.model.audio.layers
                del model.model.audio.layers
            model = PeftModel.from_pretrained(model, model_args.lora_ckpt)
            model = model.to(torch.bfloat16) if training_args.bf16 else model
            if not training_args.no_audio:
                model.model.model.audio.layers = audio_layers
            model = model.merge_and_unload()
    else:
        model = model.to(torch.bfloat16) if training_args.bf16 else model

    model.cuda()
    training_args.base_interval = data_args.base_interval

    # trainer = QwenVLTrainer(
    #     model=model, processing_class=tokenizer, args=training_args, **data_module
    # )

    result = []
    test_data = data_module["train_dataset"]
    loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=collate_fn,
        in_order=False
    )
    for idx, inputs in tqdm(enumerate(loader), total=len(loader)):
        res_i = {
            "tos_key": inputs.pop("tos_key", None),
            "video": inputs.pop("video", None),
            "image": inputs.pop("image", None),
            "prompt": inputs.pop("prompt", None),
            "ref": inputs.pop("ref", None),
            "audio": inputs.pop("audio", None),
            "tos_audio": inputs.pop("tos_audio", None),
            "use_audio": inputs.pop("use_audio", False),
            "should_use": inputs.pop("should_use", True),
            "pred": [],
        }
        to_pop = []
        for k, v in inputs.items():
            if k != "pixel_values_videos" and isinstance(v, torch.Tensor):
                inputs[k] = v.to(f"cuda:{torch.cuda.current_device()}")
            elif not isinstance(v, torch.Tensor):
                to_pop.append(k)
        for k in to_pop:
            inputs.pop(k)
        print(res_i["video"])
        preds = prediction_step(
            model,
            inputs,
            tokenizer,
        )
        torch.cuda.empty_cache()
        res_i["pred"] = preds
        result.append(res_i)
    with open(os.path.join(training_args.output_dir, training_args.run_name, "test_results.json"), "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    return

if __name__ == "__main__":
    train(attn_implementation="flash_attention_3")
