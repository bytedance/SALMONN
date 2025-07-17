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

# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl.txt'.

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

from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
from qwenvl.data.dataset import make_supervised_data_module
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, WhisperFeatureExtractor
from qwenvl.train.trainer import QwenVLTrainer

from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

from tqdm import tqdm
import torch.distributed as dist

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

    print("Applying Liger kernels to Qwen2.5-VL model...")

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from qwenvl.model import modeling_qwen2_5_vl

    if rope:
        modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
    if rms_norm:
        modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
    if swiglu:
        modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP


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
        model.visual.requires_grad_(True)
    else:
        model.visual.requires_grad_(False)

    if model_args.tune_mm_mlp:
        model.visual.merger.requires_grad_(True)
    else:
        model.visual.merger.requires_grad_(False)

    if model_args.tune_mm_audio:
        model.audio.requires_grad_(True)
    else:
        model.audio.requires_grad_(False)

    if model_args.tune_mm_qformer:
        model.audio.qformer.requires_grad_(True)
        model.audio.q_tokens.requires_grad_(True)
        model.audio.audio_proj.requires_grad_(True)
    else:
        model.audio.qformer.requires_grad_(False)
        model.audio.q_tokens.requires_grad_(False)
        model.audio.audio_proj.requires_grad_(False)

    if model_args.tune_mm_llm:
        if model_args.use_lora:
            raise Exception("tune_mm_llm is not supported when use_lora is True")
        model.model.requires_grad_(True)
        model.lm_head.requires_grad_(True)
    else:
        model.model.requires_grad_(False)
        model.lm_head.requires_grad_(False)


def train(attn_implementation="flash_attention_2"):
    global local_rank

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

    assert data_args.train_type in ["sft", "dpo", "gdpo", "grpo"], f"train_type {data_args.train_type} is not supported"

    training_args.remove_unused_columns = False

    apply_liger_kernel_to_qwen2_5_vl()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(
        model_args.model_base,
    )
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size, 
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )
    data_args.model_type = "qwen2.5vl"

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_base,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    

    if not data_args.run_test:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        # time.sleep(random.randint(0, 20))
        # print(f"RANK {dist.get_rank()} before barrier")
        dist.barrier(device_ids=dist.get_rank())
        # print(f"RANK {dist.get_rank()} after barrier")
        model = video_SALMONN2_plus.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        model.config.use_cache = False

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            if "3" not in training_args.deepspeed:
                if training_args.gradient_checkpointing_kwargs is None:
                    training_args.gradient_checkpointing_kwargs={"use_reentrant": False}
                else:
                    training_args.gradient_checkpointing_kwargs["use_reentrant"] = False

        if model_args.lora_ckpt != "No":
            from peft import PeftModel
            audio_layers = model.audio.layers
            del model.audio.layers
            model = PeftModel.from_pretrained(model, model_args.lora_ckpt)
            model.model.audio.layers = audio_layers
            model = model.merge_and_unload()
            model.save_pretrained(os.path.join(training_args.output_dir, "base/"))

        set_model(model_args, model)

        if training_args.no_audio:
            del model.audio

        if model_args.use_lora:
            from peft import LoraConfig, get_peft_model
            module_to_save = []
            if model_args.tune_mm_vision:
                module_to_save.append("visual")
            if model_args.tune_mm_mlp:
                module_to_save.append("visual.merger")
            if model_args.tune_mm_audio:
                module_to_save.append("audio")
            if model_args.tune_mm_qformer:
                module_to_save.append("audio.qformer")
                module_to_save.append("audio.q_tokens")
                module_to_save.append("audio.audio_proj")
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"], # find_all_linear_names(model),
                lora_dropout=model_args.lora_dropout,
                bias=model_args.lora_bias,
                task_type="CAUSAL_LM",
                modules_to_save=module_to_save,
            )
            if not training_args.no_audio:
                audio_layers = model.audio.layers
                del model.audio.layers
            model = get_peft_model(model, lora_config)
            if not training_args.no_audio:
                model.model.audio.layers = audio_layers

            for k, v in model.named_parameters():
                if "lora" in k:
                    v.requires_grad_(True)
        
        if dist.get_rank() == 0:
            for k, v in model.named_parameters():
                if v.requires_grad:
                    print(k, v.shape)
            # print(model.model.visual.merger)

        trainer = QwenVLTrainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module
        )
        
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logging.info("checkpoint found, resume training")
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()
        data_args.image_processor.save_pretrained(training_args.output_dir)

        model.config.use_cache = True

        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    else:
        pred_rank = training_args.pred_rank
        if torch.cuda.device_count() > 1:
            pred_rank = pred_rank * torch.cuda.device_count() + torch.cuda.current_device()
            data_args.dataset_use = f"dataset/{pred_rank}.json"
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

        os.makedirs(os.path.join(training_args.output_dir, training_args.run_name), exist_ok=True)

        if model_args.lora_ckpt != "No":
            if dist.get_rank() == 0:
                model = video_SALMONN2_plus.from_pretrained(
                    model_args.model_name_or_path,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    device_map="cpu"
                )
                from peft import PeftModel
                if not training_args.no_audio:
                    audio_layers = model.audio.layers
                    del model.audio.layers
                model = PeftModel.from_pretrained(model, model_args.lora_ckpt)
                if not training_args.no_audio:
                    model.model.audio.layers = audio_layers
                model = model.merge_and_unload()

                if torch.cuda.device_count() > 1:
                    model.save_pretrained(os.path.join(training_args.output_dir, "generation"))
                else:
                    model.save_pretrained(os.path.join(training_args.output_dir, f"generation_{pred_rank}"))
            dist.barrier(device_ids=local_rank)
            

        if torch.cuda.device_count() > 1:
            ds_config = {
                "fp16": {"enabled": False},
                "bf16": {"enabled": True},
                "zero_optimization": {
                    "stage": 3
                },
                "train_micro_batch_size_per_gpu": 1,
            }
            from transformers.integrations.deepspeed import HfDeepSpeedConfig
            hfdsc = HfDeepSpeedConfig(ds_config)

        if model_args.lora_ckpt == "No":
            model = video_SALMONN2_plus.from_pretrained(
                model_args.model_name_or_path,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        else:
            if torch.cuda.device_count() > 1:
                model = video_SALMONN2_plus.from_pretrained(
                    os.path.join(training_args.output_dir, "generation"),
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                )
            else:
                model = video_SALMONN2_plus.from_pretrained(
                    os.path.join(training_args.output_dir, f"generation_{pred_rank}"),
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                )
        if training_args.no_audio:
            del model.audio

        
        if torch.cuda.device_count() > 1:
            import deepspeed
            ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
            ds_engine.module.eval()
            model = ds_engine.module
        else:
            model.cuda()

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
        for inputs in tqdm(loader, desc=f"RANK {pred_rank}"):
            if inputs:
                res_i = {
                    "video": inputs.pop("video", None),
                    "image": inputs.pop("image", None),
                    "prompt": inputs.pop("prompt", None),
                    "ref": inputs.pop("ref", None),
                    "audio": inputs.pop("audio", None),
                    "use_audio": inputs.pop("use_audio", False),
                    "should_use": inputs.pop("should_use", True),
                }
                inputs = {k: v.to(f"cuda:{torch.cuda.current_device()}") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                for _ in range(data_args.num_sample):
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            do_sample=data_args.do_sample,
                            top_p=0.9)
                    output_trimmed = outputs[0, len(inputs["input_ids"][0]):]
                    output_text = tokenizer.decode(output_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    if data_args.num_sample == 1:
                        res_i["pred"] = output_text
                    else:
                        if "pred" in res_i:
                            res_i["pred"].append(output_text)
                        else:
                            res_i["pred"] = [output_text]
                if not res_i["should_use"]:
                    continue
                result.append(res_i)
        with open(os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
