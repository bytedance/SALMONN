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
import re
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

from qwenvl.model.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration
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

HDFS_BASE_PATH = "some HDFS Paths"  # Change to your own path

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

    from qwenvl.model import modeling_qwen3_vl_moe
    from qwenvl.model import modeling_qwen3_vl

    if rms_norm:
        modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm = LigerRMSNorm
        modeling_qwen3_vl.Qwen3VLTextRMSNorm = LigerRMSNorm
    if swiglu:
        modeling_qwen3_vl_moe.Qwen3VLMoeTextMLP = LigerSwiGLUMLP
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


def train(attn_implementation="flash_attention_3"):
    print("Start")
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
    training_args.video_max_frames = data_args.video_max_frames

    if "Fuse" in model_args.model_base:
        apply_fused_kernel_to_moe()
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
    data_args.distill_maxframes = 768 if "distill" in model_args.ttt_type else 0

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_base,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )
    

    if not data_args.run_test:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        if dist.get_rank() == 0:
            hmkdir(os.path.join(HDFS_BASE_PATH, training_args.output_dir))
        # time.sleep(random.randint(0, 20))
        # print(f"RANK {dist.get_rank()} before barrier")
        dist.barrier(device_ids=dist.get_rank())
        # print(f"RANK {dist.get_rank()} after barrier")
        if model_args.model_type == "moe":
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        elif model_args.model_type == "dense":
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        else:
            raise Exception(f"model_type {model_args.model_type} is not supported")
        model.config.use_cache = False

        # Stream
        model.model.fixed_memory_size = model_args.fixed_memory_size
        model.model.fixed_memory_size_audio = model_args.fixed_memory_size_audio
        model.model.stepsize = model_args.stepsize
        model.model.ttt_type = model_args.ttt_type
        pretrained_layers = []
        if "ttt" in model_args.ttt_type:
            all_ttt_layers = [int(layerid) for layerid in model_args.all_ttt_layers.split(",")]
            pretrained_layers = [int(layerid) for layerid in model_args.pretrained_ttt_layers.split(",") if int(layerid) >= 0]
            additional_layer = []
            if pretrained_layers != []:
                for layer_id in model_args.all_ttt_layers.split(","):
                    if int(layer_id) not in pretrained_layers:
                        additional_layer.append(int(layer_id))
                        all_ttt_layers.remove(int(layer_id))
            model.model.init_ttt_layers(
                num_heads=model_args.ttt_num_heads,
                ttt_hidden_size=model_args.ttt_hidden_size,
                CG_max_iter=model_args.cg_max_iter,
                freeze_ttt=training_args.freeze_ttt,
                all_ttt_layers=all_ttt_layers,
                search_type=model_args.search_type,
                lag_distances=model_args.lag_distances,
                ema_factor=model_args.ema_factor,
                slot_type=model_args.slot_type,
            )
        shrinksize_config = None
        if os.path.exists(model_args.shrinksize_config):
            with open(model_args.shrinksize_config) as f:
                shrinksize_config = json.load(f)
        model.model.init_mem_search(
            workingmemsize=model_args.workingmemsize,
            memgroupsize=model_args.memgroupsize,
            div_factor=model_args.div_factor,
            search_type=model_args.search_type,
            pretrained_ttt_layers=pretrained_layers,
            global_factor=model_args.global_factor,
            loss_factor=model_args.loss_factor,
            savemode_stoplayer=model_args.savemode_stoplayer,
            shrinksize=model_args.shrinksize if shrinksize_config is None else shrinksize_config,
        )

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            if training_args.deepspeed is not None and "3" not in training_args.deepspeed:
                if training_args.gradient_checkpointing_kwargs is None:
                    training_args.gradient_checkpointing_kwargs={"use_reentrant": False}
                else:
                    training_args.gradient_checkpointing_kwargs["use_reentrant"] = False

        if model_args.lora_ckpt != "No":
            from peft import PeftModel
            audio_layers = model.model.audio.layers
            del model.model.audio.layers
            print("Loading LoRA ckpt from: ", model_args.lora_ckpt)
            model = PeftModel.from_pretrained(model, model_args.lora_ckpt)
            model.model.model.audio.layers = audio_layers
            model = model.merge_and_unload()
            model.save_pretrained(os.path.join(training_args.output_dir, "base/"))
            dist.barrier()
            if dist.get_rank() == 0:
                hcopy(os.path.join(training_args.output_dir, "base"), os.path.join(HDFS_BASE_PATH, training_args.output_dir))
                print(os.path.join(HDFS_BASE_PATH, training_args.output_dir, "base"))

        if pretrained_layers != [] and additional_layer != []:
            model.model.additional_ttt_layer(additional_layer)

        set_model(model_args, model)

        if training_args.no_audio:
            del model.model.audio

        if model_args.use_lora:
            from peft import LoraConfig, get_peft_model
            modules_to_save = []
            if model_args.tune_mm_vision:
                modules_to_save.append("visual")
            if model_args.tune_mm_mlp:
                modules_to_save.append("visual.merger")
            if model_args.tune_mm_audio:
                modules_to_save.append("audio")
            if model_args.tune_mm_qformer:
                modules_to_save.append("audio.qformer")
                modules_to_save.append("audio.q_tokens")
                modules_to_save.append("audio.audio_proj")
            if "ttt" in model_args.ttt_type:
                modules_to_save.append("ttt_layers")
                modules_to_save.append("ttt_gating")
                modules_to_save.append("ttt_select_gating")
                modules_to_save.append("ttt_gating_enc")
                modules_to_save.append("ttt_selection_proj")
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"], # find_all_linear_names(model),
                lora_dropout=model_args.lora_dropout,
                bias=model_args.lora_bias,
                task_type="CAUSAL_LM",
                modules_to_save=modules_to_save,
            )
            if not training_args.no_audio:
                audio_layers = model.model.audio.layers
                del model.model.audio.layers
            model = get_peft_model(model, lora_config)
            if not training_args.no_audio:
                model.model.model.audio.layers = audio_layers

            for k, v in model.named_parameters():
                if "lora" in k and not training_args.freeze_lora:
                    v.requires_grad_(True)
                elif "lora" in k and training_args.freeze_lora:
                    v.requires_grad_(False)

        if "ttt" in model_args.ttt_type:
            for k, v in model.named_parameters():
                if "ttt" in k and not training_args.freeze_ttt:
                    if "layers." in k and int(re.findall("layers.[0-9]+", k)[0].split(".")[-1]) in pretrained_layers:
                        print("not updating: ", k)
                        v.requires_grad_(False)
                    else:
                        v.requires_grad_(True)
                elif "ttt" in k and training_args.freeze_ttt:
                    v.requires_grad_(False)
                if "search" in k:
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
            pred_rank = pred_rank * 8 + torch.cuda.current_device()
            data_args.dataset_use = f"dataset/{pred_rank}.json"
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        
        if pred_rank == 0:
            hmkdir(os.path.join(HDFS_BASE_PATH, training_args.output_dir, training_args.run_name))

        os.makedirs(os.path.join(training_args.output_dir, training_args.run_name), exist_ok=True)

        if torch.cuda.current_device() == 0:
            if model_args.model_type == "moe":
                model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                    model_args.model_name_or_path,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    device_map="cpu"
                )
            elif model_args.model_type == "dense":
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_args.model_name_or_path,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    device_map="cpu"
                )
            else:
                raise Exception(f"model_type {model_args.model_type} is not supported")

            # Stream
            model.model.fixed_memory_size = model_args.fixed_memory_size
            model.model.fixed_memory_size_audio = model_args.fixed_memory_size_audio
            model.model.stepsize = model_args.stepsize
            model.model.ttt_type = model_args.ttt_type
            if "ttt" in model_args.ttt_type:
                all_ttt_layers = [int(layerid) for layerid in model_args.all_ttt_layers.split(",")]
                model.model.init_ttt_layers(
                    num_heads=model_args.ttt_num_heads,
                    ttt_hidden_size=model_args.ttt_hidden_size,
                    CG_max_iter=model_args.cg_max_iter,
                    slot_type=model_args.slot_type,
                    ema_factor=model_args.ema_factor,
                    lag_distances=model_args.lag_distances,
                    all_ttt_layers=all_ttt_layers,
                    search_type=model_args.search_type,
                )
            shrinksize_config = None
            if os.path.exists(model_args.shrinksize_config):
                with open(model_args.shrinksize_config) as f:
                    shrinksize_config = json.load(f)
            model.model.init_mem_search(
                workingmemsize=model_args.workingmemsize,
                memgroupsize=model_args.memgroupsize,
                div_factor=model_args.div_factor,
                search_type=model_args.search_type,
                global_factor=model_args.global_factor,
                loss_factor=model_args.loss_factor,
                savemode_stoplayer=model_args.savemode_stoplayer,
                shrinksize=model_args.shrinksize if shrinksize_config is None else shrinksize_config,
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

        trainer = QwenVLTrainer(
            model=model, processing_class=tokenizer, args=training_args, **data_module
        )

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
        for idx, inputs in tqdm(enumerate(loader), total=len(loader), desc=f"RANK {pred_rank}"):
            if inputs:
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
                # inputs = {k: v.to(f"cuda:{torch.cuda.current_device()}") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                for _ in range(data_args.num_sample):
                    print(res_i["video"])
                    preds = trainer.prediction_step(
                        model,
                        inputs,
                        prompts=res_i["prompt"],
                        do_sample=data_args.do_sample,
                        processor=tokenizer,
                    )
                    torch.cuda.empty_cache()
                    res_i["pred"] = preds
                    print(preds)
                if not res_i["should_use"]:
                    continue
                result.append(res_i)

        hmkdir(os.path.join(HDFS_BASE_PATH, training_args.output_dir, training_args.run_name))
        with open(os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.json"), "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        hcopy(
            os.path.join(training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.json"),
            os.path.join(HDFS_BASE_PATH, training_args.output_dir, training_args.run_name, f"test_results_rank{pred_rank}.json")
        )
        print(os.path.join(HDFS_BASE_PATH, training_args.output_dir, training_args.run_name))
        return

if __name__ == "__main__":
    train(attn_implementation="flash_attention_3")
