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
import copy
from dataclasses import dataclass, field, asdict
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import ast
import sys
import torch
from datetime import datetime
import time
import random
import torch.distributed as dist
import transformers
import tokenizers
from peft import PeftModel
import yaml

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer, HDFS_BASE_PATH

try:
    from cruise.utilities.distributed import DIST_ENV
    from cruise.utilities.hdfs_io import hcopy, hmkdir
except:
    pass

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token

from PIL import Image
from decord import VideoReader, cpu

import cv2
from llava.model.utils import load_qwen_lora_model

from packaging import version

import numpy as np

from transformers import AutoConfig

import math

from llava.dataset import make_supervised_data_module, make_test_data_module

# from datasets import load_metric

from transformers import TrainerCallback
import shutil


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="average")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    patchify_video_feature: bool = field(default=False)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)
    mm_vlmattention_pretrained: Optional[str] = field(default=None)
    mm_vlmattention_bert_type: Optional[str] = field(default="qformer_pretrain")
    mm_vlmattention_num_query: Optional[int] = field(default=32)
    mm_vlmattention_compress_type: Optional[str] = field(default=None)
    from_stage_1_5: Optional[bool] = field(default=False)
    mm_pooling_position: Optional[str] = field(default="before")
    mm_newline_position: Optional[str] = field(default="grid")
    modality_max_length: Optional[str] = field(default="None")
    audio_visual: bool = False
    whisper_path: str = "/mnt/bn/tiktok-mm-2/aiic/public/model/whisper-large-v3"
    freeze_whisper: bool = True
    beats_path: str = None
    freeze_beats: bool = True
    use_speech_Qformer: bool = True
    num_speech_query_token: int = 1
    freeze_speech_QFormer: bool = False
    window_level_Qformer: bool = True
    second_per_window: float = 0.333333
    second_stride: float = 0.333333
    salmonn_path: str = None
    use_final_linear: bool = False
    freeze_final_linear: bool = False
    use_niv: bool = False
    niv_in_channels: int = 4
    niv_out_channels: int = 401
    niv_cnn_params: str = "[(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]"
    whisper_lora: bool = False
    flash_attn: bool = False
    multi_frame_projector: bool = False
    multi_frame_num: int = 30
    add_time_token: bool = False
    use_mfcnn: bool = False
    use_mftrans: bool = False
    use_flash_tower: bool = False
    mf_split_init: bool = False
    spt_projector: bool = False
    segmentation: int = -1
    segoverlapping: int = -1
    do_rag: bool = False
    rag_input_frames: int = 30
    rag_type: str = "direct"
    rag_topk: int = 5
    streamdecoder: bool = False

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    test_data_path: str = field(default=None, metadata={"help": "Path to the test data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    video_token: Optional[int] = field(default=2)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: int = 224
    image_split_resolution: int = 224
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: Optional[bool] = field(default=False)
    use_audio: bool = False
    audio_processor : str = "/mnt/bn/tiktok-mm-2/aiic/public/model/whisper-large-v3"
    random_video: bool = False
    val_ratio: float = 0.0
    val_path: str = None
    config_path: str = None
    max_time: int = 30
    use_tos: bool = False
    online_self_dpo: bool = False
    insert_time_precision: int = 0
    include_time_prompt: bool = False
    insert_image_num: int = 1
    use_only_insert: bool = False
    text_only_ratio: float = 0.0
    random_text_type: bool = False
    caption_data: str = ""


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    use_dora: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    train_llm_layers: str = None
    seed: int = 2024
    save_torch_ckpt: bool = True
    load_from_lora: bool = False
    run_test: bool = False
    test_output_dir: str = None
    evaluation_strategy: str = "steps"
    eval_steps: int = 1000
    disable_tqdm: bool = True
    lora_path: str = None
    save_lora_only: bool = False
    do_reasoning: bool = False
    do_sample: bool = False
    model_base: str = None
    lora_llm_only: bool = False
    max_new_tokens: int = 1024
    merge_and_reload: bool = False
    prediction_loss_only: bool = False
    ckpt: str = None
    do_demo: bool = False
    pretrain_weight: str = None
    ddp_timeout: int = 3600
    train_orm: str = ""
    beam: int = 1
    rollout: str = "none"
    vcdalpha: float = 0.0
    vcdbeta: float = 0.0
    vcdlogp: bool = False
    contrastive_adaptive: str = "none"
    load_full: bool = False
    interp_factor: float = 0.0


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector', 'vision_resampler']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

# def compute_metrics(eval_pred):
#     with torch.no_grad():
#         metrics = ["accuracy", "recall", "precision", "f1"] # List of metrics to return
#         metric={}
#         for met in metrics:
#             metric[met] = load_metric(met)
#         logits, labels = eval_pred
#         predictions = np.argmax(logits, axis=-1)
#         metric_res={}
#         for met in metrics:
#             metric_res[met]=metric[met].compute(predictions=predictions, references=labels)[met]
#         return metric_res

def train():
    global local_rank
    current_time = datetime.now()
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.pretrain_mm_mlp_adapter == 'None':
        model_args.pretrain_mm_mlp_adapter = None

    if data_args.config_path is not None:
        config_path = data_args.config_path
        output_dir = training_args.output_dir
        with open(f"{config_path}/model_args.json", "r") as f:
            dct = json.load(f)
            model_args = ModelArguments(**dct)
        parser_training = transformers.HfArgumentParser(TrainingArguments)
        training_args, = parser_training.parse_json_file(json_file=f"{config_path}/training_args.json")
        with open(f"{config_path}/data_args.json", "r") as f:
            dct = json.load(f)
            data_args = DataArguments(**dct)
        data_args.config_path = config_path
        training_args.output_dir = output_dir
    
    if dist.get_rank() == 0:
        if not training_args.run_test:
            if not os.path.exists(f"{training_args.output_dir}/train_config"):
                os.makedirs(f"{training_args.output_dir}/train_config", exist_ok=True)
            with open(f"{training_args.output_dir}/train_config/model_args.json", "w") as f:
                f.write(json.dumps(asdict(model_args)))
            with open(f"{training_args.output_dir}/train_config/data_args.json", "w") as f:
                f.write(json.dumps(asdict(data_args)))
            with open(f"{training_args.output_dir}/train_config/training_args.json", "w") as f:
                f.write(training_args.to_json_string())
            
            if True: # data_args.use_tos:
                hmkdir(os.path.join(HDFS_BASE_PATH, training_args.output_dir))
                hmkdir(os.path.join(HDFS_BASE_PATH, f"{training_args.output_dir}/train_config"))
                hcopy(f"{training_args.output_dir}/train_config/model_args.json", os.path.join(HDFS_BASE_PATH, f"{training_args.output_dir}/train_config/model_args.json"), chunk_thread_num=1)
                hcopy(f"{training_args.output_dir}/train_config/data_args.json", os.path.join(HDFS_BASE_PATH, f"{training_args.output_dir}/train_config/data_args.json"), chunk_thread_num=1)
                hcopy(f"{training_args.output_dir}/train_config/training_args.json", os.path.join(HDFS_BASE_PATH, f"{training_args.output_dir}/train_config/training_args.json"), chunk_thread_num=1)

        
    training_args.use_tos = data_args.use_tos
    data_args.train_orm = training_args.train_orm
    data_args.do_rag = model_args.do_rag
    data_args.rag_type = model_args.rag_type
    model_args.fps = data_args.video_fps
    model_args.lora_enable = training_args.lora_enable
    model_args.lora_r = training_args.lora_r
    model_args.lora_alpha = training_args.lora_alpha
    model_args.lora_dropout = training_args.lora_dropout


    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    seed = training_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")

    audio_config = dict(
        video_fps=data_args.video_fps,
        whisper_path=model_args.whisper_path,
        freeze_whisper=model_args.freeze_whisper,
        beats_path=model_args.beats_path,
        freeze_beats=model_args.freeze_beats,
        use_speech_Qformer = model_args.use_speech_Qformer,
        num_speech_query_token = model_args.num_speech_query_token,
        freeze_speech_QFormer = model_args.freeze_speech_QFormer,
        window_level_Qformer = model_args.window_level_Qformer,
        second_per_window = model_args.second_per_window,
        second_stride = model_args.second_stride,
        salmonn_path = model_args.salmonn_path,
        use_final_linear=model_args.use_final_linear,
        use_niv=model_args.use_niv,
        niv_in_channels=model_args.niv_in_channels,
        niv_out_channels=model_args.niv_out_channels,
        niv_cnn_params=model_args.niv_cnn_params,
        segmentation=model_args.segmentation,
        streamdecoder=model_args.streamdecoder,
        train_orm=training_args.train_orm,
        do_rag=model_args.do_rag,
        rag_input_frames=model_args.rag_input_frames,
        rag_type=model_args.rag_type,
        rag_topk=model_args.rag_topk,
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if not training_args.load_from_lora:
        if model_args.vision_tower is not None:
            if 'mpt' in model_args.model_name_or_path:
                model = LlavaMptForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
            elif 'mixtral' in model_args.model_name_or_path.lower():
                model = LlavaMixtralForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation="flash_attention_2",
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
            elif True: # 'qwen' in model_args.model_name_or_path.lower():
                audio_config.update(bnb_model_from_pretrained_args)
                if True: # not model_args.pretrain_mm_mlp_adapter:
                    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
                    overwrite_config = {"model_args": vars(model_args), "multi_frame_projector": model_args.multi_frame_projector, "multi_frame_num": model_args.multi_frame_num, "add_time_token": model_args.add_time_token, "use_mfcnn": model_args.use_mfcnn, "use_mftrans": model_args.use_mftrans, "use_flash_tower": model_args.use_flash_tower, "mf_split_init": model_args.mf_split_init, "spt_projector": model_args.spt_projector}
                    # overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
                    # overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
                    # overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

                    print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)
                    if not model_args.audio_visual:
                        model = LlavaQwenForCausalLM.from_pretrained(
                            model_args.model_name_or_path,
                            config=cfg_pretrained,
                            cache_dir=training_args.cache_dir,
                            attn_implementation="flash_attention_2",
                            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                            **audio_config
                        )
                    else:
                        model = LlavaAVQwenForCausalLM.from_pretrained(
                            model_args.model_name_or_path,
                            config=cfg_pretrained,
                            cache_dir=training_args.cache_dir,
                            attn_implementation="flash_attention_2",
                            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                            **audio_config
                        )

                else:
                    if not model_args.audio_visual:
                        model = LlavaQwenForCausalLM.from_pretrained(
                            model_args.model_name_or_path,
                            cache_dir=training_args.cache_dir,
                            attn_implementation="flash_attention_2",
                            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                            **audio_config
                        )
                    else:
                        model = LlavaAVQwenForCausalLM.from_pretrained(
                            model_args.model_name_or_path,
                            cache_dir=training_args.cache_dir,
                            attn_implementation="flash_attention_2",
                            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                            **audio_config
                        )
                    
            elif 'mistral' in model_args.model_name_or_path.lower() or 'zephyr' in model_args.model_name_or_path.lower():
                if not model_args.pretrain_mm_mlp_adapter:
                    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
                    overwrite_config = {}
                    # overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
                    # overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
                    # overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

                    print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)

                if model_args.pretrain_mm_mlp_adapter:
                    model = LlavaMistralForCausalLM.from_pretrained(
                        model_args.model_name_or_path,
                        cache_dir=training_args.cache_dir,
                        attn_implementation="flash_attention_2",
                        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                        **bnb_model_from_pretrained_args
                    )
                else:
                    model = LlavaMistralForCausalLM.from_pretrained(
                        model_args.model_name_or_path,
                        config=cfg_pretrained,
                        cache_dir=training_args.cache_dir,
                        attn_implementation="flash_attention_2",
                        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                        **bnb_model_from_pretrained_args
                    )         
            else:
                # finetune from a image trained model
                if not model_args.pretrain_mm_mlp_adapter:
                    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
                    overwrite_config = {}
                    # overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
                    # overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
                    # overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

                    if "224" in model_args.vision_tower:
                        # suppose the length of text tokens is around 1000, from bo's report
                        least_token_number = data_args.frames_upbound*(16//model_args.mm_spatial_pool_stride)**2 + 1000
                    else:
                        least_token_number = data_args.frames_upbound*(24//model_args.mm_spatial_pool_stride)**2 + 1000
                    
                    scaling_factor = math.ceil(least_token_number/4096)
                    if scaling_factor >= 2:
                        overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                        overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                        overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor
                        training_args.model_max_length = 4096 * scaling_factor

                    print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(cfg_pretrained, k, v)

                if not model_args.audio_visual:            
                    if transformers.__version__ == "4.31.0":
                        model = LlavaLlamaForCausalLM.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, config=cfg_pretrained, torch_dtype=(torch.bfloat16 if training_args.bf16 else None), **bnb_model_from_pretrained_args)
                        print("Using LlavaLlamaForCausalLM from transformers==4.31.0")
                    else:
                        if model_args.pretrain_mm_mlp_adapter:
                            ## For the stage2 ft, we should not load the config of a pure vicuna model
                            model = LlavaLlamaForCausalLM.from_pretrained(
                                model_args.model_name_or_path, cache_dir=training_args.cache_dir, attn_implementation="flash_attention_2", torch_dtype=(torch.bfloat16 if training_args.bf16 else None), **bnb_model_from_pretrained_args
                            )
                        else:
                            model = LlavaLlamaForCausalLM.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, attn_implementation="flash_attention_2", config=cfg_pretrained, torch_dtype=(torch.bfloat16 if training_args.bf16 else None), **bnb_model_from_pretrained_args)
                else:
                    if True: # [TODO] if ckpt_path not exists:
                        # llava_model_temp = LlavaLlamaForCausalLM.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, attn_implementation="flash_attention_2", config=cfg_pretrained, torch_dtype=(torch.bfloat16 if training_args.bf16 else None), **bnb_model_from_pretrained_args)
                        model = LlavaAVLlamaForCausalLM.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, attn_implementation="flash_attention_2", config=cfg_pretrained, torch_dtype=(torch.bfloat16 if training_args.bf16 else None), **audio_config)
                        # model = LlavaAVLlamaForCausalLM(config=cfg_pretrained, **audio_config)
                        # model.model = llava_model_temp.model
                        # model.lm_head = llava_model_temp.lm_head
        else:
            model = transformers.LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation="flash_attention_2",
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )

        model.config.use_cache = False
        if model_args.freeze_backbone:
            model.model.requires_grad_(False)
            model.lm_head.requires_grad_(False)

        if training_args.bits in [4, 8]:
            from peft import prepare_model_for_kbit_training
            model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if training_args.lora_enable:
            nonLora_path = os.path.join(training_args.ckpt, "nonLora.bin")
            if os.path.exists(nonLora_path):
                nonLora_ckpt = torch.load(nonLora_path, map_location='cpu')
                kk = model.load_state_dict(nonLora_ckpt, strict=False)
                if dist.get_rank() == 0:
                    print("Load nonLora: ", len(kk.unexpected_keys), len(kk.missing_keys))
                
                v_flag = False
                if hasattr(model.model, "vision_tower"):
                    vision_tower = model.model.vision_tower
                    del model.model.vision_tower
                    v_flag = True
                
                model = PeftModel.from_pretrained(model, training_args.ckpt)
                print("Peft Load Lora")
                if v_flag:
                    model.model.vision_tower = vision_tower
            else:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=training_args.lora_r,
                    lora_alpha=training_args.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj"], # find_all_linear_names(model),
                    lora_dropout=training_args.lora_dropout,
                    bias=training_args.lora_bias,
                    task_type="CAUSAL_LM",
                    use_dora=training_args.use_dora,
                )

                if training_args.bits == 16:
                    if training_args.bf16:
                        model.to(torch.bfloat16)
                    if training_args.fp16:
                        model.to(torch.float16)
                rank0_print("Adding LoRA adapters...")

                if model_args.audio_visual:
                    speech_encoder = model.speech_encoder
                    model.speech_encoder = None
                    
                    v_flag = False
                    if hasattr(model, "vision_tower"):
                        vision_tower = model.vision_tower
                        model.vision_tower = None
                        v_flag = True

                    model = get_peft_model(model, lora_config)
                    model.model.speech_encoder = speech_encoder
                    
                    if v_flag:
                        model.model.model.vision_tower = vision_tower
                        
                else:
                    rag_flag = False
                    vidrag_flag = False
                    if hasattr(model, "text_encoder"):
                        text_encoder = model.text_encoder
                        del model.text_encoder
                        rag_flag = True
                    if hasattr(model, "video_encoder"):
                        video_encoder = model.video_encoder
                        del model.video_encoder
                        vidrag_flag = True
                    v_flag = False
                    if hasattr(model.model, "vision_tower"):
                        vision_tower = model.model.vision_tower
                        del model.model.vision_tower
                        v_flag = True

                        model = get_peft_model(model, lora_config)
                    
                    if v_flag:
                        model.model.model.vision_tower = vision_tower
                    if rag_flag:
                        model.model.text_encoder = text_encoder
                    if vidrag_flag:
                        model.model.video_encoder = video_encoder

            model.to(torch.bfloat16)

        if False: # 'mpt' in model_args.model_name_or_path:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right"
            )
        elif False: # 'mistral' in model_args.model_name_or_path.lower() or 'mixtral' in model_args.model_name_or_path.lower() or 'zephyr' in model_args.model_name_or_path.lower():
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="left"
            )
        elif True: # "qwen" in model_args.model_name_or_path.lower():
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
    
    else:
        model, tokenizer = load_qwen_lora_model(
            training_args.ckpt,
            model_base=training_args.model_base,
            lora_enable=training_args.lora_enable,
            audio_visual=model_args.audio_visual,
            pretrain_weight=training_args.pretrain_weight,
            use_dora=training_args.use_dora,
            load_full=training_args.load_full,
            lora_r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            **audio_config,
        )
        # if training_args.use_dora:
        #     model = model.merge_and_unload()
        if training_args.merge_and_reload:
            tmp_dir = "/opt/tiger/model_demo"
            if dist.get_rank() == 0:
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)
                model = model.merge_and_unload()
                model.to(torch.bfloat16)
                model.save_pretrained(tmp_dir)
            dist.barrier()
            # model = LlavaAVQwenForCausalLM.from_pretrained(tmp_dir, low_cpu_mem_usage=True, device_map="cuda", attn_implementation="flash_attention_2", **audio_config)
            # model.to(torch.bfloat16)
        model.config.use_cache = False
        if model_args.freeze_backbone:
            model.model.requires_grad_(False)
            model.lm_head.requires_grad_(False)
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = True
        
        if training_args.bits in [4, 8]:
            from peft import prepare_model_for_kbit_training
            model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version == "qwen_1_5":
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.pad_token_id = tokenizer.bos_token_id
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    if True: 
        if model_args.vision_tower is not None:
            model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = copy.deepcopy(vision_tower.image_processor)
        # if "video" in model_args.rag_type:
        #     data_args.retrieve_image_processor = model.text_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        if model_args.spt_projector:
            model_args.mm_pooling_position = "no"
        assert model_args.mm_pooling_position in ["before", "after", "no"], "mm_pooling_position must be either 'before' or 'after'"
        model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride
        model.config.mm_pooling_position = model_args.mm_pooling_position
        model.config.mm_spatial_pool_mode = model_args.mm_spatial_pool_mode
        model.config.modality_max_length = model_args.modality_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler

        if training_args.lora_path != None:
            new_state_dict = torch.load(training_args.lora_path, map_location=torch.device('cpu'))
            for key in new_state_dict:
                if key[len("module."):] in model.state_dict():
                    model.state_dict()[key[len("module."):]].copy_(new_state_dict[key])
                    print(f"Loading {key}")
                elif f"base_model.model.{key[len('module.'):]}" in model.state_dict():
                    model.state_dict()[f"base_model.model.{key[len('module.'):]}"].copy_(new_state_dict[key])
                    print(f"Loading {key}")
                else:
                    print(f"error: {key} not found")

        if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
            model.requires_grad_(False)

        if training_args.lora_enable:
            for n, p in model.named_parameters():
                if 'lora' in n:
                    p.requires_grad = True

        if model_args.audio_visual:
            if not model_args.freeze_whisper:
                for p in model.speech_encoder.parameters():
                    p.requires_grad = True
                for p in model.ln_speech.parameters():
                    p.requires_grad = True
            else:
                for p in model.speech_encoder.parameters():
                    p.requires_grad = False
                for p in model.ln_speech.parameters():
                    p.requires_grad = False
                model.speech_encoder.eval()

            if model_args.whisper_lora:
                for n, p in model.speech_encoder.named_parameters():
                    if 'lora' in n:
                        p.requires_grad = True
                model.speech_encoder.train()

            if model_args.beats_path:
                if not model_args.freeze_beats:
                    for p in model.beats.parameters():
                        p.requires_grad = True
                    for p in model.ln_audio.parameters():
                        p.requires_grad = True
                else:
                    for p in model.beats.parameters():
                        p.requires_grad = False
                    for p in model.ln_audio.parameters():
                        p.requires_grad = False
                    model.beats.eval()

            if model_args.use_speech_Qformer:
                if not model_args.freeze_speech_QFormer:
                    for p in model.speech_Qformer.parameters():
                        p.requires_grad = True
                    model.speech_query_tokens.requires_grad = True
                    for p in model.speech_llama_proj.parameters():
                        p.requires_grad = True
                else:
                    for p in model.speech_Qformer.parameters():
                        p.requires_grad = False
                    model.speech_query_tokens.requires_grad = False
                    for p in model.speech_llama_proj.parameters():
                        p.requires_grad = False
                    model.speech_Qformer.eval()

            if model_args.use_final_linear:
                for p in model.final_linear.parameters():
                    p.requires_grad = True
            if model_args.freeze_final_linear:
                for p in model.final_linear.parameters():
                    p.requires_grad = False

        if model_args.tune_mm_vision_resampler:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
            model.get_model().image_newline.requires_grad = False
        else:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            if hasattr(model.get_model(), "image_newline"):
                model.get_model().image_newline.requires_grad = True
                
        # gs534 - high res
        # if model_args.segmentation > 0:
        #     for p in model.segmentTFM.parameters():
        #         p.requires_grad = True
        if model_args.do_rag:
            if "llm" not in model_args.rag_type and "text" not in model_args.rag_type:
                for p in model.segmentTFM.parameters():
                    p.requires_grad = True
                for p in model.query_proj.parameters():
                    p.requires_grad = True
            elif ("text" in model_args.rag_type and "video" not in model_args.rag_type) or "both" in model_args.rag_type:
                for p in model.text_encoder.parameters():
                    p.requires_grad = False
            if "video" in model_args.rag_type and "train" in model_args.rag_type:
                for p in model.video_encoder.parameters():
                    p.requires_grad = False
                for p in model.video_encoder.vision_model.head.parameters():
                    p.requires_grad = True
                for p in model.video_encoder.text_model.head.parameters():
                    p.requires_grad = True

        model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
        if training_args.freeze_mm_vision_resampler:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = False

        model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
        if model_args.unfreeze_mm_vision_tower:
            vision_tower.requires_grad_(True)
        else:
            vision_tower.requires_grad_(False)

        if model_args.use_mfcnn:
            model.mfcnn._initialize_weights()
            model.mfcnn.requires_grad_(True)

        if training_args.train_llm_layers is not None:
            train_llm_layers = eval(training_args.train_llm_layers)
            train_llm_module_name = [f'model.layers.{idx}.' for idx in train_llm_layers]
            for n, p in model.named_parameters():
                for train_n in train_llm_module_name:
                    if train_n in n:
                        p.requires_grad = True
                        break
            pass

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        data_args.do_rag = model_args.do_rag
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if torch.cuda.current_device() == 0:
        import nltk
        nltk.download("words")

    dist.barrier()

    if training_args.run_test or training_args.do_demo:
        if not training_args.prediction_loss_only or training_args.do_demo:
            data_module = make_test_data_module(tokenizer=tokenizer, data_args=data_args)
        else:
            data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        
        model.to(torch.bfloat16).cuda()
        if training_args.lora_enable:
            model.model.tokenizer = tokenizer
        else:
            model.tokenizer = tokenizer

        trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    # compute_metrics=compute_metrics,
                    **data_module)
        
        # if not training_args.lora_enable:
        #    trainer._load_from_checkpoint(training_args.model_base)
        
        if training_args.do_demo:
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import KeywordsStoppingCriteria
            from transformers import set_seed
            test_dataset = data_module["eval_dataset"]
            data_collator = data_module["data_collator"]
            while True:
                try:
                    yaml_file = input("yaml file: ")
                    with open(yaml_file, 'r') as file:
                        yaml_data = yaml.safe_load(file)
                    if data_args.use_audio:
                        audio_path = yaml_data.get('audio_path', None)
                    video_path = yaml_data['video_path']
                    tt = yaml_data.get("tt", False)
                    if not tt:
                        assert os.path.exists(video_path) #  and os.path.exists(audio_path)

                    qs = yaml_data['question']
                    max_time = yaml_data.get("max_time", 30)
                    fps = yaml_data.get("fps", 1)
                    max_new_tokens = yaml_data.get("max_new_tokens", 1024)
                    min_new_tokens = yaml_data.get("min_new_tokens", 1)
                    do_sample = yaml_data.get("do_sample", False)
                    beam = yaml_data.get("beam", 1)
                    top_p = yaml_data.get("top_p", 0.9)
                    seed = yaml_data.get("seed", 2024)
                    insert_white = yaml_data.get("insert_white", -1)
                    insert_image = yaml_data.get("insert_image", -1)
                    insert_image_path = yaml_data.get("insert_image_path", None)
                    insert_image_tos_key = yaml_data.get("insert_image_tos_key", None)
                    include_time_prompt = yaml_data.get("include_time_prompt", False)
                    insert_image_num = yaml_data.get("insert_image_num", 1)
                    insert_text = yaml_data.get("insert_text", -1)
                    insert_sentence = yaml_data.get("insert_sentence", -1)
                    demo_text = yaml_data.get("demo_text", None)
                    demo_size = yaml_data.get("demo_size", 6)
                    demo_thick = yaml_data.get("demo_thick", 15)
                    demo_rect = yaml_data.get("demo_rect", True)

                    test_dataset.data_args.insert_image_num = insert_image_num
                    test_dataset.demo = True
                    test_dataset.demo_text = demo_text
                    test_dataset.demo_size = demo_size
                    test_dataset.demo_thick = demo_thick
                    test_dataset.demo_rect = demo_rect

                    test_dataset.max_time = max_time
                    test_dataset.data_args.video_fps = fps
                    test_dataset.max_frame_num = round(test_dataset.max_time * test_dataset.data_args.video_fps)

                    test_dataset.list_data_dict = [{}]
                    if video_path != "":
                        test_dataset.list_data_dict[0]["video"] = video_path
                    if tt:
                        test_dataset.list_data_dict[0]["tt_id"] = video_path

                    if data_args.use_audio:
                        test_dataset.list_data_dict[0]["audio"] = audio_path

                    test_dataset.list_data_dict[0]["conversations"] = [
                        {
                            "from": "human",
                            "value": "<image>\n" + qs.strip()
                        },
                        {
                            "from": "gpt",
                            "value": ""
                        }
                    ]
                    test_dataset.list_data_dict[0]["insert_white"] = insert_white
                    test_dataset.list_data_dict[0]["insert_image"] = insert_image
                    test_dataset.list_data_dict[0]["insert_text"] = insert_text
                    test_dataset.list_data_dict[0]["insert_sentence"] = insert_sentence

                    if insert_image_path is not None:
                        test_dataset.list_data_dict[0]["insert_image_path"] = insert_image_path
                    if insert_image_tos_key is not None:
                        test_dataset.list_data_dict[0]["insert_image_tos_key"] = insert_image_tos_key

                    test_dataset.data_args.include_time_prompt = include_time_prompt
                    if insert_image > 0 or insert_text > 0 or insert_sentence > 0:
                        item = test_dataset._get_item(0)
                        print(item["prompt"][0]["value"])
                        print(item["text"])
                    else:
                        item = test_dataset._get_item(0)

                    batch = data_collator([item])
                    batch["input_ids"] = batch["input_ids"].cuda()
                    batch["labels"] = batch["labels"].cuda()
                    batch["attention_mask"] = batch["attention_mask"].cuda()
                    batch["images"] = [it.to(torch.bfloat16).cuda() for it in batch["images"]]
                    batch["spectrogram"] = batch["spectrogram"].to(torch.bfloat16).cuda()

                    batch.pop("ids")
                    batch.pop("prompts")
                    texts = batch.pop("texts")
                    real_time = batch["real_time"]
                    if insert_white >= 0:
                        print("Insert time: ", real_time[0] * insert_white)

                    conv = conv_templates['qwen_1_5'].copy()
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, batch["input_ids"])

                    set_seed(seed)
                    result = model.generate(
                        do_sample=do_sample,
                        num_beams=beam,
                        stopping_criteria=[stopping_criteria],
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        top_p=top_p,
                        **batch
                    )
                    res_ids = result.tolist()
                    res_text = [tokenizer.decode(it) for it in res_ids]
                    print("======================")
                    print(res_text[0])
                    with open("/opt/tiger/tmp.json", "w") as fp:
                        json.dump([res_text[0]], fp)
                    print("======================")
                except Exception as e:
                    print(e, e.__traceback__.tb_lineno)
                    import pdb; pdb.set_trace()
        
        if training_args.test_output_dir is not None:
            test_output_dir = training_args.test_output_dir
            timestamp = current_time.strftime("%Y%m%d%H")
            test_output_dir = os.path.join(test_output_dir, timestamp)
            if dist.get_rank() == 0:
                os.makedirs(test_output_dir, exist_ok=True)
                if not os.path.exists(f"{test_output_dir}/train_config"):
                    os.makedirs(f"{test_output_dir}/train_config", exist_ok=True)
                with open(f"{test_output_dir}/train_config/model_args.json", "w") as f:
                    f.write(json.dumps(asdict(model_args)))
                with open(f"{test_output_dir}/train_config/data_args.json", "w") as f:
                    f.write(json.dumps(asdict(data_args)))
                with open(f"{test_output_dir}/train_config/training_args.json", "w") as f:
                    f.write(training_args.to_json_string())

                if True: # data_args.use_tos:
                    hmkdir(os.path.join(HDFS_BASE_PATH, test_output_dir))
                    hmkdir(os.path.join(HDFS_BASE_PATH, f"{test_output_dir}/train_config"))
                    hcopy(f"{test_output_dir}/train_config/model_args.json", os.path.join(HDFS_BASE_PATH, f"{test_output_dir}/train_config/model_args.json"), chunk_thread_num=1)
                    hcopy(f"{test_output_dir}/train_config/data_args.json", os.path.join(HDFS_BASE_PATH, f"{test_output_dir}/train_config/data_args.json"), chunk_thread_num=1)
                    hcopy(f"{test_output_dir}/train_config/training_args.json", os.path.join(HDFS_BASE_PATH, f"{test_output_dir}/train_config/training_args.json"), chunk_thread_num=1)

            os.makedirs(test_output_dir, exist_ok=True)
        
        results = trainer.predict(
            data_module['eval_dataset'],
            do_sample=training_args.do_sample,
            do_reasoning=training_args.do_reasoning,
            beam=training_args.beam,
            rollout=training_args.rollout,
            interp_factor=training_args.interp_factor,
        )

        print(f"rank {dist.get_rank()} finish predict")
        
        
        if training_args.test_output_dir is not None:

            output_path = os.path.join(test_output_dir, f"test_results_rank{dist.get_rank()}.json")
            with open(output_path, 'w') as fp:
                json.dump(results, fp)
            if True: # data_args.use_tos:
                print(f"rank {dist.get_rank()} start upload")
                hcopy(output_path, os.path.join(HDFS_BASE_PATH, test_output_dir, f"test_results_rank{dist.get_rank()}.json"), chunk_thread_num=1)
                print(f"rank {dist.get_rank()} finish upload")

            if is_dist_avail_and_initialized():
                dist.barrier()

            if dist.get_rank() == 0:
                res = []
                print("start merging")
                for i in range(dist.get_world_size()):
                    print(f"rank {i} start merging")
                    if True: # data_args.use_tos:
                        hcopy(os.path.join(HDFS_BASE_PATH, test_output_dir, f"test_results_rank{i}.json"), os.path.join(test_output_dir, f"test_results_rank{i}.json"), chunk_thread_num=1)
                    with open(os.path.join(test_output_dir, f"test_results_rank{i}.json"), 'r') as fp:
                        res_i = json.load(fp)
                    res += res_i

                temp_dict = {}
                new_res = []
                for it in res:
                    key_id = str(it['id']) + ' '.join([_it["value"] for _it in it["prompt"]])
                    if key_id not in temp_dict:
                        temp_dict[key_id] = 1
                        new_res.append(it)

                res = new_res
                
                with open(tp_path := os.path.join(test_output_dir, f"test_results.json"), 'w') as fp:
                    json.dump(res, fp, indent=4)

                # print(tp_path)
                print(os.path.abspath(tp_path))
                if True: # data_args.use_tos:
                    hcopy(tp_path, os.path.join(HDFS_BASE_PATH, test_output_dir, f"test_results.json"), chunk_thread_num=1)

                    print(os.path.join(HDFS_BASE_PATH, test_output_dir, f"test_results.json"))

                if training_args.prediction_loss_only:
                    print("=" * 50)
                    print(" " * 50)
                    loss_list = [it['loss'] for it in res]
                    avg = sum(loss_list) / len(loss_list)
                    print(f"avg loss: {avg}")
                    print(" " * 50)
                    print("=" * 50)

        else:
            raise NotImplementedError

    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        if training_args.lora_enable:
            model.model.tokenizer = tokenizer
        else:
            model.tokenizer = tokenizer
        trainer = LLaVATrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        **data_module)

        # if "checkpoint" in training_args.ckpt:
        #     if not training_args.lora_enable:
        #         trainer._load_from_checkpoint(training_args.ckpt)

        if training_args.evaluation_strategy != "no":
            trainer.add_callback(EvaluateFirstStepCallback())

        temp_cnt, temp_total = 0, 0
        if dist.get_rank() == 0:
            for k, p in model.named_parameters():
                temp_total += 1
                if p.requires_grad:
                    print(k)
                    temp_cnt += 1

            print(temp_cnt, temp_total)
        
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
        trainer.save_state()

        model.config.use_cache = True

        if training_args.lora_enable:
            state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), training_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model.named_parameters()
            )
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                model.config.save_pretrained(training_args.output_dir)
                model.save_pretrained(training_args.output_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
        else:
            safe_save_model_for_hf_trainer(trainer=trainer,
                                        output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
