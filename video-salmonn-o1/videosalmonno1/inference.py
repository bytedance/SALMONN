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
from llava.conversation import conv_templates, SeparatorStyle
import llava.conversation as conversation_lib
from llava.mm_utils import KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader
from llava.train.llava_trainer import LLaVATrainer, HDFS_BASE_PATH
from llava.dataset.av_test_dataset import LazyAVTestDataset, DataCollatorForAVTestDataset

try:
    from cruise.utilities.distributed import DIST_ENV
    from cruise.utilities.hdfs_io import hcopy, hmkdir
except:
    pass
from llava.model import LlavaAVQwenForCausalLM
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token
from PIL import Image
from decord import VideoReader, cpu
from packaging import version
import numpy as np
from transformers import AutoConfig
import math
from tqdm import tqdm


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="qwen_1_5")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='spatial_unpad')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default="spatial_pool")
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=2)
    mm_spatial_pool_mode: str = field(default="max")
    mm_spatial_pool_out_channels: Optional[int] = field(default=1152)
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
    audio_visual: bool = True
    whisper_path: str = "/mnt/bn/tiktok-mm-2/aiic/public/model/whisper-large-v3"
    freeze_whisper: bool = True
    beats_path: str = None
    freeze_beats: bool = True
    use_speech_Qformer: bool = True
    num_speech_query_token: int = 1
    freeze_speech_QFormer: bool = False
    window_level_Qformer: bool = True
    second_per_window: float = 0.2
    second_stride: float = 0.2
    salmonn_path: str = None
    use_final_linear: bool = True
    freeze_final_linear: bool = False
    use_niv: bool = False
    niv_in_channels: int = 4
    niv_out_channels: int = 401
    niv_cnn_params: str = "[(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]"
    whisper_lora: bool = False
    flash_attn: bool = False
    multi_frame_projector: bool = False
    multi_frame_num: int = 30
    add_time_token: bool = True
    use_mfcnn: bool = False
    use_mftrans: bool = False
    use_flash_tower: bool = True
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
    video_fps: Optional[int] = field(default=2)
    frames_upbound: Optional[int] = field(default=30)
    video_token: Optional[int] = field(default=2)
    image_aspect_ratio: str = 'anyres'
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: int = 224
    image_split_resolution: int = 224
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: Optional[bool] = field(default=False)
    use_audio: bool = True
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
        default=32768,
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
    group_by_modality_length: bool = field(default=True)
    group_by_modality_length_auto: bool = field(default=False)
    train_llm_layers: str = None
    seed: int = 2024
    save_torch_ckpt: bool = True
    load_from_lora: bool = False
    run_test: bool = False
    test_output_dir: str = None
    evaluation_strategy: str = "steps"
    eval_steps: int = 1000
    disable_tqdm: bool = False
    lora_path: str = None
    save_lora_only: bool = False
    do_reasoning: bool = True
    do_sample: bool = False
    model_base: str = None
    lora_llm_only: bool = False
    max_new_tokens: int = 512
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


def prediction_generate(model, inputs, frames, do_sample, stopping_criteria, max_tokens=512):
    with torch.no_grad():
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
            max_new_tokens=max_tokens,
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


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
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

    # Copy parameters
    training_args.use_tos = data_args.use_tos
    data_args.train_orm = training_args.train_orm
    data_args.do_rag = model_args.do_rag
    data_args.rag_type = model_args.rag_type
    model_args.fps = data_args.video_fps
    model_args.lora_enable = training_args.lora_enable
    model_args.lora_r = training_args.lora_r
    model_args.lora_alpha = training_args.lora_alpha
    model_args.lora_dropout = training_args.lora_dropout

    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
    overwrite_config = {
        "model_args": vars(model_args),
        "multi_frame_projector": model_args.multi_frame_projector,
        "multi_frame_num": model_args.multi_frame_num,
        "add_time_token": model_args.add_time_token,
        "use_mfcnn": model_args.use_mfcnn,
        "use_mftrans": model_args.use_mftrans,
        "use_flash_tower": model_args.use_flash_tower,
        "mf_split_init": model_args.mf_split_init,
        "spt_projector": model_args.spt_projector,
    }

    print(f"Overwriting config with {overwrite_config}")
    for k, v in overwrite_config.items():
        setattr(cfg_pretrained, k, v)

    # Load model
    model = LlavaAVQwenForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=cfg_pretrained,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **audio_config
    ).to("cuda").to(torch.bfloat16)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    model.tokenizer = tokenizer

    # Specific configuration
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.pad_token_id = tokenizer.bos_token_id
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device="cuda")

    data_args.image_processor = copy.deepcopy(vision_tower.image_processor)
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
    model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride
    model.config.mm_pooling_position = model_args.mm_pooling_position
    model.config.mm_spatial_pool_mode = model_args.mm_spatial_pool_mode
    model.config.modality_max_length = model_args.modality_max_length
    data_args.do_rag = model_args.do_rag
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    test_dataset = LazyAVTestDataset(tokenizer=tokenizer, data_path=data_args.test_data_path, data_args=data_args)
    data_collator = DataCollatorForAVTestDataset(tokenizer=tokenizer)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=data_collator, num_workers=16)

    results = []
    for inputs in tqdm(test_dataloader):
        new_prompt2 = "\n<|im_start|>user\nWhat is the final answer to the question based on your thinking steps?<|im_end|>\n<|im_start|>assistant\n"
        new_input_tokens2 = tokenizer(new_prompt2, return_tensors="pt").input_ids[0].to(model.device)
        ids = inputs.pop("ids", None)
        prompts = inputs.pop("prompts", None)
        labels = inputs.pop("labels", None)
        texts = inputs.pop("texts", None)

        inputs['input_ids'] = inputs['input_ids'].to(model.device)
        inputs['images'] = [it.to(torch.bfloat16).to(model.device) for it in inputs['images']]
        inputs["raw_wav"] = inputs["raw_wav"].to(torch.bfloat16).to(model.device)
        inputs["spectrogram"] = inputs["spectrogram"].to(torch.bfloat16).to(model.device)

        conv = conv_templates['qwen_1_5']
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, inputs['input_ids'])

        frames = inputs.get("frames")
        if frames:
            frames = [frame.to(torch.bfloat16) for frame in frames]
        generated_tokens, _ = prediction_generate(model, inputs, frames, False, stopping_criteria)
        preds = [tokenizer.decode(t) for t in generated_tokens]
        inputs["input_ids"] = torch.cat([inputs["input_ids"][0], generated_tokens[0], new_input_tokens2], dim=0).unsqueeze(0)
        generated_tokens, _ = prediction_generate(model, inputs, frames, False, stopping_criteria)
        preds = ["Reasoning: {}\nAnswer: {}".format(preds[k], tokenizer.decode(t)) for k, t in enumerate(generated_tokens)]
        results.append(preds)

    output_path = os.path.join(training_args.test_output_dir, "test_results.json")
    with open(output_path, 'w') as fp:
        json.dump(results, fp)