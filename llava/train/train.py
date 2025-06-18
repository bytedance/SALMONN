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

# Adopted from https://github.com/LLaVA-VL/LLaVA-NeXT. The original license is located at 'third-party-license/llava_next.txt'.
# Adopted from https://github.com/lm-sys/FastChat. The original license is located at 'third-party-license/fastchat.txt'.
# Adopted from tatsu-lab@stanford_alpaca. The original license is located at 'third-party-license/stanford_alpaca.txt'.

import os
import copy
from dataclasses import dataclass, field, asdict
import json
import pathlib
from typing import Dict, Optional
import ast
import torch
import random
import torch.distributed as dist
import transformers
import yaml
from llava.train.llava_trainer import LLaVATrainer
from llava.train.dpo_trainer import LLaVADPOTrainer
from llava import conversation as conversation_lib
from llava.model import VideoSALMONN2ForCausalLM
from llava.model.utils import load_qwen_lora_model
import numpy as np
from transformers import AutoConfig
from llava.dataset import make_supervised_data_module, make_test_data_module
from transformers import TrainerCallback

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="average")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_pooling_position: Optional[str] = field(default="before")
    mm_newline_position: Optional[str] = field(default="grid")
    modality_max_length: Optional[str] = field(default="None")
    audio_visual: bool = False
    whisper_path: str = "openai/whisper-large-v3"
    freeze_whisper: bool = True
    num_speech_query_token: int = 1
    freeze_speech_QFormer: bool = False
    window_level_Qformer: bool = True
    second_per_window: float = 0.333333
    second_stride: float = 0.333333
    use_final_linear: bool = False
    freeze_final_linear: bool = False
    add_time_token: bool = False

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    test_data_path: str = field(default=None, metadata={"help": "Path to the test data."})
    is_multimodal: bool = True
    video_fps: Optional[int] = field(default=1)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: int = 224
    image_split_resolution: int = 224
    audio_processor : str = "openai/whisper-large-v3"
    max_time: int = 30

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
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
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    seed: int = 2024
    load_from_lora: bool = False
    do_test: bool = False
    test_output_dir: str = None
    evaluation_strategy: str = "steps"
    eval_steps: int = 1000
    disable_tqdm: bool = True
    lora_path: str = None
    do_sample: bool = False
    model_base: str = None
    max_new_tokens: int = 1024
    ckpt: str = None
    do_demo: bool = False
    pretrain_weight: str = None
    load_full: bool = False
    merge_and_new_lora: bool = False
    dpo_train: bool = False
    loss_type: str = "sigmoid"
    ce_loss_weight: float = 0.1
    with_ce_loss: bool = False
    beta: float = 0.1

class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args.fps = data_args.video_fps
    model_args.lora_enable = training_args.lora_enable
    model_args.lora_r = training_args.lora_r
    model_args.lora_alpha = training_args.lora_alpha
    model_args.lora_dropout = training_args.lora_dropout

    seed = training_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")

    audio_config = dict(
        audio_visual=model_args.audio_visual,
        video_fps=data_args.video_fps,
        whisper_path=model_args.whisper_path,
        num_speech_query_token = model_args.num_speech_query_token,
        window_level_Qformer = model_args.window_level_Qformer,
        second_per_window = model_args.second_per_window,
        second_stride = model_args.second_stride,
        use_final_linear=model_args.use_final_linear,
    )

    if not training_args.load_from_lora:
        cfg_pretrained = AutoConfig.from_pretrained(training_args.model_base)
        overwrite_config = {"model_args": vars(model_args), "add_time_token": model_args.add_time_token}

        print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)
        model = VideoSALMONN2ForCausalLM.from_pretrained(
            training_args.ckpt,
            config=cfg_pretrained,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **audio_config
        )

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


        if training_args.lora_enable:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if training_args.bits == 16:
                if training_args.bf16:
                    model.to(torch.bfloat16)
                if training_args.fp16:
                    model.to(torch.float16)
            print("Adding LoRA adapters...")

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
                v_flag = False
                if hasattr(model.model, "vision_tower"):
                    vision_tower = model.model.vision_tower
                    del model.model.vision_tower
                    v_flag = True

                model = get_peft_model(model, lora_config)
                
                if v_flag:
                    model.model.model.vision_tower = vision_tower

            if training_args.bf16:
                model.to(torch.bfloat16)

        tokenizer = transformers.AutoTokenizer.from_pretrained(training_args.ckpt, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")

    else:
        model, tokenizer = load_qwen_lora_model(training_args.ckpt, model_base=training_args.model_base, lora_enable=training_args.lora_enable, pretrain_weight=training_args.pretrain_weight, load_full=training_args.load_full, lora_r=training_args.lora_r, lora_alpha=training_args.lora_alpha, lora_dropout=training_args.lora_dropout, model_max_length=training_args.model_max_length, new_model_args=model_args, **audio_config)
        # if True:
        #     tmp_dir = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/video-SALMONN2/output/video_SALMONN_2"
        #     if dist.get_rank() == 0:
        #         import shutil
        #         if os.path.exists(tmp_dir):
        #             shutil.rmtree(tmp_dir)
        #         breakpoint()
        #         model = model.merge_and_unload()
        #         model.to(torch.bfloat16)
        #         model.save_pretrained(tmp_dir)
        #         exit()
        #     dist.barrier()
        #     model = VideoSALMONN2ForCausalLM.from_pretrained(tmp_dir, low_cpu_mem_usage=True, device_map="cuda", attn_implementation="flash_attention_2", **audio_config)
        #     model.to(torch.bfloat16)

        if training_args.merge_and_new_lora:
            print("Merging LoRA")
            model = model.merge_and_unload()
            if training_args.lora_enable:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=training_args.lora_r,
                    lora_alpha=training_args.lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj"],
                    lora_dropout=training_args.lora_dropout,
                    bias=training_args.lora_bias,
                    task_type="CAUSAL_LM",
                )
                if model_args.audio_visual:
                    if True:
                        speech_encoder = model.speech_encoder
                        model.speech_encoder = None
                        v_flag = False
                        if hasattr(model.model, "vision_tower"):
                            vision_tower = model.model.vision_tower
                            del model.model.vision_tower
                            v_flag = True

                        model = get_peft_model(model, lora_config)

                        model.model.speech_encoder = speech_encoder
                        if v_flag:
                            model.model.model.vision_tower = vision_tower
                else:
                    v_flag = False
                    if hasattr(model.model, "vision_tower"):
                        vision_tower = model.model.vision_tower
                        del model.model.vision_tower
                        v_flag = True

                    model = get_peft_model(model, lora_config)
                    
                    if v_flag:
                        model.model.model.vision_tower = vision_tower
        else:
            print("Not merging LoRA")

        model.config.use_cache = False
        if model_args.freeze_backbone:
            model.model.requires_grad_(False)
            model.lm_head.requires_grad_(False)
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = True

        if training_args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if model_args.version == "qwen_1_5":
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.pad_token_id = tokenizer.bos_token_id
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        raise NotImplementedError

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

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
    assert model_args.mm_pooling_position in ["before", "after", "no"] # "mm_pooling_position must be either 'before' or 'after' or 'no'"
    model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride
    model.config.mm_pooling_position = model_args.mm_pooling_position
    model.config.mm_spatial_pool_mode = model_args.mm_spatial_pool_mode
    model.config.modality_max_length = model_args.modality_max_length

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
        model.lm_head.requires_grad_(False)
    else:
        model.model.requires_grad_(True)
        model.lm_head.requires_grad_(True)

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

        if model_args.freeze_speech_QFormer:
            for name, param in model.speech_Qformer.named_parameters():
                param.requires_grad = False
            model.speech_Qformer.eval()
            model.speech_query_tokens.requires_grad = False
        else:
            for name, param in model.speech_Qformer.named_parameters():
                param.requires_grad = True
            model.speech_Qformer.train()
            model.speech_query_tokens.requires_grad = True

        if model_args.use_final_linear:
            for p in model.final_linear.parameters():
                p.requires_grad = True
        if model_args.freeze_final_linear:
            for p in model.final_linear.parameters():
                p.requires_grad = False

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

    model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
    if model_args.unfreeze_mm_vision_tower:
        vision_tower.requires_grad_(True)
    else:
        vision_tower.requires_grad_(False)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

    if training_args.do_test or training_args.do_demo:
        data_module = make_test_data_module(tokenizer=tokenizer, data_args=data_args)
        
        model.to(torch.bfloat16).cuda()
        model.eval()
        if training_args.lora_enable:
            model.model.tokenizer = tokenizer
        else:
            model.tokenizer = tokenizer

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
                    if model_args.audio_visual:
                        audio_path = yaml_data.get('audio_path', None)
                    text_only = yaml_data.get("text_only", False)
                    if text_only:
                        video_path = ""
                    else:
                        video_path = yaml_data['video_path']
                    if not text_only:
                        assert os.path.exists(video_path)

                    qs = yaml_data['question']
                    max_time = yaml_data.get("max_time", 30)
                    fps = yaml_data.get("fps", 1)
                    max_new_tokens = yaml_data.get("max_new_tokens", 1024)
                    do_sample = yaml_data.get("do_sample", False)
                    top_p = yaml_data.get("top_p", 0.9)
                    seed = yaml_data.get("seed", 2024)
                    prefix = yaml_data.get("prefix", "")

                    test_dataset.max_time = max_time
                    test_dataset.data_args.video_fps = fps
                    test_dataset.max_frame_num = round(test_dataset.max_time * test_dataset.data_args.video_fps)

                    test_dataset.list_data_dict = [{}]
                    if not text_only:
                        if video_path != "":
                            test_dataset.list_data_dict[0]["video"] = video_path

                        if model_args.audio_visual and not text_only:
                            test_dataset.list_data_dict[0]["audio"] = audio_path

                        test_dataset.list_data_dict[0]["conversations"] = [
                            {
                                "from": "human",
                                "value": "<image>\n" + qs.strip(),
                            },
                            {
                                "from": "gpt",
                                "value": "",
                                "prefix": prefix,
                            }
                        ]
                    else:
                        test_dataset.list_data_dict[0]["conversations"] = [
                            {
                                "from": "human",
                                "value": qs.strip(),
                                "prefix": prefix,
                            },
                            {
                                "from": "gpt",
                                "value": ""
                            }
                        ]
                    item = test_dataset._get_item(0)

                    batch = data_collator([item])
                    batch["input_ids"] = batch["input_ids"].cuda()
                    batch["labels"] = batch["labels"].cuda()
                    batch["attention_mask"] = batch["attention_mask"].cuda()
                    if not text_only:
                        batch["images"] = [it.to(torch.bfloat16).cuda() for it in batch["images"]]
                        batch["spectrogram"] = batch["spectrogram"].to(torch.bfloat16).cuda()

                    batch.pop("ids")
                    batch.pop("prompts")
                    batch.pop("ce_only")
                    batch.pop("texts")

                    conv = conv_templates['qwen_1_5'].copy()
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, batch["input_ids"])

                    set_seed(seed)
                    _ = batch.pop("ori_item", None)
                    result = model.generate(
                        do_sample=do_sample,
                        num_beams=1,
                        stopping_criteria=[stopping_criteria],
                        max_new_tokens=max_new_tokens,
                        top_p=top_p,
                        **batch
                    )
                    res_ids = result.tolist()
                    res_text = [tokenizer.decode(it) for it in res_ids]
                    print("======================")
                    print(res_text[0])
                    print("======================")                        

                except Exception as e:
                    # raise e
                    print(e, e.__traceback__.tb_lineno)
                    breakpoint()

        else:
            test_output_dir = training_args.test_output_dir
            if dist.get_rank() == 0:
                os.makedirs(test_output_dir, exist_ok=True)
            if training_args.dpo_train:
                training_args.gradient_checkpointing_kwargs = {'use_reentrant': False}
                trainer = LLaVADPOTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
                results = trainer.predict(data_module['eval_dataset'])
            else:
                trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
                results = trainer.predict(data_module['eval_dataset'], do_sample=training_args.do_sample)
            print(f"rank {dist.get_rank()} finish predict")

            output_path = os.path.join(test_output_dir, f"test_results_rank{dist.get_rank()}.json")
            with open(output_path, 'w') as fp:
                json.dump(results, fp)
            dist.barrier()

            if dist.get_rank() == 0:
                res = []
                print("start merging")
                for i in range(dist.get_world_size()):
                    print(f"rank {i} start merging")
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
                print(os.path.abspath(tp_path))

    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        if training_args.lora_enable:
            model.model.tokenizer = tokenizer
        else:
            model.tokenizer = tokenizer

        if training_args.dpo_train:
            training_args.gradient_checkpointing_kwargs = {'use_reentrant': False}
            trainer = LLaVADPOTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        else:
            trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

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
