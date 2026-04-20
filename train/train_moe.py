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
import os.path as osp
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List
import pathlib
import transformers as tf
from datasets import Emu3SFTDataset
import sys
ELLSA_BASE_PATH = os.environ.get("ELLSA_BASE_PATH")
COSY_CKPT_PATH = os.environ.get("COSY_CKPT_PATH")
LLAMA_CKPT_PATH = os.environ.get("LLAMA_CKPT_PATH")
UNIVLA_CKPT_PATH = os.environ.get("UNIVLA_CKPT_PATH")

sys.path.append(os.path.join(ELLSA_BASE_PATH,"reference/Emu3"))
from safetensors.torch import safe_open

from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM, Emu3MoE, Emu3MoEConfig, Emu3MoEWithSpeech, Emu3ForMix, LlamaWithSpeech, Emu3ForMix_FourExpert, Emu3ForMix_FourExpert_Text
from transformers import AutoModel, Trainer, AutoTokenizer, AutoConfig
from datasets import Emu3WorldModelDataset,Emu3RealRobotDataset,Emu3CoTDataset,Emu3SpeechDataset,Emu3SpeechOnlyDataset,Emu3MixDataset
from torch.utils.data import WeightedRandomSampler, DataLoader
from torch.nn.utils.rnn import pad_sequence

class WeightedSamplerTrainer(Trainer):
    def get_train_dataloader(self):
        # Assuming train_dataset has a sample_weights attribute
        sample_weights = torch.tensor(
            self.train_dataset.sample_weights, dtype=torch.double
        )

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Gen")
    model_config_path: Optional[str] = field(default="pretrain/Emu3-Base")
    config_speech_path: Optional[str] = field(default="pretrain/Emu3-Base")
    config_vision_path: Optional[str] = field(default="pretrain/Emu3-Base")
    speech_encoder_path: Optional[str] = field(default="")
    speech_expert_path: Optional[str] = field(default="BAAI/Emu3-Gen")
    vision_expert_path: Optional[str] = field(default="BAAI/Emu3-Gen")
    llama_path: Optional[str] = field(default=LLAMA_CKPT_PATH)
    encoder_type: Optional[str] = field(default="mamba")

@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    data_speech_path: Optional[str] = field(default=None)
    null_prompt_prob: float = field(default=0.05)
    apply_loss_on_only_vision: bool = field(default=True)
    apply_loss_on_only_text: bool = field(default=False)
    apply_loss_on_only_action: bool = field(default=False) 
    ignore_index: int = field(default=-100)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: Optional[int] = field(default=32768)
    frames: int = field(default=4)
    VL: bool = field(default=False)
    actions: bool = field(default=False)
    actions_format: str = field(default="fast")
    action_frames: int = field(default=8)
    use_gripper: bool = field(default=False)
    action_tokenizer_path: Optional[str] = field(default=os.path.join(ELLSA_BASE_PATH,"pretrain/fast"))
    video_format: str = field(default=None)
    random_frame_sampling: bool = field(default=True)
    raw_image: bool = field(default=False)
    post_training: bool = field(default=False)
    datasets_weight: bool = field(default=False)
    without_text: bool = field(default=False)
    real_robot: bool = field(default=False)
    with_cot: bool = field(default=False)
    token_per_second: int = field(default=8)
    contemporary: bool = field(default=False)
    vqa: bool = field(default=False)
    context_vqa: bool = field(default=False)
    stop: bool = field(default=False)
    refuse: bool = field(default=False)
    refuse_ratio: float = field(default=0.05)
    stop_ratio: float = field(default=0.05)
    time_block: float = field(default=1.0)
    mix_init_data: bool = field(default=False)

@dataclass
class TrainingArguments(tf.TrainingArguments):
    report_to: Optional[str] = field(default="wandb")
    run_name: Optional[str] = field(default="test")
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    attn_type: Optional[str] = field(default="fa2")
    image_area: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)
    from_scratch: bool = field(default=False)
    dataloader_num_workers: Optional[int] = field(default=0)
    speech: bool = field(default=False)
    speech_only: bool = field(default=False)
    mix: bool = field(default=False)
    moe: bool = field(default=False)
    peft: bool = field(default=False)
    freeze: bool = field(default=False)
    debug_mode: bool = field(default=False)
    llama: bool = field(default=False)
    attn_adapter: bool = field(default=False)
    attn_adapter_type: str = field(default="Linear")
    merge_speech_lora: bool = field(default=False)
    lora_modules: str = field(default="default")
    pt_ckpt: Optional[str] = field(default=None)
    generate: bool = field(default=False)
    action_loss_weight: float = field(default=1.0)

def load_model(model_args, model_config, training_args, tokenizer=None, time_block=1.0):
    """
    Load model based on whether to train from scratch or fine-tune from a pre-trained model.
    """
    if training_args.speech:
        if training_args.moe:
            model = Emu3ForMix(model_config[0],model_config[1],tokenizer,model_args.speech_encoder_path,training_args.peft,training_args.freezetraining_args.debug_mode,attn_adapter=training_args.attn_adapter,attn_adapter_type=training_args.attn_adapter_typemerge_speech_lora=training_args.merge_speech_lora,lora_modules=training_args.lora_modules,generate=training_args.generateaction_loss_weight=training_args.action_loss_weight,time_block=time_block)
            model.set_from_pretrained(
                speech_path=model_args.speech_expert_path,
                vision_path=model_args.vision_expert_path,
                config_speech=model_config[0],
                config_vision=model_config[1],
                tokenizer=tokenizer,
                speech_encoder_path=model_args.speech_encoder_path,
                attn_implementation="flash_attention_2" if training_args.attn_type == "fa2" else None,
                torch_dtype=torch.bfloat16 if training_args.bf16 else None,
                trust_remote_code=True,
                peft=training_args.peft,
                freeze=training_args.freeze,
                debug=training_args.debug_mode,
                merge_speech_lora=training_args.merge_speech_lora,
                lora_modules=training_args.lora_modules,
                generate=training_args.generate,
                encoder_type=model_args.encoder_type,
                time_block=time_block
            )
            return model
        else:
            return LlamaWithSpeech(
                config=model_config,
                tokenizer=tokenizer,
                llama_path=model_args.llama_path,
                speech_encoder_path=model_args.speech_encoder_path,
                peft=training_args.peft,
                freeze=training_args.freeze,
                debug=training_args.debug_mode,
                mix=training_args.mix,
                generate=training_args.generate,
                encoder_type=model_args.encoder_type,
                time_block=time_block
            )
    else:
        if training_args.from_scratch:
            model_config.torch_dtype = torch.bfloat16 if training_args.bf16 else None
            model_config.attn_implementation = "flash_attention_2" if training_args.attn_type == "fa2" else None
            return Emu3MoE(config=model_config)
        else:
            return Emu3MoE.from_pretrained(
                model_args.model_name_or_path,
                config=model_config,
                attn_implementation="flash_attention_2" if training_args.attn_type == "fa2" else None,
                torch_dtype=torch.bfloat16 if training_args.bf16 else None,
                trust_remote_code=True
            )

def get_dataset(data_args, tokenizer, speech, speech_only, mix, moe, generate, encoder_type):
    """
    Initialize and return the training dataset.
    """
    if data_args.post_training:
        return Emu3WorldModelDataset(data_args, tokenizer=tokenizer)
        # return Emu3SFTDataset(data_args, tokenizer=tokenizer)
    elif data_args.real_robot:
        return Emu3RealRobotDataset(data_args, tokenizer=tokenizer)
    elif data_args.with_cot:
        return Emu3CoTDataset(data_args, tokenizer=tokenizer)
    elif speech:
        if mix:
            return Emu3MixDataset(data_args, tokenizer=tokenizer, moe=moe, contemporary=data_args.contemporary, stop=data_args.stop, stop_ratio=data_args.stop_ratio, vqa=data_args.vqa, context_vqa=data_args.context_vqa, generate=generate, encoder_type=encoder_type)
        elif speech_only:
            return Emu3SpeechOnlyDataset(data_args, tokenizer=tokenizer, generate=generate, encoder_type=encoder_type)
        else:
            return Emu3SpeechDataset(data_args, tokenizer=tokenizer, moe=moe, encoder_type=encoder_type)
    return Emu3SFTDataset(data_args, tokenizer=tokenizer)

def get_dataset_split(data_args, tokenizer):
    """
    Initialize and return the training dataset.
    """
    if data_args.post_training:
        full_dataset = Emu3WorldModelDataset(data_args, tokenizer=tokenizer)
    else:
        full_dataset = Emu3SFTDataset(data_args, tokenizer=tokenizer)
    # 自动划分 90% train, 10% val
    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]

def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None else
        setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)

def load_sharded_safetensors_into_model(model, ckpt_dir, key_transform=None, device="cpu"):
    ckpt_dir = pathlib.Path(ckpt_dir)
    index_path = ckpt_dir / "model.safetensors.index.json"
    assert index_path.exists()

    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index["weight_map"]  # param_name -> shard filename
    shard_files = sorted(set(weight_map.values()))

    model_state = model.state_dict()
    loaded = {}
    shape_mismatch = []

    def maybe_transform(k):
        return key_transform(k) if key_transform else k

    for shard in shard_files:
        shard_path = ckpt_dir / shard
        with safe_open(shard_path.as_posix(), framework="pt", device=device) as f:
            for k in f.keys():
                k2 = maybe_transform(k)
                if k2 in model_state:
                    tensor = f.get_tensor(k)
                    if model_state[k2].shape == tensor.shape:
                        loaded[k2] = tensor
                    else:
                        shape_mismatch.append((k, model_state[k2].shape, tensor.shape))

    missing, unexpected = model.load_state_dict(loaded, strict=False)

    return missing, unexpected, shape_mismatch

def train():
    """
    Main function to train the model.
    """
    # Parse arguments
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set environment variable for WANDB logging
    os.environ["WANDB_PROJECT"] = "vla_speech"

    # Load model configuration and tokenizer
    if training_args.moe:
        config_speech = AutoConfig.from_pretrained(model_args.config_speech_path)
        config_vision = Emu3MoEConfig.from_pretrained(model_args.config_vision_path)
        update_configs(config_vision, training_args, ["image_area", "max_position_embeddings"])
        model_config = [config_speech, config_vision]
    else:
        model_config = AutoConfig.from_pretrained(model_args.model_config_path)
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate
    if training_args.moe:
        speech_tokenizer = AutoTokenizer.from_pretrained(model_args.speech_expert_path)
        speech_tokenizer.bosp_token = "<bosp>" # modified
        speech_tokenizer.eosp_token = "<eosp>" # modified
        speech_tokenizer.bot_token = "<bot>"
        speech_tokenizer.eot_token = "<eot>"
        speech_tokenizer.silence_token = "<silence>"
        speech_tokenizer.bop_token = "<bop>"
        speech_tokenizer.eop_token = "<eop>"
        speech_tokenizer.padding_side = "right"
        vision_tokenizer = Emu3Tokenizer.from_pretrained(
            model_args.vision_expert_path,
            model_max_length=training_args.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        tokenizer = [speech_tokenizer, vision_tokenizer]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({"additional_special_tokens": ["<bosp>","<eosp>","<bot>","<eot>","<silence>","<bop>","<eop>"]})
        tokenizer.bosp_token = "<bosp>" # modified
        tokenizer.eosp_token = "<eosp>" # modified
        tokenizer.bot_token = "<bot>"
        tokenizer.eot_token = "<eot>"
        tokenizer.silence_token = "<silence>"
        tokenizer.bop_token = "<bop>"
        tokenizer.eop_token = "<eop>"
        tokenizer.padding_side = "right"
        if training_args.mix:
            vision_tokenizer = Emu3Tokenizer.from_pretrained(
                UNIVLA_CKPT_PATH,
                model_max_length=training_args.max_position_embeddings,
                padding_side="right",
                use_fast=False,
            )
            tokenizer = [tokenizer, vision_tokenizer]
    
    # Initialize model
    if training_args.speech:
        model = load_model(model_args, model_config, training_args, tokenizer, data_args.time_block)
        if training_args.peft:
            if training_args.moe:
                model.peft()
    else:
        model = load_model(model_args, model_config, training_args)
    
    if training_args.pt_ckpt:
        missing, unexpected, shape_mismatch = load_sharded_safetensors_into_model(model, training_args.pt_ckpt)

    # Initialize dataset
    train_dataset = get_dataset(data_args, tokenizer, training_args.speech, training_args.speech_only, training_args.mix, training_args.moe, training_args.generate, model_args.encoder_type)

    if data_args.datasets_weight:
        trainer = WeightedSamplerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset, 
            tokenizer=tokenizer,
        )
    elif training_args.speech:
        class Collator:
            def __call__(self, samples):
                input_ids = [s["input_ids"] for s in samples]
                attention_mask = [s["attention_mask"] for s in samples]
                labels = [s["labels"] for s in samples]
                if "fbank_feature" in samples[0].keys():
                    fbank_feature = pad_sequence(
                        [s["fbank_feature"][0] for s in samples], batch_first=True
                    )
                    fbank_feature_len = torch.tensor([s["fbank_feature_len"] for s in samples])
                    sent_lens = [s["sent_lens"] for s in samples]
                    codecs = [s["codecs"] for s in samples]
                    codec_lens = [s["codec_lens"] for s in samples]
                else:
                    fbank_feature = None
                    fbank_feature_len = None
                    sent_lens = None
                    codecs = None
                    codec_lens = None

                if "context_qa" in samples[0].keys():
                    context_qa = [s["context_qa"] for s in samples]
                else:
                    context_qa = [False]
                
                if "type" in samples[0].keys():
                    data_type = samples[0]["type"]
                else:
                    data_type = "mix"
                
                return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "fbank_feature": fbank_feature, "fbank_feature_len": fbank_feature_len, "sent_lens": sent_lens, "codecs": codecs, "codec_lens": codec_lens, "context_qa": context_qa, "data_type": data_type}
        
        collator = Collator()
        
        if training_args.moe or training_args.mix:
            trainer = tf.Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset, 
                tokenizer=tokenizer[0],
                data_collator=collator,
            )
        else:
            trainer = tf.Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset, 
                tokenizer=tokenizer,
                data_collator=collator,
            )
    else:
        # Setup Trainer
        trainer = tf.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,  # Pass tokenizer to trainer
        )

    # Check if resuming from checkpoint
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model and training state
    trainer.save_state()
    torch.cuda.synchronize()
    # trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()
