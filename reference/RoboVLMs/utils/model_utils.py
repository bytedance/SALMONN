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

# Adapted from https://github.com/Robot-VLAs/RoboVLMs. The original license is located at 'third-party-license/RoboVLMs.txt'.

import copy
import math
import os
import warnings
import numpy as np

from torch import nn
import transformers


def adjust_learning_rate(iter, configs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    warmup_iters = configs["warmup_iters"]
    total_iters = configs["iters"]
    min_lr_scale = configs["min_lr_scale"]

    if iter < configs["warmup_iters"]:
        lr_scaler = 1.0 * iter / warmup_iters
    else:
        lr_scaler = min_lr_scale + (1.0 - min_lr_scale) * 0.5 * (
            1.0
            + math.cos(math.pi * (iter - warmup_iters) / (total_iters - warmup_iters))
        )

    return lr_scaler


def default_tokenizer_config(tokenizer):
    tokenizer_cfg = {
        "BertTokenizer": {
            "type": "BertTokenizer",
            "pretrained_model_name_or_path": "bert-base-uncased",
        },
        "BertTokenizerFast": {
            "type": "BertTokenizerFast",
            "pretrained_model_name_or_path": "bert-base-uncased",
        },
        "CLIPTokenizer": {
            "type": "CLIPTokenizer",
            "pretrained_model_name_or_path": "openai/clip-vit-base-patch32",
        },
        "MPT1b": {
            "type": "AutoTokenizer",
            "pretrained_model_name_or_path": "VLMs/mpt-1b-redpajama-200b",
        },
        "MPT1bIFTTokenizer": {
            "type": "AutoTokenizer",
            "pretrained_model_name_or_path": "VLMs/mpt-1b-redpajama-200b-dolly",
        },
        "MPT3bTokenizer": {
            "type": "AutoTokenizer",
            "pretrained_model_name_or_path": "VLMs/RedPajama-INCITE-Base-3B-v1",
        },
        "MPT3bIFTTokenizer": {
            "type": "AutoTokenizer",
            "pretrained_model_name_or_path": "VLMs/RedPajama-INCITE-Instruct-3B-v1",
        },
        "MPT7bTokenizer": {
            "type": "AutoTokenizer",
            "pretrained_model_name_or_path": "VLMs/mpt-7b",
        },
        "LLaMA7bTokenizer": {
            "type": "AutoTokenizer",
            "pretrained_model_name_or_path": "VLMs/.cache/llama-7b-hf-jxu124",
        },
        "LLaVA7bVicunaTokenizer": {
            "type": "AutoTokenizer",
            "pretrained_model_name_or_path": "VLMs/llava-v1.6-vicuna-7b",
        },
        "kosmos2Tokenizer": {
            "type": "AutoProcessor",
            "pretrained_model_name_or_path": "VLMs/kosmos-2-patch14-224",
        },
    }

    # synonyms
    tokenizer_cfg["bert"] = tokenizer_cfg["BertTokenizer"]
    tokenizer_cfg["clip"] = tokenizer_cfg["CLIPTokenizer"]

    tokenizer_cfg["mpt_1b"] = tokenizer_cfg["MPT1bTokenizer"]
    tokenizer_cfg["mpt_1b_ift"] = tokenizer_cfg["MPT1bIFTTokenizer"]

    tokenizer_cfg["mpt_3b"] = tokenizer_cfg["MPT3bTokenizer"]
    tokenizer_cfg["mpt_3b_ift"] = tokenizer_cfg["MPT3bIFTTokenizer"]

    tokenizer_cfg["mpt_7b"] = tokenizer_cfg["MPT7bTokenizer"]

    tokenizer_cfg["llama_7b"] = tokenizer_cfg["LLaMA7bTokenizer"]

    return tokenizer_cfg[tokenizer]


def update_tokenizer(tokenizer, tokenizer_config):
    tokenizer_config = copy.deepcopy(tokenizer_config)
    # print(tokenizer_config)
    tokenizer_type = tokenizer_config.get("type")
    new_tokens = tokenizer_config.get("additional_special_tokens", None)
    if new_tokens is not None and len(new_tokens) > 0:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    if tokenizer.pad_token is None:
        try:
            tokenizer.pad_token_id = tokenizer.eod_id
        except:
            try:
                tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            except:
                warnings.warn(
                    f"Tokenizer {tokenizer_type} does not have a pad token. "
                    f"The pad token will be added to the tokenizer."
                )
    assert tokenizer.vocab_size < 2**18
    return tokenizer


def build_tokenizer(tokenizer_config):
    if isinstance(tokenizer_config, str):
        tokenizer_config = default_tokenizer_config(tokenizer_config)
    else:
        assert isinstance(tokenizer_config, dict) and "type" in tokenizer_config

    # prevent changes to the original dict
    tokenizer_config = copy.deepcopy(tokenizer_config)
    tokenizer_type = tokenizer_config.get("type")
    # tokenizer_path = os.path.join('pretrained/tokenizers', tokenizer_type)
    tokenizer_path = tokenizer_config.get("pretrained_model_name_or_path")

    try:
        tokenizer = getattr(transformers, tokenizer_type).from_pretrained(
            tokenizer_path,
            local_files_only=tokenizer_config.get("use_local_files", False),
            trust_remote_code=True,
        )

    except:
        warnings.warn(
            f"Tokenizer initialization failed with the given path {tokenizer_path}. "
            f"The tokenizer will be initialized from scratch with default settings. "
            f"Please refer to {__file__} for details."
        )

        tokenizer_config["trust_remote_code"] = True
        tokenizer = getattr(transformers, tokenizer_type).from_pretrained(
            **tokenizer_config
        )

        try:
            if os.path.exists(tokenizer_path):
                import shutil

                shutil.rmtree(tokenizer_path)

            os.makedirs(tokenizer_path, exist_ok=True)

            tokenizer.save_pretrained(tokenizer_path)
            print(f"Tokenizer saved to {tokenizer_path}.")

        except:
            print(f"Saving tokenizer to {tokenizer_path} failed...")

    if tokenizer_type == "AutoProcessor":
        tokenizer.tokenizer = update_tokenizer(tokenizer.tokenizer, tokenizer_config)
    else:
        tokenizer = update_tokenizer(tokenizer, tokenizer_config)

    return tokenizer


def get_target_modal_tokens(tok_seq, tok_mask):
    index = tok_mask.nonzero(as_tuple=True)
    return tok_seq[index]


def preprocess_text_flamingo(sample, tokenizer):
    tokenizer.padding_side = "right"
    sample = [
        # (f"{s.strip()}{tokenizer.eos_token}")
        # for s in sample
        (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}")
        for s in sample
    ]
    text = tokenizer(
        sample,
        max_length=32,
        padding="longest",
        truncation="only_first",
        return_tensors="pt",
    )
    return text["input_ids"], text["attention_mask"]


# def get_modal_tokens(tok_seq, tok_mask_dict, modal_name):
#     assert modal_name in tok_mask_dict, f"{modal_name} not in token sequence"
#     return get_target_modal_tokens(tok_seq, tok_mask_dict[modal_name])


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
