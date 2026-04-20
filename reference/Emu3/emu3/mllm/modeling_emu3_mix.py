# coding=utf-8
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
# Adapted from https://github.com/baaivision/Emu3. The original license is located at 'third-party-license/Emu.txt'.
# Adapted from https://github.com/huggingface/transformers/blob/52daf4ec768fb9ffe84a0c373834172a7c54aecc/src/transformers/models/llama/modeling_llama.py

""" PyTorch Emu3 model."""
import math
import warnings
import random
from typing import List, Optional, Tuple, Union

import torch
import os
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn.utils import rnn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_emu3 import Emu3Config
from .tokenization_emu3 import Emu3Tokenizer
from transformers import LogitsProcessorList
from transformers import StoppingCriteriaList, StoppingCriteria
from .modeling_llama_new2 import LlamaForCausalLM, LlamaPreTrainedModel

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

import sys
ELLSA_BASE_PATH = os.environ.get("ELLSA_BASE_PATH")
COSY_CKPT_PATH = os.environ.get("COSY_CKPT_PATH")
LLAMA_CKPT_PATH = os.environ.get("LLAMA_CKPT_PATH")

sys.path.append(ELLSA_BASE_PATH)
from models.policy_head.noise_schedulers import FlowMatchingScheduler

# try:
sys.path.append(os.path.join(ELLSA_BASE_PATH,"reference/cosyvoice/third_party/Matcha-TTS"))
sys.path.append(os.path.join(ELLSA_BASE_PATH,"reference"))
from cosyvoice.cli.cosyvoice import CosyVoice2
# except:
#     print("failed to load CosyVoice2")

from peft import LoraConfig, TaskType, get_peft_model
try:
    from liger_kernel.transformers import LigerCrossEntropyLoss
except:
    print("failed to load LigerCrossEntropyLoss")

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Emu3Config"

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

def is_peft_model(model):
    return getattr(model, "peft_config", None) is not None

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.emu3.modeling_emu3._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.emu3.modeling_emu3._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.emu3.modeling_emu3.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


class Emu3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Emu3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(Emu3RMSNorm)


class Emu3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # 强制 float32，不依赖 arange(dtype=...)，规避环境污染
        t = torch.arange(self.max_seq_len_cached, device="cpu").float().to(device=device, dtype=torch.float32)
        # inv_freq = self.inv_freq.to(torch.float32).to(device=device)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)).to(torch.float32).to(device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    # def _set_cos_sin_cache(self, seq_len, device, dtype):
    #     self.max_seq_len_cached = seq_len
    #     t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

    #     freqs = torch.outer(t, self.inv_freq)
    #     # Different from paper, but it uses a different permutation in order to obtain the same calculation
    #     emb = torch.cat((freqs, freqs), dim=-1)
    #     self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
    #     self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class Emu3LinearScalingRotaryEmbedding(Emu3RotaryEmbedding):
    """Emu3RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class Emu3DynamicNTKScalingRotaryEmbedding(Emu3RotaryEmbedding):
    """Emu3RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb_llama(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Emu3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Emu3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Emu3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # modify here
        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = Emu3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = Emu3LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = Emu3DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Emu3FlashAttention2(Emu3Attention):
    """
    Emu3 flash attention module. This module inherits from `Emu3Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Emu3FlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (Emu3RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in Emu3FlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class Emu3SdpaAttention(Emu3Attention):
    """
    Emu3 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Emu3Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Emu3Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Emu3Model is using Emu3SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


EMU3_ATTENTION_CLASSES = {
    "eager": Emu3Attention,
    "flash_attention_2": Emu3FlashAttention2,
    "sdpa": Emu3SdpaAttention,
}


class Emu3DecoderLayer(nn.Module):
    def __init__(self, config: Emu3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.attention_dropout)
        self.self_attn = EMU3_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = Emu3MLP(config)
        self.input_layernorm = Emu3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Emu3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


EMU3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Emu3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Emu3 Model outputting raw hidden-states without any specific head on top.",
    EMU3_START_DOCSTRING,
)
class Emu3PreTrainedModel(PreTrainedModel):
    config_class = Emu3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Emu3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


EMU3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Emu3 Model outputting raw hidden-states without any specific head on top.",
    EMU3_START_DOCSTRING,
)
class Emu3Model(Emu3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Emu3DecoderLayer`]

    Args:
        config: Emu3Config
    """

    def __init__(self, config: Emu3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.dropout = nn.Dropout(config.attention_dropout)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Emu3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = Emu3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = self.dropout(inputs_embeds)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class Emu3ForCausalLM(Emu3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Emu3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
        >>> from transformers.generation.configuration_utils import GenerationConfig
        >>> from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
        >>> from transformers import Emu3Processor
        >>> from PIL import Image

        >>> model = AutoModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_EMU3_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> image_processor = AutoImageProcessor.from_pretrained(PATH_TO_CONVERTED_IMAGE_PROCESSER)
        >>> image_tokenizer = AutoModel.from_pretrained(PATH_TO_CONVERTED_TOKENIZER_WEIGHTS).eval()
        >>> processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

        >>> # Generation
        >>> prompt = "An Emu in cartoon style, it is wearing sunglasses."

        >>> pos_inputs = processor(text=prompt, mode='G', ratio="4:3", image_area=model.config.image_area, return_tensors="pt")
        >>> neg_inputs = processor(text="", mode='G', ratio="4:3", image_area=model.config.image_area, return_tensors="pt")

        >>> GENERATION_CONFIG = GenerationConfig(
        >>>     use_cache=True,
        >>>     eos_token_id=model.config.eos_token_id,
        >>>     pad_token_id=model.config.pad_token_id,
        >>>     max_new_tokens=40960,
        >>>     do_sample=True,
        >>>     top_k=2048,
        >>> )

        >>> h, w = pos_inputs.image_size[0]
        >>> constrained_fn = processor.build_prefix_constrained_fn(h, w)
        >>> logits_processor = LogitsProcessorList([
        >>>     UnbatchedClassifierFreeGuidanceLogitsProcessor(
        >>>         classifier_free_guidance, 
        >>>         model,
        >>>         unconditional_ids=neg_inputs.input_ids.to("cuda:0"),
        >>>     ),
        >>>     PrefixConstrainedLogitsProcessor(
        >>>         constrained_fn,
        >>>         num_beams=1,
        >>>     ),
        >>> ])

        >>> outputs = model.generate(pos_inputs.input_ids.to("cuda:0"), GENERATION_CONFIG, logits_processor=logits_processor)
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        >>> mm_list = processor.decode(outputs[0])

        >>> # Understanding
        >>> prompt = "Provide a one-sentence caption for the provided image."
        >>> image = Image.open(TEST_IMAGE_PATH)

        >>> inputs = processor(text=text, image=image, mode='U', padding_side="left", padding="longest", return_tensors="pt")
        >>> input_ids = inputs.input_ids.to("cuda:0")
        >>> GENERATION_CONFIG = GenerationConfig(
        >>>     pad_token_id=tokenizer.pad_token_id,
        >>>     bos_token_id=tokenizer.bos_token_id,
        >>>     eos_token_id=tokenizer.eos_token_id,
        >>> )

        >>> outputs = model.generate(input_ids, GENERATION_CONFIG, max_new_tokens=100)
        >>> outputs = outputs[:, input_ids.shape[-1]:]
        >>> answer = processor.batch_decode(outputs, skip_special_tokens=True)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

class ActionProjector(nn.Module):
    def __init__(self, in_channels, dim):
        super(ActionProjector, self).__init__()
        # Initialize the linear layers W1, W2, W3
        self.W1 = nn.Linear(in_channels, dim)
        self.W2 = nn.Linear(dim + dim, dim)  # Concatenating 2 encodings (dim + dim)
        self.W3 = nn.Linear(dim, dim)
        self.nonlinearity = nn.SiLU()  # swish
        
        # Initialize the weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Use Xavier initialization for the linear layer weights
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
        nn.init.xavier_uniform_(self.W3.weight)
        
        # Initialize the biases to zeros
        if self.W1.bias is not None:
            nn.init.zeros_(self.W1.bias)
        if self.W2.bias is not None:
            nn.init.zeros_(self.W2.bias)
        if self.W3.bias is not None:
            nn.init.zeros_(self.W3.bias)

    def forward(self, x, tau):
        """
        Forward pass through the ActionProjector.

        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, dim)
            tau (torch.Tensor): Timestep tensor, shape (batch_size, seq_len, dim)

        Returns:
            torch.Tensor: Output tensor, shape (batch_size, seq_len, dim)
        """
        # Apply linear transformation W1 to each element in the sequence (along dim=2)
        out1 = self.W1(x)  # Shape: (batch_size, seq_len, dim)

        # Concatenate out1 and tau along the last dimension
        out2 = self.W2(torch.cat([out1, tau], dim=-1))  # Shape: (batch_size, seq_len, dim)

        # Apply linear transformation W3
        out3 = self.W3(self.nonlinearity(out2))  # Shape: (batch_size, seq_len, dim)

        return out3

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        # # init zero
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def modulate(self, x, shift, scale):
        return x * (1 + scale) + shift

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = self.modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Emu3MoE(Emu3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, lora_modules="default", freeze=False):
        super().__init__(config)
        
        # Base model (the same as in Emu3ForCausalLM)
        self.model = Emu3Model(config)
        self.vocab_size = config.vocab_size
        self.lora_modules = lora_modules
        self.freeze = freeze

        if hasattr(config, "vision_loss_weight"):
            self.use_weight = True
            self.vision_loss_weight = config.vision_loss_weight
            self.eov_token_id = config.eov_token_id
            self.bov_token_id = config.bov_token_id
        else:
            self.use_weight = False

        if config.action_experts:
            self.action_experts = config.action_experts
            action_config = Emu3Config.from_dict(config.action_config)
            self.vision_loss_weight = action_config.vision_loss_weight
            self.action_projector = ActionProjector(config.action_dim, action_config.hidden_size)
            self.action_layers = nn.ModuleList(
                [Emu3DecoderLayer(action_config, layer_idx) for layer_idx in range(action_config.num_hidden_layers)]
            )
            self.action_decoder = FinalLayer(action_config.hidden_size, config.action_dim)
            # self.rf = FlowMatchingScheduler(sample_method="uniform", s = 1.0)
            self.rf = FlowMatchingScheduler(sample_method="beta", s = 1.0)
            self.tau_emb = SinusoidalPosEmb(action_config.hidden_size)
        
        # Output head (same as Emu3ForCausalLM)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def peft(self):
        if self.lora_modules == "default" or self.lora_modules == "qk":
            target_modules = ["q_proj", "v_proj"]
        elif self.lora_modules == "qkv":
            target_modules = ["q_proj", "k_proj", "v_proj"]
        elif self.lora_modules == "qkvo":
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,    
            r=256,   # 32                          
            lora_alpha=256,  # 32 
            lora_dropout=0.05,
            target_modules=target_modules
        )
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.model = get_peft_model(self.model, peft_config)
        if self.freeze:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
        
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        action: Optional[torch.Tensor] = None,
        return_logits: bool = False
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Example output will be the same as in Emu3ForCausalLM, with the inclusion of MoE-based processing.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        seq_len = hidden_states.shape[1]

        # processing action
        if action is not None and self.action_experts and self.training:
            # Generate noise with the same shape and data type as the action tensor
            noise = torch.randn_like(action, dtype=action.dtype)

            # Sample tau values and ensure the data type matches the noise tensor
            tau = self.rf.sample_t(noise.shape[0]).to(noise.dtype)

            noise_action = self.rf.add_noise(action, noise, tau)

            # Use forward_action to compute predictions and updated hidden states
            velo_pred, hidden_states_refine = self.forward_action(noise_action, tau, hidden_states)

            # flow matching loss
            loss_action = F.mse_loss(noise - action, velo_pred)

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            if self.use_weight:
                weights = torch.ones(self.config.vocab_size)
                vision_token_range = range(self.bov_token_id,self.eov_token_id+1)
                weights[vision_token_range] = self.vision_loss_weight
                loss_fct = CrossEntropyLoss(weight=weights.to(logits.device))

            else:
                loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            
            loss = loss_fct(shift_logits, shift_labels)
            if action is not None and self.action_experts:
                loss += loss_action * self.vision_loss_weight
            # loss = loss_action
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        if return_logits:
            return logits

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def forward_action(self, z, t, cond):

        # Embed the sampled tau values and adjust the data type to match the noise tensor
        tau_emb = self.tau_emb(t).to(z.dtype)

        # Repeat tau embeddings along the action dimension to match the input shape
        tau_emb = tau_emb.repeat(1, z.shape[1], 1)

        seq_len = cond.shape[1]

        # Compute action embeddings using the action projector and the tau embeddings
        action_hidden_states = self.action_projector(z, tau_emb)

        # Concat in sequence dimension
        action_hidden_states = torch.cat([cond, action_hidden_states], dim=1)
        # transformer layers
        for action_layer in self.action_layers:
            action_hidden_states = action_layer(
                action_hidden_states
            )[0]
        hidden_states, action_hidden_states = action_hidden_states[:, :seq_len, :], action_hidden_states[:, seq_len:, :]
        velo_pred = self.action_decoder(action_hidden_states, tau_emb)

        return velo_pred, hidden_states

    def generate_action(self, outputs, sample_steps = 20, frames = 8, action_dim = 7):

        input_ids = outputs
        batch_size, seq_len = input_ids.shape
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        hidden_states = outputs[0]

        # action generation 
        z = torch.randn((batch_size, frames, action_dim), dtype=hidden_states.dtype).to(hidden_states.device)
        dt = 1.0 / sample_steps

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * batch_size).to(hidden_states.device)

            velo_pred, hidden_states_i = self.forward_action(z, t, cond = hidden_states)

            z = z - dt * velo_pred
        
        return z

class LlamaWithSpeech(LlamaPreTrainedModel):
    def __init__(self, config, tokenizer, llama_path, speech_encoder_path, peft, freeze, debug=False, lora_modules="default", mix=False, generate=False, tts_loss_weight=0.5, encoder_type="mamba", time_block=1.0):
        super().__init__(config)
        
        # Base model (the same as in Emu3ForCausalLM)
        self.debug_mode = debug
        self.lora_modules = lora_modules
        self.generation = generate
        self.tts_loss_weight = tts_loss_weight
        self.encoder_type = encoder_type
        self.time_block = time_block
        if llama_path:
            self.model = LlamaForCausalLM.from_pretrained(
                llama_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            )
        else:
            self.model = LlamaForCausalLM(config)
        if not mix:
            self.model.resize_token_embeddings(len(tokenizer)) # old
        if peft:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
        if not mix:
            self.model.add_learnable_embeddings(7) # old
        else:
            self.model.add_learnable_embeddings(40000)
        # self.model.add_learnable_embeddings(8) # new
        if peft:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,    
                r=256,   # 64                          
                lora_alpha=256,  # 64 
                lora_dropout=0.05
                )
            self.model = get_peft_model(self.model, peft_config)
        if freeze:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        if mix:
            self.tokenizer = tokenizer[0]
        else:
            self.tokenizer = tokenizer
        self.speechencoder = self.get_speech_encoder(self.encoder_type)

        if speech_encoder_path:
            if self.encoder_type == "mamba":
                ckpt = torch.load(speech_encoder_path, map_location="cpu")
                new_ckpt = {"model": {}}
                for k in ckpt['model']:
                    if k.startswith('speech_encoder.'):
                        new_ckpt['model'][k.replace('speech_encoder.', '')] = ckpt['model'][k]
                ckpt = new_ckpt
                self.speechencoder.load_state_dict(ckpt["model"])
            elif self.encoder_type == "zipformer2":
                self.speechencoder.load_state_dict(torch.load(speech_encoder_path)["model"], strict=False)
        
        for name, param in self.speechencoder.named_parameters():
            param.requires_grad = False
        self.speechencoder.eval()
        self.ln_speech = nn.LayerNorm(self.speechencoder.encoder_dim)

        if self.encoder_type == "mamba":
            if self.time_block == 1.0:
                self.seg_size = 5
                self.hid_size = 4096
                self.speech_fps = 5
            elif self.time_block == 0.24:
                self.seg_size = 6
                self.hid_size = 5120
                self.speech_fps = 1
            elif self.time_block == 0.48:
                self.seg_size = 6
                self.hid_size = 5120
                self.speech_fps = 2
        elif self.encoder_type == "zipformer2":
            if self.time_block == 1.0:
                self.seg_size = 10
                self.hid_size = 6400
                self.speech_fps = 5
            elif self.time_block == 0.16:
                self.seg_size = 8
                self.hid_size = 6400
                self.speech_fps = 1

        self.speech_proj = nn.Sequential(
            nn.Linear(self.speechencoder.encoder_dim * self.seg_size, self.hid_size),
            nn.ReLU(),
            nn.Linear(self.hid_size, config.hidden_size)
        )

        if freeze:
            for name, param in self.ln_speech.named_parameters():
                param.requires_grad = False
            for name, param in self.speech_proj.named_parameters():
                param.requires_grad = False
        
        if self.generation:
            self.get_speech_generator()
    
    def merge_lora(self):
        self.model = self.model.merge_and_unload()

    def peft(self):
        if self.lora_modules == "default" or self.lora_modules == "qk":
            target_modules = ["q_proj", "v_proj"]
        elif self.lora_modules == "qkv":
            target_modules = ["q_proj", "k_proj", "v_proj"]
        elif self.lora_modules == "qkvo":
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,    
            r=256,   # 32                          
            lora_alpha=256,  # 32
            target_modules=target_modules, 
            lora_dropout=0.05
        )
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.model = get_peft_model(self.model, peft_config)
    
    def get_speech_encoder(self,encoder_type):
        def _to_int_tuple(s: str):
            return tuple(map(int, s.split(",")))
        
        if encoder_type == "mamba":
            sys.path.append(os.path.join(ELLSA_BASE_PATH,"reference"))
            from mamba_ssm import Mamba, Mamba2
            from stream_zf.mamba import MambaEncoder
            from stream_zf.model import MultiKDModel
            from stream_zf.scaling import ScheduledFloat
            from stream_zf.subsampling import Conv2dSubsampling

            encoder_embed = Conv2dSubsampling(
                in_channels=80,
                out_channels=_to_int_tuple("192,256,384,512,384,256")[0],
                dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
            )

            encoder = MambaEncoder(
                exp_factor=2,
                d_model=2048,
                n_layer=32,
                fused_add_norm=False,
                rms_norm=False,
                residual_in_fp32=True,
                bidirectional=False,
                d_intermediate=2048,
            )
            encoder_dim = 2048

            model = MultiKDModel(
                encoder_embed=encoder_embed,
                encoder=encoder,
                encoder_dim=encoder_dim,
                use_beats=True,
                use_ecapa=False,
                use_whisper=True,
                whisper_dim=1280,
                speaker_input_idx=-1,
                mvq_KD=False,
                use_subsampled_output=True,
                delta_t=6,
                use_mamba=True,
                whisper_init=False,
                whisper_chunk_size=1
            )
        elif encoder_type == "zipformer2":
            sys.path.append(os.path.join(ELLSA_BASE_PATH,"reference"))
            from spear_encoder.model import MultiKDModel
            from spear_encoder.scaling import ScheduledFloat
            from spear_encoder.subsampling import Conv2dSubsampling
            from spear_encoder.zipformer import Zipformer2

            encoder_embed = Conv2dSubsampling(
                in_channels=128,
                out_channels=_to_int_tuple("1280,1280,1280,1280,1280,1280,1280")[0],
                dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
            )

            encoder = Zipformer2(
                output_downsampling_factor=1,
                downsampling_factor=_to_int_tuple("1,2,4,8,4,2,1"),
                num_encoder_layers=_to_int_tuple("1,2,3,4,1,1,1"),
                encoder_dim=_to_int_tuple("1280,1280,1280,1280,1280,1280,1280"),
                encoder_unmasked_dim=_to_int_tuple("768,768,768,768,768,768,768"),
                query_head_dim=_to_int_tuple("32"),
                pos_head_dim=_to_int_tuple("4"),
                value_head_dim=_to_int_tuple("12"),
                pos_dim=48,
                num_heads=_to_int_tuple("8,8,8,8,8,8,8"),
                feedforward_dim=_to_int_tuple("3840,3840,3840,3840,3840,3840,3840"),
                cnn_module_kernel=_to_int_tuple("31,31,15,15,15,31,31"),
                dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
                warmup_batches=4000.0,
                causal=True,
                chunk_size=_to_int_tuple("8"),
                left_context_frames=_to_int_tuple("256"),
            )

            model = MultiKDModel(
                encoder_embed=encoder_embed,
                encoder=encoder,
                encoder_dim=max(_to_int_tuple("1280,1280,1280,1280,1280,1280,1280")),
                num_codebooks=0,
            )

        return model
    
    def get_speech_generator(self):
        self.generator = CosyVoice2(COSY_CKPT_PATH, load_jit=False, load_trt=False).model
        for param in self.generator.flow.parameters():
            param.requires_grad = False
        for param in self.generator.hift.parameters():
            param.requires_grad = False

        self.generator.llm.text_chunk = 8
        self.generator.llm.audio_chunk = 25
        self.generator.llm.only_streaming_training = True

        self.generator_proj = nn.Sequential(
            nn.Linear(self.hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.generator.llm.llm_input_size),
            nn.LayerNorm(self.generator.llm.llm_input_size)
        )

        self.embedding_num = -1

    def encode_speech(self, fbank_feature, fbank_feature_len):
        with self.maybe_autocast():
            if self.encoder_type == "mamba":
                speech_embeds, encoder_out_lens, spkr_embedding = self.speechencoder.get_embeddings(
                    fbank_feature,
                    fbank_feature_len,
                    extract_spkr_embed=False
                )
            elif self.encoder_type == "zipformer2":
                speech_embeds, encoder_out_lens = self.speechencoder.forward_encoder(
                    fbank_feature,
                    fbank_feature_len
                )
            bsz, seqlen, ndim = speech_embeds.size()
            x = self.ln_speech(speech_embeds)
            if seqlen % self.seg_size != 0:
                pad_embeds = torch.zeros(
                    (bsz, (seqlen // self.seg_size + 1) * self.seg_size - seqlen, ndim), dtype=speech_embeds.dtype, device=speech_embeds.device
                )
                speech_embeds = torch.cat((speech_embeds, pad_embeds), dim=1)

                bsz, seqlen, ndim = speech_embeds.size()
            speech_embeds = speech_embeds.view(bsz, seqlen // self.seg_size, ndim * self.seg_size)
            x = self.speech_proj(speech_embeds)
            return x
    
    def prepare_inputs_labels_for_speech(self, input_ids, attention_mask, labels, fbank_feature, fbank_feature_len, mix=False):
        bos_id = 128000
        boi_id = 151852 # not_used
        bot_id = 151842 # not_used
        boa_id = 151844 # not_used
        if labels is not None:
            speech_embeds = self.encode_speech(fbank_feature, fbank_feature_len)
            bsz, seqlen, dim = speech_embeds.size()
            input_embeds_list = []
            attention_mask_list = []
            labels_list = []
            for i in range(bsz):
                if is_peft_model(self.model):
                    current_input_embeds = self.model.model.model.embed_tokens(input_ids[i][0].to(torch.int64))
                else:
                    current_input_embeds = self.model.model.embed_tokens(input_ids[i][0].to(torch.int64))
                current_labels = labels[i]
                # find where currect_input_ids == bos_id, return all the index
                bos_idx = (input_ids[i][0] == bos_id).nonzero(as_tuple=True)[0]
                if mix:
                    idx_bos = (input_ids[i][0] == bos_id).nonzero(as_tuple=True)[0].tolist()
                    idx_boi = (input_ids[i][0] == boi_id).nonzero(as_tuple=True)[0].tolist()
                    idx_bot = (input_ids[i][0] == bot_id).nonzero(as_tuple=True)[0].tolist()
                    idx_boa = (input_ids[i][0] == boa_id).nonzero(as_tuple=True)[0].tolist()
                    idx_bos[0] = 0
                    idx_bos.append(-1)
                    token_maps = []
                    if len(idx_boi) != len(idx_boa):
                        for num in range(len(idx_bos)-1):
                            if num > len(idx_bos) - 4:
                                speech_len = (num + 1) * self.speech_fps
                                speech_idx = idx_bos[num] + speech_len - self.speech_fps
                                image_idx = idx_boi[2*(num - len(idx_bos) + 3)] + speech_len
                                text_idx = idx_bot[num] + speech_len
                                action_idx = idx_boa[num - len(idx_bos) + 3] + speech_len
                                next_speech_idx = idx_bos[num+1] + speech_len if idx_bos[num+1] > 0 else None
                                token_maps.append(["speech",[speech_idx,image_idx]])
                                token_maps.append(["vision",[image_idx,text_idx]])
                                token_maps.append(["speech",[text_idx,action_idx]])
                                token_maps.append(["vision",[action_idx,next_speech_idx]])
                            else:
                                speech_len = (num + 1) * self.speech_fps
                                speech_idx = idx_bos[num] + speech_len - self.speech_fps
                                text_idx = idx_bot[num] + speech_len
                                next_speech_idx = idx_bos[num+1] + speech_len if idx_bos[num+1] > 0 else None
                                token_maps.append(["speech",[speech_idx,next_speech_idx]])
                    
                    else:
                        for num in range(len(idx_bos)-1):
                            speech_len = (num + 1) * self.speech_fps
                            speech_idx = idx_bos[num] + speech_len - self.speech_fps
                            image_idx = idx_boi[num] + speech_len
                            text_idx = idx_bot[num] + speech_len
                            action_idx = idx_boa[num] + speech_len
                            next_speech_idx = idx_bos[num+1] + speech_len if idx_bos[num+1] > 0 else None
                            token_maps.append(["speech",[speech_idx,image_idx]])
                            token_maps.append(["vision",[image_idx,text_idx]])
                            token_maps.append(["speech",[text_idx,action_idx]])
                            token_maps.append(["vision",[action_idx,next_speech_idx]])
                step = 0
                for idx in bos_idx:
                    current_input_embeds = torch.cat((current_input_embeds[:idx+1],speech_embeds[i,step*self.speech_fps:(step+1)*self.speech_fps,:],current_input_embeds[idx+1:]),dim=0)
                    current_labels = torch.cat((current_labels[:idx+1], torch.full((self.speech_fps,), fill_value=-100, dtype=torch.long).to(current_labels.device),current_labels[idx+1:]),dim=0)
                    step += 1
                    bos_idx += self.speech_fps
                current_labels = current_labels[:len(current_input_embeds)]
                current_attention_mask = torch.ones((len(current_input_embeds),), dtype=torch.long).to(current_labels.device)
                input_embeds_list.append(current_input_embeds)
                attention_mask_list.append(current_attention_mask)
                labels_list.append(current_labels)
            input_embeds = rnn.pad_sequence(input_embeds_list, batch_first=True)
            attention_mask = rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
            labels = rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
            if mix:
                return input_embeds, attention_mask, labels, token_maps
            else:
                return input_embeds, attention_mask, labels
        else:
            speech_embeds = self.encode_speech(fbank_feature, fbank_feature_len)
            bsz, seqlen, dim = speech_embeds.size()
            input_embeds_list = []
            attention_mask_list = []
            for i in range(bsz):
                if is_peft_model(self.model):
                    current_input_embeds = self.model.model.model.embed_tokens(input_ids[i][0].to(torch.int64))
                else:
                    current_input_embeds = self.model.model.embed_tokens(input_ids[i][0].to(torch.int64))
                # find where currect_input_ids == bos_id, return all the index
                bos_idx = (input_ids[i][0] == bos_id).nonzero(as_tuple=True)[0]
                step = 0
                for idx in bos_idx:
                    current_input_embeds = torch.cat((current_input_embeds[:idx+1],speech_embeds[i,step*self.speech_fps:(step+1)*self.speech_fps,:],current_input_embeds[idx+1:]),dim=0)
                    step += 1
                    bos_idx += self.speech_fps
                current_attention_mask = torch.ones((len(current_input_embeds),), dtype=torch.long).to(current_input_embeds.device)
                input_embeds_list.append(current_input_embeds)
                attention_mask_list.append(current_attention_mask)
            input_embeds = rnn.pad_sequence(input_embeds_list, batch_first=True)
            attention_mask = rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
            return input_embeds, attention_mask, None

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast(device_type="cuda",dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def forward_speech_generate(self,hidden_states,labels,sent_lens,codecs,codec_lens):
        text_tokens = []
        text_token_lens = []
        speech_tokens = []
        speech_token_lens = []
        for i in range(len(sent_lens)):
            if len(sent_lens[i]) > 0:
                tts_mask = ((labels[i] != -100) & (labels[i] != 128260) & (labels[i] != 128261))
                valid_hidden_states = hidden_states[i, :-1][tts_mask]
                if valid_hidden_states.shape[0] != sum(sent_lens[i]):
                    tts_total = 0
                    for tts_idx, sent_len in enumerate(sent_lens[i]):
                        tts_total += sent_len
                        if tts_total > valid_hidden_states.shape[0]:
                            break
                    sent_lens[i] = sent_lens[i][:tts_idx]
                    valid_hidden_states = valid_hidden_states[:sum(sent_lens[i])]
                tts_features = torch.split(
                    valid_hidden_states, sent_lens[i]
                )
                for j in range(len(sent_lens[i])):
                    text_tokens.append(tts_features[j])
                    text_token_lens.append(sent_lens[i][j])
                    speech_tokens.append(codecs[i][j].squeeze(0))
                    speech_token_lens.append(codec_lens[i][j])

                    if text_token_lens[-1] / self.generator.llm.text_chunk > speech_token_lens[-1] / self.generator.llm.audio_chunk:
                        text_tokens = text_tokens[:-1]
                        text_token_lens = text_token_lens[:-1]
                        speech_tokens = speech_tokens[:-1]
                        speech_token_lens = speech_token_lens[:-1]

        if len(text_tokens) > 0:
            used_idx = random.sample(range(len(text_tokens)), min(len(text_tokens), 9))
            text_tokens = [text_tokens[i] for i in used_idx]
            text_token_lens = [text_token_lens[i] for i in used_idx]
            speech_tokens = [speech_tokens[i] for i in used_idx]
            speech_token_lens = [speech_token_lens[i] for i in used_idx]

            text_tokens = rnn.pad_sequence(text_tokens, batch_first=True).to(self.model.device)
            speech_tokens = rnn.pad_sequence(speech_tokens, batch_first=True).to(text_tokens.device)
            text_token_lens = torch.tensor(text_token_lens).to(text_tokens.device)
            speech_token_lens = torch.tensor(speech_token_lens).to(text_tokens.device)

            tts_loss_weight = self.tts_loss_weight
        else:
            text_tokens = hidden_states[0, :20].unsqueeze(0)
            text_token_lens = torch.tensor([text_tokens.shape[1]]).to(text_tokens.device)
            speech_tokens = torch.tensor([[0 for i in range(100)] for j in range(1)]).to(text_tokens.device)
            speech_token_lens = torch.tensor([100]).to(text_tokens.device)

            tts_loss_weight = 0.

        prompt_text_token = [[] for _ in range(len(text_tokens))]

        with self.maybe_autocast():
            text_tokens = self.generator_proj(text_tokens)
            cosy_outputs = self.generator.llm(text_tokens, text_token_lens, speech_tokens, speech_token_lens, prompt_text_token=prompt_text_token)
        
        return tts_loss_weight * cosy_outputs["loss"]

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        action = None,
        fbank_feature = None,
        fbank_feature_len = None,
        sent_lens = None,
        codecs = None,
        codec_lens = None,
        context_qa = None,
        distillation_labels = None,
        speech_distillation_labels = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Example output will be the same as in Emu3ForCausalLM, with the inclusion of MoE-based processing.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        input_embeds, attention_mask, labels = self.prepare_inputs_labels_for_speech(
            input_ids, attention_mask, labels, fbank_feature, fbank_feature_len
        )

        # Decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        if self.debug_mode:
            import pdb; pdb.set_trace()
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True if self.generation else output_hidden_states,
                return_dict=return_dict,
            )

        if self.generation and sent_lens is not None:
            generate_loss = self.forward_speech_generate(
                hidden_states=outputs.hidden_states[self.embedding_num],
                labels=labels[..., 1:].contiguous(),
                sent_lens=sent_lens,
                codecs=codecs,
                codec_lens=codec_lens
            )

            outputs.loss += generate_loss

            self._extra_logs = {
                "main_tts_loss": float(generate_loss.detach().mean())
            }

        return outputs
    
    def generate(
        self,
        input_ids,
        fbank_feature,
        fbank_feature_len,
        max_new_tokens=0,
        logits_processor=[],
        generation_config=None,
        **kwargs,
    ):

        if generation_config is None:
            from transformers import GenerationConfig
            generation_config = GenerationConfig.from_model_config(self.config)

        # 使用 kwargs 更新 generation_config
        for key, value in kwargs.items():
            if hasattr(generation_config, key):
                setattr(generation_config, key, value)

        if max_new_tokens == 0:
            max_new_tokens = generation_config.max_new_tokens
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        inputs_embeds, attention_mask, _ = self.prepare_inputs_labels_for_speech(
            input_ids, None, None, fbank_feature, fbank_feature_len
        )

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[torch.tensor([generation_config.eos_token_id]).cuda()])])

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
                num_beams=generation_config.num_beams,
                do_sample=generation_config.do_sample,
                min_length=generation_config.min_length,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                repetition_penalty=generation_config.repetition_penalty,
                length_penalty=generation_config.length_penalty,
                attention_mask=attention_mask,
                pad_token_id=self.model.config.eos_token_id[0]
            )
        
        return outputs

class Emu3ForMix(LlamaPreTrainedModel):
    def __init__(self, config_speech, config_vision, tokenizer, speech_encoder_path, peft, freeze, debug, attn_adapter=False, attn_adapter_type="None", merge_speech_lora=False, lora_modules="default", generate=False, action_loss_weight=1.0, test=False, encoder_type="mamba", time_block=1.0):
        super().__init__(config_speech)

        if test:
            speech_peft = False if merge_speech_lora else peft
            self.speech_expert = LlamaWithSpeech(config_speech, tokenizer[0], LLAMA_CKPT_PATH, speech_encoder_path, speech_peft, freeze, debug, lora_modules, generate=generate, encoder_type=encoder_type, time_block=time_block)
            self.vision_expert = Emu3MoE(config_vision, lora_modules=lora_modules, freeze=freeze)
            if peft:
                if merge_speech_lora:
                    self.speech_expert.peft()
                self.vision_expert.peft()
        self.debug_mode = debug
        self.gradient_checkpointing = False
        if time_block == 0.48:
            self.speech_fps = 2
        elif time_block == 1.0:
            self.speech_fps = 5
        self.attn_adapter = attn_adapter
        self.attn_adapter_type = attn_adapter_type
        self.generation = generate
        self.action_loss_weight = action_loss_weight
        if self.attn_adapter:
            vision_head_dim = config_vision.hidden_size // config_vision.num_attention_heads
            speech_head_dim = config_speech.hidden_size // config_speech.num_attention_heads
            vision_kv_dimension = config_vision.num_key_value_heads * vision_head_dim
            speech_kv_dimension = config_speech.num_key_value_heads * speech_head_dim
            if self.attn_adapter_type == "Linear":
                self.vision_expert_k_adapter = [nn.Linear(vision_kv_dimension, speech_kv_dimension) for _ in range(config_vision.num_hidden_layers)]
                self.vision_expert_v_adapter = [nn.Linear(vision_kv_dimension, speech_kv_dimension) for _ in range(config_vision.num_hidden_layers)]
                self.speech_expert_k_adapter = [nn.Linear(speech_kv_dimension, vision_kv_dimension) for _ in range(config_speech.num_hidden_layers)]
                self.speech_expert_v_adapter = [nn.Linear(speech_kv_dimension, vision_kv_dimension) for _ in range(config_speech.num_hidden_layers)]
            elif self.attn_adapter_type == "Only_Speech_Linear":
                self.speech_expert_k_adapter = [nn.Linear(speech_kv_dimension, vision_kv_dimension) for _ in range(config_speech.num_hidden_layers)]
                self.speech_expert_v_adapter = [nn.Linear(speech_kv_dimension, vision_kv_dimension) for _ in range(config_speech.num_hidden_layers)]
            elif self.attn_adapter_type == "MLP":
                self.vision_expert_k_adapter = [nn.Sequential(nn.Linear(vision_kv_dimension, speech_kv_dimension),nn.ReLU(),nn.Linear(speech_kv_dimension, speech_kv_dimension)) for _ in range(config_vision.num_hidden_layers)]
                self.vision_expert_v_adapter = [nn.Sequential(nn.Linear(vision_kv_dimension, speech_kv_dimension),nn.ReLU(),nn.Linear(speech_kv_dimension, speech_kv_dimension)) for _ in range(config_vision.num_hidden_layers)]
                self.speech_expert_k_adapter = [nn.Sequential(nn.Linear(speech_kv_dimension, vision_kv_dimension),nn.ReLU(),nn.Linear(vision_kv_dimension, vision_kv_dimension)) for _ in range(config_speech.num_hidden_layers)]
                self.speech_expert_v_adapter = [nn.Sequential(nn.Linear(speech_kv_dimension, vision_kv_dimension),nn.ReLU(),nn.Linear(vision_kv_dimension, vision_kv_dimension)) for _ in range(config_speech.num_hidden_layers)]
    
    def set_from_pretrained(self, speech_path, config_speech, vision_path, config_vision, tokenizer, speech_encoder_path, attn_implementation, torch_dtype, trust_remote_code, peft, freeze, debug, merge_speech_lora=False, lora_modules="default", generate=False, encoder_type="mamba", time_block=1.0):
        self.merge_speech_lora = merge_speech_lora
        self.speech_expert = LlamaWithSpeech.from_pretrained(
            speech_path,
            config=config_speech,
            tokenizer=tokenizer[0],
            llama_path=LLAMA_CKPT_PATH,
            speech_encoder_path=speech_encoder_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            peft=False if merge_speech_lora else peft,
            freeze=freeze,
            lora_modules=lora_modules,
            debug=debug,
            encoder_type=encoder_type,
            time_block=time_block
        )
        if generate:
            self.speech_expert.get_speech_generator()
        self.vision_expert = Emu3MoE.from_pretrained(
            vision_path,
            config=config_vision,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            lora_modules=lora_modules,
            freeze=freeze,
            trust_remote_code=trust_remote_code
        )
        self._use_sdpa = attn_implementation == "sdpa"
        self._use_flash_attention_2 = attn_implementation == "flash_attention_2"
    
    def peft(self):
        if self.merge_speech_lora:
            self.speech_expert.peft()
        self.vision_expert.peft()

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast(device_type="cuda",dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def pad_sequence_with_min_len(self, seqs, min_len=1200, batch_first=True, padding_value=0):

        padded = rnn.pad_sequence(
            seqs,
            batch_first=batch_first,
            padding_value=padding_value
        )

        cur_len = padded.size(1 if batch_first else 0)

        if cur_len < min_len:
            pad_size = list(padded.shape)
            pad_size[1 if batch_first else 0] = min_len - cur_len
            extra_pad = padded.new_full(pad_size, padding_value)

            padded = torch.cat([padded, extra_pad], dim=1 if batch_first else 0)

        return padded
    
    def prepare_inputs_labels_for_speech(self, input_ids, attention_mask, labels, fbank_feature, fbank_feature_len, mix=True, vla=True):
        bos_id = 128000
        boi_id = 151852 # not_used
        bot_id = 128259
        boa_id = 151844 # not_used
        if labels is not None:
            speech_embeds = self.speech_expert.encode_speech(fbank_feature, fbank_feature_len)
            bsz, seqlen, dim = speech_embeds.size()
            input_embeds_list = []
            attention_mask_list = []
            labels_list = []
            for i in range(bsz):
                if mix:
                    idx_bos = (input_ids[i][0] == bos_id).nonzero(as_tuple=True)[0].tolist()
                    idx_boi = (input_ids[i][0] == boi_id).nonzero(as_tuple=True)[0].tolist()
                    idx_bot = (input_ids[i][0] == bot_id).nonzero(as_tuple=True)[0].tolist()
                    idx_boa = (input_ids[i][0] == boa_id).nonzero(as_tuple=True)[0].tolist()
                    idx_bos[0] = 0
                    idx_bos.append(-1)
                    token_maps = []
                    token_maps_old = []
                    if len(idx_boi) != len(idx_boa):
                        for num in range(len(idx_bos)-1):
                            if num > len(idx_bos) - 4:
                                speech_len = (num + 1) * self.speech_fps
                                speech_idx = idx_bos[num] + speech_len - self.speech_fps
                                image_idx = idx_boi[2*(num - len(idx_bos) + 3)] + speech_len
                                text_idx = idx_bot[num] + speech_len
                                action_idx = idx_boa[num - len(idx_bos) + 3] + speech_len
                                next_speech_idx = idx_bos[num+1] + speech_len if idx_bos[num+1] > 0 else None
                                next_speech_idx_old = idx_bos[num+1] if idx_bos[num+1] > 0 else None
                                token_maps.append(["speech",[speech_idx,image_idx]])
                                token_maps.append(["vision",[image_idx,text_idx]])
                                token_maps.append(["speech",[text_idx,action_idx]])
                                token_maps.append(["vision",[action_idx,next_speech_idx]])
                                token_maps_old.append(["speech",[idx_bos[num],idx_boi[2*(num - len(idx_bos) + 3)]]])
                                token_maps_old.append(["vision",[idx_boi[2*(num - len(idx_bos) + 3)],idx_bot[num]]])
                                token_maps_old.append(["speech",[idx_bot[num],idx_boa[num - len(idx_bos) + 3]]])
                                token_maps_old.append(["vision",[idx_boa[num - len(idx_bos) + 3],next_speech_idx_old]])
                            else:
                                speech_len = (num + 1) * self.speech_fps
                                speech_idx = idx_bos[num] + speech_len - self.speech_fps
                                text_idx = idx_bot[num] + speech_len
                                next_speech_idx = idx_bos[num+1] + speech_len if idx_bos[num+1] > 0 else None
                                next_speech_idx_old = idx_bos[num+1] if idx_bos[num+1] > 0 else None
                                token_maps.append(["speech",[speech_idx,next_speech_idx]])
                                token_maps_old.append(["speech",[idx_bos[num],next_speech_idx_old]])
                    else:
                        for num in range(len(idx_bos)-1):
                            speech_len = (num + 1) * self.speech_fps
                            speech_idx = idx_bos[num] + speech_len - self.speech_fps
                            image_idx = idx_boi[num] + speech_len
                            text_idx = idx_bot[num] + speech_len
                            action_idx = idx_boa[num] + speech_len
                            next_speech_idx = idx_bos[num+1] + speech_len if idx_bos[num+1] > 0 else None
                            next_speech_idx_old = idx_bos[num+1] if idx_bos[num+1] > 0 else None
                            token_maps.append(["speech",[speech_idx,image_idx]])
                            token_maps.append(["vision",[image_idx,text_idx]])
                            token_maps.append(["speech",[text_idx,action_idx]])
                            token_maps.append(["vision",[action_idx,next_speech_idx]])
                            token_maps_old.append(["speech",[idx_bos[num],idx_boi[num]]])
                            token_maps_old.append(["vision",[idx_boi[num],idx_bot[num]]])
                            token_maps_old.append(["speech",[idx_bot[num],idx_boa[num]]])
                            token_maps_old.append(["vision",[idx_boa[num],next_speech_idx_old]])
                current_input_embeds = []
                if self.debug_mode:
                    import pdb; pdb.set_trace()
                for mapping in token_maps_old:
                    if mapping[0] == "speech":
                        current_input_embeds.append(self.speech_expert.model.model.model.embed_tokens(input_ids[i][0][mapping[1][0]:mapping[1][1]].to(torch.int64)))
                    elif mapping[0] == "vision":
                        current_input_embeds.append(self.vision_expert.model.embed_tokens(input_ids[i][0][mapping[1][0]:mapping[1][1]].to(torch.int64)))
                current_input_embeds = torch.cat(current_input_embeds,dim=0)
                current_labels = labels[i]
                # find where currect_input_ids == bos_id, return all the index
                bos_idx = (input_ids[i][0] == bos_id).nonzero(as_tuple=True)[0]
                step = 0
                for idx in bos_idx:
                    current_input_embeds = torch.cat((current_input_embeds[:idx+1],speech_embeds[i,step*self.speech_fps:(step+1)*self.speech_fps,:],current_input_embeds[idx+1:]),dim=0)
                    current_labels = torch.cat((current_labels[:idx+1], torch.full((self.speech_fps,), fill_value=-100, dtype=torch.long).to(current_labels.device),current_labels[idx+1:]),dim=0)
                    step += 1
                    bos_idx += self.speech_fps
                current_labels = current_labels[:len(current_input_embeds)]
                current_attention_mask = torch.ones((len(current_input_embeds),), dtype=torch.long).to(current_labels.device)
                input_embeds_list.append(current_input_embeds)
                attention_mask_list.append(current_attention_mask)
                labels_list.append(current_labels)
            """
            input_embeds = rnn.pad_sequence(input_embeds_list, batch_first=True)
            attention_mask = rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
            labels = rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
            """
            input_embeds = self.pad_sequence_with_min_len(input_embeds_list, batch_first=True)
            attention_mask = self.pad_sequence_with_min_len(attention_mask_list, batch_first=True, padding_value=0)
            labels = self.pad_sequence_with_min_len(labels_list, batch_first=True, padding_value=-100)
            if mix:
                return input_embeds, attention_mask, labels, token_maps
            else:
                return input_embeds, attention_mask, labels
        else:
            speech_embeds = self.speech_expert.encode_speech(fbank_feature, fbank_feature_len)
            bsz, seqlen, dim = speech_embeds.size()
            input_embeds_list = []
            attention_mask_list = []
            for i in range(bsz):
                if mix:
                    idx_bos = (input_ids[i][0] == bos_id).nonzero(as_tuple=True)[0].tolist()
                    idx_boi = (input_ids[i][0] == boi_id).nonzero(as_tuple=True)[0].tolist()
                    idx_bot = (input_ids[i][0] == bot_id).nonzero(as_tuple=True)[0].tolist()
                    idx_boa = (input_ids[i][0] == boa_id).nonzero(as_tuple=True)[0].tolist()
                    idx_bos[0] = 0
                    idx_bos.append(-1)
                    token_maps = []
                    token_maps_old = []
                    if vla:
                        for num in range(len(idx_bos)-1):
                            if len(idx_boi) == 2:
                                control_num = 3
                            elif len(idx_boi) == 4:
                                control_num = 4
                            if num > len(idx_bos) - control_num:
                                speech_len = (num + 1) * self.speech_fps
                                speech_idx = idx_bos[num] + speech_len - self.speech_fps
                                image_idx = idx_boi[2*(num - len(idx_bos) + control_num - 1)] + speech_len
                                if num - len(idx_bos) + control_num - 1 < len(idx_boa):
                                    text_idx = idx_bot[num] + speech_len
                                    action_idx = idx_boa[num - len(idx_bos) + control_num - 1] + speech_len
                                    next_speech_idx = idx_bos[num+1] + speech_len if idx_bos[num+1] > 0 else None
                                    next_speech_idx_old = idx_bos[num+1] if idx_bos[num+1] > 0 else None
                                else:
                                    text_idx = idx_bot[num] + speech_len
                                token_maps.append(["speech",[speech_idx,image_idx]])
                                token_maps.append(["vision",[image_idx,text_idx]])
                                token_maps_old.append(["speech",[idx_bos[num],idx_boi[2*(num - len(idx_bos) + control_num - 1)]]])
                                token_maps_old.append(["vision",[idx_boi[2*(num - len(idx_bos) + control_num - 1)],idx_bot[num]]])
                                if num - len(idx_bos) + control_num - 1 < len(idx_boa):
                                    token_maps.append(["speech",[text_idx,action_idx]])
                                    token_maps.append(["vision",[action_idx,next_speech_idx]])
                                    token_maps_old.append(["speech",[idx_bot[num],idx_boa[num - len(idx_bos) + control_num - 1]]])
                                    token_maps_old.append(["vision",[idx_boa[num - len(idx_bos) + control_num - 1],next_speech_idx_old]])
                                else:
                                    token_maps.append(["speech",[text_idx,None]])
                                    token_maps_old.append(["speech",[idx_bot[num],None]])
                            else:
                                speech_len = (num + 1) * self.speech_fps
                                speech_idx = idx_bos[num] + speech_len - self.speech_fps
                                text_idx = idx_bot[num] + speech_len
                                next_speech_idx = idx_bos[num+1] + speech_len if idx_bos[num+1] > 0 else None
                                next_speech_idx_old = idx_bos[num+1] if idx_bos[num+1] > 0 else None
                                token_maps.append(["speech",[speech_idx,next_speech_idx]])
                                token_maps_old.append(["speech",[idx_bos[num],next_speech_idx_old]])
                    else:
                        for num in range(len(idx_bos)-1):
                            speech_len = (num + 1) * self.speech_fps
                            speech_idx = idx_bos[num] + speech_len - self.speech_fps
                            image_idx = idx_boi[num] + speech_len
                            text_idx = idx_bot[num] + speech_len
                            if num < len(idx_boa):
                                action_idx = idx_boa[num] + speech_len
                                next_speech_idx = idx_bos[num+1] + speech_len if idx_bos[num+1] > 0 else None
                                next_speech_idx_old = idx_bos[num+1] if idx_bos[num+1] > 0 else None
                            else:
                                pass
                            token_maps.append(["speech",[speech_idx,image_idx]])
                            token_maps.append(["vision",[image_idx,text_idx]])
                            token_maps_old.append(["speech",[idx_bos[num],idx_boi[num]]])
                            token_maps_old.append(["vision",[idx_boi[num],idx_bot[num]]])
                            if num < len(idx_boa):
                                token_maps.append(["speech",[text_idx,action_idx]])
                                token_maps.append(["vision",[action_idx,next_speech_idx]])
                                token_maps_old.append(["speech",[idx_bot[num],idx_boa[num]]])
                                token_maps_old.append(["vision",[idx_boa[num],next_speech_idx_old]])
                            else:
                                token_maps.append(["speech",[text_idx,None]])
                                token_maps_old.append(["speech",[idx_bot[num],None]])
                current_input_embeds = []
                if self.debug_mode:
                    print(token_maps_old)
                    print(token_maps)
                    print(idx_bos,idx_boi,idx_bot,idx_boa)
                    import pdb; pdb.set_trace()
                for mapping in token_maps_old:
                    if mapping[0] == "speech":
                        current_input_embeds.append(self.speech_expert.model.model.model.embed_tokens(input_ids[i][0][mapping[1][0]:mapping[1][1]].to(torch.int64)))
                    elif mapping[0] == "vision":
                        current_input_embeds.append(self.vision_expert.model.embed_tokens(input_ids[i][0][mapping[1][0]:mapping[1][1]].to(torch.int64)))
                current_input_embeds = torch.cat(current_input_embeds,dim=0)
                # find where currect_input_ids == bos_id, return all the index
                bos_idx = (input_ids[i][0] == bos_id).nonzero(as_tuple=True)[0]
                step = 0
                for idx in bos_idx:
                    current_input_embeds = torch.cat((current_input_embeds[:idx+1],speech_embeds[i,step*self.speech_fps:(step+1)*self.speech_fps,:],current_input_embeds[idx+1:]),dim=0)
                    step += 1
                    bos_idx += self.speech_fps
                current_attention_mask = torch.ones((len(current_input_embeds),), dtype=torch.long).to(current_input_embeds.device)
                input_embeds_list.append(current_input_embeds)
                attention_mask_list.append(current_attention_mask)
            input_embeds = rnn.pad_sequence(input_embeds_list, batch_first=True)
            attention_mask = rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
            return input_embeds, attention_mask, None, token_maps

    def joint_attn(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        token_maps: list = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Emu3FlashAttention2 attention does not support output_attentions
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        speech_layer = self.speech_expert.model.model.model.layers[layer_idx].self_attn
        vision_layer = self.vision_expert.model.layers[layer_idx].self_attn
        if self.attn_adapter:
            speech_k_adapter = self.speech_expert_k_adapter[layer_idx].to(hidden_states.device)
            speech_v_adapter = self.speech_expert_v_adapter[layer_idx].to(hidden_states.device)
            if self.attn_adapter_type == "Only_Speech_Linear":
                vision_k_adapter = None
                vision_v_adapter = None
            else:
                vision_k_adapter = self.vision_expert_k_adapter[layer_idx].to(hidden_states.device)
                vision_v_adapter = self.vision_expert_v_adapter[layer_idx].to(hidden_states.device)

        query_states_speech = speech_layer.q_proj(hidden_states)
        key_states_speech = speech_layer.k_proj(hidden_states)
        value_states_speech = speech_layer.v_proj(hidden_states)

        query_states_vision = vision_layer.q_proj(hidden_states)
        key_states_vision = vision_layer.k_proj(hidden_states)
        value_states_vision = vision_layer.v_proj(hidden_states)

        query_states = []
        key_states = []
        value_states = []
        for mapping in token_maps:
            if mapping[0] == "speech":
                query_states.append(query_states_speech[..., mapping[1][0]:mapping[1][1], :])
                key_states.append(key_states_speech[..., mapping[1][0]:mapping[1][1], :])
                value_states.append(value_states_speech[..., mapping[1][0]:mapping[1][1], :])
            elif mapping[0] == "vision":
                query_states.append(query_states_vision[..., mapping[1][0]:mapping[1][1], :])
                key_states.append(key_states_vision[..., mapping[1][0]:mapping[1][1], :])
                value_states.append(value_states_vision[..., mapping[1][0]:mapping[1][1], :])
        query_states = torch.cat(query_states,dim=1)
        key_states = torch.cat(key_states,dim=1)
        value_states = torch.cat(value_states,dim=1)
        del query_states_speech
        del key_states_speech
        del value_states_speech
        del query_states_vision
        del key_states_vision
        del value_states_vision

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, speech_layer.num_heads, speech_layer.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, speech_layer.num_key_value_heads, speech_layer.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, speech_layer.num_key_value_heads, speech_layer.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, layer_idx)
        
        cos, sin = speech_layer.rotary_emb(value_states, position_ids)
        query_states_speech, key_states_speech = apply_rotary_pos_emb_llama(query_states, key_states, cos, sin)
        cos_vision, sin_vision = vision_layer.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states_vision, key_states_vision = apply_rotary_pos_emb(query_states, key_states, cos_vision, sin_vision, position_ids)
        query_states = []
        key_states = []
        for mapping in token_maps:
            if mapping[0] == "speech":
                query_states.append(query_states_speech[..., mapping[1][0]:mapping[1][1], :])
                key_states.append(key_states_speech[..., mapping[1][0]:mapping[1][1], :])
            elif mapping[0] == "vision":
                query_states.append(query_states_vision[..., mapping[1][0]:mapping[1][1], :])
                key_states.append(key_states_vision[..., mapping[1][0]:mapping[1][1], :])
        query_states = torch.cat(query_states,dim=2)
        key_states = torch.cat(key_states,dim=2)
        del query_states_speech
        del key_states_speech
        del query_states_vision
        del key_states_vision

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # dropout_rate = speech_layer.attention_dropout if speech_layer.training else 0.0
        dropout_rate = 0.0 # no dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (Emu3RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(speech_layer.config, "_pre_quantization_dtype"):
                target_dtype = speech_layer.config._pre_quantization_dtype
            else:
                target_dtype = speech_layer.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        if self.attn_adapter:
            key_states = key_states.reshape(bsz, key_states.shape[1], speech_layer.num_key_value_heads * speech_layer.head_dim).contiguous()
            value_states = value_states.reshape(bsz, value_states.shape[1], speech_layer.num_key_value_heads * speech_layer.head_dim).contiguous()
            key_states_speech_adapted = speech_k_adapter(key_states)
            if vision_k_adapter is None:
                key_states_vision_adapted = key_states
            else:
                key_states_vision_adapted = vision_k_adapter(key_states)
            value_states_speech_adapted = speech_v_adapter(value_states)
            if vision_v_adapter is None:
                value_states_vision_adapted = value_states
            else:
                value_states_vision_adapted = vision_v_adapter(value_states)
            key_states_for_speech = []
            key_states_for_vision = []
            value_states_for_speech = []
            value_states_for_vision = []
            for mapping in token_maps:
                if mapping[0] == "speech":
                    key_states_for_speech.append(key_states[..., mapping[1][0]:mapping[1][1], :])
                    key_states_for_vision.append(key_states_speech_adapted[..., mapping[1][0]:mapping[1][1], :])
                    value_states_for_speech.append(value_states[..., mapping[1][0]:mapping[1][1], :])
                    value_states_for_vision.append(value_states_speech_adapted[..., mapping[1][0]:mapping[1][1], :])
                elif mapping[0] == "vision":
                    key_states_for_speech.append(key_states_vision_adapted[..., mapping[1][0]:mapping[1][1], :])
                    key_states_for_vision.append(key_states[..., mapping[1][0]:mapping[1][1], :])
                    value_states_for_speech.append(value_states_vision_adapted[..., mapping[1][0]:mapping[1][1], :])
                    value_states_for_vision.append(value_states[..., mapping[1][0]:mapping[1][1], :])
            key_states_for_speech = torch.cat(key_states_for_speech,dim=1)
            value_states_for_speech = torch.cat(value_states_for_speech,dim=1)
            key_states_for_vision = torch.cat(key_states_for_vision,dim=1)
            value_states_for_vision = torch.cat(value_states_for_vision,dim=1)
            key_states_for_speech = key_states_for_speech.view(bsz, key_states.shape[1], speech_layer.num_key_value_heads, speech_layer.head_dim)
            value_states_for_speech = value_states_for_speech.view(bsz, value_states.shape[1], speech_layer.num_key_value_heads, speech_layer.head_dim)
            key_states_for_vision = key_states_for_vision.view(bsz, key_states.shape[1], speech_layer.num_key_value_heads, speech_layer.head_dim)
            value_states_for_vision = value_states_for_vision.view(bsz, value_states.shape[1], speech_layer.num_key_value_heads, speech_layer.head_dim)
            del key_states_speech_adapted
            del key_states_vision_adapted
            del value_states_speech_adapted
            del value_states_vision_adapted
            attn_output_speech = vision_layer._flash_attention_forward(
                query_states, key_states_for_speech, value_states_for_speech, attention_mask, q_len, dropout=dropout_rate
            )
            attn_output_vision = vision_layer._flash_attention_forward(
                query_states, key_states_for_vision, value_states_for_vision, attention_mask, q_len, dropout=dropout_rate
            )
            attn_output_speech = attn_output_speech.reshape(bsz, q_len, speech_layer.hidden_size).contiguous()
            attn_output_vision = attn_output_vision.reshape(bsz, q_len, vision_layer.hidden_size).contiguous()
            attn_output = []
            for mapping in token_maps:
                if mapping[0] == "speech":
                    attn_output.append(attn_output_speech[..., mapping[1][0]:mapping[1][1], :])
                elif mapping[0] == "vision":
                    attn_output.append(attn_output_vision[..., mapping[1][0]:mapping[1][1], :])
            attn_output = torch.cat(attn_output, dim=1)
            del key_states_for_speech
            del value_states_for_speech
            del key_states_for_vision
            del value_states_for_vision
            del attn_output_speech
            del attn_output_vision
        else:
            attn_output = vision_layer._flash_attention_forward(
                query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
            )
            attn_output = attn_output.reshape(bsz, q_len, speech_layer.hidden_size).contiguous()

        attn_output_speech = speech_layer.o_proj(attn_output)
        attn_output_vision = vision_layer.o_proj(attn_output)
        attn_output = []
        for mapping in token_maps:
            if mapping[0] == "speech":
                attn_output.append(attn_output_speech[..., mapping[1][0]:mapping[1][1], :])
            elif mapping[0] == "vision":
                attn_output.append(attn_output_vision[..., mapping[1][0]:mapping[1][1], :])
        attn_output = torch.cat(attn_output, dim=1)
        del attn_output_speech
        del attn_output_vision

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def joint_layer_decode(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        token_maps: list = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states
        speech_layer = self.speech_expert.model.model.model.layers[layer_idx]
        vision_layer = self.vision_expert.model.layers[layer_idx]

        hidden_states_speech = speech_layer.input_layernorm(hidden_states)
        hidden_states_vision = vision_layer.input_layernorm(hidden_states)
        hidden_states = []
        for mapping in token_maps:
            if mapping[0] == "speech":
                hidden_states.append(hidden_states_speech[..., mapping[1][0]:mapping[1][1], :])
            elif mapping[0] == "vision":
                hidden_states.append(hidden_states_vision[..., mapping[1][0]:mapping[1][1], :])
        hidden_states = torch.cat(hidden_states,dim=1)
        del hidden_states_speech
        del hidden_states_vision

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.joint_attn(
            hidden_states=hidden_states,
            layer_idx=layer_idx,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            token_maps=token_maps,
            **kwargs,
        )
        # hidden_states = residual + speech_layer.dropout(hidden_states)
        hidden_states = residual + hidden_states # no dropout

        # Fully Connected
        residual = hidden_states

        hidden_states_speech = speech_layer.post_attention_layernorm(hidden_states)
        hidden_states_vision = vision_layer.post_attention_layernorm(hidden_states)
        hidden_states = []
        for mapping in token_maps:
            if mapping[0] == "speech":
                hidden_states.append(hidden_states_speech[..., mapping[1][0]:mapping[1][1], :])
            elif mapping[0] == "vision":
                hidden_states.append(hidden_states_vision[..., mapping[1][0]:mapping[1][1], :])
        hidden_states = torch.cat(hidden_states,dim=1)
        del hidden_states_speech
        del hidden_states_vision


        hidden_states_speech = speech_layer.mlp(hidden_states)
        hidden_states_vision = vision_layer.mlp(hidden_states)
        hidden_states = []
        for mapping in token_maps:
            if mapping[0] == "speech":
                hidden_states.append(hidden_states_speech[..., mapping[1][0]:mapping[1][1], :])
            elif mapping[0] == "vision":
                hidden_states.append(hidden_states_vision[..., mapping[1][0]:mapping[1][1], :])
        hidden_states = torch.cat(hidden_states,dim=1)
        del hidden_states_speech
        del hidden_states_vision

        # hidden_states = residual + speech_layer.dropout(hidden_states)
        hidden_states = residual + hidden_states # no dropout

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    def joint_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_maps: list = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            if token_maps[0][0] == "speech":
                inputs_embeds = self.speech_expert.model.model.model.embed_tokens(input_ids)
            elif token_maps[0][0] == "vision":
                inputs_embeds = self.vision_expert.model.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        # hidden_states = self.speech_expert.model.model.model.dropout(inputs_embeds)
        hidden_states = inputs_embeds # no dropout

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx in range(len(self.speech_expert.model.model.model.layers)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    self.joint_layer_decode,
                    hidden_states,
                    layer_idx,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    token_maps,
                )
            else:
                layer_outputs = self.joint_layer_decode(
                    hidden_states,
                    layer_idx=layer_idx,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    token_maps=token_maps
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states_speech = self.speech_expert.model.model.model.norm(hidden_states)
        hidden_states_vision = self.vision_expert.model.norm(hidden_states)
        hidden_states = []
        for mapping in token_maps:
            if mapping[0] == "speech":
                hidden_states.append(hidden_states_speech[..., mapping[1][0]:mapping[1][1], :])
            elif mapping[0] == "vision":
                hidden_states.append(hidden_states_vision[..., mapping[1][0]:mapping[1][1], :])
        hidden_states = torch.cat(hidden_states,dim=1)
        del hidden_states_speech
        del hidden_states_vision

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
        action = None,
        fbank_feature = None,
        fbank_feature_len = None,
        sent_lens = None,
        codecs = None,
        codec_lens = None,
        context_qa = None,
        distillation_labels = None,
        speech_distillation_labels = None,
        data_type = "mix"
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Example output will be the same as in Emu3ForCausalLM, with the inclusion of MoE-based processing.
        """
        if data_type == "speech_only":
            outputs = self.speech_expert.forward(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels,
                fbank_feature = fbank_feature,
                fbank_feature_len = fbank_feature_len,
                return_dict=True
            )
            """
            placeholder_outputs = self.vision_expert.forward(
                input_ids = torch.LongTensor([0 for _ in range(500)]).unsqueeze(0).to(input_ids[0][0].device),
                attention_mask = torch.ones((500,), dtype=torch.long).unsqueeze(0).to(input_ids[0][0].device),
                labels = torch.ones((500,), dtype=torch.long).unsqueeze(0).to(input_ids[0][0].device),
                return_dict=True
            )
            outputs.loss = outputs.loss + 0 * placeholder_outputs.loss
            """
            return outputs
        elif data_type == "action_only":
            outputs = self.vision_expert.forward(
                input_ids = input_ids[0].unsqueeze(0),
                attention_mask = attention_mask[0].unsqueeze(0),
                labels = labels[0].unsqueeze(0),
                return_dict=True
            )
            """
            placeholder_outputs = self.speech_expert.forward(
                input_ids = [torch.LongTensor([0 for _ in range(199)]+[128000]+[0 for _ in range(300)]).unsqueeze(0).to(input_ids[0].device)],
                attention_mask = torch.ones((500,), dtype=torch.long).unsqueeze(0).to(input_ids[0].device),
                labels = torch.ones((500,), dtype=torch.long).unsqueeze(0).to(input_ids[0].device),
                fbank_feature = torch.ones((200,128)).unsqueeze(0).to(input_ids[0].device),
                fbank_feature_len = torch.LongTensor([200]).to(input_ids[0].device),
                return_dict=True
            )
            outputs.loss = outputs.loss + 0 * placeholder_outputs.loss
            """
            return outputs

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_embeds, attention_mask, labels, token_maps = self.prepare_inputs_labels_for_speech(
            input_ids, attention_mask, labels, fbank_feature, fbank_feature_len, mix=True
        )
        # token_maps like [["speech",[0,1]],["vision",[1,3]],...]

        # Decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # if self.debug_mode:
        #     import pdb; pdb.set_trace()
        with self.maybe_autocast():
            outputs = self.joint_forward(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True if self.generation else output_hidden_states,
                return_dict=return_dict,
                token_maps=token_maps
            )

        hidden_states = outputs[0]
        if self.generation:
            tts_hidden_states = outputs.hidden_states[self.speech_expert.embedding_num]
            tts_hidden_states_speech_used = []

        seq_len = hidden_states.shape[1]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits_speech = self.speech_expert.model.model.lm_head(hidden_states)
            logits_vision = self.vision_expert.lm_head(hidden_states)
            logits_speech_used = []
            logits_vision_used = []
            labels_speech_used = []
            labels_vision_used = []
            for mapping in token_maps:
                if mapping[0] == "speech":
                    logits_speech_used.append(logits_speech[..., mapping[1][0]:mapping[1][1], :])
                    labels_speech_used.append(labels[..., mapping[1][0]:mapping[1][1]])
                    if self.generation:
                        tts_hidden_states_speech_used.append(tts_hidden_states[..., mapping[1][0]:mapping[1][1], :])
                elif mapping[0] == "vision":
                    logits_vision_used.append(logits_vision[..., mapping[1][0]:mapping[1][1], :])
                    labels_vision_used.append(labels[..., mapping[1][0]:mapping[1][1]])
            logits_speech_used = torch.cat(logits_speech_used,dim=1)
            logits_vision_used = torch.cat(logits_vision_used,dim=1)
            labels_speech_used = torch.cat(labels_speech_used,dim=1)
            labels_vision_used = torch.cat(labels_vision_used,dim=1)
            if self.generation:
                tts_hidden_states_speech_used = torch.cat(tts_hidden_states_speech_used,dim=1)
            del logits_speech
            del logits_vision
        logits_speech_used = logits_speech_used.float()
        logits_vision_used = logits_vision_used.float()
        if self.generation:
            tts_hidden_states_speech_used = tts_hidden_states_speech_used.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            logits_speech_used = logits_speech_used[..., :-1, :].contiguous()
            logits_vision_used = logits_vision_used[..., :-1, :].contiguous()
            labels_speech_used = labels_speech_used[..., 1:].contiguous()
            labels_vision_used = labels_vision_used[..., 1:].contiguous()
            # Flatten the tokens
            # loss_fct = LigerCrossEntropyLoss()
            loss_fct = CrossEntropyLoss()
            shift_logits_speech = logits_speech_used.view(-1, self.speech_expert.model.config.vocab_size)
            shift_logits_vision = logits_vision_used.view(-1, self.vision_expert.config.vocab_size)
            shift_labels_speech = labels_speech_used.view(-1)
            shift_labels_vision = labels_vision_used.view(-1)
            # Enable model parallelism
            shift_labels_vision = shift_labels_vision.to(shift_logits_vision.device)
            shift_labels_speech = shift_labels_speech.to(shift_logits_speech.device)
            valid_speech = (shift_labels_speech != -100).sum().item()
            valid_vision = (shift_labels_vision != -100).sum().item()
            if distillation_labels is not None:
                distribution = distillation_labels[0]
                new_labels_for_vision = torch.zeros_like(shift_logits_vision)
                bos_idx = (shift_labels_vision.squeeze() != -100).nonzero(as_tuple=True)[0]
                continuous_list = []
                last = bos_idx[0]
                start = bos_idx[0]
                for idx in bos_idx[1:]:
                    if idx != last + 1:
                        continuous_list.append([start,last])
                        start = idx
                        last = idx
                    else:
                        last = idx
                continuous_list.append([start,last])
                num = 0
                for item in continuous_list:
                    start = item[0]
                    end = item[1]
                    new_labels_for_vision[start:end+1, 149594:151846] = distribution[num]
                    num += 1
                loss = loss_fct(shift_logits_vision, new_labels_for_vision) * valid_vision / (valid_speech + valid_vision) * self.action_loss_weight + loss_fct(shift_logits_vision, shift_labels_vision) * valid_vision / (valid_speech + valid_vision) * self.action_loss_weight + loss_fct(shift_logits_speech, shift_labels_speech) * valid_speech / (valid_speech + valid_vision)
            elif speech_distillation_labels is not None:
                if self.debug_mode:
                    import pdb; pdb.set_trace()
                distribution = speech_distillation_labels[0]
                new_labels_for_speech = torch.zeros_like(shift_logits_speech)
                bos_idx = (shift_labels_speech.squeeze() != -100).nonzero(as_tuple=True)[0]
                continuous_list = []
                last = bos_idx[0]
                start = bos_idx[0]
                for idx in bos_idx[1:]:
                    if idx != last + 1:
                        continuous_list.append([start,last])
                        start = idx
                        last = idx
                    else:
                        last = idx
                continuous_list.append([start,last])
                num = 0
                for item in continuous_list:
                    start = item[0]
                    end = item[1]
                    new_labels_for_speech[start:end+1, :] = distribution[num]
                    num += 1
                loss = loss_fct(shift_logits_vision, shift_labels_vision) * valid_vision / (valid_speech + valid_vision) * self.action_loss_weight + loss_fct(shift_logits_speech, shift_labels_speech) * valid_speech / (valid_speech + valid_vision) + loss_fct(shift_logits_speech, new_labels_for_speech) * valid_speech / (valid_speech + valid_vision)
            else:
                loss = loss_fct(shift_logits_vision, shift_labels_vision) * valid_vision / (valid_speech + valid_vision) * self.action_loss_weight + loss_fct(shift_logits_speech, shift_labels_speech) * valid_speech / (valid_speech + valid_vision)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        if self.generation and sent_lens is not None:
            generate_loss = self.speech_expert.forward_speech_generate(
                hidden_states=tts_hidden_states_speech_used,
                labels=labels_speech_used,
                sent_lens=sent_lens,
                codecs=codecs,
                codec_lens=codec_lens
            )

            loss = 0 * loss + generate_loss

            self._extra_logs = {
                "main_tts_loss": float(generate_loss.detach().mean())
            }

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits_speech_used,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def generate(
        self,
        input_ids,
        fbank_feature,
        fbank_feature_len,
        max_new_tokens=0,
        logits_processor=[],
        generation_config=None,
        vla=False,
        **kwargs,
    ):  
        tts_hidden_states_speech_used = []

        if generation_config is None:
            from transformers import GenerationConfig
            generation_config = GenerationConfig.from_model_config(self.config)

        # 使用 kwargs 更新 generation_config
        for key, value in kwargs.items():
            if hasattr(generation_config, key):
                setattr(generation_config, key, value)

        if max_new_tokens == 0:
            max_new_tokens = generation_config.max_new_tokens
        # eos_token_id = generation_config.eos_token_id
        # if isinstance(eos_token_id, int):
        #     eos_token_id = [eos_token_id]
        eos_token_id_text = [128260] # <eot>
        # eos_token_id_text = [128261] # <silence>
        eos_token_id_action = [151845]

        inputs_embeds, attention_mask, _, token_maps = self.prepare_inputs_labels_for_speech(
            input_ids, None, None, fbank_feature, fbank_feature_len, mix=True, vla=vla
        )

        # forward text
        with self.maybe_autocast():
            outputs = self.joint_forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                token_maps=token_maps,
                output_hidden_states=True if self.generation else False
            )
        
        if self.generation:
            tts_hidden_states = outputs.hidden_states[self.speech_expert.embedding_num]
            tts_hidden_states_speech_used.append(
                tts_hidden_states[..., -1:, :]
            )

        past_key_values = outputs.past_key_values
        logits = self.speech_expert.model.model.lm_head(outputs.last_hidden_state)
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        generated_ids_text = next_tokens.unsqueeze(-1)

        # 3. 自回归生成循环
        for _ in range(max_new_tokens - 1):
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

            with self.maybe_autocast():
                outputs = self.joint_forward(
                    input_ids=next_tokens.unsqueeze(-1),
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    use_cache=True,
                    token_maps=[["speech",[0,None]]],
                    output_hidden_states=True if self.generation else False
                )
                """
                outputs = self.speech_expert.model.model.model(
                    input_ids=next_tokens.unsqueeze(-1),
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    use_cache=True,
                    output_hidden_states=True if self.generation else False
                )
                """

            past_key_values = outputs.past_key_values
            logits = self.speech_expert.model.model.lm_head(outputs.last_hidden_state)
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            generated_ids_text = torch.cat([generated_ids_text, next_tokens.unsqueeze(-1)], dim=1)

            if eos_token_id_text is not None and next_tokens.item() in eos_token_id_text:
                break

            if self.generation:
                tts_hidden_states = outputs.hidden_states[self.speech_expert.embedding_num]
                tts_hidden_states_speech_used.append(
                    tts_hidden_states[..., -1:, :]
                )

        if self.generation and generated_ids_text[0, 0] == 128261:
            tts_hidden_states_speech_used = []

        # forward action
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

        with self.maybe_autocast():
            outputs = self.joint_forward(
                input_ids=next_tokens.unsqueeze(-1),
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                token_maps=[["speech",[0,None]]]
            )
            """
            outputs = self.speech_expert.model.model.model(
                input_ids=next_tokens.unsqueeze(-1),
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True
            )
            """

        past_key_values = outputs.past_key_values
        next_tokens = torch.tensor([151844]).to(outputs.last_hidden_state.device) # boa
        generated_ids_action = next_tokens.unsqueeze(-1)

        for _ in range(max_new_tokens - 1):
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

            with self.maybe_autocast():
                outputs = self.joint_forward(
                    input_ids=next_tokens.unsqueeze(-1),
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    use_cache=True,
                    token_maps=[["vision",[0,None]]]
                )
                """
                outputs = self.vision_expert.model(
                    input_ids=next_tokens.unsqueeze(-1),
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    use_cache=True
                )
                """

            past_key_values = outputs.past_key_values
            logits = self.vision_expert.lm_head(outputs.last_hidden_state)
            next_token_logits = logits[:, -1, :]
            if logits_processor:
                for logits_p in logits_processor:
                    next_token_logits = logits_p(None, next_token_logits)
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            generated_ids_action = torch.cat([generated_ids_action, next_tokens.unsqueeze(-1)], dim=1)

            if eos_token_id_action is not None and next_tokens.item() in eos_token_id_action:
                break

        return generated_ids_text, generated_ids_action, tts_hidden_states_speech_used