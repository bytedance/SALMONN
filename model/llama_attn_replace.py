from typing import Optional, Tuple
import warnings

import torch
from torch import nn
from .modeling_llama import apply_rotary_pos_emb
from .modeling_llama import LlamaAttention, LlamaModel

from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_kvpacked_func, flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input


# def forward(
#     self,
#     hidden_states: torch.Tensor,
#     attention_mask: Optional[torch.Tensor] = None,
#     position_ids: Optional[torch.Tensor] = None,
#     past_key_value: Optional[Tuple[torch.Tensor]] = None,
#     output_attentions: bool = False,
#     use_cache: bool = False,
# ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#     if output_attentions:
#         warnings.warn(
#             "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
#         )

#     bsz, q_len, _ = hidden_states.size()

#     query_states = (
#         self.q_proj(hidden_states)
#         .view(bsz, q_len, self.num_heads, self.head_dim)
#         .transpose(1, 2)
#     )
#     key_states = (
#         self.k_proj(hidden_states)
#         .view(bsz, q_len, self.num_heads, self.head_dim)
#         .transpose(1, 2)
#     )
#     value_states = (
#         self.v_proj(hidden_states)
#         .view(bsz, q_len, self.num_heads, self.head_dim)
#         .transpose(1, 2)
#     )  # shape: (b, num_heads, s, head_dim)

#     kv_seq_len = key_states.shape[-2]
#     if past_key_value is not None:
#         kv_seq_len += past_key_value[0].shape[-2]
#     cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#     query_states, key_states = apply_rotary_pos_emb(
#         query_states, key_states, cos, sin, position_ids
#     )

#     if past_key_value is not None:
#         # reuse k, v
#         key_states = torch.cat([past_key_value[0], key_states], dim=2)
#         value_states = torch.cat([past_key_value[1], value_states], dim=2)

#     past_key_value = (key_states, value_states) if use_cache else None

#     # Transform the data into the format required by flash attention
#     qkv = torch.stack([query_states, key_states, value_states], dim=2)
#     qkv = qkv.transpose(1, 3)  # shape: [b, s, 3, num_heads, head_dim]
#     key_padding_mask = attention_mask

#     if key_padding_mask is None:
#         qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)
#         cu_q_lens = torch.arange(
#             0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=qkv.device
#         )
#         max_s = q_len
#         output = flash_attn_varlen_qkvpacked_func(
#             qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
#         )
#         output = output.view(bsz, q_len, -1)
#     else:
#         qkv = qkv.reshape(bsz, q_len, -1)
#         qkv, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)
#         qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
#         output_unpad = flash_attn_varlen_qkvpacked_func(
#             qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True
#         )
#         output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
#         output = pad_input(output_unpad, indices, bsz, q_len)

#     return self.o_proj(output), None, past_key_value


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        warnings.warn(
            "Output attentions is not supported for patched `LlamaAttention`, returning `None` instead."
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )  # shape: (b, num_heads, s, head_dim)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        # reuse k, v
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    if attention_mask is None:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        output = flash_attn_func(query_states, key_states, value_states, 0.0, softmax_scale=None, causal=True).view(
            bsz, q_len, -1
        )
    else:
        q, indices, cu_q_lens, max_s = unpad_input(query_states.transpose(1, 2), attention_mask[:, -q_len:])
        kv = torch.stack((key_states, value_states), dim=2).transpose(1, 3)
        kv = kv.reshape(bsz, kv_seq_len, -1)
        kv, _, cu_k_lens, max_k = unpad_input(kv, attention_mask)
        kv = kv.view(-1, 2, self.num_heads, self.head_dim)
        output_unpad = flash_attn_varlen_kvpacked_func(
            q,
            kv,
            cu_q_lens,
            cu_k_lens,
            max_s,
            max_k,
            0.0,
            softmax_scale=None,
            causal=True,
        )
        output_unpad = output_unpad.reshape(-1, self.num_heads * self.head_dim)
        output = pad_input(output_unpad, indices, bsz, q_len)

    return self.o_proj(output), None, past_key_value


def _prepare_decoder_attention_mask_inference(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask


# Disable the transformation of the attention mask in LlamaModel as the flash attention
# requires the attention mask to be the same as the key_padding_mask
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


# def _prepare_decoder_attention_mask(
#     self, attention_mask, input_shape, inputs_embeds, past_key_values_length
# ):
#     # [bsz, seq_len]
#     if past_key_values_length > 0 and attention_mask is not None:
#         attention_mask = torch.cat(
#             (
#                 torch.full(
#                     (input_shape[0], past_key_values_length),
#                     True,
#                     dtype=attention_mask.dtype,
#                     device=attention_mask.device,
#                 ),
#                 attention_mask,
#             ),
#             dim=-1,
#         )

#     if attention_mask is not None and torch.all(attention_mask):
#         return None  # This uses the faster call when training with full samples

#     return attention_mask


def replace_llama_attn_with_flash_attn(inference=False):
    cuda_major, cuda_minor = torch.cuda.get_device_capability()
    if cuda_major < 8:
        warnings.warn(
            "Flash attention is only supported on A100 or H100 GPU during training due to head dim > 64 backward."
            "ref: https://github.com/HazyResearch/flash-attention/issues/190#issuecomment-1523359593"
        )
    if inference:
        LlamaModel._prepare_decoder_attention_mask = (
            _prepare_decoder_attention_mask_inference
        )
    else:
        LlamaModel._prepare_decoder_attention_mask = (
            _prepare_decoder_attention_mask
        )
    LlamaAttention.forward = forward