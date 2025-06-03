import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from cruise.utilities.distributed import DIST_ENV
from torch import nn
from transformers.utils.import_utils import is_flash_attn_2_available

# from xenon_generation.utils.comm_utils import all_to_all, initialize_parallel_state, all_to_all_h2s, all_to_all_s2h
from .communication import (
    all_to_all,
    all_to_all_h2s,
    all_to_all_s2h,
    initialize_parallel_state,
)

if is_flash_attn_2_available():
    try:
        from flash_attn_interface import flash_attn_varlen_func

        if DIST_ENV.rank == 0:
            print("FalshAttn Version: FlashAttention-3")
    except:
        from flash_attn import flash_attn_varlen_func

        if DIST_ENV.rank == 0:
            print("FalshAttn Version: FlashAttention-2")

# ======================================= self attention =======================================


class Attention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.use_gqa = self.num_attention_heads != self.num_key_value_heads

        self.sequence_parallel_size = config.sequence_parallel_size if hasattr(config, 'sequence_parallel_size') else 1
        self.enable_sequence_parallel = self.sequence_parallel_size > 1

        assert self.num_key_value_heads % self.sequence_parallel_size == 0
        assert self.num_attention_heads % self.sequence_parallel_size == 0

        self.use_flash_attn = config._attn_implementation == "flash_attention_2"
        #  if hasattr(config, 'use_ft_flash_attn') else False
        self.use_sdpa = config.use_sdpa if hasattr(config, 'use_sdpa') else False

        if self.use_flash_attn:
            assert is_flash_attn_2_available()
            self.attn = FlashAttention(config=config)
        else:
            self.attn = CoreAttention(config=config)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor = None, **args
    ):
        return self.attn(query, key, value, attention_mask, **args)


class CoreAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.use_sdpa = config.use_sdpa if hasattr(config, 'use_sdpa') else False
        self.scale = self.head_dim**-0.5
        # self.num_key_value_heads = config.num_key_value_heads
        # self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        # self.use_gqa = (self.num_attention_heads != self.num_key_value_heads)

        self.sequence_parallel_size = config.sequence_parallel_size if hasattr(config, 'sequence_parallel_size') else 1
        self.enable_sequence_parallel = self.sequence_parallel_size > 1

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor = None, **args
    ):
        # input_shape: [bs, seqlen_parallel, head_num, head_dim]
        # output_shape: [bs, seqlen_parallel, head_num, head_dim]

        # 如果用gqa, k,v需要先repeat

        bs, seqlen_parallel, head_num, head_dim = query.shape
        assert (
            head_num == self.num_attention_heads and head_dim == self.head_dim
        ), f"head_num={head_num}, head_dim={head_dim}"

        # TODO: 支持fuse qkv
        # [bs, seqlen_parallel, head_num, head_dim] -> [bs, seqlen, head_num_parallel, head_dim]
        if self.enable_sequence_parallel:
            query = all_to_all(query, DIST_ENV.distributed_sequence_parallel_group, scatter_dim=2, gather_dim=1)
            key = all_to_all(key, DIST_ENV.distributed_sequence_parallel_group, scatter_dim=2, gather_dim=1)
            value = all_to_all(value, DIST_ENV.distributed_sequence_parallel_group, scatter_dim=2, gather_dim=1)

        _, seqlen, head_num_parallel, _ = query.shape

        if attention_mask is not None:
            assert attention_mask.shape == (bs, 1, seqlen, seqlen)

        # [bs, seqlen, head_num_parallel, head_dim] -> [bs, head_num_parallel, seqlen, head_dim]
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        if self.use_sdpa:
            attn_output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, scale=self.scale
            ).contiguous()
        else:
            attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scale
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2)

        assert attn_output.shape == (bs, seqlen, head_num_parallel, head_dim), f"attn_output.shape={attn_output.shape}"

        # [bs, seqlen, head_num_parallel, head_dim] -> [bs, seqlen_parallel, head_num, head_dim]
        if self.enable_sequence_parallel:
            attn_output = all_to_all(
                attn_output, DIST_ENV.distributed_sequence_parallel_group, scatter_dim=1, gather_dim=2
            )

        assert attn_output.shape == (bs, seqlen_parallel, head_num, head_dim), f"attn_output.shape={attn_output.shape}"

        return attn_output


class FlashAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scale = self.head_dim**-0.5

        self.sequence_parallel_size = config.sequence_parallel_size if hasattr(config, 'sequence_parallel_size') else 1
        self.enable_sequence_parallel = self.sequence_parallel_size > 1

        self.sliding_window = config.sliding_window if hasattr(config, 'sliding_window') else None
        self.enable_sliding_window = True if self.sliding_window is not None else False

    #    assert not (self.enable_sequence_parallel and self.enable_sliding_window)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor = None,
        dropout: int = 0.0,
        causal: bool = True,
        **args,
    ):

        # flash_attn 自动支持gqa，可以不repeat

        # input_shape: [seqlen_parallel, head_num, head_dim]
        # output_shape: [seqlen_parallel, head_num, head_dim]
        seqlen_parallel, head_num, head_dim = query.shape

        # [seqlen_parallel, head_num, head_dim] -> [seqlen, head_num_parallel, head_dim]
        if self.enable_sequence_parallel:
            # query = all_to_all(query, DIST_ENV.distributed_sequence_parallel_group, scatter_dim=1, gather_dim=0)
            # key = all_to_all(key, DIST_ENV.distributed_sequence_parallel_group, scatter_dim=1, gather_dim=0)
            # value = all_to_all(value, DIST_ENV.distributed_sequence_parallel_group, scatter_dim=1, gather_dim=0)
            query = all_to_all_s2h(query)
            key = all_to_all_s2h(key)
            value = all_to_all_s2h(value)

        # window_size = (self.sliding_window, self.sliding_window) if self.enable_sliding_window else (-1, -1)
        window_size = (-1, -1)

        if args["sp_pad_size"] > 0:
            pad_size = args["sp_pad_size"]
            query, key, value = query[:-pad_size, ...], key[:-pad_size, ...], value[:-pad_size, ...]

        attn_output = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=args["cu_seqlens_q"],
            cu_seqlens_k=args["cu_seqlens_k"],
            max_seqlen_q=args["max_seqlen_q"],
            max_seqlen_k=args["max_seqlen_k"],
            causal=causal,
            window_size=window_size,
            softmax_scale=self.scale,
            # dropout_p=dropout,
        )
        # FAv3 return out, softmax_lse
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

        if args["sp_pad_size"] > 0:
            pad_tensor = torch.zeros(
                (pad_size,) + attn_output.shape[1:], dtype=attn_output.dtype, device=attn_output.device
            )
            attn_output = torch.cat((attn_output, pad_tensor), dim=0)

        # [seqlen, head_num_parallel, head_dim] -> [seqlen_parallel, head_num, head_dim]
        if self.enable_sequence_parallel:
            attn_output = all_to_all_h2s(attn_output)

        assert attn_output.shape == (seqlen_parallel, head_num, head_dim)

        return attn_output


if __name__ == '__main__':
    import deepspeed

    from maas_engine.utils.communication import seed_everything

    deepspeed.init_distributed('nccl')
    seed_everything(42)
    torch.cuda.set_device(DIST_ENV.rank)
    data_parallel_size, sequence_parallel_size = 4, 2
    hidden_size, num_attention_heads, num_key_value_heads = 4096, 64, 64
    head_dim = hidden_size // num_attention_heads
    bs, seqlen = 1, 64

    initialize_parallel_state(data_parallel_size=data_parallel_size, sequence_parallel_size=sequence_parallel_size)

    class conf:
        def __init__(self) -> None:
            pass

    config = conf()
    config.hidden_size = hidden_size
    config.num_attention_heads = num_attention_heads
    config.num_key_value_heads = num_key_value_heads
    config.sequence_parallel_size = sequence_parallel_size

    rank = DIST_ENV.rank

    query = torch.rand((bs, num_attention_heads, seqlen, head_dim)).cuda()
    key = torch.rand((bs, num_attention_heads, seqlen, head_dim)).cuda()
    value = torch.rand((bs, num_attention_heads, seqlen, head_dim)).cuda()

    def get_dist_core_attention_output(query, key, value, sdpa=False):
        config.use_sdpa = sdpa
        config.sequence_parallel_size = sequence_parallel_size
        dist_core_attn = Attention(config)
        sp_rank = torch.distributed.get_group_rank(DIST_ENV.distributed_sequence_parallel_group, DIST_ENV.rank)

        seq_parallel = seqlen // sequence_parallel_size

        output = dist_core_attn(
            query[:, :, sp_rank * seq_parallel : (sp_rank + 1) * seq_parallel, :],
            key[:, :, sp_rank * seq_parallel : (sp_rank + 1) * seq_parallel, :],
            value[:, :, sp_rank * seq_parallel : (sp_rank + 1) * seq_parallel, :],
        )

        output_list = [
            torch.zeros(bs, num_attention_heads, seq_parallel, head_dim).cuda() for _ in range(sequence_parallel_size)
        ]
        torch.distributed.all_gather(output_list, output, group=DIST_ENV.distributed_sequence_parallel_group)
        return torch.cat(output_list, dim=2)

    def get_core_attention_output(query, key, value, sdpa=False):
        config.use_sdpa = sdpa
        config.sequence_parallel_size = 1
        core_attn = Attention(config)
        return core_attn(query, key, value)

    seq_output_non_sdpa = get_dist_core_attention_output(query, key, value, sdpa=False)
    raw_output_non_sdpa = get_core_attention_output(query, key, value, sdpa=False)
    seq_output_sdpa = get_dist_core_attention_output(query, key, value, sdpa=True)
    raw_output_sdpa = get_core_attention_output(query, key, value, sdpa=True)

    assert torch.equal(seq_output_non_sdpa, raw_output_non_sdpa)
    assert torch.allclose(seq_output_sdpa, raw_output_sdpa, atol=1e-5)
    assert torch.allclose(seq_output_non_sdpa, raw_output_sdpa, atol=1e-5)
