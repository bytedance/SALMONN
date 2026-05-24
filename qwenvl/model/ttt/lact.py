import math
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from einops import rearrange

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except:
    print(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
    )
    flash_attn_func = None


@torch.compile()
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


@torch.compile()
def l2_norm(x: torch.Tensor):
    """
    x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)  # norm will upcast to float32
    return ret.type(x_type)


@torch.compile()
def zeropower_via_newtonschulz5(G):
    """
    This is an updated version of the zeropower_via_newtonschulz5 function in here:
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L26
    The code is modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49, which contains the original muon implementation.
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Args:
        G: [b, d, d']
    Returns:
        X: [b, d, d']
    FLOPS:  When d=d', Total FLOPS=30 * b * d^3
    """
    assert len(G.shape) == 3
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


# @torch.compile
def block_causal_lact_swiglu(
    w0: torch.Tensor,  # [b, dh, dk]
    w1: torch.Tensor,  # [b, dv, dh]
    w2: torch.Tensor,  # [b, dh, dk]
    q: torch.Tensor,  # [b, l, dk]
    k: torch.Tensor,  # [b, l, dk]
    v: torch.Tensor,  # [b, l, dv]
    lr0: torch.Tensor,  # [b, l, 1]
    lr1: torch.Tensor,  # [b, l, 1]
    lr2: torch.Tensor,  # [b, l, 1]
    momentum: torch.Tensor = None, # [b, s, 1] # none means no momentum
    use_muon: bool = True,
    chunk_size: int=2048,  # test-time training chunk size
) -> torch.Tensor:
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))
    
    About precision:
        w0, w1, w2 are mostly likely fp32. 
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.
    """
    
    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32

        ########## use previous fast weights to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        ########## compute the gradient and update the fast weights
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
        
        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), vi)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(
            vi, (hidden.transpose(1, 2) * lr1i).type_as(vi)
        )  # [b, d, d]
        # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :] 
        
            m_i = m_i.mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2
        
        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)
        
        w0 = w0 + dw0
        w1 = w1 + dw1
        w2 = w2 + dw2

        w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm


    s_index = e_index # update_length - mini_batch_size
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    # use the last w0 and w1 to get the final output
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l]
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)
    # -> [b, l, dv]
    states = {
        "w0": w0,
        "w1": w1,
        "w2": w2,
    }
    return output.transpose(1, 2), states


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


class CausalLaCTSwiGLUWithSlidingWindowAttn(torch.nn.Module):
    """
    Causal LaCT with SwiGLU fast weight function and sliding window attention. Suitable for ordered 1D sequence like language. 
    The sliding window attention is computed by flash_attn_func.
    The sliding window attention is mixed into the same layer as the LaCT with shared QKV, following the style of GAU(https://arxiv.org/abs/2202.10447).

    """

    def __init__(
        self,
        dim: int,
        head_dim: int, # ttt head dim
        attn_head_dim: int, # attn head dim could be different from the fast weight head dim
        lact_chunk_size: int = 2048,
        window_size: int = 2048, # recommended to >= lact_chunk_size
        inter_multi: float = 1,
        use_o_norm: bool = True,  # recommended to be True
        use_momentum: bool = True, # recommended to be True for long sequence
        use_muon: bool = True,  # if your seq len > head_dim * 2, recommended to be True
        base_lr: float = 1e-2,
        ttt_scale_before_sum: bool = True,
        mem_len: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.inter_multi = inter_multi
        self.lact_chunk_size = lact_chunk_size
        self.window_size = window_size # sliding window size
        self.use_o_norm = use_o_norm
        self.mem_len = mem_len

        self.dim = dim
        self.head_dim = head_dim
        self.attn_head_dim = attn_head_dim
        self.num_ttt_heads = dim // head_dim
        self.num_attn_heads = dim // attn_head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.lr_dim = 1   # single scalar learning rate for each head
        self.lr_proj = nn.Linear(dim, self.lr_dim * 3 * self.num_ttt_heads, bias=False)
        self.base_lr = base_lr
        self.base_lr_inv = inv_softplus(base_lr)

        # create initial fast weights
        d_in, d_out = self.head_dim, self.head_dim
        d_h = int(self.head_dim * self.inter_multi)

        self.w0 = nn.Parameter(torch.randn(self.num_ttt_heads, d_h, d_in) / math.sqrt(d_in))
        self.w1 = nn.Parameter(torch.randn(self.num_ttt_heads, d_out, d_h) / math.sqrt(d_h))
        self.w2 = nn.Parameter(
            torch.randn(self.num_ttt_heads, d_h, d_in) / math.sqrt(d_in)
        )

        self.use_muon = use_muon
        self.use_momentum = use_momentum

        if use_momentum:
            self.momentum_proj = nn.Sequential(
                    nn.Linear(dim, self.num_ttt_heads),
                    # sigmoid
                    nn.Sigmoid(),
                )
        
        self.use_o_norm = use_o_norm
        if self.use_o_norm:
            self.o_norm = nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)
        else:
            self.o_norm = nn.Identity()

        
        # add scaling and offset for the shared QK vectors
        self.qk_scale = nn.Parameter(torch.ones(dim, 2))
        self.qk_offset = nn.Parameter(torch.zeros(dim, 2))

        if ttt_scale_before_sum:
            self.ttt_scale_proj = nn.Linear(dim, self.num_ttt_heads)

    def _rescale_qk(self, q, k):
        """
        q: [b, s, d]
        k: [b, s, d]
        """
        _dtype = q.dtype
    
        qk_scale = self.qk_scale.view(1, 1, -1, 2)
        qk_offset = self.qk_offset.view(1, 1, -1, 2)
        q = q * qk_scale[:, :, :, 0] + qk_offset[:, :, :, 0]
        k = k * qk_scale[:, :, :, 1] + qk_offset[:, :, :, 1]
        # cast back to the original dtype
        q = q.to(_dtype)
        k = k.to(_dtype)
        return q, k
    
    def forward(self, x: torch.Tensor):
        """
        x: [b, l, d]
        """
        # [b, l, d]
        attn_qkv = self.to_qkv(x)
        ttt_qkv = F.silu(attn_qkv)

        ttt_q, ttt_k, ttt_v = rearrange(
            ttt_qkv,
            "b l (qkv h d) -> qkv (b h) l d",
            qkv=3,
            h=self.num_ttt_heads,
            d=self.head_dim,
        )
        ttt_q, ttt_k = l2_norm(ttt_q), l2_norm(ttt_k)

        # better to have float32 for lr.
        # For muon, I found that float16 is still very good.
        with torch.autocast(device_type="cuda", enabled=False):
            lr = self.lr_proj(x)  # [b, l, lr_dim]

        lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)

        # [b * num_heads, l, 1] for each lr
        lr0, lr1, lr2 = rearrange(
            lr, "b l (h lrs d) -> lrs (b h) l d", lrs=3, h=self.num_ttt_heads, d=self.lr_dim
        )

         # [nh, d, d] -> [b * nh, d, d]
        w0 = self.w0.repeat(x.shape[0], 1, 1)
        w1 = self.w1.repeat(x.shape[0], 1, 1)
        w2 = self.w2.repeat(x.shape[0], 1, 1)

        if self.use_momentum:
            momentum = self.momentum_proj(x)
            momentum = rearrange(momentum, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_ttt_heads)
        else:
            momentum = None

        # [b * num_heads, l, ttt_head_dim]
        ttt_output, _ = block_causal_lact_swiglu(
            w0, w1, w2, ttt_q, ttt_k, ttt_v, lr0, lr1, lr2, momentum, self.use_muon, chunk_size=self.lact_chunk_size
        )
        ttt_output = self.o_norm(ttt_output)
        ttt_scale_per_head = F.silu(self.ttt_scale_proj(x), inplace=False)
        ttt_scale_per_head = rearrange(ttt_scale_per_head, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_ttt_heads)
        ttt_output = ttt_output * ttt_scale_per_head
        ttt_output = rearrange(
            ttt_output, "(b h) l d -> b l (h d)", h=self.num_ttt_heads, b=x.shape[0]
        )

        #### Begin Sliding Window Attention ####
        # [b, l, d]
        attn_q, attn_k, attn_v = attn_qkv.chunk(3, dim=-1)
        attn_q, attn_k = self._rescale_qk(attn_q, attn_k)

        # -> [b, l, num_attn_heads, attn_head_dim]
        attn_q = rearrange(attn_q, '... (h d) -> ... h d', d=self.attn_head_dim)
        attn_k = rearrange(attn_k, '... (h d) -> ... h d', d=self.attn_head_dim)
        attn_v = rearrange(attn_v, '... (h d) -> ... h d', d=self.attn_head_dim)
        
        # here I skip potential qk_norm, and RoPE, please use it if you need.
        # [b, l, num_attn_heads, attn_head_dim]
        attn_output = flash_attn_func(
            attn_q, attn_k, attn_v,
            causal=True,
            window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
        )

        attn_output = rearrange(attn_output, '... h d -> ... (h d)')

        output = attn_output + ttt_output # [b, l, d]
        output = self.o_proj(output)

        return output


class CausalLaCTSwiGLU(torch.nn.Module):
    """
    Causal LaCT with SwiGLU fast weight function and sliding window attention. Suitable for ordered 1D sequence like language. 
    The sliding window attention is computed by flash_attn_func.
    The sliding window attention is mixed into the same layer as the LaCT with shared QKV, following the style of GAU(https://arxiv.org/abs/2202.10447).

    """

    def __init__(
        self,
        dim: int,
        head_dim: int, # ttt head dim
        lact_chunk_size: int = 2048,
        inter_multi: float = 1,
        use_o_norm: bool = True,  # recommended to be True
        use_momentum: bool = True, # recommended to be True for long sequence
        use_muon: bool = True,  # if your seq len > head_dim * 2, recommended to be True
        base_lr: float = 1e-2,
        ttt_scale_before_sum: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim
        self.inter_multi = inter_multi
        self.lact_chunk_size = lact_chunk_size
        self.use_o_norm = use_o_norm

        self.dim = dim
        self.head_dim = head_dim
        self.num_ttt_heads = dim // head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.lr_dim = 1   # single scalar learning rate for each head
        self.lr_proj = nn.Linear(dim, self.lr_dim * 3 * self.num_ttt_heads, bias=False)
        self.base_lr = base_lr
        self.base_lr_inv = inv_softplus(base_lr)

        # create initial fast weights
        d_in, d_out = self.head_dim, self.head_dim
        d_h = int(self.head_dim * self.inter_multi)

        self.w0 = nn.Parameter(torch.randn(self.num_ttt_heads, d_h, d_in) / math.sqrt(d_in))
        self.w1 = nn.Parameter(torch.randn(self.num_ttt_heads, d_out, d_h) / math.sqrt(d_h))
        self.w2 = nn.Parameter(
            torch.randn(self.num_ttt_heads, d_h, d_in) / math.sqrt(d_in)
        )

        self.use_muon = use_muon
        self.use_momentum = use_momentum

        if use_momentum:
            self.momentum_proj = nn.Sequential(
                    nn.Linear(dim, self.num_ttt_heads),
                    # sigmoid
                    nn.Sigmoid(),
                )
        
        self.use_o_norm = use_o_norm
        if self.use_o_norm:
            self.o_norm = nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)
        else:
            self.o_norm = nn.Identity()

        
        # add scaling and offset for the shared QK vectors
        self.qk_scale = nn.Parameter(torch.ones(dim, 2))
        self.qk_offset = nn.Parameter(torch.zeros(dim, 2))

        if ttt_scale_before_sum:
            self.ttt_scale_proj = nn.Linear(dim, self.num_ttt_heads)
    
    def forward(self, x: torch.Tensor, freqs_cis=None, state_track=None):
        """
        x: [b, l, d]
        """
        # [b, l, d]
        attn_qkv = self.to_qkv(x)
        ttt_qkv = F.silu(attn_qkv)

        ttt_q, ttt_k, ttt_v = rearrange(
            ttt_qkv,
            "b l (qkv h d) -> qkv (b h) l d",
            qkv=3,
            h=self.num_ttt_heads,
            d=self.head_dim,
        )
        ttt_q, ttt_k = l2_norm(ttt_q), l2_norm(ttt_k)

        # better to have float32 for lr.
        # For muon, I found that float16 is still very good.
        with torch.autocast(device_type="cuda", enabled=False):
            lr = self.lr_proj(x)  # [b, l, lr_dim]

        lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)

        # [b * num_heads, l, 1] for each lr
        lr0, lr1, lr2 = rearrange(
            lr, "b l (h lrs d) -> lrs (b h) l d", lrs=3, h=self.num_ttt_heads, d=self.lr_dim
        )

         # [nh, d, d] -> [b * nh, d, d]
        w0 = self.w0.repeat(x.shape[0], 1, 1)
        w1 = self.w1.repeat(x.shape[0], 1, 1)
        w2 = self.w2.repeat(x.shape[0], 1, 1)

        if self.use_momentum:
            momentum = self.momentum_proj(x)
            momentum = rearrange(momentum, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_ttt_heads)
        else:
            momentum = None

        # [b * num_heads, l, ttt_head_dim]
        if state_track is not None:
            w0, w1, w2 = state_track["w0"], state_track["w1"], state_track["w2"]
        ttt_output, states = block_causal_lact_swiglu(
            w0, w1, w2, ttt_q, ttt_k, ttt_v, lr0, lr1, lr2, momentum, self.use_muon, chunk_size=self.lact_chunk_size
        )
        ttt_output = self.o_norm(ttt_output)
        ttt_scale_per_head = F.silu(self.ttt_scale_proj(x), inplace=False)
        ttt_scale_per_head = rearrange(ttt_scale_per_head, 'b s (n_h d) -> (b n_h) s d', n_h=self.num_ttt_heads)
        ttt_output = ttt_output * ttt_scale_per_head
        ttt_output = rearrange(
            ttt_output, "(b h) l d -> b l (h d)", h=self.num_ttt_heads, b=x.shape[0]
        )
        output = self.o_proj(ttt_output)

        return output, states


def _test_layer_with_random_input():
    B, L, D, HeadDim = 1, 2050, 2048, 512
    AttnHeadDim = 128
    LactChunkSize = 2048
    WindowSize = 2048
    
    layer = CausalLaCTSwiGLU(dim=D, head_dim=HeadDim, lact_chunk_size=LactChunkSize, inter_multi=1, use_o_norm=True, use_momentum=True, use_muon=True)
    layer = layer.to("cuda")
    x = torch.randn(B, L, D).to("cuda")

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output = layer(x)
    print(output.shape)

    print(output.shape, output.dtype)

    print("Input norm", x.norm(), "Output norm", output.norm())


if __name__ == "__main__":
    _test_layer_with_random_input()
