import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor

from qwenvl.model.ttt.cogvideo_utils import (SequenceMetadata, full_tensor,
                                       place_into, shard_tensor, to_local)
from qwenvl.model.ttt.configs import ModelConfig
from qwenvl.model.ttt.linear_triton import TritonLinear
from qwenvl.model.ttt.mlp_tk import TkMLP
from qwenvl.model.ttt.ops import ttt_linear, ttt_mlp, ttt_mlp_cg, ttt_mlp_cg_w_loss
from qwenvl.model.ttt.utils import apply_rotary_emb, precompute_freqs_cis_3d, apply_rotary_emb_single
from qwenvl.model.ttt.ops.cg_utils import mlp_fwd_no_backward


class SSMGating(nn.Module):
    def __init__(
        self,
        model_dim,
        gating_alpha_init: float = 0.1,
    ):
        super().__init__()

        self.gating_alpha = nn.Parameter(torch.ones(model_dim) * gating_alpha_init)

    def forward(self, x):
        gating_alpha = full_tensor(self.gating_alpha)

        gating_alpha = torch.tanh(gating_alpha)
        return gating_alpha * x


class TTTWrapper(nn.Module):
    def __init__(self, config: ModelConfig, CG_max_iter=0, discard_V=False):
        super().__init__()

        self.model_dim = config.model_dim
        self.num_heads = config.num_heads
        self.rope_theta = config.rope_theta
        self.latent_height = config.latent_height
        self.latent_width = config.latent_width
        self.compressed_num_frames = config.compressed_num_frames
        self.ttt_hidden_size = config.ttt_hidden_size

        if config.ssm_layer == "ttt_linear":
            self.ttt = TTTLinear(config)
        elif config.ssm_layer == "ttt_mlp":
            self.ttt = TTTMLP(config)
        elif config.ssm_layer == "ttt_mlp_cg":
            print("Use TTT MLP CG")
            self.ttt = TTTMLPCG(config, CG_max_iterations=CG_max_iter, ttt_hidden_size=self.ttt_hidden_size)
        elif config.ssm_layer == "ttt_mlp_ntp_cg":
            print("Use TTT MLP NTP CG")
            self.ttt = TTTMLPNTPCG(config, CG_max_iterations=CG_max_iter, discard_V=discard_V)
        else:
            raise TypeError(f"No ttt layer of type {config.ssm_layer}")

        # self.register_buffer("freqs_cis", self._precompute_freqs_cis_3d(), persistent=False)

    def _precompute_freqs_cis_3d(self, height, width, compressed_num_frames) -> torch.Tensor:
        return precompute_freqs_cis_3d(
            self.model_dim // self.num_heads,
            height,
            width,
            compressed_num_frames,
            self.rope_theta,
        )

    def init_freqs(self):
        self.freqs_cis.copy_(self._precompute_freqs_cis_3d(self.latent_height, self.latent_width, self.compressed_num_frames))

    def forward(self, x: torch.Tensor, freqs_cis=None, downsample_ids=None, text_query=None, state_track=None):
        seq_metadata = SequenceMetadata(text_length=0, seq_text_length=0, num_frames=0, num_chunks=1, tokens_per_frame=1)
        hidden_states, state_track = self.ttt(x, freqs_cis, seq_metadata, downsample_ids, text_query, state_track=state_track)
        return hidden_states, state_track


class TTTBase(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.width = config.model_dim
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.mini_batch_size = config.mini_batch_size
        self.slot_type = config.slot_type
        self.head_gating = "gating" in config.slot_type
        self.forget_gating = "forget" in config.slot_type
        self.predict_ema = "ema" in config.slot_type
        print("Output gating: ", self.head_gating)
        print("Forget gating: ", self.forget_gating)
        print("EMA target: ", self.predict_ema)
        self.ema_factor = config.ema_factor
        self.lag_distances = [int(k) for k in config.lag_distances.split(",")]
        print("LAG distances: ", self.lag_distances)
        if "forward" in self.slot_type:
            print("LAG loss forward direction")
        self.predict_lag = True if self.lag_distances != [0] else False

        self.ttt_base_lr = config.ttt_base_lr
        self.scan_checkpoint_group_size = config.scan_checkpoint_group_size

        self.tp_mesh: None | DeviceMesh = None

        self._init_qkvo_proj()
        self._init_ttt_lr_gate()
        self._init_ttt_ln()

        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

    # We must reinitialize after meta initialization
    def init_weights(self):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wo.weight, mean=0.0, std=0.02)

        self.post_norm.reset_parameters()
        nn.init.ones_(self.ttt_norm_weight.data)
        nn.init.zeros_(self.ttt_norm_bias)
        nn.init.normal_(self.learnable_ttt_lr_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.learnable_ttt_lr_bias)

    def _init_qkvo_proj(self):
        self.wq = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
        self.wv = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
        self.wo = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
        if self.predict_ema:
            self.wv_ema = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
        if self.head_gating:
            self.gating_linear = nn.Linear(self.config.model_dim, self.config.model_dim)
        if self.predict_lag:
            self.lag_linear = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
            if 'usekey' in self.slot_type:
                self.lag_linear_key = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
            if "dual" in self.slot_type:
                self.lag_linear_dual = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)

    def _init_ttt_lr_gate(self):
        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )

        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim=0,
            )
        )

    def _init_ttt_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

    def init_device_mesh(self, tp_mesh: DeviceMesh):
        self.tp_mesh = tp_mesh

        self.ttt_norm_weight = nn.Parameter(distribute_tensor(self.ttt_norm_weight, tp_mesh, [Shard(0)]))
        self.ttt_norm_bias = nn.Parameter(distribute_tensor(self.ttt_norm_bias, tp_mesh, [Shard(0)]))

        self.learnable_ttt_lr_weight = nn.Parameter(
            distribute_tensor(self.learnable_ttt_lr_weight, tp_mesh, [Replicate()])
        )
        self.learnable_ttt_lr_bias = nn.Parameter(distribute_tensor(self.learnable_ttt_lr_bias, tp_mesh, [Replicate()]))

    def shard_inputs(self, inputs):
        assert self.tp_mesh is not None, "Tensor parallel mesh must be initialized before sharding inputs."

        for key in inputs:
            assert inputs[key].shape[1] == self.num_heads, "Sharding is only supported on the head dimension."
            inputs[key] = shard_tensor(inputs[key], self.tp_mesh, dim=1)

        return inputs

    @torch.compile
    def get_qkv_projections(self, hidden_states):
        XQ, XK, XV = (
            self.wq(hidden_states),
            self.wk(hidden_states),
            self.wv(hidden_states),
        )
        return XQ, XK, XV

    @torch.compile
    def get_eta(self, X):
        learnable_ttt_lr_weight = full_tensor(self.learnable_ttt_lr_weight)
        learnable_ttt_lr_bias = full_tensor(self.learnable_ttt_lr_bias)

        ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, learnable_ttt_lr_weight) + learnable_ttt_lr_bias.reshape(
            1, -1, 1, 1, 1
        )  # [B,nc,cs,c] @ [nh,1,c] -> [B,nh,nc,cs,1] + [1,nh,1,1,1] -> [B,nh,nc,cs,1]

        if "slot" in self.slot_type:
            ttt_lr = F.softmax(ttt_lr, dim=1) * ttt_lr.size(1)
        else:
            ttt_lr = F.sigmoid(ttt_lr)  # [B,H,nc,K,1]

        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        return self.ttt_base_lr * ttt_lr / self.head_dim

    @torch.compile
    def interleave(self, x: torch.Tensor, seq_metadata: SequenceMetadata):
        init_offset, num_chunks, text_length = (
            seq_metadata.init_offset,
            seq_metadata.num_chunks,
            seq_metadata.text_length,
        )
        assert init_offset is not None, "Init offset must be provided for interleaving."

        seq_text_length = text_length * num_chunks

        B, H, NC, C, HD = x.shape
        x_flatten = x.reshape(B, H, NC * C, HD)

        x_text = x_flatten[:, :, :seq_text_length]
        x_video = x_flatten[:, :, seq_text_length:]

        # Get individual scene text embeddings.
        x_text = torch.chunk(x_text, num_chunks, dim=2)

        # The first scene will have one extra latent frame.
        video_init_offset = init_offset - text_length
        partial_chunks = torch.chunk(x_video[:, :, video_init_offset:], num_chunks - 1, dim=2)
        x_video = (x_video[:, :, :video_init_offset],) + partial_chunks

        x_interleaved = []
        for i in range(num_chunks):
            x_interleaved.append(torch.cat((x_text[i], x_video[i]), dim=2))

        return torch.cat(x_interleaved, dim=2).reshape(B, H, NC, C, HD)

    @torch.compile
    def undo_interleave(self, x: torch.Tensor, seq_metadata: SequenceMetadata):
        text_length, init_offset, base_offset, num_chunks = (
            seq_metadata.text_length,
            seq_metadata.init_offset,
            seq_metadata.base_offset,
            seq_metadata.num_chunks,
        )

        assert base_offset is not None, "Base offset must be provided for undoing interleaving."
        assert init_offset is not None, "Init offset must be provided for undoing interleaving."

        text_embs, vid_embs = torch.tensor([], dtype=x.dtype, device=x.device), torch.tensor(
            [], dtype=x.dtype, device=x.device
        )

        for i in range(num_chunks):
            if i == 0:
                scene_start_idx = 0
                scene_end_idx = init_offset
            else:
                scene_start_idx = init_offset + (i - 1) * base_offset
                scene_end_idx = init_offset + i * base_offset

            scene_emb = x[:, scene_start_idx:scene_end_idx]

            text_embs = torch.cat((text_embs, scene_emb[:, :text_length]), dim=1)
            vid_embs = torch.cat((vid_embs, scene_emb[:, text_length:]), dim=1)

        return torch.cat((text_embs, vid_embs), dim=1)

    @torch.compile
    def ln_reconstruction_target(self, XV, XK):
        XV = XV - XK
        eps = 1e-8
        # Compute mean and std over the head dimension (last dimension)
        mean = XV.mean(dim=-1, keepdim=True)
        std = place_into(to_local(XV).std(dim=-1, keepdim=True), XV)

        # Normalize
        XV = (XV - mean) / (std + eps)

        # Apply per-head weight and bias.
        # self.ttt_norm_weight and self.ttt_norm_bias have shape [num_heads, head_dim].
        # We unsqueeze to make them broadcastable with XV_norm which is [B, L, num_heads, head_dim].
        XV = self.ttt_norm_weight.unsqueeze(0).unsqueeze(0) * XV + self.ttt_norm_bias.unsqueeze(0).unsqueeze(0)

        return XV + XK

    @torch.compile
    def reshape_to_mini_batch(self, X, XQ, XK, XV):
        B, L = X.shape[:2]
        num_mini_batch = L // self.mini_batch_size

        XQ, XK, XV = XQ.transpose(1, 2), XK.transpose(1, 2), XV.transpose(1, 2)

        X = X.reshape(B, num_mini_batch, self.mini_batch_size, self.width)

        XQ = XQ.reshape(B, self.num_heads, num_mini_batch, self.mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, num_mini_batch, self.mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, num_mini_batch, self.mini_batch_size, self.head_dim)

        return X, XQ, XK, XV

    def process_input(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, seq_metadata: SequenceMetadata):
        seq_text_length = seq_metadata.seq_text_length

        B, L = hidden_states.shape[:2]
        mini_batch_size = self.mini_batch_size

        XQ, XK, XV = self.get_qkv_projections(hidden_states)

        XQ = XQ.view(B, L, -1, self.head_dim)
        XK = XK.view(B, L, -1, self.head_dim)
        XV = XV.view(B, L, -1, self.head_dim)

        # L2 Norm
        XQ = place_into(torch.nn.functional.normalize(to_local(XQ), p=2, dim=-1), XQ)
        XK = place_into(torch.nn.functional.normalize(to_local(XK), p=2, dim=-1), XK)

        XQ_text, XQ_video = XQ[:, :seq_text_length], XQ[:, seq_text_length:]
        XK_text, XK_video = XK[:, :seq_text_length], XK[:, seq_text_length:]

        if freqs_cis is not None:
            XQ_rope_video, XK_rope_video = apply_rotary_emb(
                to_local(XQ_video), to_local(XK_video), freqs_cis=to_local(freqs_cis)
            )

            XQ_video = place_into(XQ_rope_video, XQ_video)
            XK_video = place_into(XK_rope_video, XK_video)

        XQ = torch.cat((XQ_text, XQ_video), dim=1)
        XK = torch.cat((XK_text, XK_video), dim=1)

        XV = self.ln_reconstruction_target(XV, XK)

        hidden_states, XQ, XK, XV = self.reshape_to_mini_batch(hidden_states, XQ, XK, XV)

        ttt_lr_eta = self.get_eta(hidden_states)

        # We do not use token_eta for non-causal chunks
        eta = 1 / mini_batch_size * ttt_lr_eta.repeat(1, 1, 1, mini_batch_size, 1)

        if seq_metadata.is_multiscene:
            XQ = place_into(self.interleave(to_local(XQ), seq_metadata), XQ)
            XK = place_into(self.interleave(to_local(XK), seq_metadata), XK)
            XV = place_into(self.interleave(to_local(XV), seq_metadata), XV)
            eta = self.interleave(to_local(eta), seq_metadata)

        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
        }

        if self.tp_mesh is not None:
            inputs = self.shard_inputs(inputs)

        return inputs

    def ttt(
        self,
        inputs,
    ):
        raise NotImplementedError("ttt method must be implemented in TTTBase subclasses.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        seq_metadata: SequenceMetadata,
        downsample_ids = None,
        text_query = None,
        state_track = None,
    ):
        assert (
            hidden_states.size(1) % self.config.mini_batch_size == 0
        ), "Sequence len must be multiple of mini batch size."

        if self.head_gating:
            hidden_gating = self.gating_linear(hidden_states)
        prev_hidden_states = None
        if state_track is not None and "prev_hidden_states" in state_track:
            prev_hidden_states = state_track["prev_hidden_states"]
        tmp_hidden_states = hidden_states
        hidden_states = self.process_input(hidden_states, freqs_cis, seq_metadata, prev_hidden_states=prev_hidden_states)
        hidden_states, state_track = self.ttt(hidden_states, downsample_ids, text_query=text_query, allstates=state_track)
        state_track["prev_hidden_states"] = tmp_hidden_states
        if self.head_gating:
            hidden_states = hidden_states * torch.sigmoid(hidden_gating)
        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.wo(hidden_states)

        hidden_states = full_tensor(hidden_states)

        return hidden_states, state_track


class TTTLinear(TTTBase):
    def __init__(self, config: ModelConfig, use_kernel: bool = True):
        super().__init__(config)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

        # For acceleration
        self.use_kernel = use_kernel

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.W1, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1)

    def init_device_mesh(self, tp_mesh: DeviceMesh):
        assert self.use_kernel, "Tensor parallel is not currently supported for TTTLinear without kernel."
        super().init_device_mesh(tp_mesh)

        self.W1 = nn.Parameter(distribute_tensor(self.W1, tp_mesh, [Shard(0)]))
        self.b1 = nn.Parameter(distribute_tensor(self.b1, tp_mesh, [Shard(0)]))

        TritonLinear.sharded_mode = True

    def ttt(self, inputs):
        B = inputs["XV"].shape[0]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        num_mini_batch = inputs["XV"].shape[2]

        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))

        checkpoint_group_size = min(max(self.config.scan_checkpoint_group_size, 1), num_mini_batch)

        if self.use_kernel:
            XQW_batch = TritonLinear.apply(
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states,
                b1_states,
                inputs["XQ"],
                inputs["XV"],
                inputs["XK"],
                inputs["eta"],
                checkpoint_group_size,
            )

            XQW_batch = XQW_batch.permute(0, 2, 3, 1, 4)
        else:
            XQW_batch = ttt_linear(
                inputs["XK"],
                inputs["XQ"],
                inputs["XV"],
                inputs["eta"],
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states,
                b1_states,
                checkpoint_group_size,
            )

        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch


class TTTMLP(TTTBase):
    def __init__(self, config: ModelConfig, use_kernel: bool = False):
        super().__init__(config)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

        self.use_kernel = use_kernel

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.W1, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1)
        nn.init.normal_(self.W2, mean=0.0, std=0.02)
        nn.init.zeros_(self.b2)

    def init_device_mesh(self, tp_mesh: DeviceMesh):
        assert self.use_kernel, "Tensor parallel is not currently supported for TTTMLP without kernel."
        super().init_device_mesh(tp_mesh)

        self.W1 = nn.Parameter(distribute_tensor(self.W1, tp_mesh, [Shard(0)]))
        self.b1 = nn.Parameter(distribute_tensor(self.b1, tp_mesh, [Shard(0)]))
        self.W2 = nn.Parameter(distribute_tensor(self.W2, tp_mesh, [Shard(0)]))
        self.b2 = nn.Parameter(distribute_tensor(self.b2, tp_mesh, [Shard(0)]))

        TkMLP.sharded_mode = True

    def ttt(self, inputs, downsample_ids=None):
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]

        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))
        W2_states = torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1))
        b2_states = torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1))

        checkpoint_group_size = min(max(self.config.scan_checkpoint_group_size, 1), num_mini_batch)

        if self.use_kernel:
            XQW_batch = TkMLP.apply(
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states,
                b1_states,
                W2_states,
                b2_states,
                inputs["XQ"],
                inputs["XV"],
                inputs["XK"],
                inputs["eta"],
                checkpoint_group_size,
            )

            XQW_batch = XQW_batch.permute(0, 2, 3, 1, 4)
        else:
            XQW_batch = ttt_mlp(
                inputs["XK"],
                inputs["XQ"],
                inputs["XV"],
                inputs["eta"],
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states,
                b1_states,
                W2_states,
                b2_states,
                checkpoint_group_size,
            )

        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch

class TTTMLPCG(TTTBase):
    def __init__(self, config: ModelConfig, use_kernel: bool = False, CG_max_iterations: int = 4, ttt_hidden_size: int = 4):
        super().__init__(config)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, ttt_hidden_size * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, ttt_hidden_size * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, ttt_hidden_size * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        if self.forget_gating:
            self.forget_gate = nn.Parameter(torch.normal(0, 0.1, size=(2 * self.head_dim, 1)))

        self.use_kernel = use_kernel
        assert use_kernel is False, "CG is not supported with kernel."
        self.CG_max_iter = CG_max_iterations
        self.downsample_option = config.downsample_option
        self.downsample_freq = config.downsample_freq

        if self.downsample_option in ("rope_simple", "rope_post"):
            self.rope_soft_prompt = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim)))
        if self.downsample_option == "fastweight_direct":
            self.w2summary = nn.Linear(2 * self.num_heads * self.head_dim, self.width, bias=True)
            nn.init.normal_(self.w2summary.weight, mean=0.0, std=0.02)

    # def _init_qkvo_proj(self):
    #     self.wq = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.W1, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1)
        nn.init.normal_(self.W2, mean=0.0, std=0.02)
        nn.init.zeros_(self.b2)
        if self.downsample_option == "fastweight_direct":
            nn.init.normal_(self.w2summary.weight, mean=0.0, std=0.02)

    def init_device_mesh(self, tp_mesh: DeviceMesh):
        assert self.use_kernel, "Tensor parallel is not currently supported for TTTMLP without kernel."
        super().init_device_mesh(tp_mesh)

        self.W1 = nn.Parameter(distribute_tensor(self.W1, tp_mesh, [Shard(0)]))
        self.b1 = nn.Parameter(distribute_tensor(self.b1, tp_mesh, [Shard(0)]))
        self.W2 = nn.Parameter(distribute_tensor(self.W2, tp_mesh, [Shard(0)]))
        self.b2 = nn.Parameter(distribute_tensor(self.b2, tp_mesh, [Shard(0)]))

        TkMLP.sharded_mode = True

    def process_input(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        seq_metadata: SequenceMetadata,
        prev_hidden_states: torch.Tensor = None,
    ):
        seq_text_length = seq_metadata.seq_text_length

        B, L, D = hidden_states.shape[:3]
        mini_batch_size = self.mini_batch_size

        XQ, XK, XV = self.get_qkv_projections(hidden_states)

        XQ = XQ.view(B, L, -1, self.head_dim)
        XK = XK.view(B, L, -1, self.head_dim)
        XV = XV.view(B, L, -1, self.head_dim)
        XV_ema = None
        if self.predict_ema:
            with torch.no_grad():
                t = torch.arange(L, device=hidden_states.device, dtype=hidden_states.dtype)
                ema_beta = 0.9995
                beta_pows = (ema_beta ** t).view(1, L, 1)
                beta_inv_pows = (ema_beta ** t).view(1, L, 1)
                scale = (1 - ema_beta) * beta_inv_pows
                XV_ema = hidden_states * scale
                prefix = torch.cumsum(XV_ema[:, 1:], dim=1)
                z = torch.zeros_like(hidden_states)
                z[:, 0] = hidden_states[:, 0]
                z[:, 1:] = hidden_states[:, [0]] + prefix
                XV_ema = beta_pows * z
                XV_ema = self.wv_ema(XV_ema)
                XV_ema = XV_ema.view(B, L, -1, self.head_dim)

        XV_lag, XK_lag, XV_lag_dual = None, None, None
        lag_loss_mask, lag_loss_mask_dual = None, None
        if self.predict_lag:
            lag_distance = self.lag_distances[0]  # hard code for now
            if "forward" in self.slot_type:
                # Forward means using X_{t-T} to predict X_t
                clip_lag_distance = min(lag_distance, L)
                if prev_hidden_states is not None and "carryover" in self.slot_type:
                    residual = prev_hidden_states.size(1) - (lag_distance - clip_lag_distance)
                    new_hidden_states = torch.cat([prev_hidden_states[:, -lag_distance:residual], hidden_states[:, :-clip_lag_distance]], dim=1)
                    lag_loss_mask = hidden_states.new_ones(L)
                else:
                    new_hidden_states = torch.cat([hidden_states.new_zeros(B, clip_lag_distance, D), hidden_states[:, :-clip_lag_distance]], dim=1)
                    lag_loss_mask = torch.cat([hidden_states.new_zeros(clip_lag_distance), hidden_states.new_ones(L-clip_lag_distance)], dim=0)
            else:
                new_hidden_states = torch.cat([hidden_states[:, lag_distance:], hidden_states.new_zeros(B, lag_distance, D)], dim=1)
                lag_loss_mask = torch.cat([hidden_states.new_ones(L-lag_distance), hidden_states.new_zeros(lag_distance)], dim=0)
            XV_lag = self.lag_linear(new_hidden_states)
            if XV_lag.size(1) != L:
                XV_lag = XV_lag.new_zeros(B, L, XV_lag.size(2))
                lag_loss_mask = XV_lag.new_zeros(L)
            XV_lag = XV_lag.view(B, L, -1, self.head_dim)
            if "usekey" in self.slot_type:
                XK_lag = self.lag_linear_key(hidden_states)
                XK_lag = XK_lag.view(B, L, -1, self.head_dim)
            if "dual" in self.slot_type:
                dual_lag_distance = self.lag_distances[1]
                dual_clip_lag_distance = min(dual_lag_distance, L)
                if prev_hidden_states is not None and "carryover" in self.slot_type:
                    residual = prev_hidden_states.size(1) - (dual_lag_distance - dual_clip_lag_distance)
                    new_hidden_states = torch.cat([prev_hidden_states[:, -dual_lag_distance:residual], hidden_states[:, :-dual_clip_lag_distance]], dim=1)
                    lag_loss_mask_dual = hidden_states.new_ones(L)
                else:
                    new_hidden_states = torch.cat([hidden_states.new_zeros(B, dual_clip_lag_distance, D), hidden_states[:, :-dual_clip_lag_distance]], dim=1)
                    lag_loss_mask_dual = torch.cat([hidden_states.new_zeros(dual_clip_lag_distance), hidden_states.new_ones(L-dual_clip_lag_distance)], dim=0)
                XV_lag_dual = self.lag_linear_dual(new_hidden_states)
                if XV_lag_dual.size(1) != L:
                    XV_lag_dual = XV_lag_dual.new_zeros(B, L, XV_lag_dual.size(2))
                    lag_loss_mask_dual = XV_lag_dual.new_zeros(L)
                XV_lag_dual = XV_lag_dual.view(B, L, -1, self.head_dim)

        forget_factor = None
        if self.forget_gating:
            XK_mean = XK.transpose(1, 2).view(B, self.num_heads, -1, mini_batch_size, self.head_dim).mean(dim=3)
            XK_diff = torch.cat([XK_mean[:, :, 0:1], XK_mean[:, :, 1:] - XK_mean[:, :, :-1]], dim=2)  # B H N D
            XK_diff = torch.cat([XK_mean, XK_diff], dim=-1)  # B H N 2D
            forget_factor = torch.sigmoid(torch.einsum("bhnd,di->bhni", XK_diff, self.forget_gate)) / 4 + 0.75  # B H N 1

        # L2 Norm
        XQ = place_into(torch.nn.functional.normalize(to_local(XQ), p=2, dim=-1), XQ)
        XK = place_into(torch.nn.functional.normalize(to_local(XK), p=2, dim=-1), XK)

        XQ_text, XQ_video = XQ[:, :seq_text_length], XQ[:, seq_text_length:]
        XK_text, XK_video = XK[:, :seq_text_length], XK[:, seq_text_length:]

        if freqs_cis is not None:
            XQ_rope_video, XK_rope_video = apply_rotary_emb(
                to_local(XQ_video), to_local(XK_video), freqs_cis=to_local(freqs_cis)
            )

            XQ_video = place_into(XQ_rope_video, XQ_video)
            XK_video = place_into(XK_rope_video, XK_video)

        XQ = torch.cat((XQ_text, XQ_video), dim=1)
        XK = torch.cat((XK_text, XK_video), dim=1)
        if "usekey" in self.slot_type:
            XK_lag = place_into(torch.nn.functional.normalize(to_local(XK_lag), p=2, dim=-1), XK_lag)
            _, XK_lag = apply_rotary_emb(
                to_local(XQ), to_local(XK_lag), freqs_cis=to_local(freqs_cis)
            )

        XV = self.ln_reconstruction_target(XV, XK)
        # EMA loss
        if self.predict_ema:
            XV_ema = self.ln_reconstruction_target(XV_ema, XK)
        # Lag loss
        if self.predict_lag:
            if "usekey" in self.slot_type:
                XV_lag = self.ln_reconstruction_target(XV_lag, XK_lag)
                if "dual" in self.slot_type:
                    XV_lag_dual = self.ln_reconstruction_target(XV_lag_dual, XK_lag)
            else:
                XV_lag = self.ln_reconstruction_target(XV_lag, XK)

        hidden_states, XQ, XK, XV = self.reshape_to_mini_batch(hidden_states, XQ, XK, XV)
        if self.predict_ema:
            XV_ema = XV_ema.transpose(1, 2).reshape(B, self.num_heads, -1, mini_batch_size, self.head_dim)
        if self.predict_lag:
            XV_lag = XV_lag.transpose(1, 2).reshape(B, self.num_heads, -1, mini_batch_size, self.head_dim)
            if "usekey" in self.slot_type:
                XK_lag = XK_lag.transpose(1, 2).reshape(B, self.num_heads, -1, mini_batch_size, self.head_dim)
            lag_loss_mask = lag_loss_mask.reshape(-1, mini_batch_size).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            if "dual" in self.slot_type:
                XV_lag_dual = XV_lag_dual.transpose(1, 2).reshape(B, self.num_heads, -1, mini_batch_size, self.head_dim) # (B, H, N, bsize, 1)
                lag_loss_mask_dual = lag_loss_mask_dual.reshape(-1, mini_batch_size).unsqueeze(0).unsqueeze(0).unsqueeze(-1) # (B, H, N, bsize, 1)

        ttt_lr_eta = self.get_eta(hidden_states)

        # We do not use token_eta for non-causal chunks
        eta = 1 / mini_batch_size * ttt_lr_eta.repeat(1, 1, 1, mini_batch_size, 1)

        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
            "XV_ema": XV_ema,
            "XV_lag": XV_lag,
            "XV_lag_dual": XV_lag_dual,
            "XK_lag": XK_lag,
            "forget_factor": forget_factor,
            "lag_loss_mask": lag_loss_mask,
            "lag_loss_mask_dual": lag_loss_mask_dual,
        }

        if self.downsample_option in ["rope_post"]:
            # use pure rope query, output during TTT scan or after TTT scan
            extended_rope_soft_prompt = torch.tile(self.rope_soft_prompt.unsqueeze(0).unsqueeze(0), (B, L, 1, 1))
            extended_rope_soft_prompt = apply_rotary_emb_single(to_local(extended_rope_soft_prompt), freqs_cis=to_local(freqs_cis))
            inputs["rope_soft"] = extended_rope_soft_prompt

        if self.tp_mesh is not None:
            inputs = self.shard_inputs(inputs)

        return inputs

    def ttt(self, inputs, downsample_ids=None, text_query=None, allstates=None):
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]

        if allstates is not None:
            W1_states = allstates["W1_states"]
            b1_states = allstates["b1_states"]
            W2_states = allstates["W2_states"]
            b2_states = allstates["b2_states"]
        else:
            W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
            b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))
            W2_states = torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1))
            b2_states = torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1))

        checkpoint_group_size = min(max(self.config.scan_checkpoint_group_size, 1), num_mini_batch)

        if self.downsample_option == "none":
            states, XQW_batch = ttt_mlp_cg(
                inputs["XK"],
                inputs["XQ"],
                inputs["XV"],
                inputs["eta"],
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states if allstates is None else allstates["W1_states"],
                b1_states if allstates is None else allstates["b1_states"],
                W2_states if allstates is None else allstates["W2_states"],
                b2_states if allstates is None else allstates["b2_states"],
                checkpoint_group_size,
                cg_max_iter=self.CG_max_iter,
                XV_ema=inputs["XV_ema"],
                XV_lag=inputs["XV_lag"],
                XV_lag_dual=inputs["XV_lag_dual"],
                XK_lag=inputs["XK_lag"],
                ema_factor=self.ema_factor,
                forget_factor=inputs["forget_factor"],
                lag_loss_mask=inputs["lag_loss_mask"],
                lag_loss_mask_dual=inputs["lag_loss_mask_dual"],
            )
        elif self.downsample_option == "loss":
            states, XQW_batch = ttt_mlp_cg_w_loss(
                inputs["XK"],
                inputs["XQ"],
                inputs["XV"],
                inputs["eta"],
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states if allstates is None else allstates["W1_states"],
                b1_states if allstates is None else allstates["b1_states"],
                W2_states if allstates is None else allstates["W2_states"],
                b2_states if allstates is None else allstates["b2_states"],
                checkpoint_group_size,
                cg_max_iter=self.CG_max_iter,
                XV_ema=inputs["XV_ema"],
                XV_lag=inputs["XV_lag"],
                XV_lag_dual=inputs["XV_lag_dual"],
                XK_lag=inputs["XK_lag"],
                ema_factor=self.ema_factor,
                forget_factor=inputs["forget_factor"],
                lag_loss_mask=inputs["lag_loss_mask"],
                lag_loss_mask_dual=inputs["lag_loss_mask_dual"],
            )

        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, states