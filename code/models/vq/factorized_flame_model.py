import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vq.model import RVQVAE


class FactorizedFlameRVQVAE(nn.Module):
    """Two-branch FLAME tokenizer with a stable six-token interface.

    The first four codebooks encode expression components orthogonal to the
    mouth subspace together with the eye pose.  The last two codebooks encode
    the mouth-expression subspace and jaw-x.  Global/head rotation (100:103)
    is deliberately excluded and decodes to zero in normalized coordinates.
    """

    def __init__(
        self,
        args,
        mouth_basis,
        mouth_vertex_matrix=None,
        nb_code=2048,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=6,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
        nonmouth_num_quantizers=4,
        mouth_num_quantizers=2,
    ):
        super().__init__()
        mouth_basis = torch.as_tensor(mouth_basis, dtype=torch.float32)
        if mouth_basis.ndim != 2 or mouth_basis.shape[0] != 100:
            raise ValueError(f"mouth_basis must have shape (100, rank), got {tuple(mouth_basis.shape)}")
        gram = mouth_basis.T @ mouth_basis
        if not torch.allclose(gram, torch.eye(gram.shape[0]), atol=1e-4, rtol=1e-4):
            raise ValueError("mouth_basis columns must be orthonormal")

        self.register_buffer("mouth_basis", mouth_basis)
        if mouth_vertex_matrix is None:
            mouth_vertex_matrix = torch.empty(100, 0, dtype=torch.float32)
        mouth_vertex_matrix = torch.as_tensor(mouth_vertex_matrix, dtype=torch.float32)
        if mouth_vertex_matrix.ndim != 2 or mouth_vertex_matrix.shape[0] != 100:
            raise ValueError(
                "mouth_vertex_matrix must map normalized expression coefficients and have shape (100, D)"
            )
        self.register_buffer("mouth_vertex_matrix", mouth_vertex_matrix)

        self.mouth_rank = int(mouth_basis.shape[1])
        self.nonmouth_num_quantizers = int(nonmouth_num_quantizers)
        self.mouth_num_quantizers = int(mouth_num_quantizers)
        self.num_quantizers = self.nonmouth_num_quantizers + self.mouth_num_quantizers
        self.code_dim = int(code_dim)
        self.num_code = int(nb_code)

        nonmouth_args = copy.copy(args)
        nonmouth_args.num_quantizers = self.nonmouth_num_quantizers
        mouth_args = copy.copy(args)
        mouth_args.num_quantizers = self.mouth_num_quantizers

        # 100-D non-mouth expression residual + four selected eye-pose dims.
        self.nonmouth_vq = RVQVAE(
            nonmouth_args,
            104,
            nb_code,
            code_dim,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation,
            norm,
        )
        # Low-rank mouth-expression coordinates + jaw-x.
        self.mouth_vq = RVQVAE(
            mouth_args,
            self.mouth_rank + 1,
            nb_code,
            code_dim,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation,
            norm,
        )

    def set_context_frames(self, frames):
        self.nonmouth_vq.set_context_frames(frames)
        self.mouth_vq.set_context_frames(frames)
        return self

    def split_features(self, motion):
        if motion.ndim != 3 or motion.shape[-1] != 108:
            raise ValueError(f"expected normalized FLAME motion (B,T,108), got {tuple(motion.shape)}")
        expression = motion[..., :100]
        mouth_expression = expression @ self.mouth_basis
        nonmouth_expression = expression - mouth_expression @ self.mouth_basis.T
        nonmouth = torch.cat([nonmouth_expression, motion[..., 104:108]], dim=-1)
        mouth = torch.cat([mouth_expression, motion[..., 103:104]], dim=-1)
        return nonmouth, mouth

    def project_nonmouth_features(self, nonmouth):
        expression = nonmouth[..., :100]
        expression = expression - (expression @ self.mouth_basis) @ self.mouth_basis.T
        return torch.cat([expression, nonmouth[..., 100:104]], dim=-1)

    def merge_features(self, nonmouth, mouth):
        if nonmouth.shape[:-1] != mouth.shape[:-1]:
            raise ValueError(
                f"branch shapes do not align: {tuple(nonmouth.shape)} vs {tuple(mouth.shape)}"
            )
        nonmouth = self.project_nonmouth_features(nonmouth)
        expression = nonmouth[..., :100] + mouth[..., : self.mouth_rank] @ self.mouth_basis.T
        output = expression.new_zeros(*expression.shape[:-1], 108)
        output[..., :100] = expression
        output[..., 103:104] = mouth[..., self.mouth_rank : self.mouth_rank + 1]
        output[..., 104:108] = nonmouth[..., 100:104]
        return output

    def forward(self, motion):
        nonmouth_target, mouth_target = self.split_features(motion)
        nonmouth_output = self.nonmouth_vq(nonmouth_target)
        mouth_output = self.mouth_vq(mouth_target)
        rec_nonmouth = self.project_nonmouth_features(nonmouth_output["rec_pose"])
        rec_pose = self.merge_features(rec_nonmouth, mouth_output["rec_pose"])
        return {
            "rec_pose": rec_pose,
            "rec_nonmouth": rec_nonmouth,
            "rec_mouth": mouth_output["rec_pose"],
            "target_nonmouth": nonmouth_target,
            "target_mouth": mouth_target,
            "commit_loss": 0.5
            * (nonmouth_output["commit_loss"] + mouth_output["commit_loss"]),
            "perplexity": 0.5
            * (nonmouth_output["perplexity"] + mouth_output["perplexity"]),
            "nonmouth_perplexity": nonmouth_output["perplexity"],
            "mouth_perplexity": mouth_output["perplexity"],
        }

    @torch.no_grad()
    def encode(self, motion):
        nonmouth, mouth = self.split_features(motion)
        nonmouth_idx, nonmouth_codes = self.nonmouth_vq.encode(nonmouth)
        mouth_idx, mouth_codes = self.mouth_vq.encode(mouth)
        if nonmouth_idx.shape[:2] != mouth_idx.shape[:2]:
            raise ValueError(
                f"branch token lengths differ: {tuple(nonmouth_idx.shape)} vs {tuple(mouth_idx.shape)}"
            )
        return torch.cat([nonmouth_idx, mouth_idx], dim=-1), {
            "nonmouth": nonmouth_codes,
            "mouth": mouth_codes,
        }

    def forward_decoder(self, code_idx):
        if code_idx.ndim != 3 or code_idx.shape[-1] != self.num_quantizers:
            raise ValueError(
                f"expected code indices (B,T,{self.num_quantizers}), got {tuple(code_idx.shape)}"
            )
        nonmouth = self.nonmouth_vq.forward_decoder(
            code_idx[..., : self.nonmouth_num_quantizers]
        )
        mouth = self.mouth_vq.forward_decoder(
            code_idx[..., self.nonmouth_num_quantizers :]
        )
        return self.merge_features(nonmouth, mouth)

    def mouth_vertex_mse_mm2(self, prediction, target):
        if self.mouth_vertex_matrix.shape[1] == 0:
            return prediction.new_zeros(())
        expression_error = prediction[..., :100] - target[..., :100]
        vertex_error_mm = (expression_error @ self.mouth_vertex_matrix) * 1000.0
        return F.mse_loss(vertex_error_mm, torch.zeros_like(vertex_error_mm))


def build_factorized_flame_rvqvae(checkpoint, device="cpu"):
    if checkpoint.get("model_type") != "factorized_flame_rvq_4plus2":
        raise ValueError("not a factorized FLAME RVQ checkpoint")
    state = checkpoint["net"]
    config = checkpoint["args"]
    args = SimpleNamespace(**config)
    args.shared_codebook = False
    model = FactorizedFlameRVQVAE(
        args,
        mouth_basis=state["mouth_basis"],
        mouth_vertex_matrix=state["mouth_vertex_matrix"],
        nb_code=int(config["nb_code"]),
        code_dim=int(config["code_dim"]),
        output_emb_width=int(config["code_dim"]),
        down_t=int(config["down_t"]),
        stride_t=int(config["stride_t"]),
        width=int(config["width"]),
        depth=int(config["depth"]),
        dilation_growth_rate=int(config["dilation_growth_rate"]),
        activation=config["vq_act"],
        norm=config["vq_norm"],
        nonmouth_num_quantizers=int(checkpoint["nonmouth_num_quantizers"]),
        mouth_num_quantizers=int(checkpoint["mouth_num_quantizers"]),
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_factorized_flame_rvqvae(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return build_factorized_flame_rvqvae(checkpoint, device=device)
