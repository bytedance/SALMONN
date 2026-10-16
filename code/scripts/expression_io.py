"""FLAME normalization and decoding helpers for joint inference."""
import json
from pathlib import Path
import numpy as np
import torch
from models.vq.factorized_flame_model import build_factorized_flame_rvqvae
from models.vq.config import build_motion_tokenizer_cfg, build_motion_rvqvae


from release_paths import CHECKPOINT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = str(CHECKPOINT_ROOT / "standalone_generator")

DEFAULT_BASE_LLM_PATH = str(CHECKPOINT_ROOT / "standalone_generator")

DEFAULT_AUDIO_ENCODER_PATH = ""

DEFAULT_FLAME_VQ_PATH = str(CHECKPOINT_ROOT / "tokenizers/flame.pth")

DEFAULT_FLAME_STATS_PATH = str(PROJECT_ROOT / "assets/flame_stats.json")


def _load_flame_decoder(vqvae_path: str, device: torch.device, context_frames=0):
    ckpt = torch.load(vqvae_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("model_type") == "factorized_flame_rvq_4plus2":
        return build_factorized_flame_rvqvae(ckpt, device=device).set_context_frames(context_frames)

    cfg = build_motion_tokenizer_cfg("flame_motioncode")
    cfg.gesture_body_part = "flame"
    cfg.causal = bool(ckpt.get("args", {}).get("causal", False))
    model = build_motion_rvqvae(cfg)
    model.load_state_dict(ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt, strict=True)
    model.to(device)
    model.eval()
    return model.set_context_frames(context_frames)


def load_motion_stats(stats_path: str):
    with open(stats_path, "r", encoding="utf-8") as f:
        motion_stats = json.load(f)
    motion_mean = np.asarray(motion_stats["motion_mean"], dtype=np.float32)
    motion_std = np.asarray(motion_stats["motion_std"], dtype=np.float32)
    return motion_mean, motion_std


def decode_expression_tokens_to_motioncode(
    motion_token_ids: torch.Tensor,
    flame_decoder,
    motion_mean: np.ndarray,
    motion_std: np.ndarray,
    motion_vocab_size: int,
):
    raw_motion_ids = motion_token_ids.clone().long().clamp(min=0, max=motion_vocab_size - 1)
    rec_motion_norm = flame_decoder.forward_decoder(
        raw_motion_ids.unsqueeze(0).to(next(flame_decoder.parameters()).device)
    )[0].detach().cpu().numpy()
    return rec_motion_norm * (motion_std + 1e-8) + motion_mean


def load_flame_decoder(vqvae_path: str, device: torch.device, context_frames=0,
                       encoder_dtype='float32', distance_dtype='float32'):
    from models.vq.precision import configure_encoding_precision
    return configure_encoding_precision(_load_flame_decoder(vqvae_path, device, context_frames),
                                        encoder_dtype, distance_dtype)


def resolve_style_flame_path(pose_path, flame_path=''):
    pose = Path(pose_path)
    candidates = ([Path(flame_path)] if flame_path else
                  [pose.with_name(pose.stem + '_flame.npz'),
                   pose.parent.parent / 'style_flame' / (pose.stem + '_flame.npz')])
    for path in candidates:
        if path.is_file():
            return str(path)
    raise FileNotFoundError('Expression style requires a matching FLAME reference; provide '
                            '--motion-style-flame-path or explicitly use --motion-style-expression false')


def load_flame_style_features(path, start, length, motion_mean, motion_std, fps=30):
    """Match Gesture_Dataset's raw UniLS slicing, crop centering and FP32 normalization."""
    with np.load(path) as data:
        if 'motion_fps' in data and float(data['motion_fps']) != fps:
            raise ValueError('FLAME style frame rate must match --flame-fps')
        motion = np.array(data['motioncode'][start:start + length], dtype=np.float32, copy=True)
        if len(motion) != length or not np.isfinite(motion).all():
            raise ValueError('FLAME style must contain the complete finite reference interval')
        if 'valid_mask' in data:
            valid = np.asarray(data['valid_mask'][start:start + length])
            if len(valid) != length or not valid.all():
                raise ValueError('FLAME style reference contains invalid frames')
    if motion.shape[1] != 108:
        if motion.shape[1] < 111:
            raise ValueError('FLAME motioncode must have 108 or at least 111 features')
        motion = np.concatenate([motion[:, :104], motion[:, 106:108], motion[:, 109:111]], axis=1)
    motion[:, 100:103] -= motion[:, 100:103].mean(axis=0, keepdims=True)
    return torch.from_numpy((motion - motion_mean) / (motion_std + 1e-8))
