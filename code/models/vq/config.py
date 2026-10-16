"""Shared gesture/FLAME RVQ layout for training, style encoding and inference."""
from types import SimpleNamespace

from models.vq.model import RVQVAE

# Joint indices refer to the 21 body joints after removing the SMPL root.
UPPER_JOINTS = [2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
FACTORIZED_GESTURE_PARTS = ("torso", "left_arm", "right_arm", "left_hand", "right_hand")
FACTORIZED_UPPER_INDICES = {
    "torso": [0, 1, 2, 3, 6],
    "left_arm": [4, 7, 9, 11],
    "right_arm": [5, 8, 10, 12],
}
FACTORIZED_NUM_QUANTIZERS = {
    "torso": 2, "left_arm": 2, "right_arm": 2, "left_hand": 3, "right_hand": 3,
}

_BASE_CONFIG = dict(
    nb_code=2048, code_dim=512, down_t=2, stride_t=2, width=512,
    depth=6, dilation_growth_rate=3, vq_act="relu", vq_norm=False,
)
_PART_DIMS = {
    "torso": 30, "left_arm": 24, "right_arm": 24, "left_hand": 90, "right_hand": 90,
    "upper": 78, "hands": 180, "upper_hands_trans": 261, "flame_motioncode": 108,
}
MOTION_TOKENIZER_CONFIGS = {}
for part, dim in _PART_DIMS.items():
    cfg = dict(_BASE_CONFIG, dim_pose=dim)
    if part in FACTORIZED_NUM_QUANTIZERS:
        cfg["num_quantizers"] = FACTORIZED_NUM_QUANTIZERS[part]
    if part in {"upper_hands_trans", "flame_motioncode"}:
        cfg["vq_norm"] = None
    MOTION_TOKENIZER_CONFIGS[part] = cfg


def build_motion_tokenizer_cfg(body_part):
    cfg = dict(MOTION_TOKENIZER_CONFIGS[body_part])
    return SimpleNamespace(
        num_quantizers=cfg.pop("num_quantizers", 6), shared_codebook=False,
        quantize_dropout_prob=0.2, mu=0.99, gesture_body_part=body_part, **cfg,
    )


def build_motion_rvqvae(cfg):
    """Keep the architecture identical for online encoding and offline decoding."""
    return RVQVAE(
        cfg, cfg.dim_pose, cfg.nb_code, cfg.code_dim, cfg.code_dim,
        cfg.down_t, cfg.stride_t, cfg.width, cfg.depth,
        cfg.dilation_growth_rate, cfg.vq_act, cfg.vq_norm,
    )
