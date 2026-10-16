"""Locations of the separately distributed code, data and checkpoint packages."""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("ELLSA2_DATA_ROOT", CODE_ROOT.parent / "data")).expanduser().resolve()
CHECKPOINT_ROOT = Path(os.environ.get("ELLSA2_CHECKPOINT_ROOT", CODE_ROOT.parent / "checkpoints")).expanduser().resolve()


def resolve_path(value):
    """Resolve portable recipe paths; preserve explicitly supplied paths."""
    if isinstance(value, str):
        for prefix, root in (("data/", DATA_ROOT), ("checkpoints/", CHECKPOINT_ROOT),
                             ("assets/", CODE_ROOT / "assets"), ("configs/", CODE_ROOT / "configs"),
                             ("runs/", CODE_ROOT / "runs")):
            if value.startswith(prefix):
                return str(root / value[len(prefix):])
    return value


def resolve_config(config, stage="gesture_expression"):
    """Merge a training stage with shared defaults and resolve portable paths."""
    stages = config.get("stages", {})
    if stages and stage not in stages:
        raise ValueError(f"Unknown training stage: {stage}")
    overrides = stages.get(stage, {})
    return {section: {key: resolve_path(value) for key, value in
                      {**config[section], **overrides.get(section, {})}.items()}
            for section in ("model_args", "data_args", "training_args")}
