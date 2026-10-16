"""Curated speaking/listening motion references from the released test set."""
import csv
import json
import re

from release_paths import DATA_ROOT

DEFAULT_PREFIX = "r2d_00073"
# Reference motions paired with the three selected dialogue examples.
PREFIX_LABELS = {
    "r2d_00073": "r2d_00073 · P0421",
    "fro_00079": "fro_00079 · P0093",
    "fro_00251": "fro_00251 · P1307A",
}


def load_motion_prefixes():
    rows = {row["group_id"]: row for row in map(json.loads, (DATA_ROOT / "testset/manifest.jsonl").read_text().splitlines())}
    with (DATA_ROOT / "manifests/speaker_quality.csv").open() as handle:
        quality = {row["speaker_id"]: row for row in csv.DictReader(handle)}
    selected = set(json.loads((DATA_ROOT / "manifests/selected_speakers.json").read_text()))
    prefixes = {}
    for prefix_id, label in PREFIX_LABELS.items():
        row = rows[prefix_id]
        speaker = re.search(r"_P([^/.]+)", row["source_pose_id"]).group(1)
        stats = quality.get(speaker, {})
        prefixes[prefix_id] = {
            "id": prefix_id, "label": label, "speaker_id": speaker,
            "in_training_inventory": speaker in selected,
            "pose_path": DATA_ROOT / row["motion_style_pose_path"],
            "reference_pose": row["motion_style_pose_path"],
            "source_start_frame": row["source_motion_style_start"],
            "source_frames": row["motion_style_length"],
            "speaking_seconds": row["motion_style_speaking_seconds"],
            "listening_seconds": row["motion_style_listening_seconds"],
            "speaker_quality_mean": float(stats["mean_score"]) if stats else None,
            "speaker_pass_rate": float(stats["pass_rate"]) if stats else None,
        }
    return prefixes
