"""Audio preprocessing, motion decoding and style features for joint inference."""
import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import librosa
import numpy as np
import torch
from lhotse import Fbank
from torch.nn.utils import rnn
from models.vq.config import (
    UPPER_JOINTS, FACTORIZED_GESTURE_PARTS, FACTORIZED_UPPER_INDICES,
    build_motion_tokenizer_cfg, build_motion_rvqvae,
)
from inline_motion_exemplar import match_action_families, normalize_text
from utils import rotation_conversions as rc


from release_paths import CHECKPOINT_ROOT
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = str(CHECKPOINT_ROOT / "standalone_generator")

DEFAULT_BASE_LLM_PATH = str(CHECKPOINT_ROOT / "standalone_generator")

DEFAULT_AUDIO_ENCODER_PATH = ""

DEFAULT_VQ_PATH = ""

DEFAULT_UPPER_VQ_PATH = str(CHECKPOINT_ROOT / "tokenizers/upper.pth")

DEFAULT_HANDS_VQ_PATH = str(CHECKPOINT_ROOT / "tokenizers/hands.pth")

DEFAULT_STATS_DIR = str(REPO_ROOT / "assets/motion_stats")

DEFAULT_STATS_PREFIX = "seamless_naturalistic_improvised_train"

DEFAULT_STATS_SUFFIX = "filtered2"

DEFAULT_INLINE_EXEMPLAR_LIBRARY = ""


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {value}")


def extract_speaker_id(audio_file: str) -> str:
    if not isinstance(audio_file, str) or audio_file.strip() == "":
        return ""
    base_name = os.path.basename(audio_file)
    match = re.search(r"(?:^|_)P([^_/.]+)", base_name)
    if match is not None:
        return match.group(1)
    match = re.search(r"P([^_/.]+)", base_name)
    return match.group(1) if match is not None else ""


def build_system_prompt(base_prompt: str, speaker_id: str) -> str:
    speaker_id = str(speaker_id).strip()
    if speaker_id == "":
        return base_prompt
    return f"{base_prompt}. Speaker ID :{speaker_id}."


def resolve_speaker_id(speaker_audio_path: str, add_speaker_id: bool, speaker_id: str = "") -> str:
    if not add_speaker_id:
        return ""
    speaker_id = str(speaker_id).strip()
    if speaker_id != "":
        return speaker_id
    return extract_speaker_id(speaker_audio_path)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def choose_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": torch.bfloat16,
    }
    if dtype_arg not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_arg}")
    return mapping[dtype_arg]


def load_audio(audio_path: str, target_sr: int) -> np.ndarray:
    if not audio_path:
        return np.zeros(0, dtype=np.float32)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    wav, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    return wav.astype(np.float32)


def fit_audio_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
    if audio.shape[0] == target_samples:
        return audio
    if audio.shape[0] > target_samples:
        return audio[:target_samples]
    return np.pad(audio, (0, target_samples - audio.shape[0]), mode="constant")


def compute_num_windows(
    speaker_audio: np.ndarray,
    listener_audio: np.ndarray,
    samples_per_window: int,
    dialog: bool,
    num_windows: int = None,
    max_windows: int = None,
):
    if num_windows is not None:
        result = int(num_windows)
    else:
        speaker_windows = speaker_audio.shape[0] // samples_per_window
        if dialog:
            listener_windows = listener_audio.shape[0] // samples_per_window
            result = min(speaker_windows, listener_windows)
        else:
            result = speaker_windows

    if max_windows is not None:
        result = min(result, int(max_windows))
    if result <= 0:
        raise ValueError(
            "Audio is too short to form one 0.8s generation window. "
            "Please provide longer audio or pass --num-windows explicitly."
        )
    return result


def build_fbank_batch(fbank_extractor: Fbank, audio_list, audio_sr: int):
    features = fbank_extractor.extract_batch(audio_list, sampling_rate=audio_sr)
    if isinstance(features, list):
        fbank_list = [feat if feat.ndim == 2 else feat.squeeze(0) for feat in features]
    else:
        fbank_list = [feat for feat in features]
    fbank_lens = torch.tensor([feat.size(0) for feat in fbank_list], dtype=torch.long)
    fbank_feature = rnn.pad_sequence(fbank_list, batch_first=True)
    return fbank_feature, fbank_lens


def split_audio_into_chunks(audio: np.ndarray, chunk_size: int):
    if audio.shape[0] == 0:
        return []
    return [audio[start : start + chunk_size] for start in range(0, audio.shape[0], chunk_size)]


def concat_audio_chunks(chunks):
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
    if temperature <= 0:
        return logits.argmax(dim=-1)

    scores = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(int(top_k), scores.size(-1))
        topk_values, _ = torch.topk(scores, k=k, dim=-1)
        kth_values = topk_values[..., -1, None]
        scores = scores.masked_fill(scores < kth_values, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_remove_mask = cumulative_probs > top_p
        sorted_remove_mask[..., 1:] = sorted_remove_mask[..., :-1].clone()
        sorted_remove_mask[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(sorted_remove_mask, float("-inf"))

        scores = torch.full_like(scores, float("-inf"))
        scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)

    probs = torch.softmax(scores, dim=-1)
    sampled = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)
    return sampled.view(*probs.shape[:-1])


def _timed_words(word_file: str):
    if not word_file or not os.path.exists(word_file):
        return []
    try:
        with open(word_file, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return []
    out = []
    for chunk in obj.get("metadata:transcript", []) if isinstance(obj, dict) else []:
        for word_info in chunk.get("words", []) if isinstance(chunk, dict) else []:
            try:
                start = float(word_info.get("start"))
                end = float(word_info.get("end"))
            except Exception:
                continue
            word = str(word_info.get("word", "")).strip()
            if word and end > start:
                out.append((start, end, word))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def load_inline_exemplar_library(path: str, args):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Inline exemplar library must be a JSON list or a dict containing `data`.")
    by_family = {}
    for row in rows:
        family = str(row.get("action_family", ""))
        pose_file = str(row.get("pose_file", row.get("pose_path", "")))
        source_start = int(row.get(
            "reference_start_frame",
            row.get("causal_reference_start_frame", 0),
        ))
        source_length = int(row.get(
            "reference_length_frames",
            row.get("causal_reference_length_frames", args.inline_exemplar_reference_frames),
        ))
        start_frame = source_start + (source_length - int(args.inline_exemplar_reference_frames)) // 2
        if not family or start_frame < 0 or not os.path.exists(pose_file):
            continue
        if not bool(row.get("normal_posture", True)):
            continue
        item = dict(row)
        item["event_uid"] = str(row.get("event_uid", row.get("exemplar_id", "")))
        item["pose_file"] = pose_file
        item["flame_file"] = str(
            row.get("flame_file", str(Path(pose_file).with_name(Path(pose_file).stem + "_flame.npz")))
        )
        item["causal_reference_start_frame"] = start_frame
        item["causal_reference_length_frames"] = int(args.inline_exemplar_reference_frames)
        if row.get("reference_anchor_frame") is not None:
            item["reference_anchor_frame"] = int(round(float(row["reference_anchor_frame"])))
        else:
            item["reference_anchor_frame"] = (
                int(round(float(row.get("trigger_time_sec", 0.0)) * args.pose_fps)) - start_frame
            )
        item["quality_score"] = float(row.get("quality_score", row.get("motion_quality", 0.0)))
        by_family.setdefault(family, []).append(item)
    for family in by_family:
        by_family[family].sort(key=lambda row: (-float(row.get("quality_score", 0.0)), row.get("event_uid", "")))
    return by_family


def build_inline_exemplar_schedule(word_file: str, num_windows: int, segment_start_sec: float, args, library):
    if not bool(getattr(args, "inline_exemplar_enabled", False)):
        return {}
    forced_family = str(getattr(args, "inline_exemplar_force_family", "")).strip()
    forced_uid = str(getattr(args, "inline_exemplar_event_uid", "")).strip()
    exemplar_chunks = int(math.ceil(
        args.inline_exemplar_reference_frames / args.pose_fps / args.window_sec
    ))

    def select_reference(family, selector):
        candidates = library.get(family, [])
        if forced_uid:
            candidates = [row for row in candidates if str(row.get("event_uid", "")) == forced_uid]
        if not candidates:
            return None, [], -1
        top_k = max(1, int(getattr(args, "inline_exemplar_top_k", 4)))
        pool = candidates[:top_k]
        if forced_uid:
            selected_index = 0
        else:
            seed = int(getattr(args, "inline_exemplar_selection_seed", 42))
            digest = hashlib.sha1(f"{seed}|{family}|{selector}".encode("utf-8")).digest()
            selected_index = int.from_bytes(digest[:8], "big") % len(pool)
        return pool[selected_index], [str(row.get("event_uid", "")) for row in pool], selected_index

    schedule_file = str(getattr(args, "inline_exemplar_schedule_file", "")).strip()
    if schedule_file:
        payload = json.loads(Path(schedule_file).read_text(encoding="utf-8"))
        rows = payload.get("events", payload if isinstance(payload, list) else [])
        schedule = {}
        for row in rows:
            family = str(row["action_family"])
            event_uid = str(row["template_uid"])
            candidates = library.get(family, [])
            matching = [item for item in candidates if str(item.get("event_uid", "")) == event_uid]
            if len(matching) != 1:
                raise ValueError(f"Router schedule template not found exactly once: {family}/{event_uid}")
            insert_chunk = int(row["insert_chunk"])
            if insert_chunk < 0 or insert_chunk + exemplar_chunks > num_windows:
                raise ValueError(f"Router schedule interval outside stream: {insert_chunk}")
            if insert_chunk in schedule:
                raise ValueError(f"Router schedule has duplicate insert chunk: {insert_chunk}")
            schedule[insert_chunk] = {
                **row,
                "insert_chunk": insert_chunk,
                "query_text": "<learned_router>",
                "matched_text": "<learned_router>",
                "timing_mode": "learned_router",
                "reference": matching[0],
                "reference_candidate_event_uids": [event_uid],
                "reference_candidate_rank": 0,
            }
        return schedule

    if forced_family:
        reference, candidate_uids, candidate_rank = select_reference(
            forced_family, f"forced|{args.inline_exemplar_force_chunk}"
        )
        if reference is None:
            raise ValueError(f"No valid inline exemplar for forced family={forced_family!r}, event_uid={forced_uid!r}.")
        chunk_idx = int(args.inline_exemplar_force_chunk)
        if chunk_idx < 0 or chunk_idx + exemplar_chunks > num_windows:
            raise ValueError(
                f"Forced inline exemplar interval [{chunk_idx}, {chunk_idx + exemplar_chunks}) "
                f"is outside [0, {num_windows})."
            )
        return {
            chunk_idx: {
                "insert_chunk": chunk_idx,
                "action_family": forced_family,
                "query_text": "<forced>",
                "matched_text": "<forced>",
                "predicted_trigger_time_sec": float(segment_start_sec) + chunk_idx * args.window_sec + args.inline_exemplar_lead_sec,
                "reference": reference,
                "reference_candidate_event_uids": candidate_uids,
                "reference_candidate_rank": candidate_rank,
            }
        }

    words = _timed_words(word_file)
    proposals = []
    timing_mode = str(getattr(args, "inline_exemplar_timing_mode", "action_anchor")).strip().lower()
    if timing_mode not in {"fixed_lead", "action_anchor"}:
        raise ValueError(f"Unsupported inline exemplar timing mode: {timing_mode}")
    for detection_chunk in range(num_windows):
        future_start = float(segment_start_sec) + detection_chunk * args.window_sec + args.inline_exemplar_lead_sec
        future_end = future_start + args.window_sec
        query_items = [(start, end, normalize_text(word)) for start, end, word in words if end >= future_start and start < future_end + 0.4]
        query_items = [(start, end, word) for start, end, word in query_items if word]
        query_text = " ".join(word for _, _, word in query_items)
        matches = [row for row in match_action_families(query_text) if row["action_family"] in library]
        matches.sort(key=lambda row: (row["action_family"] == "backchannel_nod", row["char_start"]))
        if not matches:
            continue
        match = matches[0]
        reference, candidate_uids, candidate_rank = select_reference(
            match["action_family"], f"{detection_chunk}|{query_text}"
        )
        if reference is None:
            continue
        char_end = int(match["char_start"]) + len(str(match["match"]))
        cursor = 0
        matched_items = []
        for item in query_items:
            word_start = cursor
            word_end = word_start + len(item[2])
            if min(word_end, char_end) > max(word_start, int(match["char_start"])):
                matched_items.append(item)
            cursor = word_end + 1
        if matched_items:
            text_event_start_sec = float(matched_items[0][0])
            text_event_end_sec = float(matched_items[-1][1])
            predicted_trigger_time_sec = 0.5 * (text_event_start_sec + text_event_end_sec)
        else:
            text_event_start_sec = future_start
            text_event_end_sec = future_end
            predicted_trigger_time_sec = 0.5 * (future_start + future_end)

        if timing_mode == "action_anchor":
            desired_insert_time_sec = (
                predicted_trigger_time_sec
                - int(reference["reference_anchor_frame"]) / float(args.pose_fps)
            )
            insert_chunk = int(round(
                (desired_insert_time_sec - float(segment_start_sec)) / float(args.window_sec)
            ))
            insert_chunk = max(detection_chunk, insert_chunk)
        else:
            insert_chunk = detection_chunk
            desired_insert_time_sec = float(segment_start_sec) + insert_chunk * args.window_sec
        if insert_chunk < 0 or insert_chunk + exemplar_chunks > num_windows:
            continue
        actual_insert_time_sec = float(segment_start_sec) + insert_chunk * args.window_sec
        proposals.append({
            "insert_chunk": insert_chunk,
            "detection_chunk": detection_chunk,
            "detection_time_sec": float(segment_start_sec) + detection_chunk * args.window_sec,
            "action_family": match["action_family"],
            "query_text": query_text,
            "matched_text": match["match"],
            "text_event_start_sec": text_event_start_sec,
            "text_event_end_sec": text_event_end_sec,
            "predicted_trigger_time_sec": predicted_trigger_time_sec,
            "desired_insert_time_sec": desired_insert_time_sec,
            "actual_insert_time_sec": actual_insert_time_sec,
            "action_anchor_time_error_sec": (
                actual_insert_time_sec
                + int(reference["reference_anchor_frame"]) / float(args.pose_fps)
                - predicted_trigger_time_sec
            ),
            "timing_mode": timing_mode,
            "reference": reference,
            "reference_candidate_event_uids": candidate_uids,
            "reference_candidate_rank": candidate_rank,
        })

    schedule = {}
    last_insert_chunk = -10**9
    cooldown_chunks = exemplar_chunks
    for event in sorted(proposals, key=lambda row: (row["insert_chunk"], row["detection_chunk"])):
        insert_chunk = int(event["insert_chunk"])
        if insert_chunk in schedule or insert_chunk - last_insert_chunk < cooldown_chunks:
            continue
        schedule[insert_chunk] = event
        last_insert_chunk = insert_chunk
    return schedule


def maybe_load_legacy_qwen35_state_dict(model_path: str):
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        return None

    from safetensors import safe_open

    old_prefix = "base_llm.model.language_model."
    old_expert_pattern = re.compile(
        r"^(base_llm\.model(?:\.language_model)?\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    needs_remap = False
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(old_prefix):
                needs_remap = True
                break
    if not needs_remap:
        return None

    from safetensors.torch import load_file as load_safetensors_file

    state_dict = load_safetensors_file(safetensors_path, device="cpu")
    remapped_state_dict = {}
    expert_parts = {}
    for key, value in state_dict.items():
        if key.startswith(old_prefix):
            key = "base_llm.model." + key[len(old_prefix) :]
        match = old_expert_pattern.match(key)
        if match is not None:
            expert_prefix, expert_idx, proj_name = match.groups()
            expert_parts.setdefault(expert_prefix, {}).setdefault(int(expert_idx), {})[proj_name] = value
            continue
        remapped_state_dict[key] = value

    for expert_prefix, by_expert in expert_parts.items():
        expert_ids = sorted(by_expert)
        remapped_state_dict[f"{expert_prefix}.gate_up_proj"] = torch.stack(
            [
                torch.cat([by_expert[idx]["gate_proj"], by_expert[idx]["up_proj"]], dim=0)
                for idx in expert_ids
            ],
            dim=0,
        )
        remapped_state_dict[f"{expert_prefix}.down_proj"] = torch.stack(
            [by_expert[idx]["down_proj"] for idx in expert_ids],
            dim=0,
        )
    return remapped_state_dict


def reset_motion_weight_buffers(model, model_args):
    weights = torch.tensor(
        model_args.motion_level_weights,
        dtype=model.motion_embed_weights.dtype,
        device=model.motion_embed_weights.device,
    )
    with torch.no_grad():
        model.motion_embed_weights.copy_(weights)
        model.motion_loss_weights.copy_(weights.to(dtype=model.motion_loss_weights.dtype, device=model.motion_loss_weights.device))


def build_motion_decoder_cfg(motion_tokenizer_type: str):
    if motion_tokenizer_type not in {"upper_hands_trans", "upper", "hands", *FACTORIZED_GESTURE_PARTS}:
        raise ValueError(f"Unsupported motion tokenizer type for decoder: {motion_tokenizer_type}")
    cfg = vars(build_motion_tokenizer_cfg(motion_tokenizer_type)).copy()
    cfg["body_part"] = cfg.pop("gesture_body_part")
    for key in ("shared_codebook", "quantize_dropout_prob", "mu"):
        cfg.pop(key)
    return cfg


def load_motion_decoder(vqvae_path: str, motion_tokenizer_type: str, device: torch.device):
    # Validate the public decoder layout before constructing the shared RVQ model.
    build_motion_decoder_cfg(motion_tokenizer_type)
    ckpt = torch.load(vqvae_path, map_location="cpu", weights_only=False)
    cfg = build_motion_tokenizer_cfg(motion_tokenizer_type)
    cfg.causal = bool(ckpt.get("args", {}).get("causal", False))
    model = build_motion_rvqvae(cfg)
    state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_motion_stats(stats_dir: str, prefix: str, suffix: str):
    body_mean = torch.from_numpy(np.load(os.path.join(stats_dir, f"{prefix}_body_mean_{suffix}.npy"))).float()
    body_std = torch.from_numpy(np.load(os.path.join(stats_dir, f"{prefix}_body_std_{suffix}.npy"))).float()

    upper_indices = []
    for joint_idx in UPPER_JOINTS:
        upper_indices.extend(range(6 * joint_idx, 6 * (joint_idx + 1)))

    stats = {
        "mean_upper": body_mean[upper_indices],
        "std_upper": body_std[upper_indices],
        "mean_left_hand": torch.from_numpy(np.load(os.path.join(stats_dir, f"{prefix}_left_hand_mean_{suffix}.npy"))).float(),
        "std_left_hand": torch.from_numpy(np.load(os.path.join(stats_dir, f"{prefix}_left_hand_std_{suffix}.npy"))).float(),
        "mean_right_hand": torch.from_numpy(np.load(os.path.join(stats_dir, f"{prefix}_right_hand_mean_{suffix}.npy"))).float(),
        "std_right_hand": torch.from_numpy(np.load(os.path.join(stats_dir, f"{prefix}_right_hand_std_{suffix}.npy"))).float(),
        "mean_transv": torch.from_numpy(np.load(os.path.join(stats_dir, f"{prefix}_translation_mean_{suffix}.npy"))).float(),
        "std_transv": torch.from_numpy(np.load(os.path.join(stats_dir, f"{prefix}_translation_std_{suffix}.npy"))).float(),
    }
    return stats


def pick_pose_key(pose_data, plain_key: str, namespaced_key: str) -> str:
    if plain_key in pose_data:
        return plain_key
    if namespaced_key in pose_data:
        return namespaced_key
    raise KeyError(f"Cannot find '{plain_key}' or '{namespaced_key}' in pose file.")


def build_motion_features_from_pose(pose_path: str, start_idx: int, length: int, stats, include_translation=True):
    if not pose_path or not os.path.exists(pose_path):
        raise FileNotFoundError(f"Motion style pose file not found: {pose_path}")
    start_idx = int(start_idx)
    length = int(length)
    if start_idx < 0 or length <= 0:
        raise ValueError("`motion_style_start` must be non-negative and `motion_style_length` must be positive.")
    fin_idx = start_idx + length

    with np.load(pose_path) as pose_data:
        body_key = pick_pose_key(pose_data, "body_pose", "smplh:body_pose")
        left_key = pick_pose_key(pose_data, "left_hand_pose", "smplh:left_hand_pose")
        right_key = pick_pose_key(pose_data, "right_hand_pose", "smplh:right_hand_pose")
        pose_body_np = pose_data[body_key]
        if fin_idx > int(pose_body_np.shape[0]):
            raise ValueError(
                f"Motion style segment [{start_idx}, {fin_idx}) exceeds pose length {pose_body_np.shape[0]}."
            )
        pose_body = torch.from_numpy(pose_body_np[start_idx:fin_idx]).float().reshape(length, 21, 3)[:, UPPER_JOINTS, :]
        pose_left_hand = torch.from_numpy(pose_data[left_key][start_idx:fin_idx]).float().reshape(length, 15, 3)
        pose_right_hand = torch.from_numpy(pose_data[right_key][start_idx:fin_idx]).float().reshape(length, 15, 3)
        pose_trans = torch.zeros((length, 3))
        if include_translation:
            trans_key = pick_pose_key(pose_data, "translation", "smplh:translation")
            pose_trans = torch.from_numpy(pose_data[trans_key][start_idx:fin_idx]).float()

    n = pose_body.shape[0]
    tar_pose_upper = rc.axis_angle_to_matrix(pose_body)
    tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(n, 78)
    tar_pose_upper = (tar_pose_upper - stats["mean_upper"]) / stats["std_upper"]

    tar_pose_left_hand = rc.axis_angle_to_matrix(pose_left_hand)
    tar_pose_left_hand = rc.matrix_to_rotation_6d(tar_pose_left_hand).reshape(n, 90)
    tar_pose_left_hand = (tar_pose_left_hand - stats["mean_left_hand"]) / stats["std_left_hand"]

    tar_pose_right_hand = rc.axis_angle_to_matrix(pose_right_hand)
    tar_pose_right_hand = rc.matrix_to_rotation_6d(tar_pose_right_hand).reshape(n, 90)
    tar_pose_right_hand = (tar_pose_right_hand - stats["mean_right_hand"]) / stats["std_right_hand"]

    centered_trans = pose_trans.clone()
    centered_trans[:, 0] -= centered_trans[0, 0]
    centered_trans[:, 2] -= centered_trans[0, 2]
    trans_v = torch.zeros_like(centered_trans)
    if n > 1:
        trans_v[1:, 0] = centered_trans[1:, 0] - centered_trans[:-1, 0]
        trans_v[0, 0] = trans_v[1, 0]
        trans_v[1:, 2] = centered_trans[1:, 2] - centered_trans[:-1, 2]
        trans_v[0, 2] = trans_v[1, 2]
    trans_v[:, 1] = centered_trans[:, 1]
    trans_v = (trans_v - stats["mean_transv"]) / stats["std_transv"]

    return {
        "upper": tar_pose_upper,
        "hands": torch.cat([tar_pose_left_hand, tar_pose_right_hand], dim=1),
        "torso": torch.cat([tar_pose_upper[:, 6 * idx : 6 * (idx + 1)] for idx in FACTORIZED_UPPER_INDICES["torso"]], dim=1),
        "left_arm": torch.cat([tar_pose_upper[:, 6 * idx : 6 * (idx + 1)] for idx in FACTORIZED_UPPER_INDICES["left_arm"]], dim=1),
        "right_arm": torch.cat([tar_pose_upper[:, 6 * idx : 6 * (idx + 1)] for idx in FACTORIZED_UPPER_INDICES["right_arm"]], dim=1),
        "left_hand": tar_pose_left_hand,
        "right_hand": tar_pose_right_hand,
        "upper_hands_trans": torch.cat(
            [tar_pose_upper, tar_pose_left_hand, tar_pose_right_hand, trans_v], dim=1
        ),
    }


def resolve_rvq_context_frames(args, part):
    key = f"rvq_{part}_context_frames"
    override = getattr(args, key, None)
    if override is not None:
        return int(override)
    model_path = Path(args.model_path)
    for config_path in (model_path / "config.json", model_path.parent / "config.json"):
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if key in config:
                return int(config[key])
            if "data_args" in config:
                return int(config["data_args"].get(key, 0))
    return 0


def _load_motion_decoders(args, device: torch.device):
    context = resolve_rvq_context_frames(args, "gesture")
    if args.motion_tokenizer_type == "upper_hands_trans":
        return {
            "upper_hands_trans": load_motion_decoder(
                args.vqvae_upper_hands_trans_path,
                "upper_hands_trans",
                device,
            ).set_context_frames(context)
        }
    if args.motion_tokenizer_type == "upper_hands":
        if not args.vqvae_upper_path or not args.vqvae_hands_path:
            raise ValueError(
                "`vqvae_upper_path` and `vqvae_hands_path` are required when "
                "`motion_tokenizer_type='upper_hands'`."
            )
        return {
            "upper": load_motion_decoder(args.vqvae_upper_path, "upper", device).set_context_frames(context),
            "hands": load_motion_decoder(args.vqvae_hands_path, "hands", device).set_context_frames(context),
        }
    if args.motion_tokenizer_type == "factorized_gesture_5":
        decoders = {}
        for part in FACTORIZED_GESTURE_PARTS:
            path = getattr(args, f"vqvae_{part}_path")
            if not path:
                raise ValueError(f"`vqvae_{part}_path` is required for factorized_gesture_5.")
            decoders[part] = load_motion_decoder(path, part, device).set_context_frames(context)
        return decoders
    raise ValueError(f"Unsupported motion_tokenizer_type: {args.motion_tokenizer_type}")


def rot6d_to_axis_angle(rot6d_tensor: torch.Tensor, num_joints: int) -> torch.Tensor:
    rot_mats = rc.rotation_6d_to_matrix(rot6d_tensor.reshape(-1, num_joints, 6))
    return rc.matrix_to_axis_angle(rot_mats).reshape(-1, num_joints, 3)


def _decode_motion_tokens(motion_token_ids, decoder, motion_vocab_size):
    ids = motion_token_ids.clone().long().clamp(0, motion_vocab_size - 1)
    return decoder.forward_decoder(ids.to(next(decoder.parameters()).device))[0].detach().cpu()


def _motion_features_to_smplh(upper, left_hand, right_hand, stats, translation=None):
    """Unnormalize 6D rotations; ungenerated lower-body joints and root stay zero."""
    num_frames = upper.size(0)
    body = torch.zeros(num_frames, 21, 3, dtype=torch.float32)
    body[:, UPPER_JOINTS, :] = rot6d_to_axis_angle(
        upper * stats["std_upper"] + stats["mean_upper"], len(UPPER_JOINTS)
    )
    left = rot6d_to_axis_angle(
        left_hand * stats["std_left_hand"] + stats["mean_left_hand"], 15
    )
    right = rot6d_to_axis_angle(
        right_hand * stats["std_right_hand"] + stats["mean_right_hand"], 15
    )
    return {
        "body_pose": body.reshape(num_frames, -1).numpy(),
        "left_hand_pose": left.reshape(num_frames, -1).numpy(),
        "right_hand_pose": right.reshape(num_frames, -1).numpy(),
        "global_orient": np.zeros((num_frames, 3), dtype=np.float32),
        "translation": (np.zeros((num_frames, 3), dtype=np.float32)
                        if translation is None else translation.numpy()),
    }


def decode_upper_hands_trans_to_smplh(
    motion_token_ids: torch.Tensor,
    motion_decoder,
    stats,
    motion_vocab_size: int,
):
    rec_motion = _decode_motion_tokens(motion_token_ids, motion_decoder, motion_vocab_size)
    trans_v = rec_motion[:, 258:261] * stats["std_transv"] + stats["mean_transv"]
    # X/Z store per-frame displacement, while Y stores absolute height.
    translation = torch.cumsum(trans_v, dim=0)
    translation[:, 1] = trans_v[:, 1]
    return _motion_features_to_smplh(
        rec_motion[:, :78], rec_motion[:, 78:168], rec_motion[:, 168:258], stats, translation
    )


def decode_upper_hands_to_smplh(
    motion_token_ids: torch.Tensor,
    motion_decoders,
    stats,
    motion_vocab_size: int,
):
    if motion_token_ids.size(-1) < 12:
        raise ValueError(
            f"`upper_hands` decoding requires at least 12 codebooks, got {motion_token_ids.size(-1)}."
        )

    # The first six RVQ indices are upper body; the next six are both hands.
    upper = _decode_motion_tokens(motion_token_ids[:, :, :6], motion_decoders["upper"], motion_vocab_size)
    hands = _decode_motion_tokens(motion_token_ids[:, :, 6:12], motion_decoders["hands"], motion_vocab_size)
    return _motion_features_to_smplh(upper[:, :78], hands[:, :90], hands[:, 90:180], stats)


def decode_factorized_gesture_to_smplh(
    motion_token_ids: torch.Tensor,
    motion_decoders,
    stats,
    motion_vocab_size: int,
):
    if motion_token_ids.size(-1) < 12:
        raise ValueError(
            f"`factorized_gesture_5` decoding requires 12 codebooks, got {motion_token_ids.size(-1)}."
        )

    offsets = {"torso": (0, 2), "left_arm": (2, 4), "right_arm": (4, 6), "left_hand": (6, 9), "right_hand": (9, 12)}
    decoded = {}
    for part in FACTORIZED_GESTURE_PARTS:
        start, end = offsets[part]
        decoded[part] = _decode_motion_tokens(
            motion_token_ids[:, :, start:end], motion_decoders[part], motion_vocab_size
        )

    num_frames = decoded["torso"].size(0)
    rec_upper_norm = torch.zeros(num_frames, 78, dtype=torch.float32)
    for part in ("torso", "left_arm", "right_arm"):
        source = decoded[part]
        for source_idx, upper_idx in enumerate(FACTORIZED_UPPER_INDICES[part]):
            rec_upper_norm[:, 6 * upper_idx : 6 * (upper_idx + 1)] = source[
                :, 6 * source_idx : 6 * (source_idx + 1)
            ]
    return _motion_features_to_smplh(
        rec_upper_norm, decoded["left_hand"], decoded["right_hand"], stats
    )


def decode_motion_to_smplh(
    motion_token_ids: torch.Tensor,
    motion_decoders,
    stats,
    motion_vocab_size: int,
    motion_tokenizer_type: str,
):
    if motion_tokenizer_type == "upper_hands_trans":
        return decode_upper_hands_trans_to_smplh(
            motion_token_ids=motion_token_ids,
            motion_decoder=motion_decoders["upper_hands_trans"],
            stats=stats,
            motion_vocab_size=motion_vocab_size,
        )
    if motion_tokenizer_type == "upper_hands":
        return decode_upper_hands_to_smplh(
            motion_token_ids=motion_token_ids,
            motion_decoders=motion_decoders,
            stats=stats,
            motion_vocab_size=motion_vocab_size,
        )
    if motion_tokenizer_type == "factorized_gesture_5":
        return decode_factorized_gesture_to_smplh(
            motion_token_ids=motion_token_ids,
            motion_decoders=motion_decoders,
            stats=stats,
            motion_vocab_size=motion_vocab_size,
        )
    raise ValueError(f"Unsupported motion_tokenizer_type: {motion_tokenizer_type}")


def json_load_compat(file_obj):
    data = json.load(file_obj)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    if isinstance(data, list):
        return data
    raise ValueError("Expected a json list or a dict with a `data` list.")


def load_motion_decoders(args, device: torch.device):
    """Use checkpoint/CLI precision for encoders without changing decoded motion."""
    from models.vq.precision import configure_encoding_precision, resolve_encoding_precision
    encoder = resolve_encoding_precision(args, 'rvq_encoder_dtype')
    distance = resolve_encoding_precision(args, 'rvq_quantizer_distance_dtype')
    return {part: configure_encoding_precision(model, encoder, distance)
            for part, model in _load_motion_decoders(args, device).items()}
