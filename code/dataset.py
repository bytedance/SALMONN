"""Sample aligned audio/motion windows and assemble autoregressive training examples."""
import pathlib
import hashlib
from torch.utils.data import Dataset
from collections import OrderedDict
import json
import random
import warnings
import torch
import math
import numpy as np
import librosa
import soundfile as sf
import os
import re
import zipfile
from torch.nn.utils import rnn
from lhotse import Fbank, FbankConfig

from utils import rotation_conversions as rc
from inline_motion_exemplar import EXEMPLAR_END_TOKEN, EXEMPLAR_START_TOKEN, exemplar_blend_weights, slerp_axis_angle
from models.vq.config import (
    UPPER_JOINTS, FACTORIZED_GESTURE_PARTS, FACTORIZED_UPPER_INDICES,
    MOTION_TOKENIZER_CONFIGS, build_motion_tokenizer_cfg, build_motion_rvqvae,
)
from safetensors.torch import load_file
from release_paths import resolve_path

DEFAULT_FLAME_STATS_PATH = (
    str(pathlib.Path(__file__).resolve().parent / "assets/flame_stats.json")
)

def _pick_key(motion_data, plain_key: str, namespaced_key: str) -> str:
    """Return the first available key from an ordered list of aliases."""
    if plain_key in motion_data:
        return plain_key
    if namespaced_key in motion_data:
        return namespaced_key
    raise KeyError(f"Cannot find '{plain_key}' or '{namespaced_key}' in npz keys: {list(motion_data.keys())[:50]} ...")

def _read_npz_array_shape(npz_file, plain_key: str, namespaced_key: str):
    """Read an NPZ array shape from its header without loading the array."""
    candidate_members = (plain_key + ".npy", namespaced_key + ".npy")
    with zipfile.ZipFile(npz_file, "r") as zf:
        members = set(zf.namelist())
        member_name = None
        for candidate in candidate_members:
            if candidate in members:
                member_name = candidate
                break
        if member_name is None:
            raise KeyError(f"Cannot find {candidate_members} in npz members: {list(members)[:50]} ...")
        with zf.open(member_name, "r") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(f)
            else:
                shape, _, _ = np.lib.format._read_array_header(f, version)
            return shape
POSE_KEY_ALIASES = {
    "smplh:body_pose": ("body_pose", "smplh:body_pose"),
    "smplh:left_hand_pose": ("left_hand_pose", "smplh:left_hand_pose"),
    "smplh:right_hand_pose": ("right_hand_pose", "smplh:right_hand_pose"),
    "smplh:translation": ("translation", "smplh:translation"),
}

def _get_alignment_score_threshold(args):
    """Resolve the alignment threshold from arguments, aliases or the environment."""
    for k in (
        'alignment_score_threshold',
        'alignment_threshold',
        'min_alignment',
    ):
        v = getattr(args, k, None)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue

    for env_k in (
        'GESTURE_ALIGNMENT_SCORE_THRESHOLD',
        'COSYVOICE_ALIGNMENT_SCORE_THRESHOLD',
        'COSYVOICE_ALIGN_SCORE_THRESHOLD',
    ):
        v = os.environ.get(env_k)
        if v is None or str(v) == '':
            continue
        try:
            return float(v)
        except Exception:
            continue

    return None


def _read_transcript(word_file):
    """Read transcript segments; absent/unreadable JSON provides no text supervision."""
    if word_file is None or str(word_file) == '' or not str(word_file).endswith('.json'):
        return []
    try:
        with open(word_file, 'r', encoding='utf-8') as handle:
            meta = json.load(handle)
    except Exception:
        return []
    segments = meta.get('metadata:transcript')
    return segments if isinstance(segments, list) else []


def _iter_transcript_segments(word_file):
    """Read timed transcript segments, falling back to their word timestamps."""
    vads = _read_transcript(word_file)

    segs = []
    for seg in vads:
        if not isinstance(seg, dict):
            continue
        s = seg.get('start')
        e = seg.get('end')
        if s is None or e is None:
            words = seg.get('words') or []
            if isinstance(words, list) and len(words) > 0:
                starts = [w.get('start') for w in words if isinstance(w, dict) and w.get('start') is not None]
                ends = [w.get('end') for w in words if isinstance(w, dict) and w.get('end') is not None]
                if starts and ends:
                    s = min(starts)
                    e = max(ends)
        try:
            s = float(s)
            e = float(e)
        except Exception:
            continue
        if e > s:
            segs.append((s, e))
    return segs


def _iter_transcript_words(word_file):
    """Read nonempty transcript words and order them by end time."""
    transcripts = _read_transcript(word_file)

    words = []
    for seg in transcripts:
        if not isinstance(seg, dict):
            continue
        for word_info in seg.get("words") or []:
            if not isinstance(word_info, dict):
                continue
            word = word_info.get("word")
            end = word_info.get("end")
            if not isinstance(word, str) or word.strip() == "" or end is None:
                continue
            try:
                end = float(end)
            except Exception:
                continue
            words.append({
                "word": word.strip(),
                "end": end,
            })
    words.sort(key=lambda item: item["end"])
    return words


def _infer_listener_word_file(audio_file_listen: str):
    """Infer the transcript JSON path beside a listener audio file."""
    if not isinstance(audio_file_listen, str) or audio_file_listen.strip() == "":
        return ""
    if audio_file_listen.endswith("_16k.wav"):
        return audio_file_listen[: -len("_16k.wav")] + ".json"
    base, _ = os.path.splitext(audio_file_listen)
    return base + ".json"


def _select_aligned_segment(word_file, alignment_score, threshold: float, min_len_sec: float):
    """Randomly choose an aligned transcript segment of sufficient duration."""
    segs = _iter_transcript_segments(word_file)
    if not segs:
        return None

    scores = None
    if isinstance(alignment_score, (list, tuple)):
        scores = list(alignment_score)

    if not isinstance(scores, list) or len(scores) == 0:
        return None

    th = float(threshold)
    min_len_sec = float(min_len_sec)

    cands = []
    for i, (s, e) in enumerate(segs):
        if i >= len(scores):
            break
        try:
            sc = float(scores[i])
        except Exception:
            sc = 0.0
        if not (sc > th):
            continue
        if (e - s) < min_len_sec:
            continue
        cands.append((s, e))

    if len(cands) == 0:
        return None
    return random.choice(cands)














def _extract_speaker_id(audio_file: str) -> str:
    """Extract the speaker identifier from a pose filename."""
    if not isinstance(audio_file, str) or audio_file.strip() == "":
        return ""
    base_name = os.path.basename(audio_file)
    match = re.search(r"(?:^|_)P([^_/.]+)", base_name)
    if match is not None:
        return match.group(1)
    match = re.search(r"P([^_/.]+)", base_name)
    return match.group(1) if match is not None else ""


def _build_system_prompt(base_prompt: str, speaker_id: str) -> str:
    """Build task instructions with optional speaker identity conditioning."""
    speaker_id = str(speaker_id).strip()
    if speaker_id == "":
        return base_prompt
    return f"{base_prompt}. Speaker ID :{speaker_id}."

def _resolve_flame_file(sample, output_suffix: str, file_key: str = "flame_file") -> str:
    """Resolve the explicit FLAME path or infer it from the pose filename."""
    flame_file = sample.get(file_key)
    if flame_file:
        return os.path.abspath(flame_file)

    pose_file = sample.get("pose_file", "")
    if not pose_file:
        return ""

    pose_root, pose_ext = os.path.splitext(pose_file)
    if pose_ext:
        return os.path.abspath(pose_root + output_suffix)
    return os.path.abspath(pose_file + output_suffix)


def _describe_generation_target(generation_target: str):
    """Return the task's human-readable gesture/expression target description."""
    generation_target = str(generation_target).strip().lower()
    if generation_target in {"gesture_expression", "expression_gesture"}:
        return "co-speech gesture and facial expression", "gesture and facial expression"
    if generation_target == "expression":
        return "co-speech facial expression", "facial expression"
    return "co-speech gesture", "gesture"


def _is_flame_motion_representation(pose_rep: str) -> bool:
    """Recognize the supported FLAME representation aliases."""
    return str(pose_rep).strip().lower() in {
        "flame",
        "flame_motioncode",
        "expression",
        "facial_expression",
    }


def _task_has_asr(task_type: str) -> bool:
    """Return whether the task includes text supervision."""
    return str(task_type).strip().lower() in {
        "asr",
        "asr_gesture",
        "asr_expression",
        "asr_gesture_expression",
    }


def _task_has_gesture(task_type: str) -> bool:
    """Return whether the task includes gesture supervision."""
    return str(task_type).strip().lower() in {
        "gesture",
        "gesture_expression",
        "asr_gesture",
        "asr_gesture_expression",
    }


def _task_has_expression(task_type: str) -> bool:
    """Return whether the task includes expression supervision."""
    return str(task_type).strip().lower() in {
        "expression",
        "gesture_expression",
        "asr_expression",
        "asr_gesture_expression",
    }


def _resolve_motion_target_flags(task_type: str, pose_rep: str):
    """Resolve gesture/expression targets, including legacy FLAME gesture tasks."""
    task_type = str(task_type).strip().lower()
    legacy_expression_mode = (
        _is_flame_motion_representation(pose_rep)
        and task_type in {"gesture", "asr_gesture"}
    )
    has_gesture_target = _task_has_gesture(task_type) and (not legacy_expression_mode)
    has_expression_target = _task_has_expression(task_type) or legacy_expression_mode
    return legacy_expression_mode, has_gesture_target, has_expression_target


def _build_generation_target_name(has_gesture: bool, has_expression: bool) -> str:
    """Name the motion target used in the conditioning prompt."""
    if has_gesture and has_expression:
        return "gesture_expression"
    if has_expression:
        return "expression"
    return "gesture"

class Gesture_Dataset(Dataset):
    _SMPLH_UPPER_JOINTS = UPPER_JOINTS
    _FACTORIZED_GESTURE_PARTS = FACTORIZED_GESTURE_PARTS
    _FACTORIZED_UPPER_INDICES = FACTORIZED_UPPER_INDICES
    _MOTION_TOKENIZER_CONFIGS = MOTION_TOKENIZER_CONFIGS

    def __init__(self, args):
        """Configure task targets, motion tokenizers, sampling metadata and data caches."""
        self.args = args
        self.pose_rep = str(getattr(self.args, "pose_rep", "smplh")).lower()
        self.task_type = str(getattr(self.args, "task_type", "gesture")).lower()
        if self.task_type not in {
            "gesture",
            "expression",
            "gesture_expression",
            "asr",
            "asr_gesture",
            "asr_expression",
            "asr_gesture_expression",
        }:
            raise ValueError(
                "`task_type` must be one of 'gesture', 'expression', 'gesture_expression', "
                "'asr', 'asr_gesture', 'asr_expression', or 'asr_gesture_expression'."
            )

        (
            self.legacy_expression_mode,
            self.has_gesture_target,
            self.has_expression_target,
        ) = _resolve_motion_target_flags(self.task_type, self.pose_rep)
        self.has_motion_target = self.has_gesture_target or self.has_expression_target
        self.has_asr_target = _task_has_asr(self.task_type)
        self.generation_target = _build_generation_target_name(
            self.has_gesture_target,
            self.has_expression_target,
        )

        self.motion_tokenizer_type = None
        self.max_motion_codebooks = 0
        self.max_expression_codebooks = 6 if self.has_expression_target else 0
        self.used_num_motion_codebooks = int(getattr(self.args, "used_num_motion_codebooks", 6))
        raw_used_num_expression_codebooks = getattr(
            self.args,
            "used_num_expression_codebooks",
            getattr(self.args, "used_num_flame_codebooks", None),
        )
        if raw_used_num_expression_codebooks is None:
            raw_used_num_expression_codebooks = self.used_num_motion_codebooks
        self.used_num_expression_codebooks = int(raw_used_num_expression_codebooks)
        if self.has_gesture_target:
            self.motion_tokenizer_type = self._resolve_motion_tokenizer_type()
            self.max_motion_codebooks = (
                12
                if self.motion_tokenizer_type in {"upper_hands", "factorized_gesture_5"}
                else 6
            )
            if self.used_num_motion_codebooks <= 0 or self.used_num_motion_codebooks > self.max_motion_codebooks:
                raise ValueError(
                    f"`used_num_motion_codebooks` must be in [1, {self.max_motion_codebooks}] for "
                    f"motion_tokenizer_type='{self.motion_tokenizer_type}'."
                )
        if self.has_expression_target:
            if self.used_num_expression_codebooks <= 0 or self.used_num_expression_codebooks > self.max_expression_codebooks:
                raise ValueError(
                    f"`used_num_expression_codebooks` must be in [1, {self.max_expression_codebooks}] "
                    "when expression generation is enabled."
                )

        if self.has_gesture_target and "smplh" not in self.pose_rep:
            raise ValueError("Gesture_Dataset now only supports smplh pose representations.")

        self.ori_length = int(self.args.pose_length)
        self.dialog = bool(getattr(self.args, "dialog", False))
        self.listener_fusion_delta = bool(getattr(self.args, "listener_fusion_delta", False))
        self.use_listener_audio = self.dialog or self.listener_fusion_delta
        self.dialog_text = bool(getattr(self.args, "dialog_text", False)) and self.dialog
        self.inline_imitation_boundary_loss_weight = float(
            getattr(self.args, "inline_imitation_boundary_loss_weight", 1.0)
        )
        self.inline_imitation_core_loss_weight = float(
            getattr(self.args, "inline_imitation_core_loss_weight", 1.0)
        )
        self.interleave_motion_parts = bool(getattr(self.args, "interleave_motion_parts", False))
        self.add_speaker_id = bool(getattr(self.args, "add_speaker_id", False))
        self.speaker_id_type = str(getattr(self.args, "speaker_id_type", "text")).strip().lower()
        if self.speaker_id_type not in {"text", "motion"}:
            raise ValueError("`speaker_id_type` must be either 'text' or 'motion'.")
        raw_motion_style_length = getattr(self.args, "motion_style_length", None)
        self.motion_style_length = self.ori_length if raw_motion_style_length is None else int(raw_motion_style_length)
        self.inline_exemplar_enabled = bool(getattr(self.args, "inline_exemplar_enabled", False))
        self.inline_exemplar_copy_mode = str(
            getattr(self.args, "inline_exemplar_copy_mode", "learned")
        )
        if self.inline_exemplar_copy_mode not in {"learned", "hard_core"}:
            raise ValueError("inline_exemplar_copy_mode must be learned or hard_core")
        if self.motion_style_length <= 0:
            raise ValueError("`motion_style_length` must be positive.")
        if self.add_speaker_id and self.speaker_id_type == "motion" and not self.has_gesture_target:
            raise ValueError("`speaker_id_type=motion` requires a gesture motion target.")
        self.audio_sr = int(getattr(self.args, "audio_sr", getattr(self.args, "audio_fps", 16000)))
        self.audio_fps = int(getattr(self.args, "audio_fps", self.audio_sr))
        self.pose_fps = int(self.args.pose_fps)
        self.flame_fps = int(getattr(self.args, "flame_fps", 30))
        self.motion_downsample_factor = int(getattr(self.args, "motion_downsample_factor", 4))
        self.window_sec = float(getattr(self.args, "window_sec", 0.8))
        self.flame_output_suffix = str(getattr(self.args, "flame_output_suffix", "_flame.npz"))
        self.flame_file_key = str(getattr(self.args, "flame_file_key", "flame_file"))
        self.flame_motion_key = str(getattr(self.args, "flame_motion_key", "motioncode"))
        self.flame_stats_path = str(
            getattr(self.args, "flame_stats_path", DEFAULT_FLAME_STATS_PATH)
        )
        self.motion_mean = None
        self.motion_std = None
        if self.has_expression_target:
            motion_mean, motion_std = self._load_flame_stats(self.flame_stats_path)
            self.motion_mean = torch.from_numpy(motion_mean).float()
            self.motion_std = torch.from_numpy(motion_std).float()

        self.small_dataset_threshold = int(getattr(self.args, "small_dataset_threshold", 1024))
        self.enable_small_dataset_cache = bool(getattr(self.args, "enable_small_dataset_cache", True))
        self.audio_cache_size = int(getattr(self.args, "audio_cache_size", 1024))
        self.pose_cache_size = int(getattr(self.args, "pose_cache_size", 1024))
        self.transcript_cache_size = int(getattr(self.args, "transcript_cache_size", 4096))
        self.enable_runtime_cache = bool(getattr(self.args, "enable_runtime_cache", False))
        self.motion_token_cache_dir = str(getattr(self.args, "motion_token_cache_dir", "")).strip()
        self.use_precomputed_motion_tokens = self.motion_token_cache_dir != ""
        if self.use_precomputed_motion_tokens and not os.path.isdir(self.motion_token_cache_dir):
            raise ValueError(f"Motion-token cache directory does not exist: {self.motion_token_cache_dir}")
        self.precompute_sample_meta = bool(getattr(self.args, "precompute_sample_meta", False))
        self.max_sampling_retries = max(1, int(getattr(self.args, "max_sampling_retries", 100)))
        self.sampling_warning_limit = max(0, int(getattr(self.args, "sampling_warning_limit", 10)))
        self._sampling_warning_count = 0

        self.gesture_motion_steps_per_window = max(
            1,
            int(round(self.window_sec * float(self.pose_fps) / float(self.motion_downsample_factor))),
        )
        self.expression_motion_steps_per_window = max(
            1,
            int(round(self.window_sec * float(self.flame_fps) / float(self.motion_downsample_factor))),
        )
        self.total_motion_steps_per_window = (
            self.gesture_motion_steps_per_window
            if self.has_gesture_target
            else self.expression_motion_steps_per_window
        )
        if self.interleave_motion_parts:
            if self.has_gesture_target and self.motion_tokenizer_type == "upper_hands":
                self.total_motion_steps_per_window = self.gesture_motion_steps_per_window * 2
                if self.has_expression_target:
                    self.total_motion_steps_per_window += self.expression_motion_steps_per_window
            else:
                raise ValueError("`interleave_motion_parts` only supports motion_tokenizer_type=upper_hands gesture targets.")
        self.total_num_motion_codebooks = self._get_total_num_motion_codebooks()

        self._audio_cache = None
        self._pose_cache = None
        self._flame_cache = None
        self._audio_len_cache = {}
        self._audio_info_cache = {}
        self._motion_len_cache = {}
        self._flame_len_cache = {}
        self._transcript_word_cache = None

        with open(args.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)["data"]
        self.data = [{key: resolve_path(value) for key, value in row.items()} for row in self.data]

        self._sample_meta = None
        self._small_dataset_mode = (
            self.enable_small_dataset_cache and len(self.data) <= self.small_dataset_threshold
        )
        if len(self.data) > self.small_dataset_threshold and not self.precompute_sample_meta:
            self.valid_indices = range(len(self.data))
        else:
            self._sample_meta = [self._build_sample_meta(sample) for sample in self.data]
            self.valid_indices = [i for i, meta in enumerate(self._sample_meta) if meta["static_valid"]]

        if self._small_dataset_mode or self.enable_runtime_cache:
            self._audio_cache = OrderedDict()
            self._pose_cache = OrderedDict()
            self._flame_cache = OrderedDict()
            self._transcript_word_cache = OrderedDict()
        elif self.transcript_cache_size > 0:
            self._transcript_word_cache = OrderedDict()

        if self._small_dataset_mode:
            self._warmup_small_dataset_cache()

        if len(self.valid_indices) == 0:
            raise RuntimeError("No valid training samples found after static filtering.")

        self.motion_tokenizers = {}
        if self.has_motion_target and not self.use_precomputed_motion_tokens:
            if self.has_gesture_target:
                body_mean = torch.from_numpy(
                    np.load(f"{pathlib.Path(__file__).resolve().parent}/assets/motion_stats/seamless_naturalistic_improvised_train_body_mean_filtered2.npy")
                ).float()
                body_std = torch.from_numpy(
                    np.load(f"{pathlib.Path(__file__).resolve().parent}/assets/motion_stats/seamless_naturalistic_improvised_train_body_std_filtered2.npy")
                ).float()
                self.mean_left_hand = torch.from_numpy(
                    np.load(f"{pathlib.Path(__file__).resolve().parent}/assets/motion_stats/seamless_naturalistic_improvised_train_left_hand_mean_filtered2.npy")
                ).float()
                self.std_left_hand = torch.from_numpy(
                    np.load(f"{pathlib.Path(__file__).resolve().parent}/assets/motion_stats/seamless_naturalistic_improvised_train_left_hand_std_filtered2.npy")
                ).float()
                self.mean_right_hand = torch.from_numpy(
                    np.load(f"{pathlib.Path(__file__).resolve().parent}/assets/motion_stats/seamless_naturalistic_improvised_train_right_hand_mean_filtered2.npy")
                ).float()
                self.std_right_hand = torch.from_numpy(
                    np.load(f"{pathlib.Path(__file__).resolve().parent}/assets/motion_stats/seamless_naturalistic_improvised_train_right_hand_std_filtered2.npy")
                ).float()
                self.mean_transv = torch.from_numpy(
                    np.load(f"{pathlib.Path(__file__).resolve().parent}/assets/motion_stats/seamless_naturalistic_improvised_train_translation_mean_filtered2.npy")
                ).float()
                self.std_transv = torch.from_numpy(
                    np.load(f"{pathlib.Path(__file__).resolve().parent}/assets/motion_stats/seamless_naturalistic_improvised_train_translation_std_filtered2.npy")
                ).float()

                upper_6d_indices = [
                    idx
                    for joint_idx in self._SMPLH_UPPER_JOINTS
                    for idx in range(6 * joint_idx, 6 * (joint_idx + 1))
                ]
                self.mean_upper = body_mean[upper_6d_indices]
                self.std_upper = body_std[upper_6d_indices]

            from models.vq.precision import configure_encoding_precision
            self.motion_tokenizers = {
                part: configure_encoding_precision(model, getattr(args, 'rvq_encoder_dtype', 'float32'),
                                                   getattr(args, 'rvq_quantizer_distance_dtype', 'float32'))
                for part, model in self._create_motion_tokenizers().items()
            }
            for tokenizer in self.motion_tokenizers.values():
                for param in tokenizer.parameters():
                    param.requires_grad = False
                tokenizer.eval()
    def _resolve_motion_tokenizer_type(self):
        """Validate the requested tokenizer family and retain the legacy default."""
        tokenizer_type = getattr(self.args, "motion_tokenizer_type", "")
        if not tokenizer_type:
            return "upper_hands_trans"

        if tokenizer_type not in {"upper_hands_trans", "upper_hands", "factorized_gesture_5"}:
            raise ValueError(
                "`motion_tokenizer_type` must be 'upper_hands_trans', 'upper_hands', "
                "or 'factorized_gesture_5'."
            )
        return tokenizer_type

    def _resolve_motion_tokenizer_paths(self):
        """Collect checkpoint paths required by the selected motion tokenizer family."""
        tokenizer_paths = {}
        if self.has_gesture_target:
            if self.motion_tokenizer_type == "upper_hands_trans":
                checkpoint_path = getattr(self.args, "vqvae_upper_hands_trans_path", "")
                if not checkpoint_path:
                    raise ValueError("`vqvae_upper_hands_trans_path` is required for single-tokenizer motion encoding.")
                tokenizer_paths["upper_hands_trans"] = checkpoint_path
            elif self.motion_tokenizer_type == "upper_hands":
                upper_path = getattr(self.args, "vqvae_upper_path", "")
                hands_path = getattr(self.args, "vqvae_hands_path", "")
                if not upper_path or not hands_path:
                    raise ValueError(
                    "`vqvae_upper_path` and `vqvae_hands_path` are required when "
                    "`motion_tokenizer_type='upper_hands'`."
                )
                tokenizer_paths["upper"] = upper_path
                tokenizer_paths["hands"] = hands_path
            elif self.motion_tokenizer_type == "factorized_gesture_5":
                for body_part in self._FACTORIZED_GESTURE_PARTS:
                    checkpoint_path = getattr(self.args, f"vqvae_{body_part}_path", "")
                    if not checkpoint_path:
                        raise ValueError(
                            f"`vqvae_{body_part}_path` is required when "
                            "`motion_tokenizer_type='factorized_gesture_5'`."
                        )
                    tokenizer_paths[body_part] = checkpoint_path

        if self.has_expression_target:
            checkpoint_path = getattr(self.args, "vqvae_flame_path", "")
            if not checkpoint_path:
                raise ValueError("`vqvae_flame_path` is required for flame motioncode encoding.")
            tokenizer_paths["flame_motioncode"] = checkpoint_path

        return tokenizer_paths

    def _load_checkpoint_state_dict(self, checkpoint_path):
        """Extract tokenizer weights from a raw or wrapped checkpoint."""
        if checkpoint_path.endswith(".safetensors"):
            return load_file(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        return ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt

    def _build_motion_tokenizer_cfg(self, body_part):
        """Build the RVQVAE configuration for one motion representation."""
        return build_motion_tokenizer_cfg(body_part)

    def _create_motion_tokenizer(self, body_part, checkpoint_path):
        """Load the checkpoint's causal or original RVQ architecture."""
        if checkpoint_path.endswith(".safetensors"):
            loaded_checkpoint = {"net": load_file(checkpoint_path)}
        else:
            loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        context_key = ("rvq_expression_context_frames" if body_part == "flame_motioncode"
                       else "rvq_gesture_context_frames")
        context_frames = getattr(self.args, context_key, 0)
        if loaded_checkpoint.get("model_type") == "factorized_flame_rvq_4plus2":
            from models.vq.factorized_flame_model import build_factorized_flame_rvqvae
            return build_factorized_flame_rvqvae(loaded_checkpoint).set_context_frames(context_frames)
        model_cfg = self._build_motion_tokenizer_cfg(body_part)
        model_cfg.causal = bool(loaded_checkpoint.get("args", {}).get("causal", False))
        model = build_motion_rvqvae(model_cfg)
        model.load_state_dict(loaded_checkpoint.get("net", loaded_checkpoint), strict=True)
        return model.set_context_frames(context_frames)

    def _create_motion_tokenizers(self):
        """Create the gesture and expression tokenizers required by this task."""
        return {
            part: self._create_motion_tokenizer(part, path)
            for part, path in self._resolve_motion_tokenizer_paths().items()
        }

    def _get_gesture_output_codebook_count(self):
        """Count gesture heads, accounting for separate upper-body and hand tokenizers."""
        if not self.has_gesture_target:
            return 0
        if self.motion_tokenizer_type == "upper_hands":
            return min(6, self.used_num_motion_codebooks) * 2
        return self.used_num_motion_codebooks

    def _get_expression_output_codebook_count(self):
        """Count expression heads selected from the FLAME tokenizer."""
        if not self.has_expression_target:
            return 0
        return min(6, self.used_num_expression_codebooks)

    def _get_total_num_motion_codebooks(self):
        """Count all active gesture and expression codebooks."""
        return self._get_gesture_output_codebook_count() + self._get_expression_output_codebook_count()



    def _get_expression_target_length(self, pose_num_frames=None):
        """Convert the pose crop duration to an expression frame count."""
        if pose_num_frames is None:
            pose_num_frames = self.ori_length
        return max(
            1,
            int(round(float(pose_num_frames) * float(self.flame_fps) / float(self.pose_fps))),
        )

    def _expression_frames_to_pose_frames(self, expression_num_frames):
        """Convert expression-frame counts to the pose-frame timeline."""
        return int(math.floor(float(expression_num_frames) * float(self.pose_fps) / float(self.flame_fps)))

    def _convert_expression_spans_to_pose_timeline(self, expression_spans):
        """Convert expression validity spans to pose-frame coordinates."""
        pose_spans = []
        for span_start, span_end in expression_spans:
            pose_start = int(math.ceil(float(span_start) * float(self.pose_fps) / float(self.flame_fps)))
            pose_end = int(math.floor(float(span_end) * float(self.pose_fps) / float(self.flame_fps)))
            if pose_end > pose_start:
                pose_spans.append((pose_start, pose_end))
        return pose_spans

    def _warn_sampling_issue(self, message):
        """Report sampling failures up to the configured warning limit."""
        if self._sampling_warning_count >= self.sampling_warning_limit:
            return
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        self._sampling_warning_count += 1

    def _get_expression_slice_indices(self, start_idx, fin_idx):
        """Convert pose crop endpoints to expression-frame indices."""
        expression_start = int(round(float(start_idx) * float(self.flame_fps) / float(self.pose_fps)))
        expression_length = self._get_expression_target_length(fin_idx - start_idx)
        expression_end = expression_start + expression_length
        return expression_start, expression_end

    def _build_asr_text_blocks(self, word_file, segment_start_sec: float, num_windows: int, window_sec: float, eot_token: str):
        """Assign transcript words to streaming windows by their end timestamps."""
        blocks = [[] for _ in range(num_windows)]
        eps = 1e-8
        segment_end_sec = float(segment_start_sec) + float(num_windows) * float(window_sec)

        for word_info in self._get_transcript_words(word_file):
            end_sec = float(word_info["end"])
            if end_sec < float(segment_start_sec):
                continue
            if end_sec > segment_end_sec + eps:
                continue
            block_idx = int(max(0.0, end_sec - float(segment_start_sec) - eps) // float(window_sec))
            block_idx = min(max(block_idx, 0), num_windows - 1)
            blocks[block_idx].append(word_info["word"])

        out = []
        for words in blocks:
            if len(words) == 0:
                out.append(eot_token)
            else:
                out.append(" ".join(words) + eot_token)
        return out


    def _source_motion_step_for_mask_row(self, row_idx: int):
        """Map a possibly interleaved mask row back to its source gesture step."""
        source_step = int(row_idx)
        if self.interleave_motion_parts:
            row_in_window = int(row_idx) % self.total_motion_steps_per_window
            window_idx = int(row_idx) // self.total_motion_steps_per_window
            if row_in_window < self.gesture_motion_steps_per_window * 2:
                source_step = window_idx * self.gesture_motion_steps_per_window + row_in_window // 2
            else:
                source_step = window_idx * self.gesture_motion_steps_per_window + min(
                    row_in_window - self.gesture_motion_steps_per_window * 2,
                    self.gesture_motion_steps_per_window - 1,
                )
        return source_step


    def _build_inline_imitation_motion_loss_mask(self, mask, metadata, motion_steps: int):
        """Apply exemplar core and boundary loss weights to the motion timeline."""
        if metadata is None or (
            self.inline_imitation_boundary_loss_weight == 1.0
            and self.inline_imitation_core_loss_weight == 1.0
        ):
            return mask
        if mask is None:
            mask = torch.ones(int(motion_steps), dtype=torch.float32)
        splice_start = int(metadata["target_splice_start_frame"])
        reference_length = int(metadata["reference_length_frames"])
        blend_frames = int(metadata["blend_frames"])
        core_start = splice_start + blend_frames
        core_end = splice_start + reference_length - blend_frames
        splice_end = splice_start + reference_length
        for row_idx in range(int(motion_steps)):
            source_step = self._source_motion_step_for_mask_row(row_idx)
            step_start = int(source_step) * int(self.motion_downsample_factor)
            step_end = step_start + int(self.motion_downsample_factor)
            if min(step_end, core_end) > max(step_start, core_start):
                if self.inline_exemplar_copy_mode == "hard_core":
                    mask[row_idx] = 0.0
                else:
                    mask[row_idx] = max(
                        float(mask[row_idx]), self.inline_imitation_core_loss_weight
                    )
            elif min(step_end, splice_end) > max(step_start, splice_start):
                mask[row_idx] = max(
                    float(mask[row_idx]), self.inline_imitation_boundary_loss_weight
                )
        return mask


    def _getitem_asr(self, index):
        """Sample an aligned audio/text crop for a task without motion targets."""
        alignment_th = _get_alignment_score_threshold(self.args)
        target_length = self.ori_length
        audio_seg_len = math.floor(target_length * self.audio_fps / self.pose_fps)
        samples_per_window = int(round(float(getattr(self.args, "window_sec", 0.8)) * self.audio_fps))
        window_sec = float(getattr(self.args, "window_sec", 0.8))
        eot_token = str(getattr(self.args, "asr_eot_token", "<|endoftext|>"))

        if audio_seg_len <= 0 or samples_per_window <= 0:
            raise ValueError("Invalid ASR segment configuration.")

        for _ in range(10):
            sample = self.data[index]
            audio_file = sample["audio_file"]
            audio_file_listen = sample.get("audio_file_listen", "")
            word_file = sample.get("word_file", "")
            word_file_listen = _infer_listener_word_file(audio_file_listen) if self.dialog_text else ""
            speaker_id = _extract_speaker_id(audio_file) if self.add_speaker_id and self.speaker_id_type == "text" else ""

            selected_segment = None
            if alignment_th is not None:
                selected_segment = _select_aligned_segment(
                    word_file,
                    sample.get("alignment_score", None),
                    threshold=alignment_th,
                    min_len_sec=float(target_length) / float(self.pose_fps),
                )
                if selected_segment is None:
                    index = random.randint(0, len(self.data) - 1)
                    continue

            audio_full, _ = librosa.load(audio_file, sr=self.audio_sr)
            if self.dialog and audio_file_listen and os.path.exists(audio_file_listen):
                audio_listen_full, _ = librosa.load(audio_file_listen, sr=self.audio_sr)
            elif self.dialog:
                audio_listen_full = np.zeros_like(audio_full)
            else:
                audio_listen_full = None

            max_audio_samples = int(audio_full.shape[-1])
            if self.dialog and audio_listen_full is not None:
                max_audio_samples = min(max_audio_samples, int(audio_listen_full.shape[-1]))
            if max_audio_samples < audio_seg_len:
                index = random.randint(0, len(self.data) - 1)
                continue

            max_start_sample = max_audio_samples - audio_seg_len
            start_min_sample = 0
            start_max_sample = max_start_sample
            if selected_segment is not None:
                seg_start_s, seg_end_s = selected_segment
                start_min_sample = max(start_min_sample, int(math.ceil(float(seg_start_s) * float(self.audio_fps))))
                start_max_sample = min(start_max_sample, int(math.floor(float(seg_end_s) * float(self.audio_fps))) - audio_seg_len)

            if start_max_sample < start_min_sample:
                index = random.randint(0, len(self.data) - 1)
                continue

            audio_start = random.randint(int(start_min_sample), int(start_max_sample))
            audio_end = audio_start + audio_seg_len
            audio = torch.from_numpy(audio_full[audio_start:audio_end]).float()
            if self.dialog and audio_listen_full is not None:
                audio_listen = torch.from_numpy(audio_listen_full[audio_start:audio_end]).float()
            else:
                audio_listen = None

            num_windows = int(audio.shape[0] // samples_per_window)
            if num_windows <= 0:
                index = random.randint(0, len(self.data) - 1)
                continue

            valid_num_samples = num_windows * samples_per_window
            audio = audio[:valid_num_samples]
            if audio_listen is not None:
                audio_listen = audio_listen[:valid_num_samples]

            segment_start_sec = float(audio_start) / float(self.audio_fps)
            asr_text_blocks = self._build_asr_text_blocks(
                word_file=word_file,
                segment_start_sec=segment_start_sec,
                num_windows=num_windows,
                window_sec=window_sec,
                eot_token=eot_token,
            )

            out = {
                "audio": audio,
                "asr_text_blocks": asr_text_blocks,
                "speaker_id": speaker_id,
            }
            if self.dialog_text:
                out["asr_text_blocks_listen"] = self._build_asr_text_blocks(
                    word_file=word_file_listen,
                    segment_start_sec=segment_start_sec,
                    num_windows=num_windows,
                    window_sec=window_sec,
                    eot_token=eot_token,
                )
            if self.dialog and audio_listen is not None:
                out["audio_listen"] = audio_listen
            return out

        raise RuntimeError("Failed to sample a valid ASR segment after multiple retries.")

    def _attach_asr_text_blocks_to_sample(self, sample, word_file, audio_file_listen, segment_start_sec: float, num_windows: int):
        """Attach speaker and optional listener text blocks to a sampled audio crop."""
        window_sec = float(getattr(self.args, "window_sec", 0.8))
        eot_token = str(getattr(self.args, "asr_eot_token", "<|endoftext|>"))
        sample["asr_text_blocks"] = self._build_asr_text_blocks(
            word_file=word_file,
            segment_start_sec=segment_start_sec,
            num_windows=num_windows,
            window_sec=window_sec,
            eot_token=eot_token,
        )
        if self.dialog_text:
            sample["asr_text_blocks_listen"] = self._build_asr_text_blocks(
                word_file=_infer_listener_word_file(audio_file_listen),
                segment_start_sec=segment_start_sec,
                num_windows=num_windows,
                window_sec=window_sec,
                eot_token=eot_token,
            )
        return sample

    def _cache_get(self, cache_obj, key):
        """Retrieve an LRU entry and promote it to the most recently used position."""
        if cache_obj is None:
            return None
        value = cache_obj.get(key)
        if value is None:
            return None
        cache_obj.move_to_end(key)
        return value

    def _cache_put(self, cache_obj, key, value, max_size):
        """Store an LRU entry and evict the oldest entries beyond the size limit."""
        if cache_obj is None or max_size <= 0:
            return
        cache_obj[key] = value
        cache_obj.move_to_end(key)
        while len(cache_obj) > max_size:
            cache_obj.popitem(last=False)

    def _get_transcript_words(self, word_file):
        """Read transcript words, reusing the runtime cache when enabled."""
        cached = self._cache_get(self._transcript_word_cache, word_file)
        if cached is not None:
            return cached
        words = _iter_transcript_words(word_file)
        self._cache_put(self._transcript_word_cache, word_file, words, self.transcript_cache_size)
        return words

    def _get_audio_info(self, audio_file):
        """Read audio sample rate and length, reusing cached metadata."""
        if not audio_file or not os.path.exists(audio_file):
            return None
        cached = self._audio_info_cache.get(audio_file)
        if cached is not None:
            return cached
        info = sf.info(audio_file)
        cached = (int(info.samplerate), int(info.frames))
        self._audio_info_cache[audio_file] = cached
        return cached

    def _fast_load_audio(self, audio_file):
        """Read and resample a complete audio file to the training sample rate."""
        audio_data, sr = sf.read(audio_file, dtype="float32", always_2d=False)
        if isinstance(audio_data, np.ndarray) and audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=1)
        audio_data = np.asarray(audio_data, dtype=np.float32)
        if sr != self.audio_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.audio_sr)
            audio_data = np.asarray(audio_data, dtype=np.float32)
        return audio_data

    def _fast_load_audio_segment(self, audio_file, audio_start, audio_end):
        """Read and resample the requested audio interval."""
        info = self._get_audio_info(audio_file)
        if info is None:
            return np.zeros(0, dtype=np.float32)
        sr, _ = info
        if sr != self.audio_sr:
            audio_full = self._fast_load_audio(audio_file)
            return audio_full[audio_start:audio_end]
        audio_data, _ = sf.read(
            audio_file,
            start=int(audio_start),
            stop=int(audio_end),
            dtype="float32",
            always_2d=False,
        )
        if isinstance(audio_data, np.ndarray) and audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=1)
        return np.asarray(audio_data, dtype=np.float32)

    def _get_audio_num_samples(self, audio_file):
        """Measure audio length at the configured training sample rate."""
        if not audio_file or not os.path.exists(audio_file):
            return 0
        cached = self._audio_len_cache.get(audio_file)
        if cached is not None:
            return cached

        info = self._get_audio_info(audio_file)
        if info is None:
            return 0
        sr, frames = info
        if sr == self.audio_sr:
            num_samples = int(frames)
        else:
            num_samples = int(round(frames * float(self.audio_sr) / float(sr)))

        self._audio_len_cache[audio_file] = num_samples
        return num_samples

    def _load_audio_full(self, audio_file):
        """Load a complete waveform through the optional runtime cache."""
        if not audio_file or not os.path.exists(audio_file):
            return np.zeros(0, dtype=np.float32)

        cached = self._cache_get(self._audio_cache, audio_file)
        if cached is not None:
            return cached

        audio_data = self._fast_load_audio(audio_file)
        self._cache_put(self._audio_cache, audio_file, audio_data, self.audio_cache_size)
        return audio_data

    def _load_audio_segment(self, audio_file, audio_start, audio_end, ref_audio=None):
        """Load an audio interval or provide a zero listener waveform when absent."""
        cached = self._cache_get(self._audio_cache, audio_file)
        if cached is not None and cached.shape[0] > 0:
            return cached[audio_start:audio_end], cached
        audio_segment = self._fast_load_audio_segment(audio_file, audio_start, audio_end)
        if audio_segment.shape[0] > 0:
            return audio_segment, None
        if ref_audio is None:
            return np.zeros(audio_end - audio_start, dtype=np.float32), np.zeros(0, dtype=np.float32)
        return np.zeros(audio_end - audio_start, dtype=ref_audio.dtype), np.zeros_like(ref_audio)

    @staticmethod
    def _intersect_valid_spans(left_spans, right_spans):
        """Intersect two collections of half-open validity intervals."""
        if left_spans is None:
            return list(right_spans)
        if right_spans is None:
            return list(left_spans)

        out = []
        i = 0
        j = 0
        left_spans = sorted(left_spans)
        right_spans = sorted(right_spans)
        while i < len(left_spans) and j < len(right_spans):
            left_start, left_end = left_spans[i]
            right_start, right_end = right_spans[j]
            start = max(int(left_start), int(right_start))
            end = min(int(left_end), int(right_end))
            if end > start:
                out.append((start, end))
            if left_end <= right_end:
                i += 1
            else:
                j += 1
        return out

    def _resolve_gesture_source_file(self, sample):
        """Resolve the pose source used for gesture supervision."""
        return sample.get("pose_file", "")

    def _resolve_expression_source_file(self, sample):
        """Resolve the FLAME source used for expression supervision."""
        return _resolve_flame_file(sample, self.flame_output_suffix, self.flame_file_key)

    def _resolve_motion_source_files(self, sample):
        """Resolve both motion sources required by the current task."""
        return {
            "gesture": self._resolve_gesture_source_file(sample) if self.has_gesture_target else "",
            "expression": self._resolve_expression_source_file(sample) if self.has_expression_target else "",
        }

    def _load_pose_record_from_npz(self, pose_file):
        """Read pose arrays and optional validity masks from an NPZ file."""
        pose_record = {}
        with np.load(pose_file) as pose_data:
            for canonical_key, (plain_key, namespaced_key) in POSE_KEY_ALIASES.items():
                resolved_key = _pick_key(pose_data, plain_key, namespaced_key)
                pose_record[canonical_key] = np.asarray(pose_data[resolved_key])
        return pose_record

    def _load_pose_record(self, pose_file):
        """Load a pose record through the optional runtime cache."""
        cached = self._cache_get(self._pose_cache, pose_file)
        if cached is not None:
            return cached

        pose_record = self._load_pose_record_from_npz(pose_file)

        self._cache_put(self._pose_cache, pose_file, pose_record, self.pose_cache_size)
        return pose_record

    def _load_flame_record(self, flame_file):
        """Load FLAME motion codes and validity masks through the runtime cache."""
        cached = self._cache_get(self._flame_cache, flame_file)
        if cached is not None:
            return cached

        flame_record = {}
        with np.load(flame_file) as flame_data:
            if self.flame_motion_key not in flame_data:
                raise KeyError(f"Cannot find '{self.flame_motion_key}' in {flame_file}")
            motioncode = self._select_unils_motioncode(flame_data[self.flame_motion_key])
            valid_mask = flame_data["valid_mask"] if "valid_mask" in flame_data else None

        total_frames = motioncode.shape[0]
        if valid_mask is None:
            valid_mask = np.ones(total_frames, dtype=np.bool_)
        else:
            valid_mask = np.asarray(valid_mask).astype(np.bool_)
            if valid_mask.shape[0] != total_frames:
                aligned_frames = min(total_frames, valid_mask.shape[0])
                motioncode = motioncode[:aligned_frames]
                valid_mask = valid_mask[:aligned_frames]

        if motioncode.shape[1] != self.motion_mean.shape[0]:
            raise ValueError(
                f"Stats dim {self.motion_mean.shape[0]} mismatches flame feature dim {motioncode.shape[1]} "
                f"in {flame_file}"
            )

        flame_record["flame:motioncode"] = motioncode.astype(np.float32, copy=False)
        flame_record["flame:valid_mask"] = valid_mask
        self._cache_put(self._flame_cache, flame_file, flame_record, self.pose_cache_size)
        return flame_record

    def _load_pose_arrays(self, pose_file, required_keys):
        """Return the pose and translation arrays for gesture feature construction."""
        pose_record = self._load_pose_record(pose_file)
        missing_keys = [key for key in required_keys if key not in pose_record]
        if missing_keys:
            raise KeyError(f"Missing keys in pose record for {pose_file}: {missing_keys}")
        return {key: pose_record[key] for key in required_keys}

    def _get_motion_num_frames(self, pose_file):
        """Read the pose length from its array header, with a full-load fallback."""
        cached = self._motion_len_cache.get(pose_file)
        if cached is not None:
            return cached

        try:
            shape = _read_npz_array_shape(pose_file, "body_pose", "smplh:body_pose")
            motion_frames = int(shape[0])
        except Exception:
            pose_record = self._load_pose_record(pose_file)
            motion_frames = int(pose_record["smplh:body_pose"].shape[0])
        self._motion_len_cache[pose_file] = motion_frames
        return motion_frames

    def _get_flame_motion_num_frames(self, flame_file):
        """Read the expression length from its array header, with a full-load fallback."""
        cached = self._flame_len_cache.get(flame_file)
        if cached is not None:
            return cached

        flame_record = self._load_flame_record(flame_file)
        motion_frames = int(flame_record["flame:motioncode"].shape[0])
        self._flame_len_cache[flame_file] = motion_frames
        return motion_frames


    def _build_sampling_info(self, sample):
        """Intersect gesture/face validity with whole seconds shared by both audio streams.

        Valid spans are half-open pose-frame intervals; sampling ranges derived
        from them later use inclusive crop-start endpoints.
        """
        motion_files = self._resolve_motion_source_files(sample)
        pose_file, flame_file = motion_files['gesture'], motion_files['expression']
        audio_file, listener_file = sample['audio_file'], sample.get('audio_file_listen', '')
        has_listener = bool(listener_file and os.path.exists(listener_file))
        info = {
            'motion_frames': 0, 'max_motion_frames': 0, 'valid_spans': [],
            'speaker_id': (_extract_speaker_id(audio_file)
                           if self.add_speaker_id and self.speaker_id_type == 'text' else ''),
            'has_listener_audio': has_listener,
        }
        required = [(self.has_gesture_target, pose_file), (self.has_expression_target, flame_file)]
        if any(enabled and (not path or not os.path.exists(path)) for enabled, path in required):
            return info
        motion_frames = valid_spans = None
        if self.has_gesture_target:
            motion_frames = self._get_motion_num_frames(pose_file)
            valid_spans = self._intersect_valid_spans(valid_spans, [(0, motion_frames)])
        if self.has_expression_target:
            face_frames = self._get_flame_motion_num_frames(flame_file)
            face_record = self._load_flame_record(flame_file)
            face_spans = self._find_valid_spans(face_record['flame:valid_mask'],
                                              min_length=self._get_expression_target_length())
            face_spans = self._convert_expression_spans_to_pose_timeline(face_spans)
            face_frames = self._expression_frames_to_pose_frames(face_frames)
            motion_frames = face_frames if motion_frames is None else min(motion_frames, face_frames)
            valid_spans = self._intersect_valid_spans(valid_spans, face_spans)
        audio_samples = self._get_audio_num_samples(audio_file)
        listener_samples = (self._get_audio_num_samples(listener_file)
                            if self.dialog and has_listener else audio_samples)
        seconds = min(audio_samples // self.audio_fps, listener_samples // self.audio_fps,
                      motion_frames // self.pose_fps)
        max_frames = seconds * self.pose_fps
        info.update(motion_frames=motion_frames, max_motion_frames=max_frames,
                    valid_spans=self._clamp_valid_spans(valid_spans, max_frames))
        return info

    def _build_sample_meta(self, sample):
        """Add static crop eligibility to the shared audio/motion sampling metadata."""
        info = self._build_sampling_info(sample)
        static_valid = info['max_motion_frames'] >= self.ori_length and len(info['valid_spans']) > 0
        return dict(info, static_valid=static_valid)

    def _build_runtime_sampling_info(self, sample):
        """Compute the crop metadata needed when static precomputation is disabled."""
        info = self._build_sampling_info(sample)
        return {key: info[key] for key in ('speaker_id', 'max_motion_frames', 'valid_spans')}

    def _warmup_small_dataset_cache(self):
        """Preload reusable media records for small datasets when caching is enabled."""
        unique_audio_files = set()
        unique_pose_files = set()
        unique_flame_files = set()
        for idx in self.valid_indices:
            sample = self.data[idx]
            unique_audio_files.add(sample["audio_file"])
            motion_files = self._resolve_motion_source_files(sample)
            pose_file = motion_files["gesture"]
            flame_file = motion_files["expression"]
            if pose_file:
                unique_pose_files.add(pose_file)
            if flame_file:
                unique_flame_files.add(flame_file)
            audio_file_listen = sample.get("audio_file_listen", "")
            if self.dialog and audio_file_listen and os.path.exists(audio_file_listen):
                unique_audio_files.add(audio_file_listen)

        for audio_file in unique_audio_files:
            self._load_audio_full(audio_file)
        for pose_file in unique_pose_files:
            self._load_pose_record(pose_file)
        for flame_file in unique_flame_files:
            self._load_flame_record(flame_file)

    @staticmethod
    def _load_flame_stats(stats_path):
        """Load the mean and standard deviation used by the FLAME tokenizer."""
        with open(stats_path, "r", encoding="utf-8") as f:
            motion_stats = json.load(f)

        motion_mean = np.asarray(motion_stats["motion_mean"], dtype=np.float32)
        motion_std = np.asarray(motion_stats["motion_std"], dtype=np.float32)
        if motion_mean.shape != motion_std.shape:
            raise ValueError(
                f"motion_mean and motion_std shape mismatch: {motion_mean.shape} vs {motion_std.shape}"
            )
        return motion_mean, motion_std

    @staticmethod
    def _select_unils_motioncode(motioncode):
        """Select the expression, jaw and eye channels expected by the FLAME tokenizer."""
        motioncode = np.asarray(motioncode)
        if motioncode.ndim != 2:
            raise ValueError(f"motioncode must be 2D, got {motioncode.shape}")
        if motioncode.shape[1] == 108:
            return motioncode
        if motioncode.shape[1] < 111:
            raise ValueError(
                f"motioncode dim must be 108 or >=111 for UniLS slicing, got {motioncode.shape[1]}"
            )
        return np.concatenate(
            [motioncode[:, :104], motioncode[:, 106:108], motioncode[:, 109:111]],
            axis=1,
        )

    @staticmethod
    def _zero_center_global_pose(motioncode):
        """Center global pose channels around their clip mean before expression encoding."""
        motioncode = motioncode.copy()
        if motioncode.shape[1] >= 103:
            motioncode[:, 100:103] = motioncode[:, 100:103] - motioncode[:, 100:103].mean(
                axis=0, keepdims=True
            )
        return motioncode

    def _normalize_flame_motioncode(self, motioncode):
        """Normalize FLAME channels using the loaded training statistics."""
        motion_mean = (
            self.motion_mean.detach().cpu().numpy()
            if isinstance(self.motion_mean, torch.Tensor)
            else np.asarray(self.motion_mean, dtype=np.float32)
        )
        motion_std = (
            self.motion_std.detach().cpu().numpy()
            if isinstance(self.motion_std, torch.Tensor)
            else np.asarray(self.motion_std, dtype=np.float32)
        )
        if motioncode.shape[1] != motion_mean.shape[0]:
            raise ValueError(
                f"motioncode dim {motioncode.shape[1]} mismatches stats dim {motion_mean.shape[0]}"
            )
        return (motioncode - motion_mean) / (motion_std + 1e-8)

    def _find_valid_spans(self, valid_mask, min_length=None):
        """Convert a frame validity mask to contiguous half-open intervals."""
        if min_length is None:
            min_length = self.ori_length
        min_length = int(min_length)
        spans = []
        start = None
        for idx, is_valid in enumerate(np.asarray(valid_mask).astype(np.bool_).tolist()):
            if is_valid and start is None:
                start = idx
            elif (not is_valid) and start is not None:
                if idx - start >= min_length:
                    spans.append((start, idx))
                start = None
        if start is not None and len(valid_mask) - start >= min_length:
            spans.append((start, len(valid_mask)))
        return spans

    @staticmethod
    def _clamp_valid_spans(valid_spans, max_motion_frames):
        """Clip validity intervals to the shared audio/motion length."""
        clipped_spans = []
        for span_start, span_end in valid_spans:
            clipped_start = max(0, int(span_start))
            clipped_end = min(int(span_end), int(max_motion_frames))
            if clipped_end > clipped_start:
                clipped_spans.append((clipped_start, clipped_end))
        return clipped_spans

    def _get_valid_sampling_ranges(self, valid_spans, start_min, start_max, target_length):
        """Find inclusive crop-start ranges inside valid frame intervals."""
        candidate_ranges = []
        for span_start, span_end in valid_spans:
            range_start = max(int(start_min), int(span_start))
            range_end = min(int(start_max), int(span_end) - int(target_length))
            if range_end >= range_start:
                candidate_ranges.append((range_start, range_end))
        return candidate_ranges


    def _sample_start_from_ranges(self, candidate_ranges, rng=None):
        """Choose a start range then a start frame, aligned for cached motion tokens."""
        rng = random if rng is None else rng
        step = self.motion_downsample_factor if self.use_precomputed_motion_tokens else 1
        aligned_ranges = []
        for range_start, range_end in candidate_ranges:
            range_start = int(math.ceil(float(range_start) / float(step))) * step
            range_end = int(math.floor(float(range_end) / float(step))) * step
            if range_end >= range_start:
                aligned_ranges.append((range_start, range_end))
        if len(aligned_ranges) == 0:
            raise ValueError(f"No start aligned to motion downsample factor {step}: {candidate_ranges}")
        range_start, range_end = rng.choice(aligned_ranges)
        return rng.randrange(range_start, range_end + 1, step)

    def _sample_non_overlapping_style_start(self, valid_spans, max_motion_frames, target_start, target_end, rng=None):
        """Sample a valid style crop that does not overlap the target crop."""
        style_length = int(self.motion_style_length)
        start_max = int(max_motion_frames) - style_length
        if start_max < 0:
            return None

        candidate_ranges = self._get_valid_sampling_ranges(
            valid_spans=valid_spans,
            start_min=0,
            start_max=start_max,
            target_length=style_length,
        )
        non_overlap_ranges = []
        for range_start, range_end in candidate_ranges:
            left_end = min(int(range_end), int(target_start) - style_length)
            if left_end >= int(range_start):
                non_overlap_ranges.append((int(range_start), int(left_end)))
            right_start = max(int(range_start), int(target_end))
            if int(range_end) >= right_start:
                non_overlap_ranges.append((right_start, int(range_end)))
        if len(non_overlap_ranges) == 0:
            return None
        return self._sample_start_from_ranges(non_overlap_ranges, rng=rng)

    def motion_cache_sample_key(self, sample_index):
        """Bind positional cache files to their actual gesture/expression sources."""
        files = self._resolve_motion_source_files(self.data[sample_index])
        sources = {key: os.path.abspath(value) for key, value in files.items() if value}
        args = getattr(self, 'args', None)
        settings = {key: getattr(args, key, default) for key, default in (
            ('rvq_encoder_dtype', 'float32'), ('rvq_quantizer_distance_dtype', 'float32'),
            ('rvq_gesture_context_frames', 0), ('rvq_expression_context_frames', 0))}
        return hashlib.sha256(json.dumps({'sources': sources, 'encoding': settings}, sort_keys=True).encode()).hexdigest()

    def _load_precomputed_motion_codes(self, sample_index):
        """Load cached crop starts, motion codes and optional inline exemplar codes."""
        cache_path = os.path.join(self.motion_token_cache_dir, f"{int(sample_index):06d}.npz")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Missing precomputed motion tokens: {cache_path}")
        with np.load(cache_path) as cached:
            if "sample_key" not in cached or str(cached["sample_key"]) != self.motion_cache_sample_key(sample_index):
                raise ValueError(f"Motion cache source mismatch; rebuild the cache: {cache_path}")
            start_frames = np.asarray(cached["start_frames"])
            motion_codes = np.asarray(cached["motion_codes"])
            inline_exemplar_codes = (
                np.asarray(cached["inline_exemplar_codes"])
                if "inline_exemplar_codes" in cached
                else None
            )
        if (
            start_frames.ndim != 1
            or motion_codes.ndim != 3
            or motion_codes.shape[0] != start_frames.shape[0]
            or motion_codes.shape[2] != self.total_num_motion_codebooks
        ):
            raise ValueError(
                f"Cached starts/codes {start_frames.shape}/{motion_codes.shape} do not match "
                f"{self.total_num_motion_codebooks} codebooks: {cache_path}"
            )
        if inline_exemplar_codes is not None and (
            inline_exemplar_codes.ndim != 3
            or inline_exemplar_codes.shape[2] != self.total_num_motion_codebooks
        ):
            raise ValueError(
                f"Cached inline exemplar codes have invalid shape "
                f"{inline_exemplar_codes.shape}: {cache_path}"
            )
        return (
            torch.from_numpy(start_frames.astype(np.int64, copy=False)),
            torch.from_numpy(motion_codes.astype(np.int64, copy=False)),
            None if inline_exemplar_codes is None else torch.from_numpy(
                inline_exemplar_codes.astype(np.int64, copy=False)
            ),
        )

    @staticmethod
    def _select_precomputed_motion_codes(start_frames, motion_codes, start_idx):
        """Select the cached motion codes for an exact crop-start frame."""
        matches = torch.where(start_frames == int(start_idx))[0]
        if matches.numel() != 1:
            raise ValueError(f"Expected exactly one cached crop at frame {start_idx}, got {matches.numel()}")
        return motion_codes[int(matches.item())].clone()

    def _build_motion_feature_parts(self, pose_file, start_idx, fin_idx):
        """Build normalized rotation and translation features for the gesture tokenizer."""
        required_keys = [
            "smplh:body_pose",
            "smplh:left_hand_pose",
            "smplh:right_hand_pose",
        ]
        if self.motion_tokenizer_type == "upper_hands_trans":
            required_keys.append("smplh:translation")

        pose_arrays = self._load_pose_arrays(
            pose_file,
            required_keys,
        )
        pose_body_np = pose_arrays["smplh:body_pose"][start_idx:fin_idx][:, self._SMPLH_UPPER_JOINTS, :]
        pose_left_hand_np = pose_arrays["smplh:left_hand_pose"][start_idx:fin_idx]
        pose_right_hand_np = pose_arrays["smplh:right_hand_pose"][start_idx:fin_idx]
        pose_body = torch.as_tensor(pose_body_np, dtype=torch.float32)
        pose_left_hand = torch.as_tensor(pose_left_hand_np, dtype=torch.float32)
        pose_right_hand = torch.as_tensor(pose_right_hand_np, dtype=torch.float32)

        n = pose_body.shape[0]
        tar_pose_upper = rc.axis_angle_to_matrix(pose_body)
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(n, 78)
        tar_pose_upper = (tar_pose_upper - self.mean_upper) / self.std_upper

        tar_pose_left_hand = rc.axis_angle_to_matrix(pose_left_hand)
        tar_pose_left_hand = rc.matrix_to_rotation_6d(tar_pose_left_hand).reshape(n, 90)
        tar_pose_left_hand = (tar_pose_left_hand - self.mean_left_hand) / self.std_left_hand

        tar_pose_right_hand = rc.axis_angle_to_matrix(pose_right_hand)
        tar_pose_right_hand = rc.matrix_to_rotation_6d(tar_pose_right_hand).reshape(n, 90)
        tar_pose_right_hand = (tar_pose_right_hand - self.mean_right_hand) / self.std_right_hand

        if self.motion_tokenizer_type == "factorized_gesture_5":
            upper_parts = {}
            for body_part, joint_indices in self._FACTORIZED_UPPER_INDICES.items():
                feature_indices = [
                    feature_idx
                    for joint_idx in joint_indices
                    for feature_idx in range(6 * joint_idx, 6 * (joint_idx + 1))
                ]
                upper_parts[body_part] = tar_pose_upper[:, feature_indices]
            return {
                **upper_parts,
                "left_hand": tar_pose_left_hand,
                "right_hand": tar_pose_right_hand,
            }

        if self.motion_tokenizer_type == "upper_hands":
            return {
                "upper": tar_pose_upper,
                "hands": torch.cat([tar_pose_left_hand, tar_pose_right_hand], dim=1),
            }

        pose_trans = torch.as_tensor(pose_arrays["smplh:translation"][start_idx:fin_idx], dtype=torch.float32)
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
        trans_v = (trans_v - self.mean_transv) / self.std_transv

        return {
            "upper_hands_trans": torch.cat(
                [tar_pose_upper, tar_pose_left_hand, tar_pose_right_hand, trans_v], dim=1
            ),
        }

    def _apply_inline_exemplar(self, motion_features, target_pose_file, target_flame_file, target_start_idx, metadata):
        """Blend reference gesture rotations into the target and encode exemplar context."""
        if self.motion_tokenizer_type != "upper_hands":
            raise ValueError("Inline exemplar stitching currently requires motion_tokenizer_type=upper_hands.")
        reference_pose_file = str(
            metadata.get("reference_pose_file", metadata.get("pose_file", metadata.get("pose_path", "")))
        )
        if not reference_pose_file:
            raise KeyError("Inline exemplar metadata is missing reference_pose_file.")
        reference_start = int(metadata["reference_start_frame"])
        reference_length = int(metadata["reference_length_frames"])
        blend_frames = int(metadata["blend_frames"])
        splice_start = int(metadata["target_splice_start_frame"])
        splice_end = splice_start + reference_length
        if splice_start < 0 or splice_end > self.ori_length:
            raise ValueError(f"Inline exemplar splice [{splice_start}, {splice_end}) exceeds target length {self.ori_length}.")

        required_keys = ["smplh:body_pose", "smplh:left_hand_pose", "smplh:right_hand_pose"]
        target_arrays = self._load_pose_arrays(target_pose_file, required_keys)
        reference_arrays = self._load_pose_arrays(reference_pose_file, required_keys)
        weights = exemplar_blend_weights(reference_length, blend_frames)
        target_abs_start = int(target_start_idx) + splice_start
        target_abs_end = target_abs_start + reference_length

        target_upper = torch.as_tensor(
            target_arrays["smplh:body_pose"][target_abs_start:target_abs_end][:, self._SMPLH_UPPER_JOINTS],
            dtype=torch.float32,
        )
        reference_upper = torch.as_tensor(
            reference_arrays["smplh:body_pose"][reference_start:reference_start + reference_length][:, self._SMPLH_UPPER_JOINTS],
            dtype=torch.float32,
        )
        blended_upper = slerp_axis_angle(target_upper, reference_upper, weights)
        upper_6d = rc.matrix_to_rotation_6d(rc.axis_angle_to_matrix(blended_upper)).reshape(reference_length, 78)
        motion_features["upper"][splice_start:splice_end] = (upper_6d - self.mean_upper) / self.std_upper

        blended_hands = []
        for key, mean, std in (
            ("smplh:left_hand_pose", self.mean_left_hand, self.std_left_hand),
            ("smplh:right_hand_pose", self.mean_right_hand, self.std_right_hand),
        ):
            target_hand = torch.as_tensor(target_arrays[key][target_abs_start:target_abs_end], dtype=torch.float32)
            reference_hand = torch.as_tensor(
                reference_arrays[key][reference_start:reference_start + reference_length], dtype=torch.float32
            )
            blended_hand = slerp_axis_angle(target_hand, reference_hand, weights)
            hand_6d = rc.matrix_to_rotation_6d(rc.axis_angle_to_matrix(blended_hand)).reshape(reference_length, 90)
            blended_hands.append((hand_6d - mean) / std)
        motion_features["hands"][splice_start:splice_end] = torch.cat(blended_hands, dim=1)

        reference_features = self._build_motion_feature_parts(
            reference_pose_file, reference_start, reference_start + reference_length
        )
        gesture_codes = self._encode_gesture_motion_codes(reference_features)
        exemplar_codes = torch.full(
            (int(gesture_codes.size(0)), self.total_num_motion_codebooks),
            -100,
            dtype=torch.long,
        )
        exemplar_codes[:, : int(gesture_codes.size(1))] = gesture_codes
        return motion_features, exemplar_codes

    def _build_expression_motion_feature_parts(self, flame_file, start_idx, fin_idx):
        """Crop and normalize FLAME motion features on the expression timeline."""
        flame_record = self._load_flame_record(flame_file)
        expression_start, expression_end = self._get_expression_slice_indices(start_idx, fin_idx)
        clip_motion = flame_record["flame:motioncode"][expression_start:expression_end]
        clip_motion = self._zero_center_global_pose(clip_motion)
        clip_motion = self._normalize_flame_motioncode(clip_motion)
        return {
            "flame_motioncode": torch.as_tensor(clip_motion, dtype=torch.float32),
        }

    def _build_multitask_motion_feature_parts(self, gesture_pose_file, flame_file, start_idx, fin_idx):
        """Collect the gesture and expression feature groups required by the task."""
        motion_features = {}
        if self.has_gesture_target:
            motion_features.update(self._build_motion_feature_parts(gesture_pose_file, start_idx, fin_idx))
        if self.has_expression_target:
            motion_features.update(self._build_expression_motion_feature_parts(flame_file, start_idx, fin_idx))
        return motion_features

    @torch.no_grad()
    def _encode_gesture_motion_codes(self, motion_features):
        """Encode a single gesture feature item into CPU RVQ codes."""
        return self._encode_gesture_motion_codes_batch([motion_features])[0]

    @torch.no_grad()
    def _encode_gesture_motion_codes_batch(self, motion_features_batch):
        """Batch gesture RVQ encoding and concatenate the selected body-part codebooks."""
        if self.motion_tokenizer_type == "upper_hands_trans":
            tokenizer = self.motion_tokenizers["upper_hands_trans"]
            device = next(tokenizer.parameters()).device
            code_idx, _ = self.motion_tokenizers["upper_hands_trans"].encode(
                torch.stack([features["upper_hands_trans"] for features in motion_features_batch]).to(device)
            )
            code_idx = code_idx[:, :, : self.used_num_motion_codebooks].detach().cpu().to(torch.long)
            return list(code_idx.unbind(0))

        if self.motion_tokenizer_type == "factorized_gesture_5":
            part_codes = []
            for body_part in self._FACTORIZED_GESTURE_PARTS:
                tokenizer = self.motion_tokenizers[body_part]
                device = next(tokenizer.parameters()).device
                code_idx, _ = tokenizer.encode(
                    torch.stack([features[body_part] for features in motion_features_batch]).to(device)
                )
                num_quantizers = int(
                    self._MOTION_TOKENIZER_CONFIGS[body_part]["num_quantizers"]
                )
                part_codes.append(code_idx[:, :, :num_quantizers])
            code_idx = torch.cat(part_codes, dim=2)
            code_idx = code_idx[:, :, : self.used_num_motion_codebooks].detach().cpu().to(torch.long)
            return list(code_idx.unbind(0))

        upper_tokenizer = self.motion_tokenizers["upper"]
        hands_tokenizer = self.motion_tokenizers["hands"]
        upper_device = next(upper_tokenizer.parameters()).device
        hands_device = next(hands_tokenizer.parameters()).device
        upper_code_idx, _ = upper_tokenizer.encode(
            torch.stack([features["upper"] for features in motion_features_batch]).to(upper_device)
        )
        hands_code_idx, _ = hands_tokenizer.encode(
            torch.stack([features["hands"] for features in motion_features_batch]).to(hands_device)
        )

        upper_codebooks = min(6, self.used_num_motion_codebooks)
        hands_codebooks = min(6, self.used_num_motion_codebooks)
        code_idx = torch.cat(
            [
                upper_code_idx[:, :, :upper_codebooks],
                hands_code_idx[:, :, :hands_codebooks],
            ],
            dim=2,
        ).detach().cpu().to(torch.long)
        return list(code_idx.unbind(0))

    @torch.no_grad()
    def _encode_expression_motion_codes_batch(self, motion_features_batch):
        """Batch FLAME RVQ encoding and retain the selected expression codebooks."""
        tokenizer = self.motion_tokenizers["flame_motioncode"]
        device = next(tokenizer.parameters()).device
        code_idx, _ = tokenizer.encode(
            torch.stack([features["flame_motioncode"] for features in motion_features_batch]).to(device)
        )
        code_idx = code_idx[:, :, : self.used_num_expression_codebooks].detach().cpu().to(torch.long)
        return list(code_idx.unbind(0))

    def _interleave_motion_codes(self, motion_codes):
        """Arrange upper-body, hand and expression codes in streaming step order."""
        if not self.interleave_motion_parts:
            return motion_codes
        if self.motion_tokenizer_type != "upper_hands":
            return motion_codes

        total_steps = int(motion_codes.size(0))
        num_windows = total_steps // self.gesture_motion_steps_per_window
        interleaved_rows = []
        for window_idx in range(num_windows):
            gesture_start = window_idx * self.gesture_motion_steps_per_window
            for step_idx in range(self.gesture_motion_steps_per_window):
                source_idx = gesture_start + step_idx
                upper_row = torch.full_like(motion_codes[source_idx], -100)
                upper_row[:6] = motion_codes[source_idx, :6]
                interleaved_rows.append(upper_row)

                hands_row = torch.full_like(motion_codes[source_idx], -100)
                hands_row[6:12] = motion_codes[source_idx, 6:12]
                interleaved_rows.append(hands_row)

                if self.has_expression_target and step_idx < self.expression_motion_steps_per_window:
                    expr_row = torch.full_like(motion_codes[source_idx], -100)
                    expr_row[12:] = motion_codes[source_idx, 12:]
                    interleaved_rows.append(expr_row)
        return torch.stack(interleaved_rows, dim=0)

    def _merge_motion_code_groups(self, gesture_motion_codes, expression_motion_codes):
        """Align gesture/expression windows and apply the configured code layout."""
        if self.has_gesture_target and self.has_expression_target:
            gesture_total_steps = int(gesture_motion_codes.size(0))
            expression_total_steps = int(expression_motion_codes.size(0))
            
            gesture_num_windows = gesture_total_steps // self.gesture_motion_steps_per_window
            expression_num_windows = expression_total_steps // self.expression_motion_steps_per_window
            num_windows = min(gesture_num_windows, expression_num_windows)
            if num_windows <= 0:
                raise ValueError(
                    f"Gesture/expression window mismatch: {gesture_num_windows} vs {expression_num_windows}"
                )

            gesture_num_codebooks = int(gesture_motion_codes.size(1))
            expression_num_codebooks = int(expression_motion_codes.size(1))
            motion_codes = torch.full(
                (
                    num_windows * self.gesture_motion_steps_per_window,
                    gesture_num_codebooks + expression_num_codebooks,
                ),
                -100,
                dtype=torch.long,
            )
            for window_idx in range(num_windows):
                gesture_start = window_idx * self.gesture_motion_steps_per_window
                gesture_end = gesture_start + self.gesture_motion_steps_per_window
                expression_start = window_idx * self.expression_motion_steps_per_window
                expression_end = expression_start + self.expression_motion_steps_per_window
                motion_codes[gesture_start:gesture_end, :gesture_num_codebooks] = gesture_motion_codes[
                    gesture_start:gesture_end
                ]
                motion_codes[
                    gesture_start : gesture_start + self.expression_motion_steps_per_window,
                    gesture_num_codebooks:,
                ] = expression_motion_codes[expression_start:expression_end]
            return self._interleave_motion_codes(motion_codes)

        motion_codes = gesture_motion_codes if self.has_gesture_target else expression_motion_codes
        return self._interleave_motion_codes(motion_codes.to(torch.long))

    @torch.no_grad()
    def _encode_motion_tokens_batch(self, motion_features_batch):
        """Batch compatible feature shapes; encode incompatible crops individually."""
        if len(motion_features_batch) == 0:
            return []
        reference_shapes = {key: tuple(value.shape) for key, value in motion_features_batch[0].items()}
        if any(
            {key: tuple(value.shape) for key, value in features.items()} != reference_shapes
            for features in motion_features_batch[1:]
        ):
            return [self._encode_motion_tokens(features) for features in motion_features_batch]

        gesture_codes_batch = (
            self._encode_gesture_motion_codes_batch(motion_features_batch)
            if self.has_gesture_target
            else [None] * len(motion_features_batch)
        )
        expression_codes_batch = (
            self._encode_expression_motion_codes_batch(motion_features_batch)
            if self.has_expression_target
            else [None] * len(motion_features_batch)
        )
        return [
            self._merge_motion_code_groups(gesture_codes, expression_codes)
            for gesture_codes, expression_codes in zip(gesture_codes_batch, expression_codes_batch)
        ]

    @torch.no_grad()
    def _encode_motion_tokens(self, motion_features):
        """Encode one feature item and merge its active gesture/expression codebooks."""
        return self._encode_motion_tokens_batch([motion_features])[0]
    
    def __len__(self):
        """Return the number of source samples."""
        return len(self.data)


    def _event_target_start(self, event, target_length):
        """Use an explicit crop start, or center the crop on a requested semantic event."""
        if event.get('target_start_frame') is not None:
            return int(event['target_start_frame'])
        start = float(event['source_start_sec']) * float(self.pose_fps)
        end = float(event['source_end_sec']) * float(self.pose_fps)
        return int(round(0.5 * (start + end) - 0.5 * target_length))

    def _sample_style_start(self, sample, event, cached_start_frames, valid_spans, max_motion_frames,
                            target_start, target_end):
        """Select a valid style crop disjoint from the target, using fixed/cache/random rules."""
        fixed_style_start = (
            event.get("motion_style_start")
            if event is not None and event.get("motion_style_start") is not None
            else sample.get("ipo_motion_style_start")
        )
        if fixed_style_start is not None:
            style_start_idx = int(fixed_style_start)
            if style_start_idx < 0 or style_start_idx + int(self.motion_style_length) > max_motion_frames:
                raise ValueError(
                    f"Fixed IPO style range [{style_start_idx}, "
                    f"{style_start_idx + int(self.motion_style_length)}) exceeds pose length {max_motion_frames}."
                )
        else:
            style_rng = None
            if event is not None:
                style_rng = random.Random(int(event.get("sampling_seed", 0)))
            if self.use_precomputed_motion_tokens:
                cached_style_candidates = []
                for value in cached_start_frames.tolist():
                    value = int(value)
                    style_end = value + int(self.motion_style_length)
                    within_valid = any(
                        int(span_start) <= value and style_end <= int(span_end)
                        for span_start, span_end in valid_spans
                    )
                    non_overlapping = style_end <= target_start or value >= target_end
                    if within_valid and style_end <= max_motion_frames and non_overlapping:
                        cached_style_candidates.append(value)
                if cached_style_candidates:
                    chooser = random if style_rng is None else style_rng
                    style_start_idx = chooser.choice(cached_style_candidates)
                else:
                    style_start_idx = None
            else:
                style_start_idx = self._sample_non_overlapping_style_start(
                    valid_spans=valid_spans,
                    max_motion_frames=max_motion_frames,
                    target_start=target_start,
                    target_end=target_end,
                    rng=style_rng,
                )
        return style_start_idx

    def __getitem__(self, index):
        """Sample aligned audio/motion crops, optional style context and supervision masks.

        Crop coordinates use pose frames; expression and audio slices are converted
        separately. Cached tokens select the same crop through exact start frames.
        """
        requested_event = None
        if isinstance(index, (tuple, list)) and len(index) >= 2 and isinstance(index[1], dict):
            index, requested_event = index[0], dict(index[1])
        if self.task_type == "asr":
            return self._getitem_asr(index)

        alignment_th = _get_alignment_score_threshold(self.args)
        target_length = self.ori_length

        if len(self.valid_indices) == 0:
            raise RuntimeError("No valid indices available for sampling.")

        index = self.valid_indices[index % len(self.valid_indices)]

        for _ in range(self.max_sampling_retries):
            sample = self.data[index]
            motion_files = self._resolve_motion_source_files(sample)
            pose_file = motion_files["gesture"]
            flame_file = motion_files["expression"]
            audio_file = sample["audio_file"]
            audio_file_listen = sample.get("audio_file_listen", "")
            word_file = sample.get("word_file", "")
            info = (self._build_runtime_sampling_info(sample) if self._sample_meta is None
                    else self._sample_meta[index])
            speaker_id = info["speaker_id"]
            max_motion_frames, valid_spans = info["max_motion_frames"], info["valid_spans"]

            selected_segment = None
            if alignment_th is not None:
                selected_segment = _select_aligned_segment(
                    word_file,
                    sample.get("alignment_score", None),
                    threshold=alignment_th,
                    min_len_sec=float(target_length) / float(self.pose_fps),
                )
                if selected_segment is None:
                    index = random.choice(self.valid_indices)
                    continue

            if max_motion_frames < target_length:
                index = random.choice(self.valid_indices)
                continue
            if len(valid_spans) == 0:
                index = random.choice(self.valid_indices)
                continue


            start_min = 0
            start_max = max_motion_frames - target_length
            if selected_segment is not None:
                seg_start_s, seg_end_s = selected_segment
                start_min = max(start_min, int(math.ceil(float(seg_start_s) * float(self.pose_fps))))
                start_max = min(start_max, int(math.floor(float(seg_end_s) * float(self.pose_fps))) - target_length)

            if start_max < start_min:
                index = random.choice(self.valid_indices)
                continue

            candidate_ranges = self._get_valid_sampling_ranges(
                valid_spans=valid_spans,
                start_min=start_min,
                start_max=start_max,
                target_length=target_length,
            )
            if len(candidate_ranges) == 0:
                index = random.choice(self.valid_indices)
                continue


            fixed_start_sec = sample.get("fixed_start_sec", sample.get("ipo_clip_start_sec"))
            fixed_start_idx = None
            if fixed_start_sec is not None:
                fixed_start_idx = int(round(float(fixed_start_sec) * float(self.pose_fps)))
                if not any(int(left) <= fixed_start_idx <= int(right) for left, right in candidate_ranges):
                    raise ValueError(
                        f"Fixed IPO start {fixed_start_idx} is outside valid sampling ranges "
                        f"for dataset index={index}: {candidate_ranges[:8]}"
                    )

            cached_start_frames = None
            cached_motion_codes = None
            cached_inline_exemplar_codes = None
            if self.use_precomputed_motion_tokens:
                cached_start_frames, cached_motion_codes, cached_inline_exemplar_codes = (
                    self._load_precomputed_motion_codes(index)
                )
                cached_candidates = [
                    int(value)
                    for value in cached_start_frames.tolist()
                    if any(int(left) <= int(value) <= int(right) for left, right in candidate_ranges)
                ]
                if len(cached_candidates) == 0:
                    index = random.choice(self.valid_indices)
                    continue
                if requested_event is not None:
                    desired_start = self._event_target_start(requested_event, target_length)
                    start_idx = min(cached_candidates, key=lambda value: abs(value - desired_start))
                elif fixed_start_idx is not None:
                    if fixed_start_idx not in cached_candidates:
                        raise ValueError(f"Fixed IPO start {fixed_start_idx} is not present in the motion-token cache.")
                    start_idx = fixed_start_idx
                else:
                    start_idx = random.choice(cached_candidates)
            elif requested_event is not None:
                desired_start = self._event_target_start(requested_event, target_length)
                start_idx = min(
                    (min(max(desired_start, int(left)), int(right)) for left, right in candidate_ranges),
                    key=lambda value: abs(int(value) - desired_start),
                )
            elif fixed_start_idx is not None:
                start_idx = fixed_start_idx
            else:
                start_idx = self._sample_start_from_ranges(candidate_ranges)

            fin_idx = start_idx + target_length
            audio_start = math.floor(start_idx * self.audio_fps / self.pose_fps)
            audio_seg_len = math.floor(target_length * self.audio_fps / self.pose_fps)
            audio_end = audio_start + audio_seg_len

            audio_segment, _ = self._load_audio_segment(audio_file, audio_start, audio_end)
            if audio_segment.shape[0] < audio_seg_len:
                index = random.choice(self.valid_indices)
                continue

            if self.use_listener_audio:
                audio_listen_segment, _ = self._load_audio_segment(audio_file_listen, audio_start, audio_end, ref_audio=audio_segment)
                if audio_listen_segment.shape[0] < audio_seg_len:
                    index = random.choice(self.valid_indices)
                    continue
            else:
                audio_listen_segment = np.zeros_like(audio_segment)

            speaker_motion_features = None
            speaker_motion_codes = None
            if self.add_speaker_id and self.speaker_id_type == "motion":
                style_start_idx = self._sample_style_start(
                    sample, requested_event, cached_start_frames, valid_spans, max_motion_frames, start_idx, fin_idx
                )
                if style_start_idx is None:
                    if requested_event is not None:
                        raise RuntimeError(
                            f"Requested GRPO event has no non-overlapping motion-style crop: {requested_event['event_uid']}"
                        )
                    index = random.choice(self.valid_indices)
                    continue
                if self.use_precomputed_motion_tokens and style_start_idx % self.motion_downsample_factor != 0:
                    raise ValueError(
                        f"Fixed style start {style_start_idx} is not aligned to motion downsample factor "
                        f"{self.motion_downsample_factor}."
                    )
                if self.use_precomputed_motion_tokens and not bool((cached_start_frames == style_start_idx).any()):
                    raise ValueError(f"Fixed style start {style_start_idx} is not present in the motion-token cache.")
                style_fin_idx = style_start_idx + int(self.motion_style_length)
                if not self.use_precomputed_motion_tokens:
                    speaker_motion_features = self._build_multitask_motion_feature_parts(
                        pose_file,
                        flame_file,
                        style_start_idx,
                        style_fin_idx,
                    )

            motion_features = None
            if not self.use_precomputed_motion_tokens:
                motion_features = self._build_multitask_motion_feature_parts(
                    pose_file,
                    flame_file,
                    start_idx,
                    fin_idx,
                )
            inline_exemplar_metadata = sample.get("inline_exemplars")
            if inline_exemplar_metadata is None:
                legacy_metadata = sample.get("inline_exemplar")
                inline_exemplar_metadata = [] if legacy_metadata is None else [legacy_metadata]
            inline_exemplar_codes = []
            if inline_exemplar_metadata:
                if not self.inline_exemplar_enabled:
                    raise ValueError("Training data contains inline_exemplars but --inline_exemplar_enabled is false.")
                if self.use_precomputed_motion_tokens:
                    if cached_inline_exemplar_codes is None:
                        raise ValueError("Inline exemplar codes are missing from the precomputed motion cache.")
                    if int(cached_inline_exemplar_codes.size(0)) != len(inline_exemplar_metadata):
                        raise ValueError(
                            "Cached inline exemplar count does not match training metadata: "
                            f"{cached_inline_exemplar_codes.size(0)} vs {len(inline_exemplar_metadata)}"
                        )
                    inline_exemplar_codes = list(cached_inline_exemplar_codes.unbind(0))
                else:
                    for metadata in inline_exemplar_metadata:
                        motion_features, exemplar_codes = self._apply_inline_exemplar(
                            motion_features, pose_file, flame_file, start_idx, metadata
                        )
                        inline_exemplar_codes.append(exemplar_codes)
            if self.use_precomputed_motion_tokens:
                motion_codes = self._select_precomputed_motion_codes(
                    cached_start_frames, cached_motion_codes, start_idx
                )
                if self.add_speaker_id and self.speaker_id_type == "motion":
                    speaker_motion_codes = self._select_precomputed_motion_codes(
                        cached_start_frames, cached_motion_codes, style_start_idx
                    )
            elif speaker_motion_features is None:
                motion_codes = self._encode_motion_tokens(motion_features)
            else:
                speaker_motion_codes, motion_codes = self._encode_motion_tokens_batch(
                    [speaker_motion_features, motion_features]
                )
            if requested_event is not None and requested_event.get("rollout_history_codes") is not None:
                history_codes = requested_event["rollout_history_codes"].to(torch.long)
                speaker_motion_codes = torch.cat([speaker_motion_codes, history_codes], dim=0)
            audio = torch.from_numpy(audio_segment).float()
            audio_listen = torch.from_numpy(audio_listen_segment).float()

            out = {
                "audio": audio,
                "speaker_id": speaker_id,
            }
            if requested_event is not None:
                clip_start_sec = float(start_idx) / float(self.pose_fps)
                clip_end_sec = float(fin_idx) / float(self.pose_fps)
                out["grpo_event_metadata"] = {
                    **requested_event,
                    "audio_file": audio_file,
                    "audio_file_listen": audio_file_listen,
                    "word_file": word_file,
                    "word_file_listen": _infer_listener_word_file(audio_file_listen),
                    "speaker_id": str(sample.get("speaker_id", speaker_id)),
                    "clip_start_sec": clip_start_sec,
                    "clip_end_sec": clip_end_sec,
                    "start_sec": float(requested_event["source_start_sec"]) - clip_start_sec,
                    "end_sec": float(requested_event["source_end_sec"]) - clip_start_sec,
                }
            out["motion_codes"] = motion_codes
            out["motion_tokens"] = motion_codes.reshape(-1)
            motion_loss_mask = None
            for metadata in inline_exemplar_metadata:
                motion_loss_mask = self._build_inline_imitation_motion_loss_mask(
                    motion_loss_mask, metadata, int(motion_codes.size(0))
                )
            if motion_loss_mask is not None:
                out["motion_loss_mask"] = motion_loss_mask
            if speaker_motion_codes is not None:
                out["speaker_motion_codes"] = speaker_motion_codes
                if requested_event is not None:
                    out["grpo_event_metadata"].update({
                        "motion_style_pose_path": pose_file,
                        "motion_style_start": int(style_start_idx),
                        "motion_style_length": int(self.motion_style_length),
                    })
            if inline_exemplar_codes:
                out["inline_exemplar_codes"] = inline_exemplar_codes
                out["inline_exemplar_metadata"] = [dict(row) for row in inline_exemplar_metadata]

            if self.task_type in {"asr_gesture", "asr_expression", "asr_gesture_expression"}:
                samples_per_window = int(round(float(getattr(self.args, "window_sec", 0.8)) * self.audio_fps))
                num_windows = int(audio.shape[0] // samples_per_window)
                if num_windows <= 0:
                    index = random.choice(self.valid_indices)
                    continue
                valid_num_samples = num_windows * samples_per_window
                audio = audio[:valid_num_samples]
                audio_listen = audio_listen[:valid_num_samples]
                out["audio"] = audio
                if self.use_listener_audio:
                    out["audio_listen"] = audio_listen
                out = self._attach_asr_text_blocks_to_sample(
                    sample=out,
                    word_file=word_file,
                    audio_file_listen=audio_file_listen,
                    segment_start_sec=float(audio_start) / float(self.audio_fps),
                    num_windows=num_windows,
                )

            if self.use_listener_audio:
                out["audio_listen"] = audio_listen
            return out

        self._warn_sampling_issue(
            f"Failed to sample a valid motion segment after {self.max_sampling_retries} retries; "
            f"dataset index={index}, task_type={self.task_type}. The sample will be retried by the wrapper dataset."
        )
        raise RuntimeError("Failed to sample a valid motion segment after multiple retries.")


def build_alternating_qwen_sequence(
    num_windows: int,
    motion_slot_token: str,
    dialog: bool,
    motion_only: bool,
    motion_steps_per_window: int,
    num_motion_codebooks: int,
    speaker_id: str = "",
    generation_target: str = "gesture",
    prefix_motion_steps: int = 0,
    text_blocks=None,
    inline_exemplar_steps_by_window=None,
):
    """Build text/audio placeholder windows with optional listener and motion context."""
    target_phrase, _ = _describe_generation_target(generation_target)
    system_prompt = _build_system_prompt(
        f"You are a {target_phrase} generation model.",
        speaker_id,
    )
    windows = []
    if text_blocks is not None and len(text_blocks) < int(num_windows):
        raise RuntimeError("Text conditioning blocks are shorter than the motion windows.")
    for idx in range(num_windows):
        audio_placeholders = "<|vision_start|><|vision_end|>"
        if dialog:
            audio_placeholders += "<|vision_start|><|vision_end|>"
        if motion_only:
            audio_placeholders = ""
        text_block = "" if text_blocks is None else str(text_blocks[idx])
        guide_steps = 0 if inline_exemplar_steps_by_window is None else int(inline_exemplar_steps_by_window[idx])
        guide = ""
        if guide_steps > 0:
            guide = EXEMPLAR_START_TOKEN + motion_slot_token * guide_steps + EXEMPLAR_END_TOKEN
        windows.append(guide + audio_placeholders + text_block + motion_slot_token * motion_steps_per_window)
    body = motion_slot_token * int(prefix_motion_steps) + "".join(windows)
    stream_desc = "speaker and listener audio stream" if dialog else "speaker audio stream"
    if text_blocks is None:
        user_prompt = f"Generate {target_phrase} tokens from the interleaved {stream_desc}."
    else:
        user_prompt = (
            f"For each {0.8:.1f}s window, condition on the {stream_desc} and the provided transcript, "
            f"then generate the {target_phrase} tokens."
        )
    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{body}<|im_end|>\n"
    )

class CoSpeechGestureDataset(Dataset):
    def __init__(self, args, tokenizer, window_sec=0.8, audio_sr=16000, motion_slot_token="<|vision_pad|>"):
        """Wrap motion sampling with Qwen tokenization."""
        self.base_dataset = Gesture_Dataset(args)
        self.tokenizer = tokenizer
        self.task_type = str(getattr(args, "task_type", "gesture")).lower()
        self.generation_target = getattr(self.base_dataset, "generation_target", "gesture")
        self.has_motion_target = getattr(self.base_dataset, "has_motion_target", self.task_type != "asr")
        self.has_asr_target = getattr(self.base_dataset, "has_asr_target", self.task_type != "gesture")
        self.dialog = bool(getattr(args, "dialog", True))
        self.listener_fusion_delta = bool(getattr(args, "listener_fusion_delta", False))
        self.use_listener_audio = self.dialog or self.listener_fusion_delta
        self.dialog_text = bool(getattr(args, "dialog_text", False)) and self.dialog
        self.inline_exemplar_copy_mode = getattr(self.base_dataset, "inline_exemplar_copy_mode", "learned")
        self.motion_only = bool(getattr(args, "motion_only", False))
        self.window_sec = float(window_sec)
        self.audio_sr = int(audio_sr)
        self.samples_per_window = int(round(self.window_sec * self.audio_sr))
        self.motion_steps_per_window = int(
            getattr(self.base_dataset, "total_motion_steps_per_window", 6)
        )
        self.num_codebooks = int(
            getattr(
                self.base_dataset,
                "total_num_motion_codebooks",
                getattr(args, "used_num_motion_codebooks", 6),
            )
        )
        self.nb_code = 2048
        self.motion_slot_token = motion_slot_token

        self.vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.motion_slot_id = self.tokenizer.convert_tokens_to_ids(self.motion_slot_token)
        self.eot_token = self.tokenizer.eos_token or "<|endoftext|>"
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids(self.eot_token)
        self.max_base_resample_retries = max(
            1,
            int(getattr(args, "max_base_resample_retries", 32)),
        )
        if self.has_motion_target and min(self.vision_start_id, self.vision_end_id, self.motion_slot_id) < 0:
            raise ValueError("Tokenizer is missing required placeholder tokens.")
        if bool(getattr(args, "inline_exemplar_enabled", False)):
            for delimiter in (EXEMPLAR_START_TOKEN, EXEMPLAR_END_TOKEN):
                delimiter_ids = self.tokenizer(delimiter, add_special_tokens=False)["input_ids"]
                if len(delimiter_ids) != 1 or self.tokenizer.convert_ids_to_tokens(delimiter_ids[0]) != delimiter:
                    raise ValueError(f"Inline exemplar delimiter is not one tokenizer token: {delimiter}")
        if self.has_asr_target:
            if min(self.vision_start_id, self.vision_end_id, self.eot_token_id) < 0:
                raise ValueError("Tokenizer is missing required ASR tokens.")

    def __len__(self):
        """Return the number of wrapped source samples."""
        return len(self.base_dataset)


    def _assemble_motion_sequence(self, sample, motion_codes, motion_loss_mask):
        """Style and pose guides condition the model but have no prediction labels (-100)."""
        speaker_codes = sample.get("speaker_motion_codes")
        style_steps = int(speaker_codes.size(0)) if speaker_codes is not None else 0
        exemplar_codes = sample.get("inline_exemplar_codes") or []
        metadata = sample.get("inline_exemplar_metadata") or []
        if len(exemplar_codes) != len(metadata):
            raise ValueError("Inline exemplar code and metadata counts differ.")
        num_windows = int(motion_codes.size(0)) // self.motion_steps_per_window
        guide_by_window = {}
        for codes, row in zip(exemplar_codes, metadata):
            codes = codes.to(torch.long)
            if int(codes.size(0)) % self.motion_steps_per_window != 0:
                raise ValueError("Inline exemplar length must be divisible by one motion window.")
            copy_start_window = int(row["copy_start_window"])
            for offset, chunk in enumerate(codes.split(self.motion_steps_per_window, dim=0)):
                window_idx = copy_start_window + offset
                if window_idx < 0 or window_idx >= num_windows:
                    raise ValueError("Inline exemplar imitation window exceeds the target stream.")
                if window_idx in guide_by_window:
                    raise ValueError("Inline exemplar imitation intervals must not overlap.")
                guide_by_window[window_idx] = (chunk, offset)

        context_parts, label_parts, loss_parts = [], [], []
        if speaker_codes is not None:
            speaker_codes = speaker_codes.to(torch.long)
            context_parts.append(speaker_codes)
            label_parts.append(torch.full_like(speaker_codes, -100))
            if motion_loss_mask is not None:
                loss_parts.append(torch.ones(style_steps))
        guide_steps_by_window = []
        for window_idx in range(num_windows):
            target_start = window_idx * self.motion_steps_per_window
            target_chunk = motion_codes[target_start:target_start + self.motion_steps_per_window]
            guide_info = guide_by_window.get(window_idx)
            guide_chunk = None if guide_info is None else guide_info[0]
            guide_offset = -1 if guide_info is None else int(guide_info[1])
            guide_steps_by_window.append(0 if guide_chunk is None else int(guide_chunk.size(0)))
            if guide_chunk is not None:
                context_parts.append(guide_chunk)
                label_parts.append(torch.full_like(guide_chunk, -100))
                if motion_loss_mask is not None:
                    loss_parts.append(torch.zeros(int(guide_chunk.size(0))))
            if guide_chunk is not None and self.inline_exemplar_copy_mode == "hard_core" and 1 <= guide_offset <= 6:
                context_parts.append(guide_chunk)
            else:
                context_parts.append(target_chunk)
            label_parts.append(target_chunk)
            if motion_loss_mask is not None:
                loss_parts.append(motion_loss_mask[target_start:target_start + self.motion_steps_per_window])
        if motion_loss_mask is not None:
            motion_loss_mask = torch.cat(loss_parts, dim=0)
        return (
            torch.cat(context_parts, dim=0),
            torch.cat(label_parts, dim=0),
            motion_loss_mask,
            style_steps,
            guide_steps_by_window,
        )


    def _tokenize_segments_with_labels(self, segments):
        """Tokenize segments separately while preserving their supervised text boundaries."""
        input_ids = []
        labels = []
        for text, predict in segments:
            if text == "":
                continue
            token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            input_ids.extend(token_ids)
            if predict:
                labels.extend(token_ids)
            else:
                labels.extend([-100] * len(token_ids))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask, labels

    def _build_asr_training_example(self, sample):
        """Build audio placeholders, text labels and filterbank source audio for ASR."""
        audio = sample["audio"]
        audio_listen = sample.get("audio_listen")
        asr_text_blocks = sample["asr_text_blocks"]
        asr_text_blocks_listen = sample.get("asr_text_blocks_listen")
        speaker_id = sample.get("speaker_id", "")

        system_prompt = _build_system_prompt(
            "You are a streaming automatic speech recognition model.",
            speaker_id,
        )
        if self.dialog_text:
            user_prompt = (
                f"Transcribe the speaker and listener audio streams in real time with {self.window_sec:.1f}s windows. "
                f"For each window, first output the words that finished in the speaker audio and end with {self.eot_token}, "
                f"then output the words that finished in the listener audio and end with {self.eot_token}."
            )
        else:
            user_prompt = (
                f"Transcribe the speaker audio stream in real time with {self.window_sec:.1f}s windows. "
                f"For each window, output the words that finished in that window and always end the window with {self.eot_token}."
            )
        segments = [
            ("<|im_start|>system\n", False),
            (system_prompt, False),
            ("<|im_end|>\n", False),
            ("<|im_start|>user\n", False),
            (user_prompt, False),
            ("<|im_end|>\n", False),
            ("<|im_start|>assistant\n", False),
        ]

        audio_placeholders = "<|vision_start|><|vision_end|>"
        if self.dialog:
            audio_placeholders += "<|vision_start|><|vision_end|>"
        for block_idx, block_text in enumerate(asr_text_blocks):
            segments.append((audio_placeholders, False))
            segments.append((block_text, True))
            if self.dialog_text:
                if asr_text_blocks_listen is None:
                    raise RuntimeError("`dialog_text=True` but listener ASR text blocks are missing.")
                if block_idx >= len(asr_text_blocks_listen):
                    raise RuntimeError("Listener ASR text blocks are shorter than speaker ASR text blocks.")
                segments.append((asr_text_blocks_listen[block_idx], True))
        segments.append(("<|im_end|>\n", False))

        input_ids, attention_mask, labels = self._tokenize_segments_with_labels(segments)
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_start_mask": input_ids == self.vision_start_id,
            "audio_end_mask": input_ids == self.vision_end_id,
            "labels": labels,
            "audio": audio.to(torch.float32),
        }
        if self.use_listener_audio:
            if audio_listen is None:
                raise RuntimeError("Listener conditioning is enabled but `audio_listen` is missing.")
            out["audio_listen"] = audio_listen.to(torch.float32)
        return out

    def _build_asr_motion_training_example(self, sample):
        """Build joint text/motion inputs with conditioning steps excluded from labels."""
        audio = sample["audio"]
        audio_listen = sample.get("audio_listen")
        motion_tokens = sample["motion_tokens"]
        motion_codes = sample.get("motion_codes")
        asr_text_blocks = sample["asr_text_blocks"]
        asr_text_blocks_listen = sample.get("asr_text_blocks_listen")
        speaker_id = sample.get("speaker_id", "")
        target_phrase, _ = _describe_generation_target(self.generation_target)

        if motion_codes is None:
            motion_codes = motion_tokens.view(-1, self.num_codebooks)
        num_windows = motion_codes.size(0) // self.motion_steps_per_window
        if num_windows <= 0:
            raise RuntimeError("No valid 0.8s windows can be formed from the sample.")
        motion_codes = motion_codes[: num_windows * self.motion_steps_per_window]
        motion_loss_mask = sample.get("motion_loss_mask")
        if motion_loss_mask is not None:
            motion_loss_mask = motion_loss_mask[: num_windows * self.motion_steps_per_window].to(torch.float32)
        motion_context_labels, motion_labels, motion_loss_mask, prefix_motion_steps, inline_exemplar_steps_by_window = self._assemble_motion_sequence(
            sample, motion_codes, motion_loss_mask
        )

        system_prompt = _build_system_prompt(
            f"You are a {target_phrase} generation model conditioned on streaming ASR.",
            speaker_id,
        )
        if self.dialog_text:
            user_prompt = (
                f"For each {self.window_sec:.1f}s window, first transcribe the speaker audio and end with {self.eot_token}, "
                f"then transcribe the listener audio and end with {self.eot_token}, then generate the {target_phrase} tokens."
            )
        else:
            user_prompt = (
                f"For each {self.window_sec:.1f}s window, first transcribe the speaker audio and end with {self.eot_token}, "
                f"then generate the {target_phrase} tokens."
            )
        segments = [
            ("<|im_start|>system\n", False),
            (system_prompt, False),
            ("<|im_end|>\n", False),
            ("<|im_start|>user\n", False),
            (user_prompt, False),
            ("<|im_end|>\n", False),
            ("<|im_start|>assistant\n", False),
        ]

        audio_placeholders = "<|vision_start|><|vision_end|>"
        if self.dialog:
            audio_placeholders += "<|vision_start|><|vision_end|>"

        if prefix_motion_steps > 0:
            segments.append((self.motion_slot_token * prefix_motion_steps, False))

        for window_idx in range(num_windows):
            guide_steps = int(inline_exemplar_steps_by_window[window_idx])
            if guide_steps > 0:
                segments.append((EXEMPLAR_START_TOKEN, False))
                segments.append((self.motion_slot_token * guide_steps, False))
                segments.append((EXEMPLAR_END_TOKEN, False))
            segments.append((audio_placeholders, False))
            segments.append((asr_text_blocks[window_idx], True))
            if self.dialog_text:
                if asr_text_blocks_listen is None:
                    raise RuntimeError("`dialog_text=True` but listener ASR text blocks are missing.")
                if window_idx >= len(asr_text_blocks_listen):
                    raise RuntimeError("Listener ASR text blocks are shorter than speaker ASR text blocks.")
                segments.append((asr_text_blocks_listen[window_idx], True))
            segments.append((self.motion_slot_token * self.motion_steps_per_window, False))
        segments.append(("<|im_end|>\n", False))

        input_ids, attention_mask, labels = self._tokenize_segments_with_labels(segments)
        motion_slot_count = int((input_ids == self.motion_slot_id).sum().item())
        if motion_slot_count != motion_labels.size(0):
            raise RuntimeError(
                f"Motion slot count ({motion_slot_count}) does not match motion label steps ({motion_labels.size(0)})."
            )

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_start_mask": input_ids == self.vision_start_id,
            "audio_end_mask": input_ids == self.vision_end_id,
            "motion_slot_mask": input_ids == self.motion_slot_id,
            "labels": labels,
            "audio": audio.to(torch.float32),
            "motion_labels": motion_labels.to(torch.long),
        }
        if motion_loss_mask is not None:
            out["motion_loss_mask"] = motion_loss_mask.to(torch.float32)
        if sample.get("grpo_event_metadata") is not None:
            out["grpo_event_metadata"] = dict(sample["grpo_event_metadata"])
        if motion_context_labels is not None:
            out["motion_context_labels"] = motion_context_labels.to(torch.long)
        if self.use_listener_audio:
            if audio_listen is None:
                raise RuntimeError("Listener conditioning is enabled but `audio_listen` is missing.")
            out["audio_listen"] = audio_listen.to(torch.float32)
        return out

    def _fetch_base_sample(self, idx):
        """Fetch a requested event or regular source sample from the base dataset."""
        last_exc = None
        base_idx = int(idx)
        for _ in range(self.max_base_resample_retries):
            try:
                return self.base_dataset[base_idx]
            except Exception as exc:
                last_exc = exc
                base_idx = random.randrange(len(self.base_dataset))
        raise RuntimeError(
            "CoSpeechGestureDataset failed to fetch a valid sample after "
            f"{self.max_base_resample_retries} retries."
        ) from last_exc


    def _build_training_example(self, sample):
        """Dispatch sample assembly to ASR, joint or motion-only tokenization."""
        if self.task_type == "asr":
            return self._build_asr_training_example(sample)
        if self.task_type in {"asr_gesture", "asr_expression", "asr_gesture_expression"}:
            return self._build_asr_motion_training_example(sample)

        audio = sample["audio"]
        audio_listen = sample.get("audio_listen")
        motion_tokens = sample["motion_tokens"]
        motion_codes = sample.get("motion_codes")

        if motion_codes is None:
            motion_codes = motion_tokens.view(-1, self.num_codebooks)
        num_windows = motion_codes.size(0) // self.motion_steps_per_window
        if num_windows <= 0:
            raise RuntimeError("No valid 0.8s windows can be formed from the sample.")
        motion_codes = motion_codes[: num_windows * self.motion_steps_per_window]
        motion_loss_mask = sample.get("motion_loss_mask")
        if motion_loss_mask is not None:
            motion_loss_mask = motion_loss_mask[: num_windows * self.motion_steps_per_window].to(torch.float32)
        motion_context_labels, motion_labels, motion_loss_mask, prefix_motion_steps, inline_exemplar_steps_by_window = self._assemble_motion_sequence(
            sample, motion_codes, motion_loss_mask
        )

        text = build_alternating_qwen_sequence(
            num_windows,
            self.motion_slot_token,
            self.dialog,
            self.motion_only,
            self.motion_steps_per_window,
            self.num_codebooks,
            speaker_id=sample.get("speaker_id", ""),
            generation_target=self.generation_target,
            prefix_motion_steps=prefix_motion_steps,
            text_blocks=None,
            inline_exemplar_steps_by_window=inline_exemplar_steps_by_window,
        )
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        motion_slot_count = int((input_ids == self.motion_slot_id).sum().item())
        if motion_slot_count != motion_labels.size(0):
            raise RuntimeError(
                f"Motion slot count ({motion_slot_count}) does not match motion label steps ({motion_labels.size(0)})."
            )

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio_start_mask": input_ids == self.vision_start_id,
            "audio_end_mask": input_ids == self.vision_end_id,
            "motion_slot_mask": input_ids == self.motion_slot_id,
            "audio": audio.to(torch.float32),
            "motion_labels": motion_labels.to(torch.long),
        }
        if motion_loss_mask is not None:
            out["motion_loss_mask"] = motion_loss_mask.to(torch.float32)
        if sample.get("grpo_event_metadata") is not None:
            out["grpo_event_metadata"] = dict(sample["grpo_event_metadata"])
        if motion_context_labels is not None:
            out["motion_context_labels"] = motion_context_labels.to(torch.long)
        if self.use_listener_audio:
            if audio_listen is None:
                raise RuntimeError("Listener conditioning is enabled but `audio_listen` is missing from Gesture_Dataset output.")
            out["audio_listen"] = audio_listen.to(torch.float32)
        return out


    def __getitem__(self, idx):
        """Sample a valid crop and assemble its training sequence."""
        return self._build_training_example(self._fetch_base_sample(idx))


class CoSpeechGestureCollator:
    def __init__(
        self,
        tokenizer,
        audio_sr=16000,
        dialog=True,
        listener_fusion_delta=False,
        num_motion_codebooks=6,
        task_type="gesture",
    ):
        """Configure task fields, padding and streaming filterbank extraction."""
        self.tokenizer = tokenizer
        self.audio_sr = int(audio_sr)
        self.dialog = bool(dialog)
        self.use_listener_audio = self.dialog or bool(listener_fusion_delta)
        self.num_motion_codebooks = int(num_motion_codebooks)
        self.task_type = str(task_type).lower()
        self.fbank = Fbank(FbankConfig(num_mel_bins=128))
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def _pad_bool_or_long(self, items, padding_value=0):
        """Pad variable-length tensors along the time dimension without changing dtype."""
        return rnn.pad_sequence(items, batch_first=True, padding_value=padding_value)

    def _build_fbank_batch(self, audio_list):
        """Extract padded filterbanks and their lengths from a batch of waveforms."""
        if len(audio_list) == 0:
            raise RuntimeError("No audio waveforms found in batch.")

        features = self.fbank.extract_batch(audio_list, sampling_rate=self.audio_sr)
        if isinstance(features, list):
            fbank_list = [feat if feat.ndim == 2 else feat.squeeze(0) for feat in features]
        else:
            fbank_list = [feat for feat in features]
        fbank_lens = torch.tensor([feat.size(0) for feat in fbank_list], dtype=torch.long)
        fbank_feature = rnn.pad_sequence(fbank_list, batch_first=True)
        return fbank_feature, fbank_lens


    def _pad_field(self, batch, name, padding, dtype):
        """Pad one batch field; boolean masks use integer padding before conversion back."""
        values = [item[name].to(torch.long) if dtype == torch.bool else item[name] for item in batch]
        return self._pad_bool_or_long(values, padding_value=padding).to(dtype)

    def __call__(self, batch):
        """Pad text/motion labels, extract FBanks, and include optional listener and loss masks."""
        fields = {
            'input_ids': (self.pad_token_id, torch.long), 'attention_mask': (0, torch.long),
            'audio_start_mask': (0, torch.bool), 'audio_end_mask': (0, torch.bool),
        }
        has_motion = _task_has_gesture(self.task_type) or _task_has_expression(self.task_type)
        if has_motion:
            fields.update(motion_slot_mask=(0, torch.bool), motion_labels=(-100, torch.long))
            if 'motion_context_labels' in batch[0]:
                fields['motion_context_labels'] = (-100, torch.long)
        if _task_has_asr(self.task_type):
            fields['labels'] = (-100, torch.long)
        out = {name: self._pad_field(batch, name, padding, dtype) for name, (padding, dtype) in fields.items()}
        out['fbank_feature'], out['fbank_feature_len'] = self._build_fbank_batch([item['audio'] for item in batch])
        if has_motion and any('motion_loss_mask' in item for item in batch):
            weights = [item.get('motion_loss_mask', torch.ones(item['motion_labels'].size(0), dtype=torch.float32))
                       for item in batch]
            out['motion_loss_mask'] = self._pad_bool_or_long(weights, padding_value=1.0).to(torch.float32)
        if self.use_listener_audio:
            out['audio_listen_fbank'], out['audio_listen_fbank_len'] = self._build_fbank_batch(
                [item['audio_listen'] for item in batch]
            )
        return out
