"""Stream 0.8-second audio blocks into gesture/face tokens, then decode complete sequences."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from contextlib import closing

import numpy as np
import torch
from lhotse import Fbank, FbankConfig
from transformers import AutoConfig, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.expression_io import (  # noqa: E402
    DEFAULT_FLAME_STATS_PATH,
    decode_expression_tokens_to_motioncode,
    load_flame_decoder,
    load_motion_stats as load_flame_stats,
    load_flame_style_features, resolve_style_flame_path,
)
from scripts.gesture_io import (  # noqa: E402
    DEFAULT_AUDIO_ENCODER_PATH,
    DEFAULT_BASE_LLM_PATH,
    DEFAULT_HANDS_VQ_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_STATS_DIR,
    DEFAULT_STATS_PREFIX,
    DEFAULT_STATS_SUFFIX,
    DEFAULT_UPPER_VQ_PATH,
    DEFAULT_VQ_PATH,
    build_fbank_batch,
    build_inline_exemplar_schedule,
    build_motion_features_from_pose,
    build_system_prompt,
    choose_device,
    choose_dtype,
    compute_num_windows,
    concat_audio_chunks,
    decode_motion_to_smplh,
    fit_audio_length,
    json_load_compat,
    load_audio,
    load_inline_exemplar_library,
    load_motion_decoders,
    resolve_rvq_context_frames,
    load_motion_stats as load_gesture_stats,
    maybe_load_legacy_qwen35_state_dict,
    resolve_speaker_id,
    reset_motion_weight_buffers,
    sample_from_logits,
    split_audio_into_chunks,
    str2bool,
)
from models.vq.precision import resolve_encoding_precision
from scripts.pose_bias import MODEL_PATH as POSE_BIAS_MODEL, pose_guide_schedule
from inline_motion_exemplar import EXEMPLAR_END_TOKEN, EXEMPLAR_START_TOKEN  # noqa: E402
from modeling_ellsa2 import CoSpeechGestureLM  # noqa: E402
from models.vq.streaming import make_decoder_stream, supports_strict_streaming


from release_paths import CHECKPOINT_ROOT
DEFAULT_FLAME_VQ_PATH = str(CHECKPOINT_ROOT / "tokenizers/flame.pth")
DEFAULT_INLINE_EXEMPLAR_LIBRARY = ""


@dataclass
class _StreamWindow:
    """Keep aligned audio and generated motion together when truncating history."""
    index: int
    speaker_audio: np.ndarray
    listener_audio: np.ndarray
    motion_codes: np.ndarray = None


def build_streaming_gesture_expression_sequence(
    num_history_windows: int,
    motion_slot_token: str,
    dialog: bool,
    motion_steps_per_window: int,
    speaker_id: str = "",
    prefix_motion_steps: int = 0,
    inline_exemplar_steps_by_window=None,
):
    """Build the alternating audio sequence with known motion and style placeholders."""
    system_prompt = build_system_prompt(
        "You are a co-speech gesture and facial expression generation model.",
        speaker_id,
    )
    guide_steps = inline_exemplar_steps_by_window or [0] * (num_history_windows + 1)
    if len(guide_steps) != num_history_windows + 1:
        raise ValueError("Inline exemplar guide list must cover history windows plus the target window.")
    audio_placeholders = "<|vision_start|><|vision_end|>" * (2 if dialog else 1)
    windows = []
    for idx, steps in enumerate(guide_steps):
        guide = ""
        if int(steps) > 0:
            guide = EXEMPLAR_START_TOKEN + motion_slot_token * int(steps) + EXEMPLAR_END_TOKEN
        # History includes known motion; the target block ends at audio for generation.
        motion = motion_slot_token * motion_steps_per_window if idx < num_history_windows else ""
        windows.append(guide + audio_placeholders + motion)
    body = motion_slot_token * int(prefix_motion_steps) + "".join(windows)
    stream_desc = "speaker and listener audio stream" if dialog else "speaker audio stream"
    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"Generate co-speech gesture and facial expression tokens from the interleaved {stream_desc}.<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{body}"
    )


def build_model_inputs(
    tokenizer,
    fbank_extractor: Fbank,
    speaker_audio: np.ndarray,
    listener_audio: np.ndarray,
    args,
    num_history_windows: int,
    speaker_id: str = "",
    prefix_motion_steps: int = 0,
    inline_exemplar_steps_by_window=None,
):
    """Tokenize placeholders and extract aligned speaker/listener filterbanks."""
    text = build_streaming_gesture_expression_sequence(
        num_history_windows=num_history_windows,
        motion_slot_token=args.motion_slot_token,
        dialog=args.dialog,
        motion_steps_per_window=args.motion_steps_per_window,
        speaker_id=speaker_id,
        prefix_motion_steps=prefix_motion_steps,
        inline_exemplar_steps_by_window=inline_exemplar_steps_by_window,
    )
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    motion_slot_id = tokenizer.convert_tokens_to_ids(args.motion_slot_token)
    if min(vision_start_id, vision_end_id, motion_slot_id) < 0:
        raise ValueError("Tokenizer is missing required placeholder tokens.")
    if args.inline_exemplar_enabled:
        for delimiter in (EXEMPLAR_START_TOKEN, EXEMPLAR_END_TOKEN):
            delimiter_ids = tokenizer(delimiter, add_special_tokens=False)["input_ids"]
            if len(delimiter_ids) != 1 or tokenizer.convert_ids_to_tokens(delimiter_ids[0]) != delimiter:
                raise ValueError(f"Inline exemplar delimiter is not one tokenizer token: {delimiter}")

    speaker_audio_tensor = torch.from_numpy(speaker_audio).float()
    fbank_feature, fbank_feature_len = build_fbank_batch(
        fbank_extractor,
        [speaker_audio_tensor],
        args.audio_sr,
    )

    batch = {
        "input_ids": input_ids.to(torch.long),
        "attention_mask": attention_mask.to(torch.long),
        "audio_start_mask": (input_ids == vision_start_id),
        "audio_end_mask": (input_ids == vision_end_id),
        "motion_slot_mask": (input_ids == motion_slot_id),
        "fbank_feature": fbank_feature,
        "fbank_feature_len": fbank_feature_len,
    }

    if args.dialog:
        listener_audio_tensor = torch.from_numpy(listener_audio).float()
        audio_listen_fbank, audio_listen_fbank_len = build_fbank_batch(
            fbank_extractor,
            [listener_audio_tensor],
            args.audio_sr,
        )
        batch["audio_listen_fbank"] = audio_listen_fbank
        batch["audio_listen_fbank_len"] = audio_listen_fbank_len

    return batch


def default_motion_level_weights(num_motion_codebooks: int):
    """Return the release's upper-body, hand and expression RVQ head weights."""
    base_weights = [
        1.0, 0.5, 0.25, 0.1, 0.05, 0.025,
        1.0, 0.5, 0.25, 0.1, 0.05, 0.025,
        1.0, 0.5, 0.25, 0.1, 1.0, 0.5,
    ]
    repeats = (int(num_motion_codebooks) + len(base_weights) - 1) // len(base_weights)
    return (base_weights * repeats)[: int(num_motion_codebooks)]


def build_model_args(args):
    """Translate inference options and the selected RVQ layout into model arguments."""
    motion_level_weights = (
        args.motion_level_weights
        if args.motion_level_weights is not None
        else default_motion_level_weights(args.num_motion_codebooks)
    )
    if len(motion_level_weights) < args.num_motion_codebooks:
        raise ValueError(
            "`motion_level_weights` must contain at least "
            f"{args.num_motion_codebooks} values, got {len(motion_level_weights)}."
        )
    motion_level_weights = motion_level_weights[: args.num_motion_codebooks]
    return SimpleNamespace(
        base_llm_path=args.base_llm_path,
        llm_type="Qwen",
        encoder_type="transformer_v2",
        connector_type="MLP",
        audio_encoder_path=args.audio_encoder_path,
        freeze_encoder=True,
        freeze_llm=False,
        dialog=bool(args.dialog),
        task_type="gesture_expression",
        motion_vocab_size=args.motion_vocab_size,
        num_motion_codebooks=args.num_motion_codebooks,
        gesture_num_codebooks=args.gesture_num_codebooks,
        expression_num_codebooks=args.expression_num_codebooks,
        motion_steps_per_window=args.motion_steps_per_window,
        expression_steps_per_window=args.expression_steps_per_window,
        interleave_motion_parts=bool(args.interleave_motion_parts),
        audio_tokens_per_window=args.audio_tokens_per_window,
        connector_seg_size=args.connector_seg_size,
        connector_hid_size=args.connector_hid_size,
        motion_level_weights=motion_level_weights,
        debug_mode=bool(args.debug_mode),
    )


def inspect_checkpoint_motion_codebooks(model_path: str):
    """Infer the RVQ head count from checkpoint weights without loading the model."""
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        return None

    from safetensors import safe_open

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        motion_head_ids = sorted(
            int(key.split(".")[1])
            for key in f.keys()
            if key.startswith("motion_heads.") and key.endswith(".weight")
        )
    if not motion_head_ids:
        return 0
    expected_ids = list(range(motion_head_ids[-1] + 1))
    if motion_head_ids != expected_ids:
        raise RuntimeError(f"Checkpoint motion_heads are not contiguous: {motion_head_ids}")
    return len(motion_head_ids)


def load_ellsa2_model(args, device: torch.device, torch_dtype: torch.dtype):
    """Load the configured backbone and compatible ELLSA2 checkpoint for inference."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.torch_dtype = torch_dtype
    config.attn_implementation = args.attn_implementation if device.type == "cuda" else "eager"
    model_args = build_model_args(args)
    checkpoint_codebooks = inspect_checkpoint_motion_codebooks(args.model_path)
    if checkpoint_codebooks is not None and checkpoint_codebooks != model_args.num_motion_codebooks:
        raise RuntimeError(
            "Checkpoint motion layout does not match gesture_expression inference args: "
            f"checkpoint has {checkpoint_codebooks} motion codebooks, "
            f"but inference requested {model_args.num_motion_codebooks} "
            f"({model_args.gesture_num_codebooks} gesture + {model_args.expression_num_codebooks} expression). "
            "Do not run a gesture-only checkpoint with gesture_expression inference."
        )

    remapped_state_dict = maybe_load_legacy_qwen35_state_dict(args.model_path)
    if remapped_state_dict is None:
        model, loading_info = CoSpeechGestureLM.from_pretrained(
            args.model_path,
            config=config,
            model_args=model_args,
            torch_dtype=torch_dtype,
            output_loading_info=True,
        )
        loading_problems = {
            key: value
            for key, value in loading_info.items()
            if key in {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"} and value
        }
        if loading_problems:
            raise RuntimeError(f"Checkpoint did not load cleanly: {loading_problems}")
    else:
        print("Detected legacy Qwen3.5 MoE checkpoint key layout; loading with remapped state_dict.")
        model = CoSpeechGestureLM(config, model_args)
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Remapped checkpoint did not match model state_dict: "
                f"missing={list(missing_keys)[:20]}, unexpected={list(unexpected_keys)[:20]}"
            )

    reset_motion_weight_buffers(model, model_args)
    model.to(device=device, dtype=torch_dtype if device.type == "cuda" else torch.float32)
    model.eval()
    return tokenizer, model


def mask_expression_padding_steps(motion_ids: torch.Tensor, args) -> torch.Tensor:
    """Set unused expression heads to code 0, matching training-time clamp(min=0) context."""
    motion_ids = motion_ids.clone()
    if args.interleave_motion_parts:
        return motion_ids
    if args.expression_steps_per_window >= args.motion_steps_per_window:
        return motion_ids
    for start in range(0, motion_ids.size(0), args.motion_steps_per_window):
        pad_start = start + args.expression_steps_per_window
        pad_end = min(start + args.motion_steps_per_window, motion_ids.size(0))
        if pad_start < pad_end:
            motion_ids[pad_start:pad_end, args.gesture_num_codebooks :] = 0
    return motion_ids


def gesture_steps_per_window(args) -> int:
    """Compute the number of gesture steps in one streaming window."""
    if args.interleave_motion_parts:
        return (args.motion_steps_per_window - args.expression_steps_per_window) // 2
    return args.motion_steps_per_window


def interleave_active_codebooks(args, step_idx: int):
    """Place upper-body, hand and expression codes into their active interleaved rows."""
    if not args.interleave_motion_parts:
        return range(args.num_motion_codebooks)
    if args.motion_tokenizer_type != "upper_hands":
        raise ValueError("`interleave_motion_parts` only supports motion_tokenizer_type=upper_hands.")
    step_in_window = step_idx % args.motion_steps_per_window
    cursor = 0
    for frame_idx in range(gesture_steps_per_window(args)):
        if step_in_window == cursor:
            return range(0, 6)
        cursor += 1
        if step_in_window == cursor:
            return range(6, 12)
        cursor += 1
        if frame_idx < args.expression_steps_per_window:
            if step_in_window == cursor:
                return range(args.gesture_num_codebooks, args.num_motion_codebooks)
            cursor += 1
    return range(0, 0)


def fold_interleaved_gesture_expression_tokens(motion_token_ids: torch.Tensor, args):
    """Fold interleaved upper-body/hand/expression rows into per-frame code groups."""
    if not args.interleave_motion_parts:
        return motion_token_ids
    gesture_steps = gesture_steps_per_window(args)
    gesture_rows = []
    expression_rows = []
    for start in range(0, motion_token_ids.size(0), args.motion_steps_per_window):
        window = motion_token_ids[start : start + args.motion_steps_per_window]
        if window.size(0) != args.motion_steps_per_window:
            raise ValueError("Interleaved gesture-expression tokens require full windows before decoding.")
        cursor = 0
        for frame_idx in range(gesture_steps):
            gesture_row = torch.zeros(
                (args.gesture_num_codebooks,),
                dtype=motion_token_ids.dtype,
                device=motion_token_ids.device,
            )
            gesture_row[:6] = window[cursor, :6]
            cursor += 1
            gesture_row[6:12] = window[cursor, 6:12]
            cursor += 1
            gesture_rows.append(gesture_row)
            if frame_idx < args.expression_steps_per_window:
                expression_rows.append(window[cursor, args.gesture_num_codebooks :].clone())
                cursor += 1
    gesture_tokens = torch.stack(gesture_rows, dim=0) if gesture_rows else motion_token_ids.new_zeros((0, args.gesture_num_codebooks))
    expression_tokens = torch.stack(expression_rows, dim=0) if expression_rows else motion_token_ids.new_zeros((0, args.expression_num_codebooks))
    return gesture_tokens, expression_tokens


def split_gesture_expression_tokens(motion_token_ids: torch.Tensor, args):
    """Separate generated codes into gesture and expression token sequences."""
    if args.interleave_motion_parts:
        return fold_interleaved_gesture_expression_tokens(motion_token_ids, args)
    gesture_tokens = motion_token_ids[:, : args.gesture_num_codebooks].clone()

    expression_chunks = []
    for start in range(0, motion_token_ids.size(0), args.motion_steps_per_window):
        expr_end = min(start + args.expression_steps_per_window, motion_token_ids.size(0))
        if expr_end > start:
            expression_chunks.append(
                motion_token_ids[start:expr_end, args.gesture_num_codebooks :].clone()
            )
    if expression_chunks:
        expression_tokens = torch.cat(expression_chunks, dim=0)
    else:
        expression_tokens = torch.zeros(
            (0, args.expression_num_codebooks),
            dtype=motion_token_ids.dtype,
            device=motion_token_ids.device,
        )
    return gesture_tokens, expression_tokens


@torch.no_grad()
def generate_motion_tokens_with_history(
    ellsa2_model,
    model_inputs,
    known_motion_ids: torch.Tensor,
    sampling_args,
):
    """Generate the current window conditioned on known motion slots and audio history."""
    if known_motion_ids.ndim != 2 or known_motion_ids.size(-1) != ellsa2_model.num_motion_codebooks:
        raise ValueError(
            f"`known_motion_ids` must have shape (T, {ellsa2_model.num_motion_codebooks}), "
            f"got {tuple(known_motion_ids.shape)}."
        )

    expected_history_steps = int(model_inputs["motion_slot_mask"].long().sum(dim=1).item())
    if known_motion_ids.size(0) != expected_history_steps:
        raise ValueError(
            f"Known motion length {known_motion_ids.size(0)} does not match history motion slots "
            f"{expected_history_steps}."
        )

    motion_input_embeds = None
    if known_motion_ids.numel() > 0:
        motion_input_embeds = ellsa2_model._motion_ids_to_embeds(
            known_motion_ids.unsqueeze(0).to(model_inputs["input_ids"].device)
        )

    prefix_embeds, expanded_attention_mask, _ = ellsa2_model.prepare_inputs_labels_for_speech(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        audio_start_mask=model_inputs["audio_start_mask"],
        audio_end_mask=model_inputs["audio_end_mask"],
        motion_slot_mask=model_inputs["motion_slot_mask"],
        fbank_feature=model_inputs["fbank_feature"],
        fbank_feature_len=model_inputs["fbank_feature_len"],
        audio_listen_fbank=model_inputs.get("audio_listen_fbank"),
        audio_listen_fbank_len=model_inputs.get("audio_listen_fbank_len"),
        motion_input_embeds=motion_input_embeds,
    )

    generated_steps = []
    current_embeds = prefix_embeds
    current_attention = expanded_attention_mask

    for _ in range(ellsa2_model.motion_steps_per_window):
        with ellsa2_model.maybe_autocast():
            # Motion heads consume backbone features, never text-vocabulary logits.
            llm_outputs = ellsa2_model.base_llm.model(
                inputs_embeds=current_embeds,
                attention_mask=current_attention,
                use_cache=False,
                return_dict=True,
            )

        last_hidden = llm_outputs.last_hidden_state[:, -1:, :]
        with ellsa2_model.maybe_autocast():
            step_logits = [head(last_hidden) for head in ellsa2_model.motion_heads]

        active_codebooks = interleave_active_codebooks(sampling_args, len(generated_steps))
        step_pred = torch.zeros(
            (last_hidden.size(0), ellsa2_model.num_motion_codebooks),
            dtype=torch.long,
            device=last_hidden.device,
        )
        for level in active_codebooks:
            step_pred[:, level] = sample_from_logits(
                step_logits[level].squeeze(1).float(),
                temperature=sampling_args.temperature,
                top_k=sampling_args.top_k,
                top_p=sampling_args.top_p,
            )
        generated_steps.append(step_pred)

        step_embed = ellsa2_model._motion_ids_to_embeds(step_pred.unsqueeze(1))
        current_embeds = torch.cat([current_embeds, step_embed], dim=1)
        current_attention = torch.cat(
            [
                current_attention,
                torch.ones(
                    (current_attention.size(0), 1),
                    dtype=current_attention.dtype,
                    device=current_attention.device,
                ),
            ],
            dim=1,
        )

    return torch.cat(generated_steps, dim=0)


@torch.no_grad()
def encode_pose_tokens(pose_path, start_idx, length, motion_decoders, gesture_stats, args):
    """Encode upper body and hands; absent expression is masked, never a neutral face."""
    if args.interleave_motion_parts:
        raise ValueError("Inline gesture_expression exemplars currently require interleave_motion_parts=false.")
    if args.motion_tokenizer_type != "upper_hands":
        raise ValueError("Inline gesture_expression exemplars require motion_tokenizer_type=upper_hands.")
    gesture_features = build_motion_features_from_pose(
        pose_path=pose_path,
        start_idx=start_idx,
        length=length,
        stats=gesture_stats,
        include_translation=False,
    )
    upper_encoder = motion_decoders["upper"]
    hands_encoder = motion_decoders["hands"]
    upper_device = next(upper_encoder.parameters()).device
    hands_device = next(hands_encoder.parameters()).device
    upper_codes, _ = upper_encoder.encode(gesture_features["upper"].unsqueeze(0).to(upper_device))
    hands_codes, _ = hands_encoder.encode(gesture_features["hands"].unsqueeze(0).to(hands_device))
    gesture_codes = torch.cat(
        [
            upper_codes[0, :, :6].detach().cpu().long(),
            hands_codes[0, :, :6].detach().cpu().long(),
        ],
        dim=1,
    )
    if int(gesture_codes.size(1)) != int(args.gesture_num_codebooks):
        raise ValueError(
            f"Encoded exemplar has {gesture_codes.size(1)} gesture codebooks; "
            f"expected {args.gesture_num_codebooks}."
        )
    combined = torch.full(
        (int(gesture_codes.size(0)), int(args.num_motion_codebooks)),
        -100,
        dtype=torch.long,
    )
    combined[:, : args.gesture_num_codebooks] = gesture_codes
    return combined


@torch.no_grad()
def encode_motion_style_tokens(pose_path, flame_path, start_idx, length, motion_decoders,
                               gesture_stats, flame_decoder, flame_mean, flame_std, args):
    """Style conditioning includes expression, unlike pose-only inline guides."""
    combined = encode_pose_tokens(pose_path, start_idx, length, motion_decoders, gesture_stats, args)
    if args.motion_style_expression:
        path = resolve_style_flame_path(pose_path, flame_path)
        features = load_flame_style_features(path, start_idx, length, flame_mean, flame_std, args.flame_fps)
        codes, _ = flame_decoder.encode(features.unsqueeze(0).to(next(flame_decoder.parameters()).device))
        face = codes[0, :, :args.expression_num_codebooks].detach().cpu().long()
        if face.shape != combined[:, args.gesture_num_codebooks:].shape:
            raise ValueError('Gesture and expression style token lengths do not match')
        combined[:, args.gesture_num_codebooks:] = face
    return combined


@torch.no_grad()
def iter_gesture_expression_tokens(
    speaker_audio: np.ndarray,
    listener_audio: np.ndarray,
    tokenizer,
    ellsa2_model,
    fbank_extractor: Fbank,
    args,
    speaker_id: str = "",
    speaker_motion_ids: torch.Tensor = None,
    motion_decoders=None,
    gesture_stats=None,
    word_file: str = "",
    segment_start_sec: float = 0.0,
    audio_windows=None,
):
    """Yield offline or live audio windows with bounded, freshly encoded history."""
    if audio_windows is None:
        samples_per_window = int(round(args.window_sec * args.audio_sr))
        num_windows = compute_num_windows(
            speaker_audio=speaker_audio,
            listener_audio=listener_audio,
            samples_per_window=samples_per_window,
            dialog=args.dialog,
            num_windows=args.num_windows,
            max_windows=args.max_windows,
        )
        target_samples = num_windows * samples_per_window
        speaker_audio = fit_audio_length(speaker_audio, target_samples)
        if args.dialog:
            listener_audio = fit_audio_length(listener_audio, target_samples)

        speaker_chunks = split_audio_into_chunks(speaker_audio, samples_per_window)
        listener_chunks = split_audio_into_chunks(listener_audio, samples_per_window) if args.dialog else []
        if args.pose_guide:
            inline_schedule = pose_guide_schedule(
                args.pose_guide, args.pose_guide_start, args.pose_guide_time, num_windows)
        else:
            inline_library = (load_inline_exemplar_library(args.inline_exemplar_library, args)
                              if args.inline_exemplar_enabled else {})
            inline_schedule = build_inline_exemplar_schedule(
                word_file, num_windows, segment_start_sec, args, inline_library)

        audio_windows = zip(speaker_chunks, listener_chunks if args.dialog else [None] * num_windows)
    else:
        if args.pose_guide or args.inline_exemplar_enabled:
            raise ValueError("Live microphone inference does not support scheduled pose guides.")
        num_windows = None
        inline_schedule = {}

    max_history_chunks = max(1, int(round(args.max_history_sec / args.window_sec)))
    reset_history_chunks = max(1, int(args.reset_history_chunks))
    if reset_history_chunks >= max_history_chunks:
        raise ValueError(
            f"`reset_history_chunks` ({reset_history_chunks}) must be smaller than "
            f"the max history chunks ({max_history_chunks}) to leave room for the current window."
        )

    model_device = next(ellsa2_model.parameters()).device
    if speaker_motion_ids is None:
        speaker_motion_ids = torch.zeros((0, args.num_motion_codebooks), dtype=torch.long)
    if speaker_motion_ids.ndim != 2 or speaker_motion_ids.size(1) != args.num_motion_codebooks:
        raise ValueError(
            f"`speaker_motion_ids` must have shape (T, {args.num_motion_codebooks}), "
            f"got {tuple(speaker_motion_ids.shape)}."
        )
    history = []
    inline_exemplar_code_cache = {}
    inline_guide_schedule = {}
    inline_exemplar_log = []

    for chunk_idx, (speaker_chunk, listener_chunk) in enumerate(audio_windows):
        current_window = _StreamWindow(chunk_idx, speaker_chunk, listener_chunk)
        # Keep completed windows before appending the pending audio window.
        if len(history) + 1 > max_history_chunks:
            history = history[-reset_history_chunks:]
            print(
                f"Refreshed context before chunk {chunk_idx + 1} | "
                f"retained_chunks={len(history)} | first_history_chunk={history[0].index + 1}"
            )
        history.append(current_window)

        if chunk_idx in inline_schedule:
            event = inline_schedule[chunk_idx]
            reference = event["reference"]
            event_uid = str(reference["event_uid"])
            if event_uid not in inline_exemplar_code_cache:
                if motion_decoders is None or gesture_stats is None:
                    raise ValueError("Gesture encoders and stats are required for pose guides.")
                inline_exemplar_code_cache[event_uid] = encode_pose_tokens(
                    pose_path=reference["pose_file"],
                    start_idx=int(reference["causal_reference_start_frame"]),
                    length=int(reference["causal_reference_length_frames"]),
                    motion_decoders=motion_decoders,
                    gesture_stats=gesture_stats,
                    args=args,
                )
            reference_codes = inline_exemplar_code_cache[event_uid]
            expected_steps = int(args.inline_exemplar_reference_frames // args.motion_downsample_factor)
            if int(reference_codes.size(0)) != expected_steps:
                raise ValueError(
                    f"Inline exemplar encoded to {reference_codes.size(0)} steps; expected {expected_steps}."
                )
            reference_chunks = reference_codes.split(args.motion_steps_per_window, dim=0)
            for offset, guide_chunk in enumerate(reference_chunks):
                guide_chunk_idx = chunk_idx + offset
                if guide_chunk_idx >= num_windows:
                    raise ValueError("Inline exemplar copy interval exceeds the inference stream.")
                if guide_chunk_idx in inline_guide_schedule:
                    raise ValueError("Inline exemplar copy intervals overlap.")
                inline_guide_schedule[guide_chunk_idx] = {
                    "event": event,
                    "offset": offset,
                    "motion_codes": guide_chunk,
                }
            inline_exemplar_log.append({
                "insert_chunk": int(chunk_idx),
                "insert_time_sec": float(segment_start_sec) + chunk_idx * args.window_sec,
                "action_family": event["action_family"],
                "query_text": event["query_text"],
                "matched_text": event["matched_text"],
                "reference_event_uid": event_uid,
                "reference_pose_file": reference["pose_file"],
                "reference_flame_file": reference.get("flame_file", ""),
                "reference_start_frame": int(reference["causal_reference_start_frame"]),
                "reference_length_frames": int(reference["causal_reference_length_frames"]),
                "reference_anchor_frame": int(reference["reference_anchor_frame"]),
                "guide_chunk_count": len(reference_chunks),
                "guide_motion_steps_per_chunk": int(args.motion_steps_per_window),
                "guide_num_codebooks": int(reference_codes.size(1)),
                "guide_valid_gesture_codebooks": int(args.gesture_num_codebooks),
                "guide_expression_codebooks_masked": True,
                "copy_mode": args.inline_exemplar_copy_mode,
                "hard_copy_core_chunk_offsets": (
                    [1, 2, 3, 4, 5, 6] if args.inline_exemplar_copy_mode == "hard_core" else []
                ),
            })

        known_parts = [speaker_motion_ids.cpu().long()]
        inline_exemplar_steps_by_window = []
        for window in history[:-1]:
            guide = inline_guide_schedule.get(window.index)
            if guide is not None:
                known_parts.append(guide["motion_codes"].long())
                inline_exemplar_steps_by_window.append(int(guide["motion_codes"].size(0)))
            else:
                inline_exemplar_steps_by_window.append(0)
            known_parts.append(torch.from_numpy(window.motion_codes).long())
        current_guide = inline_guide_schedule.get(chunk_idx)
        if current_guide is not None:
            known_parts.append(current_guide["motion_codes"].long())
            inline_exemplar_steps_by_window.append(int(current_guide["motion_codes"].size(0)))
        else:
            inline_exemplar_steps_by_window.append(0)
        known_motion_ids = torch.cat(known_parts, dim=0)

        # Re-extract fbank and re-encode this crop from its first retained window;
        # audio embeddings and LLM states from the discarded left context are not reused.
        model_inputs = build_model_inputs(
            tokenizer=tokenizer,
            fbank_extractor=fbank_extractor,
            speaker_audio=concat_audio_chunks([window.speaker_audio for window in history]),
            listener_audio=concat_audio_chunks([window.listener_audio for window in history])
            if args.dialog
            else np.zeros(0, dtype=np.float32),
            args=args,
            num_history_windows=len(history) - 1,
            speaker_id=speaker_id,
            prefix_motion_steps=int(speaker_motion_ids.size(0)),
            inline_exemplar_steps_by_window=inline_exemplar_steps_by_window,
        )
        model_inputs = {
            key: value.to(model_device) if isinstance(value, torch.Tensor) else value
            for key, value in model_inputs.items()
        }

        hard_copy_current = bool(
            current_guide is not None
            and args.inline_exemplar_copy_mode == "hard_core"
            and 1 <= int(current_guide["offset"]) <= 6
        )
        if hard_copy_current:
            current_chunk_motion = current_guide["motion_codes"].cpu().numpy()
        else:
            current_chunk_motion = generate_motion_tokens_with_history(
                ellsa2_model=ellsa2_model,
                model_inputs=model_inputs,
                known_motion_ids=known_motion_ids.to(model_device),
                sampling_args=args,
            ).detach().cpu()
            current_chunk_motion = mask_expression_padding_steps(current_chunk_motion, args).numpy()

        current_window.motion_codes = current_chunk_motion

        print(
            f"Decoded chunk {chunk_idx + 1}/{num_windows or 'live'} | "
            f"context_chunks={len(history)} | "
            f"history_steps={known_motion_ids.size(0)} | "
            f"exemplar_steps={inline_exemplar_steps_by_window[-1]} | "
            f"hard_copy_core={hard_copy_current}"
        )
        yield torch.from_numpy(current_chunk_motion).long()

    if args.inline_exemplar_enabled:
        args._last_inline_exemplar_log = inline_exemplar_log


def stream_generate_gesture_expression_tokens(*args, **kwargs):
    """Collect the same causal token iterator for the offline CLI."""
    chunks = list(iter_gesture_expression_tokens(*args, **kwargs))
    return torch.cat(chunks, dim=0), len(chunks)


def build_code_chunk_ranges(num_windows: int, chunk_size: int):
    """Compute gesture/expression token ranges corresponding to decoded output chunks."""
    return np.asarray(
        [(idx * chunk_size, (idx + 1) * chunk_size) for idx in range(num_windows)],
        dtype=np.int32,
    )


def save_split_outputs(output_path, smplh_dict, motioncode, combined_tokens, gesture_tokens, expression_tokens, args, num_windows):
    """Decode and save gesture and expression predictions with streaming chunk metadata."""
    root, _ = os.path.splitext(output_path)
    smplh_path = f"{root}_smplh.npz"
    flame_path = f"{root}_flame.npz"

    np.savez(
        smplh_path,
        body_pose=smplh_dict["body_pose"],
        left_hand_pose=smplh_dict["left_hand_pose"],
        right_hand_pose=smplh_dict["right_hand_pose"],
        global_orient=smplh_dict["global_orient"],
        translation=smplh_dict["translation"],
        model="smplh",
        gender="neutral",
        mocap_frame_rate=args.pose_fps,
        rvq_code_idx=gesture_tokens.cpu().numpy().astype(np.int64, copy=False),
    )

    np.savez_compressed(
        flame_path,
        motioncode=motioncode,
        expression=motioncode[:, :100],
        valid_mask=np.ones((motioncode.shape[0],), dtype=np.uint8),
        motion_fps=np.asarray(args.flame_fps, dtype=np.int32),
        rvq_motioncode_unils_108=motioncode,
        rvq_code_idx=expression_tokens.cpu().numpy().astype(np.int64, copy=False),
        rvq_code_chunk_ranges=build_code_chunk_ranges(num_windows, args.expression_steps_per_window),
        rvq_num_chunks=np.asarray(num_windows, dtype=np.int32),
        rvq_chunk_num_frames=np.asarray(args.expression_steps_per_window, dtype=np.int32),
        rvq_codec_ckpt=np.asarray(str(args.vqvae_flame_path)),
        rvq_stats_path=np.asarray(str(args.flame_stats_path)),
        model=np.asarray("flame"),
    )
    return smplh_path, flame_path


@torch.no_grad()
def infer_single_audio(
    speaker_audio_path: str,
    output_path: str,
    tokenizer,
    ellsa2_model,
    motion_decoders,
    gesture_stats,
    flame_decoder,
    flame_mean,
    flame_std,
    fbank_extractor: Fbank,
    args,
    listener_audio_path: str = "",
    speaker_id: str = "",
    motion_style_pose_path: str = "",
    motion_style_flame_path: str = "",
    motion_style_start: int = 0,
    motion_style_length: int = 612,
    word_file: str = "",
    segment_start_sec: float = 0.0,
):
    """Load one audio job, generate motion tokens and save optional decoded/rendered outputs."""
    speaker_audio = load_audio(speaker_audio_path, args.audio_sr)
    if args.dialog:
        listener_audio = load_audio(listener_audio_path, args.audio_sr) if listener_audio_path else np.zeros_like(speaker_audio)
    else:
        listener_audio = np.zeros(0, dtype=np.float32)

    speaker_motion_ids = None
    if args.speaker_id_type == "motion":
        speaker_motion_ids = encode_motion_style_tokens(
            pose_path=motion_style_pose_path, flame_path=motion_style_flame_path,
            start_idx=int(motion_style_start), length=int(motion_style_length),
            motion_decoders=motion_decoders, gesture_stats=gesture_stats,
            flame_decoder=flame_decoder, flame_mean=flame_mean, flame_std=flame_std, args=args,
        )
        complete_steps = (speaker_motion_ids.size(0) // args.motion_steps_per_window) * args.motion_steps_per_window
        speaker_motion_ids = speaker_motion_ids[:complete_steps]
        speaker_id = ""

    token_stream = iter_gesture_expression_tokens(
        speaker_audio=speaker_audio,
        listener_audio=listener_audio,
        tokenizer=tokenizer,
        ellsa2_model=ellsa2_model,
        fbank_extractor=fbank_extractor,
        args=args,
        speaker_id=speaker_id,
        speaker_motion_ids=speaker_motion_ids,
        motion_decoders=motion_decoders,
        gesture_stats=gesture_stats,
        word_file=word_file,
        segment_start_sec=segment_start_sec,
    )
    decoded_motion = None
    chunks = []
    args.rvq_active_decode_mode = "whole"
    if use_streaming_rvq(motion_decoders, flame_decoder, args):
        args.rvq_active_decode_mode = "stream"
        decoder = StreamingMotionDecoder(motion_decoders, gesture_stats,
                                         flame_decoder, flame_mean, flame_std, args)
        blocks = []
        with closing(token_stream):
            for tokens in token_stream:
                chunks.append(tokens)
                poses, face = decoder.decode(tokens)
                blocks.append({**poses, "motioncode": face})
                print(f"Emitted RVQ block {len(chunks)} | frames={len(face)} | right_context=0", flush=True)
        combined = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
        face = combined.pop("motioncode")
        decoded_motion = combined, face
    else:
        with closing(token_stream):
            chunks = list(token_stream)
    combined_tokens = torch.cat(chunks)
    decode_and_save_tokens(
        combined_tokens, output_path, motion_decoders, gesture_stats,
        flame_decoder, flame_mean, flame_std, args, decoded_motion=decoded_motion,
    )


def use_streaming_rvq(motion_decoders, flame_decoder, args):
    """Use strict streaming for compatible causal checkpoints by default."""
    mode = getattr(args, "rvq_decode_mode", "auto")
    if mode == "whole":
        return False
    compatible = (args.motion_tokenizer_type == "upper_hands"
                  and args.motion_steps_per_window == args.expression_steps_per_window
                  and bool(motion_decoders) and flame_decoder is not None
                  and all(supports_strict_streaming(model) for model in motion_decoders.values())
                  and supports_strict_streaming(flame_decoder))
    if mode == "stream" and not compatible:
        raise ValueError("Strict streaming requires causal upper/hands/FLAME RVQs with finite context caps and aligned steps")
    return compatible


class StreamingMotionDecoder:
    """Decode only the new block while retaining bounded RVQ latent history."""
    def __init__(self, motion_decoders, gesture_stats, flame_decoder, flame_mean, flame_std, args):
        assert use_streaming_rvq(motion_decoders, flame_decoder, args)
        self.motion_decoders = {part: make_decoder_stream(model) for part, model in motion_decoders.items()}
        self.flame_decoder = make_decoder_stream(flame_decoder)
        self.gesture_stats, self.flame_mean, self.flame_std = gesture_stats, flame_mean, flame_std
        self.args = args

    @torch.no_grad()
    def decode(self, tokens):
        assert len(tokens) == self.args.motion_steps_per_window
        return decode_tokens(tokens, self.motion_decoders, self.gesture_stats,
                             self.flame_decoder, self.flame_mean, self.flame_std, self.args)


@torch.no_grad()
def decode_tokens(combined_tokens, motion_decoders, gesture_stats,
                  flame_decoder, flame_mean, flame_std, args):
    """Decode tokens through either complete-sequence or stateful RVQ decoders."""
    gesture_tokens, expression_tokens = split_gesture_expression_tokens(combined_tokens, args)
    smplh_dict = decode_motion_to_smplh(
        motion_token_ids=gesture_tokens.unsqueeze(0),
        motion_decoders=motion_decoders,
        stats=gesture_stats,
        motion_vocab_size=args.motion_vocab_size,
        motion_tokenizer_type=args.motion_tokenizer_type,
    )
    motioncode = decode_expression_tokens_to_motioncode(
        motion_token_ids=expression_tokens,
        flame_decoder=flame_decoder,
        motion_mean=flame_mean,
        motion_std=flame_std,
        motion_vocab_size=args.motion_vocab_size,
    ).astype(np.float32, copy=False)

    return smplh_dict, motioncode


def decode_and_save_tokens(combined_tokens, output_path, motion_decoders, gesture_stats,
                           flame_decoder, flame_mean, flame_std, args, *, decoded_motion=None):
    """Save full decoding, or reuse the exact committed motion shown by the demo."""
    num_windows = len(combined_tokens) // args.motion_steps_per_window
    gesture_tokens, expression_tokens = split_gesture_expression_tokens(combined_tokens, args)

    if decoded_motion is None:
        decoded_motion = decode_tokens(combined_tokens, motion_decoders, gesture_stats,
                                       flame_decoder, flame_mean, flame_std, args)
    smplh_dict, motioncode = decoded_motion
    decode_metadata = {
        "rvq_encoder_dtype": np.asarray(getattr(args, "rvq_encoder_dtype", "float32")),
        "rvq_quantizer_distance_dtype": np.asarray(getattr(args, "rvq_quantizer_distance_dtype", "float32")),
        "motion_style_expression": np.asarray(bool(getattr(args, "motion_style_expression", False))
                                                   and getattr(args, "speaker_id_type", "text") == "motion"),
        "rvq_decode_mode": np.asarray(getattr(args, "rvq_active_decode_mode", "whole")),
        "rvq_gesture_context_frames": np.asarray(resolve_rvq_context_frames(args, "gesture"), dtype=np.int32),
        "rvq_expression_context_frames": np.asarray(resolve_rvq_context_frames(args, "expression"), dtype=np.int32),
    }
    if hasattr(args, "decode_window_blocks"):
        decode_metadata["rvq_decode_window_blocks"] = np.asarray(args.decode_window_blocks, dtype=np.int32)
    if hasattr(args, "boundary_smoothing_frames"):
        decode_metadata["motion_boundary_smoothing_frames"] = np.asarray(args.boundary_smoothing_frames, dtype=np.int32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        body_pose=smplh_dict["body_pose"],
        left_hand_pose=smplh_dict["left_hand_pose"],
        right_hand_pose=smplh_dict["right_hand_pose"],
        global_orient=smplh_dict["global_orient"],
        translation=smplh_dict["translation"],
        gender=np.asarray("neutral"),
        mocap_frame_rate=np.asarray(args.pose_fps, dtype=np.int32),
        motioncode=motioncode,
        expression=motioncode[:, :100],
        valid_mask=np.ones((motioncode.shape[0],), dtype=np.uint8),
        motion_fps=np.asarray(args.flame_fps, dtype=np.int32),
        rvq_motioncode_unils_108=motioncode,
        rvq_combined_code_idx=combined_tokens.cpu().numpy().astype(np.int64, copy=False),
        rvq_gesture_code_idx=gesture_tokens.cpu().numpy().astype(np.int64, copy=False),
        rvq_expression_code_idx=expression_tokens.cpu().numpy().astype(np.int64, copy=False),
        rvq_combined_code_chunk_ranges=build_code_chunk_ranges(num_windows, args.motion_steps_per_window),
        rvq_expression_code_chunk_ranges=build_code_chunk_ranges(num_windows, args.expression_steps_per_window),
        rvq_num_chunks=np.asarray(num_windows, dtype=np.int32),
        rvq_gesture_chunk_num_frames=np.asarray(gesture_steps_per_window(args), dtype=np.int32),
        rvq_expression_chunk_num_frames=np.asarray(args.expression_steps_per_window, dtype=np.int32),
        rvq_gesture_codec_type=np.asarray(str(args.motion_tokenizer_type)),
        rvq_flame_codec_ckpt=np.asarray(str(args.vqvae_flame_path)),
        rvq_flame_stats_path=np.asarray(str(args.flame_stats_path)),
        model=np.asarray("smplh+flame"),
        **decode_metadata,
    )

    if args.save_motion_tokens:
        root, _ = os.path.splitext(output_path)
        np.save(f"{root}_combined_motion_tokens.npy", combined_tokens.cpu().numpy())
        np.save(f"{root}_gesture_tokens.npy", gesture_tokens.cpu().numpy())
        np.save(f"{root}_expression_tokens.npy", expression_tokens.cpu().numpy())

    if args.save_split_outputs:
        smplh_path, flame_path = save_split_outputs(
            output_path=output_path,
            smplh_dict=smplh_dict,
            motioncode=motioncode,
            combined_tokens=combined_tokens,
            gesture_tokens=gesture_tokens,
            expression_tokens=expression_tokens,
            args=args,
            num_windows=num_windows,
        )
        print(f"Saved split outputs: gesture={smplh_path} | expression={flame_path}")

    if args.inline_exemplar_enabled:
        root, _ = os.path.splitext(output_path)
        with open(f"{root}_inline_exemplar.json", "w", encoding="utf-8") as f:
            json.dump(getattr(args, "_last_inline_exemplar_log", []), f, ensure_ascii=False, indent=2)

    print(
        f"Inference finished | "
        f"windows={num_windows} | combined_steps={combined_tokens.shape[0]} | "
        f"gesture_frames={smplh_dict['translation'].shape[0]} | expression_frames={motioncode.shape[0]} | "
        f"output={output_path}"
    )


def iter_jobs(args):
    """Read inference jobs from a manifest or construct a single command-line job."""
    if args.audio_path:
        output_path = args.output_path
        if not output_path:
            output_dir = args.output_dir or os.path.dirname(args.audio_path) or "."
            output_path = os.path.join(output_dir, f"{Path(args.audio_path).stem}_gesture_expression.npz")
        yield {
            "audio_path": args.audio_path,
            "audio_listen_path": args.audio_listen_path,
            "output_path": output_path,
            "word_file": args.word_file,
            "segment_start_sec": args.segment_start_sec,
            "motion_style_pose_path": args.motion_style_pose_path,
            "motion_style_flame_path": args.motion_style_flame_path,
            "motion_style_start": args.motion_style_start,
            "motion_style_length": args.motion_style_length,
            "speaker_id": resolve_speaker_id(
                speaker_audio_path=args.audio_path,
                add_speaker_id=args.add_speaker_id and args.speaker_id_type == "text",
                speaker_id=args.speaker_id,
            ),
        }
        return

    if not args.audio_json:
        raise ValueError("Please provide either --audio-path or --audio-json.")

    with open(args.audio_json, "r", encoding="utf-8") as f:
        rows = json_load_compat(f)

    output_dir = args.output_dir or os.path.dirname(args.audio_json) or "."
    os.makedirs(output_dir, exist_ok=True)
    for row in rows:
        if "audio_file" not in row:
            raise KeyError("Each json item must contain `audio_file`.")
        audio_path = row["audio_file"]
        output_path = row.get(
            "output_path",
            os.path.join(output_dir, f"{Path(audio_path).stem}_gesture_expression.npz"),
        )
        yield {
            "audio_path": audio_path,
            "audio_listen_path": row.get("audio_file_listen", ""),
            "output_path": output_path,
            "word_file": row.get("word_file", ""),
            "segment_start_sec": float(row.get("fixed_start_sec", row.get("segment_start_sec", 0.0))),
            "motion_style_pose_path": row.get("motion_style_pose_path", row.get("motion_style_pose_file", args.motion_style_pose_path)),
            "motion_style_flame_path": row.get("motion_style_flame_path", row.get("motion_style_flame_file", args.motion_style_flame_path)),
            "motion_style_start": int(row.get("motion_style_start", args.motion_style_start)),
            "motion_style_length": int(row.get("motion_style_length", args.motion_style_length)),
            "speaker_id": resolve_speaker_id(
                speaker_audio_path=audio_path,
                add_speaker_id=args.add_speaker_id and args.speaker_id_type == "text",
                speaker_id=row.get("speaker_id", args.speaker_id),
            ),
        }


def parse_args(argv=None):
    """Define and parse model, audio, streaming, guidance and rendering options."""
    parser = argparse.ArgumentParser(
        description="Run ELLSA2 task_type=gesture_expression inference from audio and save SMPL-H + FLAME npz."
    )
    parser.add_argument("--audio-path", type=str, default="", help="single speaker audio path")
    parser.add_argument("--audio-listen-path", type=str, default="", help="listener audio path when dialog=True")
    parser.add_argument("--word-file", type=str, default="", help="timed-word JSON used by the inline exemplar router")
    parser.add_argument("--segment-start-sec", type=float, default=0.0, help="absolute transcript time of the first audio sample")
    parser.add_argument("--audio-json", type=str, default="", help="json list with `audio_file` and optional `audio_file_listen`")
    parser.add_argument("--output-path", type=str, default="", help="single output npz path")
    parser.add_argument("--output-dir", type=str, default="./outputs/inference", help="output directory for batch inference")
    parser.add_argument("--render", type=str2bool, nargs="?", const=True, default=False,
                        help="render each generated NPZ as an SMPL-X body/hand/face MP4")
    parser.add_argument("--smplx-model", type=str,
                        default=str(CHECKPOINT_ROOT / "external_assets/smplx/SMPLX_NEUTRAL_2020.npz"),
                        help="SMPL-X neutral 2020 NPZ with 100 expression coefficients, for --render true")
    parser.add_argument("--render-concurrent", type=int, default=2, help="number of rendering workers")

    parser.add_argument("--model-path", "--model_path", type=str, default="", help="ELLSA2 checkpoint directory")
    parser.add_argument("--motion-bias", action="store_true", help="use the bundled pose-only guidance checkpoint")
    parser.add_argument("--pose-guide", default="", help="30 fps pose NPZ; also selects --motion-bias unless --model-path is explicit")
    parser.add_argument("--pose-guide-start", type=int, default=0, help="first source frame of the 192-frame guide")
    parser.add_argument("--pose-guide-time", type=float, default=3.2, help="guide insertion time, a multiple of 0.8 seconds")
    parser.add_argument("--base-llm-path", type=str, default=DEFAULT_BASE_LLM_PATH, help="base LLM directory used by ELLSA2")
    parser.add_argument("--audio-encoder-path", type=str, default=DEFAULT_AUDIO_ENCODER_PATH, help="audio encoder checkpoint path")
    parser.add_argument(
        "--motion-tokenizer-type",
        type=str,
        default="upper_hands",
        choices=["upper_hands_trans", "upper_hands"],
        help="gesture tokenizer type used by the trained checkpoint",
    )
    parser.add_argument("--vqvae-upper-hands-trans-path", type=str, default=DEFAULT_VQ_PATH, help="joint upper+hands+translation RVQVAE checkpoint path")
    parser.add_argument("--vqvae-upper-path", type=str, default=DEFAULT_UPPER_VQ_PATH, help="upper-body RVQVAE checkpoint path for upper_hands mode")
    parser.add_argument("--vqvae-hands-path", type=str, default=DEFAULT_HANDS_VQ_PATH, help="hands RVQVAE checkpoint path for upper_hands mode")
    parser.add_argument("--vqvae-flame-path", type=str, default=DEFAULT_FLAME_VQ_PATH, help="flame RVQVAE checkpoint path")
    parser.add_argument("--rvq-decode-mode", choices=("auto", "stream", "whole"), default="auto",
                        help="auto streams causal RVQs block by block; whole waits for all tokens")
    parser.add_argument("--rvq-encoder-dtype", choices=("float32", "float64"), default=None)
    parser.add_argument("--rvq-quantizer-distance-dtype", choices=("float32", "float64"), default=None)
    parser.add_argument("--rvq-gesture-context-frames", type=int, default=None)
    parser.add_argument("--rvq-expression-context-frames", type=int, default=None)
    parser.add_argument("--stats-dir", type=str, default=DEFAULT_STATS_DIR, help="directory containing SMPL-H motion statistics")
    parser.add_argument("--stats-prefix", type=str, default=DEFAULT_STATS_PREFIX, help="SMPL-H statistics filename prefix")
    parser.add_argument("--stats-suffix", type=str, default=DEFAULT_STATS_SUFFIX, help="SMPL-H statistics filename suffix")
    parser.add_argument("--flame-stats-path", type=str, default=DEFAULT_FLAME_STATS_PATH, help="flame motion statistics json path")

    parser.add_argument("--dialog", type=str2bool, default=False, help="whether to use speaker-listener interleaved audio")
    parser.add_argument("--add-speaker-id", type=str2bool, default=True, help="whether to append speaker id to the system prompt")
    parser.add_argument("--speaker-id-type", choices=("text", "motion"), default="text")
    parser.add_argument("--speaker-id", type=str, default="", help="speaker id used in the prompt when --add-speaker-id=true; defaults to parsing from audio filename")
    parser.add_argument("--motion-style-pose-path", type=str, default="")
    parser.add_argument("--motion-style-flame-path", type=str, default="")
    parser.add_argument("--motion-style-expression", type=str2bool, default=None,
                        help="include FLAME style tokens; false explicitly reproduces pose-only conditioning")
    parser.add_argument("--motion-style-start", type=int, default=0)
    parser.add_argument("--motion-style-length", type=int, default=612)
    parser.add_argument("--window-sec", type=float, default=0.8, help="streaming block length in seconds")
    parser.add_argument("--audio-sr", type=int, default=16000, help="audio sample rate")
    parser.add_argument("--pose-fps", type=int, default=30, help="saved SMPL-H frame rate")
    parser.add_argument("--flame-fps", type=int, default=30, help="saved FLAME motion frame rate")
    parser.add_argument("--motion-downsample-factor", type=int, default=4, help="RVQVAE temporal downsample factor")
    parser.add_argument("--num-windows", type=int, default=None, help="force the number of 0.8s windows")
    parser.add_argument("--max-windows", type=int, default=None, help="truncate generation to at most this many windows")
    parser.add_argument("--save-motion-tokens", type=str2bool, default=False, help="whether to save raw token ids as npy sidecars")
    parser.add_argument("--save-split-outputs", type=str2bool, default=False, help="also save separate *_smplh.npz and *_flame.npz files")

    parser.add_argument("--device", type=str, default="auto", help="device, e.g. auto, cuda, cuda:0, cpu")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"], help="model dtype")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2", help="attention backend for the LLM")

    parser.add_argument("--motion-slot-token", type=str, default="<|vision_pad|>", help="motion placeholder token")
    parser.add_argument("--motion-vocab-size", type=int, default=2048, help="per-codebook motion vocabulary size")
    parser.add_argument("--num-motion-codebooks", type=int, default=None, help="total gesture+expression codebooks; defaults to gesture codebooks + expression codebooks")
    parser.add_argument("--gesture-num-codebooks", type=int, default=None, help="gesture codebooks at the beginning of each combined token")
    parser.add_argument("--expression-num-codebooks", type=int, default=6, help="expression codebooks after gesture codebooks")
    parser.add_argument("--motion-level-weights", type=float, nargs="*", default=None, help="optional per-codebook motion embed/loss weights")
    parser.add_argument("--motion-steps-per-window", type=int, default=6, help="combined/gesture motion steps in one 0.8s window")
    parser.add_argument("--interleave-motion-parts", type=str2bool, default=False, help="generate upper/hands/expression in alternating autoregressive steps")
    parser.add_argument("--expression-steps-per-window", type=int, default=None, help="expression steps in one 0.8s window; default derived from flame_fps/downsample")
    parser.add_argument("--audio-tokens-per-window", type=int, default=10, help="audio embedding tokens in one 0.8s window")
    parser.add_argument("--connector-seg-size", type=int, default=4, help="audio connector segment size")
    parser.add_argument("--connector-hid-size", type=int, default=4096, help="audio connector hidden size")
    parser.add_argument("--max-history-sec", type=float, default=12.0, help="maximum streaming history length in seconds")
    parser.add_argument("--reset-history-chunks", type=int, default=3, help="completed audio/motion windows retained at refresh, excluding the current window")
    parser.add_argument("--temperature", type=float, default=0.0, help="temperature for sampling")
    parser.add_argument("--top-k", type=int, default=0, help="top-k for sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p for sampling")
    parser.add_argument("--debug-mode", type=str2bool, default=False, help="pass debug flag to model args")

    parser.add_argument("--inline-exemplar-enabled", type=str2bool, default=False)
    parser.add_argument("--inline-exemplar-copy-mode", choices=("learned", "hard_core"), default="learned")
    parser.add_argument("--inline-exemplar-library", type=str, default=DEFAULT_INLINE_EXEMPLAR_LIBRARY)
    parser.add_argument("--inline-exemplar-schedule-file", type=str, default="")
    parser.add_argument("--inline-exemplar-lead-sec", type=float, default=5.0)
    parser.add_argument("--inline-exemplar-timing-mode", choices=("fixed_lead", "action_anchor"), default="action_anchor")
    parser.add_argument("--inline-exemplar-reference-frames", type=int, default=192)
    parser.add_argument("--inline-exemplar-top-k", type=int, default=4)
    parser.add_argument("--inline-exemplar-selection-seed", type=int, default=42)
    parser.add_argument("--inline-exemplar-force-family", type=str, default="")
    parser.add_argument("--inline-exemplar-force-chunk", type=int, default=4)
    parser.add_argument("--inline-exemplar-event-uid", type=str, default="")

    return parser.parse_args(argv)


def finalize_args(args):
    """Resolve release-relative assets and validate the requested inference configuration."""
    if args.render and not Path(args.smplx_model).is_file():
        raise FileNotFoundError(f"SMPL-X asset not found: {args.smplx_model}")
    if args.pose_guide:
        if args.inline_exemplar_library or args.inline_exemplar_schedule_file or args.inline_exemplar_force_family:
            raise ValueError("Use either --pose-guide or the legacy exemplar router, not both.")
        if args.inline_exemplar_copy_mode != "learned" or args.inline_exemplar_reference_frames != 192:
            raise ValueError("Pose-only guidance requires learned generation and a 192-frame reference.")
        if (args.window_sec, args.pose_fps, args.motion_downsample_factor, args.motion_steps_per_window) != (0.8, 30, 4, 6):
            raise ValueError("Pose-only guidance requires the trained 30 fps / 0.8 s / six-token layout.")
        args.motion_bias = args.inline_exemplar_enabled = True
    if not args.model_path:
        args.model_path = str(POSE_BIAS_MODEL) if args.motion_bias else DEFAULT_MODEL_PATH
    config_path = Path(args.model_path) / "config.json"
    if config_path.is_file():
        checkpoint_config = json.loads(config_path.read_text())
        tokenizer_subdir = checkpoint_config.get("rvq_tokenizer_subdir")
        if tokenizer_subdir:
            for part, default in (("upper", DEFAULT_UPPER_VQ_PATH),
                                  ("hands", DEFAULT_HANDS_VQ_PATH),
                                  ("flame", DEFAULT_FLAME_VQ_PATH)):
                key = f"vqvae_{part}_path"
                if getattr(args, key) == default:
                    setattr(args, key, str(Path(args.model_path).parent / tokenizer_subdir / f"{part}.pth"))
    if args.gesture_num_codebooks is None:
        args.gesture_num_codebooks = 12 if args.motion_tokenizer_type == "upper_hands" else 6
    if args.expression_num_codebooks is None:
        args.expression_num_codebooks = 6
    if args.num_motion_codebooks is None:
        args.num_motion_codebooks = int(args.gesture_num_codebooks) + int(args.expression_num_codebooks)
    if args.num_motion_codebooks != args.gesture_num_codebooks + args.expression_num_codebooks:
        raise ValueError(
            "`num_motion_codebooks` must equal `gesture_num_codebooks + expression_num_codebooks`, "
            f"got {args.num_motion_codebooks} vs {args.gesture_num_codebooks}+{args.expression_num_codebooks}."
        )
    if args.expression_num_codebooks <= 0 or args.expression_num_codebooks > 6:
        raise ValueError("`expression_num_codebooks` must be in [1, 6].")
    if args.expression_steps_per_window is None:
        args.expression_steps_per_window = max(
            1,
            int(round(float(args.window_sec) * float(args.flame_fps) / float(args.motion_downsample_factor))),
        )
    if args.expression_steps_per_window > args.motion_steps_per_window:
        raise ValueError("`expression_steps_per_window` cannot exceed `motion_steps_per_window`.")
    if args.motion_tokenizer_type == "upper_hands" and args.gesture_num_codebooks < 12:
        raise ValueError("`upper_hands` decoding requires `gesture_num_codebooks >= 12`.")
    if args.interleave_motion_parts:
        if args.motion_tokenizer_type != "upper_hands":
            raise ValueError("`interleave_motion_parts` only supports motion_tokenizer_type=upper_hands.")
        gesture_steps = max(
            1,
            int(round(float(args.window_sec) * float(args.pose_fps) / float(args.motion_downsample_factor))),
        )
        args.motion_steps_per_window = gesture_steps * 2 + args.expression_steps_per_window
    if args.inline_exemplar_enabled:
        if args.interleave_motion_parts:
            raise ValueError("Inline gesture_expression exemplars require interleave_motion_parts=false.")
        if args.motion_tokenizer_type != "upper_hands":
            raise ValueError("Inline gesture_expression exemplars require motion_tokenizer_type=upper_hands.")
        if args.inline_exemplar_reference_frames % args.motion_downsample_factor != 0:
            raise ValueError("inline_exemplar_reference_frames must be divisible by motion_downsample_factor.")
        reference_steps = args.inline_exemplar_reference_frames // args.motion_downsample_factor
        if reference_steps % args.motion_steps_per_window != 0:
            raise ValueError("Encoded inline exemplar steps must divide into complete streaming windows.")
        if not args.pose_guide and not os.path.exists(args.inline_exemplar_library):
            raise FileNotFoundError(f"Inline exemplar library not found: {args.inline_exemplar_library}")
    if not args.vqvae_flame_path:
        raise ValueError("Please provide `--vqvae-flame-path`.")
    if args.speaker_id_type == "motion":
        if not os.path.exists(args.motion_style_pose_path) and not args.audio_json:
            raise FileNotFoundError(f"Motion-style pose not found: {args.motion_style_pose_path}")
        if args.motion_style_flame_path and not os.path.exists(args.motion_style_flame_path) and not args.audio_json:
            raise FileNotFoundError(f"Motion-style FLAME not found: {args.motion_style_flame_path}")
    for key in ('rvq_encoder_dtype', 'rvq_quantizer_distance_dtype'):
        setattr(args, key, resolve_encoding_precision(args, key))
    if args.motion_style_expression is None:
        config_path = Path(args.model_path) / 'config.json'
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        args.motion_style_expression = config.get('motion_style_expression', not args.motion_bias)
    return args


def render_prediction(job, args):
    """Render the saved gesture and expression together, without demo dependencies."""
    motion_path = Path(job["output_path"]).resolve()
    video_path = motion_path.with_suffix(".mp4")
    subprocess.run([
        sys.executable, str(REPO_ROOT / "evaluation/render.py"),
        "--npz-path", str(motion_path), "--wav-path", str(Path(job["audio_path"]).resolve()),
        "--output-path", str(video_path), "--smplx-model", str(Path(args.smplx_model).resolve()),
        "--device", args.device, "--fps", str(args.pose_fps),
        "--concurrent", str(args.render_concurrent),
        "--zero-lower", "--zero-global-orient", "--overwrite",
    ], check=True, cwd=REPO_ROOT)
    print(f"Rendered gesture and expression video: {video_path}")


def main():
    """Load shared models once and run each requested audio inference job."""
    args = finalize_args(parse_args())
    device = choose_device(args.device)
    args.device = str(device)
    torch_dtype = choose_dtype(args.dtype, device)

    tokenizer, ellsa2_model = load_ellsa2_model(args, device, torch_dtype)
    motion_decoders = load_motion_decoders(args, device)
    gesture_stats = load_gesture_stats(args.stats_dir, args.stats_prefix, args.stats_suffix)
    flame_decoder = load_flame_decoder(
        args.vqvae_flame_path, device, resolve_rvq_context_frames(args, "expression"),
        args.rvq_encoder_dtype, args.rvq_quantizer_distance_dtype)
    flame_mean, flame_std = load_flame_stats(args.flame_stats_path)
    fbank_extractor = Fbank(FbankConfig(num_mel_bins=128))

    resources = dict(
        tokenizer=tokenizer, ellsa2_model=ellsa2_model, motion_decoders=motion_decoders,
        gesture_stats=gesture_stats, flame_decoder=flame_decoder,
        flame_mean=flame_mean, flame_std=flame_std, fbank_extractor=fbank_extractor, args=args,
    )
    for job in iter_jobs(args):
        inputs = dict(job)
        inputs['speaker_audio_path'] = inputs.pop('audio_path')
        inputs['listener_audio_path'] = inputs.pop('audio_listen_path')
        infer_single_audio(**resources, **inputs)
        if args.render:
            render_prediction(job, args)


if __name__ == "__main__":
    main()

