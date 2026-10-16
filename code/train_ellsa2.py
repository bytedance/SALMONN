"""Configure, initialize and train the standalone audio-to-motion model with Trainer."""
import os
import json
import pathlib
from dataclasses import asdict, dataclass, field
from typing import Optional, List

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, Trainer

from dataset import (
    CoSpeechGestureCollator, CoSpeechGestureDataset,
    _resolve_motion_target_flags as _dataset_motion_target_flags,
)
from modeling_ellsa2 import CoSpeechGestureLM
from scripts.gesture_io import (
    maybe_load_legacy_qwen35_state_dict,
    reset_motion_weight_buffers,
)

DEFAULT_FLAME_STATS_PATH = str(pathlib.Path(__file__).resolve().parent / "assets/flame_stats.json")


def _patch_deepspeed_universal_step_device():
    """Keep the universal checkpoint step tensor on its parameter's device."""
    import deepspeed.checkpoint.universal_checkpoint as universal_checkpoint

    original_load = universal_checkpoint.load_hp_checkpoint_state

    def load_hp_checkpoint_state(parameter, folder, tp_rank, tp_world_size):
        """Load a parameter checkpoint and move its step tensor to the parameter device."""
        step = original_load(parameter, folder, tp_rank, tp_world_size)
        return step.to(parameter.device) if torch.is_tensor(step) else step

    universal_checkpoint.load_hp_checkpoint_state = load_hp_checkpoint_state


def _resolve_motion_target_flags(data_args):
    """Use the dataset's shared task and legacy FLAME target resolution."""
    return _dataset_motion_target_flags(data_args.task_type, data_args.pose_rep)


def _resolve_motion_layout(data_args):
    """Derive codebook counts and streaming steps from task, frame rates and RVQ layout."""
    legacy_expression_mode, has_gesture_target, has_expression_target = _resolve_motion_target_flags(data_args)

    gesture_output_codebooks = 0
    if has_gesture_target:
        used_num_motion_codebooks = int(data_args.used_num_motion_codebooks)
        if str(data_args.motion_tokenizer_type) == "upper_hands":
            gesture_output_codebooks = min(6, used_num_motion_codebooks) * 2
        else:
            gesture_output_codebooks = used_num_motion_codebooks

    expression_output_codebooks = 0
    if has_expression_target:
        used_num_expression_codebooks = int(getattr(data_args, "used_num_expression_codebooks", 6))
        if legacy_expression_mode and used_num_expression_codebooks == 6:
            used_num_expression_codebooks = int(data_args.used_num_motion_codebooks)
        expression_output_codebooks = used_num_expression_codebooks

    gesture_steps_per_window = max(
        1,
        int(round(float(data_args.window_sec) * float(data_args.pose_fps) / float(data_args.motion_downsample_factor))),
    )
    expression_steps_per_window = max(
        1,
        int(round(float(data_args.window_sec) * float(data_args.flame_fps) / float(data_args.motion_downsample_factor))),
    )
    motion_steps_per_window = gesture_steps_per_window if has_gesture_target else expression_steps_per_window
    interleave_motion_parts = bool(getattr(data_args, "interleave_motion_parts", False))
    if interleave_motion_parts:
        if has_gesture_target and str(data_args.motion_tokenizer_type) == "upper_hands":
            motion_steps_per_window = gesture_steps_per_window * 2
            if has_expression_target:
                motion_steps_per_window += expression_steps_per_window
        else:
            raise ValueError("`interleave_motion_parts` only supports motion_tokenizer_type=upper_hands gesture targets.")

    return {
        "gesture_num_codebooks": gesture_output_codebooks,
        "expression_num_codebooks": expression_output_codebooks,
        "num_motion_codebooks": gesture_output_codebooks + expression_output_codebooks,
        "motion_steps_per_window": motion_steps_per_window,
        "gesture_steps_per_window": gesture_steps_per_window,
        "expression_steps_per_window": expression_steps_per_window,
        "interleave_motion_parts": interleave_motion_parts,
    }


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = ""
    base_llm_path: Optional[str] = ""
    llm_type: str = "Qwen"
    attn_implementation: str = "flash_attention_2"
    encoder_type: str = "transformer_v2"
    audio_encoder_path: str = ""
    freeze_encoder: bool = True
    freeze_llm: bool = False
    freeze_motion_io: bool = False
    listener_connector_delta: bool = False
    listener_delta_rank: int = 0
    listener_fusion_delta: bool = False
    listener_fusion_scale: float = 0.1
    listener_fusion_gate: bool = False
    listener_gate_activation: str = "sigmoid"
    listener_gate_power: float = 1.0
    listener_gate_loss_weight: float = 0.2
    listener_gate_positive_weight: float = 10.0
    listener_gate_init_bias: float = -3.0
    listener_gate_type: str = "linear"
    listener_gate_hidden_size: int = 64
    listener_gate_kernel_size: int = 3
    listener_gate_num_layers: int = 3
    listener_gate_deadzone_loss_weight: float = 0.0
    listener_gate_deadzone_max_logit: float = -3.0
    train_listener_delta_only: bool = False
    train_listener_gate_only: bool = False
    freeze_listener_gate: bool = False
    connector_type: str = "MLP"
    connector_seg_size: int = 4
    connector_hid_size: int = 4096
    motion_vocab_size: int = 2048
    num_motion_codebooks: int = 6
    motion_steps_per_window: int = 6
    gesture_num_codebooks: int = 6
    expression_num_codebooks: int = 0
    expression_steps_per_window: int = 6
    audio_tokens_per_window: int = 10
    motion_level_weights: List[float] = field(
        default_factory=lambda: [
            1.0, 0.5, 0.25, 0.1, 0.05, 0.025,
            1.0, 0.5, 0.25, 0.1, 0.05, 0.025,
            1.0, 0.5, 0.25, 0.1, 0.05, 0.025,
        ]
    )
    debug_mode: bool = False


@dataclass
class DataArguments:
    data_path: str = ""
    task_type: str = "gesture"
    pose_rep: str = "smplh"
    pose_length: int = 600
    pose_fps: int = 30
    flame_fps: int = 30
    motion_downsample_factor: int = 4
    audio_sr: int = 16000
    audio_fps: int = 16000
    dialog: bool = True
    dialog_text: bool = False
    inline_imitation_boundary_loss_weight: float = 1.0
    inline_imitation_core_loss_weight: float = 1.0
    alignment_score_threshold: Optional[float] = None
    motion_tokenizer_type: str = "upper_hands_trans"
    vqvae_upper_hands_trans_path: str = ""
    vqvae_upper_path: str = ""
    vqvae_hands_path: str = ""
    vqvae_torso_path: str = ""
    vqvae_left_arm_path: str = ""
    vqvae_right_arm_path: str = ""
    vqvae_left_hand_path: str = ""
    vqvae_right_hand_path: str = ""
    vqvae_flame_path: str = ""
    used_num_motion_codebooks: int = 6
    used_num_expression_codebooks: int = 6
    interleave_motion_parts: bool = False
    window_sec: float = 0.8
    motion_slot_token: str = "<|vision_pad|>"
    motion_only: bool = False
    asr_eot_token: str = "<|endoftext|>"
    add_speaker_id: bool = False
    speaker_id_type: str = "text"
    motion_style_length: Optional[int] = None
    inline_exemplar_enabled: bool = False
    inline_exemplar_copy_mode: str = "learned"
    precompute_sample_meta: bool = False
    enable_runtime_cache: bool = False
    motion_token_cache_dir: str = ""
    audio_cache_size: int = 1024
    pose_cache_size: int = 1024
    transcript_cache_size: int = 4096
    flame_output_suffix: str = "_flame.npz"
    flame_file_key: str = "flame_file"
    flame_motion_key: str = "motioncode"
    rvq_encoder_dtype: str = "float32"
    rvq_quantizer_distance_dtype: str = "float32"
    rvq_gesture_context_frames: int = 0
    rvq_expression_context_frames: int = 0
    flame_stats_path: str = DEFAULT_FLAME_STATS_PATH


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    report_to: Optional[str] = "wandb"
    run_name: Optional[str] = "ellsa2_cospeech_gesture"
    min_learning_rate: Optional[float] = None
    from_scratch: bool = False


def _configure_trainable_parameters(model, model_args):
    """Apply listener-only/gate-only training and optional motion-I/O freezing."""
    if model_args.train_listener_delta_only or model_args.train_listener_gate_only:
        if model.listener_connector_delta is None:
            raise ValueError("listener-only training requires listener_connector_delta")
        model.requires_grad_(False)
        if model_args.train_listener_delta_only:
            model.listener_connector_delta.requires_grad_(True)
        if model.listener_gate is not None and not model_args.freeze_listener_gate:
            model.listener_gate.requires_grad_(True)
        elif model_args.train_listener_gate_only:
            if model.listener_gate is None:
                raise ValueError("train_listener_gate_only requires listener_fusion_gate")
            raise ValueError("train_listener_gate_only conflicts with freeze_listener_gate")
    if model_args.freeze_listener_gate:
        if model.listener_gate is None:
            raise ValueError("freeze_listener_gate requires listener_fusion_gate")
        model.listener_gate.requires_grad_(False)
    if model_args.freeze_motion_io:
        for module in (model.motion_embeds, model.motion_heads):
            module.requires_grad_(False)


def remap_expression_state_dict(state_dict, gesture_codebooks, expression_codebooks):
    """Place expression-only I/O weights after the joint model's gesture codebooks."""
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith(("motion_embeds.", "motion_heads.")):
            prefix, index, suffix = key.split(".", 2)
            if not 0 <= int(index) < expression_codebooks:
                raise ValueError(f"Unexpected expression-only weight: {key}")
            key = f"{prefix}.{gesture_codebooks + int(index)}.{suffix}"
        remapped[key] = value
    return remapped


def load_model_and_dataset(model_args, data_args, training_args):
    """Initialize tokenizer, model and datasets with a consistent motion layout."""
    model_args.dialog = bool(data_args.dialog)
    data_args.listener_fusion_delta = bool(model_args.listener_fusion_delta)
    model_args.task_type = str(data_args.task_type).lower()
    if bool(data_args.dialog_text) and not bool(data_args.dialog):
        raise ValueError("`dialog_text=True` requires `dialog=True`.")
    if data_args.task_type not in {
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

    _, has_gesture_target, has_expression_target = _resolve_motion_target_flags(data_args)
    has_motion_target = has_gesture_target or has_expression_target
    motion_layout = _resolve_motion_layout(data_args)
    model_args.num_motion_codebooks = int(motion_layout["num_motion_codebooks"])
    model_args.motion_steps_per_window = int(motion_layout["motion_steps_per_window"])
    model_args.gesture_num_codebooks = int(motion_layout["gesture_num_codebooks"])
    model_args.expression_num_codebooks = int(motion_layout["expression_num_codebooks"])
    model_args.expression_steps_per_window = int(motion_layout["expression_steps_per_window"])
    model_args.interleave_motion_parts = bool(motion_layout["interleave_motion_parts"])
    if has_motion_target and model_args.num_motion_codebooks <= 0:
        raise ValueError("The total number of motion/expression codebooks must be positive.")
    if len(model_args.motion_level_weights) < model_args.num_motion_codebooks:
        raise ValueError(
            "`motion_level_weights` must provide at least the total number of motion/expression codebooks."
        )
    model_args.motion_level_weights = model_args.motion_level_weights[: model_args.num_motion_codebooks]
    tokenizer_path = model_args.base_llm_path if training_args.from_scratch else model_args.model_name_or_path
    if not tokenizer_path:
        tokenizer_path = model_args.base_llm_path
    if not tokenizer_path:
        raise ValueError("Please provide `base_llm_path` or `model_name_or_path`.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config_path = model_args.base_llm_path if training_args.from_scratch else model_args.model_name_or_path
    if not config_path:
        config_path = tokenizer_path
    model_config = AutoConfig.from_pretrained(config_path)
    source_task = getattr(model_config, "task_type", None)
    for key in ("task_type", "num_motion_codebooks", "gesture_num_codebooks", "expression_num_codebooks"):
        setattr(model_config, key, getattr(model_args, key))
    model_config.torch_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)
    model_config.attn_implementation = model_args.attn_implementation
    model_config.rvq_gesture_context_frames = data_args.rvq_gesture_context_frames
    model_config.rvq_expression_context_frames = data_args.rvq_expression_context_frames
    model_config.rvq_encoder_dtype = data_args.rvq_encoder_dtype
    model_config.rvq_quantizer_distance_dtype = data_args.rvq_quantizer_distance_dtype
    model_config.motion_style_expression = bool(data_args.task_type == 'gesture_expression'
                                               and data_args.add_speaker_id
                                               and data_args.speaker_id_type == 'motion')

    if training_args.from_scratch:
        model = CoSpeechGestureLM(model_config, model_args)
        if model_args.base_llm_path:
            model.load_base_llm_from_pretrained(model_args.base_llm_path)
    else:
        remapped_state_dict = maybe_load_legacy_qwen35_state_dict(model_args.model_name_or_path)
        if source_task == "expression" and has_gesture_target and has_expression_target:
            from safetensors.torch import load_file
            state_dict = remapped_state_dict or load_file(
                os.path.join(model_args.model_name_or_path, "model.safetensors"))
            state_dict = remap_expression_state_dict(
                state_dict, model_args.gesture_num_codebooks, model_args.expression_num_codebooks)
            model, loading_info = CoSpeechGestureLM.from_pretrained(
                None, config=model_config, model_args=model_args, state_dict=state_dict,
                torch_dtype="auto", output_loading_info=True)
            if loading_info.get("unexpected_keys") or loading_info.get("mismatched_keys") or loading_info.get("error_msgs"):
                raise RuntimeError(f"Expression-to-joint initialization failed: {loading_info}")
            expected_missing = {key for key in model.state_dict()
                                if key.startswith(("motion_embeds.", "motion_heads."))
                                and int(key.split(".")[1]) < model_args.gesture_num_codebooks}
            if set(loading_info.get("missing_keys", [])) != expected_missing:
                raise RuntimeError(f"Unexpected missing stage-transfer weights: {loading_info}")
            model.init_missing_parameters(loading_info["missing_keys"])
        elif remapped_state_dict is None:
            model, loading_info = CoSpeechGestureLM.from_pretrained(
                model_args.model_name_or_path,
                config=model_config,
                model_args=model_args,
                torch_dtype="auto",
                output_loading_info=True,
            )
            bad_info = {
                name: list(loading_info.get(name, []))
                for name in ("unexpected_keys", "mismatched_keys", "error_msgs")
                if loading_info.get(name)
            }
            if bad_info:
                summary = {name: values[:20] for name, values in bad_info.items()}
                raise RuntimeError(f"Failed to load ELLSA2 checkpoint cleanly: {summary}")
            model.init_missing_parameters(loading_info.get("missing_keys", []))
        else:
            print("Detected legacy Qwen3.5 ELLSA2 checkpoint key layout; loading with remapped state_dict.")
            model = CoSpeechGestureLM(model_config, model_args)
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "Remapped checkpoint did not match model state_dict: "
                    f"missing={list(missing_keys)[:20]}, unexpected={list(unexpected_keys)[:20]}"
                )

    reset_motion_weight_buffers(model, model_args)
    _configure_trainable_parameters(model, model_args)

    dataset = CoSpeechGestureDataset(
        data_args,
        tokenizer,
        window_sec=data_args.window_sec,
        audio_sr=data_args.audio_sr,
        motion_slot_token=data_args.motion_slot_token,
    )
    collator = CoSpeechGestureCollator(
        tokenizer,
        audio_sr=data_args.audio_sr,
        dialog=data_args.dialog,
        listener_fusion_delta=model_args.listener_fusion_delta,
        num_motion_codebooks=model_args.num_motion_codebooks,
        task_type=data_args.task_type,
    )
    return tokenizer, model, dataset, collator


def main():
    """Parse configuration, train with Trainer and save model/tokenizer artifacts."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if "universal" in str(training_args.deepspeed):
        _patch_deepspeed_universal_step_device()

    os.environ["WANDB_PROJECT"] = "ellsa2_cospeech_gesture"
    if training_args.min_learning_rate is not None:
        if training_args.lr_scheduler_kwargs is None:
            training_args.lr_scheduler_kwargs = {}
        training_args.lr_scheduler_kwargs["min_lr_rate"] = training_args.min_learning_rate / training_args.learning_rate

    training_args.remove_unused_columns = False

    # Seed before constructing new motion heads and sampling dataset metadata.
    transformers.set_seed(training_args.seed)
    tokenizer, model, dataset, collator = load_model_and_dataset(model_args, data_args, training_args)

    # Keep the resolved recipe separate from Hugging Face's model config.json.
    if training_args.should_save:
        os.makedirs(training_args.output_dir, exist_ok=True)
        config = {"model_args": asdict(model_args), "data_args": asdict(data_args),
                  "training_args": asdict(training_args)}
        with open(os.path.join(training_args.output_dir, "training_config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )

    resume = training_args.resume_from_checkpoint
    if resume is None and list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        resume = True
    trainer.train(resume_from_checkpoint=resume)
    # Always leave a usable model, even when max_steps is not a save interval.
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
