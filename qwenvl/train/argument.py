import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    model_base: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_mm_audio: bool = field(default=False)
    tune_mm_qformer: bool = field(default=False)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_ckpt: str = field(default="No")
    model_type: str = field(default="moe")
    fixed_memory_size: int = field(default=0)
    fixed_memory_size_audio: int = field(default=0)
    stepsize: int = field(default=0)
    ttt_type: str = field(default="tttsim")
    cg_max_iter: int = field(default=0)
    ttt_hidden_size: int = field(default=4)
    ttt_num_heads: int = field(default=8)
    memgroupsize: int = field(default=0)
    workingmemsize: int = field(default=0)
    search_type: str = field(default="none")
    retain_factor: str = field(default="diversity")
    lambdas: str = field(default="0.0")
    div_factor: float = field(default=0.0)
    slot_type: str = field(default="")
    ema_factor: float = field(default=0.1)
    lag_distances: str = field(default="0")

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    base_interval: float = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    run_test: bool = field(default=False)
    do_sample: bool = field(default=False)
    num_sample: int = field(default=1)
    train_type: str = field(default="sft")
    feature_size: int = field(default=128)
    chunk_length: int = field(default=30)
    hop_length: int = field(default=160)
    sampling_rate: int = field(default=16000)
    validation_data: str = field(default="")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    fsdp: str = field(default="")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    pred_rank: int = field(default=0)
    no_audio: bool = field(default=False)
    use_modality_sampler: bool = field(default=False)
    do_validation: bool = field(default=False)
    distill_factor: float = field(default=1.0)
    freeze_ttt: bool = field(default=False)
    freeze_lora: bool = field(default=False)
    train_memory: bool = field(default=False)
