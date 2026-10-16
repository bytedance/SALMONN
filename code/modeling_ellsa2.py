"""Autoregressive audio-conditioned gesture/expression model with independent RVQ heads."""
from dataclasses import dataclass
import contextlib
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import rnn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput


def _task_has_asr(task_type: str) -> bool:
    """Return whether the task supervises text tokens."""
    return str(task_type).strip().lower() in {
        "asr",
        "asr_gesture",
        "asr_expression",
        "asr_gesture_expression",
    }


def _task_has_motion(task_type: str) -> bool:
    """Return whether the task supervises gesture or expression codes."""
    return str(task_type).strip().lower() in {
        "gesture",
        "expression",
        "gesture_expression",
        "asr_gesture",
        "asr_expression",
        "asr_gesture_expression",
    }


def _restore_qwen2_rope_fp32(module, inputs):
    # The training environment recomputed default RoPE frequencies in FP32.
    # Recompute after dtype conversion to preserve that checkpoint behavior.
    """Restore default Qwen2 rotary frequencies in FP32 before each forward pass."""
    module.inv_freq = module.compute_default_rope_parameters(module.config, inputs[0].device)[0]


@dataclass
class CoSpeechGestureOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None
    motion_loss: Optional[torch.FloatTensor] = None
    motion_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    motion_pred: Optional[torch.LongTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor, ...], ...]] = None
    llm_outputs: Optional[CausalLMOutputWithPast] = None


class ListenerLowRankDelta(nn.Module):
    def __init__(self, input_size, hidden_size, rank):
        """Initialize a low-rank listener residual with zero initial output."""
        super().__init__()
        self.down = nn.Linear(input_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=True)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, inputs):
        """Project listener features through the low-rank residual adapter."""
        return self.up(self.down(inputs))


class ListenerCausalConvGate(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers, init_bias):
        """Build a causal, dilated listener gate with the requested initial bias."""
        super().__init__()
        if kernel_size <= 0 or num_layers <= 0:
            raise ValueError("Temporal listener gate kernel size and layer count must be positive.")
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_size,
                    hidden_size,
                    kernel_size,
                    dilation=2 ** layer_idx,
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.output = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, init_bias)

    def forward(self, inputs):
        """Compute gate logits using only the current and preceding audio frames."""
        hidden = F.gelu(self.input_proj(inputs)).transpose(1, 2)
        for layer_idx, conv in enumerate(self.convs):
            left_padding = (conv.kernel_size[0] - 1) * (2 ** layer_idx)
            residual = hidden
            hidden = conv(F.pad(hidden, (left_padding, 0)))
            hidden = F.gelu(hidden) + residual
        return self.output(hidden.transpose(1, 2)).squeeze(-1)


class CoSpeechGestureLM(PreTrainedModel):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _no_split_modules = []
    all_tied_weights_keys = {}

    def __init__(self, config, model_args):
        """Build the audio connector, Qwen backbone and independent RVQ motion heads."""
        super().__init__(config)
        self.config = config

        self.llm_type = getattr(model_args, "llm_type", "Qwen")
        self.encoder_type = getattr(model_args, "encoder_type", "transformer_v2")
        self.connector_type = getattr(model_args, "connector_type", "MLP")
        if self.llm_type != "Qwen":
            raise ValueError(f"Only Qwen is supported, got '{self.llm_type}'.")
        if self.connector_type != "MLP":
            raise ValueError(f"Only MLP connector is supported, got '{self.connector_type}'.")

        self.freeze_encoder = bool(getattr(model_args, "freeze_encoder", False))
        self.freeze_llm = bool(getattr(model_args, "freeze_llm", False))
        self.dialog = bool(getattr(model_args, "dialog", True))
        self.listener_fusion_delta = bool(getattr(model_args, "listener_fusion_delta", False))
        self.listener_fusion_scale = float(getattr(model_args, "listener_fusion_scale", 0.1))
        self.listener_fusion_gate = bool(getattr(model_args, "listener_fusion_gate", False))
        self.listener_gate_activation = str(getattr(model_args, "listener_gate_activation", "sigmoid"))
        self.listener_gate_power = float(getattr(model_args, "listener_gate_power", 1.0))
        self.listener_gate_loss_weight = float(getattr(model_args, "listener_gate_loss_weight", 0.2))
        self.listener_gate_positive_weight = float(getattr(model_args, "listener_gate_positive_weight", 10.0))
        self.listener_gate_init_bias = float(getattr(model_args, "listener_gate_init_bias", -3.0))
        self.listener_gate_type = str(getattr(model_args, "listener_gate_type", "linear"))
        self.listener_gate_hidden_size = int(getattr(model_args, "listener_gate_hidden_size", 64))
        self.listener_gate_kernel_size = int(getattr(model_args, "listener_gate_kernel_size", 3))
        self.listener_gate_num_layers = int(getattr(model_args, "listener_gate_num_layers", 3))
        self.listener_gate_deadzone_loss_weight = float(
            getattr(model_args, "listener_gate_deadzone_loss_weight", 0.0)
        )
        self.listener_gate_deadzone_max_logit = float(
            getattr(model_args, "listener_gate_deadzone_max_logit", -3.0)
        )
        if self.dialog and self.listener_fusion_delta:
            raise ValueError("listener_fusion_delta requires dialog=False to preserve the standalone token sequence.")
        if self.listener_fusion_gate and not self.listener_fusion_delta:
            raise ValueError("listener_fusion_gate requires listener_fusion_delta.")
        self.task_type = str(getattr(model_args, "task_type", "gesture")).lower()
        if self.task_type not in {
            "gesture",
            "expression",
            "gesture_expression",
            "asr",
            "asr_gesture",
            "asr_expression",
            "asr_gesture_expression",
        }:
            raise ValueError(f"Unsupported `task_type`: {self.task_type}")
        self.has_asr_target = _task_has_asr(self.task_type)
        self.has_motion_target = _task_has_motion(self.task_type)
        self.motion_vocab_size = int(getattr(model_args, "motion_vocab_size", 2048))
        self.num_motion_codebooks = int(getattr(model_args, "num_motion_codebooks", 6))
        if self.has_motion_target and self.num_motion_codebooks <= 0:
            raise ValueError("`num_motion_codebooks` must be positive.")
        if (not self.has_motion_target) and self.num_motion_codebooks < 0:
            raise ValueError("`num_motion_codebooks` cannot be negative.")
        self.motion_steps_per_window = int(getattr(model_args, "motion_steps_per_window", 6))
        self.gesture_num_codebooks = int(getattr(model_args, "gesture_num_codebooks", self.num_motion_codebooks))
        self.expression_num_codebooks = int(getattr(model_args, "expression_num_codebooks", 0))
        self.expression_steps_per_window = int(
            getattr(model_args, "expression_steps_per_window", self.motion_steps_per_window)
        )
        self.interleave_motion_parts = bool(getattr(model_args, "interleave_motion_parts", False))
        if self.interleave_motion_parts:
            if self.expression_num_codebooks > 0:
                self.interleave_split = 3
            else:
                self.interleave_split = 2
        else:
            self.interleave_split = 1
        if self.has_motion_target:
            if self.interleave_motion_parts and self.gesture_num_codebooks < 12:
                raise ValueError("`interleave_motion_parts` requires at least 12 gesture codebooks for upper/hands.")
            if self.gesture_num_codebooks < 0 or self.expression_num_codebooks < 0:
                raise ValueError("Gesture/expression codebook counts cannot be negative.")
            if self.gesture_num_codebooks + self.expression_num_codebooks not in {0, self.num_motion_codebooks}:
                raise ValueError(
                    "`gesture_num_codebooks + expression_num_codebooks` must equal `num_motion_codebooks` "
                    "when both are configured."
                )
            if self.expression_steps_per_window <= 0 or self.expression_steps_per_window > self.motion_steps_per_window:
                raise ValueError(
                    "`expression_steps_per_window` must be in [1, motion_steps_per_window]."
                )
        self.audio_tokens_per_window = int(getattr(model_args, "audio_tokens_per_window", 10))
        self.connector_seg_size = int(getattr(model_args, "connector_seg_size", 4))
        self.llm_config = self._get_llm_config(config)
        self.hidden_size = self.llm_config.hidden_size
        self.connector_hid_size = int(getattr(model_args, "connector_hid_size", self.hidden_size))
        self.debug_mode = bool(getattr(model_args, "debug_mode", False))
        motion_level_weights = self._parse_motion_level_weights(
            getattr(model_args, "motion_level_weights", None),
            "motion_level_weights",
        )
        motion_embed_weights = self._parse_motion_level_weights(
            getattr(model_args, "motion_embed_weights", None),
            "motion_embed_weights",
            default=motion_level_weights,
        )
        motion_loss_weights = self._parse_motion_level_weights(
            getattr(model_args, "motion_loss_weights", None),
            "motion_loss_weights",
            default=motion_level_weights,
        )
        self.register_buffer("motion_embed_weights", motion_embed_weights, persistent=False)
        self.register_buffer("motion_loss_weights", motion_loss_weights, persistent=False)

        self.base_llm = AutoModelForCausalLM.from_config(self.llm_config)
        self._no_split_modules = list(self.base_llm._no_split_modules)
        self._apply_qwen2_rope_precision()
        self._apply_freeze_llm()

        self.audio_encoder = self._build_audio_encoder()
        audio_encoder_path = getattr(model_args, "audio_encoder_path", None)
        if audio_encoder_path:
            ckpt = torch.load(audio_encoder_path, map_location="cpu")
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self.audio_encoder.load_state_dict(state_dict, strict=False)

        if self.freeze_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            self.audio_encoder.eval()

        encoder_dim = self.audio_encoder.encoder_dim
        self.ln_audio = nn.LayerNorm(encoder_dim)
        self.connector = nn.Sequential(
            nn.Linear(encoder_dim * self.connector_seg_size, self.connector_hid_size),
            nn.ReLU(),
            nn.Linear(self.connector_hid_size, self.hidden_size),
        )
        self.listener_connector_delta = None
        if bool(getattr(model_args, "listener_connector_delta", False)):
            listener_delta_rank = int(getattr(model_args, "listener_delta_rank", 0))
            if listener_delta_rank > 0:
                self.listener_connector_delta = ListenerLowRankDelta(
                    encoder_dim * self.connector_seg_size,
                    self.hidden_size,
                    listener_delta_rank,
                )
            else:
                self.listener_connector_delta = nn.Linear(
                    encoder_dim * self.connector_seg_size,
                    self.hidden_size,
                )
                nn.init.zeros_(self.listener_connector_delta.weight)
                nn.init.zeros_(self.listener_connector_delta.bias)
        self.listener_gate = None
        if self.listener_fusion_gate:
            if self.listener_gate_type == "linear":
                self.listener_gate = nn.Linear(self.hidden_size * 2, 1)
                nn.init.zeros_(self.listener_gate.weight)
                nn.init.constant_(self.listener_gate.bias, self.listener_gate_init_bias)
            elif self.listener_gate_type == "causal_conv":
                self.listener_gate = ListenerCausalConvGate(
                    self.hidden_size * 2,
                    self.listener_gate_hidden_size,
                    self.listener_gate_kernel_size,
                    self.listener_gate_num_layers,
                    self.listener_gate_init_bias,
                )
            else:
                raise ValueError(f"Unsupported listener gate type: {self.listener_gate_type}")
        self._last_listener_gate_logits = None
        self._last_listener_gate_lens = None
        self.motion_embeds = nn.ModuleList(
            [nn.Embedding(self.motion_vocab_size, self.hidden_size) for _ in range(self.num_motion_codebooks)]
        )
        self.motion_heads = nn.ModuleList(
            [nn.Linear(self.hidden_size, self.motion_vocab_size) for _ in range(self.num_motion_codebooks)]
        )

    @staticmethod
    def _get_llm_config(config):
        """Select the text configuration from either a text or multimodal config."""
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            return text_config
        return config

    @staticmethod
    def _get_initializer_range(config):
        """Read the backbone initialization scale, defaulting to 0.02."""
        initializer_range = getattr(config, "initializer_range", None)
        if initializer_range is not None:
            return float(initializer_range)
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            initializer_range = getattr(text_config, "initializer_range", None)
            if initializer_range is not None:
                return float(initializer_range)
        return 0.02

    def _apply_qwen2_rope_precision(self):
        """Preserve the training checkpoint's FP32 default Qwen2 RoPE behavior."""
        if self.llm_config.model_type == "qwen2":
            rotary = self.base_llm.model.rotary_emb
            if rotary.rope_type == "default":
                rotary.register_forward_pre_hook(_restore_qwen2_rope_fp32)

    def _apply_freeze_llm(self):
        """Disable backbone gradients when freeze_llm is configured."""
        if self.freeze_llm:
            for param in self.base_llm.parameters():
                param.requires_grad = False
            self.base_llm.eval()

    def load_base_llm_from_pretrained(self, llm_path):
        """Load the pretrained text backbone and reject missing or unexpected weights."""
        torch_dtype = getattr(self.llm_config, "torch_dtype", getattr(self.config, "torch_dtype", None))
        self.base_llm, loading_info = AutoModelForCausalLM.from_pretrained(
            llm_path,
            config=self.llm_config,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            output_loading_info=True,
        )
        bad_info = {
            name: list(loading_info.get(name, []))
            for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading_info.get(name)
        }
        if bad_info:
            summary = {name: values[:20] for name, values in bad_info.items()}
            raise RuntimeError(f"Failed to load base LLM cleanly from {llm_path}: {summary}")
        self._apply_qwen2_rope_precision()
        self._apply_freeze_llm()

    def init_missing_parameters(self, missing_keys):
        """Initialize newly added parameters omitted by a pretrained checkpoint."""
        missing_keys = set(missing_keys or [])
        param_dict = dict(self.named_parameters())
        initialized_keys = []
        initializer_range = self._get_initializer_range(self.llm_config)

        for key in sorted(missing_keys):
            param = param_dict.get(key)
            if param is None:
                continue
            with torch.no_grad():
                if key in {"listener_gate.bias", "listener_gate.output.bias"}:
                    torch.nn.init.constant_(param, self.listener_gate_init_bias)
                elif key == "listener_connector_delta.down.weight":
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
                elif key == "listener_gate.output.weight":
                    torch.nn.init.zeros_(param)
                elif key.startswith("listener_gate.") and key.endswith(".weight"):
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
                elif key.startswith("listener_connector_delta.") or key.startswith("listener_gate."):
                    torch.nn.init.zeros_(param)
                elif param.ndim > 1:
                    torch.nn.init.normal_(param, mean=0.0, std=initializer_range)
                else:
                    torch.nn.init.zeros_(param)
            initialized_keys.append(key)

        bad_keys = []
        for key in initialized_keys:
            param = param_dict[key]
            if not torch.isfinite(param).all():
                bad_keys.append(key)
        if bad_keys:
            raise ValueError(f"Non-finite values found after initializing missing parameters: {bad_keys}")
        if initialized_keys:
            print(f"Initialized missing parameters with normal/zero init: {initialized_keys}")

    def _parse_motion_level_weights(self, weights, name, default=None):
        """Validate one nonnegative weight per codebook, with at least one active head."""
        if self.num_motion_codebooks == 0:
            return torch.zeros(0, dtype=torch.float32, device="cpu")

        if weights is None:
            if default is not None:
                return default.clone().detach().to(dtype=torch.float32)
            return torch.ones(self.num_motion_codebooks, dtype=torch.float32, device="cpu")

        if isinstance(weights, str):
            weights = [item.strip() for item in weights.split(",") if item.strip()]

        weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device="cpu")
        if weight_tensor.ndim != 1 or weight_tensor.numel() != self.num_motion_codebooks:
            raise ValueError(
                f"`{name}` must provide exactly {self.num_motion_codebooks} weights, got shape {tuple(weight_tensor.shape)}."
            )
        if torch.any(weight_tensor < 0):
            raise ValueError(f"`{name}` must be non-negative.")
        if torch.all(weight_tensor == 0):
            raise ValueError(f"`{name}` cannot be all zeros.")
        return weight_tensor

    def _build_audio_encoder(self):
        """Construct the supported causal transformer_v2 audio encoder."""
        if self.encoder_type != "transformer_v2":
            raise ValueError("This release supports encoder_type=transformer_v2 only")
        from streaming_transformer_encoder_v2.audio_encoder import AttributeDict, CausalAudioEncoder

        speech_encoder_params = AttributeDict()
        speech_encoder_params.causal = True
        speech_encoder_params.encoder_dim= 1280
        speech_encoder_params.num_layers = 22
        speech_encoder_params.use_flash_attention = 1
        speech_encoder_params.attention_dropout = 0.1
        speech_encoder_params.num_heads = 16
        speech_encoder_params.subsampling_factor = 2
        speech_encoder_params.feature_dim = 128
        speech_encoder_params.dtype = torch.bfloat16

        speech_encoder = CausalAudioEncoder(
            config=speech_encoder_params,
            num_mels=speech_encoder_params.feature_dim, # fbank dim
        )

        speech_encoder.encoder_dim = 1280
        return speech_encoder

    def maybe_autocast(self, dtype=torch.bfloat16):
        """Use CUDA autocast when available; keep CPU execution in its native dtype."""
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        if device.type == "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=dtype)
        return contextlib.nullcontext()

    def encode_audio(
        self,
        fbank_feature,
        fbank_feature_len,
        listener=False,
        listener_delta_only=False,
        listener_fusion_features=False,
    ):
        """Encode filterbanks and group encoder frames into backbone-sized embeddings."""
        with self.maybe_autocast(next(self.audio_encoder.parameters()).dtype):
            audio_embeds, encoder_out_lens = self.audio_encoder.forward_encoder(
                fbank_feature[:, : int(fbank_feature_len.max().item()), :],
                fbank_feature_len,
            )
            audio_embeds = self.ln_audio(audio_embeds)

            batch_size, seq_len, dim = audio_embeds.size()
            if seq_len % self.connector_seg_size != 0:
                pad_len = (seq_len // self.connector_seg_size + 1) * self.connector_seg_size - seq_len
                pad_embeds = torch.zeros((batch_size, pad_len, dim), dtype=audio_embeds.dtype, device=audio_embeds.device)
                audio_embeds = torch.cat((audio_embeds, pad_embeds), dim=1)
                seq_len = audio_embeds.size(1)

            audio_embeds = audio_embeds.view(
                batch_size,
                seq_len // self.connector_seg_size,
                dim * self.connector_seg_size,
            )
            connector_input = audio_embeds
            audio_lens = torch.ceil(encoder_out_lens / self.connector_seg_size).to(torch.int64)
            if listener_fusion_features:
                if self.listener_connector_delta is None:
                    raise ValueError("listener_fusion_features requires listener_connector_delta")
                return (
                    self.connector(connector_input),
                    self.listener_connector_delta(connector_input),
                    audio_lens,
                )
            if listener_delta_only:
                if self.listener_connector_delta is None:
                    raise ValueError("listener_delta_only requires listener_connector_delta")
                return self.listener_connector_delta(connector_input), audio_lens
            audio_embeds = self.connector(connector_input)
            if listener and self.listener_connector_delta is not None:
                audio_embeds = audio_embeds + self.listener_connector_delta(connector_input)
            return audio_embeds, audio_lens

    def _fuse_listener_audio(
        self,
        audio_embeds,
        audio_embeds_lens,
        audio_listen_fbank,
        audio_listen_fbank_len,
    ):
        """Add an aligned listener residual, optionally scaled by a learned gate."""
        listener_embeds, audio_delta, audio_delta_lens = self.encode_audio(
            audio_listen_fbank,
            audio_listen_fbank_len,
            listener_fusion_features=True,
        )
        if audio_delta.shape != audio_embeds.shape or not torch.equal(audio_delta_lens, audio_embeds_lens):
            raise ValueError("Speaker and listener audio must be frame aligned for residual fusion.")
        if self.listener_fusion_gate:
            gate_logits = self.listener_gate(torch.cat((audio_embeds, listener_embeds), dim=-1))
            if gate_logits.ndim == 3:
                gate_logits = gate_logits.squeeze(-1)
            if self.listener_gate_activation == "sigmoid":
                gate = torch.sigmoid(gate_logits)
            elif self.listener_gate_activation == "hard_sigmoid":
                gate = F.hardsigmoid(gate_logits)
            else:
                raise ValueError(f"Unsupported listener gate activation: {self.listener_gate_activation}")
            gate = gate.pow(self.listener_gate_power).unsqueeze(-1)
            self._last_listener_gate_logits = gate_logits
            self._last_listener_gate_lens = audio_embeds_lens
            audio_delta = gate * audio_delta
        return audio_embeds + self.listener_fusion_scale * audio_delta

    def _encode_audio_streams(
        self, fbank_feature, fbank_feature_len, audio_listen_fbank, audio_listen_fbank_len, fuse_listener=True,
    ):
        """Dialog keeps two streams; listener fusion adds a delta to the target stream."""
        audio_embeds, audio_embeds_lens = self.encode_audio(fbank_feature, fbank_feature_len)
        audio_listen_embeds = None
        audio_listen_embeds_lens = None
        if fuse_listener and self.listener_fusion_delta:
            if audio_listen_fbank is None or audio_listen_fbank_len is None:
                raise ValueError("listener_fusion_delta requires listener audio.")
            audio_embeds = self._fuse_listener_audio(
                audio_embeds,
                audio_embeds_lens,
                audio_listen_fbank,
                audio_listen_fbank_len,
            )
        elif self.dialog:
            if audio_listen_fbank is None or audio_listen_fbank_len is None:
                raise ValueError("dialog=True requires `audio_listen_fbank` and `audio_listen_fbank_len`.")
            audio_listen_embeds, audio_listen_embeds_lens = self.encode_audio(
                audio_listen_fbank, audio_listen_fbank_len, listener=True
            )
        return audio_embeds, audio_embeds_lens, audio_listen_embeds, audio_listen_embeds_lens

    def _embed_tokens(self, input_ids):
        """Look up text embeddings using integer token IDs."""
        return self.base_llm.get_input_embeddings()(input_ids.to(torch.int64))

    def _motion_ids_to_embeds(self, motion_token_ids):
        """Sum weighted RVQ embeddings; negative IDs contribute zero."""
        if self.num_motion_codebooks <= 0:
            raise RuntimeError("Motion embeddings are unavailable because `num_motion_codebooks` is zero.")
        motion_embeds = []
        for level in range(self.num_motion_codebooks):
            level_ids = motion_token_ids[:, :, level].clamp(min=0, max=self.motion_vocab_size - 1)
            invalid_mask = motion_token_ids[:, :, level] < 0
            level_embeds = self.motion_embeds[level](level_ids)
            # Mask on-device: testing .any() here would synchronize every codebook.
            level_embeds = level_embeds.masked_fill(invalid_mask.unsqueeze(-1), 0.0)
            motion_embeds.append(level_embeds * self.motion_embed_weights[level].to(level_embeds.dtype))
        return torch.stack(motion_embeds, dim=0).sum(dim=0)

    def _apply_motion_layout_mask(self, motion_token_ids, fill_value=-100):
        """Mask inactive codebooks and expression padding for the configured layout."""
        if self.interleave_motion_parts:
            masked_ids = motion_token_ids.clone()
            total_steps = masked_ids.size(1)
            expression_steps = self.expression_steps_per_window if self.expression_num_codebooks > 0 else 0
            window_steps = self.motion_steps_per_window
            gesture_steps = (window_steps - expression_steps) // 2
            for start in range(0, total_steps, window_steps):
                cursor = start
                window_end = min(start + window_steps, total_steps)
                for frame_idx in range(gesture_steps):
                    if cursor < window_end:
                        masked_ids[:, cursor, :] = fill_value
                        masked_ids[:, cursor, :6] = motion_token_ids[:, cursor, :6]
                        cursor += 1
                    if cursor < window_end:
                        masked_ids[:, cursor, :] = fill_value
                        masked_ids[:, cursor, 6:12] = motion_token_ids[:, cursor, 6:12]
                        cursor += 1
                    if expression_steps > 0 and frame_idx < expression_steps and cursor < window_end:
                        masked_ids[:, cursor, : self.gesture_num_codebooks] = fill_value
                        cursor += 1
                if cursor < window_end:
                    masked_ids[:, cursor:window_end, :] = fill_value
            return masked_ids

        if self.expression_num_codebooks <= 0:
            return motion_token_ids
        if self.expression_steps_per_window >= self.motion_steps_per_window:
            return motion_token_ids
        if self.gesture_num_codebooks >= self.num_motion_codebooks:
            return motion_token_ids

        masked_ids = motion_token_ids.clone()
        total_steps = masked_ids.size(1)
        for start in range(0, total_steps, self.motion_steps_per_window):
            pad_start = start + self.expression_steps_per_window
            pad_end = min(start + self.motion_steps_per_window, total_steps)
            if pad_start < pad_end:
                masked_ids[:, pad_start:pad_end, self.gesture_num_codebooks:] = fill_value
        return masked_ids

    def _take_stream_chunk(self, audio_embeds, audio_lens, row_idx, cursor):
        """Take the next audio window, allowing a short final chunk."""
        total_len = int(min(audio_lens[row_idx].item(), audio_embeds.size(1)))
        start = int(cursor)
        end = min(start + self.audio_tokens_per_window, total_len)
        if start >= total_len:
            raise ValueError(
                f"Not enough audio embeddings for streaming insertion: row={row_idx}, cursor={cursor}, total={total_len}."
            )
        return audio_embeds[row_idx, start:end, :], end


    def _expand_streaming_inputs(
        self, input_ids, attention_mask, audio_start_mask, audio_end_mask, audio_streams,
        motion_slot_mask=None, motion_input_embeds=None, labels=None,
    ):
        """Expand placeholders into audio/motion embeddings and pad the resulting batch.

        Inputs have shape (batch, text_steps); motion embeddings have shape
        (batch, motion_steps, hidden_size). Audio inserts carry label -100.
        The motion target mask marks the position
        immediately before each motion input, matching next-token prediction.
        """
        speaker, speaker_lens, listener, listener_lens = audio_streams
        streams = [
            (speaker, speaker_lens.cpu()),
            (listener, None if listener_lens is None else listener_lens.cpu()),
        ]
        # Make layout decisions on CPU; embedding tensors retain their device and gradients.
        attention_mask, audio_start_mask, audio_end_mask = (
            mask.cpu() for mask in (attention_mask, audio_start_mask, audio_end_mask)
        )
        if motion_slot_mask is not None:
            motion_slot_mask = motion_slot_mask.cpu()
        embed_rows, attention_rows, target_rows, label_rows = [], [], [], []
        for row in range(input_ids.size(0)):
            token_embeds = self._embed_tokens(input_ids[row])
            seq_len = int(attention_mask[row].sum().item())
            parts, motion_positions, label_parts = [], [], []
            cursors = [0, 0]
            col = audio_slot = motion_cursor = 0
            while col < seq_len:
                if motion_slot_mask is not None and motion_slot_mask[row, col]:
                    motion = (
                        token_embeds.new_zeros((1, self.hidden_size)) if motion_input_embeds is None
                        else motion_input_embeds[row, motion_cursor:motion_cursor + 1]
                    )
                    parts.append(motion)
                    motion_positions.append(True)
                    if labels is not None:
                        label_parts.append(labels[row, col:col + 1])
                    motion_cursor += 1
                    col += 1
                elif col + 1 < seq_len and audio_start_mask[row, col] and audio_end_mask[row, col + 1]:
                    stream = int(self.dialog and audio_slot % 2 == 1)
                    embeds, lengths = streams[stream]
                    chunk, cursors[stream] = self._take_stream_chunk(embeds, lengths, row, cursors[stream])
                    parts.extend((token_embeds[col:col + 1], chunk, token_embeds[col + 1:col + 2]))
                    motion_positions.extend([False] * (chunk.size(0) + 2))
                    if labels is not None:
                        label_parts.extend((
                            labels[row, col:col + 1],
                            torch.full((chunk.size(0),), -100, dtype=torch.long, device=input_ids.device),
                            labels[row, col + 1:col + 2],
                        ))
                    audio_slot += 1
                    col += 2
                else:
                    parts.append(token_embeds[col:col + 1])
                    motion_positions.append(False)
                    if labels is not None:
                        label_parts.append(labels[row, col:col + 1])
                    col += 1
            row_embeds = torch.cat(parts, dim=0)
            motion_positions = torch.tensor(motion_positions, dtype=torch.bool, device=input_ids.device)
            motion_targets = torch.zeros_like(motion_positions)
            motion_targets[:-1] = motion_positions[1:]
            embed_rows.append(row_embeds)
            attention_rows.append(torch.ones(len(motion_positions), dtype=torch.long, device=input_ids.device))
            target_rows.append(motion_targets)
            if labels is not None:
                label_rows.append(torch.cat(label_parts))
        return (
            rnn.pad_sequence(embed_rows, batch_first=True),
            rnn.pad_sequence(attention_rows, batch_first=True, padding_value=0),
            rnn.pad_sequence(target_rows, batch_first=True, padding_value=False),
            rnn.pad_sequence(label_rows, batch_first=True, padding_value=-100) if labels is not None else None,
        )

    def prepare_llm_inputs_for_streaming_speech(
        self,
        input_ids,
        attention_mask,
        audio_start_mask,
        audio_end_mask,
        fbank_feature,
        fbank_feature_len,
        audio_listen_fbank=None,
        audio_listen_fbank_len=None,
        labels=None,
    ):
        """Expand audio placeholders and insert ignored labels for the ASR task."""
        audio_streams = self._encode_audio_streams(
            fbank_feature, fbank_feature_len, audio_listen_fbank, audio_listen_fbank_len,
        )
        embeds, attention, motion_targets, expanded_labels = self._expand_streaming_inputs(
            input_ids, attention_mask, audio_start_mask, audio_end_mask, audio_streams,
            labels=labels,
        )
        return embeds, attention, expanded_labels

    def prepare_inputs_labels_for_speech(
        self,
        input_ids,
        attention_mask,
        audio_start_mask,
        audio_end_mask,
        motion_slot_mask,
        fbank_feature,
        fbank_feature_len,
        audio_listen_fbank=None,
        audio_listen_fbank_len=None,
        motion_input_embeds=None,
    ):
        """Expand audio and motion placeholders and mark motion prediction positions."""
        audio_streams = self._encode_audio_streams(
            fbank_feature, fbank_feature_len, audio_listen_fbank, audio_listen_fbank_len,
        )
        embeds, attention, motion_targets, expanded_labels = self._expand_streaming_inputs(
            input_ids, attention_mask, audio_start_mask, audio_end_mask, audio_streams,
            motion_slot_mask=motion_slot_mask, motion_input_embeds=motion_input_embeds,
        )
        return embeds, attention, motion_targets

    def prepare_inputs_labels_for_speech_and_motion(
        self,
        input_ids,
        attention_mask,
        audio_start_mask,
        audio_end_mask,
        motion_slot_mask,
        fbank_feature,
        fbank_feature_len,
        audio_listen_fbank=None,
        audio_listen_fbank_len=None,
        motion_input_embeds=None,
        labels=None,
    ):
        """Expand joint ASR/motion inputs, preserving the separate-stream audio path."""
        audio_streams = self._encode_audio_streams(
            fbank_feature, fbank_feature_len, audio_listen_fbank, audio_listen_fbank_len, fuse_listener=False,
        )
        embeds, attention, motion_targets, expanded_labels = self._expand_streaming_inputs(
            input_ids, attention_mask, audio_start_mask, audio_end_mask, audio_streams,
            motion_slot_mask=motion_slot_mask, motion_input_embeds=motion_input_embeds, labels=labels,
        )
        return embeds, attention, motion_targets, expanded_labels

    def _forward_asr(
        self,
        input_ids,
        attention_mask,
        audio_start_mask,
        audio_end_mask,
        fbank_feature,
        fbank_feature_len,
        audio_listen_fbank=None,
        audio_listen_fbank_len=None,
        labels=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=True,
    ):
        """Run the text model with expanded audio inputs and optional ASR labels."""
        input_embeds, expanded_attention_mask, expanded_labels = self.prepare_llm_inputs_for_streaming_speech(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_start_mask=audio_start_mask,
            audio_end_mask=audio_end_mask,
            fbank_feature=fbank_feature,
            fbank_feature_len=fbank_feature_len,
            audio_listen_fbank=audio_listen_fbank,
            audio_listen_fbank_len=audio_listen_fbank_len,
            labels=labels,
        )

        with self.maybe_autocast():
            llm_outputs = self.base_llm(
                inputs_embeds=input_embeds,
                attention_mask=expanded_attention_mask,
                labels=expanded_labels,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        return llm_outputs

    def _gather_motion_hidden(self, hidden_states, motion_position_mask):
        """Gather prediction states in slot order and pad shorter rows with zeros."""
        gathered_hidden = []
        for row_hidden, row_mask in zip(hidden_states, motion_position_mask):
            gathered_hidden.append(row_hidden[row_mask])

        padded_hidden = rnn.pad_sequence(gathered_hidden, batch_first=True)
        valid_mask = rnn.pad_sequence(
            [torch.ones(item.size(0), dtype=torch.bool, device=hidden_states.device) for item in gathered_hidden],
            batch_first=True,
            padding_value=False,
        )
        return padded_hidden, valid_mask


    def _project_motion_hidden(self, hidden_states, motion_target_mask):
        """Gather predictive hidden states and apply one head per RVQ codebook."""
        motion_hidden, valid = self._gather_motion_hidden(hidden_states, motion_target_mask)
        with self.maybe_autocast():
            logits = tuple(head(motion_hidden) for head in self.motion_heads)
        prediction = None if self.training else torch.stack([head.argmax(dim=-1) for head in logits], dim=-1)
        return logits, prediction, valid

    def _run_backbone(
        self,
        input_ids,
        attention_mask,
        audio_start_mask,
        audio_end_mask,
        motion_slot_mask,
        fbank_feature,
        fbank_feature_len,
        audio_listen_fbank=None,
        audio_listen_fbank_len=None,
        motion_input_embeds=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=True,
    ):
        """Run the text backbone and project motion prediction states to RVQ logits."""
        input_embeds, expanded_attention_mask, motion_target_mask = self.prepare_inputs_labels_for_speech(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_start_mask=audio_start_mask,
            audio_end_mask=audio_end_mask,
            motion_slot_mask=motion_slot_mask,
            fbank_feature=fbank_feature,
            fbank_feature_len=fbank_feature_len,
            audio_listen_fbank=audio_listen_fbank,
            audio_listen_fbank_len=audio_listen_fbank_len,
            motion_input_embeds=motion_input_embeds,
        )

        # Motion-only training does not consume text-vocabulary logits. Calling
        # AutoModelForCausalLM here would project every sequence position to the
        # full 151k text vocabulary and immediately discard the result.
        with self.maybe_autocast():
            decoder_outputs = self.base_llm.model(
                inputs_embeds=input_embeds,
                attention_mask=expanded_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        llm_outputs = CausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

        motion_logits, motion_pred, valid_motion_mask = self._project_motion_hidden(
            decoder_outputs.last_hidden_state, motion_target_mask
        )
        return llm_outputs, motion_logits, motion_pred, valid_motion_mask

    def _forward_asr_gesture(
        self,
        input_ids,
        attention_mask,
        audio_start_mask,
        audio_end_mask,
        motion_slot_mask,
        fbank_feature,
        fbank_feature_len,
        audio_listen_fbank=None,
        audio_listen_fbank_len=None,
        motion_input_embeds=None,
        labels=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=True,
    ):
        """Run joint supervision with separate text loss and RVQ motion predictions."""
        input_embeds, expanded_attention_mask, motion_target_mask, expanded_labels = self.prepare_inputs_labels_for_speech_and_motion(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_start_mask=audio_start_mask,
            audio_end_mask=audio_end_mask,
            motion_slot_mask=motion_slot_mask,
            fbank_feature=fbank_feature,
            fbank_feature_len=fbank_feature_len,
            audio_listen_fbank=audio_listen_fbank,
            audio_listen_fbank_len=audio_listen_fbank_len,
            motion_input_embeds=motion_input_embeds,
            labels=labels,
        )

        with self.maybe_autocast():
            decoder_outputs = self.base_llm.model(
                inputs_embeds=input_embeds,
                attention_mask=expanded_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = decoder_outputs.hidden_states[-1]
        text_loss = self._compute_text_loss_from_hidden(hidden_states, expanded_labels)
        llm_outputs = CausalLMOutputWithPast(
            loss=text_loss,
            logits=None,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )
        motion_logits, motion_pred, valid_motion_mask = self._project_motion_hidden(hidden_states, motion_target_mask)
        return llm_outputs, motion_logits, motion_pred, valid_motion_mask, text_loss

    def _autoregressive_motion_decode(
        self,
        input_ids,
        attention_mask,
        audio_start_mask,
        audio_end_mask,
        motion_slot_mask,
        fbank_feature,
        fbank_feature_len,
        audio_listen_fbank=None,
        audio_listen_fbank_len=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=True,
    ):
        """Generate each motion step from context and previously predicted RVQ codes."""
        batch_size = input_ids.size(0)
        motion_counts = motion_slot_mask.long().sum(dim=1)
        max_motion_steps = int(motion_counts.max().item())
        motion_input_ids = torch.zeros(
            (batch_size, max_motion_steps, self.num_motion_codebooks),
            dtype=torch.long,
            device=input_ids.device,
        )

        logits_storage = [[] for _ in range(self.num_motion_codebooks)]
        final_outputs = None
        final_valid_mask = None

        for step in range(max_motion_steps):
            motion_input_embeds = self._motion_ids_to_embeds(motion_input_ids)
            llm_outputs, motion_logits, motion_pred, valid_motion_mask = self._run_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_start_mask=audio_start_mask,
                audio_end_mask=audio_end_mask,
                motion_slot_mask=motion_slot_mask,
                fbank_feature=fbank_feature,
                fbank_feature_len=fbank_feature_len,
                audio_listen_fbank=audio_listen_fbank,
                audio_listen_fbank_len=audio_listen_fbank_len,
                motion_input_embeds=motion_input_embeds,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            final_outputs = llm_outputs
            final_valid_mask = valid_motion_mask
            step_valid = valid_motion_mask[:, step]
            for level in range(self.num_motion_codebooks):
                step_logits = motion_logits[level][:, step, :]
                logits_storage[level].append(step_logits)

            active_rows = step_valid & (step < motion_counts)
            motion_input_ids[active_rows, step] = motion_pred[active_rows, step]
            motion_input_ids = self._apply_motion_layout_mask(motion_input_ids, fill_value=0)

        stacked_logits = tuple(torch.stack(level_logits, dim=1) for level_logits in logits_storage)
        stacked_pred = torch.stack([head_logits.argmax(dim=-1) for head_logits in stacked_logits], dim=-1)
        stacked_pred = self._apply_motion_layout_mask(stacked_pred, fill_value=0)
        return final_outputs, stacked_logits, stacked_pred, final_valid_mask

    def _compute_text_loss_from_hidden(self, hidden_states, labels):
        """Compute shifted text cross-entropy only at nonignored label positions."""
        shift_hidden = hidden_states[..., :-1, :]
        shift_labels = labels[..., 1:].to(hidden_states.device)
        valid_mask = shift_labels.ne(-100)
        if not bool(valid_mask.any()):
            return hidden_states.sum() * 0.0
        with self.maybe_autocast():
            selected_logits = self.base_llm.lm_head(shift_hidden[valid_mask])
        return F.cross_entropy(selected_logits.float(), shift_labels[valid_mask])

    def _compute_motion_loss(self, motion_logits, motion_labels, motion_loss_mask=None):
        """Average weighted per-codebook cross-entropy over valid motion labels."""
        head_losses = []
        valid_head_indices = []
        if motion_loss_mask is not None:
            motion_loss_mask = motion_loss_mask.to(device=motion_labels.device, dtype=torch.float32)
        for head_idx, head_logits in enumerate(motion_logits):
            head_labels = motion_labels[:, :, head_idx]
            valid_mask = head_labels >= 0
            if not torch.any(valid_mask):
                continue
            if motion_loss_mask is None:
                head_loss = F.cross_entropy(
                    head_logits.reshape(-1, head_logits.size(-1)),
                    head_labels.reshape(-1),
                    ignore_index=-100,
                )
            else:
                token_loss = F.cross_entropy(
                    head_logits.reshape(-1, head_logits.size(-1)),
                    head_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(head_labels)
                token_weights = motion_loss_mask.masked_fill(~valid_mask, 0.0).to(dtype=token_loss.dtype)
                head_loss = (token_loss * token_weights).sum() / token_weights.sum().clamp_min(1e-8)
            head_losses.append(head_loss)
            valid_head_indices.append(head_idx)
        if len(head_losses) == 0:
            return motion_logits[0].sum() * 0.0
        head_losses = torch.stack(head_losses)
        loss_weights = self.motion_loss_weights[valid_head_indices].to(
            device=head_losses.device,
            dtype=head_losses.dtype,
        )
        return (head_losses * loss_weights).sum() / loss_weights.sum()

    def _build_listener_gate_targets(self, motion_backchannel_mask):
        """Expand per-window backchannel targets to the audio embedding timeline."""
        logits = self._last_listener_gate_logits
        targets = torch.zeros_like(logits)
        valid = torch.zeros_like(logits, dtype=torch.bool)
        motion_backchannel_mask = motion_backchannel_mask.to(logits.device)
        for row_idx in range(logits.size(0)):
            audio_steps = int(min(self._last_listener_gate_lens[row_idx].item(), logits.size(1)))
            num_windows = (audio_steps + self.audio_tokens_per_window - 1) // self.audio_tokens_per_window
            motion_steps = num_windows * self.motion_steps_per_window
            row_mask = motion_backchannel_mask[row_idx]
            if row_mask.numel() < motion_steps:
                raise ValueError("Backchannel mask is shorter than the aligned motion sequence.")
            row_mask = row_mask[-motion_steps:].reshape(num_windows, self.motion_steps_per_window)
            window_targets = row_mask.amax(dim=1).to(dtype=logits.dtype)
            token_targets = window_targets.repeat_interleave(self.audio_tokens_per_window)[:audio_steps]
            targets[row_idx, :audio_steps] = token_targets
            valid[row_idx, :audio_steps] = True
        return targets, valid

    def _compute_listener_gate_loss(self, motion_backchannel_mask):
        """Compute valid-frame gate supervision and the optional negative deadzone loss."""
        if (
            not self.listener_fusion_gate
            or self._last_listener_gate_logits is None
            or motion_backchannel_mask is None
        ):
            return None
        logits = self._last_listener_gate_logits
        targets, valid = self._build_listener_gate_targets(motion_backchannel_mask)
        token_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
            pos_weight=torch.as_tensor(
                self.listener_gate_positive_weight,
                dtype=logits.dtype,
                device=logits.device,
            ),
        )
        gate_loss = token_loss.masked_select(valid).mean()
        if self.listener_gate_deadzone_loss_weight > 0.0:
            negative_valid = valid & (targets == 0)
            if negative_valid.any():
                deadzone_loss = F.relu(
                    logits.masked_select(negative_valid) - self.listener_gate_deadzone_max_logit
                ).square().mean()
                gate_loss = gate_loss + self.listener_gate_deadzone_loss_weight * deadzone_loss
        return gate_loss


    def _masked_motion_loss(self, motion_logits, motion_labels, motion_loss_mask, valid_motion_mask):
        """Validate sequence lengths and exclude padded/inactive motion targets from loss."""
        pred_steps = motion_logits[0].size(1)
        if motion_labels.size(1) != pred_steps:
            raise ValueError(
                f"Motion label length ({motion_labels.size(1)}) does not match motion slot count ({pred_steps})."
            )
        motion_labels = self._apply_motion_layout_mask(motion_labels, fill_value=-100)
        motion_labels = motion_labels.masked_fill((~valid_motion_mask).unsqueeze(-1), -100)
        if motion_loss_mask is not None:
            if motion_loss_mask.size(1) != pred_steps:
                raise ValueError(
                    f"Motion loss mask length ({motion_loss_mask.size(1)}) does not match motion slot count ({pred_steps})."
                )
            motion_loss_mask = motion_loss_mask.masked_fill(~valid_motion_mask, 0.0)
        return self._compute_motion_loss(motion_logits, motion_labels, motion_loss_mask)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        audio_start_mask=None,
        audio_end_mask=None,
        motion_slot_mask=None,
        motion_labels=None,
        motion_labels_per_head=None,
        motion_context_labels=None,
        motion_loss_mask=None,
        motion_semantic_mask=None,
        motion_backchannel_mask=None,
        motion_backchannel_timing_weight=None,
        motion_backchannel_appropriateness_weight=None,
        labels=None,
        fbank_feature=None,
        fbank_feature_len=None,
        audio_listen_fbank=None,
        audio_listen_fbank_len=None,
        position_ids=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, CoSpeechGestureOutput]:
        """Run ASR, motion, or joint teacher-forced training; decode motion when labels are absent.

        Context labels may contain style/guide steps ignored by the target labels.
        Returns the task's legacy tuple or CoSpeechGestureOutput when return_dict=True.
        """
        if input_ids is None or attention_mask is None:
            raise ValueError("`input_ids` and `attention_mask` are required.")
        if audio_start_mask is None or audio_end_mask is None:
            raise ValueError("Audio placeholder masks are required.")
        if fbank_feature is None or fbank_feature_len is None:
            raise ValueError("`fbank_feature` and `fbank_feature_len` are required.")
        self._last_listener_gate_logits = self._last_listener_gate_lens = None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = True if output_hidden_states is None else output_hidden_states
        return_dict = True if return_dict is None else return_dict
        inputs = dict(
            input_ids=input_ids, attention_mask=attention_mask,
            audio_start_mask=audio_start_mask, audio_end_mask=audio_end_mask,
            fbank_feature=fbank_feature, fbank_feature_len=fbank_feature_len,
            audio_listen_fbank=audio_listen_fbank, audio_listen_fbank_len=audio_listen_fbank_len,
            position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        )
        text_loss = motion_loss = motion_logits = motion_pred = loss = None
        if self.task_type != "asr":
            if motion_slot_mask is None:
                message = "ASR+motion tasks" if self.has_asr_target else "motion training/inference"
                raise ValueError(f"`motion_slot_mask` is required for {message}.")
            if self.has_asr_target and labels is None:
                raise ValueError("`labels` are required for ASR+motion tasks.")
            if motion_labels is None and motion_labels_per_head is not None:
                motion_labels = torch.stack(motion_labels_per_head, dim=-1)
            if self.has_asr_target and motion_labels is None:
                raise ValueError("`motion_labels` are required for ASR+motion tasks.")
            inputs["motion_slot_mask"] = motion_slot_mask
        if self.debug_mode:
            import pdb
            pdb.set_trace()

        if self.task_type == "asr":
            llm_outputs = self._forward_asr(**inputs, labels=labels)
            loss = text_loss = llm_outputs.loss
        else:
            if motion_labels is not None:
                motion_labels = motion_labels.to(input_ids.device)
                if motion_loss_mask is not None:
                    motion_loss_mask = motion_loss_mask.to(input_ids.device)
                # Style and guide codes condition the model even when their labels are ignored.
                context = (motion_labels if motion_context_labels is None else motion_context_labels).to(input_ids.device)
                slot_counts = motion_slot_mask.long().sum(dim=1)
                context_counts = (context[:, :, 0] >= 0).long().sum(dim=1)
                if not torch.equal(slot_counts / self.interleave_split, context_counts):
                    raise ValueError(
                        f"Motion slot count {slot_counts.tolist()} does not match motion context count {context_counts.tolist()}."
                    )
                context = self._apply_motion_layout_mask(context, fill_value=-100)
                inputs["motion_input_embeds"] = self._motion_ids_to_embeds(context)
            if self.has_asr_target:
                llm_outputs, motion_logits, motion_pred, valid_mask, text_loss = self._forward_asr_gesture(
                    **inputs, labels=labels.to(input_ids.device)
                )
            elif motion_labels is not None:
                llm_outputs, motion_logits, motion_pred, valid_mask = self._run_backbone(**inputs)
            else:
                llm_outputs, motion_logits, motion_pred, valid_mask = self._autoregressive_motion_decode(**inputs)
            if motion_labels is not None:
                motion_loss = self._masked_motion_loss(motion_logits, motion_labels, motion_loss_mask, valid_mask)
                if self.has_asr_target:
                    loss = text_loss + motion_loss
                else:
                    gate_loss = self._compute_listener_gate_loss(motion_backchannel_mask)
                    loss = motion_loss if gate_loss is None else motion_loss + self.listener_gate_loss_weight * gate_loss
        if not return_dict:
            if self.task_type == "asr":
                return loss, llm_outputs
            if self.has_asr_target:
                return loss, text_loss, motion_loss, motion_logits, motion_pred, llm_outputs
            return loss, motion_logits, motion_pred, llm_outputs
        return CoSpeechGestureOutput(
            loss=loss, text_loss=text_loss, motion_loss=motion_loss,
            motion_logits=motion_logits, motion_pred=motion_pred,
            hidden_states=llm_outputs.hidden_states, attentions=llm_outputs.attentions,
            past_key_values=llm_outputs.past_key_values, llm_outputs=llm_outputs,
        )

    @torch.no_grad()
    def generate_motion(
        self,
        input_ids,
        attention_mask,
        audio_start_mask,
        audio_end_mask,
        motion_slot_mask,
        fbank_feature,
        fbank_feature_len,
        audio_listen_fbank=None,
        audio_listen_fbank_len=None,
    ):
        """Generate RVQ IDs and offset each codebook into a disjoint token range."""
        if (not self.has_motion_target) or self.has_asr_target:
            raise ValueError("`generate_motion` is only available for motion-only tasks.")
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_start_mask=audio_start_mask,
            audio_end_mask=audio_end_mask,
            motion_slot_mask=motion_slot_mask,
            fbank_feature=fbank_feature,
            fbank_feature_len=fbank_feature_len,
            audio_listen_fbank=audio_listen_fbank,
            audio_listen_fbank_len=audio_listen_fbank_len,
            motion_labels=None,
            return_dict=True,
        )
        motion_pred = outputs.motion_pred.clone()
        for level in range(self.num_motion_codebooks):
            motion_pred[:, :, level] += level * self.motion_vocab_size
        return motion_pred
