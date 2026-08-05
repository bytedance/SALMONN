import contextlib

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_salmonn import SalmonnConfig
from .spear.model import MultiKDModel
from .spear.scaling import ScheduledFloat
from .spear.subsampling import Conv2dSubsampling
from .spear.zipformer_layerwise import Zipformer2 as SPEAREncoder


def _ints(value):
    return tuple(map(int, value.split(",")))


def build_spear():
    dims = _ints("1280,1280,1280,1280,1280,1280,1280")
    encoder_embed = Conv2dSubsampling(
        in_channels=128,
        out_channels=dims[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    encoder = SPEAREncoder(
        output_downsampling_factor=1,
        downsampling_factor=_ints("1,2,4,8,4,2,1"),
        num_encoder_layers=_ints("1,2,3,4,1,1,1"),
        encoder_dim=dims,
        encoder_unmasked_dim=_ints("768,768,768,768,768,768,768"),
        query_head_dim=_ints("32"),
        pos_head_dim=_ints("4"),
        value_head_dim=_ints("12"),
        pos_dim=48,
        num_heads=_ints("8,8,8,8,8,8,8"),
        feedforward_dim=_ints("3840,3840,3840,3840,3840,3840,3840"),
        cnn_module_kernel=_ints("31,31,15,15,15,31,31"),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=False,
        chunk_size=_ints("-1"),
        left_context_frames=_ints("-1"),
    )
    return MultiKDModel(encoder_embed, encoder, max(dims), num_codebooks=0)


class SalmonnForConditionalGeneration(PreTrainedModel):
    """SALMONN-2: a SPEAR audio encoder, an MLP connector, and Qwen3."""

    config_class = SalmonnConfig
    base_model_prefix = "salmonn"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer", "Qwen3MoeDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        from transformers import AutoConfig

        qwen_config = AutoConfig.for_model(**config.qwen_config)
        self.base_llm = AutoModelForCausalLM.from_config(qwen_config)
        self.audio_encoder = build_spear()

        encoder_dim = self.audio_encoder.encoder_dim
        self.ln_audio = nn.LayerNorm(encoder_dim)
        self.concat_proj = None
        if config.concatenate_encoder_layers:
            self.concat_proj = nn.Linear(self.audio_encoder.num_encoder_layers * encoder_dim, encoder_dim)
        self.connector = nn.Sequential(
            nn.Linear(encoder_dim * config.connector_segment_size, config.connector_hidden_size),
            nn.ReLU(),
            nn.Linear(config.connector_hidden_size, qwen_config.hidden_size),
        )
        self.audio_encoder.requires_grad_(not config.freeze_audio_encoder)
        self.nl_timestamp_token_ids_list = None
        if config.inject_temporal_embedding_nl:
            self.output_frame_rate = config.encoder_frame_rate / config.connector_segment_size
            frames_per_stamp = config.temporal_granularity * self.output_frame_rate
            if abs(frames_per_stamp - round(frames_per_stamp)) >= 1e-6:
                raise ValueError("temporal_granularity must align with the connector output frame rate")
        self.post_init()

    def get_input_embeddings(self):
        return self.base_llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_llm.set_input_embeddings(value)

    def _autocast(self):
        if next(self.parameters()).device.type == "cuda":
            return torch.autocast("cuda", dtype=next(self.audio_encoder.parameters()).dtype)
        return contextlib.nullcontext()

    def encode_audio(self, features, feature_lengths):
        with self._autocast(), torch.set_grad_enabled(not self.config.freeze_audio_encoder):
            encoded, lengths, middle = self.audio_encoder.forward_encoder(
                features[:, : int(feature_lengths.max())], feature_lengths
            )
        if self.concat_proj is not None:
            middle = [F.layer_norm(x.permute(1, 0, 2), (x.shape[-1],)) for x in middle]
            concatenated = torch.cat(middle, dim=-1).to(self.concat_proj.weight.dtype)
            encoded = self.concat_proj(concatenated)
        encoded = self.ln_audio(encoded)
        segment = self.config.connector_segment_size
        padding = (-encoded.size(1)) % segment
        if padding:
            encoded = F.pad(encoded, (0, 0, 0, padding))
        encoded = self.connector(encoded.reshape(encoded.size(0), -1, encoded.size(2) * segment))
        lengths = torch.div(lengths + segment - 1, segment, rounding_mode="floor")
        if self.config.inject_temporal_embedding_nl:
            encoded, lengths = self.inject_nl_timestamps(encoded, lengths)
        return encoded, lengths

    def register_nl_timestamp_tokenizer(self, tokenizer):
        """Pre-tokenize the natural-language timestamps used by the released model."""
        self.nl_timestamp_token_ids_list = [
            tokenizer(f"<{i * 0.1:.1f} seconds>", add_special_tokens=False)["input_ids"] for i in range(1201)
        ]

    def inject_nl_timestamps(self, audio_embeds, audio_lengths):
        if self.nl_timestamp_token_ids_list is None:
            raise RuntimeError(
                "Natural-language temporal injection is enabled. Call "
                "model.register_nl_timestamp_tokenizer(tokenizer) before inference or training."
            )
        k = max(1, round(self.config.temporal_granularity * self.output_frame_rate))
        batch, sequence, dim = audio_embeds.shape
        blocks = (sequence + k - 1) // k
        if blocks * k != sequence:
            audio_embeds = F.pad(audio_embeds, (0, 0, 0, blocks * k - sequence))
        embeddings = self.get_input_embeddings()
        timestamp_embeddings, timestamp_lengths = [], []
        for index in range(blocks):
            timestamp = round((index + 1) * self.config.temporal_granularity, 10)
            token_index = max(0, min(round(timestamp / 0.1), 1200))
            token_ids = torch.tensor(
                self.nl_timestamp_token_ids_list[token_index],
                dtype=torch.long,
                device=embeddings.weight.device,
            )
            value = embeddings(token_ids).to(device=audio_embeds.device, dtype=audio_embeds.dtype)
            timestamp_embeddings.append(value)
            timestamp_lengths.append(value.size(0))
        pieces = []
        for index, timestamp_embedding in enumerate(timestamp_embeddings):
            block = audio_embeds[:, index * k : (index + 1) * k]
            stamp = timestamp_embedding.unsqueeze(0).expand(batch, -1, -1)
            pieces.append(torch.cat((block, stamp), dim=1))
        cumulative = torch.tensor(timestamp_lengths, device=audio_embeds.device).cumsum(0)
        valid_blocks = torch.div(audio_lengths + k - 1, k, rounding_mode="floor").clamp(1, blocks)
        return torch.cat(pieces, dim=1), audio_lengths + cumulative[valid_blocks - 1]

    def prepare_multimodal_inputs(
        self, input_ids, attention_mask, audio_features, audio_lengths, audio_counts, labels=None
    ):
        audio_embeds, audio_embed_lengths = self.encode_audio(audio_features, audio_lengths)
        token_embeds = self.get_input_embeddings()(input_ids)
        audio_embeds = audio_embeds.to(device=token_embeds.device, dtype=token_embeds.dtype)
        vision_start_id = self.config.qwen_config.get("vision_start_token_id", 151652)
        output_embeds, output_masks, output_labels, cursor = [], [], [], 0
        for row in range(input_ids.size(0)):
            starts = (input_ids[row] == vision_start_id).nonzero(as_tuple=True)[0].tolist()
            count = int(audio_counts[row])
            if len(starts) != count:
                raise ValueError(f"Sample {row} has {len(starts)} <audio> placeholders but {count} audio files")
            pieces, masks, label_pieces, previous = [], [], [], 0
            for position in starts:
                pieces.append(token_embeds[row, previous : position + 1])
                masks.append(attention_mask[row, previous : position + 1])
                if labels is not None:
                    label_pieces.append(labels[row, previous : position + 1])
                length = int(audio_embed_lengths[cursor])
                pieces.append(audio_embeds[cursor, :length])
                masks.append(torch.ones(length, dtype=attention_mask.dtype, device=attention_mask.device))
                if labels is not None:
                    label_pieces.append(torch.full((length,), -100, dtype=labels.dtype, device=labels.device))
                cursor += 1
                previous = position + 1
            pieces.append(token_embeds[row, previous:])
            masks.append(attention_mask[row, previous:])
            if labels is not None:
                label_pieces.append(labels[row, previous:])
            output_embeds.append(torch.cat(pieces))
            output_masks.append(torch.cat(masks))
            if labels is not None:
                output_labels.append(torch.cat(label_pieces))
        result = (
            pad_sequence(output_embeds, batch_first=True),
            pad_sequence(output_masks, batch_first=True, padding_value=0),
        )
        if labels is not None:
            result += (pad_sequence(output_labels, batch_first=True, padding_value=-100),)
        return result

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        audio_features=None,
        audio_lengths=None,
        audio_counts=None,
        **kwargs,
    ):
        if audio_features is None:
            return self.base_llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        prepared = self.prepare_multimodal_inputs(
            input_ids, attention_mask, audio_features, audio_lengths, audio_counts, labels
        )
        inputs_embeds, expanded_mask = prepared[:2]
        expanded_labels = prepared[2] if labels is not None else None
        return self.base_llm(
            inputs_embeds=inputs_embeds, attention_mask=expanded_mask, labels=expanded_labels, **kwargs
        )

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, audio_features, audio_lengths, audio_counts, **kwargs):
        inputs_embeds, expanded_mask = self.prepare_multimodal_inputs(
            input_ids, attention_mask, audio_features, audio_lengths, audio_counts
        )
        return self.base_llm.generate(inputs_embeds=inputs_embeds, attention_mask=expanded_mask, **kwargs)
