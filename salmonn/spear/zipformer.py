#!/usr/bin/env python3
# Copyright    2022-2023  Xiaomi Corp.        (authors: Daniel Povey,
#                                                       Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import warnings
from typing import List, Optional, Tuple, Union
import logging
import torch
import random
from .scaling import (
    Balancer,
    BiasNorm,
    Dropout2,
    ChunkCausalDepthwiseConv1d,
    ActivationDropoutAndLinear,
    ScaledLinear,  # not as in other dirs.. just scales down initial parameter values.
    Whiten,
    Identity,  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
    penalize_abs_values_gt,
    softmax,
    ScheduledFloat,
    FloatLike,
    limit_param_value,
    convert_num_channels,
)
from torch import Tensor, nn


class EncoderInterface(nn.Module):
    def forward(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A tensor of shape (batch_size, input_seq_len, num_features)
            containing the input features.
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames
            in `x` before padding.
        Returns:
          Return a tuple containing two tensors:
            - encoder_out, a tensor of (batch_size, out_seq_len, output_dim)
              containing unnormalized probabilities, i.e., the output of a
              linear layer.
            - encoder_out_lens, a tensor of shape (batch_size,) containing
              the number of frames in `encoder_out` before padding.
        """
        raise NotImplementedError("Please implement it in a subclass")


class Zipformer2(EncoderInterface):
    """
    Args:

    Note: all "int or Tuple[int]" arguments below will be treated as lists of the same length
    as downsampling_factor if they are single ints or one-element tuples.  The length of
    downsampling_factor defines the number of stacks.

        output_downsampling_factor (int): how much to downsample at the output.  Note:
            we also downsample by a factor of 2 in the Conv2dSubsampling encoder.
            You should probably leave this at 2.
        downsampling_factor (Tuple[int]): downsampling factor for each encoder stack.
           Note: this is in addition to the downsampling factor of 2 that is applied in
           the frontend (self.encoder_embed).
        encoder_dim (Tuple[int]): embedding dimension of each of the encoder stacks, one per
           encoder stack.
        num_encoder_layers (int or Tuple[int])): number of encoder layers for each stack
        encoder_unmasked_dim (int or Tuple[int]): unmasked dimension in each of
            the encoder stacks for purposes of per-frame dropout (recommend 256 for
            now).
        query_head_dim (int or Tuple[int]): dimension of query and key per attention
           head: per stack, if a tuple..
        pos_head_dim (int or Tuple[int]): dimension of positional-encoding projection per
           attention head
        value_head_dim (int or Tuple[int]): dimension of value in each attention head
        num_heads: (int or Tuple[int]): number of heads in the self-attention mechanism.
              Must be at least 4.
        feedforward_dim (int or Tuple[int]): hidden dimension in feedforward modules
        cnn_module_kernel (int or Tuple[int])): Kernel size of convolution module

        pos_dim (int): the dimension of each positional-encoding vector prior to projection,
            e.g. 128.

        dropout (float): dropout rate
        warmup_batches (float): number of batches to warm up over; this controls
          dropout of encoder layers.
        causal (bool): if True, support chunkwise causal convolution.  This should
          not hurt WER as no modeling power is lost, but the convolution modules will be
          slightly slower and use more memory.  Enables use of the chunk_size and
          left_context_chunks options in forward(), which simulates streaming
          decoding.
        chunk_size: (list of int): only set this to other than [-1] if causal;
           the chunk size will be randomly chosen from this list.  -1 means no chunking.
        left_context_frames: (list of int): determines the number of left-
           context chunks for causal training; will be rounded to a number of
           chunks.  Must not be less than cnn_module_kernel (after factoring in
           rounding and downsampling); an error will be thrown if this is violated.
    """

    def __init__(
        self,
        output_downsampling_factor: int = 2,
        downsampling_factor: Tuple[int] = (2, 4),
        encoder_dim: Union[int, Tuple[int]] = 384,
        num_encoder_layers: Union[int, Tuple[int]] = 4,
        encoder_unmasked_dim: Union[int, Tuple[int]] = 256,
        query_head_dim: Union[int, Tuple[int]] = 24,
        pos_head_dim: Union[int, Tuple[int]] = 4,
        value_head_dim: Union[int, Tuple[int]] = 12,
        num_heads: Union[int, Tuple[int]] = 8,
        feedforward_dim: Union[int, Tuple[int]] = 1536,
        cnn_module_kernel: Union[int, Tuple[int]] = 31,
        pos_dim: int = 192,
        dropout: FloatLike = None,  # see code below for default
        warmup_batches: float = 4000.0,
        causal: bool = False,
        chunk_size: Tuple[int] = [-1],
        left_context_frames: Tuple[int] = [-1],
    ) -> None:
        super(Zipformer2, self).__init__()

        if dropout is None:
            dropout = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))

        def _to_tuple(x):
            """Converts a single int or a 1-tuple of an int to a tuple with the same length
            as downsampling_factor"""
            if isinstance(x, int):
                x = (x,)
            if len(x) == 1:
                x = x * len(downsampling_factor)
            else:
                assert len(x) == len(downsampling_factor) and isinstance(x[0], int)
            return x

        self.output_downsampling_factor = output_downsampling_factor  # int
        self.downsampling_factor = downsampling_factor  # tuple
        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim)  # tuple
        self.encoder_unmasked_dim = encoder_unmasked_dim = _to_tuple(
            encoder_unmasked_dim
        )  # tuple
        num_encoder_layers = _to_tuple(num_encoder_layers)
        self.num_encoder_layers = num_encoder_layers
        self.query_head_dim = query_head_dim = _to_tuple(query_head_dim)
        self.value_head_dim = value_head_dim = _to_tuple(value_head_dim)
        pos_head_dim = _to_tuple(pos_head_dim)
        self.num_heads = num_heads = _to_tuple(num_heads)
        feedforward_dim = _to_tuple(feedforward_dim)
        self.cnn_module_kernel = cnn_module_kernel = _to_tuple(cnn_module_kernel)

        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames

        for u, d in zip(encoder_unmasked_dim, encoder_dim):
            assert u <= d

        # each one will be Zipformer2Encoder or DownsampledZipformer2Encoder
        encoders = []

        num_encoders = len(downsampling_factor)
        for i in range(num_encoders):
            encoder_layer = Zipformer2EncoderLayer(
                embed_dim=encoder_dim[i],
                pos_dim=pos_dim,
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                pos_head_dim=pos_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_dim=feedforward_dim[i],
                dropout=dropout,
                cnn_module_kernel=cnn_module_kernel[i],
                causal=causal,
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = Zipformer2Encoder(
                encoder_layer,
                num_encoder_layers[i],
                pos_dim=pos_dim,
                dropout=dropout,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
                final_layerdrop_rate=0.035 * (downsampling_factor[i] ** 0.5),
            )

            if downsampling_factor[i] != 1:
                encoder = DownsampledZipformer2Encoder(
                    encoder,
                    dim=encoder_dim[i],
                    downsample=downsampling_factor[i],
                    dropout=dropout,
                )

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        if output_downsampling_factor >= 2:
            self.downsample_output = SimpleDownsample(
                max(encoder_dim), downsample=output_downsampling_factor, dropout=dropout
            )
        else:
            self.downsample_output = None
            

    def get_feature_masks(self, x: Tensor) -> Union[List[float], List[Tensor]]:
        """
        In eval mode, returns [1.0] * num_encoders; in training mode, returns a number of
        randomized feature masks, one per encoder.
        On e.g. 15% of frames, these masks will zero out all enocder dims larger than
        some supplied number, e.g. >256, so in effect on those frames we are using
        a smaller encoer dim.

        We generate the random masks at this level because we want the 2 masks to 'agree'
        all the way up the encoder stack. This will mean that the 1st mask will have
        mask values repeated self.zipformer_subsampling_factor times.

        Args:
           x: the embeddings (needed for the shape and dtype and device), of shape
             (1, batch_size, encoder_dims0)
        """
        num_encoders = len(self.encoder_dim)
        if not self.training:
            return [1.0] * num_encoders

        (num_frames0, batch_size, _encoder_dims0) = x.shape

        assert self.encoder_dim[0] == _encoder_dims0, (
            self.encoder_dim[0],
            _encoder_dims0,
        )

        feature_mask_dropout_prob = 0.125

        # mask1 shape: (1, batch_size, 1)
        mask1 = (
            torch.rand(1, batch_size, 1, device=x.device) > feature_mask_dropout_prob
        ).to(x.dtype)

        # mask2 has additional sequences masked, about twice the number.
        mask2 = torch.logical_and(
            mask1,
            (
                torch.rand(1, batch_size, 1, device=x.device)
                > feature_mask_dropout_prob
            ).to(x.dtype),
        )

        # dim: (1, batch_size, 2)
        mask = torch.cat((mask1, mask2), dim=-1)

        feature_masks = []
        for i in range(num_encoders):
            channels = self.encoder_dim[i]
            feature_mask = torch.ones(
                1, batch_size, channels, dtype=x.dtype, device=x.device
            )
            u1 = self.encoder_unmasked_dim[i]
            u2 = u1 + (channels - u1) // 2

            feature_mask[:, :, u1:u2] *= mask[..., 0:1]
            feature_mask[:, :, u2:] *= mask[..., 1:2]

            feature_masks.append(feature_mask)

        return feature_masks

    def get_chunk_info(self) -> Tuple[int, int]:
        """
        Returns chunk_size and left_context_chunks.
        """
        if not self.causal:
            return -1, -1

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            assert len(self.chunk_size) == 1, self.chunk_size
            chunk_size = self.chunk_size[0]
        else:
            chunk_size = random.choice(self.chunk_size)

        if chunk_size == -1:
            left_context_chunks = -1
        else:
            if torch.jit.is_scripting() or torch.jit.is_tracing():
                assert len(self.left_context_frames) == 1, self.left_context_frames
                left_context_frames = self.left_context_frames[0]
            else:
                left_context_frames = random.choice(self.left_context_frames)
            # Note: in Python, -1 // n == -1 for n > 0
            left_context_chunks = left_context_frames // chunk_size
            if left_context_chunks == 0:
                left_context_chunks = 1

        return chunk_size, left_context_chunks

    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        return_middle_out: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (seq_len, batch_size, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          src_key_padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
            masked position. May be None.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (output_seq_len, batch_size, max(encoder_dim))
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        outputs = []
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            feature_masks = [1.0] * len(self.encoder_dim)
        else:
            feature_masks = self.get_feature_masks(x)

        chunk_size, left_context_chunks = self.get_chunk_info()

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            # Not support exporting a model for simulating streaming decoding
            attn_mask = None
        else:
            attn_mask = self._get_attn_mask(x, chunk_size, left_context_chunks)

        for i, module in enumerate(self.encoders):
            ds = self.downsampling_factor[i]
            x = convert_num_channels(x, self.encoder_dim[i])

            x = module(
                x,
                chunk_size=chunk_size,
                feature_mask=feature_masks[i],
                src_key_padding_mask=(
                    None
                    if src_key_padding_mask is None
                    else src_key_padding_mask[..., ::ds]
                ),
                attn_mask=attn_mask,
            )
            outputs.append(x)

        # if the last output has the largest dimension, x will be unchanged,
        # it will be the same as outputs[-1].  Otherwise it will be concatenated
        # from different pieces of 'outputs', taking each dimension from the
        # most recent output that has it present.
        x = self._get_full_dim_output(outputs)
        
        if self.output_downsampling_factor >= 2:
            x = self.downsample_output(x)
            # class Downsample has this rounding behavior..
            assert self.output_downsampling_factor == 2, self.output_downsampling_factor
            if torch.jit.is_scripting() or torch.jit.is_tracing():
                lengths = (x_lens + 1) // 2
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lengths = (x_lens + 1) // 2
        else:
            lengths = x_lens
        if return_middle_out:
            return x, lengths, outputs
        else:
            return x, lengths

    def _get_attn_mask(
        self, x: Tensor, chunk_size: int, left_context_chunks: int
    ) -> Optional[Tensor]:
        """
        Return None if chunk_size == -1, else return attention mask of shape
          (seq_len, seq_len), interpreted as (tgt_seq_len, src_seq_len).  True
           means a masked position.
        Args:
           x: embeddings after self.encoder_embed(), of shape (seq_len, batch_size, embed_dim).
          chunk_size: chunk size, must divide
        """
        if chunk_size <= 0:
            return None
        assert all(chunk_size % d == 0 for d in self.downsampling_factor)
        if left_context_chunks >= 0:
            num_encoders = len(self.encoder_dim)
            assert all(
                chunk_size * left_context_chunks
                >= (self.cnn_module_kernel[i] // 2) * self.downsampling_factor[i]
                for i in range(num_encoders)
            )
        else:
            left_context_chunks = 1000000

        seq_len = x.shape[0]

        # t is frame index, shape (seq_len,)
        t = torch.arange(seq_len, dtype=torch.int32, device=x.device)
        # c is chunk index for each frame, shape (seq_len,)
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            c = t // chunk_size
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = t // chunk_size
        src_c = c
        tgt_c = c.unsqueeze(-1)

        attn_mask = torch.logical_or(src_c > tgt_c, src_c < tgt_c - left_context_chunks)
        if __name__ == "__main__":
            logging.info(f"attn_mask = {attn_mask}")
        return attn_mask

    def _get_full_dim_output(self, outputs: List[Tensor]):
        num_encoders = len(self.encoder_dim)
        assert len(outputs) == num_encoders
        output_dim = max(self.encoder_dim)
        output_pieces = [outputs[-1]]
        cur_dim = self.encoder_dim[-1]
        for i in range(num_encoders - 2, -1, -1):
            d = self.encoder_dim[i]
            if d > cur_dim:
                this_output = outputs[i]
                output_pieces.append(this_output[..., cur_dim:d])
                cur_dim = d
        assert cur_dim == output_dim
        return torch.cat(output_pieces, dim=-1)

    def streaming_forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        states: List[Tensor],
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """
        Args:
          x:
            The input tensor. Its shape is (seq_len, batch_size, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          states: list of cached tensors of all encoder layers. For layer-i,
            states[i*6:(i+1)*6] is (cached_key, cached_nonlin_attn, cached_val1, cached_val2,
            cached_conv1, cached_conv2).
          src_key_padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
            masked position. May be None.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (output_seq_len, batch_size, max(encoder_dim))
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
            - updated states
        """
        outputs = []
        new_states = []
        layer_offset = 0

        for i, module in enumerate(self.encoders):
            num_layers = module.num_layers
            ds = self.downsampling_factor[i]
            x = convert_num_channels(x, self.encoder_dim[i])

            x, new_layer_states = module.streaming_forward(
                x,
                states=states[layer_offset * 6 : (layer_offset + num_layers) * 6],
                left_context_len=self.left_context_frames[0] // ds,
                src_key_padding_mask=src_key_padding_mask[..., ::ds],
            )
            layer_offset += num_layers
            outputs.append(x)
            new_states += new_layer_states

        # if the last output has the largest dimension, x will be unchanged,
        # it will be the same as outputs[-1].  Otherwise it will be concatenated
        # from different pieces of 'outputs', taking each dimension from the
        # most recent output that has it present.
        x = self._get_full_dim_output(outputs)
        
        if self.output_downsampling_factor >= 2:
            x = self.downsample_output(x)
            # class Downsample has this rounding behavior..
            assert self.output_downsampling_factor == 2, self.output_downsampling_factor
            if torch.jit.is_scripting() or torch.jit.is_tracing():
                lengths = (x_lens + 1) // 2
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lengths = (x_lens + 1) // 2
        else:
            lengths = x_lens

        return x, lengths, new_states

    @torch.jit.export
    def get_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> List[Tensor]:
        """Get initial states.

        A list of cached tensors of all encoder layers. For layer-i, states[i*6:(i+1)*6]
        is (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
        """
        states = []
        for i, module in enumerate(self.encoders):
            num_layers = module.num_layers
            embed_dim = self.encoder_dim[i]
            ds = self.downsampling_factor[i]
            num_heads = self.num_heads[i]
            key_dim = self.query_head_dim[i] * num_heads
            value_dim = self.value_head_dim[i] * num_heads
            downsample_left = self.left_context_frames[0] // ds
            nonlin_attn_head_dim = 3 * embed_dim // 4
            conv_left_pad = self.cnn_module_kernel[i] // 2
            for layer in range(num_layers):
                cached_key = torch.zeros(downsample_left, batch_size, key_dim).to(
                    device
                )
                cached_nonlin_attn = torch.zeros(
                    1, batch_size, downsample_left, nonlin_attn_head_dim
                ).to(device)
                cached_val1 = torch.zeros(downsample_left, batch_size, value_dim).to(
                    device
                )
                cached_val2 = torch.zeros(downsample_left, batch_size, value_dim).to(
                    device
                )
                cached_conv1 = torch.zeros(batch_size, embed_dim, conv_left_pad).to(
                    device
                )
                cached_conv2 = torch.zeros(batch_size, embed_dim, conv_left_pad).to(
                    device
                )
                states += [
                    cached_key,
                    cached_nonlin_attn,
                    cached_val1,
                    cached_val2,
                    cached_conv1,
                    cached_conv2,
                ]

        return states


def _whitening_schedule(x: float, ratio: float = 2.0) -> ScheduledFloat:
    return ScheduledFloat((0.0, x), (20000.0, ratio * x), default=x)


def _balancer_schedule(min_prob: float):
    return ScheduledFloat((0.0, 0.4), (8000.0, min_prob))


class Zipformer2EncoderLayer(nn.Module):
    """
    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module.

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_dim: int,
        dropout: FloatLike = 0.1,
        cnn_module_kernel: int = 31,
        causal: bool = False,
        attention_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0
        ),
        conv_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0
        ),
        const_attention_rate: FloatLike = ScheduledFloat(
            (0.0, 0.25), (4000.0, 0.025), default=0
        ),
        ff2_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)
        ),
        ff3_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)
        ),
        bypass_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.5), (4000.0, 0.02), default=0
        ),
    ) -> None:
        super(Zipformer2EncoderLayer, self).__init__()
        self.embed_dim = embed_dim

        # self.bypass implements layer skipping as well as bypass; see its default values.
        self.bypass = BypassModule(
            embed_dim, skip_rate=bypass_skip_rate, straight_through_rate=0
        )
        # bypass_mid is bypass used in the middle of the layer.
        self.bypass_mid = BypassModule(embed_dim, straight_through_rate=0)

        # skip probability for dynamic modules (meaning: anything but feedforward).
        self.attention_skip_rate = copy.deepcopy(attention_skip_rate)
        # an additional skip probability that applies to ConvModule to stop it from
        # contributing too much early on.
        self.conv_skip_rate = copy.deepcopy(conv_skip_rate)

        # ff2_skip_rate is to prevent the ff2 module from having output that's too big
        # compared to its residual.
        self.ff2_skip_rate = copy.deepcopy(ff2_skip_rate)
        self.ff3_skip_rate = copy.deepcopy(ff3_skip_rate)

        self.const_attention_rate = copy.deepcopy(const_attention_rate)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            dropout=0.0,
        )

        self.self_attn1 = SelfAttention(embed_dim, num_heads, value_head_dim)

        self.self_attn2 = SelfAttention(embed_dim, num_heads, value_head_dim)

        self.feed_forward1 = FeedforwardModule(
            embed_dim, (feedforward_dim * 3) // 4, dropout
        )

        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim, dropout)

        self.feed_forward3 = FeedforwardModule(
            embed_dim, (feedforward_dim * 5) // 4, dropout
        )

        self.nonlin_attention = NonlinAttention(
            embed_dim, hidden_channels=3 * embed_dim // 4
        )

        self.conv_module1 = ConvolutionModule(
            embed_dim, cnn_module_kernel, causal=causal
        )

        self.conv_module2 = ConvolutionModule(
            embed_dim, cnn_module_kernel, causal=causal
        )

        # TODO: remove it
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))

        self.norm = BiasNorm(embed_dim)

        self.balancer1 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            min_abs=0.2,
            max_abs=4.0,
        )

        # balancer for output of NonlinAttentionModule
        self.balancer_na = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.004), (4000.0, 0.02)),
            prob=0.05,  # out of concern for memory usage
        )

        # balancer for output of feedforward2, prevent it from staying too
        # small.  give this a very small probability, even at the start of
        # training, it's to fix a rare problem and it's OK to fix it slowly.
        self.balancer_ff2 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.1), default=0.0),
            max_abs=2.0,
            prob=0.05,
        )

        self.balancer_ff3 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.2), default=0.0),
            max_abs=4.0,
            prob=0.05,
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(4.0, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.balancer2 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            min_abs=0.1,
            max_abs=4.0,
        )

    def get_sequence_dropout_mask(
        self, x: Tensor, dropout_rate: float
    ) -> Optional[Tensor]:
        if (
            dropout_rate == 0.0
            or not self.training
            or torch.jit.is_scripting()
            or torch.jit.is_tracing()
        ):
            return None
        batch_size = x.shape[1]
        mask = (torch.rand(batch_size, 1, device=x.device) > dropout_rate).to(x.dtype)
        return mask

    def sequence_dropout(self, x: Tensor, dropout_rate: float) -> Tensor:
        """
        Apply sequence-level dropout to x.
        x shape: (seq_len, batch_size, embed_dim)
        """
        dropout_mask = self.get_sequence_dropout_mask(x, dropout_rate)
        if dropout_mask is None:
            return x
        else:
            return x * dropout_mask

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
            Pass the input through the encoder layer.
            Args:
                src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
             pos_emb: (1, 2*seq_len-1, pos_emb_dim) or (batch_size, 2*seq_len-1, pos_emb_dim)
             chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
           feature_mask: something that broadcasts with src, that we'll multiply `src`
                  by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
             attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                    interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                   True means masked position. May be None.
        src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

            Returns:
               A tensor which has the same shape as src
        """
        src_orig = src

        # dropout rate for non-feedforward submodules
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            attention_skip_rate = 0.0
        else:
            attention_skip_rate = (
                float(self.attention_skip_rate) if self.training else 0.0
            )

        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        attn_weights = self.self_attn_weights(
            src,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
        )

        src = src + self.feed_forward1(src)

        self_attn_dropout_mask = self.get_sequence_dropout_mask(
            src, attention_skip_rate
        )

        selected_attn_weights = attn_weights[0:1]
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif not self.training and random.random() < float(self.const_attention_rate):
            # Make attention weights constant.  The intention is to
            # encourage these modules to do something similar to an
            # averaging-over-time operation.
            # only need the mask, can just use the 1st one and expand later
            selected_attn_weights = selected_attn_weights[0:1]
            selected_attn_weights = (selected_attn_weights > 0.0).to(
                selected_attn_weights.dtype
            )
            selected_attn_weights = selected_attn_weights * (
                1.0 / selected_attn_weights.sum(dim=-1, keepdim=True)
            )

        na = self.balancer_na(self.nonlin_attention(src, selected_attn_weights))

        src = src + (
            na if self_attn_dropout_mask is None else na * self_attn_dropout_mask
        )

        self_attn = self.self_attn1(src, attn_weights)

        src = src + (
            self_attn
            if self_attn_dropout_mask is None
            else self_attn * self_attn_dropout_mask
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            conv_skip_rate = 0.0
        else:
            conv_skip_rate = float(self.conv_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(
            self.conv_module1(
                src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask
            ),
            conv_skip_rate,
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            ff2_skip_rate = 0.0
        else:
            ff2_skip_rate = float(self.ff2_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(
            self.balancer_ff2(self.feed_forward2(src)), ff2_skip_rate
        )

        # bypass in the middle of the layer.
        src = self.bypass_mid(src_orig, src)

        self_attn = self.self_attn2(src, attn_weights)

        src = src + (
            self_attn
            if self_attn_dropout_mask is None
            else self_attn * self_attn_dropout_mask
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            conv_skip_rate = 0.0
        else:
            conv_skip_rate = float(self.conv_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(
            self.conv_module2(
                src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask
            ),
            conv_skip_rate,
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            ff3_skip_rate = 0.0
        else:
            ff3_skip_rate = float(self.ff3_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(
            self.balancer_ff3(self.feed_forward3(src)), ff3_skip_rate
        )

        src = self.balancer1(src)
        src = self.norm(src)

        src = self.bypass(src_orig, src)

        src = self.balancer2(src)
        src = self.whiten(src)

        return src

    def streaming_forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        cached_key: Tensor,
        cached_nonlin_attn: Tensor,
        cached_val1: Tensor,
        cached_val2: Tensor,
        cached_conv1: Tensor,
        cached_conv2: Tensor,
        left_context_len: int,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Pass the input through the encoder layer in streaming forward mode.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            pos_emb: (1, left_context_len+2*seq_len-1, pos_emb_dim) or
              (batch_size, left_context_len+2*seq_len-1, pos_emb_dim)
            cached_key: cached attention key tensor of left context,
              of shape (left_context_len, batch_size, key_dim)
            cached_nonlin_attn: left context for nonlin_attention module, a Tensor of shape
              (num_heads, batch_size, left_context_len, head_dim)
            cached_val1: cached left context for the first attention module,
              of shape (left_context_len, batch_size, value_dim)
            cached_val2: cached left context for the second attention module,
              of shape (left_context_len, batch_size, value_dim)
            cached_conv1: cached left context for the first convolution module,
              of shape (batch_size, channels, left_pad)
            cached_conv2: cached left context for the second convolution module,
              of shape (batch_size, channels, left_pad)
            left_context_len: number of left context frames.
            src_key_padding_mask:  the mask for padding, of shape
              (batch_size, left_context_len + seq_len); True means masked position.
              May be None.

        Returns:
            - x, with the same shape as src
            - updated cached_key
            - updated cached_nonlin_attn
            - updated cached_val1
            - updated cached_val2
            - updated cached_conv1
            - updated cached_conv2
        """
        src_orig = src

        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        attn_weights, cached_key = self.self_attn_weights.streaming_forward(
            src,
            pos_emb=pos_emb,
            cached_key=cached_key,
            left_context_len=left_context_len,
            key_padding_mask=src_key_padding_mask,
        )

        src = src + self.feed_forward1(src)

        na, cached_nonlin_attn = self.nonlin_attention.streaming_forward(
            src,
            attn_weights[0:1],
            cached_x=cached_nonlin_attn,
            left_context_len=left_context_len,
        )
        src = src + na

        self_attn, cached_val1 = self.self_attn1.streaming_forward(
            src,
            attn_weights=attn_weights,
            cached_val=cached_val1,
            left_context_len=left_context_len,
        )
        src = src + self_attn

        src_conv, cached_conv1 = self.conv_module1.streaming_forward(
            src,
            cache=cached_conv1,
            src_key_padding_mask=src_key_padding_mask[:, left_context_len:],
        )
        src = src + src_conv

        src = src + self.feed_forward2(src)

        # bypass in the middle of the layer.
        src = self.bypass_mid(src_orig, src)

        self_attn, cached_val2 = self.self_attn2.streaming_forward(
            src,
            attn_weights=attn_weights,
            cached_val=cached_val2,
            left_context_len=left_context_len,
        )
        src = src + self_attn

        src_conv, cached_conv2 = self.conv_module2.streaming_forward(
            src,
            cache=cached_conv2,
            src_key_padding_mask=src_key_padding_mask[:, left_context_len:],
        )
        src = src + src_conv

        src = src + self.feed_forward3(src)

        src = self.norm(src)

        src = self.bypass(src_orig, src)

        return (
            src,
            cached_key,
            cached_nonlin_attn,
            cached_val1,
            cached_val2,
            cached_conv1,
            cached_conv2,
        )


class Zipformer2Encoder(nn.Module):
    r"""Zipformer2Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the Zipformer2EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
       pos_dim: the dimension for the relative positional encoding

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> zipformer_encoder = Zipformer2Encoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = zipformer_encoder(src)
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        pos_dim: int,
        dropout: float,
        warmup_begin: float,
        warmup_end: float,
        initial_layerdrop_rate: float = 0.5,
        final_layerdrop_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder_pos = CompactRelPositionalEncoding(
            pos_dim, dropout_rate=0.15, length_factor=1.0
        )

        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

        assert 0 <= warmup_begin <= warmup_end

        delta = (1.0 / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin  # interpreted as a training batch index
        for i in range(num_layers):
            cur_end = cur_begin + delta
            self.layers[i].bypass.skip_rate = ScheduledFloat(
                (cur_begin, initial_layerdrop_rate),
                (cur_end, final_layerdrop_rate),
                default=0.0,
            )
            cur_begin = cur_end

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

        Returns: a Tensor with the same shape as src.
        """
        pos_emb = self.encoder_pos(src)
        output = src

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            output = output * feature_mask

        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                pos_emb,
                chunk_size=chunk_size,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                output = output * feature_mask

        return output

    def streaming_forward(
        self,
        src: Tensor,
        states: List[Tensor],
        left_context_len: int,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            states: list of cached tensors of N encoder layers. For layer-i, states[i*6:(i+1)*6] is
              (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
            left_context_len: Number of left context frames.
            src_key_padding_mask:  the mask for padding, of shape
              (batch_size, left_context_len + seq_len); True means masked position.
              May be None.

        Returns:
          - output, a Tensor with the same shape as src.
          - updated states
        """
        pos_emb = self.encoder_pos(src, left_context_len)
        output = src

        new_states = []
        for i, mod in enumerate(self.layers):
            (
                cached_key,
                cached_nonlin_attn,
                cached_val1,
                cached_val2,
                cached_conv1,
                cached_conv2,
            ) = states[i * 6 : (i + 1) * 6]
            (
                output,
                new_cached_key,
                new_cached_nonlin_attn,
                new_cached_val1,
                new_cached_val2,
                new_cached_conv1,
                new_cached_conv2,
            ) = mod.streaming_forward(
                output,
                pos_emb,
                cached_key=cached_key,
                cached_nonlin_attn=cached_nonlin_attn,
                cached_val1=cached_val1,
                cached_val2=cached_val2,
                cached_conv1=cached_conv1,
                cached_conv2=cached_conv2,
                left_context_len=left_context_len,
                src_key_padding_mask=src_key_padding_mask,
            )
            new_states += [
                new_cached_key,
                new_cached_nonlin_attn,
                new_cached_val1,
                new_cached_val2,
                new_cached_conv1,
                new_cached_conv2,
            ]

        return output, new_states


class BypassModule(nn.Module):
    """
    An nn.Module that implements a learnable bypass scale, and also randomized per-sequence
    layer-skipping.  The bypass is limited during early stages of training to be close to
    "straight-through", i.e. to not do the bypass operation much initially, in order to
    force all the modules to learn something.
    """

    def __init__(
        self,
        embed_dim: int,
        skip_rate: FloatLike = 0.0,
        straight_through_rate: FloatLike = 0.0,
        scale_min: FloatLike = ScheduledFloat((0.0, 0.9), (20000.0, 0.2), default=0),
        scale_max: FloatLike = 1.0,
    ):
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))
        self.skip_rate = copy.deepcopy(skip_rate)
        self.straight_through_rate = copy.deepcopy(straight_through_rate)
        self.scale_min = copy.deepcopy(scale_min)
        self.scale_max = copy.deepcopy(scale_max)

    def _get_bypass_scale(self, batch_size: int):
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 correponds to bypassing
        # this module.
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(
                self.bypass_scale, min=float(self.scale_min), max=float(self.scale_max)
            )
            skip_rate = float(self.skip_rate)
            if skip_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) > skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for sequences
                # on which we have randomly chosen to do layer-skipping.
            straight_through_rate = float(self.straight_through_rate)
            if straight_through_rate != 0.0:
                mask = (
                    torch.rand((batch_size, 1), device=ans.device)
                    < straight_through_rate
                )
                ans = torch.maximum(ans, mask.to(ans.dtype))
            return ans

    def forward(self, src_orig: Tensor, src: Tensor):
        """
        Args: src_orig and src are both of shape (seq_len, batch_size, num_channels)
        Returns: something with the same shape as src and src_orig
        """
        bypass_scale = self._get_bypass_scale(src.shape[1])
        return src_orig + (src - src_orig) * bypass_scale


class DownsampledZipformer2Encoder(nn.Module):
    r"""
    DownsampledZipformer2Encoder is a zipformer encoder evaluated at a reduced frame rate,
    after convolutional downsampling, and then upsampled again at the output, and combined
    with the origin input, so that the output has the same shape as the input.
    """

    def __init__(
        self, encoder: nn.Module, dim: int, downsample: int, dropout: FloatLike
    ):
        super(DownsampledZipformer2Encoder, self).__init__()
        self.downsample_factor = downsample
        self.downsample = SimpleDownsample(dim, downsample, dropout)
        self.num_layers = encoder.num_layers
        self.encoder = encoder
        self.upsample = SimpleUpsample(dim, downsample)
        self.out_combiner = BypassModule(dim, straight_through_rate=0)

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

        Returns: a Tensor with the same shape as src.
        """
        src_orig = src
        src = self.downsample(src)
        ds = self.downsample_factor
        if attn_mask is not None:
            attn_mask = attn_mask[::ds, ::ds]

        src = self.encoder(
            src,
            chunk_size=chunk_size // ds,
            feature_mask=feature_mask,
            attn_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[: src_orig.shape[0]]

        return self.out_combiner(src_orig, src)

    def streaming_forward(
        self,
        src: Tensor,
        states: List[Tensor],
        left_context_len: int,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        r"""Downsample, go through encoder, upsample, in streaming forward mode.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            states: list of cached tensors of N encoder layers. For layer-i, states[i*6:(i+1)*6] is
              (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
            left_context_len: Number of left context frames.
            src_key_padding_mask: the mask for padding, of shape (batch_size, left_context_len+seq_len);
              True means masked position. May be None.

        Returns:
            - output, a Tensor with the same shape as src.
            - updated states
        """
        src_orig = src
        src = self.downsample(src)

        src, new_states = self.encoder.streaming_forward(
            src,
            states=states,
            left_context_len=left_context_len,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[: src_orig.shape[0]]

        return self.out_combiner(src_orig, src), new_states


class SimpleDownsample(torch.nn.Module):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """

    def __init__(self, channels: int, downsample: int, dropout: FloatLike):
        super(SimpleDownsample, self).__init__()

        self.bias = nn.Parameter(torch.zeros(downsample))

        self.name = None  # will be set from training code
        self.dropout = copy.deepcopy(dropout)

        self.downsample = downsample

    def forward(self, src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, channels)
        """
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        # Pad to an exact multiple of self.downsample
        # right-pad src, repeating the last element.
        pad = d_seq_len * ds - seq_len
        src_extra = src[src.shape[0] - 1 :].expand(pad, src.shape[1], src.shape[2])
        src = torch.cat((src, src_extra), dim=0)
        assert src.shape[0] == d_seq_len * ds

        src = src.reshape(d_seq_len, ds, batch_size, in_channels)

        weights = self.bias.softmax(dim=0)
        # weights: (downsample, 1, 1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        # ans1 is the first `in_channels` channels of the output
        ans = (src * weights).sum(dim=1)

        return ans


class SimpleUpsample(torch.nn.Module):
    """
    A very simple form of upsampling that mostly just repeats the input, but
    also adds a position-specific bias.
    """

    def __init__(self, num_channels: int, upsample: int):
        super(SimpleUpsample, self).__init__()
        self.upsample = upsample

    def forward(self, src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, num_channels)
        Returns a tensor of shape
           ( (seq_len*upsample), batch_size, num_channels)
        """
        upsample = self.upsample
        (seq_len, batch_size, num_channels) = src.shape
        src = src.unsqueeze(1).expand(seq_len, upsample, batch_size, num_channels)
        src = src.reshape(seq_len * upsample, batch_size, num_channels)
        return src


class CompactRelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module.  This version is "compact" meaning it is able to encode
    the important information about the relative position in a relatively small number of dimensions.
    The goal is to make it so that small differences between large relative offsets (e.g. 1000 vs. 1001)
    make very little difference to the embedding.   Such differences were potentially important
    when encoding absolute position, but not important when encoding relative position because there
    is now no need to compare two large offsets with each other.

    Our embedding works done by projecting the interval [-infinity,infinity] to a finite interval
    using the atan() function, before doing the fourier transform of that fixed interval.  The
    atan() function would compress the "long tails" too small,
    making it hard to distinguish between different magnitudes of large offsets, so we use a logarithmic
    function to compress large offsets to a smaller range before applying atan().
    Scalings are chosen in such a way that the embedding can clearly distinguish invidual offsets as long
    as they are quite close to the origin, e.g. abs(offset) <= about sqrt(embedding_dim)


    Args:
        embed_dim: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length: just a heuristic for initialization.
        length_factor: a heuristic scale (should be >= 1.0) which, if larger, gives
           less weight to small differences of offset near the origin.
    """

    def __init__(
        self,
        embed_dim: int,
        dropout_rate: FloatLike,
        max_len: int = 1000,
        length_factor: float = 1.0,
    ) -> None:
        """Construct a CompactRelPositionalEncoding object."""
        super(CompactRelPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0
        self.dropout = Dropout2(dropout_rate)
        self.pe = None
        assert length_factor >= 1.0
        self.length_factor = length_factor
        self.extend_pe(torch.tensor(0.0).expand(max_len))

    def extend_pe(self, x: Tensor, left_context_len: int = 0) -> None:
        """Reset the positional encodings."""
        T = x.size(0) + left_context_len

        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(0) >= T * 2 - 1:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        # if T == 4, x would contain [ -3, -2, 1, 0, 1, 2, 3 ]
        x = torch.arange(-(T - 1), T, device=x.device).to(torch.float32).unsqueeze(1)

        freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)

        # `compression_length` this is arbitrary/heuristic, if it is larger we have more resolution
        # for small time offsets but less resolution for large time offsets.
        compression_length = self.embed_dim**0.5
        # x_compressed, like X, goes from -infinity to infinity as T goes from -infinity to infinity;
        # but it does so more slowly than T for large absolute values of T.
        # The formula is chosen so that d(x_compressed )/dx is 1 around x == 0, which
        # is important.
        x_compressed = (
            compression_length
            * x.sign()
            * ((x.abs() + compression_length).log() - math.log(compression_length))
        )

        # if self.length_factor == 1.0, then length_scale is chosen so that the
        # FFT can exactly separate points close to the origin (T == 0).  So this
        # part of the formulation is not really heuristic.
        # But empirically, for ASR at least, length_factor > 1.0 seems to work better.
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)

        # note for machine implementations: if atan is not available, we can use:
        #   x.sign() * ((1 / (x.abs() + 1)) - 1)  * (-math.pi/2)
        #  check on wolframalpha.com: plot(sign(x) *  (1 / ( abs(x) + 1) - 1 ) * -pi/2 , atan(x))
        x_atan = (x_compressed / length_scale).atan()  # results between -pi and pi

        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()

        pe = torch.zeros(x.shape[0], self.embed_dim, device=x.device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0  # for bias.

        self.pe = pe.to(dtype=x.dtype)

    def forward(self, x: Tensor, left_context_len: int = 0) -> Tensor:
        """Create positional encoding.

        Args:
            x (Tensor): Input tensor (time, batch, `*`).
            left_context_len: (int): Length of cached left context.

        Returns:
            positional embedding, of shape (batch, left_context_len + 2*time-1, `*`).
        """
        self.extend_pe(x, left_context_len)
        x_size_left = x.size(0) + left_context_len
        # length of positive side: x.size(0) + left_context_len
        # length of negative side: x.size(0)
        pos_emb = self.pe[
            self.pe.size(0) // 2
            - x_size_left
            + 1 : self.pe.size(0) // 2  # noqa E203
            + x.size(0),
            :,
        ]
        pos_emb = pos_emb.unsqueeze(0)
        return self.dropout(pos_emb)


class RelPositionMultiheadAttentionWeights(nn.Module):
    r"""Module that computes multi-head attention weights with relative position encoding.
    Various other modules consume the resulting attention weights: see, for example, the
    SimpleAttention module which allows you to compute conventional attention.

    This is a quite heavily modified from: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    we have to write up the differences.


    Args:
           embed_dim: number of channels at the input to this module, e.g. 256
             pos_dim: dimension of the positional encoding vectors, e.g. 128.
           num_heads:  number of heads to compute weights for, e.g. 8
     query_head_dim: dimension of the query (and key), per head.  e.g. 24.
       pos_head_dim: dimension of the projected positional encoding per head, e.g. 4.
            dropout: dropout probability for attn_output_weights. Default: 0.0.
       pos_emb_skip_rate: probability for skipping the pos_emb part of the scores on
                     any given call to forward(), in training time.
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        dropout: float = 0.0,
        pos_emb_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.0)),
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout
        self.pos_emb_skip_rate = copy.deepcopy(pos_emb_skip_rate)
        self.name = None  # will be overwritten in training code; for diagnostics.

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.in_proj = ScaledLinear(
            embed_dim, in_proj_dim, bias=True, initial_scale=query_head_dim**-0.25
        )

        self.whiten_keys = Whiten(
            num_groups=num_heads,
            whitening_limit=_whitening_schedule(3.0),
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )

        # add a balancer for the keys that runs with very small probability, and
        # tries to enforce that all dimensions have mean around zero.  The
        # weights produced by this module are invariant to adding a constant to
        # the keys, so the derivative of the bias is mathematically zero; but
        # due to how Adam/ScaledAdam work, it can learn a fairly large nonzero
        # bias because the small numerical roundoff tends to have a non-random
        # sign.  This module is intended to prevent that.  Use a very small
        # probability; that should be suffixient to fix the problem.
        self.balance_keys = Balancer(
            key_head_dim * num_heads,
            channel_dim=-1,
            min_positive=0.4,
            max_positive=0.6,
            min_abs=0.0,
            max_abs=100.0,
            prob=0.025,
        )

        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(
            pos_dim, num_heads * pos_head_dim, bias=False, initial_scale=0.05
        )

        # the following are for diagnosics only, see --print-diagnostics option
        self.copy_pos_query = Identity()
        self.copy_query = Identity()

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            x: input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional embedding tensor, of shape (1, 2*seq_len - 1, pos_dim)
            key_padding_mask: a bool tensor of shape (batch_size, seq_len).  Positions that
               are True in this mask will be ignored as sources in the attention weighting.
            attn_mask: mask of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len),
               interpreted as ([batch_size,] tgt_seq_len, src_seq_len)
               saying which positions are allowed to attend to which other positions.
        Returns:
           a tensor of attention weights, of shape (hum_heads, batch_size, seq_len, seq_len)
           interpreted as (hum_heads, batch_size, tgt_seq_len, src_seq_len).
        """
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads

        seq_len, batch_size, _ = x.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = x[..., 0:query_dim]
        k = x[..., query_dim : 2 * query_dim]
        # p is the position-encoding query
        p = x[..., 2 * query_dim :]
        assert p.shape[-1] == num_heads * pos_head_dim

        q = self.copy_query(q)  # for diagnostics only, does nothing.
        k = self.whiten_keys(self.balance_keys(k))  # does nothing in the forward pass.
        p = self.copy_pos_query(p)  # for diagnostics only, does nothing.

        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        p = p.permute(2, 1, 0, 3)  # (head, batch, time1, pos_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        use_pos_scores = False
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            # We can't put random.random() in the same line
            use_pos_scores = True
        elif not self.training or random.random() >= float(self.pos_emb_skip_rate):
            use_pos_scores = True

        if use_pos_scores:
            pos_emb = self.linear_pos(pos_emb)
            seq_len2 = 2 * seq_len - 1
            pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(
                2, 0, 3, 1
            )
            # pos shape now: (head, {1 or batch_size}, pos_dim, seq_len2)

            # (head, batch, time1, pos_dim) x (head, 1, pos_dim, seq_len2) -> (head, batch, time1, seq_len2)
            #  [where seq_len2 represents relative position.]
            pos_scores = torch.matmul(p, pos_emb)
            # the following .as_strided() expression converts the last axis of pos_scores from relative
            # to absolute position.  I don't know whether I might have got the time-offsets backwards or
            # not, but let this code define which way round it is supposed to be.
            if torch.jit.is_tracing():
                (num_heads, batch_size, time1, n) = pos_scores.shape
                rows = torch.arange(start=time1 - 1, end=-1, step=-1)
                cols = torch.arange(seq_len)
                rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
                indexes = rows + cols
                pos_scores = pos_scores.reshape(-1, n)
                pos_scores = torch.gather(pos_scores, dim=1, index=indexes)
                pos_scores = pos_scores.reshape(num_heads, batch_size, time1, seq_len)
            else:
                pos_scores = pos_scores.as_strided(
                    (num_heads, batch_size, seq_len, seq_len),
                    (
                        pos_scores.stride(0),
                        pos_scores.stride(1),
                        pos_scores.stride(2) - pos_scores.stride(3),
                        pos_scores.stride(3),
                    ),
                    storage_offset=pos_scores.stride(3) * (seq_len - 1),
                )

            attn_scores = attn_scores + pos_scores

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif self.training and random.random() < 0.1:
            # This is a harder way of limiting the attention scores to not be
            # too large.  It incurs a penalty if any of them has an absolute
            # value greater than 50.0.  this should be outside the normal range
            # of the attention scores.  We use this mechanism instead of, say,
            # something added to the loss function involving the entropy,
            # because once the entropy gets very small gradients through the
            # softmax can become very small, and we'd get zero derivatives.  The
            # choices of 1.0e-04 as the scale on the penalty makes this
            # mechanism vulnerable to the absolute scale of the loss function,
            # but we view this as a failsafe to avoid "implausible" parameter
            # values rather than a regularization method that should be active
            # under normal circumstances.
            attn_scores = penalize_abs_values_gt(
                attn_scores, limit=25.0, penalty=1.0e-04, name=self.name
            )

        assert attn_scores.shape == (num_heads, batch_size, seq_len, seq_len)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            # use -1000 to avoid nan's where attn_mask and key_padding_mask make
            # all scores zero.  It's important that this be large enough that exp(-1000)
            # is exactly zero, for reasons related to const_attention_rate, it
            # compares the final weights with zero.
            attn_scores = attn_scores.masked_fill(attn_mask, -1000)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                batch_size,
                seq_len,
            ), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        # We use our own version of softmax, defined in scaling.py, which should
        # save a little of the memory used in backprop by, if we are in
        # automatic mixed precision mode (amp / autocast), by only storing the
        # half-precision output for backprop purposes.
        attn_weights = softmax(attn_scores, dim=-1)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif random.random() < 0.001 and not self.training:
            self._print_attn_entropy(attn_weights)

        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        return attn_weights

    def streaming_forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        cached_key: Tensor,
        left_context_len: int,
        key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            x: input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional embedding tensor, of shape (1, left_context_len+2*seq_len-1, pos_dim)
            cached_key: cached attention key tensor of left context,
              of shape (left_context_len, batch_size, key_dim)
            left_context_len: number of left context frames.
            key_padding_mask: a bool tensor of shape (batch_size, seq_len).  Positions that
              are True in this mask will be ignored as sources in the attention weighting.

        Returns:
           - attention weights, of shape (hum_heads, batch_size, seq_len, seq_len2),
             interpreted as (hum_heads, batch_size, tgt_seq_len, src_seq_len).
           - updated cached attention key tensor of left context.
        """
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads

        seq_len, batch_size, _ = x.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = x[..., 0:query_dim]
        k = x[..., query_dim : 2 * query_dim]
        # p is the position-encoding query
        p = x[..., 2 * query_dim :]
        assert p.shape[-1] == num_heads * pos_head_dim

        # Pad cached left contexts
        assert cached_key.shape[0] == left_context_len, (
            cached_key.shape[0],
            left_context_len,
        )
        k = torch.cat([cached_key, k], dim=0)
        # Update cached left contexts
        cached_key = k[-left_context_len:, ...]

        # The length of key
        k_len = k.shape[0]

        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(k_len, batch_size, num_heads, query_head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        p = p.permute(2, 1, 0, 3)  # (head, batch, time1, pos_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        pos_emb = self.linear_pos(pos_emb)
        seq_len2 = 2 * seq_len - 1 + left_context_len
        pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(
            2, 0, 3, 1
        )
        # pos shape now: (head, {1 or batch_size}, pos_dim, seq_len2)

        # (head, batch, time1, pos_dim) x (head, 1, pos_dim, seq_len2) -> (head, batch, time1, seq_len2)
        #  [where seq_len2 represents relative position.]
        pos_scores = torch.matmul(p, pos_emb)

        if torch.jit.is_tracing():
            (num_heads, batch_size, time1, n) = pos_scores.shape
            rows = torch.arange(start=time1 - 1, end=-1, step=-1)
            cols = torch.arange(k_len)
            rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
            indexes = rows + cols
            pos_scores = pos_scores.reshape(-1, n)
            pos_scores = torch.gather(pos_scores, dim=1, index=indexes)
            pos_scores = pos_scores.reshape(num_heads, batch_size, time1, k_len)
        # the following .as_strided() expression converts the last axis of pos_scores from relative
        # to absolute position.  I don't know whether I might have got the time-offsets backwards or
        # not, but let this code define which way round it is supposed to be.
        else:
            pos_scores = pos_scores.as_strided(
                (num_heads, batch_size, seq_len, k_len),
                (
                    pos_scores.stride(0),
                    pos_scores.stride(1),
                    pos_scores.stride(2) - pos_scores.stride(3),
                    pos_scores.stride(3),
                ),
                storage_offset=pos_scores.stride(3) * (seq_len - 1),
            )

        attn_scores = attn_scores + pos_scores

        assert attn_scores.shape == (
            num_heads,
            batch_size,
            seq_len,
            k_len,
        ), attn_scores.shape

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, k_len), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        attn_weights = attn_scores.softmax(dim=-1)

        return attn_weights, cached_key

    def _print_attn_entropy(self, attn_weights: Tensor):
        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        (num_heads, batch_size, seq_len, seq_len) = attn_weights.shape

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                attn_weights = attn_weights.to(torch.float32)
                attn_weights_entropy = (
                    -((attn_weights + 1.0e-20).log() * attn_weights)
                    .sum(dim=-1)
                    .mean(dim=(1, 2))
                )
                logging.info(
                    f"name={self.name}, attn_weights_entropy = {attn_weights_entropy}"
                )


class SelfAttention(nn.Module):
    """
    The simplest possible attention module.  This one works with already-computed attention
    weights, e.g. as computed by RelPositionMultiheadAttentionWeights.

    Args:
          embed_dim: the input and output embedding dimension
          num_heads: the number of attention heads
          value_head_dim: the value dimension per head
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        value_head_dim: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, num_heads * value_head_dim, bias=True)

        self.out_proj = ScaledLinear(
            num_heads * value_head_dim, embed_dim, bias=True, initial_scale=0.05
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """
        Args:
          x: input tensor, of shape (seq_len, batch_size, embed_dim)
         attn_weights: a tensor of shape (num_heads, batch_size, seq_len, seq_len),
          with seq_len being interpreted as (tgt_seq_len, src_seq_len).  Expect
          attn_weights.sum(dim=-1) == 1.
        Returns:
           a tensor with the same shape as x.
        """
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = self.in_proj(x)  # (seq_len, batch_size, num_heads * value_head_dim)
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, value_head_dim)
        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, seq_len, value_head_dim)

        x = (
            x.permute(2, 1, 0, 3)
            .contiguous()
            .view(seq_len, batch_size, num_heads * value_head_dim)
        )

        # returned value is of shape (seq_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)
        x = self.whiten(x)

        return x

    def streaming_forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
        cached_val: Tensor,
        left_context_len: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: input tensor, of shape (seq_len, batch_size, embed_dim)
            attn_weights: a tensor of shape (num_heads, batch_size, seq_len, seq_len),
              with seq_len being interpreted as (tgt_seq_len, src_seq_len).  Expect
              attn_weights.sum(dim=-1) == 1.
            cached_val: cached attention value tensor of left context,
              of shape (left_context_len, batch_size, value_dim)
            left_context_len: number of left context frames.

        Returns:
           - attention weighted output, a tensor with the same shape as x.
           - updated cached attention value tensor of left context.
        """
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        seq_len2 = seq_len + left_context_len
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len2)

        x = self.in_proj(x)  # (seq_len, batch_size, num_heads * value_head_dim)

        # Pad cached left contexts
        assert cached_val.shape[0] == left_context_len, (
            cached_val.shape[0],
            left_context_len,
        )
        x = torch.cat([cached_val, x], dim=0)
        # Update cached left contexts
        cached_val = x[-left_context_len:, ...]

        x = x.reshape(seq_len2, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, value_head_dim)
        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, seq_len, value_head_dim)

        x = (
            x.permute(2, 1, 0, 3)
            .contiguous()
            .view(seq_len, batch_size, num_heads * value_head_dim)
        )

        # returned value is of shape (seq_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)

        return x, cached_val


class FeedforwardModule(nn.Module):
    """Feedforward module in Zipformer2 model."""

    def __init__(self, embed_dim: int, feedforward_dim: int, dropout: FloatLike):
        super(FeedforwardModule, self).__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)

        self.hidden_balancer = Balancer(
            feedforward_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=1.0,
            min_abs=0.75,
            max_abs=5.0,
        )

        # shared_dim=0 means we share the dropout mask along the time axis
        self.out_proj = ActivationDropoutAndLinear(
            feedforward_dim,
            embed_dim,
            activation="SwooshL",
            dropout_p=dropout,
            dropout_shared_dim=0,
            bias=True,
            initial_scale=0.1,
        )

        self.out_whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(self, x: Tensor):
        x = self.in_proj(x)
        x = self.hidden_balancer(x)
        # out_proj contains SwooshL activation, then dropout, then linear.
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x


class NonlinAttention(nn.Module):
    """This is like the ConvolutionModule, but refactored so that we use multiplication by attention weights (borrowed
       from the attention module) in place of actual convolution.  We also took out the second nonlinearity, the
       one after the attention mechanism.

    Args:
        channels (int): The number of channels of conv layers.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels

        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)

        # balancer that goes before the sigmoid.  Have quite a large min_abs value, at 2.0,
        # because we noticed that well-trained instances of this module have abs-value before the sigmoid
        # starting from about 3, and poorly-trained instances of the module have smaller abs values
        # before the sigmoid.
        self.balancer = Balancer(
            hidden_channels,
            channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.25), (20000.0, 0.05)),
            max_positive=ScheduledFloat((0.0, 0.75), (20000.0, 0.95)),
            min_abs=0.5,
            max_abs=5.0,
        )
        self.tanh = nn.Tanh()

        self.identity1 = Identity()  # for diagnostics.
        self.identity2 = Identity()  # for diagnostics.
        self.identity3 = Identity()  # for diagnostics.

        self.out_proj = ScaledLinear(
            hidden_channels, channels, bias=True, initial_scale=0.05
        )

        self.whiten1 = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(5.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.whiten2 = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(5.0, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """.
                Args:
                   x: a Tensor of shape (seq_len, batch_size, num_channels)
        attn_weights: a Tensor of shape (num_heads, batch_size, seq_len, seq_len)
                Returns:
                   a Tensor with the same shape as x
        """
        x = self.in_proj(x)

        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels

        s, x, y = x.chunk(3, dim=-1)

        # s will go through tanh.

        s = self.balancer(s)
        s = self.tanh(s)

        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = self.whiten1(x)
        x = x * s
        x = self.identity1(x)  # diagnostics only, it's the identity.

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = torch.matmul(attn_weights, x)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)

        y = self.identity2(y)
        x = x * y
        x = self.identity3(x)

        x = self.out_proj(x)
        x = self.whiten2(x)
        return x

    def streaming_forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
        cached_x: Tensor,
        left_context_len: int,
    ) -> Tuple[Tensor, Tensor]:
        """.
        Args:
            x: a Tensor of shape (seq_len, batch_size, num_channels)
            attn_weights: a Tensor of shape (num_heads, batch_size, seq_len, seq_len)
            cached_x: left context, a Tensor of shape
              (num_heads, batch_size, left_context_len, head_dim)
            left_context_len: number of left context frames.
        Returns:
            - a Tensor with the same shape as x
            - updated left context with same shape as cached_x
        """
        x = self.in_proj(x)

        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels

        s, x, y = x.chunk(3, dim=-1)

        # s will go through tanh.
        s = self.tanh(s)

        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = x * s

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (
            num_heads,
            batch_size,
            seq_len,
            left_context_len + seq_len,
        )

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, head_dim)

        # Pad cached tensor
        assert cached_x.shape[2] == left_context_len, (
            cached_x.shape[2],
            left_context_len,
        )
        x_pad = torch.cat([cached_x, x], dim=2)
        # Update cached tensor
        cached_x = x_pad[:, :, -left_context_len:, :]

        x = torch.matmul(attn_weights, x_pad)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)

        x = x * y

        x = self.out_proj(x)
        return x, cached_x


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zipformer2 model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/zipformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        causal: bool,
    ) -> None:
        """Construct a ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        bottleneck_dim = channels
        self.causal = causal

        self.in_proj = nn.Linear(
            channels,
            2 * bottleneck_dim,
        )
        # the gradients on in_proj are a little noisy, likely to do with the
        # sigmoid in glu.

        # after in_proj we put x through a gated linear unit (nn.functional.glu).
        # For most layers the normal rms value of channels of x seems to be in the range 1 to 4,
        # but sometimes, for some reason, for layer 0 the rms ends up being very large,
        # between 50 and 100 for different channels.  This will cause very peaky and
        # sparse derivatives for the sigmoid gating function, which will tend to make
        # the loss function not learn effectively.  (for most layers the average absolute values
        # are in the range 0.5..9.0, and the average p(x>0), i.e. positive proportion,
        # at the output of pointwise_conv1.output is around 0.35 to 0.45 for different
        # layers, which likely breaks down as 0.5 for the "linear" half and
        # 0.2 to 0.3 for the part that goes into the sigmoid.  The idea is that if we
        # constrain the rms values to a reasonable range via a constraint of max_abs=10.0,
        # it will be in a better position to start learning something, i.e. to latch onto
        # the correct range.
        self.balancer1 = Balancer(
            bottleneck_dim,
            channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.05), (8000.0, 0.025)),
            max_positive=1.0,
            min_abs=1.5,
            max_abs=ScheduledFloat((0.0, 5.0), (8000.0, 10.0), default=1.0),
        )

        self.activation1 = Identity()  # for diagnostics

        self.sigmoid = nn.Sigmoid()

        self.activation2 = Identity()  # for diagnostics

        assert kernel_size % 2 == 1

        self.depthwise_conv = (
            ChunkCausalDepthwiseConv1d(channels=bottleneck_dim, kernel_size=kernel_size)
            if causal
            else nn.Conv1d(
                in_channels=bottleneck_dim,
                out_channels=bottleneck_dim,
                groups=bottleneck_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

        self.balancer2 = Balancer(
            bottleneck_dim,
            channel_dim=1,
            min_positive=ScheduledFloat((0.0, 0.1), (8000.0, 0.05)),
            max_positive=1.0,
            min_abs=ScheduledFloat((0.0, 0.2), (20000.0, 0.5)),
            max_abs=10.0,
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.out_proj = ActivationDropoutAndLinear(
            bottleneck_dim,
            channels,
            activation="SwooshR",
            dropout_p=0.0,
            initial_scale=0.05,
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        chunk_size: int = -1,
    ) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
           src_key_padding_mask: the mask for the src keys per batch (optional):
               (batch, #time), contains True in masked positions.

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """

        x = self.in_proj(x)  # (time, batch, 2*channels)

        x, s = x.chunk(2, dim=-1)
        s = self.balancer1(s)
        s = self.sigmoid(s)
        x = self.activation1(x)  # identity.
        x = x * s
        x = self.activation2(x)  # identity

        # (time, batch, channels)

        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        if (
            not torch.jit.is_scripting()
            and not torch.jit.is_tracing()
            and chunk_size >= 0
        ):
            # Not support exporting a model for simulated streaming decoding
            assert (
                self.causal
            ), "Must initialize model with causal=True if you use chunk_size"
            x = self.depthwise_conv(x, chunk_size=chunk_size)
        else:
            x = self.depthwise_conv(x)

        x = self.balancer2(x)
        x = x.permute(2, 0, 1)  # (time, batch, channels)

        x = self.whiten(x)  # (time, batch, channels)
        x = self.out_proj(x)  # (time, batch, channels)

        return x

    def streaming_forward(
        self,
        x: Tensor,
        cache: Tensor,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute convolution module in streaming forward mode.

        Args:
            x: Input tensor (#time, batch, channels).
            cache: cached left context for depthwise_conv of shape
              (#batch, channels, left_pad)
            src_key_padding_mask: the mask for the src keys per batch (optional):
              (batch, #time), contains True in masked positions.

        Returns:
            - Output tensor (#time, batch, channels).
            - Updated cache (#batch, channels, left_pad)
        """

        x = self.in_proj(x)  # (time, batch, 2*channels)

        x, s = x.chunk(2, dim=2)
        s = self.sigmoid(s)
        x = x * s
        # (time, batch, channels)

        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        x, cache = self.depthwise_conv.streaming_forward(x, cache=cache)

        x = x.permute(2, 0, 1)  # (time, batch, channels)

        x = self.out_proj(x)  # (time, batch, channels)

        return x, cache


class ScalarMultiply(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


def _test_zipformer_main(causal: bool = False):
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.

    c = Zipformer2(
        encoder_dim=(64, 96),
        encoder_unmasked_dim=(48, 64),
        num_heads=(4, 4),
        causal=causal,
        chunk_size=(4,) if causal else (-1,),
        left_context_frames=(64,),
    )
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(seq_len, batch_size, 64),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f[0].sum().backward()
    c.eval()
    f = c(
        torch.randn(seq_len, batch_size, 64),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_zipformer_main(False)
    _test_zipformer_main(True)
