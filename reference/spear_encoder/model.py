# Copyright (2026) Tsinghua University, Bytedance Ltd. and/or its affiliates
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

# Adapted from https://huggingface.co/collections/marcoyang/spear-encoders. The original license is located at 'third-party-license/SPEAR.txt'.

import logging
from typing import Optional, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


class MultiKDModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: nn.Module,
        encoder_dim: int,
        num_codebooks: int=8,
        distillation_layer: int=9,
        distillation_delta: int=0,
        teacher_frame_ratio: int = 2,
        interpolate_teacher: bool = False,
        n_mels: int = 128,
        num_events: int = 527,
        mask_mode: str = "w2v2",
        mask_prob: float = 0.65,
        mask_length: int = 10,
        mask_selection: str = "static",
        mask_other: float = 0.0,
        min_masks: int = 2,
        mask_channel_prob: float = 0.0,
        mask_channel_length: int = 10,
        mask_channel_selection: str = "static",
        mask_channel_other: float = 0.0,
        loss_only_mask: bool = False,
    ):
        """A model that performs MVQ KD pre-training .

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          num_codebooks:
            The number of codebooks used in the target
          distillation_layer:
            Use which layer to do MVQ pre-training
          distillation_delta:
            How many frames to delay the alignment between the model and the target frames.
            Should be zero for non-streaming models, and a positive number for streaming models
          teacher_frame_ratio:
            The frame rate ratio between the target and the model output
          mask_mode:
            The masking mode.
                w2v2: the wav2vec2 style of masking, allows overlap
                custom: no overlap, therefore bigger masking ratio 
          mask_prob:
            The probability of selecting choosing one frame as the start index
          mask_length:
            The length of each mask
          mask_selection:
            How to determine the length of the mask, see ``compute_mask_indices''
        """
        super().__init__()
        
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_dim = encoder_dim
            
        self.distillation_layer = distillation_layer
        # the frame ratio between the teacher and student
        # if larger than one, we are basically having more than one set of
        # codebooks for each frame
        self.num_codebooks= num_codebooks
        self.teacher_frame_ratio = teacher_frame_ratio 
        self.interpolate_teacher = interpolate_teacher
        self.distillation_delta = distillation_delta
        
        if num_codebooks > 0:
            from multi_quantization.prediction import JointCodebookLoss
            self.codebook_loss_net = JointCodebookLoss(
                predictor_channels=encoder_dim,
                num_codebooks=num_codebooks * self.teacher_frame_ratio,
                is_joint=False,
                reduction="none",
            )
        else:
            self.codebook_loss_net = None
        
        self.audio_tagging_proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(encoder_dim, num_events),
        ) # 527 classes
        
        # masking related
        assert mask_mode in ["w2v2", "block"], f"Unseen mask mode: {mask_mode}"
        self.mask_mode = mask_mode
        
        self.mask_emb = nn.Parameter(torch.FloatTensor(n_mels).normal_()) 
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.min_masks = min_masks
        
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_length = mask_channel_length
        self.mask_channel_selection = mask_channel_selection
        self.mask_channel_other = mask_channel_other
        
        self.loss_only_mask = loss_only_mask

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        codebook_indexes: torch.Tensor = None,
        at_targets: torch.Tensor = None,
        mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          codebook_indexes:
            Codebook indexes of teacher embeddings
          mask:
            If we perform w2v2 style of masking over the fbank frames
            
        Returns:
          Return the codebook loss
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert codebook_indexes is not None or at_targets is not None

        # apply masking
        if self.training and mask:
            padding_mask = make_pad_mask(x_lens)
            
            # apply masking to the fbank features
            x, mask_indices = self.apply_mask(
                x.clone(),
                padding_mask=padding_mask
            ) # (N,T,C), (N,T)
        else:
            mask_indices = None
        
        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)
            
        if codebook_indexes is not None and self.codebook_loss_net is not None:
            codebook_loss = self.forward_codebook_loss(
                encoder_out, encoder_out_lens, codebook_indexes, reduction="none"
            )
            if self.loss_only_mask and mask_indices is not None:
                # downsample the mask 
                mask_indices = nn.functional.avg_pool1d(mask_indices, 4) >= 0.5
                assert mask_indices.size(1) >= codebook_loss.size(1)
                mask_indices = mask_indices[:, :codebook_loss.size(1)].float()
                codebook_loss = codebook_loss * mask_indices
            codebook_loss = codebook_loss.sum(dim=1) # (B,)    
        else:
            codebook_loss = None
        
        if at_targets is not None:
            at_loss = self.forward_audio_tagging(encoder_out, encoder_out_lens, at_targets, return_logits=False)
        else:
            at_loss = None
        
        return codebook_loss, at_loss

    def forward_codebook_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        codebook_indexes: torch.Tensor,
        reduction: str = "sum",
    ):
        # align the encoder features with the codebook indexes
        if self.interpolate_teacher:
            codebook_indexes = self.interpolate_codebook_indexes(
                encoder_out, codebook_indexes
            )
        else:
            if codebook_indexes.shape[1] != encoder_out.shape[1]:
                # align the codebook indexes to the frame rate of the student encoder out
                codebook_indexes = self.concat_successive_codebook_indexes(
                    encoder_out, codebook_indexes, ratio=self.teacher_frame_ratio
                )
                
        # the delta is associated with the frame-rate of the encoder
        # so a bigger delta maybe necessary for 50Hz student encoder
        if self.distillation_delta > 0:
            codebook_indexes = codebook_indexes[:,:-self.distillation_delta, :]
            encoder_out = encoder_out[:, self.distillation_delta:, :]
            truncated_padding_mask = make_pad_mask(encoder_out_lens - self.distillation_delta)
            codebook_indexes = codebook_indexes.masked_fill(truncated_padding_mask.unsqueeze(-1), value=-100)
            
        N,T,_ = encoder_out.shape
        codebook_loss = self.codebook_loss_net(encoder_out.float(), codebook_indexes)
        codebook_loss = codebook_loss.reshape(N,T,-1)
        num_cb = codebook_loss.size(-1)
        # normalize the loss by the number of codebooks
        if reduction == "sum":
            codebook_loss = codebook_loss.sum(dim=(1,2)) / num_cb # (B,)
        elif reduction == "none":
            codebook_loss = codebook_loss.sum(dim=2) / num_cb # (B,T)
        else:
            raise NotImplementedError()
        
        return codebook_loss

    def forward_audio_tagging(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        target: torch.Tensor = None,
        return_logits: bool = False,
    ):
        # target: (N, num_events)
        logits = self.audio_tagging_proj(encoder_out) # (N, T, num_classes)
        padding_mask = make_pad_mask(encoder_out_lens) # (N,T)
        logits[padding_mask] = 0
        logits = logits.sum(dim=1)
        logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits) # (N, num_events)
        if return_logits:
            return logits
        
        at_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        return at_loss
    
    def apply_mask(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mask according to the mask_mode, return the masked features and the masked positions

        Args:
            x (torch.Tensor): The input fbank features
            padding_mask (torch.Tensor, optional): The padding mask

        Returns:
            The masked fbank feature and the masked_indices, with masked positions as 1
        """
        # apply mask to the fbank features, two modes applicable
        if self.mask_mode == "w2v2":
            x, masked_indices = self.apply_mask_w2v2(x, padding_mask)
        elif self.mask_mode == "block":
            x, masked_indices = self.apply_mask_block(x, padding_mask)
        else:
            raise NotImplementedError()
        
        if random.random() > 0.97:
            logging.info(f"Apply {self.mask_mode} masking. A proportion of {masked_indices.sum()/masked_indices.numel():.2f} frames are masked")
        return x, masked_indices
        
    
    def apply_mask_block(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None
    ):
        B,T,C = x.shape
        assert self.mask_prob > 0.0

        mask_indices = compute_mask_indices_block(
            shape=(B,T),
            padding_mask=padding_mask,
            mask_prob=self.mask_prob,
            mask_length=self.mask_length,
            min_masks=self.min_masks,
        ).to(x.device)
        
        x = index_put(x, mask_indices.bool(), self.mask_emb)

        return x, mask_indices
    
    def apply_mask_w2v2(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor = None
    ):
        # this function is modified from fairseq: https://github.com/facebookresearch/fairseq/blob/bedb259bf34a9fc22073c13a1cee23192fa70ef3/fairseq/models/wav2vec/wav2vec2.py#L429
        # The masked indices have value 1
        B, T, C = x.shape
        
        # we mask channel first, then mask timestamps
        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=False,
                min_space=1,
                require_same_masks=False,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            if random.random() > 0.98:
                logging.info(f"A proportion of {mask_channel_indices.sum()/mask_channel_indices.numel():.2f} feature dims are masked")
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                mask_type=self.mask_selection,
                mask_other=self.mask_other,
                min_masks=2, # fixed
                no_overlap=False,  # False
                min_space=1,  # 1
                require_same_masks=False,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb)
            mask_indices = mask_indices.float()
        else:
            mask_indices = None

        return x, mask_indices
    
    @staticmethod
    def interpolate_codebook_indexes(middle_layer_output, codebook_indexes):
        # This function addresses the case where the teacher has a lower frame rate
        # than the student model
        t_expected = middle_layer_output.shape[1]
        N, T, C = codebook_indexes.shape # C should be 256
        
        codebook_indexes = codebook_indexes.permute(0,2,1).float() # (N,C,T)
        codebook_indexes = torch.nn.functional.interpolate(codebook_indexes, t_expected)
        codebook_indexes = codebook_indexes.permute(0,2,1).int() # (N,T,C)
        
        assert codebook_indexes.shape[1] == middle_layer_output.shape[1]
        return codebook_indexes
    
    @staticmethod
    def concat_successive_codebook_indexes(middle_layer_output, codebook_indexes, ratio=2):
        # Output rate of hubert is 50 frames per second,
        # while that of current encoder is 25.
        # Following code handling two issues:
        # 1.
        #   Roughly speaking, to generate another frame output,
        #   hubert needes extra two frames,
        #   while current encoder needs extra four frames.
        #   Suppose there are only extra three frames provided,
        #   hubert will generate another frame while current encoder does nothing.
        # 2.
        #   codebook loss is a frame-wise loss, to enalbe 25 frames studnet output
        #   learns from 50 frames teacher output, two successive frames of teacher model
        #   output is concatenated together.
        t_expected = middle_layer_output.shape[1]
        N, T, C = codebook_indexes.shape # C should be 256
        
        # Handling issue 1.
        if T >= t_expected * ratio:
            codebook_indexes = codebook_indexes[:, : t_expected * ratio, :]
        else:
            assert t_expected * ratio - T <= 5, (T, t_expected, ratio)
            diff = t_expected * ratio - T
            codebook_indexes = torch.cat(
                [
                    codebook_indexes,
                    torch.full((N,diff,C), -100).to(codebook_indexes.device).to(codebook_indexes.dtype)
                ],
                dim=1,
            )
        assert codebook_indexes.size(1) == middle_layer_output.size(1) * ratio
        
        # Handling issue 2.
        codebook_indexes = codebook_indexes.reshape(N, t_expected, C * ratio)
        assert middle_layer_output.shape[1] == codebook_indexes.shape[1]
        return codebook_indexes
    
def index_put(tensor, indices, value):
    tensor[indices] = value
    return tensor    

def compute_mask_indices_block(
    shape,
    padding_mask,
    mask_prob: float = 0.5,
    mask_length: int = 10,
    min_masks: int = 2,
):
    # self-implemented mask, no overlap
    B,T = shape
    mask_indices = []
    for i in range(B):
        if padding_mask is not None:
            num_segments = (T - padding_mask[i].sum()) // mask_length # discard the last few frames
        else:
            num_segments = T // mask_length 
        segment_mask = torch.rand(num_segments) < mask_prob 
        while sum(segment_mask) < min_masks:
            segment_mask = torch.rand(num_segments) < mask_prob
        segment_mask_expanded = segment_mask.unsqueeze(-1).expand(num_segments, mask_length)
        segment_mask_expanded = segment_mask_expanded.reshape(-1).float()
        if segment_mask_expanded.size(0) < T:
            pad = T - segment_mask_expanded.size(0)
            segment_mask_expanded = torch.cat([segment_mask_expanded, torch.zeros(pad)])
        mask_indices.append(segment_mask_expanded)

    mask_indices = torch.stack(mask_indices)
    return mask_indices

def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
    add_masks: bool = False,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
    idc_select_ver: int = 1,  # 2 to reproduce mask_tokens_dataset
    num_mask_ver: int = 2,  # 2 to reproduce mask_tokens_dataset
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    if num_mask_ver == 1:
        all_num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * all_sz / float(mask_length)
            + np.random.rand()
        )
        all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if seed is not None and epoch is not None and indices is not None:
            seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6)
        else:
            seed_i = None

        rng = np.random.default_rng(seed_i)

        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            assert sz >= 0, sz
        else:
            sz = all_sz

        if num_mask_ver == 1:
            if padding_mask is not None:
                num_mask = int(
                    # add a random number for probabilistic rounding
                    mask_prob * sz / float(mask_length)
                    + np.random.rand()
                )
                num_mask = max(min_masks, num_mask)
            else:
                num_mask = all_num_mask
        elif num_mask_ver == 2:
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + rng.random()
            )
            num_mask = max(min_masks, num_mask)
            hard_max = sz // mask_length
            num_mask = min(hard_max, num_mask) # prevent whole sequence being masked
        else:
            raise ValueError()

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = rng.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = rng.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = rng.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            if mask_type == "static":
                raise ValueError("this should never happens")
            else:
                lengths = [min(mask_length, sz - 1)]

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = rng.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = rng.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            if idc_select_ver == 1:
                min_len = min(lengths)
                if sz - min_len <= num_mask:
                    min_len = sz - num_mask - 1
                mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
            elif idc_select_ver == 2:
                mask_idc = rng.choice(sz, num_mask, replace=False)
            else:
                raise ValueError()

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idc = np.unique(mask_idc[mask_idc < sz])
        if len(mask_idc) >= sz:
            
            raise ValueError(
                (
                    f"the entire sequence is masked. "
                    f"sz={sz}; mask_idc[mask_idc]; "
                    f"index={indices[i] if indices is not None else None}"
                )
            )
        mask_idcs.append(mask_idc)

    target_len = None
    if require_same_masks:
        if add_masks:
            target_len = max([len(m) for m in mask_idcs])
        else:
            target_len = min([len(m) for m in mask_idcs])

    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len:
            mask_idc = rng.choice(mask_idc, target_len, replace=False)

        mask[i, mask_idc] = True

        if target_len is not None and len(mask_idc) < target_len:
            unmasked = np.flatnonzero(~mask[i])
            to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
            mask[i, to_mask] = True

        if mask_dropout > 0:
            masked = np.flatnonzero(mask[i])
            num_holes = np.rint(len(masked) * mask_dropout).astype(int)
            to_drop = rng.choice(masked, num_holes, replace=False)
            mask[i, to_drop] = False

    return mask

def _test_w2v2_channel_mask():
    x = torch.ones(100, 1000, 128)
    B, T, C = x.shape
    
    configs = [(0.25, 15), (0.25, 20), (0.5, 15),]
    # configs = [(0.2, 20), (0.3, 20), (0.4, 20),]
    for config in configs:
        mask_channel_prob, mask_channel_length = config
        ratios = []
        for i in range(20):
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                mask_channel_prob,
                mask_channel_length,
                "static",
                0.0,
                no_overlap=False,
                min_space=1,
                require_same_masks=False,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            ratio = mask_channel_indices.sum() / mask_channel_indices.numel()
            ratios.append(ratio)
        import pdb; pdb.set_trace()
        avg_ratio = sum(ratios) / len(ratios)
        print(f"Current config: mask_channel_prob = {mask_channel_prob}, mask_channel_length = {mask_channel_length}")
        print(f"Averaged masking ratio: {avg_ratio}")

def _test_w2v2_mask():
    x = torch.ones(100, 1000, 128)
    B, T, C = x.shape
    
    mask_prob = 0.65
    mask_length = 10
    
    # configs = [(0.65, 10), (0.01, 40), (0.1, 40), (0.2, 40), (0.2, 20), (0.35, 10), (0.35, 20), (0.25, 20)]
    configs = []
    for i in range(6):
        p = 0.05 + (i+1) * 0.1
        for l in [10, 20, 30, 40]:
            configs.append((p, l))
    configs = [(0.65, 10), (0.02, 40), (0.05, 40), (0.1, 40)]
    for config in configs:
        mask_prob, mask_length = config
        ratios = []
        for i in range(20):
            mask_indices = compute_mask_indices(
                (B, T),
                None,
                mask_prob,
                mask_length,
                mask_type="static",
                mask_other=0.0,
                min_masks=2,
                no_overlap=False,  # False
                min_space=1,  # 1
                require_same_masks=False,
            )
            mask_indices = torch.from_numpy(mask_indices) 
            ratio = mask_indices.sum() / mask_indices.numel()
            ratios.append(ratio)
        avg_ratio = sum(ratios) / len(ratios)
        print(f"Current config: mask_prob = {mask_prob}, mask_length = {mask_length}")
        print(f"Averaged masking ratio: {avg_ratio}")

def _test_custom_mask():
    x = torch.ones(100, 1000, 128)
    B, T, C = x.shape
    
    configs = [(0.5, 20), (0.2, 20), (0.3, 20), (0.4, 20), (0.5, 20)]
    for config in configs:
        mask_prob, mask_length = config
        ratios = []
        for i in range(20):
            all_possible_mask_lengths = [mask_length + i * 2 for i in range(-5, 6)]
            mask_length = random.sample(all_possible_mask_lengths, 1)[0]
            assert mask_length > 0, f"Sampled mask_length smaller than 0, {mask_length}"
            
            mask_indices = compute_mask_indices_block(
                shape=(B, T),
                padding_mask=None,
                mask_prob=mask_prob,
                mask_length=mask_length,
                min_masks=2,
            )
            import pdb; pdb.set_trace()
            ratio = mask_indices.sum() / mask_indices.numel()
            ratios.append(ratio)
        avg_ratio = sum(ratios) / len(ratios)
        print(f"Current config: mask_prob = {mask_prob}, mask_length = {mask_length}")
        print(f"Averaged masking ratio: {avg_ratio}")
        

if __name__=="__main__":
    _test_w2v2_channel_mask()
    # _test_w2v2_mask()
    # _test_custom_mask()