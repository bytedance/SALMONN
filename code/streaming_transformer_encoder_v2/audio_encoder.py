import logging
import math
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from torchaudio.compliance.kaldi import fbank


from .transformer_encoder import LlamaAudioEncoder
from .subsampling import Conv2dSubsampling, Conv2dSubsampling4

SAMPLING_RATE=16000
LOG_EPS = math.log(1e-10)


class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")

def get_encoder_embed(params) -> nn.Module:
    # encoder_embed converts the input of shape (N, T, num_features)
    # to the shape (N, (T - 7) // 2, encoder_dims).
    # That is, it does two things simultaneously:
    #   (1) subsampling: T -> (T - 7) // 2
    #   (2) embedding: num_features -> encoder_dims
    # In the normal configuration, we will downsample once more at the end
    # by a factor of 2, and most of the encoder stacks will run at a lower
    # sampling rate.
    if params.subsampling_factor == 2:
        logging.info(f"Using subsample factor = 2")
        encoder_embed = Conv2dSubsampling(
            idim=params.feature_dim,
            odim=params.encoder_dim,
        )
    elif params.subsampling_factor == 4:
        logging.info(f"Using subsample factor = 4")
        encoder_embed = Conv2dSubsampling4(
            idim=params.feature_dim,
            odim=params.encoder_dim,
        )
    else:
        raise ValueError()
    return encoder_embed

def get_encoder_model(params) -> nn.Module:
    encoder = LlamaAudioEncoder(
        encoder_dim=params.encoder_dim,
        num_layers=params.num_layers,
        num_attention_heads=params.num_heads,
        hidden_act="gelu",
        use_flash_attention=params.use_flash_attention,
        attention_dropout=params.attention_dropout,
        is_causal=params.causal, # should be True
    )
    return encoder

def torchaudio_fbank(
    wavs: List[torch.Tensor], wav_lens: torch.Tensor
):
    """Compute fbank features

    Args:
        wavs (torch.Tensor): the mono-channel input waveform, (N, T)
        wav_lens (torch.Tensor): the length of each waveform in samples (N)

    Returns:
        The fbank features, and their lengths
    """
    if isinstance(wavs, list):
        assert wavs[0].ndim == 1
        device = wavs[0].device
    else:
        assert instance(wavs, torch.Tensor) 
        assert wavs.ndim == 2, wavs.shape
        device = wavs.device
    low_freq = 20.0
    high_freq=-400.0
    dither=0.0
    snip_egdes=False

    features = []
    for i, wav in enumerate(wavs):
        feat = fbank(
            wav[:wav_lens[i]].unsqueeze(0),
            sample_frequency=16000, # this is fixed to 16000
            num_mel_bins=128,
            low_freq=low_freq,
            snip_edges=snip_egdes,
            high_freq=high_freq,
            dither=dither,
            energy_floor=1.0e-10,
        )
        features.append(feat)
    feat_len = torch.tensor([f.shape[0] for f in features]).to(device)
    features = torch.nn.utils.rnn.pad_sequence(
        features, batch_first=True, padding_value=LOG_EPS
    ).to(device)
    return features, feat_len

class CausalAudioEncoder(nn.Module):
    def __init__(
        self,
        config,
        num_mels: int = 128,
    ):
        super().__init__()
        encoder_embed = get_encoder_embed(config)
        encoder = get_encoder_model(config)
        
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        
        # fbank_config = FbankConfig(num_mel_bins=num_mels)
        # self.fbank_extractor = Fbank(fbank_config)
        
    def compute_fbank(
        self, audio: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute filterbank features

        Args:
            audio (List[torch.Tensor]): A list of 1-D tensor

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The fbank feat and their lengths
        """
        device = next(self.parameters()).device
        features = self.fbank_extractor.extract_batch(audio, sampling_rate=SAMPLING_RATE)
        feat_len = torch.tensor([f.shape[0] for f in features]).to(device)
        features = pad_sequence(features, batch_first=True, padding_value=math.log(1e-10)).to(device)
        return features, feat_len
    
    def compute_fbank_torch(
        self, audio: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # audio: a list of 1D-tensor in shape (num_samples)
        device = audio[0].device
        audio_lens = [a.shape[0] for a in audio]
        audio_lens = torch.tensor(audio_lens).to(device)
        fbank, fbank_lens = torchaudio_fbank(audio, audio_lens)
        return fbank, fbank_lens
    
    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward the log-mel fbank features

        Args:
            x (torch.Tensor): filterbank feat (N,T,C)
            x_lens (torch.Tensor): the length of the fbank (N,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The encoder output and the corresponding length
        """
        assert torch.all(x_lens > 0)
        x, x_lens = self.encoder_embed(x, x_lens) # (B,T,C)

        output = self.encoder(x, x_lens)
        encoder_out = output.last_hidden_state

        return encoder_out, x_lens
    
    def forward(
        self, audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x, x_lens = self.compute_fbank(audio=audio)
        x2, x_lens2 = self.compute_fbank_torch(audio=audio)
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)
        return encoder_out, encoder_out_lens
