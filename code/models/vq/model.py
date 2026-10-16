"""Temporal RVQ-VAE: features (B, T, D), indices (B, T/4, Q) in the release layout."""
import torch
import torch.nn as nn
from models.vq.encdec import Encoder, Decoder, Causal_Encoder, Causal_Decoder
from models.vq.residual_vq import ResidualVQ

class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        assert output_emb_width == code_dim
        self.code_dim = code_dim
        self.num_code = nb_code
        self.downsample_factor = stride_t ** down_t
        self.left_context_frames = 0
        self.context_batch_size = 128
        self.causal = bool(getattr(args, "causal", False))
        if self.causal and norm not in (None, False, "LN"):
            raise ValueError("Causal RVQ requires per-timestep normalization (None/False or LN)")
        encoder_cls = Causal_Encoder if self.causal else Encoder
        decoder_cls = Causal_Decoder if self.causal else Decoder
        encoder_kwargs = {"end_aligned": True} if self.causal else {}
        self.encoder = encoder_cls(input_width, output_emb_width, down_t, stride_t, width, depth,
                                   dilation_growth_rate, activation=activation, norm=norm, **encoder_kwargs)
        self.decoder = decoder_cls(input_width, output_emb_width, down_t, stride_t, width, depth,
                                   dilation_growth_rate, activation=activation, norm=norm)
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        return x.permute(0, 2, 1)

    def set_context_frames(self, frames):
        # The cap includes the current completed stride-sized frame block.
        frames = int(frames)
        assert frames >= 0 and frames % self.downsample_factor == 0
        assert not frames or self.causal
        self.left_context_frames = frames
        return self

    def _bounded_transform(self, x, module, window, stride, output_stride):
        # x is (B,C,T). Each later output uses an independently reset window;
        # a single final crop cannot produce every earlier output.
        prefix = module(x[..., :window])
        if x.shape[-1] <= window:
            return prefix
        windows = x.unfold(-1, window, stride)[:, :, 1:, :]
        batch, channels, count, _ = windows.shape
        windows = windows.permute(0, 2, 1, 3).reshape(batch * count, channels, window)
        tails = []
        for start in range(0, len(windows), self.context_batch_size):
            tails.append(module(windows[start:start + self.context_batch_size])[..., -output_stride:])
        tail = torch.cat(tails, dim=0).reshape(batch, count, -1, output_stride)
        tail = tail.permute(0, 2, 1, 3).reshape(batch, -1, count * output_stride)
        return torch.cat([prefix, tail], dim=-1)

    def encode(self, x):
        x_in = self.preprocess(x)
        if self.left_context_frames:
            x_encoder = self._bounded_transform(x_in, self.encoder, self.left_context_frames,
                                                self.downsample_factor, 1)
        else:
            x_encoder = self.encoder(x_in)
        return self.quantizer.quantize(x_encoder, return_latent=True)

    def forward(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)

        x_quantized, _, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)

        x_out = self.decoder(x_quantized)
        return {
            'rec_pose': x_out,
            'commit_loss': commit_loss,
            'perplexity': perplexity,
        }

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        x = x_d.sum(dim=0).permute(0, 2, 1)

        if self.left_context_frames:
            # Causal_Decoder returns (B,T,D); transform internally uses (B,D,T).
            decode = lambda codes: self.decoder(codes).permute(0, 2, 1)
            x_out = self._bounded_transform(x, decode,
                                           self.left_context_frames // self.downsample_factor,
                                           1, self.downsample_factor).permute(0, 2, 1)
        else:
            x_out = self.decoder(x)
        return x_out
    
    def map2latent(self, x):
        return self.encoder(self.preprocess(x)).permute(0, 2, 1)

    def latent2origin(self, x):
        x = x.permute(0, 2, 1)
        x_quantized, _, commit_loss, perplexity = self.quantizer(x, sample_codebook_temp=0.5)
        x_out = self.decoder(x_quantized)
        return x_out, commit_loss, perplexity
