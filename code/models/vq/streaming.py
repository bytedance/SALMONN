"""Causal RVQ decoding with bounded per-request history and no right context."""
import torch


class RVQDecoderStream:
    def __init__(self, model):
        if not model.causal or not model.left_context_frames:
            raise ValueError("Strict streaming requires a causal RVQ with a finite context cap")
        self.model = model
        self.window = model.left_context_frames // model.downsample_factor
        self.history = None

    def parameters(self):
        return self.model.parameters()

    @property
    def cached_steps(self):
        return 0 if self.history is None else self.history.shape[-1]

    @torch.no_grad()
    def forward_decoder(self, indices):
        assert not self.model.training and indices.ndim == 3 and indices.shape[1] > 0
        new = self.model.quantizer.get_codes_from_indices(indices).sum(dim=0).permute(0, 2, 1)
        old_steps = self.cached_steps
        joined = new if self.history is None else torch.cat([self.history, new], dim=-1)
        decode = lambda codes: self.model.decoder(codes).permute(0, 2, 1)
        decoded = self.model._bounded_transform(
            joined, decode, self.window, 1, self.model.downsample_factor)
        # Clone so the cache never retains the backing storage of a longer block.
        self.history = (joined[..., -(self.window - 1):] if self.window > 1
                        else joined[..., :0]).detach().clone()
        return decoded[..., old_steps * self.model.downsample_factor:].permute(0, 2, 1)


class FactorizedFlameDecoderStream:
    def __init__(self, model):
        self.model = model
        self.nonmouth = RVQDecoderStream(model.nonmouth_vq)
        self.mouth = RVQDecoderStream(model.mouth_vq)

    def parameters(self):
        return self.model.parameters()

    @torch.no_grad()
    def forward_decoder(self, indices):
        assert indices.ndim == 3 and indices.shape[-1] == self.model.num_quantizers
        split = self.model.nonmouth_num_quantizers
        return self.model.merge_features(self.nonmouth.forward_decoder(indices[..., :split]),
                                         self.mouth.forward_decoder(indices[..., split:]))


def make_decoder_stream(model):
    if hasattr(model, "nonmouth_vq"):
        return FactorizedFlameDecoderStream(model)
    return RVQDecoderStream(model)


def supports_strict_streaming(model):
    branches = (model.nonmouth_vq, model.mouth_vq) if hasattr(model, "nonmouth_vq") else (model,)
    return all(getattr(branch, "causal", False) and getattr(branch, "left_context_frames", 0)
               for branch in branches)
