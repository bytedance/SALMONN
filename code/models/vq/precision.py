"""RVQ encoder precision shared by training and inference; decoders keep their dtype."""
import json
from pathlib import Path
from types import MethodType
import torch


def resolve_encoding_precision(args, key):
    value = getattr(args, key, None)
    if value is None:
        model_path = Path(getattr(args, 'model_path', ''))
        for path in (model_path / 'config.json', model_path.parent / 'config.json'):
            if not path.is_file():
                continue
            config = json.loads(path.read_text())
            value = config.get(key, config.get('data_args', {}).get(key))
            if value is None:
                policy_key = 'encoder_dtype' if key == 'rvq_encoder_dtype' else 'quantizer_distance_dtype'
                value = config.get('rvq_cache_policy', {}).get(policy_key)
            if value is not None:
                break
    value = value or 'float32'
    if value not in ('float32', 'float64'):
        raise ValueError(f'{key} must be float32 or float64, got {value}')
    return value


@torch.no_grad()
def _quantize_float64(self, x, sample_codebook_temp=0.):
    assert not self.training and sample_codebook_temp == 0.
    book = self.codebook.double()
    norm = (book * book).sum(-1).unsqueeze(0)
    result = []
    for begin in range(0, len(x), 2048):
        value = x[begin:begin + 2048].double()
        result.append(((value * value).sum(-1, keepdim=True) - 2 * (value @ book.T) + norm).argmin(-1))
    return torch.cat(result)


def _preprocess(self, x):
    return x.permute(0, 2, 1).to(next(self.encoder.parameters()).dtype)


def _split_float64(self, motion):
    if motion.ndim != 3 or motion.shape[-1] != 108:
        raise ValueError(f'expected normalized FLAME motion (B,T,108), got {tuple(motion.shape)}')
    motion = motion.double()
    basis = self.mouth_basis.double()
    mouth = motion[..., :100] @ basis
    residual = motion[..., :100] - mouth @ basis.T
    return torch.cat([residual, motion[..., 104:108]], -1), torch.cat([mouth, motion[..., 103:104]], -1)


def configure_encoding_precision(model, encoder_dtype='float32', distance_dtype='float32'):
    if encoder_dtype not in ('float32', 'float64') or distance_dtype not in ('float32', 'float64'):
        raise ValueError('RVQ encoding precision must be float32 or float64')
    if hasattr(model, 'nonmouth_vq'):
        if not hasattr(model, '_original_split_features'):
            model._original_split_features = model.split_features
        model.split_features = (MethodType(_split_float64, model) if encoder_dtype == 'float64'
                                else model._original_split_features)
        branches = (model.nonmouth_vq, model.mouth_vq)
    else:
        branches = (model,)
    for branch in branches:
        branch.encoder.to(dtype=getattr(torch, encoder_dtype))
        branch.preprocess = MethodType(_preprocess, branch)
        for layer in branch.quantizer.layers:
            if not hasattr(layer, '_original_quantize'):
                layer._original_quantize = layer.quantize
            layer.quantize = (MethodType(_quantize_float64, layer) if distance_dtype == 'float64'
                              else layer._original_quantize)
    return model
