import random
from random import randrange

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from models.vq.quantizer import QuantizeEMAReset

from einops import repeat

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        num_quantizers,
        shared_codebook=False,
        quantize_dropout_prob=0.5,
        quantize_dropout_cutoff_index=0,
        **kwargs
    ):
        super().__init__()

        self.num_quantizers = num_quantizers

        if shared_codebook:
            layer = QuantizeEMAReset(**kwargs)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])

        assert quantize_dropout_cutoff_index >= 0 and quantize_dropout_prob >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_prob = quantize_dropout_prob

    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks # 'q c d'
    
    def get_codes_from_indices(self, indices): #indices shape 'b n q' # dequantize

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # get ready for gathering

        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b = batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d = codebooks.shape[-1])

        # take care of quantizer dropout

        mask = gather_indices == -1.
        gather_indices = gather_indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        all_codes = codebooks.gather(2, gather_indices) # gather all codes

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(mask, 0.)

        return all_codes # 'q b n d'

    def get_codebook_entry(self, indices): #indices shape 'b n q'
        all_codes = self.get_codes_from_indices(indices) #'q b n d'
        latent = torch.sum(all_codes, dim=0) #'b n d'
        latent = latent.permute(0, 2, 1)
        return latent

    def forward(self, x, return_all_codes = False, sample_codebook_temp = None, force_dropout_index=-1):
        num_quant, device = self.num_quantizers, x.device

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        all_perplexity = []

        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob

        start_drop_quantize_index = num_quant
        # To ensure the first-k layers learn things as much as possible, we randomly dropout the last q - k layers
        if should_quantize_dropout:
            start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant) # keep quant layers <= quantize_dropout_cutoff_index, TODO vary in batch
        if dist.is_available() and dist.is_initialized():
            # All ranks must execute the same number of distributed EMA updates.
            dropout_state = torch.tensor(
                [int(should_quantize_dropout), int(start_drop_quantize_index)],
                dtype=torch.long,
                device=device,
            )
            dist.broadcast(dropout_state, src=0)
            should_quantize_dropout = bool(dropout_state[0].item())
            start_drop_quantize_index = int(dropout_state[1].item())
        if should_quantize_dropout:
            null_indices_shape = [x.shape[0], x.shape[-1]] # 'b*n'
            null_indices = torch.full(null_indices_shape, -1., device = device, dtype = torch.long)

        if force_dropout_index >= 0:
            should_quantize_dropout = True
            start_drop_quantize_index = force_dropout_index
            null_indices_shape = [x.shape[0], x.shape[-1]]  # 'b*n'
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)

        # go through the layers

        for quantizer_index, layer in enumerate(self.layers):

            if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
                all_indices.append(null_indices)
                continue

            quantized, *rest = layer(residual, return_idx=True, temperature=sample_codebook_temp) #single quantizer

            residual -= quantized.detach()
            quantized_out += quantized

            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            all_losses.append(loss)
            all_perplexity.append(perplexity)

        # stack all losses and indices
        all_indices = torch.stack(all_indices, dim=-1)
        all_losses = sum(all_losses)/len(all_losses)
        all_perplexity = sum(all_perplexity)/len(all_perplexity)

        ret = (quantized_out, all_indices, all_losses, all_perplexity)

        if return_all_codes:
            # whether to return all codes from all codebooks across layers
            all_codes = self.get_codes_from_indices(all_indices)

            # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)
            ret = (*ret, all_codes)

        return ret
    
    def quantize(self, x, return_latent=False):
        all_indices = []
        residual = x
        all_codes = []
        for layer in self.layers:
            quantized, *rest = layer(residual, return_idx=True) #single quantizer

            residual = residual - quantized.detach()

            embed_indices, _, _ = rest
            all_indices.append(embed_indices)
            all_codes.append(quantized)

        code_idx = torch.stack(all_indices, dim=-1)
        all_codes = torch.stack(all_codes, dim=0)
        if return_latent:
            return code_idx, all_codes
        return code_idx
