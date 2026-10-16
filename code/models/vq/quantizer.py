import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange


#Borrow from vector_quantize_pytorch

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    dim = -1,
    training = True
):

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)

    return ind

class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super(QuantizeEMAReset, self).__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu  ##TO_DO
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, requires_grad=False))

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # EMA accumulators were historically not checkpointed.  Preserve a
        # loaded codebook instead of reinitializing it on the first forward.
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                codebook = self._tile(x)[:self.nb_code]
            else:
                codebook = torch.empty(
                    self.nb_code, self.code_dim, dtype=x.dtype, device=x.device
                )
            dist.broadcast(codebook, src=0)
            self.codebook = codebook
        else:
            out = self._tile(x)
            self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def quantize(self, x, sample_codebook_temp=0.):
        # N X C -> C X N
        k_w = self.codebook.t()
        # x: NT X C
        # NT X N
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - \
                   2 * torch.matmul(x, k_w) + \
                   torch.sum(k_w ** 2, dim=0, keepdim=True)  # (N * L, b)


        code_idx = gumbel_sample(-distance, dim = -1, temperature = sample_codebook_temp, stochastic=True, training = self.training)

        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x
    
    def get_codebook_entry(self, indices):
        return self.dequantize(indices).permute(0, 2, 1)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device) # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x) # nb_code, c
        code_count = code_onehot.sum(dim=-1) # nb_code

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(code_sum)
            dist.all_reduce(code_count)

        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                code_rand = self._tile(x)[:self.nb_code]
            else:
                code_rand = torch.empty(
                    self.nb_code, self.code_dim, dtype=x.dtype, device=x.device
                )
            dist.broadcast(code_rand, src=0)
        else:
            out = self._tile(x)
            code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1-usage) * code_rand


        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = rearrange(x, 'n c t -> (n t) c')
        return x

    def forward(self, x, return_idx=False, temperature=0.):
        N, width, T = x.shape

        x = self.preprocess(x)
        if self.training and not self.init:
            self.init_codebook(x)

        code_idx = self.quantize(x, temperature)
        x_d = self.dequantize(code_idx)

        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        commit_loss = F.mse_loss(x, x_d.detach()) # It's right. the t2m-gpt paper is wrong on embed loss and commitment loss.

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        code_idx = code_idx.view(N, T).contiguous()
        if return_idx:
            return x_d, code_idx, commit_loss, perplexity
        return x_d, commit_loss, perplexity
    
class QuantizeEMA(QuantizeEMAReset):
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device) # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x) # nb_code, c
        code_count = code_onehot.sum(dim=-1) # nb_code

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1-usage) * self.codebook

        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity
