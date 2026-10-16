import torch.nn as nn
from models.vq.resnet import Resnet1D, Causal_Resnet1D
import torch

class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 1)
    
    

## replace all `nn.ReplicationPad1d((2,0))` to `nn.ZeroPad1d((1,1))`
## replace all `nn.ReplicationPad1d((filter_t-1,0))` to `nn.ZeroPad1d((pad_t,pad_t))`

class Causal_Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 end_aligned=False):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.ReplicationPad1d((2,0)))
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                # End-align each latent with its completed stride-sized input block.
                nn.ReplicationPad1d((filter_t-stride_t if end_aligned else filter_t-1,0)),
                nn.Conv1d(input_dim, width, filter_t, stride_t),
                Causal_Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.ReplicationPad1d((2,0)))
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)



class Causal_Decoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        blocks.append(nn.ReplicationPad1d((2,0)))
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Causal_Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ReplicationPad1d((2,0)),
                nn.Conv1d(width, out_dim, 3, 1)
            )
            blocks.append(block)
        blocks.append(nn.ReplicationPad1d((2,0)))
        blocks.append(nn.Conv1d(width, width, 3, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.ReplicationPad1d((2,0)))
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 1)
