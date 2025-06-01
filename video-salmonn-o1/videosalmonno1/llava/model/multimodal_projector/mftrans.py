import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MFTrans(nn.Module):
    def __init__(self):
        super().__init__()

        # For mfproj 16
        self.trans = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=1152, dim_feedforward=2048, nhead=8, batch_first=True),
            num_layers=2
        )
        self.position = nn.Embedding(16, 1152)

    def forward(self, x):
        # F, P, C
        x_lst = torch.split(x, split_size_or_sections=256, dim=0)
        y = []
        for x in x_lst:
            if x.size(0) % 16 != 0:
                pad_zero = torch.zeros((16 - x.size(0) % 16, x.size(1), x.size(2))).to(device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad_zero], dim=0)

            x = x.view(-1, 16, x.size(-2), x.size(-1))
            x = x.transpose(1, 2)
            x = x + self.position(torch.arange(16, device=x.device))
            x = x.reshape(-1, 16, x.size(-1))
            x = self.trans(x)
            x = x[:, 0, :]
            y.append(x)

        y = torch.cat(y, dim=0)
        return y

if __name__ == "__main__":
    model = MFTrans().to(device="cuda", dtype=torch.bfloat16)

    print(sum(p.numel() for p in model.parameters()))
    x = torch.randn(960, 169, 1152, device="cuda", dtype=torch.bfloat16)
    y = model(x)
    print(y.shape)