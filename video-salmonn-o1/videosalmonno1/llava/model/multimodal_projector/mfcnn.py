import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MFCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For mfproj 16
        self.cnn3d = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=2, kernel_size=(2, 7, 7), stride=(2, 1, 1), padding=(0, 3, 3)),
            nn.Conv3d(in_channels=2, out_channels=4, kernel_size=(2, 5, 7), stride=(2, 2, 1), padding=(0, 2, 3)),
            nn.Conv3d(in_channels=4, out_channels=2, kernel_size=(2, 5, 5), stride=(2, 2, 1), padding=(0, 2, 2)),
            nn.Conv3d(in_channels=2, out_channels=1, kernel_size=(2, 5, 5), stride=(2, 1, 1), padding=(0, 2, 2)),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.cnn3d:
            if isinstance(layer, nn.Conv3d):
                nn.init.kaiming_uniform_(layer.weight, a=0.1)  # Kaiming initialization
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        # F, P, C
        x_lst = torch.split(x, split_size_or_sections=256, dim=0)
        y = []
        for x in x_lst:
            if x.size(0) % 16 != 0:
                pad_zero = torch.zeros((16 - x.size(0) % 16, x.size(1), x.size(2))).to(device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad_zero], dim=0)

            x = x.unsqueeze(0)
            x = self.cnn3d(x)
            x = x.squeeze(0)
            x = x.view(-1, x.size(-1))
            y.append(x)

        y = torch.cat(y, dim=0)
        return y

if __name__ == "__main__":
    model = MFCNN().to(device="cuda", dtype=torch.bfloat16)

    print(sum(p.numel() for p in model.parameters()))
    x = torch.randn(960, 729, 1152, device="cuda", dtype=torch.bfloat16)
    y = model(x)
    print(y.shape)