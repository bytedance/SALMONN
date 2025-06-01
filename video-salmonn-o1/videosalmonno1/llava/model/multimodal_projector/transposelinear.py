import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTemporalProj(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(729, 169)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(config.multi_frame_num * config.mm_hidden_size, config.multi_frame_num * config.hidden_size)
        self.linear3 = nn.Linear(config.multi_frame_num * config.hidden_size, config.hidden_size)

    def forward(self, x):
        x = self.linear2(x)
        x = self.gelu(x)
        x = self.linear3(x)
        x = x.transpose(1, 2)
        x = self.linear1(x)
        # x = self.gelu(x)
        x = x.transpose(1, 2)
        return x

if __name__ == "__main__":
    data = {'multi_frame_num': 16, "mm_hidden_size": 1152, "hidden_size": 3584}
    from types import SimpleNamespace
    config = SimpleNamespace(**data)

    model = SpatialTemporalProj(config).to(device="cuda", dtype=torch.bfloat16)

    print(sum(p.numel() for p in model.parameters()))
    x = torch.randn(10, 729, 1152 * 16, device="cuda", dtype=torch.bfloat16)
    y = model(x)
    print(y.shape)
