import torch
from torch import nn
import time
from tqdm import tqdm

class NeuralIV(nn.Module):
    def __init__(self, input_channels, out_channels, params, use_batch_norm=True):
        super().__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.final_feature_dim = out_channels * (input_channels - 1)
        self.params = params

        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(in_features=out_channels, out_features=out_channels))
        self.linear_layers.append(nn.ReLU())
        self.linear_layers.append(nn.Linear(in_features=out_channels, out_features=out_channels))

        self.use_batch_norm = use_batch_norm
        self.conv_layers = nn.ModuleList()
        for i, (kernel_size, stride) in enumerate(params):
            if i == 0:
                self.conv_layers.append(nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2))
            else:
                self.conv_layers.append(nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2))
            if self.use_batch_norm:
                self.conv_layers.append(nn.BatchNorm1d(out_channels))

    def forward(self, raw_wav):
        B, C, T = raw_wav.shape
        
        x = raw_wav.view(-1, 1, T)
        for layer in self.conv_layers:
            x = layer(x)

        x = x.transpose(1, 2) # B * T * C
        x = x.view(B, C, -1, self.out_channels)

        main_spec = x[:, 0:1, :, :]
        other_spec = x[:, 1:, :, :]

        I = main_spec * other_spec
        for layer in self.linear_layers:
            I = layer(I)

        I = I.transpose(1, 2)
        y = I.reshape(B, -1, self.out_channels * (self.input_channels - 1))

        return y

if __name__ == "__main__":
    model = NeuralIV(
        input_channels=4,
        out_channels=401,
        params=[(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)],
        use_batch_norm=True,
    ).cuda()

    signal = torch.randn([4, 4, 160000]).cuda()
    output = model(signal)

    # ln_map = nn.Linear(in_features=768, out_features=361).cuda()
    # res = ln_map(output)
    # global_avg_pool = nn.AdaptiveAvgPool2d((1, None))
    # res = global_avg_pool(res).squeeze()
    # labels = nn.functional.softmax(res, dim=1)

    import pdb; pdb.set_trace()
