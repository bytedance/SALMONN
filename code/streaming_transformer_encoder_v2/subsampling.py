import torch
import torch.nn as nn

class Conv2dSubsampling4(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Convert an input of shape (N, T, idim) to an output
    with shape (N, T', odim), where
    T' = ((T-1)//2 - 1)//2, which approximates T' == T//4

    It is based on
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py  # noqa
    """

    def __init__(self, idim: int, odim: int, intermediate_dim: int = 256) -> None:
        """
        Args:
          idim:
            Input dim. The input shape is (N, T, idim).
            Caution: It requires: T >=7, idim >=7
          odim:
            Output dim. The output shape is (N, ((T-1)//2 - 1)//2, odim)
        """
        assert idim >= 7
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=intermediate_dim, kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv2d(in_channels=intermediate_dim, out_channels=intermediate_dim, kernel_size=3, stride=2),
            nn.GELU(),
        )
        self.out = nn.Linear(intermediate_dim * (((idim - 1) // 2 - 1) // 2), odim)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).

        Returns:
          Return a tensor of shape (N, ((T-1)//2 - 1)//2, odim)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        x = self.conv(x)
        # Now x is of shape (N, odim, ((T-1)//2 - 1)//2, ((idim-1)//2 - 1)//2)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        # Now x is of shape (N, ((T-1)//2 - 1))//2, odim)
        x_lens = ((x_lens-1) // 2 - 1) // 2
        return x, x_lens

class Conv2dSubsampling(nn.Module):
    def __init__(self, 
        idim: int = 128,
        intermediate_dim: int = 128,
        odim: int = 768,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=intermediate_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels=intermediate_dim, out_channels=intermediate_dim, kernel_size=3, stride=2),
            nn.GELU(),
        )
        self.out = nn.Linear(intermediate_dim * ((idim - 1) // 2), odim)
        
    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        """Subsample x.

        Args:
          x:
            Its shape is (N, T, idim).
          x_lens:
            Length of the input feature (N,)

        Returns:
          Return a tensor of shape (N, ((T-1)//2 , odim)
        """
        # On entry, x is (N, T, idim)
        x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
        x = self.conv(x)
        # Now x is of shape (N, odim, T, odim)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x_lens = (x_lens - 1) // 2
        # Now x is of shape (N, (T-1)//2 , odim)
        return x, x_lens
