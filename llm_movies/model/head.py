import torch
from torch import Tensor, nn
from torch.nn import functional as F


class QuantileHead(nn.Module):
    """Hidden state -> p10/p50/p90 for each target, non-crossing by construction."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 6))

    def forward(self, h: Tensor) -> Tensor:
        m, a, b = self.net(h).view(-1, 2, 3).unbind(-1)
        return torch.stack([m - F.softplus(a), m, m + F.softplus(b)], dim=-1)
