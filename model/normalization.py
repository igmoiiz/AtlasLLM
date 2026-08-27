"""LayerNorm implemented with explicit tensor math.

    y = (x - mean) / sqrt(var + eps) * gamma + beta

``var`` is the population variance (divide by N), matching ``nn.LayerNorm``.
"""

import torch
from torch import nn


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension (learnable gamma/beta)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta
