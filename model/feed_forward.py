"""Feed-forward network.

    FFN(x) = dropout(GELU(xW1 + b1)W2 + b2)
"""

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, bias: bool):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, T, D]
        return self.dropout(self.fc2(self.act(self.fc1(x))))
