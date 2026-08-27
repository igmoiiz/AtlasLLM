"""Token embedding table.

Maps token ids [B, T] -> vectors [B, T, D].
"""

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B, T]
        return self.token_emb(ids)  # [B, T, D]
