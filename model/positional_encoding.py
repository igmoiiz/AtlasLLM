"""Learned positional embeddings.

Each position index gets its own trainable vector [T, D], added to the token
embeddings. (Sinusoidal / RoPE are reserved for later experiments; only one
positional system is active per configuration.)
"""

import torch
from torch import nn


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, context_length: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Embedding(context_length, d_model)

    def forward(self, seq_len: int, device: torch.device, offset: int = 0) -> torch.Tensor:
        # positions: [T] at absolute indices [offset, offset+seq_len)
        # returns [T, D] broadcastable over the batch dim; offset lets the KV
        # cache pick up embeddings exactly where a prefill left off.
        positions = torch.arange(offset, offset + seq_len, device=device)
        return self.pos_emb(positions)
