"""Multi-head causal self-attention.

    Q = XWq      K = XWk      V = XWv
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + mask) V

The causal mask uses ``-inf`` in the upper triangle so position i can only
attend to positions 0..i. The mask is built explicitly in the forward pass
(there is no mask-free optimization path); heads run in parallel over the
[H] dimension and are concatenated and projected out.
"""

import math

import torch
from torch import nn


class MultiHeadCausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, bias: bool):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must divide d_model ({d_model})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # d_k = d_v = head_dim

        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _, _ = self._attend(x, None, None)
        return out  # [B, T, D]

    def forward_with_cache(
        self, x: torch.Tensor, past_key: torch.Tensor | None, past_value: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Attention with a cached prefix; returns (out, new_k, new_v)."""
        return self._attend(x, past_key, past_value)

    def _attend(
        self, x: torch.Tensor, past_key: torch.Tensor | None, past_value: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, D]; past_key/past_value: [B, H, P, d_k] or None
        b, t, _ = x.shape
        h, hd = self.n_heads, self.head_dim

        q = self.wq(x).view(b, t, h, hd).transpose(1, 2)  # [B, H, T, d_k]
        k = self.wk(x).view(b, t, h, hd).transpose(1, 2)  # [B, H, T, d_k]
        v = self.wv(x).view(b, t, h, hd).transpose(1, 2)  # [B, H, T, d_v]

        if past_key is not None:
            k = torch.cat((past_key, k), dim=2)  # [B, H, P + T, d_k]
            v = torch.cat((past_value, v), dim=2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)  # [B, H, T, P + T]

        # Causal mask over the full historical span; the new queries occupy
        # rows [P, P+t) of the (P+t) x (P+t) causal matrix (row i attends only
        # to columns 0..i). Slicing the last t rows keeps exactly the visible
        # columns; with past_key=None (P=0) this reduces to the plain t-by-t
        # causal mask.
        p = past_key.size(2) if past_key is not None else 0
        full_len = p + t
        mask = torch.triu(
            torch.ones(full_len, full_len, dtype=torch.bool, device=x.device), diagonal=1
        )[p : p + t, :]
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)  # [B, H, T, P + T]
        probs = self.attn_dropout(probs)

        out = probs @ v  # [B, H, T, d_v]
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.resid_dropout(self.wo(out)), k, v  # [B, T, D]
