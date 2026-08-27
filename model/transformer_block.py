"""Pre-normalization transformer block.

    x = x + attention(LayerNorm(x))
    x = x + ffn(LayerNorm(x))

Normalizing before (not after) the residual branches gives more stable
gradients and matches modern transformer practice.
"""

import torch
from torch import nn

from model.attention import MultiHeadCausalAttention
from model.config import ModelConfig
from model.feed_forward import FeedForward
from model.normalization import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.d_model)
        self.attn = MultiHeadCausalAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            bias=config.bias,
        )
        self.ln2 = LayerNorm(config.d_model)
        self.ffn = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            bias=config.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return self._forward_cached(x, None, None)[0]

    def forward_with_cache(
        self, x: torch.Tensor, past_key: torch.Tensor | None, past_value: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Block pass with a cached prefix; returns (out, new_k, new_v)."""
        return self._forward_cached(x, past_key, past_value)

    def _forward_cached(
        self, x: torch.Tensor, past_key: torch.Tensor | None, past_value: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, T, D]
        h = self.ln1(x)
        attn_out, k, v = self.attn.forward_with_cache(h, past_key, past_value)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, k, v
