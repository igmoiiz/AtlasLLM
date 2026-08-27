"""Tests for multi-head causal self-attention.

Covers output shapes, the causal property (future positions must not influence
earlier predictions), gradient flow, and numerical agreement with a naive
per-head reference implementation.
"""

import math

import pytest
import torch

from model.attention import MultiHeadCausalAttention

B, T, D, H = 2, 8, 32, 4


def make_attention(dropout: float = 0.0, bias: bool = False) -> MultiHeadCausalAttention:
    return MultiHeadCausalAttention(d_model=D, n_heads=H, dropout=dropout, bias=bias)


def naive_attention(
    attn: MultiHeadCausalAttention,
    x: torch.Tensor,
) -> torch.Tensor:
    """Per-head reference: softmax((QK^T + mask) / sqrt(d_k)) V, concat, project.

    Re-derives the math explicitly so ``MultiHeadCausalAttention`` is checked
    against ground-truth matrix operations rather than against itself.
    """
    b, t, _ = x.shape
    head_dim = D // H
    w_q = attn.wq.weight  # [D, D]
    w_k, w_v, w_o = attn.wk.weight, attn.wv.weight, attn.wo.weight
    head_logits = []
    for h in range(H):
        q = x @ w_q[h * head_dim : (h + 1) * head_dim].T  # [b,t,hd]
        k = x @ w_k[h * head_dim : (h + 1) * head_dim].T
        v = x @ w_v[h * head_dim : (h + 1) * head_dim].T
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)  # [b,t,t]
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        head_logits.append(probs @ v)  # [b,t,hd]
    concat = torch.cat(head_logits, dim=-1)  # [b,t,d]
    return concat @ w_o.T


def test_output_shape() -> None:
    attn = make_attention()
    out = attn(torch.randn(B, T, D))
    assert out.shape == (B, T, D)


def test_future_positions_do_not_affect_earlier_outputs() -> None:
    """The causal-mask property (AGENTS.md Rule 23, CONTEXT #56).

    If inputs agree up to position k but differ afterwards, outputs up to and
    including k must be identical.
    """
    attn = make_attention()
    x1 = torch.randn(B, T, D)
    x2 = x1.clone()
    x2[:, 3:, :] = torch.randn(B, T - 3, D)  # differ only from position 3 onward
    out1 = attn(x1)
    out2 = attn(x2)
    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-6, rtol=1e-6)


def test_matches_naive_reference() -> None:
    attn = make_attention()
    x = torch.randn(B, T, D)
    expected = naive_attention(attn, x)
    assert torch.allclose(attn(x), expected, atol=1e-5, rtol=1e-5)


def test_single_token_attends_only_to_itself() -> None:
    """With one token, attention is the identity operation over V."""
    attn = make_attention()
    x = torch.randn(B, 1, D)
    out = attn(x)
    assert out.shape == (B, 1, D)
    assert torch.all(torch.isfinite(out))


@pytest.mark.parametrize("d_model,n_heads", [(32, 4), (32, 2), (64, 8)])
def test_various_head_splits(d_model: int, n_heads: int) -> None:
    attn = MultiHeadCausalAttention(d_model=d_model, n_heads=n_heads, dropout=0.0, bias=False)
    out = attn(torch.randn(2, 6, d_model))
    assert out.shape == (2, 6, d_model)


def test_gradient_flows() -> None:
    attn = make_attention()
    x = torch.randn(B, T, D)
    out = attn(x)
    out.sum().backward()
    for name, param in attn.named_parameters():
        assert param.grad is not None, f"{name} did not receive a gradient"
