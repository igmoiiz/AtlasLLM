"""Token-sampling strategies for generation.

    greedy(logits)             full-argmax     (temperature == 0)
    sample_next_token(...)     temperature + optional top-k / top-p

All functions operate on the last ``logits`` row already (or on any trailing
dimension over V); they are batch-friendly: rows are independent and each row
yields one sampled index. top-k keeps the k most likely ids, top-p keeps the
smallest nucleus whose cumulative probability reaches p (always keeping the
single most-likely id), and the surviving logits are renormalized into a
distribution by softmax.
"""

import torch

_NEG_INF = float("-inf")


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Return the argmax index of each row of ``logits`` [..., V]."""
    return logits.argmax(dim=-1)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one index per row of ``logits`` [..., V].

    ``temperature``: scales logits before softmax; 0.0 is a shorthand for
    greedy decoding. Higher temperatures flatten the distribution.
    ``top_k``: keep only the k most-likely ids.
    ``top_p``: nucleus sampling — keep the smallest set whose cumulative
    probability reaches p (at minimum the top id).
    ``rng``: optional torch.Generator for reproducible sampling (pass the
    same generator to replay a sequence); None uses the global RNG.
    """
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if temperature == 0.0:
        return greedy(logits)
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    scores = logits.float() / temperature

    if top_k is not None and top_k < scores.size(-1):
        min_kept = torch.topk(scores, top_k, dim=-1).values[..., -1:]
        scores = torch.where(scores < min_kept, _NEG_INF, scores)

    if top_p is not None and top_p < 1.0:
        # Work in sorted (rank) order: compute the nucleus on the sorted
        # probabilities, censor the sorted scores, then scatter back into
        # vocabulary order. The top-ranked token is always kept.
        sorted_scores, sorted_ids = torch.sort(scores, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep_rank = (cumulative - sorted_probs) <= top_p
        keep_rank[..., 0] = True
        censored = torch.where(keep_rank, sorted_scores, _NEG_INF)
        scores = torch.scatter(scores, -1, sorted_ids, censored)

    probs = torch.softmax(scores, dim=-1)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)
    idx = torch.multinomial(probs, num_samples=1, generator=rng)
    return idx.squeeze(-1)
