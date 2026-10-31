"""Token-sampling strategies for generation.

    greedy(logits)             full-argmax     (temperature == 0)
    sample_next_token(...)     temperature + optional top-k / top-p / n-gram ban

All functions operate on the last ``logits`` row already (or on any trailing
dimension over V); they are batch-friendly: rows are independent and each row
yields one sampled index. top-k keeps the k most likely ids, top-p keeps the
smallest nucleus whose cumulative probability reaches p (always keeping the
single most-likely id), and the surviving logits are renormalized into a
distribution by softmax.

``no_repeat_ngram_size > 0`` bans any id that would complete an n-gram already
present in ``recent_ids`` (the prompt plus everything generated so far). The
ban is applied before top-k/top-p, and it can never empty the distribution: if
every id is banned the unmodified scores are kept. Greedy decoding
(``temperature == 0``) bypasses sampling entirely and therefore ignores the ban
by design; it is a diagnostic mode, not a chat mode.
"""

import torch

_NEG_INF = float("-inf")


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Return the argmax index of each row of ``logits`` [..., V]."""
    return logits.argmax(dim=-1)


def _history_rows(recent_ids, batch_size: int) -> list[list[int]]:
    """Normalize generation history to one id list per logits row.

    Accepts a flat list/tensor (shared by every row) or one list per row.
    """
    if recent_ids is None:
        return [[] for _ in range(batch_size)]
    if isinstance(recent_ids, torch.Tensor):
        rows = recent_ids.tolist()
    else:
        rows = list(recent_ids)
    if rows and isinstance(rows[0], int):
        rows = [rows]
    if len(rows) == 1:
        rows = rows * batch_size
    if len(rows) != batch_size:
        raise ValueError(
            f"recent_ids has {len(rows)} rows but logits has {batch_size}; "
            "pass one history per row or a single shared history"
        )
    return rows


def _repeat_ban(recent_rows: list[list[int]], n: int, vocab_size: int, device) -> torch.Tensor:
    """Boolean [B, V] mask of ids that would complete an already-seen n-gram."""
    ban = torch.zeros((len(recent_rows), vocab_size), dtype=torch.bool, device=device)
    for row, ids in enumerate(recent_rows):
        if len(ids) < n:
            continue
        # The n-1 ids immediately before the next position.
        prefix = tuple(ids[len(ids) - (n - 1):]) if n > 1 else ()
        for start in range(len(ids) - n + 1):
            if tuple(ids[start:start + n - 1]) == prefix:
                ban[row, ids[start + n - 1]] = True
    return ban


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    rng: torch.Generator | None = None,
    no_repeat_ngram_size: int = 0,
    recent_ids=None,
) -> torch.Tensor:
    """Sample one index per row of ``logits`` [..., V].

    ``temperature``: scales logits before softmax; 0.0 is a shorthand for
    greedy decoding. Higher temperatures flatten the distribution.
    ``top_k``: keep only the k most-likely ids.
    ``top_p``: nucleus sampling — keep the smallest set whose cumulative
    probability reaches p (at minimum the top id).
    ``rng``: optional torch.Generator for reproducible sampling (pass the
    same generator to replay a sequence); None uses the global RNG.
    ``no_repeat_ngram_size``: ban ids that would repeat an n-gram from
    ``recent_ids`` (0 disables the ban).
    ``recent_ids``: prompt + generated ids, flat or one row per batch entry.
    """
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if top_p is not None and not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if no_repeat_ngram_size < 0:
        raise ValueError(f"no_repeat_ngram_size must be >= 0, got {no_repeat_ngram_size}")
    if no_repeat_ngram_size > 0 and recent_ids is None:
        raise ValueError("recent_ids is required when no_repeat_ngram_size > 0")
    if temperature == 0.0:
        return greedy(logits)

    scores = logits.float() / temperature

    if no_repeat_ngram_size > 0:
        rows = _history_rows(recent_ids, scores.size(0))
        ban = _repeat_ban(rows, no_repeat_ngram_size, scores.size(-1), scores.device)
        if ban.any():
            banned = scores.masked_fill(ban, _NEG_INF)
            # The ban must never empty the distribution: when every id is
            # blocked, keep the unmodified scores so the top id stays legal.
            empty = torch.isneginf(banned).all(dim=-1, keepdim=True)
            scores = torch.where(empty, scores, banned)

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
