"""Held-out perplexity evaluation.

    loss = mean cross-entropy over token positions
    ppl  = exp(loss)

Reuses the training loss (:func:`training.loss.lm_cross_entropy`) so the
numbers agree with the curves in ``metrics.jsonl``. Shapes: input_ids
[B, T], logits [B, T, V]; targets are the shifted next-token stream.
"""

import math

import torch

from training.loss import lm_cross_entropy


def evaluate_loss(model: torch.nn.Module, loader, device: torch.device, max_batches: int | None = None) -> float:
    """Mean per-token cross-entropy over a data loader (no gradients).

    ``loader`` yields ``{"input_ids": [B, T] long, "targets": [B, T] long}``
    (e.g. a ``DataLoader`` over :class:`data_pipeline.dataset.TextDataset`).
    When ``max_batches`` is set only that many batches are averaged.
    """
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            total += lm_cross_entropy(model(input_ids), targets).item()
            count += 1
    model.train()
    if count == 0:
        raise ValueError("loader produced no batches")
    return total / count


def perplexity(loss: float) -> float:
    """Standard LM metric: perplexity = exp(cross-entropy loss)."""
    return math.exp(loss)
