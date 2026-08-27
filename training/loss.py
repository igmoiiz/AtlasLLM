"""Autoregressive language-modeling loss.

    loss = cross_entropy(reshape(logits), targets)

``ignore_index`` excludes positions from the expectation (house rule: padding
positions must never contribute to the loss; our fixed-window batches have no
padding, but the option exists for variable-length batches).
"""

import torch
from torch import nn


def lm_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy over flattened token positions.

    Accepts any logits shaped [..., V] (typically [B, T, V]) against targets
    [..., ]; leading dimensions are flattened into a single batch.
    """
    v = logits.shape[-1]
    flat_logits = logits.reshape(-1, v)
    flat_targets = targets.reshape(-1)
    return nn.functional.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index)
