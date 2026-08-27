"""Learning-rate schedule: linear warmup followed by cosine decay to zero.

    Step 0          -> warmup_steps:  lr = max_lr * step / warmup_steps
    warmup_steps    -> max_steps:     lr = max_lr * 0.5 * (1 + cos(pi * progress))

``progress`` goes from 0 to 1 across the decay phase. ``max_lr`` is the
optimizer's configured learning rate; the schedule only returns a multiplier.
"""

import math

import torch


def lr_prefactor(step: int, warmup_steps: int, max_steps: int) -> float:
    """LR multiplier at integer ``step`` (0-indexed). Never exceeds 1."""
    if step < 0:
        return 0.0
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def build_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
    """Wrap ``optimizer`` so its LR follows :func:`lr_prefactor`."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_prefactor(step, warmup_steps, max_steps)
    )
