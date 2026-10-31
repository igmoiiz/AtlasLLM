"""Learning-rate schedules: linear warmup plus cosine decay or WSD.

    cosine (default):  warmup -> cosine decay to zero over the whole run

        Step 0            -> warmup_steps:  lr = max_lr * step / warmup_steps
        warmup_steps      -> max_steps:     lr = max_lr * 0.5 * (1 + cos(pi * p))

    wsd (warmup-stable-decay):  warmup -> flat -> short cosine tail

        Step 0            -> warmup_steps:  lr = max_lr * step / warmup_steps
        warmup_steps      -> decay_start:   lr = max_lr          (stable)
        decay_start       -> max_steps:     lr = max_lr * 0.5 * (1 + cos(pi * p))

``p`` runs 0 -> 1 across the decay phase, so both schedules end at zero.
``decay_start = max_steps * (1 - decay_frac)`` for WSD.

WSD exists because the long plateau keeps the LR high for almost the whole
run: a checkpoint taken during the stable phase can be extended (more tokens,
bigger data) and still gets a proper anneal at the end, instead of inheriting
an LR that cosine already drove to zero. The short tail is what buys the final
loss, so ``decay_frac`` must stay small (0.1 = last 10% of steps).

``max_lr`` is the optimizer's configured learning rate; the schedule only
returns a multiplier.
"""

import math

import torch

SCHEDULES = ("cosine", "wsd")


def lr_prefactor(
    step: int,
    warmup_steps: int,
    max_steps: int,
    schedule: str = "cosine",
    decay_frac: float = 0.1,
) -> float:
    """LR multiplier at integer ``step`` (0-indexed). Never exceeds 1."""
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}")
    if not 0.0 < decay_frac <= 1.0:
        raise ValueError(f"decay_frac must be in (0, 1], got {decay_frac}")
    if step < 0:
        return 0.0
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps

    if schedule == "wsd":
        decay_start = max(warmup_steps, max_steps * (1.0 - decay_frac))
        if step <= decay_start:
            return 1.0
        progress = (step - decay_start) / max(1e-9, max_steps - decay_start)
    else:
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def build_lr_scheduler(
    optimizer,
    warmup_steps: int,
    max_steps: int,
    schedule: str = "cosine",
    decay_frac: float = 0.1,
):
    """Wrap ``optimizer`` so its LR follows :func:`lr_prefactor`."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_prefactor(step, warmup_steps, max_steps, schedule, decay_frac),
    )
