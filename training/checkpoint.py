"""Checkpoint persistence: save/restore full training state in one file.

A checkpoint bundles model, optimizer, and scheduler state with the exact
training step, the run configuration, and current metrics. Save is atomic
(write to ``.tmp`` then rename) so a crash cannot leave a truncated file.
"""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Checkpoint:
    step: int
    config: dict
    metrics: dict
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    step: int,
    config: dict,
    metrics: dict,
) -> Path:
    """Persist ``step``, ``config``, ``metrics``, and all module states."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "step": step,
            "config": config,
            "metrics": metrics,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        tmp,
    )
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, *, model, optimizer, scheduler) -> Checkpoint:
    """Load checkpoint into ``model``/``optimizer``/``scheduler`` in place.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    return Checkpoint(
        step=payload.get("step", 0),
        config=payload.get("config", {}),
        metrics=payload.get("metrics", {}),
        model_state=payload["model_state"],
        optimizer_state=payload["optimizer_state"],
        scheduler_state=payload["scheduler_state"],
    )
