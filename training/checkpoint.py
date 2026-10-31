"""Checkpoint persistence: save/restore full training state in one file.

A checkpoint bundles model, optimizer, and scheduler state with the exact
training step, the run configuration, and current metrics. Save is atomic
(write to ``.tmp`` then rename) so a crash cannot leave a truncated file. A
power cut can still truncate a file that was already renamed into place, so
:func:`newest_readable_checkpoint` exists to find the newest checkpoint that
actually deserializes (``--resume auto``).
"""

import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch

# A truncated checkpoint fails in the deserializer rather than as a missing
# file; these are the failure modes observed from interrupted writes.
_READ_ERRORS = (RuntimeError, EOFError, OSError, pickle.UnpicklingError, zipfile.BadZipFile)


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


def newest_readable_checkpoint(root: str | Path) -> Path | None:
    """Newest ``run_*/{last,best}.pt`` under ``root`` that actually loads.

    Recovery policy after an interruption: the newest checkpoint wins, and a
    truncated file falls back to the next candidate (usually ``best.pt`` at
    the same step, or the previous run's ``last.pt``). Returns ``None`` when
    no candidate is readable, which means "start fresh".
    """
    root = Path(root)
    if not root.is_dir():
        return None
    candidates = sorted(
        (*root.glob("run_*/last.pt"), *root.glob("run_*/best.pt")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except _READ_ERRORS:
            continue
        if isinstance(payload, dict) and "model_state" in payload and "step" in payload:
            return path
    return None
