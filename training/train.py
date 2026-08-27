"""Training entry point.

Usage:

    python -m training.train --config configs/debug.yaml
    python -m training.train --config configs/debug.yaml --resume checkpoints/debug/run_20260101-000000/last.pt
    python -m training.train --config configs/small.yaml --steps 500   # override max_steps

Loads a YAML config, builds the model and dataloaders, and runs the
:class:`training.trainer.Trainer`. All run artifacts (checkpoints, metrics,
reproducibility metadata) land under ``checkpoint.dir/run_<timestamp>/``.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data_pipeline.dataset import TextDataset
from model.atlas_llm import AtlasLLM
from model.config import ModelConfig
from training.trainer import Trainer


def resolve_device(request: str) -> torch.device:
    """``auto`` picks CUDA when available, else CPU."""
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(request)


def resolve_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(name, torch.float32)


def write_reproducibility(run_dir: Path, config: dict, device: torch.device, seed: int, datasets: dict) -> None:
    """Record everything needed to reproduce a run (AGENTS.md rule 25)."""
    info = {
        "seed": seed,
        "model": config["model"],
        "training": config["training"],
        "data": config["data"],
        "tokenizer_path": config["data"].get("tokenizer_path"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "device": str(device),
        "hardware": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "dataset_windows": {name: len(ds) for name, ds in datasets.items()},
    }
    (run_dir / "reproducibility.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a configs/*.yaml file.")
    parser.add_argument("--resume", default=None, help="Checkpoint .pt file to resume from.")
    parser.add_argument("--steps", type=int, default=None, help="Override training.max_steps.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = resolve_device(config.get("device", "auto"))
    dtype = resolve_dtype(config.get("dtype", "float32"))
    seed = int(config["training"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    ctx = int(config["model"]["context_length"])
    batch_size = int(config["training"].get("batch_size", 2))
    datasets = {
        "train": TextDataset(config["data"]["train_path"], ctx),
        "val": TextDataset(config["data"]["val_path"], ctx),
    }
    train_loader = torch.utils.data.DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = torch.utils.data.DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)

    model = AtlasLLM(ModelConfig.from_dict(config["model"])).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} dtype={dtype} params={n_params:,}")

    if args.steps is not None:
        config["training"]["max_steps"] = args.steps
    run_dir = Path(config["checkpoint"]["dir"]) / f"run_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    write_reproducibility(run_dir, config, device, seed, datasets)

    trainer = Trainer(model, config, train_loader, val_loader, device=device, seed=seed, run_dir=run_dir)
    summary = trainer.train(resume_from=Path(args.resume) if args.resume else None)

    print(f"\nFinished {config['training']['max_steps']} steps")
    print(f"  train loss: {summary['train_loss_first']:.4f} -> {summary['train_loss_last']:.4f}")
    if summary["best_val_loss"] is not None:
        print(f"  best val loss: {summary['best_val_loss']:.4f}")
    print(f"  last: {summary['last_checkpoint']}")
    if summary["best_checkpoint"]:
        print(f"  best: {summary['best_checkpoint']}")


if __name__ == "__main__":
    sys.exit(main())
