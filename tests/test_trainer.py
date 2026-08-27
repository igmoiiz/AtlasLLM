"""End-to-end training-loop smoke test."""

import numpy as np
import torch

from data_pipeline.dataset import TextDataset
from model.atlas_llm import AtlasLLM
from model.config import ModelConfig
from training.trainer import Trainer


def _write_tokens(path, tokens):
    np.asarray(tokens, dtype=np.uint16).tofile(path)


def test_trainer_reduces_loss_and_checkpoints(tmp_path):
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    _write_tokens(tmp_path / "train.bin", rng.integers(0, 20, size=4096).tolist())
    _write_tokens(tmp_path / "val.bin", rng.integers(0, 20, size=512).tolist())

    ctx = 16
    model = AtlasLLM(
        ModelConfig(vocab_size=256, context_length=ctx, d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.1, bias=False)
    )
    train_gen = torch.Generator().manual_seed(7)
    train_loader = torch.utils.data.DataLoader(TextDataset(tmp_path / "train.bin", ctx), batch_size=4, shuffle=True, generator=train_gen)
    val_loader = torch.utils.data.DataLoader(TextDataset(tmp_path / "val.bin", ctx), batch_size=8, shuffle=False)

    config = {
        "training": {"learning_rate": 1e-2, "weight_decay": 0.1, "max_steps": 40, "warmup_steps": 5, "grad_clip": 1.0},
        "logging": {"log_every": 10, "val_every": 20, "max_val_batches": 5},
        "checkpoint": {"dir": str(tmp_path / "ckpts"), "save_every": 20},
    }
    run_dir = tmp_path / "run"
    trainer = Trainer(model, config, train_loader, val_loader, device=torch.device("cpu"), seed=7, run_dir=run_dir)
    summary = trainer.train()

    assert summary["train_loss_first"] > summary["train_loss_last"] + 0.3
    assert summary["best_val_loss"] is not None
    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "metrics.jsonl").is_file()


def test_trainer_resume_continues_from_checkpoint(tmp_path):
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    _write_tokens(tmp_path / "train.bin", rng.integers(0, 20, size=4096).tolist())
    _write_tokens(tmp_path / "val.bin", rng.integers(0, 20, size=512).tolist())

    ctx = 16
    config = {
        "training": {"learning_rate": 1e-2, "weight_decay": 0.1, "max_steps": 20, "warmup_steps": 5, "grad_clip": 1.0},
        "logging": {"log_every": 10, "val_every": 10, "max_val_batches": 5},
        "checkpoint": {"dir": str(tmp_path / "ckpts"), "save_every": 10},
    }

    def make_trainer(run_dir, max_steps):
        cfg = {**config, "training": {**config["training"], "max_steps": max_steps}}
        model = AtlasLLM(ModelConfig(vocab_size=256, context_length=ctx, d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.1, bias=False))
        gen = torch.Generator().manual_seed(3)
        train_loader = torch.utils.data.DataLoader(TextDataset(tmp_path / "train.bin", ctx), batch_size=4, shuffle=True, generator=gen)
        val_loader = torch.utils.data.DataLoader(TextDataset(tmp_path / "val.bin", ctx), batch_size=8, shuffle=False)
        return Trainer(model, cfg, train_loader, val_loader, device=torch.device("cpu"), seed=3, run_dir=run_dir)

    first = make_trainer(tmp_path / "run1", max_steps=20).train()
    resumed = make_trainer(tmp_path / "run2", max_steps=40).train(resume_from=first["last_checkpoint"])
    assert resumed["train_loss_first"] == first["train_loss_last"]
    assert resumed["best_val_loss"] is not None
    assert (tmp_path / "run2" / "last.pt").is_file()
