"""Checkpoint save/load/resume tests."""

import pytest
import torch

from model.atlas_llm import AtlasLLM
from model.config import ModelConfig
from training.checkpoint import load_checkpoint, save_checkpoint
from training.scheduler import build_lr_scheduler

VOCAB, CTX = 256, 16


def _make_state(seed: int):
    torch.manual_seed(seed)
    model = AtlasLLM(
        ModelConfig(vocab_size=VOCAB, context_length=CTX, d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0, bias=False)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
    sched = build_lr_scheduler(opt, warmup_steps=10, max_steps=1000)
    return model, opt, sched


def _tensors_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, dict):
        return all(k in b and _tensors_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple, range)):
        return len(a) == len(b) and all(_tensors_equal(x, y) for x, y in zip(a, b))
    return bool(torch.equal(torch.as_tensor(a), torch.as_tensor(b)))


def test_save_checkpoint_writes_file(tmp_path):
    model, opt, sched = _make_state(0)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, step=10, config={"training": {"max_steps": 100}}, metrics={"val_loss": 1.2})
    assert path.is_file()
    assert path.stat().st_size > 0


def test_roundtrip_restores_full_state(tmp_path):
    model_a, opt_a, sched_a = _make_state(0)
    ids = torch.randint(0, VOCAB, (2, CTX))
    for _ in range(5):
        opt_a.zero_grad()
        model_a(ids).mean().backward()
        opt_a.step()
        sched_a.step()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model=model_a, optimizer=opt_a, scheduler=sched_a, step=5, config={"training": {"max_steps": 100}}, metrics={"train_loss": 0.3})

    model_b, opt_b, sched_b = _make_state(1)
    ckpt = load_checkpoint(path, model=model_b, optimizer=opt_b, scheduler=sched_b)
    assert ckpt.step == 5
    assert ckpt.config == {"training": {"max_steps": 100}}
    assert ckpt.metrics == {"train_loss": 0.3}
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)
    assert _tensors_equal(opt_a.state_dict(), opt_b.state_dict())
    assert _tensors_equal(sched_a.state_dict(), sched_b.state_dict())


def test_load_missing_checkpoint_raises(tmp_path):
    model, opt, sched = _make_state(0)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt", model=model, optimizer=opt, scheduler=sched)
