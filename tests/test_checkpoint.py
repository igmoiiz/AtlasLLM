"""Checkpoint save/load/resume tests."""

import os

import pytest
import torch

from model.atlas_llm import AtlasLLM
from model.config import ModelConfig
from training.checkpoint import load_checkpoint, newest_readable_checkpoint, save_checkpoint
from training.scheduler import build_lr_scheduler
from training.train import resolve_resume

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


def _write_ckpt(path, step: int, mtime: float) -> None:
    model, opt, sched = _make_state(step)
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, step=step, config={}, metrics={})
    os.utime(path, (mtime, mtime))


def test_newest_readable_checkpoint_skips_truncated_file(tmp_path):
    run = tmp_path / "run_20260101-000000"
    run.mkdir()
    _write_ckpt(run / "best.pt", step=900, mtime=1_700_000_000)
    last = run / "last.pt"
    last.write_bytes((run / "best.pt").read_bytes()[:512])  # killed mid-write
    os.utime(last, (1_700_000_001, 1_700_000_001))

    assert newest_readable_checkpoint(tmp_path) == run / "best.pt"


def test_newest_readable_checkpoint_prefers_newer_run(tmp_path):
    old = tmp_path / "run_20260101-000000"
    new = tmp_path / "run_20260102-000000"
    old.mkdir()
    new.mkdir()
    _write_ckpt(old / "last.pt", step=100, mtime=1_700_000_000)
    _write_ckpt(new / "last.pt", step=200, mtime=1_700_000_100)

    assert newest_readable_checkpoint(tmp_path) == new / "last.pt"


def test_newest_readable_checkpoint_empty_dir_returns_none(tmp_path):
    assert newest_readable_checkpoint(tmp_path) is None


def test_newest_readable_checkpoint_missing_dir_returns_none(tmp_path):
    assert newest_readable_checkpoint(tmp_path / "absent") is None


# ------------------------------------------------------ --resume resolution


def test_resolve_resume_auto_uses_checkpoint_dir(tmp_path):
    run = tmp_path / "run_20260101-000000"
    run.mkdir()
    _write_ckpt(run / "last.pt", step=300, mtime=1_700_000_000)

    assert resolve_resume("auto", {"checkpoint": {"dir": str(tmp_path)}}) == run / "last.pt"


def test_resolve_resume_none_starts_fresh():
    assert resolve_resume(None, {"checkpoint": {"dir": "unused"}}) is None


def test_resolve_resume_auto_without_readable_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_resume("auto", {"checkpoint": {"dir": str(tmp_path)}})


def test_resolve_resume_explicit_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_resume(str(tmp_path / "nope.pt"), {})
