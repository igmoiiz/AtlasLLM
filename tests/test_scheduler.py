"""Learning-rate schedule tests (warmup + cosine decay)."""

import pytest
import torch

from training.scheduler import build_lr_scheduler, lr_prefactor


def test_warmup_linear_from_zero():
    assert lr_prefactor(0, warmup_steps=100, max_steps=1000) == pytest.approx(0.0)
    assert lr_prefactor(50, warmup_steps=100, max_steps=1000) == pytest.approx(0.5)
    assert lr_prefactor(100, warmup_steps=100, max_steps=1000) == pytest.approx(1.0)


def test_cosine_decay_phase():
    warmup, max_steps = 10, 110
    assert lr_prefactor(warmup, warmup, max_steps) == pytest.approx(1.0)
    assert lr_prefactor((warmup + max_steps) // 2, warmup, max_steps) == pytest.approx(0.5)
    assert lr_prefactor(max_steps, warmup, max_steps) == pytest.approx(0.0)


def test_prefactor_not_increasing_after_warmup():
    warmup, max_steps = 50, 500
    prev = lr_prefactor(warmup, warmup, max_steps)
    for step in range(warmup + 1, max_steps + 1, 7):
        cur = lr_prefactor(step, warmup, max_steps)
        assert cur <= prev + 1e-9
        prev = cur


def test_prefactor_within_unit_bounds():
    for step in range(0, 1000, 13):
        assert 0.0 <= lr_prefactor(step, warmup_steps=100, max_steps=900) <= 1.0 + 1e-9


def test_zero_warmup_starts_at_full_lr():
    assert lr_prefactor(0, warmup_steps=0, max_steps=100) == pytest.approx(1.0)


def test_lr_scheduler_controls_optimizer_lr():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = build_lr_scheduler(opt, warmup_steps=10, max_steps=100)
    lrs = []
    for _ in range(20):
        opt.step()
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[0] == pytest.approx(0.0)
    assert max(lrs) == pytest.approx(1.0)
    assert lrs[-1] < max(lrs)
