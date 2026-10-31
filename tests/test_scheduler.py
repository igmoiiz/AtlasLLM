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


# ------------------------------------------------------------------- WSD


def test_wsd_warms_up_then_holds_full_lr():
    warmup, max_steps = 100, 1000
    assert lr_prefactor(0, warmup, max_steps, schedule="wsd") == pytest.approx(0.0)
    assert lr_prefactor(50, warmup, max_steps, schedule="wsd") == pytest.approx(0.5)
    # decay_frac=0.1 -> decay starts at step 900; everything before is flat.
    for step in (warmup, 200, 500, 899, 900):
        assert lr_prefactor(step, warmup, max_steps, schedule="wsd") == pytest.approx(1.0)


def test_wsd_decays_only_in_the_tail():
    warmup, max_steps = 100, 1000
    assert lr_prefactor(950, warmup, max_steps, schedule="wsd") == pytest.approx(0.5, abs=1e-6)
    assert lr_prefactor(max_steps, warmup, max_steps, schedule="wsd") == pytest.approx(0.0)


def test_wsd_holds_longer_than_cosine():
    warmup, max_steps = 100, 1000
    mid = 500
    assert lr_prefactor(mid, warmup, max_steps, schedule="cosine") < 0.6
    assert lr_prefactor(mid, warmup, max_steps, schedule="wsd") == pytest.approx(1.0)


def test_wsd_is_monotone_after_warmup():
    warmup, max_steps = 20, 200
    prev = lr_prefactor(warmup, warmup, max_steps, schedule="wsd")
    for step in range(warmup + 1, max_steps + 1):
        cur = lr_prefactor(step, warmup, max_steps, schedule="wsd")
        assert cur <= prev + 1e-9
        prev = cur


def test_wsd_with_decay_frac_one_matches_cosine():
    warmup, max_steps = 50, 500
    for step in range(0, max_steps + 1, 17):
        assert lr_prefactor(step, warmup, max_steps, schedule="wsd", decay_frac=1.0) == pytest.approx(
            lr_prefactor(step, warmup, max_steps, schedule="cosine")
        )


def test_unknown_schedule_raises():
    with pytest.raises(ValueError):
        lr_prefactor(10, 5, 100, schedule="linear")


def test_decay_frac_out_of_range_raises():
    with pytest.raises(ValueError):
        lr_prefactor(10, 5, 100, schedule="wsd", decay_frac=0.0)
    with pytest.raises(ValueError):
        lr_prefactor(10, 5, 100, schedule="wsd", decay_frac=1.5)


def test_wsd_scheduler_reaches_zero_at_max_steps():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = build_lr_scheduler(opt, warmup_steps=2, max_steps=20, schedule="wsd")
    for _ in range(20):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)
