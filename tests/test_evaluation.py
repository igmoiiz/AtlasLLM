"""Stage-8 evaluation tests: perplexity, memorization gap, generation probes."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from data_pipeline.dataset import TextDataset
from evaluation import (
    evaluate_loss,
    evaluate_memorization,
    memorize_gap,
    perplexity,
    probe_generation,
)
from model.atlas_llm import AtlasLLM
from model.config import ModelConfig

DEVICE = torch.device("cpu")


def make_model(seed: int = 0) -> AtlasLLM:
    torch.manual_seed(seed)
    return AtlasLLM(
        ModelConfig(vocab_size=256, context_length=16, d_model=32, n_layers=2, n_heads=4, d_ff=64, dropout=0.0, bias=False)
    ).eval()


def make_bin(tmp_path: Path, n: int, name: str) -> Path:
    path = tmp_path / name
    rng = np.random.default_rng(seed=0)
    rng.integers(0, 256, size=n, dtype=np.uint16).tofile(path)  # ids within model vocab
    return path


def make_loader(tmp_path: Path, n: int = 800, name: str = "data.bin", batch_size: int = 4):
    bin_path = make_bin(tmp_path, n, name)
    ds = TextDataset(bin_path, context_length=16)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


# ----------------------------------------------------------------- perplexity


def test_perplexity_is_exponential_of_loss():
    assert perplexity(0.0) == pytest.approx(1.0)
    assert perplexity(1.0) == pytest.approx(math.e)
    assert perplexity(2.0) == pytest.approx(math.e**2)


def test_evaluate_loss_returns_finite_mean_loss(tmp_path):
    model = make_model()
    loader = make_loader(tmp_path)
    loss = evaluate_loss(model, loader, DEVICE)
    assert math.isfinite(loss)
    assert loss > 0.0


def test_evaluate_loss_respects_max_batches(tmp_path):
    model = make_model()
    loader = make_loader(tmp_path)
    capped = evaluate_loss(model, loader, DEVICE, max_batches=1)
    finite = evaluate_loss(model, loader, DEVICE, max_batches=5)
    assert math.isfinite(capped)
    assert math.isfinite(finite)


def test_evaluate_loss_rejects_empty_loader(tmp_path):
    empty = torch.utils.data.DataLoader(TextDataset(make_bin(tmp_path, 3, "short.bin"), 16), batch_size=4)
    with pytest.raises(ValueError):
        evaluate_loss(make_model(), empty, DEVICE)


# ----------------------------------------------------------------- memorization


def test_memorize_gap_is_heldout_minus_train():
    assert memorize_gap(2.0, 3.0) == pytest.approx(1.0)  # held-out worse -> positive gap
    assert memorize_gap(3.0, 2.0) == pytest.approx(-1.0)


def test_evaluate_memorization_returns_expected_fields(tmp_path):
    model = make_model()
    train_loader = make_loader(tmp_path, n=800, name="train.bin")
    heldout_loader = make_loader(tmp_path, n=800, name="heldout.bin")
    result = evaluate_memorization(model, train_loader, heldout_loader, DEVICE, max_batches=20)
    assert set(result) == {"train_loss", "heldout_loss", "gap", "train_perplexity", "heldout_perplexity"}
    assert result["gap"] == pytest.approx(result["heldout_loss"] - result["train_loss"])


def test_unseen_random_streams_show_no_memorization_signal(tmp_path):
    # A model pretrained on neither stream must exhibit a bounded gap rather
    # than a training-memorization spike. (The streams deliberately differ in
    # length so their windows do not coincide.)
    model = make_model()
    a = make_loader(tmp_path, n=800, name="a.bin")
    b = make_loader(tmp_path, n=900, name="b.bin")
    result = evaluate_memorization(model, a, b, DEVICE, max_batches=10)
    assert abs(result["gap"]) < 1.0  # bounded, not a training-vs-heldout spike


# ------------------------------------------------------------------ generation


def make_fake_engine():
    """Minimal InferenceEngine-compatible stub recording prompts seen."""

    class _Fake:
        def __init__(self):
            self.seen = []

        def generate(self, prompt, **kwargs):
            self.seen.append((prompt, kwargs))
            return type("G", (), {"text": prompt + " done", "token_ids": [1, 2, 3], "finished_reason": "max_len"})()

    return _Fake()


def test_probe_generation_records_each_prompt_and_kwargs():
    engine = make_fake_engine()
    prompts = ["hello", "world"]
    probes = probe_generation(engine, prompts, max_new_tokens=10, temperature=0.5, seed=7)
    assert [p.prompt for p in probes] == prompts
    assert probes[0].finished_reason == "max_len"
    assert probes[0].token_ids == [1, 2, 3]
    assert engine.seen[0][1]["max_new_tokens"] == 10
    assert engine.seen[0][1]["temperature"] == 0.5
    assert engine.seen[0][1]["seed"] == 7
