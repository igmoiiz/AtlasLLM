"""Tests for the full AtlasLLM transformer and the language-modeling loss.

Covers mandatory checks from AGENTS.md Rule 23: output shapes, causal masking
at model level, gradient flow, forward pass, loss calculation, and checkpoint
(state_dict) reload.
"""

import math

import pytest
import torch

from model.atlas_llm import AtlasLLM
from model.config import ModelConfig
from training.loss import lm_cross_entropy

V, T, B = 256, 16, 2


def make_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=V,
        context_length=T,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        bias=False,
    )
    base.update(overrides)
    return ModelConfig(**base)


@pytest.fixture
def model() -> AtlasLLM:
    torch.manual_seed(0)
    return AtlasLLM(make_config())


def test_forward_output_shape(model: AtlasLLM) -> None:
    logits = model(torch.randint(0, V, (B, T)))
    assert logits.shape == (B, T, V)


def test_forward_is_pure_tensor(model: AtlasLLM) -> None:
    """Model purity (AGENTS.md Rule 8): forward returns tensors, not tuples."""
    out = model(torch.randint(0, V, (B, T)))
    assert isinstance(out, torch.Tensor)


def test_logits_are_well_conditioned(model: AtlasLLM) -> None:
    logits = model(torch.randint(0, V, (B, T)))
    assert torch.all(torch.isfinite(logits))
    assert logits.dtype == torch.float32


def test_causal_at_model_level(model: AtlasLLM) -> None:
    """Changing only future tokens must not change earlier-prefix logits."""
    model.eval()
    ids = torch.randint(0, V, (B, T))
    altered = ids.clone()
    altered[:, T // 2 :] = torch.randint(0, V, (B, T - T // 2))
    out1 = model(ids)
    out2 = model(altered)
    assert torch.allclose(out1[:, : T // 2], out2[:, : T // 2], atol=1e-5, rtol=1e-5)


def test_gradient_flows_through_every_parameter(model: AtlasLLM) -> None:
    ids = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))
    loss = lm_cross_entropy(model(ids), targets)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} did not receive a gradient"
        assert not torch.all(param.grad == 0), f"{name} has a zero gradient"


def test_loss_is_positive_scalar(model: AtlasLLM) -> None:
    ids = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))
    loss = lm_cross_entropy(model(ids), targets)
    assert loss.dim() == 0
    assert loss.detach().item() > 0.0


def test_loss_on_constant_logits_is_log_vocab() -> None:
    """Uniform logits must give exactly log(V) (nt ~ log V at init)."""
    logits = torch.zeros(B * T, V)
    targets = torch.randint(0, V, (B * T,))
    loss = lm_cross_entropy(logits, targets)
    assert float(loss) == pytest.approx(math.log(V), rel=1e-5)


def test_loss_decreases_with_perfect_prediction() -> None:
    """A model that already predicts the targets must be at the loss floor."""
    logits = torch.randn(B * T, V)
    targets = torch.randint(0, V, (B * T,))
    perfect = logits.clone()
    perfect[torch.arange(B * T), targets] += 50.0  # strongly favor the true token
    loss_perfect = float(lm_cross_entropy(perfect, targets))
    loss_random = float(lm_cross_entropy(logits, targets))
    assert 0.0 <= loss_perfect < 0.1 < loss_random


def test_state_dict_roundtrip(model: AtlasLLM, tmp_path) -> None:
    """Save/load weights (checkpoint reload, Rule 23) preserves behavior."""
    ids = torch.randint(0, V, (B, T))
    before = model(ids)
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)
    torch.manual_seed(1)
    restored = AtlasLLM(make_config())
    assert not torch.allclose(restored(ids), before)  # different init
    restored.load_state_dict(torch.load(ckpt, weights_only=True), strict=True)
    assert torch.allclose(restored(ids), before, atol=1e-6, rtol=1e-6)


def test_deterministic_init_with_seed() -> None:
    x = torch.randint(0, V, (B, T))
    torch.manual_seed(7)
    m1 = AtlasLLM(make_config())
    torch.manual_seed(7)
    m2 = AtlasLLM(make_config())
    assert m1(x).equal(m2(x))


def test_rejects_sequence_longer_than_context() -> None:
    model = AtlasLLM(make_config())
    with pytest.raises(ValueError):
        model(torch.randint(0, V, (1, T + 1)))


def test_positional_embeddings_differ_and_are_learnable() -> None:
    torch.manual_seed(3)
    model = AtlasLLM(make_config())
    pe = model.pos_emb.pos_emb.weight  # [T, D]
    assert not torch.allclose(pe[0], pe[1])
    assert pe.requires_grad
    assert pe.shape == (T, 64)


def test_embeddings_and_head_are_not_tied(model: AtlasLLM) -> None:
    """No weight tying initially (architecture.md decision 5)."""
    assert model.lm_head.weight is not model.token_emb.token_emb.weight
    assert model.lm_head.weight.shape == (V, 64)


@pytest.mark.parametrize("d_model,n_heads", [(64, 5), (128, 3)])
def test_config_rejects_heads_that_do_not_divide_model(d_model: int, n_heads: int) -> None:
    with pytest.raises(ValueError):
        make_config(d_model=d_model, n_heads=n_heads)


def test_config_rejects_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError):
        make_config(d_model=0)
    with pytest.raises(ValueError):
        make_config(n_layers=0)
