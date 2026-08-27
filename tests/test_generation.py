"""Generation, sampling, KV-cache, and inference-engine tests."""

import torch

from inference.engine import InferenceEngine
from inference.kv_cache import KVCache
from inference.sampling import greedy, sample_next_token
from model.atlas_llm import AtlasLLM
from model.config import ModelConfig
from tokenizer.tokenizer import AtlasTokenizer

DEVICE = torch.device("cpu")

# ----------------------------------------------------------------- sampling


def test_greedy_picks_argmax():
    logits = torch.tensor([[0.1, 0.9, 0.4], [3.0, 1.0, 2.0]])
    assert torch.equal(greedy(logits), torch.tensor([1, 0]))


def test_temperature_zero_is_greedy():
    logits = torch.tensor([[0.1, 0.9, 0.4]])
    rng = torch.Generator().manual_seed(0)
    for _ in range(20):
        tok = sample_next_token(logits, temperature=0.0, rng=rng)
        assert torch.equal(tok, torch.tensor([1]))


def test_rejects_negative_temperature():
    logits = torch.tensor([[0.1, 0.9]])
    try:
        sample_next_token(logits, temperature=-0.5)
    except ValueError:
        return
    raise AssertionError("negative temperature must raise ValueError")


def test_top_k_one_is_deterministic():
    logits = torch.tensor([[0.0, 1.0, 5.0, 2.0]])
    rng = torch.Generator().manual_seed(0)
    for _ in range(30):
        tok = sample_next_token(logits, temperature=1.0, top_k=1, rng=rng)
        assert tok.item() == 2


def test_top_p_keeps_dominant_token_only():
    # token 0 dominates; a tight nucleus must never draw tokens 1..3.
    logits = torch.tensor([[6.0, 0.0, 0.0, 0.0]])
    rng = torch.Generator().manual_seed(0)
    for _ in range(30):
        tok = sample_next_token(logits, temperature=1.0, top_p=0.1, rng=rng)
        assert tok.item() == 0


def test_top_k_combined_with_top_p_is_safe_and_contained():
    # Regression: top-p combined with top-k once corrupted the surviving set
    # (all cells ended up silently -inf -> NaN -> multinomial crash) because
    # the nucleus logic mixed vocab order with rank order.
    logits = torch.randn(4, 64)
    rng = torch.Generator().manual_seed(7)
    tok = sample_next_token(logits, temperature=0.8, top_k=8, top_p=0.9, rng=rng)
    assert tok.shape == (4,)
    assert not torch.isnan(logits).any()
    top8 = torch.topk(logits, 8, dim=-1).indices
    for b in range(4):
        assert tok[b].item() in top8[b].tolist()


def test_high_temperature_spreads_distribution():
    # two near-equal tokens under a huge temperature should both appear.
    logits = torch.tensor([[2.0, 0.0]])
    rng = torch.Generator().manual_seed(1)
    draws = [sample_next_token(logits, temperature=50.0, rng=rng).item() for _ in range(400)]
    assert 0 < draws.count(0) < 400  # both tokens seen


def test_batched_sampling_returns_one_index_per_row():
    logits = torch.randn(3, 16)
    rng = torch.Generator().manual_seed(0)
    tok = sample_next_token(logits, temperature=0.9, top_k=8, rng=rng)
    assert tok.shape == (3,)


# ---------------------------------------------------------------- kv cache


def test_kv_cache_store_length_and_reset():
    cache = KVCache(n_layers=2)
    assert cache.length() == 0
    k0 = torch.zeros(1, 4, 3, 8)
    v0 = torch.zeros_like(k0)
    cache.append(0, k0, v0)
    assert cache.length() == 3
    pairs = cache.pairs()
    assert pairs[0][0] is k0 and pairs[0][1] is v0
    assert pairs[1][0] is None  # untouched layer stays empty
    cache.reset()
    assert cache.length() == 0
    assert cache.pairs()[0][0] is None


# ------------------------------------------ model cache-path correctness


def make_model(dropout: float = 0.0) -> AtlasLLM:
    torch.manual_seed(0)
    return AtlasLLM(
        ModelConfig(vocab_size=256, context_length=32, d_model=32, n_layers=2, n_heads=4, d_ff=64, dropout=dropout, bias=False)
    ).eval()


def test_prefill_with_empty_cache_matches_plain_forward():
    model = make_model()
    ids = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        reference = model(ids)
        logits, pairs = model(ids, past_key_values=[(None, None)] * model.config.n_layers)
    assert torch.equal(logits, reference)
    assert len(pairs) == model.config.n_layers
    for k, v in pairs:
        assert k.shape == (2, 4, 8, 8)  # [B, H, T, head_dim]
        assert v.shape == (2, 4, 8, 8)


def test_stepwise_cached_logits_match_full_recompute():
    model = make_model()
    ids = torch.randint(0, 256, (2, 8))
    prefix = ids[:, :4]
    with torch.no_grad():
        _, past = model(prefix, past_key_values=[(None, None)] * model.config.n_layers)
        for step in range(4, 8):
            new_token = ids[:, step : step + 1]
            logits_cached, past = model(new_token, past_key_values=past)
            logits_full = model(ids[:, : step + 1])
            assert torch.allclose(logits_cached[:, -1], logits_full[:, -1], atol=1e-5), f"step {step}"
            assert past[0][0].shape[-2] == step + 1  # cache grows by one each step


# -------------------------------------------------------------- engine


TOK_DIR = "tokenizer/model/debug"


def make_tokenizer() -> AtlasTokenizer:
    return AtlasTokenizer.from_pretrained(TOK_DIR)


def make_engine() -> InferenceEngine:
    tokenizer = make_tokenizer()
    torch.manual_seed(0)
    model = AtlasLLM(
        ModelConfig(vocab_size=tokenizer.vocab_size, context_length=32, d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0, bias=False)
    ).eval()
    return InferenceEngine(model, tokenizer, DEVICE)


def force(monkeypatch, token: int):
    """Make the engine deterministically emit ``token`` every step."""
    monkeypatch.setattr("inference.engine.sample_next_token", lambda *_a, **_k: torch.tensor([token], dtype=torch.long))


def test_generate_hits_max_length_and_fills_with_forced_token(monkeypatch):
    eng = make_engine()
    force(monkeypatch, 7)
    prompt = "The capital of France is"
    out = eng.generate(prompt, max_new_tokens=10, temperature=0.0)
    prompt_len = len(eng.tokenizer.encode(prompt))
    assert out.finished_reason == "max_len"
    assert len(out.token_ids) == prompt_len + 10
    assert all(t == 7 for t in out.token_ids[prompt_len:])
    assert out.text == eng.tokenizer.decode(out.token_ids)


def test_generate_stops_on_eos(monkeypatch):
    eng = make_engine()
    force(monkeypatch, eng.tokenizer.eos_id)
    prompt = "hello"
    out = eng.generate(prompt, max_new_tokens=10, temperature=0.0)
    assert out.finished_reason == "eos"
    assert out.token_ids[-1] == eng.tokenizer.eos_id
    assert len(out.token_ids) == len(eng.tokenizer.encode(prompt)) + 1


def test_generate_stops_on_stop_sequence(monkeypatch):
    eng = make_engine()
    tid = 8
    seq = eng.tokenizer.decode([tid])
    assert seq, "token 8 must decode to a non-empty string for this test"
    force(monkeypatch, tid)
    prompt = "test prompt"
    out = eng.generate(prompt, max_new_tokens=12, temperature=0.0, stop_sequences=(seq,))
    assert out.finished_reason == "stop_string"
    assert len(out.token_ids) == len(eng.tokenizer.encode(prompt)) + 1


def test_cached_and_non_cached_generation_agree():
    # Greedy decoding is deterministic; cached and recompute paths use the
    # same math (verified to 1e-5 at the logits level by the stepwise test),
    # so identical inputs must produce identical output.
    eng = make_engine()
    prompt = "Colorless green ideas sleep furiously"
    kwargs = dict(max_new_tokens=20, temperature=0.0)
    a = eng.generate(prompt, use_cache=False, **kwargs)
    b = eng.generate(prompt, use_cache=True, **kwargs)
    assert a.token_ids == b.token_ids
    assert a.text == b.text


def test_stream_reconstructs_generated_text():
    eng = make_engine()
    prompt = "begin"
    chunks = list(eng.stream(prompt, max_new_tokens=5, temperature=0.0))
    out = eng.generate(prompt, max_new_tokens=5, temperature=0.0)
    assert len(chunks) == 5
    assert "".join(chunks) == out.text


def test_long_prompt_is_truncated_to_fit_generation_budget():
    eng = make_engine()
    long_prompt = "word word " * 60  # comfortably more than 32 tokens
    out = eng.generate(long_prompt, max_new_tokens=5, temperature=0.0)
    assert len(out.token_ids) == 32  # prompt window + 5 new, bounded by context_length
    assert out.finished_reason == "max_len"


def test_from_checkpoint_loads_model(tmp_path):
    tokenizer = make_tokenizer()
    torch.manual_seed(0)
    model = AtlasLLM(
        ModelConfig(vocab_size=tokenizer.vocab_size, context_length=32, d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0, bias=False)
    )
    path = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict()}, path)
    engine = InferenceEngine.from_checkpoint(path, ModelConfig(vocab_size=tokenizer.vocab_size, context_length=32, d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0, bias=False), tokenizer, device="cpu")
    for pa, pb in zip(model.parameters(), engine.model.parameters()):
        assert torch.equal(pa, pb)


def test_engine_rejects_tokenizer_larger_than_model():
    tokenizer = make_tokenizer()  # vocab 1280
    torch.manual_seed(0)
    model = AtlasLLM(ModelConfig(vocab_size=64, context_length=16, d_model=32, n_layers=1, n_heads=4, d_ff=64, dropout=0.0, bias=False))
    try:
        InferenceEngine(model, tokenizer, DEVICE)
    except ValueError:
        return
    raise AssertionError("mismatched vocab must raise ValueError")
