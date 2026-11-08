"""Harness tests: suites, scoring, report comparison, runner, CLI gate."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from harness import adversarial, reports, scenarios, scoring
from harness.runner import (
    evaluate_per_source,
    evaluate_perplexity,
    load_run_config,
    main,
    run_suite,
)
from inference.engine import InferenceEngine
from model.atlas_llm import AtlasLLM
from model.config import ModelConfig
from tokenizer.tokenizer import AtlasTokenizer
from training.checkpoint import save_checkpoint
from training.scheduler import build_lr_scheduler

TOK_DIR = Path(__file__).resolve().parent / "fixtures" / "tokenizer"
DEVICE = torch.device("cpu")


# ------------------------------------------------------------------ scenarios


def test_suite_ids_are_unique():
    ids = [probe.id for probe in scenarios.suite("all")]
    assert len(ids) == len(set(ids))


def test_all_suite_is_the_union():
    assert len(scenarios.suite("all")) == (
        len(scenarios.SCORECARD) + len(scenarios.ADVERSARIAL) + len(scenarios.FACTUAL)
    )


def test_factual_suite_has_twenty_checkable_probes():
    factual = scenarios.suite("factual")
    assert len(factual) == 20
    assert all(probe.expected for probe in factual)
    # Knowledge probes are greedy so the same checkpoint answers the same way.
    assert all(probe.temperature == 0.0 for probe in factual)


def test_unknown_suite_raises():
    with pytest.raises(ValueError, match="unknown suite"):
        scenarios.suite("no-such-suite")


# -------------------------------------------------------------------- scoring


def test_answer_hit_matches_any_keyword():
    assert scoring.answer_hit("The answer is Paris.", ("paris",)) is True
    assert scoring.answer_hit("The answer is Paris.", ("paris", "france")) is True
    assert scoring.answer_hit("I do not know.", ("paris",)) is False


def test_answer_hit_is_none_without_expected_answer():
    assert scoring.answer_hit("anything at all", ()) is None


def test_score_probe_echo_and_on_topic():
    probe = scenarios.long_form("t", "story", "the cat sat")
    row = scoring.score_probe(probe, "the cat sat on the mat", token_count=6, finished_reason="max_len", latency_ms=12.34)
    assert row["prompt_echo"] == pytest.approx(3 / 5)  # the, cat, sat
    assert row["on_topic"] == pytest.approx(1.0)  # every prompt word reappears
    assert row["repetition"] == pytest.approx(0.0)  # both 4-grams are new
    assert row["passed"] is None
    assert row["latency_ms"] == pytest.approx(12.3)


def test_score_probe_marks_answer_hit():
    probe = scenarios.short_form("t", "The capital of France is", ("paris",))
    row = scoring.score_probe(probe, " Paris, the capital.", token_count=4, finished_reason="max_len", latency_ms=1.0)
    assert row["passed"] is True


def test_summarize_counts_checkable_and_categories():
    rows = [
        {"category": "factual", "passed": True, "repetition": 0.0, "prompt_echo": 0.0, "on_topic": 0.0},
        {"category": "factual", "passed": False, "repetition": 0.2, "prompt_echo": 0.5, "on_topic": 1.0},
        {"category": "story", "passed": None, "repetition": 0.1, "prompt_echo": 0.1, "on_topic": 0.5},
    ]
    summary = scoring.summarize(rows)
    assert summary["n_probes"] == 3
    assert summary["checkable"] == 2
    assert summary["passed"] == 1
    assert summary["pass_rate"] == pytest.approx(0.5)
    assert summary["mean_repetition"] == pytest.approx(0.1)
    assert summary["by_category"]["factual"]["pass_rate"] == pytest.approx(0.5)
    assert summary["by_category"]["story"]["pass_rate"] is None


def test_summarize_empty_rows():
    summary = scoring.summarize([])
    assert summary["n_probes"] == 0
    assert summary["pass_rate"] is None
    assert summary["mean_repetition"] == 0.0


def test_completion_text_drops_the_echoed_prompt():
    tokenizer = AtlasTokenizer.from_pretrained(TOK_DIR)
    prompt = "The capital of France is"
    prompt_ids = tokenizer.encode(prompt)
    ids = prompt_ids + tokenizer.encode(" Paris")
    assert scoring.completion_text(tokenizer, ids, len(prompt_ids)) == " Paris"


# ------------------------------------------------------------------- reports


def _report(rows, *, val_perplexity=None, checkpoint="ckpt"):
    payload = reports.build_report(checkpoint, suite="all", rows=rows, decoding={})
    if val_perplexity is not None:
        payload["perplexity"] = {"val_perplexity": val_perplexity}
    return payload


def _row(id="p", passed=None, repetition=0.0):
    return {"id": id, "category": "factual", "passed": passed, "repetition": repetition, "prompt_echo": 0.0, "on_topic": 0.0}


def test_compare_identical_reports_has_no_regressions():
    report = _report([_row(passed=True)], val_perplexity=100.0)
    assert reports.compare(report, report)["regressions"] == []


def test_compare_flags_perplexity_above_tolerance():
    baseline = _report([_row()], val_perplexity=100.0)
    worse = _report([_row()], val_perplexity=103.0)
    assert reports.compare(baseline, worse)["regressions"]
    within = _report([_row()], val_perplexity=101.0)
    assert reports.compare(baseline, within)["regressions"] == []


def test_compare_flags_repetition_above_tolerance():
    baseline = _report([_row(repetition=0.05)])
    worse = _report([_row(repetition=0.20)])
    regressions = reports.compare(baseline, worse)["regressions"]
    assert len(regressions) == 1 and "repetition" in regressions[0]
    assert reports.compare(baseline, _report([_row(repetition=0.10)]))["regressions"] == []


def test_compare_reports_pass_flips():
    baseline = _report([_row("a", passed=True), _row("b", passed=False)])
    current = _report([_row("a", passed=False), _row("b", passed=True)])
    comparison = reports.compare(baseline, current)
    assert comparison["new_failures"] == ["a"]
    assert comparison["new_passes"] == ["b"]
    assert comparison["regressions"] == []


def test_compare_understands_legacy_eval_json():
    legacy = {"checkpoint": "v2", "val_perplexity": 125.0}
    current = _report([_row()], val_perplexity=126.0)
    comparison = reports.compare(legacy, current)
    assert comparison["val_perplexity"] == {"baseline": 125.0, "current": 126.0}
    # 126.0 is +0.8% over 125.0, inside the 2% tolerance.
    assert comparison["regressions"] == []
    # ...but 128.0 is +2.4% and must fail.
    assert reports.compare(legacy, _report([], val_perplexity=128.0))["regressions"]


def test_compare_tolerates_missing_fields():
    assert reports.compare({}, {})["regressions"] == []


def test_write_and_load_roundtrip(tmp_path):
    payload = _report([_row()])
    path = reports.write_json(tmp_path / "nested" / "report.json", payload)
    assert reports.load(path) == payload


# -------------------------------------------------------------------- runner


def make_tokenizer() -> AtlasTokenizer:
    return AtlasTokenizer.from_pretrained(TOK_DIR)


def make_model(seed: int = 0) -> AtlasLLM:
    torch.manual_seed(seed)
    tokenizer = make_tokenizer()
    return AtlasLLM(
        ModelConfig(vocab_size=tokenizer.vocab_size, context_length=32, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    ).eval()


def make_engine() -> InferenceEngine:
    return InferenceEngine(make_model(), make_tokenizer(), DEVICE)


def test_run_suite_scores_every_probe():
    engine = make_engine()
    probes = tuple(replace(p, max_new_tokens=3) for p in scenarios.suite("scorecard"))
    rows = run_suite(engine, probes)
    assert len(rows) == len(probes)
    assert [row["id"] for row in rows] == [p.id for p in probes]
    for row in rows:
        assert 0.0 <= row["repetition"] <= 1.0
        assert row["token_count"] >= 0
        assert row["finished_reason"] in {"max_len", "eos", "stop"}


def test_run_suite_with_one_probe_and_tiny_budget():
    engine = make_engine()
    probes = (replace(scenarios.suite("factual")[0], max_new_tokens=2),)
    rows = run_suite(engine, probes)
    assert len(rows) == 1
    assert rows[0]["category"] == "factual"


def test_evaluate_perplexity_on_temp_bins(tmp_path):
    engine = make_engine()
    rng = np.random.default_rng(0)
    for split in ("val", "test"):
        rng.integers(0, engine.model.config.vocab_size, size=512, dtype=np.uint16).tofile(tmp_path / f"{split}.bin")
    config = {
        "data": {"val_path": str(tmp_path / "val.bin"), "test_path": str(tmp_path / "test.bin")},
        "training": {"batch_size": 2},
    }
    result = evaluate_perplexity(engine, config, max_batches=2)
    assert set(result) == {"val_loss", "val_perplexity", "test_loss", "test_perplexity"}
    assert result["val_perplexity"] > 1.0


def test_evaluate_perplexity_without_bins_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_perplexity(make_engine(), {"data": {"val_path": str(tmp_path / "absent.bin")}})


def test_evaluate_per_source_uses_meta_json(tmp_path):
    engine = make_engine()
    rng = np.random.default_rng(0)
    vocab = engine.model.config.vocab_size
    ctx = engine.model.config.context_length
    for name, size in (("alpha", 512), ("tiny", 4)):
        rng.integers(0, vocab, size=size, dtype=np.uint16).tofile(tmp_path / f"val_{name}.bin")
    meta = {
        "per_source": {
            "val": {
                "alpha": {"path": str(tmp_path / "val_alpha.bin"), "tokens": 512},
                "tiny": {"path": str(tmp_path / "val_tiny.bin"), "tokens": 4},
            }
        }
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    config = {
        "data": {"val_path": str(tmp_path / "val.bin")},
        "training": {"batch_size": 2},
    }
    result = evaluate_per_source(engine, config, max_batches=2)
    # A stream shorter than one window is skipped, not reported as zero.
    assert set(result) == {"alpha"}
    assert result["alpha"]["perplexity"] > 1.0
    assert result["alpha"]["tokens"] == 512 > ctx


def test_evaluate_per_source_without_meta_returns_empty(tmp_path):
    result = evaluate_per_source(make_engine(), {"data": {"val_path": str(tmp_path / "val.bin")}})
    assert result == {}


def test_load_run_config_prefers_explicit_path(tmp_path):
    explicit = tmp_path / "cfg.yaml"
    explicit.write_text("model: {vocab_size: 8}\n", encoding="utf-8")
    assert load_run_config(explicit, tmp_path / "ckpt.pt")["model"] == {"vocab_size": 8}


def test_load_run_config_reads_run_dir_yaml(tmp_path):
    (tmp_path / "config.yaml").write_text("model: {vocab_size: 9}\n", encoding="utf-8")
    assert load_run_config(None, tmp_path / "ckpt.pt")["model"] == {"vocab_size": 9}


def _write_checkpoint(path: Path, seed: int = 0, bins: dict | None = None) -> Path:
    model = make_model(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_lr_scheduler(opt, warmup_steps=1, max_steps=10)
    config = {
        "model": asdict(model.config),
        "data": {"tokenizer_path": str(TOK_DIR), **(bins or {})},
        "training": {"batch_size": 1},
    }
    return save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, step=0, config=config, metrics={})


def test_cli_runs_suite_and_writes_report(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "ckpt.pt")
    out = tmp_path / "report.json"
    status = main(
        ["--checkpoint", str(checkpoint), "--suite", "factual", "--max-tokens", "2",
         "--no-repeat-ngram-size", "0", "--quiet", "--out", str(out)]
    )
    assert status == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["suite"] == "factual"
    assert len(payload["rows"]) == 20
    assert payload["summary"]["checkable"] == 20


def test_cli_gate_passes_against_its_own_report(tmp_path):
    checkpoint = _write_checkpoint(tmp_path / "ckpt.pt")
    baseline = tmp_path / "baseline.json"
    argv = ["--checkpoint", str(checkpoint), "--suite", "scorecard", "--max-tokens", "3", "--quiet"]
    assert main(argv + ["--out", str(baseline)]) == 0
    assert main(argv + ["--baseline", str(baseline)]) == 0


def test_cli_gate_fails_when_baseline_is_stronger(tmp_path):
    rng = np.random.default_rng(0)
    bins = {}
    for split in ("val", "test"):
        path = tmp_path / f"{split}.bin"
        rng.integers(0, 1280, size=512, dtype=np.uint16).tofile(path)
        bins[f"{split}_path"] = str(path)
    checkpoint = _write_checkpoint(tmp_path / "ckpt.pt", bins=bins)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"checkpoint": "imaginary", "perplexity": {"val_perplexity": 1.0}}),
        encoding="utf-8",
    )
    status = main(
        ["--checkpoint", str(checkpoint), "--suite", "factual", "--max-tokens", "2", "--quiet",
         "--perplexity", "--max-batches", "1", "--baseline", str(baseline)]
    )
    assert status == 1  # a random-init model is far above ppl 1.0 * 1.02


# --------------------------------------------------------------- stress test


def test_stress_repetition_reports_worst_case():
    engine = make_engine()
    probes = (replace(scenarios.suite("adversarial")[0], max_new_tokens=2),)
    result = adversarial.stress_repetition(engine, probes, seeds=range(2))
    assert result["n_prompts"] == 1
    assert result["seeds"] == [0, 1]
    row = result["per_prompt"][0]
    assert row["worst_repetition"] >= row["mean_repetition"] >= 0.0
    assert row["worst_seed"] in (0, 1)
    assert result["loops"] == int(row["looped"])
