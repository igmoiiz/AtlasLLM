"""Tests for the automated data pipeline (sources, documents, packing, CLI).

Covered (DATA_PIPELINE.md stages):
- acquire: idempotent cache, injected rows replace host delivery
- process: normalization, language/quality filters, dedup, sampling, split
- tokenize: streaming pack into uint16 .bin, manifest/meta, validation
- deterministic: processing and packing are reproducible across runs

All tests run offline: acquisition rows are injected and the tokenizer is
trained on a tiny in-memory corpus.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from data_pipeline import documents as docs
from data_pipeline import packing, pipeline, sources
from data_pipeline.dataset import TextDataset
from data_pipeline.manifest import validate_bin
from tokenizer.tokenizer import AtlasTokenizer
from tokenizer.train_tokenizer import train_from_texts

TRAIN_TEXTS = [
    "hello world, this is a test",
    "the quick brown fox jumps over the lazy dog",
    "numbers 1234567890 also work",
    "accented cafe text with more words here",
    "a short widget story with several words",
    "another batch of englishish sample text",
    "the end of the day was quiet and calm",
    "final line of stories for the corpus",
]

RAW_ROWS = [
    "The fox slept in the old wooden shed.",
    "A child found a shiny rock by the river and showed it to the teacher.",
    "Mary had a little lamb it followed her to school one day.",
    "The three brothers built a raft and crossed the lake in one night.",
    "This is a made up sentence without an obvious origin.",
    "Another sentence that is clearly English text for the corpus.",
]

# Rows that should be rejected by quality / language filters.
NOISY_ROWS = [
    "short",  # too_short
    "x" * 5000,  # too_long (only when max_chars set)
    "äöüäöüäöüäöüäöüäöüäöüäöüäöüäöüäöüäöüäö" * 4,  # non-ASCII-heavy -> lang filter
]


@pytest.fixture(scope="module")
def tokenizer() -> AtlasTokenizer:
    return train_from_texts(TRAIN_TEXTS, vocab_size=800, min_frequency=2)


# ---------------------------------------------------------------------------
# sources.acquire
# ---------------------------------------------------------------------------

def test_acquire_writes_records_and_source(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    result = sources.acquire(
        identifier="test/ds", hf_split="train", raw_dir=raw_dir,
        source_name="testsource", rows=iter(RAW_ROWS),
    )
    assert result["status"] == "acquired"
    assert result["documents"] == len(RAW_ROWS)
    assert (raw_dir / "records.jsonl").is_file()
    assert (raw_dir / "source.json").is_file()
    info = json.loads((raw_dir / "source.json").read_text(encoding="utf-8"))
    assert info["identifier"] == "test/ds"
    assert info["hf_split"] == "train"
    assert info["documents"] == len(RAW_ROWS)


def test_acquire_is_idempotent(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    sources.acquire(identifier="test/ds", hf_split="train", raw_dir=raw_dir,
                    source_name="testsource", rows=iter(RAW_ROWS))
    touched = {"count": 0}

    def rows_with_counter():
        touched["count"] += 1
        yield "should not be read again"

    result = sources.acquire(
        identifier="test/ds", hf_split="train", raw_dir=raw_dir,
        source_name="testsource", rows=rows_with_counter(),
    )
    assert result["status"] == "cached"
    assert result["documents"] == len(RAW_ROWS)
    assert touched["count"] == 0  # cache hit means rows iterator never consumed


def test_acquire_cache_invalidated_on_identifier_change(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    sources.acquire(identifier="test/ds", hf_split="train", raw_dir=raw_dir,
                    source_name="testsource", rows=iter(RAW_ROWS))
    result = sources.acquire(
        identifier="test/other", hf_split="train", raw_dir=raw_dir,
        source_name="testsource", rows=iter(["different payload"]),
    )
    assert result["status"] == "acquired"
    assert result["documents"] == 1


def test_acquire_respects_max_documents(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    result = sources.acquire(
        identifier="test/ds", hf_split="train", raw_dir=raw_dir,
        source_name="testsource", max_documents=2, rows=iter(RAW_ROWS),
    )
    assert result["documents"] == 2


# ---------------------------------------------------------------------------
# documents: normalization, language, filters, dedup, sampling, split
# ---------------------------------------------------------------------------

def test_normalize_text_is_conservative() -> None:
    messy = "line one\r\n\r\n\r\nline two\twith\ttabs\n\n\n"
    cleaned = docs.normalize_text(messy)
    assert "\r" not in cleaned
    assert "with\ttabs" in cleaned  # interior whitespace untouched
    assert cleaned != cleaned + "\n"
    assert docs.normalize_text(cleaned) == cleaned  # idempotent


def test_detect_language_labels_english() -> None:
    assert docs.detect_language("The quick brown fox jumps over the lazy dog") == "en"
    assert docs.detect_language("Mary had a little lamb, its fleece was white") == "en"
    assert docs.detect_language("aaaaaaaaaa bbbbbbbbbb cccccccccc") == "unknown"


def test_filter_document_quality() -> None:
    assert docs.filter_document("x" * 1000, min_chars=100) == (True, "ok")
    assert docs.filter_document("short", min_chars=100) == (False, "too_short")
    assert docs.filter_document("x" * 500, min_chars=100, max_chars=200) == (False, "too_long")
    assert docs.filter_document("1234567890" * 20, min_chars=100) == (False, "low_alpha_ratio")


def test_filter_language_enforced() -> None:
    ok, lang = docs.filter_language("The fox slept in the old wooden shed.", ["en", "unknown"])
    assert ok and lang == "en"
    rejected, lang = docs.filter_language("äöüäöüäöüäöüäö" * 3, ["en"])
    assert not rejected and lang == "unknown"


def test_dedup_is_exact_and_membership() -> None:
    seen: set[str] = set()
    text = "the-same-text"
    assert not docs.is_duplicate(text, seen)
    assert docs.is_duplicate(text, seen)


def test_sampling_is_deterministic() -> None:
    assert docs.sample_keep("abc", 42, 0.5) == docs.sample_keep("abc", 42, 0.5)
    assert docs.sample_keep("abc", 42, 1.0) is True
    assert docs.sample_keep("abc", 42, 0.0) is False
    ids = [f"doc-{i}" for i in range(200)]
    set_a = {i for i in ids if docs.sample_keep(i, 42, 0.5)}
    set_b = {i for i in ids if docs.sample_keep(i, 43, 0.5)}
    assert set_a != set_b


def test_split_assignment_is_deterministic_and_roughly_ratio() -> None:
    ratios = [("train", 0.90), ("val", 0.05), ("test", 0.05)]
    counts = {"train": 0, "val": 0, "test": 0}
    for i in range(20000):
        counts[docs.assign_split(f"sentence-number-{i}", 42, ratios)] += 1
    assert counts["train"] / 20000 == pytest.approx(0.90, abs=0.01)
    assert counts["val"] / 20000 == pytest.approx(0.05, abs=0.01)
    assert counts["test"] / 20000 == pytest.approx(0.05, abs=0.01)
    assert docs.assign_split("some-doc", 42, ratios) == docs.assign_split("some-doc", 42, ratios)


def test_process_streams_to_splits_and_counts(tmp_path: Path) -> None:
    interim = tmp_path / "interim"
    rows = [*RAW_ROWS, *NOISY_ROWS, *RAW_ROWS]
    stats = docs.process(
        iter({"text": t} for t in rows),
        source_name="testsource",
        dataset_identifier="test/ds",
        languages=["en"],  # unknown-language rows get rejected
        min_chars=20,
        max_chars=1000,
        dedup=True,
        sample_fraction=1.0,
        split_ratios=[("train", 0.9), ("val", 0.05), ("test", 0.05)],
        interim_dir=interim,
    )
    kept = stats["total_documents"]
    assert kept == len(RAW_ROWS)  # dupes and noisy rows dropped
    for split in ("train", "val", "test"):
        assert (interim / f"{split}.jsonl").is_file()
    assert sum(stats["split_counts"].values()) == kept
    assert stats["rejection_reasons"]["duplicate"] == len(RAW_ROWS)
    assert stats["rejection_reasons"]["quality_too_short"] == 1
    assert stats["rejection_reasons"]["language_unknown"] == 1


def test_process_deterministic_across_runs(tmp_path: Path) -> None:
    kwargs = dict(
        source_name="testsource", dataset_identifier="test/ds",
        languages=["en"], min_chars=10,
        split_ratios=[("train", 0.9), ("val", 0.05), ("test", 0.05)],
    )
    stats_a = docs.process(iter({"text": t} for t in RAW_ROWS), interim_dir=tmp_path / "a", **kwargs)
    stats_b = docs.process(iter({"text": t} for t in RAW_ROWS), interim_dir=tmp_path / "b", **kwargs)
    assert stats_a == stats_b
    for split in ("train", "val", "test"):
        assert (tmp_path / "a" / f"{split}.jsonl").read_bytes() == \
               (tmp_path / "b" / f"{split}.jsonl").read_bytes()


# ---------------------------------------------------------------------------
# packing / meta
# ---------------------------------------------------------------------------

def _split_records(*texts: str) -> list[dict]:
    return [{"text": t, "document_id": docs.document_id(t)} for t in texts]


def test_pack_split_matches_encode_concatenation(tmp_path: Path, tokenizer: AtlasTokenizer) -> None:
    records = _split_records("hello world", "second story line", "third")
    bin_path = tmp_path / "train.bin"
    result = packing.pack_split(tokenizer, iter(records), bin_path)
    expected_ids = [i for r in records for i in tokenizer.encode(r["text"])]
    assert result["tokens"] == len(expected_ids)
    assert result["documents"] == len(records)
    assert np.fromfile(bin_path, dtype=np.uint16).tolist() == expected_ids


def test_write_meta_records_split_counts(tmp_path: Path, tokenizer: AtlasTokenizer) -> None:
    records = _split_records("hello world", "second story line", "third")
    bin_path = tmp_path / "train.bin"
    packing.pack_split(tokenizer, iter(records), bin_path)
    meta = packing.write_meta(
        bin_paths={"train": bin_path},
        context_length=4,
        vocab_size=tokenizer.vocab_size,
        tokenizer_path="tok/path",
        meta_path=tmp_path / "meta.json",
    )
    expected = sum(len(tokenizer.encode(r["text"])) for r in records)
    assert meta["splits"]["train"]["tokens"] == expected
    assert meta["splits"]["train"]["sequences"] == len(TextDataset(bin_path, 4))
    assert meta["vocab_size"] == tokenizer.vocab_size
    assert meta["context_length"] == 4
    assert meta["dtype"] == "uint16"


def test_packing_deterministic(tmp_path: Path, tokenizer: AtlasTokenizer) -> None:
    records = _split_records("hello world", "second story line", "third")
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    packing.pack_split(tokenizer, iter(records), a)
    packing.pack_split(tokenizer, iter(records), b)
    assert a.read_bytes() == b.read_bytes()


def test_validate_bin_rejects_odd_bytes_and_range(tmp_path: Path, tokenizer: AtlasTokenizer) -> None:
    records = _split_records("hello world")
    bin_path = tmp_path / "train.bin"
    packing.pack_split(tokenizer, iter(records), bin_path)
    stat = validate_bin(bin_path, tokenizer.vocab_size)
    assert stat["tokens"] == result_tokens(tokenizer, records)

    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\x00\x01\x02")  # 3 bytes -> not a valid uint16 stream
    with pytest.raises(ValueError):
        validate_bin(bad, tokenizer.vocab_size)

    out_of_range = tmp_path / "range.bin"
    np.asarray([0, tokenizer.vocab_size], dtype=np.uint16).tofile(out_of_range)
    with pytest.raises(ValueError):
        validate_bin(out_of_range, tokenizer.vocab_size)


def result_tokens(tokenizer: AtlasTokenizer, records: list[dict]) -> int:
    return sum(len(tokenizer.encode(r["text"])) for r in records)


# ---------------------------------------------------------------------------
# end-to-end: stage functions + CLI (offline synthetic source)
# ---------------------------------------------------------------------------

def _pipeline_config(tmp_path: Path) -> dict:
    base = tmp_path / "artifacts"
    return {
        "pipeline": {
            "dataset": {"name": "synthetic", "version": 1},
            "paths": {
                "raw": str(base / "raw"),
                "interim": str(base / "interim"),
                "processed": str(base / "processed"),
                "reports": str(base / "reports"),
            },
            "registry": [{
                "name": "synth",
                "role": "debug",
                "source": "test",
                "identifier": "test/synthetic",
                "hf_split": "train",
                "enabled": True,
                "max_documents": 100,
            }],
            "processing": {
                "languages": ["en", "unknown"],
                "min_chars": 10,
                "max_chars": 1000,
                "min_alpha_ratio": 0.3,
                "dedup": True,
                "sample": {"seed": 42, "fraction": 1.0},
                "split": {"seed": 42, "ratios": {"train": 0.90, "val": 0.05, "test": 0.05}},
            },
        },
        "model": {"vocab_size": 1600, "context_length": 16},
        "tokenizer": {"vocab_size": 800, "min_frequency": 2},
        "data": {
            "tokenizer_path": str(base / "tokenizer"),
            "train_path": str(base / "processed" / "train.bin"),
            "val_path": str(base / "processed" / "val.bin"),
            "test_path": str(base / "processed" / "test.bin"),
        },
    }


def _acquire_synth(config: dict, tmp_path: Path, n: int) -> None:
    raw_dir = pipeline._paths(config)["raw"] / "synth"
    sources.acquire(
        identifier="test/synthetic", hf_split="train",
        raw_dir=raw_dir, source_name="synth", max_documents=n,
        rows=iter(f"synthetic row number {i} with enough words for the filters" for i in range(n)),
    )


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    _acquire_synth(config, tmp_path, n=60)
    ctx = {"sources": [], "source_stats": [], "tokenizer_stats": {}, "bin_paths": {}}
    ctx["sources"].extend(sources.enabled_sources(config))
    pipeline.stage_process(config, ctx, force=False)
    pipeline.stage_tokenizer(config, ctx, force=False)
    pipeline.stage_tokenize(config, ctx, force=False)
    pipeline.stage_finalize(config, ctx)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    processed = Path(config["data"]["train_path"]).parent
    assert (processed / "train.bin").is_file()
    assert (processed / "val.bin").is_file()
    assert (processed / "test.bin").is_file()
    assert (processed / "meta.json").is_file()
    assert (processed / "manifest.json").is_file()
    assert (Path(config["pipeline"]["paths"]["reports"]) / "synthetic" / "report.txt").is_file()

    meta = json.loads((processed / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["splits"]) == {"train", "val", "test"}

    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"]["name"] == "synthetic"
    assert manifest["metadata"]["max_token_id"] < meta["vocab_size"]

    # The produced dataset must be consumable by the real TextDataset.
    ds = TextDataset(processed / "train.bin", context_length=16)
    if len(ds) > 0:
        sample = ds[0]
        assert sample["input_ids"].shape == (16,)
        assert sample["targets"].shape == (16,)


def test_pipeline_resumes_from_cache(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    _acquire_synth(config, tmp_path, n=40)
    ctx_a = {"sources": [], "source_stats": [], "tokenizer_stats": {}, "bin_paths": {}}
    ctx_a["sources"].extend(sources.enabled_sources(config))
    pipeline.stage_process(config, ctx_a, force=False)
    pipeline.stage_tokenizer(config, ctx_a, force=False)
    pipeline.stage_tokenize(config, ctx_a, force=False)

    # A fresh run over existing artifacts must reuse caches, not error.
    ctx_b = {"sources": [], "source_stats": [], "tokenizer_stats": {}, "bin_paths": {}}
    ctx_b["sources"].extend(sources.enabled_sources(config))
    pipeline.stage_process(config, ctx_b, force=False)
    pipeline.stage_tokenizer(config, ctx_b, force=False)
    pipeline.stage_tokenize(config, ctx_b, force=False)
    assert ctx_b["bin_paths"]["train"].stat().st_size == ctx_a["bin_paths"]["train"].stat().st_size


def test_standalone_finalize_rebuilds_manifest(tmp_path: Path) -> None:
    """Fresh ctx of --stage finalize must recover token counts + provenance."""
    config = _pipeline_config(tmp_path)
    _acquire_synth(config, tmp_path, n=60)
    ctx = {"sources": [], "source_stats": [], "tokenizer_stats": {}, "bin_paths": {}}
    ctx["sources"].extend(sources.enabled_sources(config))
    pipeline.stage_process(config, ctx, force=False)
    pipeline.stage_tokenizer(config, ctx, force=False)
    pipeline.stage_tokenize(config, ctx, force=False)
    initial = pipeline.stage_finalize(config, ctx)  # full ctx path

    # A standalone finalize starts from an empty ctx.
    fresh = pipeline.stage_finalize(config, {})
    processed = Path(config["data"]["train_path"]).parent
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        assert manifest["metadata"]["splits"][split]["tokens"] == initial["metadata"]["splits"][split]["tokens"] > 0
    assert manifest["processing"]["language_distribution"]  # surfaced per-language


def test_cli_runs_full_pipeline_offline(tmp_path: Path) -> None:
    config = _pipeline_config(tmp_path)
    _acquire_synth(config, tmp_path, n=20)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "data_pipeline.pipeline", "--config", str(config_path)],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Pipeline complete." in result.stdout
    processed = Path(config["data"]["train_path"]).parent
    assert (processed / "manifest.json").is_file()
    assert (processed / "meta.json").is_file()
