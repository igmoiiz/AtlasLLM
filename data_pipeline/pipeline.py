"""Automated dataset pipeline: acquire -> process -> tokenizer -> tokenize -> finalize.

Usage:

    python -m data_pipeline.pipeline --config configs/tinystories.yaml
    python -m data_pipeline.pipeline --config configs/tinystories.yaml --stage acquire
    python -m data_pipeline.pipeline --config configs/tinystories.yaml --force
    python -m data_pipeline.pipeline --config configs/tinystories.yaml --stage tokenize --force

Run the full pipeline (default) or a single stage.  Stages are resumable:
an existing, valid artifact is reused instead of recomputed.  ``--force``
rebuilds the selected stage (or all stages) from scratch.

The config reuses the experiment config contract: ``tokenizer.vocab_size`` /
``tokenizer.min_frequency`` drive BPE training, ``data.tokenizer_path`` is the
tokenizer output, ``data.train_path``/``val_path``/``test_path`` are the
processed artifacts, and ``model.context_length`` / ``model.vocab_size`` bound
tokenization.  The ``pipeline`` section adds only pipeline-specific parameters
(registry, processing filters, paths for raw/interim/reports).

Artifacts:

    data/raw/<source>/records.jsonl      immutable acquired cache
    data/interim/<source>/{train,val,test}.jsonl   processed documents
    tokenizer/model/<name>/              trained tokenizer artifact
    data/processed/<name>/{train,val,test}.bin     tokenized uint16 stream
    data/processed/<name>/meta.json      split metadata
    data/processed/<name>/manifest.json  validation gate + provenance
    reports/data/<name>/                 statistics report
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from data_pipeline import documents, manifest, packing, sources
from tokenizer.train_tokenizer import train_tokenizer

_RAW = "raw"
_INTERIM = "interim"
_SPLITS = ("train", "val", "test")


def _pipeline_cfg(config: dict) -> dict:
    cfg = config.get("pipeline", {})
    if not cfg:
        raise ValueError("Config has no 'pipeline' section; add registry/processing/paths")
    return cfg


def _paths(config: dict) -> dict:
    """Single place resolving every artifact directory used by the pipeline."""
    return {
        _RAW: Path(_pipeline_cfg(config)["paths"]["raw"]),
        _INTERIM: Path(_pipeline_cfg(config)["paths"]["interim"]),
        "processed": Path(config["data"]["train_path"]).parent,
        "tokenizer": Path(config["data"]["tokenizer_path"]),
        "reports": Path(_pipeline_cfg(config)["paths"]["reports"]),
    }


def _split_ratios(config: dict) -> list[tuple[str, float]]:
    """Configured train/val/test ratios as an ordered list."""
    ratios = _pipeline_cfg(config).get("processing", {}).get("split", {}).get("ratios")
    if not ratios:
        return [("train", 0.90), ("val", 0.05), ("test", 0.05)]
    return [(name, float(value)) for name, value in ratios.items()]


def _source_budgets(config: dict) -> dict[str, dict[str, int]]:
    """Per-source token budgets for train/val/test, from the mixture plan.

    ``pipeline.dataset.train_tokens`` is the total training budget and each
    source's ``target_fraction`` is its share of it. Validation and test keep
    the configured split ratio, so a source's val budget is
    ``train * val_ratio / train_ratio``; validating on the same mixture keeps
    per-source perplexity comparable to the training distribution.

    Returns an empty dict when the config declares no ``train_tokens``, which
    means "tokenize each split in full" (the debug and small configs).
    """
    dataset = _pipeline_cfg(config).get("dataset", {})
    total = int(dataset.get("train_tokens") or 0)
    if total <= 0:
        return {}

    registry = sources.enabled_sources(config)
    fractions = [float(source.get("target_fraction", 0.0)) for source in registry]
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError(
            "pipeline.dataset.train_tokens is set, so every enabled source needs "
            f"a positive target_fraction; got {fractions}"
        )
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(
            f"target_fraction must sum to 1.0, got {sum(fractions):.6f} from {fractions}"
        )

    ratios = dict(_split_ratios(config))
    train_ratio = ratios.get("train", 0.90) or 0.90
    budgets: dict[str, dict[str, int]] = {}
    for source, fraction in zip(registry, fractions, strict=True):
        train = round(total * fraction)
        budgets[source["name"]] = {
            split: train if split == "train" else round(train * ratios.get(split, 0.0) / train_ratio)
            for split in _SPLITS
        }
    return budgets


def _char_budget(config: dict, source_name: str, budgets: dict[str, dict[str, int]]) -> int | None:
    """Characters to acquire/process for one source before its token cap binds.

    Tokens are the unit the mixture is defined in, but the pipeline reads text.
    ``chars_per_token`` (default 4.0 for English BPE) converts one to the other
    and ``acquisition_margin`` (default 1.10) leaves headroom so truncation, not
    a short stream, is what enforces the cap. Processing stops at the same
    budget, so acquisition and filtering never do more work than the mixture
    needs.
    """
    if source_name not in budgets:
        return None
    proc = _pipeline_cfg(config).get("processing", {})
    chars_per_token = float(proc.get("chars_per_token", 4.0))
    margin = float(proc.get("acquisition_margin", 1.10))
    return round(sum(budgets[source_name].values()) * chars_per_token * margin)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_acquire(config: dict, ctx: dict, force: bool) -> None:
    raw_root = _paths(config)[_RAW]
    budgets = _source_budgets(config)
    for source in sources.enabled_sources(config):
        name = source["name"]
        raw_dir = raw_root / name
        result = sources.acquire(
            identifier=source["identifier"],
            hf_split=source.get("hf_split", "train"),
            raw_dir=raw_dir,
            source_name=name,
            max_documents=source.get("max_documents"),
            max_chars=_char_budget(config, name, budgets),
            config_name=source.get("config_name"),
        )
        print(f"  [acquire] {name}: {result['status']} "
              f"({result['documents']:,} rows) -> {raw_dir}")
        ctx["sources"].append(source)


def stage_process(config: dict, ctx: dict, force: bool) -> None:
    proc = _pipeline_cfg(config).get("processing", {})
    split_cfg = proc.get("split", {})
    ratios = _split_ratios(config)
    sample_cfg = proc.get("sample", {})
    paths = _paths(config)
    budgets = _source_budgets(config)
    dedup = proc.get("dedup", True)
    seen_hashes: set[str] = set()

    for source in (ctx["sources"] or sources.enabled_sources(config)):
        name = source["name"]
        interim_dir = paths[_INTERIM] / name
        stats_path = interim_dir / "processing_stats.json"
        done = stats_path.is_file() and all(
            (interim_dir / f"{s}.jsonl").is_file() for s, _ in ratios
        )
        if done and not force:
            print(f"  [process] {name}: cached at {interim_dir}")
            ctx["source_stats"].append(_read_json(stats_path))
            if dedup:
                # Cached sources still own their documents; load them so later
                # sources deduplicate against this one as a fresh run would.
                seen_hashes |= documents.load_document_ids(interim_dir)
            continue
        print(f"  [process] {name}: filtering {source['identifier']}")
        stats = documents.process(
            sources.iter_raw_rows(paths[_RAW] / name),
            source_name=name,
            dataset_identifier=source["identifier"],
            languages=proc.get("languages", ["en"]),
            min_chars=proc.get("min_chars", 30),
            max_chars=proc.get("max_chars"),
            min_alpha_ratio=proc.get("min_alpha_ratio", 0.4),
            dedup=dedup,
            sample_seed=sample_cfg.get("seed", 42),
            sample_fraction=sample_cfg.get("fraction", 1.0),
            split_seed=split_cfg.get("seed", 42),
            split_ratios=ratios,
            char_budget=_char_budget(config, name, budgets),
            seen_hashes=seen_hashes if dedup else None,
            interim_dir=interim_dir,
        )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"  [process] {name}: kept {stats['total_documents']:,} docs -> {interim_dir}")
        ctx["source_stats"].append(stats)


def _corpus_path(config: dict) -> Path:
    """Tokenizer training corpus: configured corpus_path, else a built file."""
    tokenizer_cfg = config.get("tokenizer", {})
    if tokenizer_cfg.get("corpus_path"):
        return Path(tokenizer_cfg["corpus_path"])
    interim_root = _paths(config)[_INTERIM]
    corpus = interim_root / "corpus.txt"
    if not corpus.is_file():
        corpus.parent.mkdir(parents=True, exist_ok=True)
        with open(corpus, "w", encoding="utf-8") as out:
            for source in sources.enabled_sources(config):
                for doc in documents.iter_split_docs(interim_root / source["name"], "train"):
                    out.write(doc["text"] + "\n")
    return corpus


def stage_tokenizer(config: dict, ctx: dict, force: bool) -> None:
    tokenizer_path = _paths(config)["tokenizer"]
    if (tokenizer_path / "tokenizer.json").is_file() and not force:
        print(f"  [tokenizer] cached -> {tokenizer_path}")
        ctx["tokenizer_stats"] = _read_json(tokenizer_path / "stats.json")
        return

    tokenizer_cfg = config.get("tokenizer", {})
    corpus = _corpus_path(config)
    print(f"  [tokenizer] training on {corpus}")
    tokenizer, stats = train_tokenizer(
        corpus_path=corpus,
        vocab_size=tokenizer_cfg.get("vocab_size", 16000),
        min_frequency=tokenizer_cfg.get("min_frequency", 2),
    )
    model_vocab = config.get("model", {}).get("vocab_size")
    if model_vocab is not None and model_vocab < tokenizer.vocab_size:
        raise ValueError(
            f"model.vocab_size ({model_vocab}) < tokenizer.vocab_size "
            f"({tokenizer.vocab_size}); the LM embedding table must cover "
            "the produced vocabulary"
        )
    tokenizer.save(tokenizer_path)
    stats_path = tokenizer_path / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    ctx["tokenizer_stats"] = stats
    print(f"  [tokenizer] vocab={tokenizer.vocab_size} -> {tokenizer_path}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_tokenize(config: dict, ctx: dict, force: bool) -> None:
    from tokenizer.tokenizer import AtlasTokenizer

    paths = _paths(config)
    tokenizer = AtlasTokenizer.from_pretrained(paths["tokenizer"])
    processed_dir = paths["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    ctx["vocab_size"] = tokenizer.vocab_size

    registry = ctx["sources"] or sources.enabled_sources(config)
    budgets = _source_budgets(config)
    chunk_tokens = int(
        _pipeline_cfg(config).get("dataset", {}).get("mix_chunk_tokens", 4_000_000)
    )
    per_source: dict[str, dict[str, Path]] = {split: {} for split in _SPLITS}

    for split in _SPLITS:
        for source in registry:
            name = source["name"]
            bin_path = processed_dir / f"{split}_{name}.bin"
            budget = budgets.get(name, {}).get(split)
            if bin_path.is_file() and not force:
                per_source[split][name] = bin_path
                print(f"  [tokenize] {split}/{name}: cached "
                      f"({packing.count_tokens(bin_path):,} tokens)")
                continue
            records = documents.iter_split_docs(paths[_INTERIM] / name, split)
            stats = packing.pack_split(tokenizer, records, bin_path, max_tokens=budget)
            per_source[split][name] = bin_path
            note = ""
            if budget is not None and stats["tokens"] < budget:
                note = f" (short of the {budget:,} token budget)"
            print(f"  [tokenize] {split}/{name}: {stats['tokens']:,} tokens "
                  f"from {stats['documents']:,} docs -> {bin_path}{note}")

        mixed_path = processed_dir / f"{split}.bin"
        mixed = packing.interleave_bins(
            [
                {
                    "name": source["name"],
                    "path": per_source[split][source["name"]],
                    "tokens": packing.count_tokens(per_source[split][source["name"]]),
                    "fraction": float(source.get("target_fraction", 1.0)),
                }
                for source in registry
            ],
            mixed_path,
            chunk_tokens=chunk_tokens,
        )
        ctx["bin_paths"][split] = mixed_path
        print(f"  [tokenize] {split}: mixed {mixed['tokens']:,} tokens from "
              f"{len(mixed['per_source'])} sources -> {mixed_path}")

    ctx["per_source_bins"] = per_source


def _merge_source_stats(source_stats: list[dict]) -> dict:
    """Merge per-source processing stats into manifest-shaped aggregates."""
    merged = {
        "rejection_reasons": {},
        "per_source": {},
        "per_language": {},
        "split_counts": {s: 0 for s in _SPLITS},
        "char_stats": {},
        "total_documents": 0,
    }
    reasons: dict[str, int] = {}
    languages: dict[str, int] = {}
    for stats in source_stats:
        for k, v in stats["rejection_reasons"].items():
            reasons[k] = reasons.get(k, 0) + v
        for k, v in stats["per_source"].items():
            merged["per_source"][k] = merged["per_source"].get(k, 0) + v
        for k, v in stats.get("per_language", {}).items():
            languages[k] = languages.get(k, 0) + v
        for k, v in stats["split_counts"].items():
            merged["split_counts"][k] = merged["split_counts"].get(k, 0) + v
        merged["total_documents"] += stats["total_documents"]
    merged["rejection_reasons"] = reasons
    merged["per_language"] = languages

    char = [s["char_stats"] for s in source_stats]
    if len(char) == 1:
        merged["char_stats"] = char[0]
    else:
        # Multi-source aggregate: total/min/max/mean exact; median shown from
        # the largest source (labeled) since true merged median needs raw data.
        counts = [round(c["total"] / c["mean"]) for c in char]
        merged["char_stats"] = {
            "total": sum(c["total"] for c in char),
            "min": min(c["min"] for c in char),
            "max": max(c["max"] for c in char),
            "mean": round(sum(c["total"] for c in char) / max(sum(counts), 1), 2),
            "median": char[counts.index(max(counts))]["median"],
            "median_note": "median from largest source",
        }
    return merged


def stage_finalize(config: dict, ctx: dict) -> dict:
    paths = _paths(config)
    # A standalone --stage finalize sees an empty ctx; reconstruct provenance
    # from the persisted per-source stats and the config registry.
    source_stats = ctx.get("source_stats") or [
        _read_json(paths[_INTERIM] / src["name"] / "processing_stats.json")
        for src in sources.enabled_sources(config)
    ]
    sources_used = ctx.get("sources") or sources.enabled_sources(config)
    bin_paths = ctx.get("bin_paths") or {
        name: Path(config["data"][f"{name}_path"])
        for name in _SPLITS
        if (config["data"].get(f"{name}_path"))
    }
    if not all(p.is_file() for p in bin_paths.values()):
        missing = [str(p) for s in _SPLITS
                   if not (p := bin_paths.get(s)) or not p.is_file()]
        raise FileNotFoundError(f"Cannot finalize: missing token files {missing}")
    per_source_bins = ctx.get("per_source_bins") or {
        split: {
            source["name"]: path
            for source in sources_used
            if (path := paths["processed"] / f"{split}_{source['name']}.bin").is_file()
        }
        for split in _SPLITS
    }
    model_cfg = config.get("model", {})
    processed_dir = paths["processed"]
    vocab_size = ctx.get("vocab_size") or model_cfg["vocab_size"]

    meta = packing.write_meta(
        bin_paths=bin_paths,
        context_length=model_cfg["context_length"],
        vocab_size=vocab_size,
        tokenizer_path=str(paths["tokenizer"]),
        meta_path=processed_dir / "meta.json",
        per_source=per_source_bins,
    )

    pipeline_cfg = _pipeline_cfg(config)
    dataset_cfg = pipeline_cfg.get("dataset", {})
    merged = _merge_source_stats(source_stats)
    manifest_obj = manifest.build_manifest(
        dataset_name=dataset_cfg.get("name", "atlas-tiny"),
        dataset_version=str(dataset_cfg.get("version", 1)),
        config=config,
        sources=sources_used,
        processing_stats=merged,
        meta=meta,
        bin_paths=bin_paths,
        tokenizer_info=ctx.get("tokenizer_stats") or {"status": "cached"},
        output_dir=processed_dir,
    )
    manifest.write_reports(
        manifest_obj,
        merged,
        paths["reports"] / dataset_cfg.get("name", "atlas-tiny"),
    )
    print(f"  [finalize] manifest + report -> {processed_dir}")
    return manifest_obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_STAGES = ("acquire", "process", "tokenizer", "tokenize", "finalize")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stage", choices=[*_STAGES, "all"], default="all",
        help="Run a single stage or all stages (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild the stage/artifacts instead of reusing cached results",
    )
    args = parser.parse_args()

    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    ctx = {"sources": [], "source_stats": [], "tokenizer_stats": {},
           "bin_paths": {}}
    stages = _STAGES if args.stage == "all" else (args.stage,)

    for stage in stages:
        if stage == "acquire":
            stage_acquire(config, ctx, args.force)
        elif stage == "process":
            stage_process(config, ctx, args.force)
        elif stage == "tokenizer":
            stage_tokenizer(config, ctx, args.force)
        elif stage == "tokenize":
            stage_tokenize(config, ctx, args.force)
        elif stage == "finalize":
            stage_finalize(config, ctx)

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
