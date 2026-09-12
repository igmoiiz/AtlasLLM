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
from itertools import chain
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


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_acquire(config: dict, ctx: dict, force: bool) -> None:
    raw_root = _paths(config)[_RAW]
    for source in sources.enabled_sources(config):
        name = source["name"]
        raw_dir = raw_root / name
        result = sources.acquire(
            identifier=source["identifier"],
            hf_split=source.get("hf_split", "train"),
            raw_dir=raw_dir,
            source_name=name,
            max_documents=source.get("max_documents"),
        )
        print(f"  [acquire] {name}: {result['status']} "
              f"({result['documents']:,} rows) -> {raw_dir}")
        ctx["sources"].append(source)


def stage_process(config: dict, ctx: dict, force: bool) -> None:
    proc = _pipeline_cfg(config).get("processing", {})
    split_cfg = proc.get("split", {})
    ratios = [(k, float(v)) for k, v in
              split_cfg.get("ratios", {"train": 0.90, "val": 0.05, "test": 0.05}).items()]
    sample_cfg = proc.get("sample", {})
    paths = _paths(config)

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
            dedup=proc.get("dedup", True),
            sample_seed=sample_cfg.get("seed", 42),
            sample_fraction=sample_cfg.get("fraction", 1.0),
            split_seed=split_cfg.get("seed", 42),
            split_ratios=ratios,
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

    for split in _SPLITS:
        bin_path = processed_dir / f"{split}.bin"
        if bin_path.is_file() and not force:
            print(f"  [tokenize] {split}: cached ({bin_path.stat().st_size // 2:,} tokens)")
            ctx["bin_paths"][split] = bin_path
            continue
        records = chain.from_iterable(
            documents.iter_split_docs(paths[_INTERIM] / src["name"], split)
            for src in (ctx["sources"] or sources.enabled_sources(config))
        )
        stats = packing.pack_split(tokenizer, records, bin_path)
        ctx["bin_paths"][split] = bin_path
        print(f"  [tokenize] {split}: {stats['tokens']:,} tokens "
              f"from {stats['documents']:,} docs -> {bin_path}")


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
    model_cfg = config.get("model", {})
    processed_dir = paths["processed"]
    vocab_size = ctx.get("vocab_size") or model_cfg["vocab_size"]

    meta = packing.write_meta(
        bin_paths=bin_paths,
        context_length=model_cfg["context_length"],
        vocab_size=vocab_size,
        tokenizer_path=str(paths["tokenizer"]),
        meta_path=processed_dir / "meta.json",
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
