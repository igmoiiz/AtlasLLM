"""Dataset validation and manifest generation.

The manifest records the real measured values produced by every pipeline
stage (documents, tokens, bytes, split sizes, rejection reasons, seed,
software versions) per DATA_PIPELINE.md.  Validation runs before the
manifest is written and fails loudly when anything is corrupted, so a
manifest on disk is a trustworthy "this dataset is trainable" gate.
"""

import json
import platform
from pathlib import Path

import numpy as np
import torch

from data_pipeline.dataset import TextDataset


def validate_bin(bin_path: Path, vocab_size: int) -> dict:
    """Validate a uint16 token file: even byte size, in-range token ids."""
    bin_path = Path(bin_path)
    if not bin_path.is_file():
        raise FileNotFoundError(f"Missing token file: {bin_path}")

    size = bin_path.stat().st_size
    if size % 2 != 0:
        raise ValueError(f"Corrupt token file {bin_path}: odd byte length {size}")
    if size == 0:
        raise ValueError(f"Empty token file {bin_path}")

    tokens = np.fromfile(bin_path, dtype=np.uint16)
    if tokens.size == 0:
        raise ValueError(f"Empty token file {bin_path}")
    max_id = int(tokens.max())
    if max_id >= vocab_size:
        raise ValueError(
            f"Token id {max_id} out of range for vocab_size {vocab_size} in {bin_path}"
        )
    return {"tokens": int(tokens.size), "bytes": size, "max_token_id": max_id}


def validate_meta(meta: dict, bin_paths: dict[str, Path]) -> None:
    """Cross-check meta.json split entries against real token files."""
    for name in ("train", "val", "test"):
        if name not in bin_paths:
            continue
        path = bin_paths[name]
        meta_entry = meta["splits"].get(name)
        if meta_entry is None:
            stat = validate_bin(path, meta["vocab_size"])
            raise ValueError(
                f"meta.json missing split '{name}' but {path} exists with "
                f"{stat['tokens']} tokens; rerun packing/validation"
            )
        actual = validate_bin(path, meta["vocab_size"])
        if actual["tokens"] != meta_entry["tokens"]:
            raise ValueError(
                f"meta token count mismatch for {name}: manifest {meta_entry['tokens']} "
                f"vs file {actual['tokens']}"
            )


def build_manifest(
    *,
    dataset_name: str,
    dataset_version: str,
    config: dict,
    sources: list[dict],
    processing_stats: dict,
    meta: dict,
    bin_paths: dict[str, Path],
    tokenizer_info: dict,
    output_dir: Path,
) -> dict:
    """Validate built shards and write the dataset manifest.

    Returns manifest dict; also persists to ``output_dir/manifest.json``.
    """
    max_token_id = 0
    split_stats = {}
    for name, path in bin_paths.items():
        stat = validate_bin(path, meta["vocab_size"])
        max_token_id = max(max_token_id, stat["max_token_id"])
        split_stats[name] = {
            "tokens": stat["tokens"],
            "bytes": stat["bytes"],
            "sequences": len(TextDataset(path, meta["context_length"])),
        }

    manifest = {
        "dataset": {
            "name": dataset_name,
            "version": dataset_version,
        },
        "config": {
            "model_vocab_size": config.get("model", {}).get("vocab_size"),
            "context_length": meta["context_length"],
            "training_seed": config.get("training", {}).get("seed"),
        },
        "sources": sources,
        "processing": {
            "sample_seed": config["pipeline"]["processing"].get("sample", {}).get("seed", 42),
            "sample_fraction": config["pipeline"]["processing"].get("sample", {}).get("fraction", 1.0),
            "split_seed": config["pipeline"]["processing"].get("split", {}).get("seed", 42),
            "split_ratios": config["pipeline"]["processing"].get("split", {}).get("ratios"),
            "languages": config["pipeline"]["processing"].get("languages", ["en"]),
            "dedup": config["pipeline"]["processing"].get("dedup", True),
            "rejection_reasons": processing_stats["rejection_reasons"],
            "source_distribution": processing_stats["per_source"],
            "language_distribution": processing_stats.get("per_language", {}),
            "char_stats": processing_stats["char_stats"],
        },
        "tokenizer": tokenizer_info,
        "metadata": {
            "documents": {split: processing_stats["split_counts"].get(split, 0)
                          for split in ("train", "val", "test")},
            "total_documents": processing_stats["total_documents"],
            "splits": split_stats,
            "max_token_id": max_token_id,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_reports(manifest: dict, processing_stats: dict, output_dir: Path) -> None:
    """Write human-readable dataset statistics under ``reports/data/``.

    Per DATA_PIPELINE.md: documents, tokens, characters, average/median
    lengths, source distribution, language distribution, filter rejection
    counts, duplicate counts, train/validation/test counts.
    """
    lines: list[str] = []
    doc_count = manifest["metadata"]["total_documents"]
    lines.append(f"Dataset: {manifest['dataset']['name']} v{manifest['dataset']['version']}")
    lines.append(f"Documents (kept):   {doc_count:,}")
    for split, count in manifest["metadata"]["documents"].items():
        info = manifest["metadata"]["splits"].get(split, {})
        lines.append(f"  {split:5s}: {count:>8,} docs  {info.get('tokens', 0):>14,} tokens  "
                     f"{info.get('sequences', 0):>10,} sequences")
    lines.append(f"Sources: {json.dumps(manifest['processing']['source_distribution'])}")
    lines.append(f"Language labels: {json.dumps(manifest['processing'].get('language_distribution', {}))}")
    lines.append(f"Rejection reasons: {json.dumps(manifest['processing']['rejection_reasons'])}")
    char_stats = manifest["processing"]["char_stats"]
    lines.append(f"Chars: total={char_stats['total']:,} min={char_stats['min']:,} "
                 f"median={char_stats['median']:,} mean={char_stats['mean']:.1f} max={char_stats['max']:,}")
    lines.append(f"Tokenizer: {json.dumps(manifest['tokenizer'])}")
    lines.append(f"Config: {json.dumps(manifest['config'])}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "processing_stats.json").write_text(
        json.dumps(processing_stats, indent=2), encoding="utf-8"
    )
