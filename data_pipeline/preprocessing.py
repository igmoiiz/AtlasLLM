"""Tokenize split UTF-8 text files into ``uint16`` .bin files for training.

Usage:
    python -m data_pipeline.preprocessing --config configs/small.yaml

The config ``data`` section must provide the tokenizer path, the three raw
text sources (train/val/test), and the three output .bin paths. Each text
file is encoded once (whole file) and written as a raw uint16 array, exactly
matching the format consumed by :class:`data_pipeline.dataset.TextDataset`.

A ``meta.json`` is written next to ``train.bin`` recording vocabulary size,
context length, per-split token/sequence counts, tokenizer path, and creation
time (Rule 25 — reproducibility metadata).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from data_pipeline.dataset import TextDataset
from tokenizer.tokenizer import AtlasTokenizer


def tokenize_to_bin(
    tokenizer: AtlasTokenizer,
    text_path: Path,
    bin_path: Path,
) -> dict:
    """Encode ``text_path`` with ``tokenizer`` and write it as uint16 tokens."""
    text = text_path.read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    if ids and max(ids) > 65535:
        raise ValueError(
            f"{text_path} produced token id {max(ids)}, too large for uint16 storage"
        )
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(ids, dtype=np.uint16).tofile(bin_path)
    return {"tokens": len(ids)}


def build_processed_data(config: dict, tokenizer: AtlasTokenizer, meta_path: Path | None = None) -> dict:
    """Tokenize all configured splits and write ``meta.json`` metadata."""
    data = config.get("data")
    if not data:
        raise ValueError("Config has no 'data' section")
    context_length = config.get("model", {}).get("context_length")
    if not context_length:
        raise ValueError("Config has no 'model.context_length'")

    splits = {}
    for split in ("train", "val", "test"):
        text_path = Path(data[f"{split}_text"])
        bin_path = Path(data[f"{split}_path"])
        if not text_path.is_file():
            raise FileNotFoundError(f"Split source not found: {text_path}")
        stats = tokenize_to_bin(tokenizer, text_path, bin_path)
        splits[split] = {
            "path": str(bin_path),
            "source": str(text_path),
            "tokens": stats["tokens"],
            "sequences": len(TextDataset(bin_path, context_length)),
        }

    meta = {
        "vocab_size": tokenizer.vocab_size,
        "context_length": context_length,
        "dtype": "uint16",
        "tokenizer_path": data["tokenizer_path"],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "splits": splits,
    }

    if meta_path is None:
        meta_path = Path(data["train_path"]).parent / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AtlasLLM processed token datasets")
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML config")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data = config.get("data")
    if not data:
        raise ValueError(f"Config {args.config} has no 'data' section")

    tokenizer = AtlasTokenizer.from_pretrained(data["tokenizer_path"])
    meta = build_processed_data(config, tokenizer)

    print(f"Tokenizing with vocab {meta['vocab_size']} and context {meta['context_length']}")
    for split, info in meta["splits"].items():
        print(f"{split:5s} {info['tokens']:>10,} tokens  {info['sequences']:>8,} sequences  -> {info['path']}")
    print(f"Wrote {Path(meta['splits']['train']['path']).parent / 'meta.json'}")


if __name__ == "__main__":
    main()
