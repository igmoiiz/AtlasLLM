"""Inspect a processed uint16 token dataset.

Usage:
    python -m scripts.inspect_dataset --data data/processed/train.bin
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a processed token dataset")
    parser.add_argument("--data", required=True, type=Path, help="Path to a .bin token file")
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Token file not found: {args.data}")

    tokens = np.fromfile(args.data, dtype=np.uint16)
    print(f"File:            {args.data}")
    print(f"Tokens:          {len(tokens):,}")
    print(f"Max token id:    {tokens.max() if len(tokens) else 0}")

    meta_path = args.data.parent / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"Context length:  {meta['context_length']}")
        print(f"Vocab size:      {meta['vocab_size']}")
        for name, info in meta["splits"].items():
            marker = " <- this file" if Path(info["path"]).resolve() == args.data.resolve() else ""
            print(f"Split {name:5s} {info['tokens']:>10,} tokens  {info['sequences']:>8,} sequences{marker}")


if __name__ == "__main__":
    main()
