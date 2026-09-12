"""Streaming tokenization and packing of split documents into uint16 .bin files.

Consumes the interim per-split JSONL (``documents.iter_split_docs``), encodes
each document in a streaming pass, and writes the token stream to the exact
format :class:`data_pipeline.dataset.TextDataset` already consumes
(``data/processed/<name>/<split>.bin`` + ``meta.json``).  The whole corpus is
never materialised in RAM; only one document is tokenized at a time.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from data_pipeline.dataset import TextDataset
from tokenizer.tokenizer import AtlasTokenizer


def _encode_document(tokenizer: AtlasTokenizer, text: str) -> list[int]:
    """Encode one document and guard against uint16 overflow."""
    ids = tokenizer.encode(text)
    if max(ids) > 65535:
        raise ValueError(
            f"token id {max(ids)} exceeds uint16 range; vocabulary must stay <= 65536"
        )
    return ids


def pack_split(
    tokenizer: AtlasTokenizer,
    split_docs: Iterable[dict],
    bin_path: Path,
) -> dict:
    """Encode and pack an iterable of documents into one uint16 token file.

    Documents are concatenated end-to-end in the returned stream.  For a
    language-modeling corpus this is the standard "packed sequence" format:
    training windows are contiguous slices of the token stream (dataloader
    slicing), so there is no cross-document padding and no wasted tokens.

    Returns token-total statistics for the manifest.
    """
    bin_path = Path(bin_path)
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    token_total = 0
    doc_tokens: list[int] = []
    with open(bin_path, "wb") as out:
        for record in split_docs:
            ids = _encode_document(tokenizer, record["text"])
            doc_tokens.append(len(ids))
            token_total += len(ids)
            if ids:
                np.asarray(ids, dtype=np.uint16).tofile(out)
    return {"tokens": token_total, "documents": len(doc_tokens), "avg_doc_tokens": round(token_total / max(len(doc_tokens), 1), 2)}


def write_meta(
    bin_paths: dict[str, Path],
    context_length: int,
    vocab_size: int,
    tokenizer_path: str,
    meta_path: Path,
) -> dict:
    """Write ``meta.json`` in the schema ``preprocessing`` already produces."""
    splits = {}
    for name, bin_path in bin_paths.items():
        seqs = len(TextDataset(bin_path, context_length)) if bin_path.is_file() else 0
        splits[name] = {
            "path": str(bin_path),
            "tokens": _count_tokens(bin_path),
            "sequences": seqs,
        }
    meta = {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "dtype": "uint16",
        "tokenizer_path": tokenizer_path,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "splits": splits,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _count_tokens(bin_path: Path) -> int:
    return int(Path(bin_path).stat().st_size) // 2
