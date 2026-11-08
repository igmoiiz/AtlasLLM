"""Streaming tokenization and packing of split documents into uint16 .bin files.

Consumes the interim per-split JSONL (``documents.iter_split_docs``), encodes
each document in a streaming pass, and writes the token stream to the exact
format :class:`data_pipeline.dataset.TextDataset` already consumes
(``data/processed/<name>/<split>_<source>.bin`` + ``meta.json``).  The whole
corpus is never materialised in RAM; only one document is tokenized at a time.

A mixture is built in two steps: ``pack_split`` writes one bounded file per
source, then ``interleave_bins`` composes them into the split the trainer reads.
Keeping the per-source files means the same tokenization can be re-mixed later
without re-tokenizing, and each source's validation stream stays separately
addressable for per-source perplexity.
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


def count_tokens(bin_path: Path) -> int:
    """Number of uint16 tokens in a .bin file (0 when it does not exist)."""
    path = Path(bin_path)
    return int(path.stat().st_size) // 2 if path.is_file() else 0


def pack_split(
    tokenizer: AtlasTokenizer,
    split_docs: Iterable[dict],
    bin_path: Path,
    max_tokens: int | None = None,
) -> dict:
    """Encode and pack an iterable of documents into one uint16 token file.

    Documents are concatenated end-to-end in the returned stream.  For a
    language-modeling corpus this is the standard "packed sequence" format:
    training windows are contiguous slices of the token stream (dataloader
    slicing), so there is no cross-document padding and no wasted tokens.

    ``max_tokens`` caps the file at an exact token count, truncating the
    document that crosses the cap. That is how a source's share of a mixture is
    enforced; ``truncated`` says whether the cap actually bound the output.

    Returns token-total statistics for the manifest.
    """
    bin_path = Path(bin_path)
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    token_total = 0
    documents = 0
    truncated = False
    with open(bin_path, "wb") as out:
        for record in split_docs:
            ids = _encode_document(tokenizer, record["text"])
            if max_tokens is not None:
                remaining = max_tokens - token_total
                if remaining <= 0:
                    truncated = True
                    break
                if len(ids) > remaining:
                    ids = ids[:remaining]
                    truncated = True
            if ids:
                np.asarray(ids, dtype=np.uint16).tofile(out)
                token_total += len(ids)
                documents += 1
            if max_tokens is not None and token_total >= max_tokens:
                break
    return {
        "tokens": token_total,
        "documents": documents,
        "avg_doc_tokens": round(token_total / max(documents, 1), 2),
        "truncated": truncated,
    }


def interleave_bins(
    sources: list[dict],
    out_path: Path,
    chunk_tokens: int = 4_000_000,
) -> dict:
    """Interleave per-source token files into one mixed stream.

    Each entry is ``{"name", "path", "tokens", "fraction"}``. Every round
    appends a chunk from each remaining source, sized by that source's share of
    ``chunk_tokens``, so a contiguous slice of the output reflects the mixture
    rather than a source-ordered concatenation. Sources that run out early are
    skipped in later rounds; the realised totals are returned so a shortfall
    against the target mixture is visible in the manifest.

    Sources with zero tokens are skipped. Chunks are read with ``np.fromfile``
    offsets instead of a memory map so the inputs are never locked.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = sum(int(s["tokens"]) for s in sources)
    if total == 0:
        raise ValueError("Cannot interleave: every source produced zero tokens")

    plans = []
    for source in sources:
        tokens = int(source["tokens"])
        if tokens == 0:
            continue
        fraction = float(source.get("fraction", tokens / total))
        plans.append({
            "name": source["name"],
            "path": Path(source["path"]),
            "tokens": tokens,
            "position": 0,
            "chunk": max(1, round(chunk_tokens * fraction)),
        })

    written = {plan["name"]: 0 for plan in plans}
    total_written = 0
    with open(out_path, "wb") as out:
        while True:
            moved = 0
            for plan in plans:
                if plan["position"] >= plan["tokens"]:
                    continue
                take = min(plan["chunk"], plan["tokens"] - plan["position"])
                piece = np.fromfile(
                    plan["path"], dtype=np.uint16,
                    count=take, offset=plan["position"] * 2,
                )
                if piece.size != take:
                    raise ValueError(
                        f"{plan['path']} ended early: wanted {take} tokens at "
                        f"{plan['position']}, got {piece.size}"
                    )
                piece.tofile(out)
                plan["position"] += take
                written[plan["name"]] += take
                total_written += take
                moved += take
            if moved == 0:
                break
    return {"tokens": total_written, "per_source": written}


def write_meta(
    bin_paths: dict[str, Path],
    context_length: int,
    vocab_size: int,
    tokenizer_path: str,
    meta_path: Path,
    per_source: dict[str, dict[str, Path]] | None = None,
) -> dict:
    """Write ``meta.json`` in the schema ``preprocessing`` already produces.

    ``per_source`` (``{split: {source: path}}``) is recorded alongside the mixed
    splits so per-source validation files stay discoverable by name.
    """
    splits = {}
    for name, bin_path in bin_paths.items():
        tokens = count_tokens(bin_path)
        splits[name] = {
            "path": str(bin_path),
            "tokens": tokens,
            "sequences": len(TextDataset(bin_path, context_length)) if tokens else 0,
        }
    per_source_meta: dict[str, dict[str, dict]] = {}
    for split, source_bins in (per_source or {}).items():
        per_source_meta[split] = {}
        for name, path in source_bins.items():
            tokens = count_tokens(path)
            per_source_meta[split][name] = {
                "path": str(path),
                "tokens": tokens,
                # An empty file has no windows and cannot be memory-mapped.
                "sequences": len(TextDataset(path, context_length)) if tokens else 0,
            }
    meta = {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "dtype": "uint16",
        "tokenizer_path": tokenizer_path,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "splits": splits,
        "per_source": per_source_meta,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
