"""Dataset registry and source acquisition.

The ``pipeline.registry`` config section declares every source. Acquisition
streams a HuggingFace dataset split into an immutable raw cache:

    data/raw/<name>/
        records.jsonl   one JSON object per row: {"text": str}
        source.json     provenance: identifier, split, count, created

Re-running acquisition on an existing, matching cache yields the cache,
never a re-download. Downloads are bounded by ``max_documents`` so only the
documents actually used are ever transferred (DATA_PIPELINE.md: "must not
unnecessarily download data that will never be used").
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Iterator

from datasets import load_dataset


def enabled_sources(config: dict) -> list[dict]:
    """Return the enabled sources from ``pipeline.registry``."""
    sources = config["pipeline"]["registry"]
    enabled = [s for s in sources if s.get("enabled", True)]
    if not enabled:
        raise ValueError("pipeline.registry has no enabled sources")
    return enabled


def dataset_version_id(identifier: str, hf_split: str) -> str:
    """Stable short id for a (dataset, split) pair, used to detect cache reuse."""
    digest = hashlib.sha1(f"{identifier}:{hf_split}".encode("utf-8")).hexdigest()
    return digest[:12]


def _row_text(row: dict) -> str | None:
    """Extract plain text from a dataset row against the common field names."""
    for key in ("text", "content", "story"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def stream_hf_rows(identifier: str, hf_split: str, max_documents: int | None) -> Iterator[str]:
    """Yield raw text rows from a HuggingFace dataset split.

    Lazy streaming keeps peak memory flat regardless of dataset size
    (DATA_PIPELINE.md: tokenize/acquire without loading the corpus into RAM).
    """
    stream = load_dataset(identifier, split=hf_split, streaming=True)
    for n, row in enumerate(stream, start=1):
        if max_documents is not None and n > max_documents:
            break
        text = _row_text(row)
        if text is not None:
            yield text


def acquire(
    identifier: str,
    hf_split: str,
    raw_dir: Path,
    source_name: str,
    max_documents: int | None = None,
    rows: Iterator[str] | None = None,
) -> dict:
    """Stream a dataset split into the immutable raw cache.

    Args:
        identifier: HuggingFace dataset identifier (e.g. ``roneneldan/TinyStories``).
        hf_split: dataset split to stream (e.g. ``train``).
        raw_dir: destination directory (``data/raw/<name>``).
        source_name: registry name, recorded for provenance.
        max_documents: bounded download cap.
        rows: optional pre-built iterator (tests inject synthetic rows here).

    Returns:
        Summary dict: ``status`` (acquired|cached), ``documents``, ``source``.
    """
    raw_dir = Path(raw_dir)
    records_path = raw_dir / "records.jsonl"
    source_path = raw_dir / "source.json"

    if records_path.is_file() and source_path.is_file():
        prior = json.loads(source_path.read_text(encoding="utf-8"))
        if prior.get("identifier") == identifier and prior.get("hf_split") == hf_split:
            return {"status": "cached", "documents": prior["documents"], "source": prior}

    row_iter = rows if rows is not None else stream_hf_rows(identifier, hf_split, max_documents)

    raw_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(records_path, "w", encoding="utf-8", newline="\n") as out:
        n = 0
        for text in row_iter:
            n += 1
            if max_documents is not None and n > max_documents:
                break
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1

    if count == 0:
        raise ValueError(f"Source {identifier} ({hf_split}) produced no usable rows")

    info = {
        "identifier": identifier,
        "hf_split": hf_split,
        "source": source_name,
        "documents": count,
        "version_id": dataset_version_id(identifier, hf_split),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw_file": str(records_path),
    }
    source_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    return {"status": "acquired", "documents": count, "source": info}


def iter_raw_rows(raw_dir: Path) -> Iterator[dict]:
    """Yield cached raw records as dicts."""
    records_path = Path(raw_dir) / "records.jsonl"
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
