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
    """Extract plain text from a dataset row against the common field names.

    ``TEXT`` (uppercase) is the field name used by ``sedthh/gutenberg_english``;
    the rest cover the usual conventions. A registry entry does not need to
    declare its field because every source advertises one of these names.
    """
    for key in ("text", "TEXT", "content", "story"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def stream_hf_rows(
    identifier: str,
    hf_split: str,
    max_documents: int | None,
    config_name: str | None = None,
) -> Iterator[str]:
    """Yield raw text rows from a HuggingFace dataset split.

    Lazy streaming keeps peak memory flat regardless of dataset size
    (DATA_PIPELINE.md: tokenize/acquire without loading the corpus into RAM).

    ``config_name`` selects a dataset configuration; ``wikimedia/wikipedia``
    needs one (``20231101.en``) because the default config does not exist.

    ``datasets`` is imported here, not at module scope: acquisition is the only
    part of the pipeline that needs the optional ``data`` extra, and tests or
    callers that feed rows directly must not require it.
    """
    from datasets import load_dataset

    if config_name:
        stream = load_dataset(identifier, config_name, split=hf_split, streaming=True)
    else:
        stream = load_dataset(identifier, split=hf_split, streaming=True)
    for n, row in enumerate(stream, start=1):
        if max_documents is not None and n > max_documents:
            break
        text = _row_text(row)
        if text is not None:
            yield text


def _cache_is_enough(
    prior: dict,
    identifier: str,
    hf_split: str,
    config_name: str | None,
    max_documents: int | None,
    max_chars: int | None,
) -> bool:
    """Whether a cached acquisition satisfies the caps now being requested.

    A cache built with a smaller budget looks valid by dataset name alone, which
    is how an old 200k-row FineWeb cache silently under-filled a 558M-token
    target. ``stopped_by`` records the cap that actually bound the previous
    acquisition, so a bigger request re-acquires instead of reusing short data.
    """
    if prior.get("identifier") != identifier or prior.get("hf_split") != hf_split:
        return False
    if prior.get("config_name") != config_name:
        return False
    if "stopped_by" not in prior:
        return True  # legacy cache; cannot prove it short, so keep old behaviour
    if prior["stopped_by"] == "end":
        return True  # the previous read reached the end of the split

    prior_chars = prior.get("max_chars")
    if max_chars is not None and (prior_chars is None or prior_chars < max_chars):
        return False
    prior_docs = prior.get("max_documents")
    if max_documents is not None and (prior_docs is None or prior_docs < max_documents):
        return False
    return True


def acquire(
    identifier: str,
    hf_split: str,
    raw_dir: Path,
    source_name: str,
    max_documents: int | None = None,
    max_chars: int | None = None,
    config_name: str | None = None,
    rows: Iterator[str] | None = None,
) -> dict:
    """Stream a dataset split into the immutable raw cache.

    Args:
        identifier: HuggingFace dataset identifier (e.g. ``roneneldan/TinyStories``).
        hf_split: dataset split to stream (e.g. ``train``).
        raw_dir: destination directory (``data/raw/<name>``).
        source_name: registry name, recorded for provenance.
        max_documents: bounded download cap in rows.
        max_chars: bounded download cap in characters. This is the cap that
            matters for a token budget: documents vary in size by orders of
            magnitude across sources, so a row cap alone either over-downloads
            or stops short.
        config_name: HuggingFace dataset configuration (``wikimedia/wikipedia``
            requires one).
        rows: optional pre-built iterator (tests inject synthetic rows here).

    Returns:
        Summary dict: ``status`` (acquired|cached), ``documents``, ``source``.
    """
    raw_dir = Path(raw_dir)
    records_path = raw_dir / "records.jsonl"
    source_path = raw_dir / "source.json"

    if records_path.is_file() and source_path.is_file():
        prior = json.loads(source_path.read_text(encoding="utf-8"))
        if _cache_is_enough(prior, identifier, hf_split, config_name, max_documents, max_chars):
            return {"status": "cached", "documents": prior["documents"], "source": prior}

    # Caps are enforced here, not in the generator, so ``stopped_by`` can say
    # which one bound the cache. A cap inside the generator would make an
    # exhausted stream and a capped stream indistinguishable.
    row_iter = (
        rows if rows is not None
        else stream_hf_rows(identifier, hf_split, None, config_name)
    )

    raw_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    chars = 0
    stopped_by = "end"
    with open(records_path, "w", encoding="utf-8", newline="\n") as out:
        n = 0
        for text in row_iter:
            n += 1
            if max_documents is not None and n > max_documents:
                stopped_by = "documents"
                break
            if max_chars is not None and chars >= max_chars:
                stopped_by = "chars"
                break
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            chars += len(text)

    if count == 0:
        raise ValueError(f"Source {identifier} ({hf_split}) produced no usable rows")

    info = {
        "identifier": identifier,
        "hf_split": hf_split,
        "config_name": config_name,
        "source": source_name,
        "documents": count,
        "chars": chars,
        "max_documents": max_documents,
        "max_chars": max_chars,
        "stopped_by": stopped_by,
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
