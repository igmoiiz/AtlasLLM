"""Document processing: normalize, filter, deduplicate, sample, and split.

This module consumes the immutable raw cache (``sources.iter_raw_rows``) and
produces deterministic, filtered, split-assigned document JSONL files under
``data/interim/<name>/``.  Every stage is measurable: counters and rejection
reasons are returned for the manifest.

Document identity is a content hash so that dedup, sampling, and splitting
are deterministic, reproducible, and streamable (never holding the full
corpus in RAM).
"""

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator


def stable_int_hash(salt: str, value: str, bits: int = 64) -> int:
    """Deterministic integer hash suitable for sampling and splitting.

    Unlike Python's builtin ``hash()``, which is randomised per process,
    this is built on :mod:`hashlib` and gives identical results across
    runs, platforms, and Python invocations.
    """
    digest = hashlib.sha1(f"{salt}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[: bits // 8], "big")


def document_id(text: str) -> str:
    """Content-addressed document identity (first 16 hex chars of SHA-1)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------

# English heuristic: a conservative, honest classifier (not an NLP library).
# We record the decision per spec ("Language detection must not be assumed
# perfect ... record the detected language and filtering decision").

_EN_WORDS = frozenset(
    ("the", "a", "an", "and", "of", "in", "to", "it", "is", "was", "that",
     "with", "for", "on", "at", "from", "by", "this", "his", "her", "but",
     "not", "as", "or", "they", "had", "have", "are", "were", "been", "he",
     "she", "one", "all", "would", "there", "what", "about", "who", "which",
     "their", "when", "will", "how", "each", "than", "them", "very", "some",
     "just", "into", "if", "out", "do", "no", "so", "my", "up", "can",
     "you", "did", "its", "also")
)


def detect_language(text: str) -> str:
    """Heuristic English detection for short/medium web text.

    Returns ``"en"`` for ASCII-dominant text with recognizable English
    stopword presence, else ``"unknown"``.  The heuristic is cheap,
    dependency-free, and honest about its limitations.  Configured
    ``languages`` are enforced downstream; this only labels.
    """
    if not text:
        return "unknown"

    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "unknown"

    ascii_letters = sum(1 for c in letters if ord(c) < 128)
    ascii_ratio = ascii_letters / len(letters)

    words_lower = text.lower().split()
    word_hits = sum(1 for w in words_lower if w in _EN_WORDS)

    if ascii_ratio >= 0.88 and word_hits >= 2:
        return "en"
    return "unknown"


# ---------------------------------------------------------------------------

# Normalization and cleaning (DATA_PIPELINE.md: "Do not aggressively rewrite
# legitimate text").

_CONTROL_CHARS = frozenset(chr(i) for i in range(0x20) if chr(i) not in "\t\n")


def normalize_text(text: str) -> str:
    """Conservative, idempotent text normalization."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch not in _CONTROL_CHARS)
    lines = text.split("\n")
    result: list[str] = []
    blank_streak = 0
    for ln in lines:
        ln_stripped = ln.rstrip()
        if ln_stripped == "":
            blank_streak += 1
            if blank_streak <= 2:
                result.append("")
        else:
            blank_streak = 0
            result.append(ln_stripped)
    while result and result[-1] == "":
        result.pop()
    return "\n".join(result)


def clean_text(text: str) -> str:
    """Text cleaning pass after normalization.

    Currently lightweight (remaining control chars, whitespace). Additional
    cleaning can be layered here as corpus complexity grows.
    """
    return normalize_text(text)


# ---------------------------------------------------------------------------

# Document-level and quality filters (every filter is measurable).

def filter_document(
    text: str,
    *,
    min_chars: int = 100,
    max_chars: int | None = None,
    min_alpha_ratio: float = 0.4,
) -> tuple[bool, str]:
    """Apply structural quality filters.

    Returns ``(accepted, reason)`` where ``reason`` is one of
    ``"ok"``, ``"too_short"``, ``"too_long"``, ``"low_alpha_ratio"``.
    """
    n_chars = len(text)
    if n_chars < min_chars:
        return False, "too_short"
    if max_chars is not None and n_chars > max_chars:
        return False, "too_long"
    letters = sum(1 for c in text if c.isalpha())
    total = max(n_chars, 1)
    if letters / total < min_alpha_ratio:
        return False, "low_alpha_ratio"
    return True, "ok"


# ---------------------------------------------------------------------------

# Language filter.

def filter_language(text: str, allowed_languages: list[str]) -> tuple[bool, str]:
    """Reject documents outside the configured language list.

    Returns ``(accepted, detected_language)``.
    """
    lang = detect_language(text)
    if not allowed_languages or lang in allowed_languages:
        return True, lang
    return False, lang


# ---------------------------------------------------------------------------

# Deduplication (exact, streaming hash set, DATA_PIPELINE.md: "At minimum:
# exact document deduplication").

def is_duplicate(doc_id: str, seen: set[str]) -> bool:
    """Check and record document id for exact deduplication."""
    if doc_id in seen:
        return True
    seen.add(doc_id)
    return False


# ---------------------------------------------------------------------------

# Deterministic sampling (DATA_PIPELINE.md: "seed = 42, target_tokens =
# configurable ... Running the same configuration and dataset version
# should produce the same selection where deterministic processing is
# possible").

def sample_keep(doc_id: str, seed: int, fraction: float) -> bool:
    """Deterministic sampling via content hash; accepts ``fraction`` ∈ [0, 1]."""
    if fraction >= 1.0:
        return True
    h = stable_int_hash(f"sample:{seed}", doc_id)
    return (h % 100_000) < round(fraction * 100_000)


# ---------------------------------------------------------------------------

# Deterministic document-level split (DATA_PIPELINE.md: "90% train 5%
# validation 5% test ... The split must be deterministic ... document-level").

_SPLIT_SALT = "split"


def assign_split(doc_id: str, seed: int, split_ratios: list[tuple[str, float]]) -> str:
    """Assign a split name using a deterministic hash bucket.

    ``split_ratios`` is an ordered list like
    ``[("train", 0.90), ("val", 0.05), ("test", 0.05)]``.
    """
    h = stable_int_hash(f"{_SPLIT_SALT}:{seed}", doc_id)
    bucket = h % 100_000
    cumulative = 0
    for name, fraction in split_ratios:
        cumulative += fraction
        if bucket < round(cumulative * 100_000):
            return name
    return split_ratios[-1][0]


# ---------------------------------------------------------------------------

# Full processing pass: reads raw cache, writes interim per-split JSONL,
# returns measurable statistics for the manifest.

def process(
    raw_iter: Iterator[dict],
    *,
    source_name: str,
    dataset_identifier: str,
    languages: list[str],
    min_chars: int = 100,
    max_chars: int | None = None,
    min_alpha_ratio: float = 0.4,
    dedup: bool = True,
    sample_seed: int = 42,
    sample_fraction: float = 1.0,
    split_seed: int = 42,
    split_ratios: list[tuple[str, float]] | None = None,
    interim_dir: Path,
) -> dict:
    """Streaming processing pass over raw records.

    Returns a stats dict with per-stage accepted/rejected counts suitable
    for the manifest.  Per-split JSONL files are written to ``interim_dir``.
    """
    if split_ratios is None:
        split_ratios = [("train", 0.90), ("val", 0.05), ("test", 0.05)]

    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    reasons: Counter = Counter()
    languages_counter: Counter = Counter()
    source_dist: Counter = Counter()
    seen_hashes: set[str] = set()
    split_counters: Counter = Counter()

    char_counts: list[int] = []

    handles = {name: open(interim_dir / f"{name}.jsonl", "w", encoding="utf-8", newline="\n")
               for name, _ in split_ratios}
    try:
        for row in raw_iter:
            text = row["text"]
            reasons["raw"] += 1

            text_clean = clean_text(text)
            accepted, reason = filter_document(
                text_clean, min_chars=min_chars, max_chars=max_chars,
                min_alpha_ratio=min_alpha_ratio,
            )
            if not accepted:
                reasons[f"quality_{reason}"] += 1
                continue

            ok_lang, lang = filter_language(text_clean, languages)
            languages_counter[lang] += 1
            if not ok_lang:
                reasons[f"language_{lang}"] += 1
                continue

            did = document_id(text_clean)
            if dedup and is_duplicate(did, seen_hashes):
                reasons["duplicate"] += 1
                continue

            if not sample_keep(did, sample_seed, sample_fraction):
                reasons["sample_dropped"] += 1
                continue

            split = assign_split(did, split_seed, split_ratios)
            rec = {
                "document_id": did,
                "text": text_clean,
                "source": source_name,
                "source_dataset": dataset_identifier,
                "language": lang,
                "metadata": {"chars": len(text_clean)},
            }
            handles[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
            split_counters[split] += 1
            source_dist[source_name] += 1
            char_counts.append(len(text_clean))
    finally:
        for h in handles.values():
            h.close()

    char_arr = sorted(char_counts) if char_counts else [0]
    total = len(char_counts)
    return {
        "source_name": source_name,
        "documents": split_counters,
        "total_documents": total,
        "per_source": dict(source_dist),
        "per_language": dict(languages_counter),
        "rejection_reasons": {k: v for k, v in reasons.items()
                              if k not in ("raw",)},
        "char_stats": {
            "total": sum(char_counts),
            "min": char_arr[0],
            "median": char_arr[total // 2],
            "mean": sum(char_counts) / max(total, 1),
            "max": char_arr[-1],
        },
        "split_counts": dict(split_counters),
    }


def iter_split_docs(interim_dir: Path, split_name: str) -> Iterator[dict]:
    """Yield documents from a split JSONL file."""
    path = Path(interim_dir) / f"{split_name}.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
