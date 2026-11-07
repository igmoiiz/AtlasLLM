"""Score harness probes on fixed, comparable axes.

Every axis is a cheap heuristic computed from the completion text alone, so
scoring never touches the model. The printed transcript is the authoritative
record; the numbers exist to compare checkpoints 1:1.

    repetition    fraction of tokens inside a repeated 4-gram (0=none, 1=loop)
    prompt_echo   fraction of output words also present in the prompt
    on_topic      fraction of the prompt's words that reappear in the output
    passed        True/False if the probe has an expected answer, else None

Answers must appear as whole words: substring matching is too generous on a
rambling base model ("0" would "pass" the freezing-point probe by matching
"30 minutes"). ``repetition`` is
:func:`evaluation.repetition.repetition_score` - the same metric the
pre-harness scorecard and adversarial eval used, kept in one place so old and
new numbers stay comparable.
"""

import re

from evaluation.repetition import repetition_score
from harness.scenarios import Probe


def words(text: str) -> set[str]:
    """Lowercased word set; digits kept so numeric answers still count."""
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def completion_text(tokenizer, token_ids: list[int], prompt_tokens: int) -> str:
    """Decode only the newly generated tokens, not the echoed prompt."""
    return tokenizer.decode(token_ids[prompt_tokens:])


def answer_hit(completion: str, expected: tuple[str, ...]) -> bool | None:
    """``True``/``False`` when the probe has an answer, ``None`` when it does not."""
    if not expected:
        return None
    text = completion.lower()
    return any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in expected)


def score_probe(
    probe: Probe,
    completion: str,
    *,
    token_count: int,
    finished_reason: str,
    latency_ms: float,
) -> dict:
    """Score one completion. ``completion`` excludes the echoed prompt."""
    generated = words(completion)
    prompt = words(probe.prompt)
    return {
        "id": probe.id,
        "category": probe.category,
        "prompt": probe.prompt,
        "text": completion,
        "token_count": token_count,
        "finished_reason": finished_reason,
        "temperature": probe.temperature,
        "latency_ms": round(latency_ms, 1),
        "repetition": repetition_score(completion),
        "prompt_echo": (len(generated & prompt) / len(generated)) if generated else 0.0,
        "on_topic": (len(generated & prompt) / len(prompt)) if prompt else 0.0,
        "passed": answer_hit(completion, probe.expected),
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate rows into the report summary (means + per-category pass rates)."""
    n = len(rows)
    checkable = [r for r in rows if r["passed"] is not None]
    by_category: dict[str, dict] = {}
    for row in rows:
        bucket = by_category.setdefault(
            row["category"], {"total": 0, "checkable": 0, "passed": 0, "repetition_total": 0.0}
        )
        bucket["total"] += 1
        bucket["repetition_total"] += row["repetition"]
        if row["passed"] is not None:
            bucket["checkable"] += 1
            bucket["passed"] += int(row["passed"])
    for bucket in by_category.values():
        bucket["pass_rate"] = (bucket["passed"] / bucket["checkable"]) if bucket["checkable"] else None
        bucket["mean_repetition"] = bucket["repetition_total"] / bucket["total"]
        del bucket["repetition_total"]
    return {
        "n_probes": n,
        "mean_repetition": (sum(r["repetition"] for r in rows) / n) if n else 0.0,
        "mean_prompt_echo": (sum(r["prompt_echo"] for r in rows) / n) if n else 0.0,
        "mean_on_topic": (sum(r["on_topic"] for r in rows) / n) if n else 0.0,
        "checkable": len(checkable),
        "passed": sum(int(r["passed"]) for r in checkable),
        "pass_rate": (sum(int(r["passed"]) for r in checkable) / len(checkable)) if checkable else None,
        "by_category": by_category,
    }
