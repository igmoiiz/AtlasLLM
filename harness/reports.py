"""Harness reports: one JSON payload per checkpoint, plus baseline comparison.

The JSON file is the artifact the project docs cite, so it carries everything
needed to judge a run: checkpoint path, decoding settings, per-probe rows,
aggregate means, and (when measured) perplexity and loop-stress numbers.

``compare`` is the regression gate. It fails a new checkpoint when either:

    val perplexity   > baseline * (1 + ppl_tolerance)        (default 2%)
    mean repetition  > baseline + repetition_tolerance       (default +0.10)

Both are relative/absolute deltas, not absolute targets, because the point is
"did this change make things worse than the last checkpoint", which is the only
question a harness can answer with a 48M base model. Per-probe pass flips are
reported as information; they are too noisy to gate on.
"""

import json
from pathlib import Path

from harness import scoring

PPL_TOLERANCE = 0.02
REPETITION_TOLERANCE = 0.10


def build_report(checkpoint: str, *, suite: str, rows: list[dict], decoding: dict) -> dict:
    """Assemble the JSON payload for one checkpoint/suite run."""
    return {
        "checkpoint": checkpoint,
        "suite": suite,
        "decoding": decoding,
        "summary": scoring.summarize(rows),
        "rows": rows,
    }


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _val_perplexity(report: dict) -> float | None:
    """Accept both the harness layout and the older ``eval.json`` layout."""
    perplexity_section = report.get("perplexity") or {}
    for value in (
        perplexity_section.get("val_perplexity"),
        report.get("val_perplexity"),
        perplexity_section.get("val_ppl"),
        report.get("val_ppl"),
    ):
        if value is not None:
            return float(value)
    return None


def _mean_repetition(report: dict) -> float | None:
    value = (report.get("summary") or {}).get("mean_repetition")
    return None if value is None else float(value)


def compare(
    baseline: dict,
    current: dict,
    *,
    ppl_tolerance: float = PPL_TOLERANCE,
    repetition_tolerance: float = REPETITION_TOLERANCE,
) -> dict:
    """Compare two harness reports; ``regressions`` is what fails the gate."""
    regressions: list[str] = []

    base_ppl, cur_ppl = _val_perplexity(baseline), _val_perplexity(current)
    if base_ppl is not None and cur_ppl is not None:
        limit = base_ppl * (1.0 + ppl_tolerance)
        if cur_ppl > limit:
            regressions.append(
                f"val perplexity {cur_ppl:.1f} > {limit:.1f} "
                f"(baseline {base_ppl:.1f}, +{100.0 * (cur_ppl / base_ppl - 1.0):.1f}% > {ppl_tolerance:.0%})"
            )

    base_rep, cur_rep = _mean_repetition(baseline), _mean_repetition(current)
    if base_rep is not None and cur_rep is not None:
        limit = base_rep + repetition_tolerance
        if cur_rep > limit:
            regressions.append(
                f"mean repetition {cur_rep:.3f} > {limit:.3f} "
                f"(baseline {base_rep:.3f}, +{cur_rep - base_rep:.3f} > {repetition_tolerance:.2f})"
            )

    new_failures, new_passes = _pass_flips(baseline, current)
    return {
        "baseline_checkpoint": baseline.get("checkpoint"),
        "regressions": regressions,
        "new_failures": new_failures,
        "new_passes": new_passes,
        "val_perplexity": {"baseline": base_ppl, "current": cur_ppl},
        "mean_repetition": {"baseline": base_rep, "current": cur_rep},
    }


def _pass_flips(baseline: dict, current: dict) -> tuple[list[str], list[str]]:
    """Probes that flipped pass/fail between the two reports (by probe id)."""
    def statuses(report: dict) -> dict[str, bool]:
        return {row["id"]: row["passed"] for row in report.get("rows", []) if row.get("passed") is not None}

    before, after = statuses(baseline), statuses(current)
    shared = before.keys() & after.keys()
    failures = sorted(p for p in shared if before[p] and not after[p])
    passes = sorted(p for p in shared if not before[p] and after[p])
    return failures, passes


def print_report(payload: dict, *, verbose: bool = True) -> None:
    """Print the transcript (per probe) and the aggregate summary."""
    if verbose:
        for row in payload["rows"]:
            status = "-" if row["passed"] is None else ("PASS" if row["passed"] else "FAIL")
            print("\n" + "=" * 70)
            print(f"[{row['category']}] {row['prompt']}")
            print(
                f"[{status} rep={row['repetition']:.3f} echo={row['prompt_echo']:.3f} "
                f"ontopic={row['on_topic']:.3f} {row['token_count']}tok "
                f"{row['finished_reason']} {row['latency_ms']:.0f}ms]"
            )
            print(row["text"].strip())

    summary = payload["summary"]
    print("\n" + "=" * 70)
    print(f"summary: {summary['n_probes']} probes, mean repetition {summary['mean_repetition']:.3f}, "
          f"echo {summary['mean_prompt_echo']:.3f}, on-topic {summary['mean_on_topic']:.3f}")
    if summary["checkable"]:
        print(f"answered: {summary['passed']}/{summary['checkable']} "
              f"({100.0 * (summary['pass_rate'] or 0.0):.0f}%)")
    for category, bucket in sorted(summary["by_category"].items()):
        rate = "-" if bucket["pass_rate"] is None else f"{100.0 * bucket['pass_rate']:.0f}%"
        print(f"  {category:12s} n={bucket['total']:<3d} checkable={bucket['checkable']:<3d} "
              f"pass={rate:4s} rep={bucket['mean_repetition']:.3f}")
    if payload.get("perplexity"):
        ppl = payload["perplexity"]
        parts = [f"{split} ppl {ppl[f'{split}_perplexity']:.1f}" for split in ("val", "test") if f"{split}_perplexity" in ppl]
        print("perplexity: " + ", ".join(parts))
    if payload.get("stress"):
        stress = payload["stress"]
        print(
            f"stress: {stress['loops']}/{stress['n_prompts']} prompts looped "
            f"(worst repetition {stress['worst_repetition']:.3f} over {len(stress['seeds'])} seeds, "
            f"threshold {stress['threshold']:.2f})"
        )


def print_comparison(comparison: dict, baseline_path: str | Path) -> None:
    """Print the gate result against ``baseline_path``."""
    print("\n" + "=" * 70)
    print(f"baseline: {comparison['baseline_checkpoint']} ({baseline_path})")
    ppl = comparison["val_perplexity"]
    if ppl["baseline"] is not None and ppl["current"] is not None:
        print(f"  val perplexity {ppl['baseline']:.1f} -> {ppl['current']:.1f} "
              f"({100.0 * (ppl['current'] / ppl['baseline'] - 1.0):+.1f}%)")
    rep = comparison["mean_repetition"]
    if rep["baseline"] is not None and rep["current"] is not None:
        print(f"  mean repetition {rep['baseline']:.3f} -> {rep['current']:.3f} "
              f"({rep['current'] - rep['baseline']:+.3f})")
    if comparison["new_failures"]:
        print(f"  lost passes: {', '.join(comparison['new_failures'])}")
    if comparison["new_passes"]:
        print(f"  new passes: {', '.join(comparison['new_passes'])}")
    if comparison["regressions"]:
        print("  REGRESSIONS:")
        for line in comparison["regressions"]:
            print(f"    - {line}")
    else:
        print("  no regression")
