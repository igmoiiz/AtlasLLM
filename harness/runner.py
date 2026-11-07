"""Harness runner: drive a checkpoint through a probe suite, score, report.

    python -m harness.runner --checkpoint checkpoints/atlasbase_v2/run_<ts>/best.pt \
        --suite all --perplexity --out reports/data/AtlasBase-v2/harness.json

    python -m harness.runner --checkpoint <v3.pt> --suite all --perplexity \
        --baseline reports/data/AtlasBase-v2/harness.json

The runner is the harness entry point. It owns three things and nothing else:
loading the run config, driving ``InferenceEngine`` over
:mod:`harness.scenarios`, and handing the rows to :mod:`harness.reports`.
Scoring lives in :mod:`harness.scoring`, suite definitions in
:mod:`harness.scenarios`, the loop stress test in :mod:`harness.adversarial`.
Exit status is 0 normally and 1 when ``--baseline`` finds a regression.
"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from data_pipeline.dataset import TextDataset
from evaluation.perplexity import evaluate_loss, perplexity
from harness import adversarial, reports, scenarios, scoring
from inference.engine import InferenceEngine

DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.95
DEFAULT_SEED = 42
# Same default as scripts/chat.py: the decoding config AtlasLLM ships with.
DEFAULT_NO_REPEAT = 4


def load_run_config(config_path: str | Path | None, checkpoint: str | Path) -> dict:
    """Full run config: explicit path, else the trainer's ``config.yaml`` next
    to the checkpoint, else the config stashed inside the checkpoint."""
    if config_path is not None:
        return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    run_config = Path(checkpoint).parent / "config.yaml"
    if run_config.is_file():
        return yaml.safe_load(run_config.read_text(encoding="utf-8"))
    stashed = torch.load(checkpoint, map_location="cpu", weights_only=True).get("config")
    if not stashed:
        raise ValueError(
            f"no config for {checkpoint}: pass --config (this checkpoint predates config stashing)"
        )
    return stashed


def run_suite(
    engine: InferenceEngine,
    probes,
    *,
    top_k: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
    seed: int = DEFAULT_SEED,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT,
) -> list[dict]:
    """Generate and score every probe. Decoding settings come from the probe."""
    rows = []
    for probe in probes:
        start = time.perf_counter()
        gen = engine.generate(
            probe.prompt,
            max_new_tokens=probe.max_new_tokens,
            temperature=probe.temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        completion = scoring.completion_text(engine.tokenizer, gen.token_ids, gen.prompt_tokens)
        rows.append(
            scoring.score_probe(
                probe,
                completion,
                token_count=len(gen.token_ids) - gen.prompt_tokens,
                finished_reason=gen.finished_reason,
                latency_ms=latency_ms,
            )
        )
    return rows


def evaluate_perplexity(engine: InferenceEngine, config: dict, max_batches: int | None = None) -> dict:
    """Held-out loss/perplexity for every split the config points at.

    Reuses :func:`evaluation.perplexity.evaluate_loss`, so these numbers are the
    same math as the training curves. Splits whose ``.bin`` is missing are
    skipped rather than reported as zero.
    """
    ctx = engine.model.config.context_length
    batch_size = int(config.get("training", {}).get("batch_size", 8))
    data = config.get("data", {})
    results = {}
    for split in ("val", "test"):
        path = data.get(f"{split}_path")
        if not path or not Path(path).is_file():
            continue
        loader = torch.utils.data.DataLoader(
            TextDataset(path, ctx), batch_size=batch_size, shuffle=False
        )
        loss = evaluate_loss(engine.model, loader, engine.device, max_batches)
        results[f"{split}_loss"] = loss
        results[f"{split}_perplexity"] = perplexity(loss)
    if not results:
        raise FileNotFoundError("no val_path/test_path .bin found; run the data pipeline first")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", default=None, help="Config YAML (default: run-dir config.yaml)")
    parser.add_argument("--suite", default="all", help=f"One of {sorted(scenarios.SUITES)}")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=DEFAULT_NO_REPEAT)
    parser.add_argument("--max-tokens", type=int, default=None, help="Override every probe's token budget")
    parser.add_argument("--perplexity", action="store_true", help="Also measure val/test perplexity")
    parser.add_argument("--max-batches", type=int, default=200, help="Cap on perplexity batches per split")
    parser.add_argument("--stress", type=int, default=0, help="Run the loop stress test with N seeds per prompt")
    parser.add_argument("--baseline", type=Path, default=None, help="Previous harness JSON to compare against")
    parser.add_argument("--out", type=Path, default=None, help="Write the report JSON here")
    parser.add_argument("--quiet", action="store_true", help="Summary only, no per-probe transcripts")
    args = parser.parse_args(argv)

    config = load_run_config(args.config, args.checkpoint)
    engine = InferenceEngine.from_checkpoint(args.checkpoint, config=config, tokenizer=None, device=args.device)
    probes = scenarios.suite(args.suite)
    if args.max_tokens is not None:
        probes = tuple(replace(p, max_new_tokens=args.max_tokens) for p in probes)
    print(f"device={engine.device} checkpoint={args.checkpoint} suite={args.suite} probes={len(probes)}")

    rows = run_suite(
        engine,
        probes,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    payload = reports.build_report(
        str(args.checkpoint),
        suite=args.suite,
        rows=rows,
        decoding={
            "temperature": "per-probe",
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_tokens": args.max_tokens or "per-probe",
        },
    )
    if args.perplexity:
        payload["perplexity"] = evaluate_perplexity(
            engine, config, args.max_batches if args.max_batches > 0 else None
        )
    if args.stress > 0:
        payload["stress"] = adversarial.stress_repetition(
            engine,
            probes,
            seeds=range(args.stress),
            top_k=args.top_k,
            top_p=args.top_p,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

    reports.print_report(payload, verbose=not args.quiet)

    status = 0
    if args.baseline is not None:
        comparison = reports.compare(reports.load(args.baseline), payload)
        reports.print_comparison(comparison, args.baseline)
        status = 1 if comparison["regressions"] else 0
        payload["comparison"] = comparison
    if args.out is not None:
        reports.write_json(args.out, payload)
        print(f"\nwrote: {args.out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
