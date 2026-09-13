"""Sampling-parameter sweep: does decoding warmth cause the derailment loops?

    python -m scripts.sampling_sweep --checkpoint checkpoints/atlasbase/run_20260912-232905/last.pt \
        --out reports/data/AtlasBase-v1/sampling_sweep.json

Runs a fixed prompt suite across a temperature x top-p grid at a fixed seed and
reports, per cell, the mean repetition score (evaluation.repetition) and the
echo-accuracy of the "repeat my words" probe.  The question this answers:
would tuning the *chat defaults* (lower temperature / tighter nucleus) reduce
the "Migrat"/"Dickens" loops, or is the derailment model-side and therefore an
AtlasBase-v2 / SFT problem?

Uses the shared probe_generation + repetition_score drivers (no duplicated
decoding or metric logic).
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import yaml

from evaluation.generation_eval import probe_generation
from evaluation.repetition import repetition_score
from inference.engine import InferenceEngine

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a quiet village,",
    "The Earth orbits the Sun because",
    "List ten things you know about the planet Mars.",
    "The difference between a cat and a dog is",
    "Please repeat the last three words I wrote exactly: 'one two three'.",
]

TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9]
TOP_P_VALUES = [0.8, 0.9, 0.95]
TOP_K = 50
ECHO_PHRASE = "one two three"
MAX_TOKENS = 200
SEED = 42


def echo_accuracy(text: str) -> bool:
    """True if the completion re-states the echo phrase at least twice
    (once from the prompt, once from the model echoing it back)."""
    return text.count(ECHO_PHRASE) >= 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", default=None, help="Config YAML (default: run-dir config.yaml)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config or Path(args.checkpoint).parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    engine = InferenceEngine.from_checkpoint(args.checkpoint, config=config, tokenizer=None, device="auto")
    print(f"device={engine.device} checkpoint={args.checkpoint}")

    cells: list[dict] = []
    for temperature in TEMPERATURES:
        for top_p in TOP_P_VALUES:
            probes = probe_generation(
                engine,
                PROMPTS,
                max_new_tokens=MAX_TOKENS,
                temperature=temperature,
                top_k=TOP_K,
                top_p=top_p,
                seed=SEED,
            )
            reps = [repetition_score(p.text) for p in probes]
            echo_idx = PROMPTS.index("Please repeat the last three words I wrote exactly: 'one two three'.")
            echo = echo_accuracy(probes[echo_idx].text)
            cell = {
                "temperature": temperature,
                "top_p": top_p,
                "mean_repetition": round(mean(reps), 4),
                "per_prompt_repetition": {str(i): round(r, 3) for i, r in enumerate(reps)},
                "echo_accuracy": echo,
            }
            cells.append(cell)
            print(f"T={temperature}  p={top_p}  rep={cell['mean_repetition']:.3f}  echo={echo}")

    best = min(cells, key=lambda c: c["mean_repetition"])
    print("\nlowest-repetition cell:", best)

    results = {
        "checkpoint": str(args.checkpoint),
        "temperature": TEMPERATURES,
        "top_p": TOP_P_VALUES,
        "cells": cells,
        "best": best,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
