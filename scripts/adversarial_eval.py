"""Adversarial generation eval for a trained checkpoint.

    python -m scripts.adversarial_eval --checkpoint checkpoints/atlasbase/run_20260912-232905/last.pt --out reports/data/AtlasBase-v1/adversarial.json

Runs long-form probes (200 tokens) designed to expose the failure modes we
already see in base LMs - repetition loops, prompt echo, factual emptiness,
and claim-confidence. Each completion gets a repetition score: the fraction of
generated tokens that repeat a previous n-gram window (n=4). 0.0 = novel
continuation; values approaching 1.0 = the model loops a single phrase.

Reuses InferenceEngine + probe_generation; adds only the repetition metric.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from evaluation.generation_eval import probe_generation
from inference.engine import InferenceEngine

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a land far away,",
    "Artificial intelligence is",
    "In 1969, humans first",
    "The Earth orbits the Sun because",
    "Explain, step by step, how an internal combustion engine works.",
    "List ten things you know about the planet Mars.",
    "The king sat on his throne and",
    "If you flip a coin, the probability of heads is",
    "Please repeat the last three words I wrote exactly: 'one two three'.",
]


def repetition_score(text: str, n: int = 4) -> float:
    """Fraction of tokens that fall inside a previously-seen n-gram window."""
    tokens = text.split()
    if len(tokens) < n + 1:
        return 0.0
    seen: set[str] = set()
    repeated = 0
    for i in range(len(tokens) - n):
        window = " ".join(tokens[i : i + n])
        if window in seen:
            repeated += 1
        seen.add(window)
    return repeated / (len(tokens) - n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", default=None, help="Config YAML (default: run-dir config.yaml)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config or Path(args.checkpoint).parent / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    engine = InferenceEngine.from_checkpoint(args.checkpoint, config=config, tokenizer=None, device="auto")
    print(f"device={engine.device} checkpoint={args.checkpoint}")

    probes = probe_generation(
        engine,
        PROMPTS,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=50,
        top_p=0.95,
        seed=42,
    )

    rows = []
    avg_rep = 0.0
    for p in probes:
        rep = repetition_score(p.text)
        avg_rep += rep
        print("\n" + "=" * 70)
        print(f"> {p.prompt}")
        print(f"[{p.finished_reason}, {len(p.token_ids)} tokens, repetition={rep:.3f}]")
        print(p.text.strip())
        rows.append(
            {
                "prompt": p.prompt,
                "text": p.text,
                "token_count": len(p.token_ids),
                "finished_reason": p.finished_reason,
                "repetition": rep,
            }
        )

    summary = {"mean_repetition": avg_rep / max(len(rows), 1), "n_prompts": len(rows)}
    print("\n" + "=" * 70)
    print(f"mean repetition score: {summary['mean_repetition']:.3f}")

    results = {"checkpoint": str(args.checkpoint), "summary": summary, "probes": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote: {args.out}")


if __name__ == "__main__":
    sys.exit(main())