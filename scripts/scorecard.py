"""Manual-test scorecard: quantify the failure modes of a base LM checkpoint.

    python -m scripts.scorecard --checkpoint checkpoints/atlasbase/run_20260912-232905/last.pt \
        --out reports/data/AtlasBase-v1/scorecard.json

Each prompt in the suite is categorized (autocomplete / story / fact / echo),
and every completion is scored on the same axes so future checkpoints compare
1:1:

  repetition   - fraction of tokens inside a repeated 4-gram (0=none, 1=loop)
  prompt_echo  - fraction of output tokens that also appear in the prompt (0..1)
  on_topic     - fraction of output tokens among the prompt's content words
  finishing    - finished_reason + token count
  fact_hit     - (fact prompts only) 1 if a known answer keyword appears

Values are heuristics for *manual review* - the printed transcript is the
authoritative record; the numbers only make it comparable.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from evaluation.generation_eval import probe_generation
from evaluation.repetition import repetition_score
from inference.engine import InferenceEngine

PROMPTS = [
    ("autocomplete", "The capital of France is", {"paris"}),
    ("story", "Once upon a time, in a quiet village,", set()),
    ("autocomplete", "The Earth orbits the Sun because", set()),
    ("fact", "List ten things you know about the planet Mars.", {"mars"}),
    ("story", "The difference between a cat and a dog is", set()),
    ("echo", "Please repeat the last three words I wrote exactly: 'one two three'.", {"one two three"}),
]

MAX_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.95
SEED = 42


def _words(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z']+", text.lower()))


def generated_text(engine: InferenceEngine, prompt: str, token_ids: list[int]) -> str:
    """Decode only the newly generated tokens, not the echoed prompt."""
    prompt_len = len(engine.tokenizer.encode(prompt))
    return engine.tokenizer.decode(token_ids[prompt_len:])


def score(engine: InferenceEngine, prompt: str, kind: str, expected: set[str], token_ids: list[int]) -> dict:
    gen = generated_text(engine, prompt, token_ids)
    gen_words = _words(gen)
    prompt_words = _words(prompt)
    out_tokens = gen.split()
    base = {
        "kind": kind,
        "prompt": prompt,
        "text": gen,
        "token_count": len(out_tokens),
        "repetition": repetition_score(gen),
        # fraction of output words the model copied from the prompt (echolalia)
        "prompt_echo": (len(gen_words & prompt_words) / len(gen_words)) if gen_words else 0.0,
        # fraction of the prompt's content words the continuation stays on
        "on_topic": (len(gen_words & prompt_words) / len(prompt_words)) if prompt_words else 0.0,
    }
    if kind in ("autocomplete", "fact"):
        base["fact_hit"] = bool(expected) and any(kw in gen.lower() for kw in expected)
    return base


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

    probes = probe_generation(
        engine,
        [p for _, p, _ in PROMPTS],
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        seed=SEED,
    )

    rows = []
    for (kind, prompt, expected), probe in zip(PROMPTS, probes):
        row = score(engine, prompt, kind, expected, probe.token_ids)
        row["finished_reason"] = probe.finished_reason
        rows.append(row)
        print("\n" + "=" * 70)
        print(f"[{kind}] {probe.prompt}")
        print(f"rep={row['repetition']:.3f} echo={row['prompt_echo']:.3f} "
              f"ontopic={row['on_topic']:.3f} finish={probe.finished_reason}")
        print(row["text"].strip())

    results = {"checkpoint": str(args.checkpoint), "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
