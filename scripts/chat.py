"""Interactive chat REPL for a trained AtlasLLM checkpoint.

    python -m scripts.chat --checkpoint <ckpt> --config <config.yaml>
    python -m scripts.chat --checkpoint <ckpt> --safety [--policy safety/policy.yaml]

Each prompt is completed with the KV cache and streamed token-by-token. With
``--safety`` the prompt and the response pass through the policy filters, which
means the response is generated in one piece (output filtering cannot retract
already-printed tokens). Type ``quit`` / ``exit`` (or Ctrl-C) to leave. Empty
lines are skipped.
"""

import argparse
import logging
import sys

from inference.engine import InferenceEngine
from safety.guardrails import Guardrails
from safety.policy import load_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a trained AtlasLLM checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a training checkpoint (.pt)")
    parser.add_argument("--config", help="Training config YAML (auto-loaded from the checkpoint when omitted)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, help="Seed for reproducible sampling")
    parser.add_argument(
        "--no-repeat-ngram-size", type=int, default=4,
        help="Ban ids that would repeat an n-gram of this size (default 4; 0 disables)",
    )
    parser.add_argument("--safety", action="store_true", help="Filter prompts and responses through the safety policy")
    parser.add_argument("--policy", help="Policy YAML (default: safety/policy.yaml)")
    args = parser.parse_args()

    engine = InferenceEngine.from_checkpoint(args.checkpoint, config=args.config, tokenizer=None, device="auto")
    guardrails = None
    if args.safety:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
        guardrails = Guardrails(engine, load_policy(args.policy) if args.policy else None)
        print(f"safety: on ({len(guardrails.policy.input_rules)} input rules, "
              f"{len(guardrails.policy.output_rules)} output rules)")

    print(f"AtlasLLM chat — q to quit (model: {type(engine.model).__name__})")
    while True:
        try:
            prompt = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt.strip():
            continue
        if prompt.strip().lower() in {"quit", "exit", "q"}:
            break

        generate_kwargs = {
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "seed": args.seed,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        }
        sys.stdout.write("atlas> ")
        if guardrails is not None:
            guarded = guardrails.complete(prompt, **generate_kwargs)
            print(guarded.text.strip())
            for decision in (guarded.input, guarded.output):
                if decision is not None and decision.action == "warn":
                    print(f"  [safety warning: {decision.reason}]")
            continue
        for chunk in engine.stream(prompt, **generate_kwargs):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    main()
