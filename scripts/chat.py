"""Interactive chat REPL for a trained AtlasLLM checkpoint.

    python -m scripts.chat --checkpoint <ckpt> --config <config.yaml>

Each prompt is completed with the KV cache and streamed token-by-token.
Type ``quit`` / ``exit`` (or Ctrl-C) to leave. Empty lines are skipped.
"""

import argparse
import sys

from inference.engine import InferenceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a trained AtlasLLM checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a training checkpoint (.pt)")
    parser.add_argument("--config", help="Training config YAML (auto-loaded from the checkpoint when omitted)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, help="Seed for reproducible sampling")
    args = parser.parse_args()

    engine = InferenceEngine.from_checkpoint(args.checkpoint, config=args.config, tokenizer=None, device="auto")

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
        sys.stdout.write("atlas> ")
        for chunk in engine.stream(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        ):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    main()
