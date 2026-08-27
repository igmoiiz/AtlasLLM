"""One-shot generation CLI.

    python -m inference.generate --checkpoint <ckpt> --config <config.yaml> \
        --prompt "Once upon a time" [--max-tokens 200] [--temperature 0.8]

The tokenizer is taken from the ``--config`` ``data.tokenizer_path``; the
architecture comes from the ``model:`` section (checkpoints written during
Stage 6+ also carry their config, so ``--config`` can be omitted then).
"""

import argparse

from inference.engine import InferenceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a completion from a trained AtlasLLM checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a training checkpoint (.pt)")
    parser.add_argument("--config", help="Training config YAML (auto-loaded from the checkpoint when omitted)")
    parser.add_argument("--prompt", help="Prompt text (read from stdin when omitted)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, help="Seed for reproducible sampling")
    parser.add_argument("--no-cache", action="store_true", help="Disable the KV cache (recompute the full prefix each step)")
    args = parser.parse_args()

    engine = InferenceEngine.from_checkpoint(
        args.checkpoint,
        config=args.config,
        tokenizer=None,
        device="auto",
    )

    prompt = args.prompt if args.prompt is not None else input("Prompt: ")
    output = engine.generate(
        prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        use_cache=not args.no_cache,
    )
    print(output.text)


if __name__ == "__main__":
    main()
