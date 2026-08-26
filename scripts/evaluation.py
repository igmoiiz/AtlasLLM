"""Evaluation — measure perplexity and run generation benchmarks."""

import argparse
import json
import math
from pathlib import Path

import torch
import yaml


def compute_perplexity(loss: float) -> float:
    """Perplexity = exp(cross_entropy_loss)."""
    return math.exp(loss)


def load_checkpoint(checkpoint_path: Path):
    """Load a model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint


def evaluate_perplexity(model, data_loader, device):
    """Compute perplexity over a dataset split."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)

            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = compute_perplexity(avg_loss)
    return avg_loss, perplexity


def run_generation_benchmark(model, tokenizer, prompts: list, max_new_tokens: int = 50):
    """Run generation on fixed prompts and return results."""
    results = []
    model.eval()

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        with torch.no_grad():
            output = model.generate(input_tensor, max_new_tokens=max_new_tokens)

        generated = tokenizer.decode(output[0].tolist())
        results.append({"prompt": prompt, "generated": generated})

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate AtlasLLM")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=Path, required=True, help="Model config YAML")
    parser.add_argument("--data", type=Path, help="Evaluation data path")
    parser.add_argument("--output", type=Path, default=Path("eval_results.json"), help="Results output path")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max generation length")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("Evaluation module — to be completed with full model implementation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")

    # Placeholder output
    results = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "status": "pending_implementation",
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
