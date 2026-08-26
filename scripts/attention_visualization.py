"""Attention visualization — visualize attention patterns from a trained checkpoint."""

import argparse
from pathlib import Path

import torch
import yaml


def load_attention_weights(checkpoint_path: Path, layer_idx: int = 0):
    """Load attention weight matrices from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model", checkpoint)

    # Extract attention projection weights for the specified layer
    prefix = f"transformer_blocks.{layer_idx}.attention"
    weights = {}
    for key, value in model_state.items():
        if prefix in key:
            weights[key] = value

    return weights


def compute_attention_patterns(weights: dict, layer_idx: int):
    """Compute attention score patterns for visualization."""
    # Placeholder — will be implemented with actual model forward pass
    # For now, return weight matrices for inspection
    print(f"Layer {layer_idx} attention weights:")
    for key, tensor in weights.items():
        print(f"  {key}: {tensor.shape}")
    return weights


def save_attention_plot(patterns, output_path: Path, layer_idx: int):
    """Save attention pattern as an image."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"Layer {layer_idx} — Attention Pattern")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {output_path}")
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")


def main():
    parser = argparse.ArgumentParser(description="Visualize attention patterns")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to visualize")
    parser.add_argument("--output", type=Path, default=Path("attention_plot.png"), help="Output image path")
    parser.add_argument("--config", type=Path, default=None, help="Model config YAML (optional)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return

    weights = load_attention_weights(args.checkpoint, args.layer)
    if not weights:
        print(f"No attention weights found for layer {args.layer}")
        return

    patterns = compute_attention_patterns(weights, args.layer)
    save_attention_plot(patterns, args.output, args.layer)


if __name__ == "__main__":
    main()
