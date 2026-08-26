"""Training analysis — parse training logs and generate plots."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_tensorboard_logs(log_dir: Path) -> dict:
    """Load training metrics from TensorBoard log directory."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        ea = EventAccumulator(str(log_dir))
        ea.Reload()

        metrics = {}
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            metrics[tag] = {
                "steps": [e.step for e in events],
                "values": [e.value for e in events],
            }
        return metrics
    except ImportError:
        print("tensorboard not installed. Install with: pip install tensorboard")
        return {}
    except Exception as e:
        print(f"Error loading TensorBoard logs: {e}")
        return {}


def load_json_logs(log_path: Path) -> dict:
    """Load training metrics from a JSON log file."""
    metrics = {"steps": [], "train_loss": [], "val_loss": [], "lr": []}
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            metrics["steps"].append(entry.get("step", 0))
            metrics["train_loss"].append(entry.get("train_loss", 0))
            metrics["val_loss"].append(entry.get("val_loss", 0))
            metrics["lr"].append(entry.get("lr", 0))
    return metrics


def plot_training_curves(metrics: dict, output_path: Path):
    """Plot training loss, validation loss, and learning rate."""
    has_tb = "steps" in metrics and "train_loss" in metrics
    has_json = "steps" in metrics and len(metrics["steps"]) > 0

    if not has_tb and not has_json:
        print("No valid metrics found to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1 = axes[0]
    if "train_loss" in metrics and metrics["train_loss"]:
        ax1.plot(metrics["steps"], metrics["train_loss"], label="Train Loss", alpha=0.8)
    if "val_loss" in metrics and metrics["val_loss"]:
        ax1.plot(metrics["steps"], metrics["val_loss"], label="Val Loss", alpha=0.8)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Learning rate plot
    ax2 = axes[1]
    if "lr" in metrics and metrics["lr"]:
        ax2.plot(metrics["steps"], metrics["lr"], label="Learning Rate", color="green")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and plot training metrics")
    parser.add_argument("--log-dir", type=Path, help="TensorBoard log directory")
    parser.add_argument("--log-file", type=Path, help="JSON lines log file")
    parser.add_argument("--output", type=Path, default=Path("training_curves.png"), help="Output plot path")
    args = parser.parse_args()

    if not args.log_dir and not args.log_file:
        print("Error: provide --log-dir or --log-file")
        return

    if args.log_dir:
        metrics = load_tensorboard_logs(args.log_dir)
    else:
        metrics = load_json_logs(args.log_file)

    plot_training_curves(metrics, args.output)


if __name__ == "__main__":
    main()
