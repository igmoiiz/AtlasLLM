"""Evaluation CLI: measure a trained checkpoint with Stage-8 metrics.

    python -m scripts.evaluation --checkpoint checkpoints/wikitext103/run_<ts>/last.pt --config configs/wikitext103.yaml

Reports:
  - validation loss + perplexity        (evaluation.perplexity)
  - train-vs-heldout memorization gap   (evaluation.memorization)
  - generation probes on fixed prompts  (evaluation.generation_eval)

The run dir's ``config.yaml`` (written by the trainer) is used when ``--config``
is omitted.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from data_pipeline.dataset import TextDataset
from evaluation.generation_eval import probe_generation
from evaluation.memorization import evaluate_memorization
from evaluation.perplexity import evaluate_loss, perplexity
from inference.engine import InferenceEngine

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Once upon a time,",
    "Artificial intelligence is",
    "In 1969, humans first",
    "The Earth orbits the Sun because",
]


def _load_config(config_arg: str | None, checkpoint: Path) -> dict:
    if config_arg:
        return yaml.safe_load(Path(config_arg).read_text(encoding="utf-8"))
    # Trainer-run dirs carry their own config.yaml; the checkpoint itself holds
    # a stashed "config" dict used for things like model sizing when that is
    # missing (pre-Stage-7 checkpoints).
    run_config = Path(checkpoint).parent / "config.yaml"
    if run_config.is_file():
        return yaml.safe_load(run_config.read_text(encoding="utf-8"))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return ckpt.get("config")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to a training checkpoint (.pt)")
    parser.add_argument("--config", default=None, help="Path to a configs/*.yaml file (default: run-dir config.yaml)")
    parser.add_argument("--max-batches", type=int, default=200, help="Cap on evaluated batches per loader (0 = all)")
    parser.add_argument("--max-tokens", type=int, default=120, help="Generated tokens per prompt probe")
    parser.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS, help="Prompts to probe (default: standard suite)")
    parser.add_argument("--out", type=Path, default=None, help="Write results to this JSON file")
    args = parser.parse_args()

    config = _load_config(args.config, args.checkpoint)
    model_cfg = config["model"]
    ctx = int(model_cfg["context_length"])
    batch_size = int(config.get("training", {}).get("batch_size", 8))

    engine = InferenceEngine.from_checkpoint(args.checkpoint, config=config, tokenizer=None, device="auto")
    device = engine.device

    loader_kwargs = {"batch_size": batch_size, "shuffle": False}
    val_loader = torch.utils.data.DataLoader(TextDataset(config["data"]["val_path"], ctx), **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(TextDataset(config["data"]["test_path"], ctx), **loader_kwargs)
    train_loader = torch.utils.data.DataLoader(TextDataset(config["data"]["train_path"], ctx), **loader_kwargs)

    max_batches = args.max_batches if args.max_batches > 0 else None
    val_loss = evaluate_loss(engine.model, val_loader, device, max_batches)
    test_loss = evaluate_loss(engine.model, test_loader, device, max_batches)
    mem = evaluate_memorization(engine.model, train_loader, val_loader, device, max_batches)

    print(f"device={device} ctx={ctx} batch={batch_size}")
    print(f"val loss : {val_loss:.4f}  ppl={perplexity(val_loss):.2f}")
    print(f"test loss: {test_loss:.4f}  ppl={perplexity(test_loss):.2f}")
    print(f"memorization gap (train vs val): {mem['gap']:.4f} nats "
          f"(train {mem['train_loss']:.4f}, val {mem['heldout_loss']:.4f})")

    probes = probe_generation(engine, list(args.prompts), max_new_tokens=args.max_tokens)
    print("\ngeneration probes:")
    for p in probes:
        print(f"\n  > {p.prompt}")
        print(f"  [{p.finished_reason}, {len(p.token_ids)} tokens] {p.text.strip()}")

    results = {
        "checkpoint": str(args.checkpoint),
        "val_loss": val_loss,
        "val_perplexity": perplexity(val_loss),
        "test_loss": test_loss,
        "test_perplexity": perplexity(test_loss),
        "memorization": {"train_loss": mem["train_loss"], "heldout_loss": mem["heldout_loss"], "gap": mem["gap"]},
        "generation": [{"prompt": p.prompt, "text": p.text, "finished_reason": p.finished_reason, "tokens": len(p.token_ids)} for p in probes],
    }
    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
