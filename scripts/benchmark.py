"""Training-step benchmark: throughput and VRAM per batch size.

    python -m scripts.benchmark --config configs/atlasbase_v2.yaml
    python -m scripts.benchmark --config configs/atlasbase_v2.yaml \
        --batch-sizes 8 16 24 32 --steps 20 --out reports/data/hardware/bench.json

Runs the real training step (forward, cross-entropy, backward, AdamW) on random
token batches, one batch size at a time. Warmup steps are untimed. A
configuration that runs out of memory is recorded as ``oom`` and the grid
continues, which is the point of the script: find the largest batch size that
fits before starting a long run.

On Windows, CUDA can overcommit VRAM through WDDM instead of raising OOM: the
step still runs, but pages to host memory and becomes an order of magnitude
slower. A row whose peak allocation exceeds physical VRAM is flagged
``spills``; its throughput is real but not usable for training.

This measures speed and memory, not model quality. For loss and perplexity use
``training.train`` or ``harness.runner``.

Notes:
- Device, dtype, and AdamW hyperparameters come from the same places training
  uses, so the numbers transfer to a real run.
- Model weights are initialized once and reused for every batch size; the
  optimizer is re-created per configuration and released on return.
- Targets are random, not shifted text. Timing does not depend on the values,
  and the reported loss is only a sanity check that the step ran.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

from model.atlas_llm import AtlasLLM
from model.config import ModelConfig
from monitoring.system import device_info, vram_usage
from training.loss import lm_cross_entropy
from training.train import resolve_device, resolve_dtype
from training.trainer import ADAMW_BETAS, ADAMW_EPS

DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32)


def bench_step(
    model: torch.nn.Module,
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    steps: int,
    warmup: int,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    seed: int = 42,
) -> dict:
    """Time ``steps`` real training steps at one batch size.

    The batch is created once and reused, so the measurement is compute, not
    data generation. Returns ``ok=False`` and ``oom=True`` when the
    configuration does not fit; no other exception is swallowed. ``spills``
    marks a run that overcommitted VRAM without failing (Windows WDDM).
    """
    row = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_step": batch_size * seq_len,
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
    )
    try:
        torch.manual_seed(seed)
        model.zero_grad(set_to_none=True)
        model.train()
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        last_loss = float("nan")

        def step() -> None:
            nonlocal last_loss
            optimizer.zero_grad(set_to_none=True)
            loss = lm_cross_entropy(model(input_ids), targets)
            loss.backward()
            optimizer.step()
            last_loss = loss.detach().item()

        for _ in range(warmup):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        started = time.perf_counter()
        for _ in range(steps):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        usage = vram_usage(device)
        row.update(
            {
                "ok": True,
                "oom": False,
                "spills": bool(usage and usage.peak_mb > usage.total_mb),
                "final_loss": round(last_loss, 4),
                "ms_per_step": round(1000.0 * elapsed / steps, 2),
                "tokens_per_sec": round(steps * row["tokens_per_step"] / elapsed, 1),
                "allocated_mb": usage.allocated_mb if usage else 0.0,
                "reserved_mb": usage.reserved_mb if usage else 0.0,
                "peak_allocated_mb": usage.peak_mb if usage else 0.0,
                "total_mb": usage.total_mb if usage else 0.0,
            }
        )
    except torch.OutOfMemoryError:
        row.update(
            {
                "ok": False,
                "oom": True,
                "spills": None,
                "final_loss": None,
                "ms_per_step": None,
                "tokens_per_sec": None,
                "allocated_mb": None,
                "reserved_mb": None,
                "peak_allocated_mb": None,
                "total_mb": None,
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a configs/*.yaml file")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES),
        help=f"Batch sizes to try (default: {' '.join(map(str, DEFAULT_BATCH_SIZES))})",
    )
    parser.add_argument("--steps", type=int, default=10, help="Timed steps per batch size (default: 10)")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warmup steps per batch size (default: 3)")
    parser.add_argument("--seq-len", type=int, default=None, help="Override model.context_length")
    parser.add_argument("--device", default=None, help="Override config device (auto | cuda | cpu)")
    parser.add_argument("--out", type=Path, default=None, help="Write the results to this JSON file")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if any(size < 1 for size in args.batch_sizes):
        parser.error("--batch-sizes values must be positive")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = resolve_device(args.device or config.get("device", "auto"))
    dtype = resolve_dtype(config.get("dtype", "float32"))
    model_section = dict(config["model"])
    if args.seq_len is not None:
        model_section["context_length"] = args.seq_len
    model_config = ModelConfig.from_dict(model_section)

    model = AtlasLLM(model_config).to(device=device, dtype=dtype)
    n_params = sum(p.numel() for p in model.parameters())
    training = config.get("training", {})
    learning_rate = float(training.get("learning_rate", 1e-3))
    weight_decay = float(training.get("weight_decay", 0.1))
    info = device_info(device)

    print(
        f"device={device} dtype={str(dtype).replace('torch.', '')} "
        f"params={n_params:,} seq_len={model_config.context_length}"
    )
    print(f"hardware: {info.name} | {info.total_memory_mb:,.0f} MiB | torch {info.torch_version}")
    print(f"{'batch':>6} {'tok/step':>9} {'ms/step':>9} {'tok/s':>10} {'peak MiB':>9} {'reserved MiB':>13}")

    rows = []
    for batch_size in args.batch_sizes:
        row = bench_step(
            model,
            batch_size=batch_size,
            seq_len=model_config.context_length,
            vocab_size=model_config.vocab_size,
            steps=args.steps,
            warmup=args.warmup,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=int(training.get("seed", 42)),
        )
        rows.append(row)
        if row["ok"]:
            print(
                f"{batch_size:>6} {row['tokens_per_step']:>9,} {row['ms_per_step']:>9.2f} "
                f"{row['tokens_per_sec']:>10,.0f} {row['peak_allocated_mb']:>9.1f} "
                f"{row['reserved_mb']:>13.1f}{'  SPILL' if row['spills'] else ''}"
            )
        else:
            print(f"{batch_size:>6} {'-':>9} {'-':>9} {'-':>10} {'OOM':>9} {'-':>13}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if any(row["spills"] for row in rows if row["ok"]):
        print("\nSPILL: peak allocation exceeded physical VRAM; Windows pages to host memory, throughput not usable.")

    artifact = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": str(args.config),
        "device": info.to_dict(),
        "dtype": str(dtype).replace("torch.", ""),
        "parameters": n_params,
        "seq_len": model_config.context_length,
        "vocab_size": model_config.vocab_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "results": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    sys.exit(main())
