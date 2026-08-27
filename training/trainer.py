"""Training loop orchestration.

Drives the full per-step cycle:

    data -> model -> loss -> backward -> clip -> optimizer -> scheduler
        -> logging (console + JSONL [+ TensorBoard]) -> checkpointing

The loop owns the optimizer/scheduler machinery and never touches model
architecture internals; model and dataloaders are passed in ready to run.
"""

import json
import time
from pathlib import Path

import torch

from training.checkpoint import load_checkpoint, save_checkpoint
from training.loss import lm_cross_entropy
from training.scheduler import build_lr_scheduler

ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8


class Trainer:
    """Run a fixed number of training steps over one DataLoader.

    Args:
        model: AtlasLLM (or any ``Module`` returning ``[B, T, V]`` logits).
        config: ``configs/*.yaml`` structure; reads ``training``, ``logging``
            and ``checkpoint`` sections (each key optional with a default).
        train_loader: batches of ``{"input_ids", "targets"}`` ``[B, T]``.
        val_loader: same shape; ``shuffle=False``.
        device: compute device for model and batches.
        seed: base seed; train DataLoader's ``generator`` is reseeded as
            ``seed + epoch`` each epoch for reproducible shuffling.
        run_dir: directory for ``last.pt``, ``best.pt`` and ``metrics.jsonl``.
    """

    def __init__(self, model, config: dict, train_loader, val_loader, device: torch.device, seed: int, run_dir: str | Path):
        training = config.get("training", {})
        logging = config.get("logging", {})
        self.model = model
        self.config = config
        self.device = device
        self.seed = int(seed)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = int(training["max_steps"])
        self.grad_clip = float(training.get("grad_clip", 1.0))
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training.get("weight_decay", 0.1)),
            betas=ADAMW_BETAS,
            eps=ADAMW_EPS,
        )
        self.scheduler = build_lr_scheduler(
            self.optimizer,
            warmup_steps=int(training.get("warmup_steps", 0)),
            max_steps=self.max_steps,
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.log_every = max(1, int(logging.get("log_every", 10)))
        self.val_every = int(logging.get("val_every", 0))
        self.max_val_batches = int(logging.get("max_val_batches", 200))
        self.save_every = int(logging.get("save_every", 0))
        self.writer = None
        if logging.get("tensorboard", False):
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))

    def _write(self, record: dict) -> None:
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if self.writer is not None:
            for key, value in record.items():
                self.writer.add_scalar(key, value, record.get("step", 0))

    def evaluate(self) -> float:
        """Mean validation loss (no gradients), capped at ``max_val_batches``."""
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                if count >= self.max_val_batches:
                    break
                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)
                total += lm_cross_entropy(self.model(input_ids), targets).item()
                count += 1
        self.model.train()
        return total / max(count, 1)

    def train(self, resume_from: str | Path | None = None) -> dict:
        """Run the loop (optionally resumed) and return a summary."""
        start_step = 0
        first_loss = None
        if resume_from is not None:
            ckpt = load_checkpoint(resume_from, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler)
            start_step = ckpt.step + 1
            first_loss = ckpt.metrics.get("train_loss")
            if start_step >= self.max_steps:
                print(f"[trainer] checkpoint at step {ckpt.step} is already at max_steps={self.max_steps}; nothing to do")
                return {
                    "train_loss_first": first_loss,
                    "train_loss_last": ckpt.metrics.get("train_loss"),
                    "best_val_loss": ckpt.metrics.get("best_val_loss"),
                    "last_checkpoint": str(resume_from),
                    "best_checkpoint": None,
                }

        token_count = self.train_loader.batch_size * self.train_loader.dataset.context_length
        epoch = 0
        data_iter = iter(self.train_loader)
        best_val = None
        last_loss = 0.0
        t_section = time.perf_counter()
        steps_in_section = 0

        for step in range(start_step, self.max_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if self.train_loader.generator is not None:
                    self.train_loader.generator.manual_seed(self.seed + epoch)
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss = lm_cross_entropy(self.model(input_ids), targets)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            last_loss = loss.detach().item()
            if first_loss is None:
                first_loss = last_loss
            steps_in_section += 1

            if step % self.log_every == 0:
                now = time.perf_counter()
                tokens_per_sec = steps_in_section * token_count / max(now - t_section, 1e-9)
                t_section, steps_in_section = now, 0
                lr = self.optimizer.param_groups[0]["lr"]
                gpu_mb = torch.cuda.memory_allocated(self.device) / 2**20 if self.device.type == "cuda" else 0.0
                record = {
                    "step": step,
                    "train_loss": last_loss,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "tokens_per_sec": round(tokens_per_sec, 1),
                    "gpu_memory_mb": round(gpu_mb, 1),
                }
                print(
                    f"step {step}: loss={last_loss:.4f} lr={lr:.2e} "
                    f"grad={float(grad_norm):.3f} tok/s={round(tokens_per_sec)} "
                    f"gpu={round(gpu_mb, 1)}MB"
                )
                self._write(record)

            if self.val_every and step % self.val_every == 0:
                val_loss = self.evaluate()
                print(f"  val_loss={val_loss:.4f}")
                self._write({"step": step, "val_loss": val_loss})
                if best_val is None or val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        self.run_dir / "best.pt",
                        model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                        step=step, config=self.config, metrics={"val_loss": val_loss},
                    )

            if self.save_every and step % self.save_every == 0:
                save_checkpoint(
                    self.run_dir / "last.pt",
                    model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                    step=step, config=self.config, metrics={"train_loss": last_loss, "best_val_loss": best_val},
                )

        save_checkpoint(
            self.run_dir / "last.pt",
            model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            step=self.max_steps - 1, config=self.config, metrics={"train_loss": last_loss, "best_val_loss": best_val},
        )
        if self.writer is not None:
            self.writer.close()
        return {
            "train_loss_first": first_loss if first_loss is not None else last_loss,
            "train_loss_last": last_loss,
            "best_val_loss": best_val,
            "last_checkpoint": str(self.run_dir / "last.pt"),
            "best_checkpoint": str(self.run_dir / "best.pt") if best_val is not None else None,
        }
