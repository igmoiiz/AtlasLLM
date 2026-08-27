# Training Pipeline

AtlasLLM training system — from data loading to checkpointing.

## Overview

```
Config YAML
     ↓
Load Dataset (tokenized)
     ↓
Create DataLoader (batch, sequence packing)
     ↓
Training Loop:
     ├── Forward pass
     ├── Loss calculation (cross-entropy)
     ├── Backward pass (gradients)
     ├── Gradient clipping
     ├── Optimizer step (AdamW)
     ├── Scheduler step (warmup + cosine decay)
     └── Logging + Checkpointing
     ↓
Validation
     ↓
Metrics (loss, perplexity, throughput)
```

## Loss Function

**Autoregressive cross-entropy:**

For input sequence `x_1, x_2, ..., x_T`, the model learns to predict the next token:

```
Input:  x_1, x_2, ..., x_{T-1}
Target: x_2, x_3, ..., x_T
```

The loss at each position is:

```
L = -log(P(x_{t+1} | x_1, ..., x_t))
```

**Padding handling:** Padding positions are excluded from loss calculation via a mask.

**Total loss:** Mean over all non-padding positions.

## Optimizer

**AdamW** with the following defaults:

| Parameter | Value |
|-----------|-------|
| learning_rate | 3e-4 |
| weight_decay | 0.1 |
| betas | (0.9, 0.95) |
| eps | 1e-8 |

## Learning Rate Schedule

```
Step 0          → warmup_steps      → linear warmup from 0 to max_lr
warmup_steps   → max_steps         → cosine decay from max_lr to ~0
```

The cosine schedule decays the learning rate following:

```
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
```

where `progress` goes from 0 to 1 over the decay phase.

## Gradient Management

- **Gradient clipping:** Global norm clipping (default `max_norm=1.0`)
- **Gradient norm logging:** Tracked at every log step
- **Gradient accumulation:** Not yet implemented; batch size is the primary
  knob. Revisit if VRAM forces a smaller batch than is effective.

## Checkpointing

Checkpoints are single files written by `training/checkpoint.py`:

```
checkpoints/<...>/run_<timestamp>/
├── last.pt    # Most recent state
├── best.pt    # Lowest validation loss seen so far (when validation is on)
├── metrics.jsonl       # Per-log-step metric records
├── config.yaml         # Exact config used for the run
└── reproducibility.json # Seed, versions, hardware, dataset sizes (rule 25)
```

Each `.pt` bundles `step`, `config`, `metrics`, and the state dicts of the
model, optimizer, and scheduler. Saves write to a `.tmp` then rename, so a
crash cannot truncate a checkpoint. An interrupted/converged resume is a
no-op with a clear message.

**Checkpoint resumption:** `python -m training.train --config ... --resume <path>.pt`
restores all state and continues from `step + 1` toward `max_steps`.

## Validation

- Run every N steps (configurable)
- Compute validation loss on held-out data
- Track best validation loss for early stopping
- Never compute gradients during validation

## Logging

Metrics logged at each step:

| Metric | Description |
|--------|-------------|
| step | Current training step |
| train_loss | Training loss for the batch |
| val_loss | Validation loss (periodic) |
| lr | Current learning rate |
| grad_norm | Global gradient norm |
| tokens_per_sec | Training throughput |
| gpu_memory | GPU VRAM usage |

Logging targets:
- **Console** — one line per `logging.log_every` steps
- **`metrics.jsonl`** — machine-readable, one JSON object per logged step
- **TensorBoard** — optional, enabled with `logging.tensorboard: true`

Validation loss is capped at `logging.max_val_batches` batches for speed.

## Usage

```bash
python -m training.train --config configs/debug.yaml
python -m training.train --config configs/debug.yaml --resume checkpoints/debug/run_20260101-000000/last.pt
python -m training.train --config configs/small.yaml --steps 500   # override max_steps
```

## Reproducibility

Every training run records:

- Random seed
- Model configuration
- Dataset version
- Tokenizer version
- PyTorch version
- CUDA version
- Hardware info
- Training hyperparameters

## Debug Mode

Before GPU training, verify the pipeline with `configs/debug.yaml`:

```yaml
batch_size: 2
context_length: 32
d_model: 64
n_layers: 2
n_heads: 4
max_steps: 100
```

The model must successfully:

1. Forward pass
2. Backward pass
3. Optimizer step
4. Checkpoint save
5. Checkpoint reload

Before scaling to full training.

## Stage 5 Verification

Debug run on GPU (`configs/debug.yaml`, 100 steps, debug tokenizer/model):
train loss `7.29 -> 5.04` (uniform floor for a 1280-vocab char LM is `ln 1280 ≈ 7.16`);
warmup + cosine schedule visible; best val `4.84`; checkpoints + resume verified.

Tiny-overfit test (AGENTS.md rule 24), 512-token stream memorized on GPU:
loss `7.27 -> 0.039` over 300 steps — full dataset → tokenizer → model → loss
→ optimizer → backward path proven before pretraining.
