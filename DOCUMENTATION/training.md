# Training Pipeline

AtlasLLM training system - from data loading to checkpointing.

## For a beginner

Training teaches the model by example. For every token in the training text, the model guesses what comes next, and the loss function measures how wrong that guess was. The optimizer then nudges the model's internal numbers in the direction that would have made the guess better. Repeating this millions of times across billions of tokens is what produces a language model.

Key ideas:

- **Loss** - one number that says how bad today's predictions were. It starts near a floor (for a 16k vocabulary, uniform guessing gives loss about 9.68) and drops as the model learns.
- **Learning rate** - how large each correction step is. The run warms up slowly, then holds or decays (cosine or WSD schedule), so early steps don't overshoot and later steps settle cleanly.
- **Checkpointing** - a snapshot of the model, optimizer, and schedule so training can be paused, resumed, or interrupted without losing work. The trainer writes atomic checkpoints (write to `.tmp`, then rename) so a crash cannot corrupt them.
- **Validation** - periodically measuring loss on text the model never trained on. If training loss drops but validation loss rises, the model is memorizing instead of learning.

Anything below this line is the technical description. See [index.md](index.md) for the full picture.

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
     ├── Scheduler step (warmup + cosine/wsd decay)
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

**AdamW** with these defaults (the small config; each config sets its own):

| Parameter | Value |
|-----------|-------|
| learning_rate | 3e-4 |
| weight_decay | 0.1 |
| betas | (0.9, 0.95) |
| eps | 1e-8 |

## Learning Rate Schedule

Two schedules are available via `training.schedule`; both start with the same
linear warmup and both end at zero.

**`cosine` (default)**

```
Step 0          → warmup_steps      → linear warmup from 0 to max_lr
warmup_steps   → max_steps         → cosine decay from max_lr to ~0

lr = max_lr * 0.5 * (1 + cos(π * progress))
```

**`wsd`** (warmup-stable-decay, planned for AtlasBase-v3)

```
Step 0                 → warmup_steps   → linear warmup from 0 to max_lr
warmup_steps          → decay_start    → flat at max_lr
decay_start           → max_steps      → cosine decay from max_lr to ~0

decay_start = max_steps * (1 - training.decay_frac)   # default decay_frac = 0.1
```

`progress` goes from 0 to 1 over the decay phase in both cases.

Why WSD: the plateau holds the LR at `max_lr` for 90% of the run, so a
checkpoint taken mid-run can be extended (more tokens, more data) and still
receive a full anneal at the end. Under cosine, extending a finished run means
restarting from an LR that already decayed to zero, which is a much worse
starting point. The tail is what actually buys the final loss, which is why
`decay_frac` stays small.

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
model, optimizer, and scheduler. The stashed `config` is the full run config,
so inference (`InferenceEngine.from_checkpoint`) can rebuild the model and
tokenizer without being told the config explicitly (checkpoints trained
before Stage 7 stored an empty config and need `--config`). Saves write to a
`.tmp` then rename, so a crash cannot truncate a checkpoint. An
interrupted/converged resume is a
no-op with a clear message.

**Checkpoint resumption:** `python -m training.train --config ... --resume <path>.pt`
restores all state and continues from `step + 1` toward `max_steps`.

**Recovery after an interruption** (power cut, Ctrl-C, overnight crash):

```bash
python -m training.train --config configs/atlasbase_v2.yaml --resume auto
```

`auto` scans `checkpoint.dir/run_*/` for `last.pt` / `best.pt`, newest first,
and resumes from the newest file that actually deserializes. A checkpoint
truncated mid-write is skipped, so recovery falls back to the previous good
save instead of failing. If nothing is readable it raises rather than silently
starting a fresh run; pass `--resume` explicitly or omit it to train from
scratch. The resolved path is recorded in `reproducibility.json` as
`resume_from`.

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
- **Console** - one line per `logging.log_every` steps
- **`metrics.jsonl`** - machine-readable, one JSON object per logged step
- **TensorBoard** - optional, enabled with `logging.tensorboard: true`

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
loss `7.27 -> 0.039` over 300 steps - full dataset → tokenizer → model → loss
→ optimizer → backward path proven before pretraining.

## Measured performance

| Run | Params | Tokens | Throughput | VRAM peak | Best val loss |
|-----|--------|--------|-----------|-----------|---------------|
| AtlasBase-v1 | 13M | 186M | ~26,000 tok/s | ~217 MB | 5.4174 |
| AtlasBase-v2 | 48M | 186M | 9,091 tok/s | 2,143 MiB | 4.8282 |

Both runs trained 95k steps on the same 186M-token corpus with the same 16k
tokenizer, so the difference is model capacity. The v2 throughput comes from a
real training step at batch 8 x context 256 (225 ms/step,
`reports/data/hardware/bench_atlasbase_v2.json`); the batch-size sweep and the
Windows VRAM caveat are in [hardware.md](hardware.md). The full comparison table
is in [experiments.md](experiments.md).

Full hardware context in [hardware.md](hardware.md).

## Related documentation

- [index.md](index.md) - documentation entry point
- [hardware.md](hardware.md) - device strategy and memory budget
- [dataset.md](dataset.md) - where the training batches come from
- [architecture.md](architecture.md) - what the optimizer is training
- [tokenizer.md](tokenizer.md) - how text is tokenized first
- [inference.md](inference.md) - how the trained checkpoint is used
- [training/train.py](../training/train.py) - training entry point
