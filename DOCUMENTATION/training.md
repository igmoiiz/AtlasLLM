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

- **Gradient clipping:** Global norm clipping (default max_norm=1.0)
- **Gradient accumulation:** For effective larger batch sizes on limited VRAM
- **Gradient norm logging:** Tracked at every log step

## Checkpointing

Every N steps, save:

```
checkpoints/step_XXXXX/
├── model.pt          # Model state dict
├── optimizer.pt      # Optimizer state dict
├── scheduler.pt      # Scheduler state dict
├── config.yaml       # Training configuration
├── metrics.json      # Current metrics
└── step.txt          # Step number
```

**Checkpoint resumption:** Training can be resumed from any checkpoint by loading all saved states.

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
- **TensorBoard** — for visualization
- **Console** — periodic summary
- **JSON file** — for post-hoc analysis

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
