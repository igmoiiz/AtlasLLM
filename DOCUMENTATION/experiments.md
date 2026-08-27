# Experiments (Design)

AtlasLLM experiment system - configuration-driven, reproducible comparisons.

> **Status: PLANNED.** The comparison workflow below is the design. Today the
> repository has three working configurations - `configs/debug.yaml`,
> `configs/small.yaml`, `configs/medium.yaml` - and no `configs/experiments/`
> or `scripts/evaluation.py` results yet. Any table of measured results in the
> old version of this page was hypothetical and has been removed.

## For a beginner

A language model has many design dials: how many layers, how wide, how long a context, which word-prediction strategy. The experiment system lets you turn one dial at a time and measure the effect, so you learn *why* a model behaves the way it does instead of guessing.

The rule of the game: change one thing, run the identical training recipe, compare the identical metrics. If layer count 6 beats layer count 4 on the same data, that is evidence. If you changed two things at once, you know nothing.

Anything below this line is the technical design. See [index.md](index.md) for the project overview.

## Overview

Every experiment is defined by a YAML configuration. No source code changes between experiments.

```text
configs/
├── debug.yaml              # Minimal debug config
├── small.yaml              # Canonical small model
├── medium.yaml             # Larger model
└── experiments/            # Planned:
    ├── depth.yaml          # Varying number of layers
    ├── width.yaml          # Varying hidden dimension
    ├── context.yaml        # Varying sequence length
    ├── heads.yaml          # Varying attention heads
    ├── positional.yaml     # Varying positional encoding
    └── activation.yaml     # Varying activation function
```

## Experiment Categories

Changing a single dimension while keeping everything else constant.

### Depth

| Config | n_layers | Everything else |
|--------|----------|-----------------|
| depth_2 | 2 | same as small |
| depth_4 | 4 | same as small |
| depth_6 | 6 | same as small (== small.yaml) |
| depth_8 | 8 | same as small |

**Hypothesis:** Deeper models capture more complex patterns but are harder to train and slower.

**Measure:** validation loss, parameter count, training speed.

### Width

| Config | d_model | n_heads | Everything else |
|--------|---------|---------|-----------------|
| width_128 | 128 | 4 | same as small |
| width_256 | 256 | 8 | same as small (== small.yaml) |
| width_384 | 384 | 8 | same as small |
| width_512 | 512 | 8 | same as small |

**Hypothesis:** Wider models have more capacity but diminishing returns at small scale.

### Context Length

| Config | context_length | batch_size | tokens/step |
|--------|---------------|------------|-------------|
| ctx_128 | 128 | 16 | 2048 |
| ctx_256 | 256 | 8 | 2048 |
| ctx_512 | 512 | 4 | 2048 |

**Hypothesis:** Longer context improves coherence but increases VRAM and slows training.

> Note: batch_size is chosen to keep tokens/step (batch × context) constant, so
> the comparison isolates context length rather than total compute per step.

### Attention Heads

| Config | n_heads | head_dim |
|--------|---------|----------|
| heads_4 | 4 | 64 |
| heads_8 | 8 | 32 |
| heads_16 | 16 | 16 |

**Hypothesis:** More heads allow attending to different relationship types.

### Positional Encoding

Changing `model/positional_encoding.py` behind a config switch is future work;
today only learned positional embeddings exist.

| Config | Type |
|--------|------|
| pos_learned | Learned embeddings (current) |
| pos_sinusoidal | Sinusoidal (fixed) - planned |
| pos_rope | Rotary Position Embeddings - planned |

**Hypothesis:** RoPE generalizes better to longer sequences than learned embeddings.

### Activation

Changing `model/feed_forward.py` behind a config switch is future work; today
only GELU exists.

| Config | Activation |
|--------|-----------|
| act_gelu | GELU (current) |
| act_swiglu | SwiGLU - planned |

**Hypothesis:** SwiGLU improves performance with similar parameter count.

## Running Experiments

```bash
# Run a single experiment
python -m training.train --config configs/experiments/depth_4.yaml

# Run all depth experiments
for f in configs/experiments/depth_*.yaml; do
    python -m training.train --config "$f"
done
```

Each run writes to its own `checkpoints/run_<timestamp>/` directory.

## Experiment Tracking

Each experiment produces:

```text
checkpoints/run_<timestamp>/
├── last.pt
├── best.pt
├── config.yaml
├── metrics.jsonl
└── reproducibility.json
```

The `config.yaml` and `reproducibility.json` make each run self-describing (seed, versions, hyperparameters). See [training.md](training.md).

## Comparing Results

Comparison is planned to use a table like this (placeholders only - populate from real `metrics.jsonl` files):

| Experiment | Params | Best Val Loss | Perplexity | Tokens/sec | VRAM |
|-----------|--------|---------------|------------|------------|------|
| small | 12,982,784 | (from run) | (from run) | ~26,000 | ~217 MB |
| depth_8 | (measure) | (from run) | (from run) | (measure) | (measure) |

Perplexity = `exp(val_loss)`. This table must only ever contain measured values from real runs (see [AGENTS.md](../AGENTS.md) rule 48).

## Experiment Integrity

Rules:

1. **Never modify results manually** - Metrics are generated by code
2. **Record failed experiments** - Don't cherry-pick only successes
3. **Document hypothesis** - What were you testing?
4. **Document interpretation** - What did the results mean?
5. **Reproducible** - Same config + same seed = same results

## Implementation checklist

- [ ] Create `configs/experiments/` variants (depth, width, context, heads)
- [ ] Add positional-encoding config switch (sinusoidal/RoPE)
- [ ] Add activation config switch (SwiGLU)
- [ ] Implement `scripts/evaluation.py` to emit a real comparison table

## Related documentation

- [index.md](index.md) - documentation entry point
- [harness.md](harness.md) - behavioral comparison across model versions
- [hardware.md](hardware.md) - what the GTX 1070 allows before these experiments
- [training.md](training.md) - what each experiment records for reproducibility
- [CONTEXT.md](../CONTEXT.md) - original experiment requirements (sections 63-64)