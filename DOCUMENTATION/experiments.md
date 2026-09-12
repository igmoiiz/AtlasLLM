# Experiments (Design)

AtlasLLM experiment system - configuration-driven, reproducible comparisons.

> **Status: INITIAL RESULTS.** The comparison workflow below is the design; the
> experiment-grid configs (`configs/experiments/`) are not built yet. What IS
> measured today: one real overfit study (WikiText-2 vs WikiText-103, same model)
> with real numbers below - a concrete "what not to do" versus "healthy" pair.
> `scripts/evaluation.py` now emits these metrics for any checkpoint.

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

The overfit study is the first controlled (same-model, different-corpus) comparison:

| Run | Data | Train Tokens | Final Train Loss | Best Val Loss | Memo. Gap (val-train) | Val Perplexity |
|-----|------|-------------|------------------|---------------|----------------------|----------------|
| `run_20260827-231105/last.pt` | WikiText-2 | 2.1M | 3.68 | 7.49 @ 100k | +3.8 nats | exp(7.49) ≈ 1787 |
| `run_20260905-215250/last.pt` | WikiText-103 | 108M | 5.18 | 5.16 @ 100k | -0.02 nats | exp(5.16) ≈ 174 |
| `tinystories/run_20260912-192907/last.pt` | AtlasTiny | 22.9M | 5.80 | 5.93 @ 8k | +0.00 nats | exp(5.93) ≈ 380 |
| `tinystories-overfit/run_20260912-194102/last.pt` | AtlasTiny slice (300k) | 300k | 0.25 | 13.80 @ 5k | +13.55 nats | exp(13.80) ≈ 0.99M |

The gap metric is measured by `scripts/evaluation.py`: held-out loss minus train
loss on matching passages. WikiText-2 ran ~97 epochs and memorized (gap +3.8);
WikiText-103 ran ~1.9 epochs and generalizes (gap ~0, test ppl 165). Same model
and schedule in both runs - the corpus size is the only change. See
[training.md](training.md) and the run metrics for the full curves.

The last two rows are the AtlasTiny data-pipeline milestone (2026-09-12). The
AtlasTiny full run trained the small model for 8k steps (~1.4 epochs) on the new
tokenized corpus and generalizes (gap ~0, coherent TinyStories-style generation).
The slice run is the AGENTS rule-24 gate: dropout off, lr 1e-3, 5k steps over a
300k-token slice (~68 epochs) -> train loss collapses 9.856 -> 0.247 with a
+13.55 nats memorization gap, proving the dataset->tokenizer->model->loss->
optimizer->backward path before any pretraining proceeds.

Measured with:

```bash
python -m scripts.evaluation --checkpoint checkpoints/wikitext103/run_20260905-215250/last.pt --config configs/wikitext103.yaml
```

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
- [x] Implement `scripts/evaluation.py` (perplexity + memorization + generation) - done, see Comparing Results

## Related documentation

- [index.md](index.md) - documentation entry point
- [harness.md](harness.md) - behavioral comparison across model versions
- [hardware.md](hardware.md) - what the GTX 1070 allows before these experiments
- [training.md](training.md) - what each experiment records for reproducibility
- [CONTEXT.md](../CONTEXT.md) - original experiment requirements (sections 63-64)