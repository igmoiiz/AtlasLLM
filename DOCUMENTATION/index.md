# AtlasLLM Documentation

Complete documentation for the AtlasLLM project.

## Table of Contents

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | Model architecture — embeddings, attention, FFN, Transformer blocks |
| [Training](training.md) | Training pipeline — loss, optimizer, scheduler, checkpointing |
| [Tokenizer](tokenizer.md) | Tokenizer training, vocabulary, encode/decode interface |
| [Dataset](dataset.md) | Data pipeline — collection, cleaning, preprocessing, loading |
| [Inference](inference.md) | Generation, sampling strategies, KV cache, streaming |
| [Safety](safety.md) | Guardrail architecture — input/output filtering, policy |
| [Harness](harness.md) | Automated test harness — scenarios, scoring, regression |
| [Experiments](experiments.md) | Experiment system — configurations, tracking, comparisons |
| [Hardware](hardware.md) | Hardware constraints, VRAM budget, device strategy |

## Project Status

AtlasLLM is in **v0.1** — the educational prototype phase.

Current milestone: **Milestone 1 — Environment** (completed)

```
[x] Repository structure
[x] Python environment
[x] Configuration system
[ ] Tokenizer
[ ] Dataset
[ ] Transformer implementation
[ ] Training pipeline
[ ] Inference engine
[ ] Evaluation suite
[ ] Safety system
[ ] Test harness
```

## Architecture Overview

```
Raw Text
   ↓
Tokenizer (HF Tokenizers, 16k vocab)
   ↓
Token IDs [B, T]
   ↓
Token Embedding [B, T, D] + Position Embedding [B, T, D]
   ↓
Transformer Block × 6 (Pre-Norm, Causal Attention, FFN)
   ↓
Final LayerNorm [B, T, D]
   ↓
LM Head [B, T, 16000] (vocab logits)
   ↓
Next-Token Prediction
```

## Directory Map

```
AtlasLLM/
├── model/              # Transformer architecture (pure tensors in, tensors out)
├── tokenizer/          # Tokenizer training and interface
├── data_pipeline/      # Dataset loading and preprocessing
├── training/           # Training orchestration
├── inference/          # Generation engine
├── evaluation/         # Perplexity and benchmarks
├── safety/             # Guardrails (external to model)
├── harness/            # Automated testing
├── monitoring/         # Metrics and profiling
├── configs/            # YAML experiment configurations
├── scripts/            # CLI utilities
├── tests/              # Unit tests
├── DOCUMENTATION/      # This directory
└── checkpoints/        # Model checkpoints (git-ignored)
```

## Design Principles

1. **Correctness before performance** — working code first, optimization later
2. **One source of truth** — no duplicate implementations
3. **Model purity** — Transformer knows nothing about training, data, or safety
4. **Configuration-driven** — all experiments controlled by YAML, not code changes
5. **Educational transparency** — every component must be understandable
