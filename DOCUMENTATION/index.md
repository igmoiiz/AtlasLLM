# AtlasLLM Documentation

The technical and conceptual documentation for AtlasLLM, a small dense decoder-only Transformer language model built and trained from scratch on a GTX 1070.

> **Read this first.** This page is the entry point. Read the "How this project works" section if you are new, then follow the links in [Table of contents](#table-of-contents). Each page links back here and to its neighbors, so you can move forward and backward through the docs.

## How this project works (for beginners)

A language model is a program that learns to predict the next word of a text. This project builds that program piece by piece:

1. **Data.** Raw text (plain `.txt` files) is the only source material. See [dataset.md](dataset.md).
2. **Tokenizer.** A tool that cuts text into a fixed set of small pieces ("tokens") the model can count. See [tokenizer.md](tokenizer.md).
3. **Model.** A Transformer: a neural network that reads token history and guesses the next token. See [architecture.md](architecture.md).
4. **Training.** Repeatedly showing the model real text, measuring how wrong its guesses are, and adjusting its internal numbers to make future guesses better. See [training.md](training.md).
5. **Inference.** Using the trained model to finish a prompt, one token at a time. See [inference.md](inference.md).
6. **Measurement.** Fixed probe suites score every checkpoint and a gate catches regressions. See [harness.md](harness.md) and [experiments.md](experiments.md). Guardrails are implemented but opt-in ([safety.md](safety.md)), and hardware profiling is covered in [hardware.md](hardware.md).

Everything runs on one consumer GPU with 8 GB of memory. See [hardware.md](hardware.md).

## Table of contents

| Document | Purpose | Status |
|----------|---------|--------|
| [index.md](index.md) | Overview, status, reading order | Active |
| [architecture.md](architecture.md) | The Transformer model inside out | Active |
| [tokenizer.md](tokenizer.md) | Text to token IDs and back | Active |
| [dataset.md](dataset.md) | Raw text to training batches | Active |
| [training.md](training.md) | Loss, optimizer, scheduler, checkpoints | Active |
| [inference.md](inference.md) | Generation, sampling, KV cache | Active |
| [hardware.md](hardware.md) | Memory budget, device strategy, speed | Active |
| [experiments.md](experiments.md) | Experiment design and measured comparisons | Active, initial results |
| [safety.md](safety.md) | Policy-driven input/output guardrails | Active |
| [harness.md](harness.md) | Probe suites, scoring, regression gate | Active |

## Project status

The pipeline follows this dependency order:

```text
data -> tokenizer -> dataset -> model -> training -> evaluation -> inference -> safety -> harness
```

| Milestone | Deliverable | Status |
|-----------|-------------|--------|
| 1 | Project structure, config system, environment | Done |
| 2 | Tokenizer (BPE, lossless roundtrip) | Done, tested |
| 3 | Dataset pipeline (preprocessing, .bin store) | Done, tested |
| 4 | Transformer implementation | Done, tested |
| 5 | Debug training and tiny-overfit verification | Done |
| 6 | Real pretraining (small config on GPU) | Done - 13M (v1) and 48M (v2); v3 on 930M tokens next |
| 7 | Inference engine (sampling, KV cache, streaming) | Done, tested |
| 8 | Evaluation suite (perplexity, memorization, probes) | Done, tested |
| 9 | Instruction tuning | Planned |
| 10 | Safety guardrails (input/output filters, policy) | Done, tested, opt-in |
| 11 | Automated harness (probe suites, scoring, gate) | Done, tested, measured |
| 12 | Monitoring (device/VRAM introspection, step benchmark) | Done, tested, measured |
| 13 | Release (v1.0) | Planned |

**Quality gates currently passing:** 195 unit tests in `tests/`, `ruff check .` clean, end-to-end training verified with checkpoint save and resume. CI runs the same lint and test suite on Python 3.11 and 3.12 (CPU-only Ubuntu; CUDA-only checks are skipped there).

For the measured numbers behind the "Done" claims, see [hardware.md](hardware.md) and [training.md](training.md).

## Architecture overview

```text
Raw Text
   ↓
Tokenizer (BPE, 16k vocab)
   ↓
Token IDs [B, T]
   ↓
Token Embedding [B, T, D] + Position Embedding [B, T, D]
   ↓
Transformer Block × n_layers (6 in the small config, 10 in AtlasBase)
   ├── LayerNorm + Multi-Head Causal Self-Attention + Residual
   └── LayerNorm + Feed-Forward Network + Residual
   ↓
Final LayerNorm [B, T, D]
   ↓
LM Head [B, T, 16000] (vocab logits)
   ↓
Next-Token Prediction
```

B = batch, T = sequence length, D = model dimension, V = vocabulary size. Full details in [architecture.md](architecture.md).

## Directory map

```text
AtlasLLM/
├── model/              # Transformer architecture (pure tensors in, tensors out)
├── tokenizer/          # Tokenizer training and interface
├── data_pipeline/      # Dataset loading and preprocessing
├── training/           # Training orchestration
├── inference/          # Generation engine
├── evaluation/         # Perplexity, memorization, generation probes
├── safety/             # Policy-driven guardrails (opt-in, external)
├── harness/            # Probe suites, scoring, regression gate
├── monitoring/         # Device/VRAM introspection, step benchmark
├── configs/            # YAML experiment configurations
├── scripts/            # CLI utilities
├── tests/              # Unit tests (195)
├── DOCUMENTATION/      # This directory
└── checkpoints/        # Model runs (git-ignored)
```

## Design principles

1. **Correctness before performance** - working code first, optimization after measuring.
2. **One source of truth** - no duplicate implementations; one authoritative module per responsibility.
3. **Model purity** - the Transformer knows nothing about training, data, or safety.
4. **Configuration-driven** - experiments are controlled by YAML, not code edits.
5. **Educational transparency** - core mathematics stays readable, not hidden behind black-box libraries.
6. **Reproducibility** - every run records seed, versions, config, and hardware ([training.md](training.md)).
7. **No unmeasured claims** - throughput and memory figures in these docs come from actual runs on this machine.

## Reading paths

For a beginner: start at [dataset.md](dataset.md) and follow the pipeline in order.

For an agent or engineer modifying code: start with this page, then read the page for the subsystem you touch, plus its neighbors:

- Changing the model: [architecture.md](architecture.md) and [training.md](training.md).
- Changing text handling: [tokenizer.md](tokenizer.md) and [dataset.md](dataset.md).
- Changing generation: [inference.md](inference.md) and [training.md](training.md).
- Planning evaluation, safety, or profiling work: [experiments.md](experiments.md), [harness.md](harness.md), [safety.md](safety.md), [hardware.md](hardware.md).
- Capacity planning: [hardware.md](hardware.md).

## Related documentation

- [README.md](../README.md) - project home page with quick start.
- [CONTEXT.md](../CONTEXT.md) - the project execution specification (goals, milestones, hardware constraints).
- [AGENTS.md](../AGENTS.md) - engineering rules every contributor and agent must follow.