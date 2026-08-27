# AtlasLLM

A small dense decoder-only Transformer language model, built from scratch on one GTX 1070. The goal is to understand the complete lifecycle of a language model: data, tokenizer, architecture, training, inference, and evaluation all written and tested in this repository.

> **Reading guide.** If you have no background in machine learning, start with the "Layman's guide" below, then use the [documentation index](DOCUMENTATION/index.md). If you are here to work on the code (human or agent), the technical pages in `DOCUMENTATION/` describe each subsystem in detail.

## Layman's guide

A language model is a program that reads text and predicts the next word. Give it "The cat sat on the", and it predicts "mat". Train it on enough text and it learns grammar, facts, and style. This project builds one such program from the basic building blocks, so that every layer is visible and explainable. It is a teaching project and a baseline for larger work (see [AtlasMoE](#relationship)).

The model reads about 26,000 tokens per second during training and uses roughly 0.2 GB of the 8 GB on the GPU, leaving most of the hardware headroom for experiments.

## Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GTX 1070, 8 GB VRAM |
| CPU | Intel Xeon E3-1270 v3 |
| RAM | 32 GB DDR3 |
| Storage | 512 GB SSD + 1 TB HDD |
| OS | Windows |
| Python | 3.11+ (developed on 3.14) |
| PyTorch | 2.x with CUDA 12.x |

A CPU-only machine also works for the debug config and for inference; `device: auto` in the config selects CUDA when present.

See [DOCUMENTATION/hardware.md](DOCUMENTATION/hardware.md) for the memory budget and measured throughput.

## What is built

Status is tracked per subsystem. The repository follows an architecture where data moves through clearly separated stages:

```text
data -> tokenizer -> dataset -> model -> training -> evaluation -> inference -> safety -> harness
```

| Subsystem | Package | Status | Documentation |
|-----------|---------|--------|---------------|
| Transformer model | `model/` | Done, tested | [architecture.md](DOCUMENTATION/architecture.md) |
| Tokenizer | `tokenizer/` | Done, tested | [tokenizer.md](DOCUMENTATION/tokenizer.md) |
| Data pipeline | `data_pipeline/` | Core done, tested | [dataset.md](DOCUMENTATION/dataset.md) |
| Training pipeline | `training/` | Done, tested | [training.md](DOCUMENTATION/training.md) |
| Inference engine | `inference/` | Done, tested | [inference.md](DOCUMENTATION/inference.md) |
| Evaluation | `evaluation/` | Planned | [experiments.md](DOCUMENTATION/experiments.md) |
| Safety guardrails | `safety/` | Planned | [safety.md](DOCUMENTATION/safety.md) |
| Test harness | `harness/` | Planned | [harness.md](DOCUMENTATION/harness.md) |
| Monitoring | `monitoring/` | Planned | [experiments.md](DOCUMENTATION/experiments.md) |

The `evaluation/`, `safety/`, `harness/`, and `monitoring/` packages currently exist as stub modules. Their design is documented so implementation can be community-contributed or tackled next without guessing. The `tests/` suite (84 tests) covers the subsystems marked Done.

## Default configuration (AtlasLLM-Small)

`configs/small.yaml`:

| Parameter       | Value  |
|-----------------|--------|
| vocab_size      | 16,000 |
| context_length  | 256    |
| d_model         | 256    |
| n_layers        | 6      |
| n_heads         | 8      |
| d_ff            | 1,024  |
| dropout         | 0.1    |
| bias            | false  |
| parameters      | 12,982,784 (measured) |

Alternative configurations live in `configs/`: `debug.yaml` for smoke tests and tiny overfit runs, `medium.yaml` for a larger model. See [index.md](DOCUMENTATION/index.md) for the full picture and [experiments.md](DOCUMENTATION/experiments.md) for the planned experiment matrix.

## Setup

Requires Python 3.11+ and PyTorch with CUDA for GPU training.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[all]"
```

For a minimal install without optional extras (matplotlib, tensorboard, etc.):

```bash
pip install -e .
```

## Usage

### 1. Build the dataset

```bash
python -m data_pipeline.preprocessing --config configs/small.yaml
```

This tokenizes the raw text files named in the config and writes `data/processed/*.bin` plus `meta.json`. Document what data you put in `data/raw/` in `data/README.md` (see [dataset.md](DOCUMENTATION/dataset.md)).

### 2. Train

```bash
python -m training.train --config configs/small.yaml
```

Every run writes to a timestamped folder under `checkpoints/run_<timestamp>/` containing `last.pt`, `best.pt`, `metrics.jsonl`, `config.yaml`, and `reproducibility.json`. Resume an interrupted run with:

```bash
python -m training.train --config configs/small.yaml --resume checkpoints/run_20260101-120000/last.pt
```

Override the total step count with `--steps N`. See [training.md](DOCUMENTATION/training.md).

### 3. Chat with the model

```bash
python -m scripts.chat --checkpoint checkpoints/run_20260101-120000/last.pt
```

The config, tokenizer, and model are reconstructed from the checkpoint automatically, so `--config` is only required for checkpoints saved before the checkpoint carried its config. One-shot completion also works:

```bash
python -m inference.generate \
    --checkpoint checkpoints/run_20260101-120000/last.pt \
    --prompt "The capital of France is" \
    --max-tokens 50 --temperature 0.8 --top-k 50
```

See [inference.md](DOCUMENTATION/inference.md).

### 4. Run the tests and linter

```bash
python -m pytest                # 84 tests
python -m ruff check .
```

## Project Structure

```text
AtlasLLM/
├── model/              # Transformer architecture (embeddings, attention, blocks)
├── tokenizer/          # Tokenizer training, vocabulary, encode/decode
├── data_pipeline/      # Preprocessing, dataset, batching
├── training/           # Loss, optimizer, scheduler, trainer, checkpoints
├── inference/          # Generation, sampling, KV cache
├── evaluation/         # Planned: perplexity, benchmarks, generalization
├── safety/             # Planned: input/output guardrails
├── harness/            # Planned: automated scenario testing
├── monitoring/         # Planned: metrics, profiling
├── configs/            # YAML experiment configurations
├── scripts/            # CLI utilities (chat, dataset inspection, training analysis)
├── tests/              # Unit tests (84)
├── DOCUMENTATION/      # Project documentation (start at index.md)
├── data/               # Data: raw/ (immutable), interim/, processed/
└── checkpoints/        # Model runs (git-ignored)
```

## Documentation

The full documentation set, meant to be read as a chain:

- [Documentation index](DOCUMENTATION/index.md) - overview, status, and reading order
- [Architecture](DOCUMENTATION/architecture.md) - the Transformer model
- [Tokenizer](DOCUMENTATION/tokenizer.md) - text to token IDs and back
- [Dataset](DOCUMENTATION/dataset.md) - raw text to training batches
- [Training](DOCUMENTATION/training.md) - loss, optimizer, scheduler, checkpoints
- [Inference](DOCUMENTATION/inference.md) - generation, sampling, KV cache
- [Hardware](DOCUMENTATION/hardware.md) - memory budget, device strategy, measured throughput
- [Experiments](DOCUMENTATION/experiments.md) - planned experiment system
- [Safety](DOCUMENTATION/safety.md) - planned guardrail design
- [Harness](DOCUMENTATION/harness.md) - planned automated test harness

## Relationship

AtlasLLM is the dense baseline for [AtlasMoE](https://github.com/igmoiiz/AtlasMoE), a Mixture-of-Experts extension. AtlasLLM stays dense and simple; AtlasMoE reuses the surrounding pipeline.

## License

Proprietary - see [LICENSE.md](LICENSE.md).