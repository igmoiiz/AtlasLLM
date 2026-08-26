# AtlasLLM

A small dense decoder-only Transformer language model, built from scratch for learning.

## Hardware

- **GPU:** NVIDIA GTX 1070 (8 GB VRAM)
- **CPU:** Intel Xeon E3-1270 v3
- **RAM:** 32 GB DDR3

## Architecture

```text
Token IDs → Embedding + Position → Transformer Blocks × N → LayerNorm → LM Head → Logits
```

Decoder-only, pre-normalization, learned positional embeddings, causal self-attention.

## Default Configuration

| Parameter       | Value  |
|-----------------|--------|
| vocab_size      | 16,000 |
| context_length  | 256    |
| d_model         | 256    |
| n_layers        | 6      |
| n_heads         | 8      |
| d_ff            | 1,024  |
| dropout         | 0.1    |
| parameters      | ~5.5M  |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[all]"
```

## Usage

```bash
# Training
python -m training.train --config configs/small.yaml

# Debug / overfit test
python -m training.train --config configs/debug.yaml

# Inference
python -m inference.generate --checkpoint checkpoints/best.pt

# Count parameters
python -m scripts.count_parameters

# Visualize attention patterns
python -m scripts.attention_visualization --checkpoint checkpoints/best.pt --layer 0

# Plot training curves
python -m scripts.training_analysis --log-dir runs/ --output training_curves.png

# Run evaluation
python -m scripts.evaluation --checkpoint checkpoints/best.pt --config configs/small.yaml
```

## Project Structure

```text
AtlasLLM/
├── model/              # Transformer architecture
├── tokenizer/          # Tokenizer training and interface
├── data_pipeline/      # Dataset loading and preprocessing
├── training/           # Training loop, loss, optimizer, scheduler
├── inference/          # Generation, sampling, KV cache
├── evaluation/         # Perplexity, benchmarks, generalization
├── safety/             # Input/output guardrails
├── harness/            # Automated test harness
├── monitoring/         # Metrics and profiling
├── configs/            # YAML experiment configurations
├── scripts/            # Utility scripts (visualization, analysis, eval)
├── tests/              # Unit tests
├── DOCUMENTATION/      # Project documentation
└── checkpoints/        # Model checkpoints
```

See [DOCUMENTATION/index.md](DOCUMENTATION/index.md) for complete documentation.

## Experiments

Configuration-driven. See `configs/` for available configurations.

### Planned experiments:
- Depth: 2, 4, 6, 8 layers
- Width: 128, 256, 384, 512 hidden dim
- Context: 128, 256, 512 tokens
- Positional encoding: learned, sinusoidal, RoPE
- Activation: GELU, SwiGLU

## Relationship

AtlasLLM is the dense baseline for [AtlasMoE](https://github.com/igmoiiz/AtlasMoE) — a Mixture-of-Experts extension.

## License

Proprietary — see [LICENSE.md](LICENSE.md)
