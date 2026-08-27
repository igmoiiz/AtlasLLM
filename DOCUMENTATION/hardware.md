# Hardware

AtlasLLM hardware constraints, device strategy, and measured performance.

## For a beginner

Training a language model means running huge math on every text at once. That math lives on the graphics card (GPU), which has a fixed amount of fast memory (VRAM). This project targets one consumer GPU with 8 GB of VRAM, so every design decision keeps the model small enough to fit comfortably with room to spare. During training the model uses about 0.2 GB of the 8 GB available.

"CPU" and "GPU" are two ways to run the code: a GPU is much faster but only on machines that have one. The project falls back to CPU automatically when no GPU is present, which is useful for debugging on any machine.

Anything below this line is the technical description. See [index.md](index.md) for the full picture.

## Development System

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GTX 1070 - 8 GB VRAM |
| CPU | Intel Xeon E3-1270 v3 (4C/8T, 3.5 GHz) |
| RAM | 32 GB DDR3 1600 MHz |
| Storage | 512 GB SSD + 1 TB HDD |
| OS | Windows |
| CUDA | 12.6 |
| PyTorch | 2.12.0+cu126 |

## Memory budget

The GTX 1070 has 8 GB of VRAM. The small config is deliberately far below that ceiling.

### Measured footprint (AtlasLLM-Small)

| Item | Measured |
|------|----------|
| Model parameters | 12,982,784 (~13M) |
| Model weights in FP32 | ~52 MB |
| Optimizer states (AdamW) + gradients | ~208 MB |
| Training step VRAM peak (batch 8 × context 256) | ~217 MB |
| Checkpoint file (`last.pt` / `best.pt`) | ~156 MB |

### Headroom

8 GB with ~217 MB in use leaves ~7.8 GB of headroom. Larger configs (block width, layer count, batch, context) stay feasible up to roughly the medium scale. See [configs/medium.yaml](../configs/medium.yaml).

## Measured performance

| Metric | Measured |
|--------|----------|
| Training throughput | ~26,000 tokens/sec (~13 steps/sec at batch 8 × context 256) |
| Training VRAM | ~217 MB |
| Checkpoint size | ~156 MB |

Numbers come from an actual 100k-step run on this machine (see [training.md](training.md)). Anything not backed by a measurement is explicitly labelled estimated.

## Device strategy

Device selection is centralized in one place per entry point, never re-implemented per module:

- Training: `resolve_device()` in [training/train.py](../training/train.py) - `"auto"` picks CUDA when available, else CPU.
- Inference: `inference/engine.py` uses the same rule when passed `device="auto"`.

Code must never assume CUDA exists; every `torch.device` decision flows through these helpers.

```python
# device: auto in the YAML config selects CUDA when present, else CPU
python -m training.train --config configs/small.yaml
```

> Note: there is no `--device` CLI flag. Selection is controlled by the
> top-level `device:` key in the YAML config (`auto` | `cuda` | `cpu`); the
> default is `auto`.

## Precision

| Precision | Status |
|-----------|--------|
| FP32 | Default (`dtype: float32`) - used for all training to date |
| FP16 | Configurable (`dtype: float16`) - GTX 1070 (Pascal) supports it; untested, verify first |
| BF16 | Not supported on GTX 1070 (requires Ampere+) |

Development order: FP32 first, verified, then experiment with FP16.

## CPU fallback

When no CUDA device is present, set `device: cpu` (or leave `auto`) in the config:

```bash
python -m training.train --config configs/debug.yaml
```

Training is far slower on CPU (one to two orders of magnitude) but functional for debugging.

## Hardware-specific notes

### GTX 1070 (Pascal)

- No FlashAttention support (requires Ampere+) - fine for this model size
- FP16 works but may be numerically unstable - keep FP32 unless measured otherwise
- No BF16 support
- Compute capability 6.1
- Good enough for AtlasLLM's target scale

## Related documentation

- [index.md](index.md) - documentation entry point
- [training.md](training.md) - measured run statistics
- [architecture.md](architecture.md) - model parameter counts
- [experiments.md](experiments.md) - how bigger configs would be measured
- [configs/small.yaml](../configs/small.yaml) - the measured configuration