# Hardware

AtlasLLM hardware constraints and device strategy.

## Development System

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GTX 1070 — 8 GB VRAM |
| CPU | Intel Xeon E3-1270 v3 (4C/8T, 3.5 GHz) |
| RAM | 32 GB DDR3 1600 MHz |
| Storage | 512 GB SSD + 1 TB HDD |
| OS | Windows 10/11 |
| CUDA | 12.6 |
| PyTorch | 2.12.0+cu126 |

## VRAM Budget

The GTX 1070 has 8 GB VRAM. This is the hard constraint for model size and batch size.

### Model VRAM Breakdown

| Component | Memory |
|-----------|--------|
| Token embedding (16000 × 256) | ~16 MB |
| Position embedding (256 × 256) | ~0.25 MB |
| Per Transformer block | ~2 MB |
| 6 Transformer blocks | ~12 MB |
| FFN layers (6 × 256 × 1024) | ~12 MB |
| Attention layers (6 × 256 × 256) | ~4 MB |
| LM head (256 × 16000) | ~16 MB |
| **Model weights total** | **~60 MB** |
| Optimizer states (AdamW) | ~240 MB |
| Gradients | ~60 MB |
| Activations (batch=8, seq=256) | ~200 MB |
| **Total estimated** | **~560 MB** |

This fits comfortably within 8 GB.

### Scaling Limits

| Config | Est. VRAM | Fits GTX 1070? |
|--------|-----------|----------------|
| Small (256d, 6L) | ~0.6 GB | Yes |
| Medium (384d, 8L) | ~2 GB | Yes |
| Large (512d, 12L) | ~6 GB | Tight |
| XL (768d, 12L) | ~14 GB | No |

### Batch Size × Context Length

| batch_size | context_length | Est. VRAM |
|-----------|---------------|-----------|
| 8 | 256 | ~0.6 GB |
| 16 | 256 | ~1.0 GB |
| 4 | 512 | ~0.8 GB |
| 8 | 512 | ~1.5 GB |
| 2 | 1024 | ~1.2 GB |

## Device Strategy

AtlasLLM supports CPU and CUDA through a common device abstraction:

```python
def get_device(config):
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)
```

Code must never assume CUDA exists.

## Precision

| Precision | Status |
|-----------|--------|
| FP32 | Primary — use for initial training |
| FP16 | Supported on GTX 1070 (Pascal) but test stability |
| BF16 | Not supported on GTX 1070 (requires Ampere+) |

Development order: FP32 first → verified → then test FP16.

## Performance Considerations

### Data Loading

- Use `num_workers > 0` for parallel data loading
- Pin memory for faster CPU→GPU transfer
- Pre-tokenize data to avoid tokenization bottleneck during training

### Training Throughput

Expected for AtlasLLM-Small on GTX 1070:

| Metric | Expected |
|--------|----------|
| Forward + backward | ~8 ms |
| Optimizer step | ~2 ms |
| Total per step | ~10 ms |
| Steps per second | ~100 |
| Tokens per second | ~200,000 |

### Bottleneck Identification

During training, monitor:

- GPU utilization (target > 80%)
- GPU memory (watch for OOM)
- Data loading time (should be < compute time)
- CPU utilization (data preprocessing)

## CPU Fallback

When GPU is unavailable:

```bash
python -m training.train --config configs/debug.yaml --device cpu
```

Training is ~10-50x slower on CPU but functional for debugging.

## Hardware-Specific Notes

### GTX 1070 (Pascal)

- No FlashAttention support (requires Ampere+)
- FP16 works but may have numerical instability — use with caution
- No BF16 support
- 8 SM architecture (compute capability 6.1)
- Good enough for AtlasLLM's small model scale
