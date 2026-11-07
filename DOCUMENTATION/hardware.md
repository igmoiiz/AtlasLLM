# Hardware

AtlasLLM hardware constraints, device strategy, and measured performance.

## For a beginner

Training a language model means running huge math on every text at once. That math lives on the graphics card (GPU), which has a fixed amount of fast memory (VRAM). This project targets one consumer GPU with 8 GB of VRAM, so every design decision keeps the model small enough to fit comfortably with room to spare. The 13M small config uses about 0.2 GB; the 48M pretraining config peaks at about 2.1 GB with batch 8, still well inside the budget.

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

### Measured footprint (AtlasLLM-Small, 13M)

| Item | Measured |
|------|----------|
| Model parameters | 12,982,784 (~13M) |
| Model weights in FP32 | ~52 MB |
| Optimizer states (AdamW) + gradients | ~208 MB |
| Training step VRAM peak (batch 8 × context 256) | ~217 MB |
| Checkpoint file (`last.pt` / `best.pt`) | ~156 MB |

### Measured footprint (AtlasBase-v2, 48M)

From `reports/data/hardware/bench_atlasbase_v2.json`, a real forward,
backward and AdamW step at context 256:

| Batch | Tokens/step | Peak allocated | Reserved |
|-------|------------:|---------------:|---------:|
| 8 | 2,048 | 2,142.8 MiB | 2,312.0 MiB |
| 16 | 4,096 | 3,713.0 MiB | 4,166.0 MiB |
| 24 | 6,144 | 5,286.1 MiB | 5,598.0 MiB |
| 32 | 8,192 | 6,855.4 MiB | 7,176.0 MiB |
| 48 | 12,288 | 9,995.5 MiB | 10,756.0 MiB |
| 64 | 16,384 | 13,140.2 MiB | 14,004.0 MiB |

Physical VRAM is 8,191.9 MiB. Batches 48 and 64 exceed it without failing:
Windows overcommits VRAM through host memory, so the step still runs but at
about one tenth the speed (see below). They are not usable configurations.

### Headroom

The 13M small config leaves ~7.8 GB free. The 48M config at batch 8 leaves
~6.0 GB free; batch 32 still fits with ~1.0 GB (7,176 MiB reserved). Larger
configs (block width, layer count, batch, context) stay feasible up to roughly
the medium scale. See [configs/medium.yaml](../configs/medium.yaml).

## Measured performance

### AtlasLLM-Small (13M)

| Metric | Measured |
|--------|----------|
| Training throughput | ~26,000 tokens/sec (~13 steps/sec at batch 8 × context 256) |
| Training VRAM | ~217 MB |
| Checkpoint size | ~156 MB |

### AtlasBase-v2 (48M)

Same artifact as the footprint table, 10 timed steps after 3 warmup steps:

| Batch | ms/step | Tokens/sec |
|-------|--------:|-----------:|
| 8 | 225.37 | 9,088 |
| 16 | 396.99 | 10,318 |
| 24 | 565.40 | 10,867 |
| 32 | 675.60 | 12,126 |
| 48 | 9,892.49 | 1,242 |
| 64 | 19,683.89 | 832 |

Throughput rises with batch size up to 32. Batches 48 and 64 are slower by an
order of magnitude because they ran past physical VRAM and were served from
host memory; the numbers are recorded, not recommended. The planned v3 training
configuration stays at batch 8 (2,048 tokens/step), which measured 225 ms/step
on this machine. At 9,088 tokens/sec, 930M tokens is about 28 hours of pure
compute before interruptions.

Numbers come from measurements on this machine: the 13M row from an actual
100k-step run (see [training.md](training.md)), the 48M rows from
`scripts/benchmark.py`. Anything not backed by a measurement is explicitly
labelled estimated.

## Monitoring

`monitoring/system.py` is the one source of truth for hardware facts. No module
reads `torch.cuda` counters on its own.

| Function | Returns |
|----------|---------|
| `device_info(device)` | `DeviceInfo(type, name, total_memory_mb, cuda_version, capability, torch_version)` |
| `vram_usage(device)` | `VramUsage(allocated_mb, reserved_mb, peak_mb, total_mb)`, or `None` on CPU |

Both dataclasses provide `to_dict()` for JSON. The trainer writes
`gpu_memory_mb` (allocated) and `gpu_reserved_mb` into every `metrics.jsonl`
record; before this module existed the first field was computed inline and the
second was missing. The training entry point also uses `device_info(device).name`
for the `hardware` field in `reproducibility.json`.

`scripts/benchmark.py` measures a real training step (forward, cross-entropy,
backward, AdamW) for each batch size in a grid:

```bash
python -m scripts.benchmark --config configs/atlasbase_v2.yaml \
    --batch-sizes 8 16 24 32 --steps 10 \
    --out reports/data/hardware/bench_atlasbase_v2.json
```

Warmup steps are excluded from timing, the batch is created once and reused,
and device, dtype and AdamW settings come from the same helpers training uses.
A configuration that raises out-of-memory is recorded with `oom: true` and the
grid continues. A configuration that overcommits VRAM on Windows is recorded
with `spills: true`; its throughput is real but not usable. The sweep committed
today predates that flag and shows the same outcome as `peak_allocated_mb`
above `total_mb`. The JSON also carries the device description, parameter
count, sequence length and hyperparameters, so a benchmark can be compared
later without guessing.

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