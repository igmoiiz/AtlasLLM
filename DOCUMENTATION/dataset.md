# Dataset Pipeline

AtlasLLM data pipeline — from raw text to training batches.

## Overview

```
Raw Text
     ↓
Collection & Provenance
     ↓
Cleaning Pipeline
     ├── Unicode normalization
     ├── Whitespace normalization
     ├── Deduplication
     ├── Quality filtering
     └── Length filtering
     ↓
Train / Validation / Test Split
     ↓
Tokenization
     ↓
Sequence Construction
     ↓
Binary Storage (.bin)
     ↓
DataLoader → Training Loop
```

## Directory Structure

```
data/
├── raw/              # Immutable original data — NEVER modify
├── interim/          # Intermediate processing results
├── processed/        # Final tokenized data
│   ├── train.bin     # Training token IDs (uint16)
│   ├── val.bin       # Validation token IDs (uint16)
│   └── meta.json     # Dataset metadata
└── README.md         # Data provenance documentation
```

## Data Provenance

Every dataset added must be documented in `data/README.md`:

| Field | Description |
|-------|-------------|
| Name | Dataset name |
| Source | URL or origin |
| License | Legal terms |
| Download date | When acquired |
| Raw size | Uncompressed size |
| Language | Primary language |
| Preprocessing | What was done |
| Token count | Total tokens |

## Cleaning Pipeline

### Step 1 — Unicode Normalization

Normalize all text to NFC (Canonical Decomposition followed by Canonical Composition).

### Step 2 — Whitespace Normalization

- Replace tabs with spaces
- Collapse multiple spaces
- Normalize newlines
- Strip trailing whitespace
- Preserve meaningful paragraph breaks

### Step 3 — Deduplication

Exact deduplication at the document level.

### Step 4 — Quality Filtering

- Remove documents shorter than 100 characters
- Remove documents with excessive special characters
- Remove non-UTF-8 sequences

### Step 5 — Length Filtering

Remove extremely short records that provide no learning signal.

## Train / Validation / Test Split

```
Total data:
├── 90% → train.bin
├── 5%  → val.bin
└── 5%  → test.bin (held out for final evaluation)
```

The split is performed **before** sequence construction to prevent leakage.

## Sequence Construction

For context length `T`:

```
Full token stream: [t_0, t_1, t_2, ..., t_N]

Sequence 0: [t_0,   t_1,   ..., t_{T-1}]
Target  0:  [t_1,   t_2,   ..., t_T]

Sequence 1: [t_T,   t_{T+1}, ..., t_{2T-1}]
Target  1:  [t_{T+1}, t_{T+2}, ..., t_{2T}]
...
```

Each sequence is the next-token prediction task.

## Binary Format

Tokenized data is stored as raw `uint16` binary files:

```python
import numpy as np

# Save
tokens = np.array(token_ids, dtype=np.uint16)
tokens.tofile("data/processed/train.bin")

# Load
data = np.fromfile("data/processed/train.bin", dtype=np.uint16)
```

Using `uint16` supports vocabulary sizes up to 65,536.

## DataLoader

```python
class TextDataset:
    def __init__(self, data_path, context_length):
        self.data = np.fromfile(data_path, dtype=np.uint16)
        self.context_length = context_length

    def __getitem__(self, idx):
        start = idx * self.context_length
        chunk = self.data[start : start + self.context_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)   # input
        y = torch.tensor(chunk[1:], dtype=torch.long)     # target
        return {"input_ids": x, "targets": y}
```

## Dataset Requirements

- **Size:** 10–100 MB of clean text for initial training
- **Language:** English (initially)
- **License:** Public domain or permissive
- **Quality:** Clean, well-formed text without excessive noise

## Adding a Dataset

1. Place raw files in `data/raw/`
2. Document provenance in `data/README.md`
3. Run preprocessing: `python -m data_pipeline.preprocessing --input data/raw/ --output data/processed/`
4. Verify output: `python -m scripts.inspect_dataset --data data/processed/train.bin`
