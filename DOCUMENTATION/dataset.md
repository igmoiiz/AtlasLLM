# Dataset Pipeline

AtlasLLM data pipeline - from raw text to training batches.

## For a beginner

Models learn from examples, and the raw material is plain text. The data pipeline turns that text into the exact form the model trains on:

1. **Collect raw text** into `data/raw/` (immutable; never edited). This project ships no bundled data, so documents what you add in `data/README.md`.
2. **Split** the text into training, validation, and test parts *before* any tokenization, so no validation text leaks into training.
3. **Tokenize** each part with the tokenizer, then store the result as raw number arrays (`.bin` files) for speed.
4. **Build batches**: cut the token stream into fixed-length windows. Each window is a puzzle - the model sees tokens 1..T and must predict tokens 2..T+1.

Anything below this line is the technical description. See [index.md](index.md) for the full picture.

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
├── raw/              # Immutable original data - NEVER modify
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

For WikiText-2 (already high-quality raw Wikipedia text) the cleaning step is
**not applied**: documents are tokenized as-is, preserving punctuation, spacing,
and paragraph breaks (the tokenizer is lossless). The cleaning pipeline below is
reserved for lower-quality corpora (Phase 2, OpenWebText) and must never silently
modify the immutable `data/raw/` sources.

### Step 1 - Unicode Normalization

Normalize all text to NFC (Canonical Decomposition followed by Canonical Composition).

### Step 2 - Whitespace Normalization

- Replace tabs with spaces
- Collapse multiple spaces
- Normalize newlines
- Strip trailing whitespace
- Preserve meaningful paragraph breaks

### Step 3 - Deduplication

Exact deduplication at the document level.

### Step 4 - Quality Filtering

- Remove documents shorter than 100 characters
- Remove documents with excessive special characters
- Remove non-UTF-8 sequences

### Step 5 - Length Filtering

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

## Building the Dataset

```bash
python -m data_pipeline.preprocessing --config configs/small.yaml
python -m data_pipeline.preprocessing --config configs/debug.yaml
```

Each run tokenizes the three split text files (from the config `data.train_text /
val_text / test_text`) and writes raw `uint16` arrays to `data.train_path /
val_path / test_path` plus `meta.json`. Because bins are tokenizer- and
context-specific, `configs/debug.yaml` writes into `data/processed/debug/` so the
debug and small datasets do not clobber each other.

### Automated pipeline (registry-driven)

The supported path for real corpora is the staged pipeline:

```bash
python -m data_pipeline.pipeline --config configs/atlasbase_v2.yaml
python -m data_pipeline.pipeline --config <config> --stage tokenize --force
```

It runs `acquire -> process -> tokenizer -> tokenize -> finalize` over the
sources listed in `pipeline.registry`, and every stage reuses an existing valid
artifact unless `--force` is passed. Artifacts:

```text
data/raw/<source>/records.jsonl                  immutable acquired cache
data/interim/<source>/{train,val,test}.jsonl     filtered documents
tokenizer/model/<name>/                          trained tokenizer
data/processed/<name>/{train,val,test}.bin       tokenized streams
data/processed/<name>/{meta,manifest}.json       split metadata + provenance
reports/data/<name>/                             statistics report
```

Acquisition streams each HuggingFace split into the cache, so only the documents
actually used are downloaded. Processing normalizes, filters, deduplicates, and
splits each source; tokenization then writes one bounded `.bin` per source per
split and composes the split the trainer reads by interleaving them.

### Weighted mixture

A config that sets `pipeline.dataset.train_tokens` builds a mixture instead of
a concatenation:

- `pipeline.registry[].target_fraction` is each source's share of the training
  budget; the fractions must sum to 1.0 and the pipeline refuses to run
  otherwise.
- Per-source budgets are derived from it: `train = train_tokens * fraction`,
  `val = train * val_ratio / train_ratio` (same for test). Validation therefore
  follows the same mixture as training, which keeps per-source perplexity
  comparable.
- Acquisition and processing stop at a character budget,
  `sum(budgets) * chars_per_token * acquisition_margin` (defaults 4.0 and 1.10
  for English BPE). Tokens are the unit the mixture is defined in, but the
  pipeline reads text; the estimate is deliberately generous so truncation, not
  a short stream, enforces the cap. `source.json` records the realised character
  count and `processing_stats.json` records `char_budget.reached`.
- Deduplication is shared: one document-id set spans all sources, so a page that
  appears in two sources is kept once, by the first source in registry order.
  The kept ids are written to `data/interim/<source>/document_ids.txt`, so a
  resumed build deduplicates exactly like a fresh one.
- `pack_split(max_tokens=...)` truncates a source at its exact token budget;
  `interleave_bins` then appends chunks from every source, sized by
  `target_fraction * mix_chunk_tokens`, so a contiguous slice of `train.bin`
  reflects the mixture rather than one source's block. `meta.json` and
  `manifest.json` record realised tokens per source per split, so a source that
  came up short is visible instead of silently skewing the mixture.

Artifacts for a mixture (per source plus the mixed split):

```text
data/processed/<name>/train_<source>.bin   bounded per-source stream
data/processed/<name>/{train,val,test}.bin interleaved mixture for training
data/processed/<name>/meta.json            per-source and mixed token counts
```

The v3 corpus uses 930M training tokens over FineWeb-Edu 60%, Wikipedia 15%,
Gutenberg 15% and peS2o 10%; see [configs/atlasbase_v3.yaml](../configs/atlasbase_v3.yaml)
and [AGENT_CONTEXT.md](../AGENT_CONTEXT.md).

Verify a built dataset:

```bash
python -m scripts.inspect_dataset --data data/processed/train.bin
```

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

`data_pipeline.dataset.TextDataset` memory-maps a `uint16` .bin file and returns
contiguous, non-overlapping, next-token-shifted windows:

```python
from torch.utils.data import DataLoader
from data_pipeline.dataset import TextDataset

ds = TextDataset("data/processed/train.bin", context_length=256)
loader = DataLoader(ds, batch_size=8, shuffle=True)

batch = next(iter(loader))
batch["input_ids"]  # [B, T] torch.long
batch["targets"]    # [B, T] torch.long, input shifted by one token
```

- `__len__` = number of complete windows: `(n_tokens - 1) // context_length`
- Each window consumes `context_length + 1` tokens (input `t0..t_{T-1}`, target
  `t1..t_T`); the stream tail is dropped each epoch
- Padding is unnecessary (fixed-length windows); cross-document boundaries are
  treated as ordinary token context, matching the chained corpus used to train
  the tokenizer

## Dataset Requirements

- **Size:** 10-100 MB of clean text for initial training
- **Language:** English (initially)
- **License:** Public domain or permissive
- **Quality:** Clean, well-formed text without excessive noise

## Adding a Dataset

1. Place raw files in `data/raw/`
2. Document provenance in `data/README.md`
3. Run preprocessing: `python -m data_pipeline.preprocessing --config configs/small.yaml`
4. Verify output: `python -m scripts.inspect_dataset --data data/processed/train.bin`

## Related documentation

- [index.md](index.md) - documentation entry point
- [tokenizer.md](tokenizer.md) - how text is tokenized during preprocessing
- [training.md](training.md) - how the resulting batches are consumed
- [scripts/inspect_dataset.py](../scripts/inspect_dataset.py) - verification utility
- [DATA_PIPELINE.md](../DATA_PIPELINE.md) - full automated pipeline specification
