# AtlasLLM — Data

## Purpose

This directory holds all datasets used for tokenizer training and model pretraining.

## Structure

```text
data/
├── raw/          # Immutable original data — never modify
├── interim/      # Intermediate processing results
├── processed/    # Final tokenized training data
└── README.md     # This file
```

## Rules

- `raw/` is **immutable**. Never modify files in `raw/`.
- `processed/` must be reproducible from `raw/` and preprocessing config.
- Large data files are git-ignored. Document provenance here.

## Data Provenance

| Field          | Value              |
|----------------|--------------------|
| Dataset name   | WikiText-2 (raw)   |
| Source URL     | https://huggingface.co/datasets/Salesforce/wikitext |
| License        | CC BY-SA 3.0       |
| Download date  | 2026-08-27         |
| Raw size       | 13.4 MB (train 10.9 + valid 1.1 + test 1.3 MB) |
| Language       | English            |
| Preprocessing  | None — raw article text, one line per row (`wikitext-2-raw-v1`) |
| Token count    | ~2.3M (train+valid, 16k BPE) |

**Raw files:** `data/raw/wikitext-2-raw/{train,valid,test}.txt` — immutable sources.

**Interim:** `data/interim/wikitext-2-raw/corpus.txt` = train + validation text (used to train the tokenizer). Reproducible by concatenating the two raw files; the test split is excluded from tokenizer training.

## Processed Datasets

Tokenized by `python -m data_pipeline.preprocessing --config configs/<name>.yaml` → raw uint16 `.bin` files + `meta.json` (vocab, context, per-split token/sequence counts, tokenizer path, created). Bins are tokenizer- and context-specific:

| Config   | Tokenizer (vocab) | Context | Split       | Tokens      | Sequences |
|----------|-------------------|---------|-------------|-------------|-----------|
| small    | small (16,000)    | 256     | train/val/test | 2,116,813 / 219,289 / 258,561 | 8,268 / 856 / 1,010 |
| debug    | debug (1,280)     | 32      | train/val/test | 6,910,924 / 723,663 / 817,836 | 215,966 / 22,614 / 25,557 |

Outputs are git-ignored (`data/processed/`); `meta.json` records everything needed to reproduce them.

## Adding a Dataset

1. Place raw files in `data/raw/`
2. Document provenance in this README
3. Run preprocessing pipeline
4. Store results in `data/processed/`
