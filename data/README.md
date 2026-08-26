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

For each dataset added, record:

| Field          | Value              |
|----------------|--------------------|
| Dataset name   |                    |
| Source URL     |                    |
| License        |                    |
| Download date  |                    |
| Raw size       |                    |
| Language       |                    |
| Preprocessing  |                    |
| Token count    |                    |

## Adding a Dataset

1. Place raw files in `data/raw/`
2. Document provenance in this README
3. Run preprocessing pipeline
4. Store results in `data/processed/`
