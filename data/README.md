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

---

**WikiText-103 (raw)** — added 2026-08-28 as the overfitting fix for the 13M model (WikiText-2 = 2.1M tokens ≈ 97 epochs of memorization at ~2,048 tokens/step over 100k steps; WikiText-103 ≈ 100M tokens removes the data bottleneck).

| Field          | Value              |
|----------------|--------------------|
| Dataset name   | WikiText-103 (raw) |
| Source URL     | https://huggingface.co/datasets/Salesforce/wikitext |
| Config         | `wikitext-103-raw-v1` |
| License        | CC BY-SA 3.0       |
| Download date  | 2026-08-28         |
| Raw size       | 539.2 MB (train 538.3 + valid 1.0 + test 1.2 MB) |
| Language       | English            |
| Preprocessing  | None — raw article text, one line per row |
| Token count    | ~103M (train, 16k BPE estimate at ~0.194 tok/char) |

**Raw files:** `data/raw/wikitext-103-raw/{train,valid,test}.txt` — immutable sources.

**Interim:** `data/interim/wikitext-103-raw/corpus.txt` = train + validation text. `data/interim/wikitext-103-raw/corpus_sample.txt` = every 10th line (1/10 systematic sample, ~53.7 MB): the full 539 MB no-pretokenizer BPE pass exceeds 32 GB RAM, but a 16k vocabulary is equivalent for this domain. Both reproducibly derived from `raw/`.

---

**AtlasTiny (TinyStories)** — added 2026-09-12 as the automated data-pipeline milestone (DATA_PIPELINE.md): acquire → process → tokenizer → tokenize → finalize, driven by `configs/tinystories.yaml`.

| Field          | Value              |
|----------------|--------------------|
| Dataset name   | AtlasTiny-v1       |
| Source         | https://huggingface.co/datasets/roneneldan/TinyStories |
| License        | CC BY-SA 4.0       |
| Download date  | 2026-09-12         |
| Raw cache      | `data/raw/tinystories/records.jsonl` (185.8 MB, 199,955 docs) |
| Language       | English only (language filter) |
| Post-filter    | 197,555 docs kept, 2,400 duplicates rejected |
| Token count    | train 22,894,678 / val 1,259,977 / test 1,281,431 (16k BPE, ctx 256) |
| Checksums      | `data/processed/tinystories/manifest.json` (validation gate + provenance) |

**Interim:** `data/interim/tinystories/{train,val,test}.jsonl` + `processing_stats.json`. Tokenizer corpus: `data/interim/tinystories/corpus_sample.txt` = every 10th line of the full corpus (1/10 systematic sample, ~16 MB) — the full 162 MB no-pretokenizer BPE pass risks exceeding 32 GB RAM; a 16k vocabulary is equivalent for this domain (same reasoning as WikiText-103).

**Rules 24 gate:** the slice-overfit run `checkpoints/tinystories-overfit/run_20260912-194102/` verified train loss collapses 9.856 → 0.247 with a +13.55 nats train-vs-heldout memorization gap, proving the whole dataset→tokenizer→model→loss→optimizer→backward path before any pretraining proceeds.

---

**AtlasBase (FineWeb-Edu)** — added 2026-09-12 as the first real pretraining corpus (Base tier, 100–300M tokens), a single-source milestone of DATA_PIPELINE.md: the full automated pipeline on one edu-quality-filtered HuggingFace source, no mixture machinery yet. Driven by `configs/atlasbase.yaml`.

| Field          | Value              |
|----------------|--------------------|
| Dataset name   | AtlasBase-v1       |
| Source         | https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu |
| Source split   | train (streamed, first 200,000 rows) |
| License        | CC BY 4.0 (FineWeb-Edu) |
| Download date  | 2026-09-12         |
| Raw cache      | `data/raw/fineweb_edu/records.jsonl` (979.6 MB, 200,000 docs) |
| Language       | English only (language filter; 42 docs unknown-language rejected) |
| Post-filter    | 199,957 docs kept, 1 duplicate rejected |
| Token count    | train 186,058,386 / val 10,062,577 / test 11,013,884 (16k BPE, ctx 256) |
| Checksums      | `data/processed/atlasbase/manifest.json` (validation gate + provenance) |

**Interim:** `data/interim/fineweb_edu/{train,val,test}.jsonl` + `processing_stats.json`. Tokenizer corpus: `data/interim/fineweb_edu/corpus_sample.txt` = every 10th processed train doc (1/10 systematic sample, ~88 MB, 17,975 docs) — the full ~968 MB no-pretokenizer BPE pass exceeds the machine's safe RAM margin; a 16k vocabulary is equivalent for this domain (same reasoning as WikiText-103 / TinyStories).

## Processed Datasets

Tokenized by `python -m data_pipeline.pipeline --config configs/<name>.yaml` → raw uint16 `.bin` files + `meta.json` (vocab, context, per-split token/sequence counts, tokenizer path, created) + `manifest.json` (validation gate). Bins are tokenizer- and context-specific:

| Config       | Tokenizer (vocab) | Context | Split       | Tokens      | Sequences |
|--------------|-------------------|---------|-------------|-------------|-----------|
| small        | small (16,000)    | 256     | train/val/test | 2,116,813 / 219,289 / 258,561 | 8,268 / 856 / 1,010 |
| wikitext103  | wikitext103 (16,000) | 256 | train/val/test | 107,946,880 / 224,705 / 258,166 | 421,667 / 877 / 1,008 |
| debug        | debug (1,280)     | 32      | train/val/test | 6,910,924 / 723,663 / 817,836 | 215,966 / 22,614 / 25,557 |
| tinystories  | tinystories (16,000) | 256 | train/val/test | 22,894,678 / 1,259,977 / 1,281,431 | 89,432 / 4,921 / 5,005 |
| atlasbase    | atlasbase (16,000)   | 256 | train/val/test | 186,058,386 / 10,062,577 / 11,013,884 | 726,790 / 39,306 / 43,022 |

Outputs are git-ignored (`data/processed/`); `meta.json` and `manifest.json` record everything needed to reproduce them.

## Adding a Dataset

1. Add a registry source + processing filters under `pipeline:` in the config
2. Run `python -m data_pipeline.pipeline --config configs/<name>.yaml` (resumable, `--force` rebuilds)
3. Document provenance in this README
4. Store results in `data/processed/`
