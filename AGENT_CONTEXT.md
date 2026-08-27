# AGENT_CONTEXT.md — Project Memory

This file is the agent's persistent memory across sessions. Update it after every significant session.

---

## Last Updated: 2026-08-27

---

## Project Identity

- **Name:** AtlasLLM
- **Type:** Dense decoder-only Transformer language model
- **Purpose:** Educational systems-learning project — understand every component
- **Author:** Moiz Baloch (khanmoaiz682@gmail.com, +923067892235)
- **Successor:** AtlasMoE (MoE extension after dense baseline is complete)

---

## Repository

- **GitHub:** https://github.com/igmoiiz/AtlasLLM
- **HuggingFace:** https://huggingface.co/igmoiiz/AtlasLLM (private)
- **Git remotes:** `origin` → GitHub, `huggingface` → HuggingFace
- **License:** Proprietary (LICENSE.md under Moiz Baloch)

---

## Development Environment

- **OS:** Windows 10/11
- **Python:** 3.14.3 (64-bit, system install — used directly, no venv)
- **PyTorch:** 2.12.0+cu126
- **CUDA:** 12.6 (driver 582.66, CUDA 13.0 runtime visible)
- **GPU:** NVIDIA GTX 1070 — 8.6 GB VRAM
- **CPU:** Intel Xeon E3-1270 v3 (4C/8T, 3.5 GHz)
- **RAM:** 32 GB DDR3 1600 MHz
- **Storage:** 512 GB SSD + 1 TB HDD

## Installed Packages

| Package | Status |
|---------|--------|
| torch | 2.12.0+cu126 |
| PyYAML | 6.0.3 |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| tokenizers | 0.23.1 |
| datasets | 5.0.1 |
| tensorboard | 2.21.0 |
| matplotlib | 3.11.1 |
| pytest | 9.1.1 (+pytest-cov) |
| ruff | 0.16.4 |

**Status:** All installed into system Python 3.14.3 via `pip install -e ".[all]"` (installed editable `atlas-llm 0.1.0`). The `.venv` was deleted on user instruction. Note: a pre-existing global `numba` (not a project dependency) conflicts with numpy 2.5.2 — ignored, does not affect AtlasLLM.

---

## Model Configuration

### AtlasLLM-Small (primary target)

| Parameter | Value |
|-----------|-------|
| vocab_size | 16,000 |
| context_length | 256 |
| d_model | 256 |
| n_layers | 6 |
| n_heads | 8 |
| head_dim | 32 |
| d_ff | 1,024 |
| dropout | 0.1 |
| bias | false |
| ~Params | ~5.5M |
| Est. VRAM | ~560 MB |

### AtlasLLM-Medium (next target after Small is complete)

| Parameter | Value |
|-----------|-------|
| vocab_size | 16,000 |
| context_length | 512 |
| d_model | 384 |
| n_layers | 8 |
| n_heads | 8 |
| head_dim | 48 |
| d_ff | 1,536 |
| dropout | 0.1 |
| bias | false |
| ~Params | ~25M |
| Est. VRAM | ~2 GB |

**Decision:** User approved scaling to Medium after Small v0.1 is complete.

---

## Dataset Strategy

### Phase 1 (overfit test + baseline): WikiText-2

- **HuggingFace ID:** `Salesforce/wikitext`
- **Config:** `wikitext-2-raw-v1`
- **Size:** ~11 MB (~2M tokens)
- **License:** CC BY-SA 3.0
- **Quality:** Excellent (verified Wikipedia articles)
- **Download:** `load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")`

### Phase 2 (real pretraining): WikiText-2 + OpenWebText

- **OpenWebText HuggingFace ID:** `Skylion007/openwebtext`
- **Sample:** Stream ~50-90 MB subset
- **License:** CC0-1.0 (public domain)
- **Combined target:** 60-100 MB total

### Why this approach:

1. WikiText-2 validates the entire pipeline fast (~30 min training)
2. Known baseline perplexity (~30-40 PPL) for comparison
3. OpenWebText adds diversity for real pretraining
4. Both are permissively licensed and well-documented

---

## Innovation Policy

### Phase 1 — Small (5.5M) + Medium (25M): Traditional, From Scratch

- Traditional decoder-only Transformer architecture
- Every component implemented from scratch — no `nn.Transformer`, no `nn.MultiheadAttention`, no HuggingFace model libraries, no pre-built transformer layers
- Only torch primitives (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, tensor ops) as building blocks
- Full mathematical transparency — every equation visible in code
- No black-box replacements (AGENTS.md Rule 34)
- The goal is to understand every component completely

### Phase 2 — After Medium 25M is Complete: Innovation

- Novel attention patterns, efficient FFN variants, positional encoding improvements
- Training innovations — curriculum learning, adaptive scheduling, distillation
- Memory efficiency — gradient checkpointing, parameter sharing, activation recomputation
- Every innovation measured against the working traditional baseline
- No claimed improvements without measurements (AGENTS.md Rule 48)

---

## Key Decisions Made

1. **Notebooks removed** — replaced with scripts/ for CLI-driven workflow (version-controllable, reproducible, no hidden state)
2. **Scripts created:** attention_visualization.py, training_analysis.py, evaluation.py
3. **DOCUMENTATION/ folder created** — 10 markdown files covering all subsystems
4. **HF CLI authenticated** as `igmoiiz` — repo created and added as git remote
5. **Windows target** — all paths and commands use Windows conventions
6. **No premature optimization** — FP32 first, then test FP16 after verification
7. **Pre-normalization architecture** — x = x + attn(norm(x)), x = x + ffn(norm(x))
8. **Learned positional embeddings** — simpler for v1, RoPE/Sinusoidal in later experiments
9. **No weight tying initially** — keeps architecture simple
10. **From scratch only** — no pre-built transformer/encoder-decoder libraries, every component hand-written

---

## Implementation Stages (approved)

```
Stage 1 — Environment Setup        [x]
Stage 2 — Tokenizer                [x]
Stage 3 — Dataset Pipeline         [x]
Stage 4 — Transformer Implementation [ ]
Stage 5 — Training Pipeline        [ ]
Stage 6 — Real Training            [ ]
Stage 7 — Inference                [ ]
Stage 8 — Evaluation + Documentation [ ]
```

**Current status:** Stages 1-3 complete. WikiText-2 tokenized to uint16 .bin datasets (16k small + 1280 debug). Next: Stage 4 — Transformer Implementation.

---

## What Was Completed Today (2026-08-27)

1. ✅ Created full project directory structure (47+ files)
2. ✅ Created pyproject.toml with dependencies
3. ✅ Created 3 YAML configs: debug.yaml, small.yaml, medium.yaml
4. ✅ Created .gitignore (comprehensive)
5. ✅ Created README.md with project overview
6. ✅ Created data/README.md with provenance template
7. ✅ Wrote proprietary LICENSE.md under Moiz Baloch
8. ✅ Renamed AGENTS.md.md → AGENTS.md, CONTEXT.md.md → CONTEXT.md
9. ✅ Created HF repo (igmoiiz/AtlasLLM, private) and added as git remote
10. ✅ Deleted notebooks/ directory, created 3 replacement scripts
11. ✅ Created DOCUMENTATION/ folder with 10 documentation files
12. ✅ Updated README.md to reflect all changes

### Stage 1 — Environment Setup (completed this session)

13. ✅ Fixed `pyproject.toml` — added `[tool.setuptools.packages.find]` (unblocked editable install) and corrected `license` to match proprietary LICENSE.md
14. ✅ Deleted `.venv` (127 MB freed) per user instruction — system Python 3.14.3 is canonical
15. ✅ Installed all deps via `pip install -e ".[all]"` (~390 MB on disk; C: free 77.4 GB)
16. ✅ Verified: torch 2.12.0+cu126 CUDA enabled, GPU matmul OK (GTX 1070)
17. ✅ Verified: tokenizers, datasets, tensorboard, matplotlib, pytest, ruff all import
18. ✅ `ruff check .` — clean (fixed 2 pre-existing issues in placeholder scripts)
19. ✅ `pytest` — runs (0 tests collected; test files are still empty stubs)
20. ✅ Added `.obsidian/` to .gitignore (editor config)

### Stage 2 — Tokenizer (completed this session)

21. ✅ Implemented `tokenizer/vocabulary.py` — special tokens `<pad>/<unk>/<bos>/<eos>` (ids 0-3)
22. ✅ Implemented `tokenizer/tokenizer.py` — `AtlasTokenizer` wrapper (encode/decode/vocab/save/load/from_pretrained)
23. ✅ Implemented `tokenizer/train_tokenizer.py` — BPE training CLI with stats + config-vocab consistency checks
24. ✅ **Design:** char-level BPE, no pretokenizer/normalizer → exact lossless roundtrip; seeded 256-byte alphabet; unseen multibyte chars → `<unk>` (never silently dropped). Rejected ByteLevel (GPT-2 byte table corrupts multibyte chars) and byte_fallback (decode inserts spaces)
25. ✅ Downloaded WikiText-2-raw → `data/raw/wikitext-2-raw/` (immutable); corpus = train+valid in `data/interim/wikitext-2-raw/`
26. ✅ Trained tokenizers: small (16,000 vocab, 2.34M tokens, 12 MB corpus) + debug (1,280 vocab)
27. ✅ Configs: added `tokenizer:` sections; debug vocab 256→1280 (wiki char alphabet ≈1211+specials needs ≥1280)
28. ✅ Tests: 18 passing in `tests/test_tokenizer.py` (roundtrip, special ids, bos/eos, empty, unicode, unk, save/load)
29. ✅ `ruff check` clean; data provenance documented in `data/README.md`; `DOCUMENTATION/tokenizer.md` updated
30. ✅ Trained models saved to `tokenizer/model/small` and `tokenizer/model/debug` (git-ignored, reproducible via CLI)

### Stage 3 — Dataset Pipeline (completed this session)

31. ✅ Implemented `data_pipeline/dataset.py` — `TextDataset(torch.utils.data.Dataset)` memory-maps uint16 .bin; non-overlapping contiguous shifted windows (input `t0..t_{T-1}`, target `t1..t_T`); `len = (n-1)//T`
32. ✅ Implemented `data_pipeline/preprocessing.py` — `tokenize_to_bin` (whole-file encode → uint16, overflow guard) + `build_processed_data` (writes train/val/test .bin + `meta.json` with vocab/context/per-split counts/tokenizer path/created) + CLI (reuses `TextDataset.__len__` for sequence counts — no duplicated math)
33. ✅ Implemented `scripts/inspect_dataset.py` — verify .bin + meta.json (was an empty stub)
34. ✅ Configs: `data:` — added `test_path` + raw text sources (`train/val/test_text`); debug splits moved to `data/processed/debug/` (**different tokenizer/context ⇒ separate bins**, prevents clobbering small)
35. ✅ Built datasets: small (16k vocab, ctx 256) train 2.12M/val 219k/test 259k tokens; debug (1280, ctx 32) train 6.91M/val 724k/test 818k
36. ✅ Verified end-to-end: DataLoader batch shapes `[8, 256]` long, next-token shift OK, ids < vocab
37. ✅ Tests: 29 passing (11 new in `tests/test_dataset.py` — window math, shift contract, arbitrary index, T=1, empty/too-short, missing file, bin==encode, meta.json, roundtrip through dataset)
38. ✅ Gotcha: `python -m pkg.sub` fails (runpy RuntimeWarning) if `__init__.py` imports the target submodule — `data_pipeline/__init__.py` re-exports only `TextDataset` (mirrors tokenizer package)
39. ✅ `ruff check` clean; `data/README.md` + `DOCUMENTATION/dataset.md` synced (cleaning pipeline documented as deferred for Phase-2 corpora)

---

## User Preferences

- Clean, well-documented project
- Scripts over notebooks
- CLI-driven workflow
- Proprietary license with strict terms
- Realistic about model capabilities (not claiming ChatGPT-level)
- Educational focus — understanding > performance
- Will scale to Medium (25M) after Small (5.5M) works

---

## Next Session: Stage 4 — Transformer Implementation

1. Implement `model/` from scratch in pure PyTorch (pre-normalization per CONTEXT §19-20): `embeddings.py`, `attention.py` (causal, explicit mask, multi-head), `feed_forward.py` (GELU), `normalization.py` (LayerNorm), `transformer_block.py`, `atlas_llm.py` (full model + LM head). Learned positional embeddings (Decision 8)
2. Shape contract: `[B,T] → [B,T,D] → logits [B,T,V]`; explicit causal mask
3. Implement `training/loss.py` (cross-entropy, only after taking `loss.py` into account — CONTEXT §22)
4. Mandatory tests (AGENTS.md Rule 23): shapes, causal masking (future tokens do not affect earlier predictions — must test explicitly), attention dimensions, gradient flow, block forward, full-model forward, loss
5. Commit: "feat: implement Transformer architecture"
