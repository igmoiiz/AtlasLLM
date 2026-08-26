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
- **Python:** 3.14.3 (64-bit)
- **PyTorch:** 2.12.0+cu126
- **CUDA:** 12.6
- **GPU:** NVIDIA GTX 1070 — 8.6 GB VRAM
- **CPU:** Intel Xeon E3-1270 v3 (4C/8T, 3.5 GHz)
- **RAM:** 32 GB DDR3 1600 MHz
- **Storage:** 512 GB SSD + 1 TB HDD

---

## Installed Packages

| Package | Status |
|---------|--------|
| torch | 2.12.0+cu126 |
| PyYAML | 6.0.3 |
| numpy | MISSING |
| tokenizers | MISSING |
| datasets | MISSING |
| tensorboard | MISSING |
| matplotlib | MISSING |
| pytest | MISSING |
| ruff | MISSING |

**Action required:** Install all missing packages via `pip install -e ".[all]"`

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
Stage 1 — Environment Setup        [ ]
Stage 2 — Tokenizer                [ ]
Stage 3 — Dataset Pipeline         [ ]
Stage 4 — Transformer Implementation [ ]
Stage 5 — Training Pipeline        [ ]
Stage 6 — Real Training            [ ]
Stage 7 — Inference                [ ]
Stage 8 — Evaluation + Documentation [ ]
```

**Current status:** Scaffolding complete. Stage 1 not yet started.

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

## Next Session: Start Stage 1 — Environment Setup

1. Install all missing packages: `pip install -e ".[all]"`
2. Verify PyTorch + CUDA works
3. Set up ruff for linting
4. Set up pytest for testing
5. Run a quick smoke test
6. Commit: "feat: complete environment setup"
