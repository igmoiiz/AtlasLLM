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
| ~Params | 12.98M (measured) |
| Measured VRAM | ~270 MB peak (fwd+bwd, batch 2 × ctx 128) |

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
Stage 4 — Transformer Implementation [x]
Stage 5 — Training Pipeline        [x]
Stage 6 — Real Training            [~] (full 100k small.yaml run live)
Stage 7 — Inference                [x]
Stage 8 — Evaluation + Documentation [ ]
```

**Current status:** Stages 1-7 complete (inference built before Stage 6 per user's milestone reorder). Rule 24 tiny-overfit passed (loss to 0.039). Stage 6 full training on small.yaml is running in the background (see Stage 6 section below); chat-test progress after checkpoints.

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

### Stage 4 — Transformer Implementation (completed this session)

40. ✅ `model/config.py` — frozen `ModelConfig` dataclass (vocab_size, context_length, d_model, n_layers, n_heads, d_ff, dropout, bias) + validation (n_heads divides d_model, positivity, dropout range) + `from_dict` for YAML `model:` sections; unknown fields rejected
41. ✅ `model/normalization.py` — LayerNorm from explicit tensor math (population variance, learns gamma/beta; no nn.LayerNorm)
42. ✅ `model/attention.py` — MultiHeadCausalAttention: Q=XWq/K=XWk/V=XWv, scores = QKᵀ/√d_k, explicit triu -inf causal mask, softmax, concat + Wo; dropout on probs + residual
43. ✅ `model/feed_forward.py` (GELU), `model/embeddings.py` (TokenEmbedding), `model/positional_encoding.py` (learned pos emb), `model/transformer_block.py` (pre-norm: x = x + attn(ln1(x)); x = x + ffn(ln2(x)))
44. ✅ `model/atlas_llm.py` — AtlasLLM (embed + pos → N blocks → ln_f → lm_head(no bias)); pure: tensors in, logits out; rejects seq > context_length
45. ✅ `training/loss.py` — `lm_cross_entropy` flattens any [.., V] logits vs targets, `ignore_index` for padding (unused)
46. ✅ Tests: 53 passing (24 new: attention output shape, causal invariance at attention + model level, numerical match vs naive per-head reference, grad flow, per-head splits, single-token, model shapes, loss = log V on uniform logits, loss floor with perfect pred, state_dict roundtrip = checkpoint reload, deterministic seed init, context-length guard, pos-emb learnable/differing, no weight tying, config validation)
47. ✅ Gotchas: torch RNG consumed by forward — seed BEFORE construction AND reuse input tensor in tests; loss must accept 2D logits too
48. ✅ GPU smoke (small config): params 12,982,784 (13M); loss drops 9.83→9.43 over 2 steps; AdamW step works; peak GPU 267.6 MB; logits [2,128,16000]
49. ✅ Corrected stale "~5.5M" estimates in architecture.md + AGENT_CONTEXT → measured 13M (~5.5M assumed weight tying or smaller head)

### Stage 5 — Training Pipeline (completed this session)

50. ✅ `training/scheduler.py` — `lr_prefactor(step, warmup, max)` pure function (linear warmup → cosine to 0); `build_lr_scheduler` = LambdaLR wrapper; 6 property tests
51. ✅ `training/checkpoint.py` — single-file `Checkpoint` dataclass; `save_checkpoint` atomic (.tmp + rename); `load_checkpoint` weights_only=True, restores model/optimizer/scheduler in place; raises FileNotFoundError on missing file
52. ✅ `training/trainer.py` — `Trainer` owns the loop: AdamW (betas (0.9,0.95), eps 1e-8), warmup+cosine via config, grad clip, per-epoch reseeded shuffle (seed+epoch), evaluate() (no-grad, capped max_val_batches), best.pt on val improvement, last.pt periodic (+final), metrics.jsonl + console + optional TB; resume no-op guard
53. ✅ `training/train.py` CLI — `--config/--resume/--steps`; device auto→cuda/cpu; dtype map; run dir `checkpoint.dir/run_<ts>/` with config.yaml copy + `reproducibility.json` (seed/model/training/data/tokenizer/torch/cuda/python/hardware/dataset_windows — rule 25)
54. ✅ Configs: added `logging.max_val_batches` (debug 50, small/medium 200) + `logging.tensorboard` (small true, others false); NO `min_lr_ratio` field — scheduler decays cosine to 0, so the field would be dead config (rule 12)
55. ✅ Gotchas: LambdaLR bakes an implicit step at construction (peak LR sits at step-1 index); AdamW state_dict contains `momentum: None` → test helper must handle None; resuming a finished grid → don't print 0.0→0.0, explicit no-op message
56. ✅ Tests: 64 total (11 new: 6 scheduler + 3 checkpoint + 2 trainer incl. resume-continuation test)
57. ✅ GPU debug run (real data, 1280 vocab): loss 7.29→5.04 (ln1280=7.16 floor), best val 4.84, warmup/cosine lr trace visible, tok/s ~4000, 20MB GPU; resume from step-99 ckpt → ran, best val improved
58. ✅ Rule 24 tiny-overfit: 512-token stream memorized → loss 7.27→0.039 (300 steps, GPU); full path (dataset→tokenizer→model→loss→optimizer→backward) proven. PREREQUISITE MET; Stage 6 (real training) is authorized

---

## Stage 7 — Inference Engine (DONE: `67ad83e`, fix `88c99b7` — pushed)

**Milestone reorder (user request):** build inference BEFORE Stage-6 real training so the model can be manually tested by asking questions after every training session. Commit + push after this stage (fail-safe pattern unchanged).

59. ✅ `inference/sampling.py` — `greedy(logits)` (argmax); `sample_next_token(logits, temperature, top_k, top_p, rng)` batch-friendly; temp 0.0 == greedy, temp<0/top_k<1/top_p∉(0,1] raise; top-k via -inf mask, top-p nucleus (keep top id always) via scatter + renormalize
60. ✅ `inference/kv_cache.py` — `KVCache(n_layers)` storage only: full `[B,H,T,hd]` K/V per layer, `append`, `update(pairs)`, `pairs()` (None for never-seen layers), `length()`, `reset()`
61. ✅ `model/` KV-cache support (training path untouched, all old tests green):
    - `attention._attend(x, past_key, past_value)` → `(out, k, v)`; public `forward(x)` and `forward_with_cache` wrappers
    - **Mask bug caught by tests:** cached queries occupy rows `[P, P+t)` of the (P+t)² causal matrix → slice `triu(...)[p:p+t, :]`, NOT `[:t, :]`
    - `pos_emb(seq_len, device, offset=0)` picks up absolute positions after a prefill; `AtlasLLM.forward(input_ids, past_key_values=None)` → logits-only (old behavior) or `(logits, new_pairs)` (full K/V per layer); validates pair count + context bound
62. ✅ `inference/engine.py` — `InferenceEngine(model, tokenizer, device)`; `from_checkpoint(checkpoint_path, config=None, tokenizer=None, device)` (config = yaml path|dict|ModelConfig; tokenizer resolved from explicit arg → config `data.tokenizer_path` → ckpt config — needed because Stage 5 checkpoints store `config={}`); vocab-mismatch guard (tokenizer ≤ model); `generate()` → `Generation(text, token_ids, finished_reason ∈ max_len|eos|stop_string)`; `stream()` yields decoded chunks; `_prepare` truncates to newest `ctx - max_new_tokens` tokens; `_iter_tokens` bounded by context; seed → torch.Generator
63. ✅ **First-token-from-prefill design (another bug caught by tests):** the prefill forward's own last-position logits already predict the first new token; feeding the last prompt token again double-counts it (logits diverged ~2.3 → cached top-1 (1154) ≠ recompute argmax (445)). Sampling `pending` from prefill logits makes cached == full recompute
64. ✅ `inference/generate.py` CLI (`--checkpoint --config --prompt/--max-tokens/--temperature/--top-k/--top-p/--seed/--no-cache`); `scripts/chat.py` interactive streaming REPL (q/quit exits, empty line skipped); `inference/__init__.py` re-exports (generate NOT imported → runpy-safe)
65. ✅ `training/trainer.py`: checkpoints now stash `self.config` (3 save sites) so Stage 6+ ckpts load without `--config`
66. ✅ Tests: 82 total (18 new). Test-design lesson: ±20/−20 one-hot `lm_head` trick is unreliable (argmax = sign of Σh) → engine control-flow tests monkeypatch `inference.engine.sample_next_token` instead; sampling math tested separately
67. ✅ Manual smoke (real debug checkpoint `run_20260827-172944/best.pt`, val 4.7989): `python -m inference.generate` prints continuation; cached vs `--no-cache` identical; chat REPL streams and exits. Output is gibberish — expected at ~110 steps; the point is to watch it improve across Stage-6 training sessions

**Next Session: Stage 6 — Real Training (small.yaml) + manual chat Q/A after training**
1. `python -m training.train --config configs/small.yaml` (13M params; measured ~26k tok/s → 100k steps ≈ 2-2.5 h on GTX 1070, GPU 216 MB)
2. After each training run: `python -m scripts.chat --checkpoint <run>/last.pt` and ask the same questions to watch improvement
3. Monitor overfit via val loss vs train loss; resume with `--resume run_<ts>/last.pt` if interrupted (scheduler restores position correctly — verified live)
4. Do NOT change architecture; only hyperparameters if clearly broken
5. Stage 8 (evaluation) can start once small training produces meaningful checkpoints

---

## Stage 6 — Real Training (in progress)

**Post-Stage-7 fix found here (shipped `88c99b7`):** top-p nucleus combined with top-k was broken. The old code marked `remove[..., 0] = False` on the *rank* axis but then built the scatter source with `torch.where(remove, -inf, scores)` — which indexes *vocab* positions by rank flags. Forced-keep hit vocab token 0, usually already -inf after top-k masking → empty surviving set → softmax NaN → CUDA `input[0] != 0` device assert / `multinomial` "probability tensor contains inf, nan or <0". Fix: do the entire nucleus computation in sorted order (sort scores → softmax → cumsum → keep_rank mask → censor sorted_scores → scatter back). Regression test added (`test_top_k_combined_with_top_p_is_safe_and_contained`); suite grew 82 → 83.

**2000-step verification run** (`checkpoints/run_20260827-195547/`): loss 9.86 → 8.15, best val **8.3788**, GPU 216.6 MB, tok/s ~26k (≈13 steps/s; the old "~4k tok/s" was tiny-model scale). Checkpoints auto-load config (config-stash feature makes `--config` unnecessary post-Stage-7).

**Full 100k run (live since 2026-08-27 ~21:02, PID 2100):** resumed from the 2000-step `last.pt` — scheduler restored at the right position (LR re-entered cosine at 3.00e-4 peak-adjacent), run dir `checkpoints/run_20260827-210202/`. Logs: `%TEMP%\opencode\small_full.out.log` / `.err.log`. Chat-test at checkpoints (every 2000 steps) and after the run; expected uniform floor ln16000 = 9.68 and only-slightly-below-fabric at 2k steps → coherence improves with tokens seen.

**Power cutoff mid-run (~step 24700) + recovery:**
- Process died with the cut; GPU verified idle. `best.pt` (the ONLY recoverable snapshot) = step 13000, val 6.8413 — the run's true best; ~14 min of steps (13001–24700) re-done.
- **Root-cause bug:** periodic `last.pt` NEVER fired in any config — the trainer read `logging.save_every` (trainer.py) while all configs define `checkpoint.save_every`. Existing tests masked it (they only assert last.pt exists at run end, and the run-end save always writes it). Fixed: trainer now reads `checkpoint.save_every` (keeps `logging` fallback) + regression test (`test_save_every_reads_checkpoint_section`) → 84 tests. Commit `7fe6c50`, pushed.
- **Resumed run:** `python -m training.train --config configs/small.yaml --resume checkpoints/run_20260827-210202/best.pt` → new dir `checkpoints/run_20260827-225205/` (PID 804), clean metrics.jsonl, LR restored correctly (2.89e-4 → cosine continuity, verified), train/val back on the pre-cutoff trajectory (val 6.843 @ step 14000 ≈ pre-cutoff 6.841@13000). Periodic last.pt verified live (fired at absolute step 14000). Logs: `%TEMP%\opencode\small_recovered.{out,err}.log`. Future cut cost ≤ 2000 steps (~2.5 min).

**Second power cutoff (~step 21400 in run 225205) + recovery:**
- Resumed from `run_20260827-225205/last.pt` (step 20000, periodic save fired correctly; only ~1400 steps lost). Verified `last.pt` contains step 20000 with no `.tmp` debris.
- Relaunched: `python -m training.train --config configs/small.yaml --resume checkpoints/run_20260827-225205/last.pt` → **live run** `checkpoints/run_20260827-231105/` (PID 8224, logs `%TEMP%\opencode\small_resume3.{out,err}.log`). RESUMED AT 20K, NOT 13K (best.pt) — so it continues past the old best. By step 30k: train loss ~4.8 falling, val ~7.07 (drifting up from 6.84 best at step 13k → mild overfit signal, tracked in docs/training.md). LR 2.39e-4, cosine position correct. GPU 216.6 MB, ~20-28k tok/s.
- **Resume gotcha:** each resume makes a NEW `run_<ts>/` dir; `best.pt` in it starts tracking from the resume point. Historical best (6.8413 @ 13000) lives in `run_20260827-210202/best.pt` — preserved if a final chat-test uses that instead.

**Docs overhaul (commit `a686c5d`, pushed):**
- Rewrote README + all `DOCUMENTATION/*.md`. Added a "For a beginner" section to every doc, honest status ("Done, tested" vs "PLANNED" for the stub packages), cross-links (every page links to index + neighbors; verified all links resolve), removed fabricated numbers (README ~5.5M params → measured 13M; hardware.md invented ~200k tok/s → measured ~26k; experiments.md invented comparison table → placeholders), removed em/en-dashes per user's no-AI-slop rule. Marked evaluation/safety/harness/monitoring as stubs (they truly are: empty .py files).

---

## User Preferences

- Clean, well-documented project
- Scripts over notebooks
- CLI-driven workflow
- Proprietary license with strict terms
- Realistic about model capabilities (not claiming ChatGPT-level)
- Educational focus — understanding > performance
- Will scale to Medium after Small (13M, measured) works

---

## Next Session: Stage 6 — training running; chat-test + evaluate

1. Poll `%TEMP%\opencode\small_resume3.{out,err}.log` and `checkpoints/run_20260827-231105/metrics.jsonl` (PID 8224)
2. When the run finishes (~step 100000): chat-test the final `last.pt` with the SAME questions used on the 2k-step model (and optionally compare against `run_20260827-210202/best.pt`, the true historical best at val 6.8413)
3. Watch train vs val loss for overfit (val has drifted 6.84 → 7.07 by step 30k; decide whether the mild overfit changes the plan)
4. Do NOT change architecture during this stage; only hyperparameters if clearly broken
5. Stage 8 (evaluation) starts once small training produces meaningful checkpoints. Docs for it are already drafted in `DOCUMENTATION/experiments.md`, `harness.md` (honestly marked PLANNED; `evaluation/`, `safety/`, `harness/`, `monitoring/` are empty stubs)
