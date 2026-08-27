# Inference Engine

AtlasLLM inference — generating text from a trained model.

## Overview

```
Prompt
  ↓
Tokenizer
  ↓
Token IDs
  ↓
Model Forward Pass
  ↓
Logits [B, T, V]
  ↓
Sampling Strategy
  ↓
Next Token
  ↓
Append to Sequence
  ↓
Repeat until EOS or max_tokens
```

## Generation Strategies

### Greedy Decoding

```python
next_token = torch.argmax(logits[:, -1, :], dim=-1)
```

Always picks the highest-probability token. Deterministic but often repetitive.

### Temperature Sampling

```python
logits = logits / temperature
probs = softmax(logits)
next_token = multinomial(probs)
```

| Temperature | Effect |
|-------------|--------|
| 0.0 | Equivalent to greedy |
| 0.5 | More focused |
| 1.0 | Neutral |
| 1.5 | More random |
| 2.0 | Very random |

### Top-k Sampling

Restrict sampling to the `k` highest-probability tokens:

```python
top_k_values, top_k_indices = torch.topk(probs, k=50)
# Zero out all tokens except top-k
# Sample from the remaining distribution
```

### Top-p (Nucleus) Sampling

Sample from the smallest set of tokens whose cumulative probability exceeds `p`:

```python
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative = torch.cumsum(sorted_probs, dim=-1)
# Keep tokens where cumulative probability <= p
# Renormalize and sample
```

| p value | Effect |
|---------|--------|
| 0.5 | Very focused |
| 0.9 | Standard |
| 0.95 | Diverse |
| 1.0 | No filtering |

## KV Cache

Without caching, every generation step recomputes attention for all previous tokens:

```
Step 1: compute attention for [t_0]
Step 2: compute attention for [t_0, t_1]
Step 3: compute attention for [t_0, t_1, t_2]
```

With KV caching, only the new token's Q/K/V is computed, and previous K/V are reused:

```
Step 1: compute K_0, V_0 → cache
Step 2: compute K_1, V_1 → append to cache → attend with [K_0, K_1]
Step 3: compute K_2, V_2 → append to cache → attend with [K_0, K_1, K_2]
```

**Performance gain:** ~2-5x speedup for long sequences.

**Memory cost:** Stores K and V tensors for all layers and all previous positions.

## Streaming Generation

Instead of waiting for the full sequence, emit tokens as they are generated:

```
"The"
" cat"
" sat"
" on"
" the"
" mat"
"."
```

Streaming is an inference-engine feature, not a model feature.

## EOS Handling

Stop generation when:

1. `<eos>` token is generated
2. Maximum sequence length is reached
3. User-defined stop sequence is found

## Interface

The engine loads a checkpoint and completes prompts. The tokenizer and model
configuration come from the training config (explicitly via ``--config``, or
from the checkpoint itself for runs trained on Stage 6+ of the pipeline):

```python
from inference.engine import InferenceEngine

engine = InferenceEngine.from_checkpoint(
    "checkpoints/best.pt",
    config="configs/debug.yaml",  # optional for Stage 6+ checkpoints
)

output = engine.generate(
    prompt="The capital of France is",
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    seed=42,              # reproducible sampling
    use_cache=True,       # KV cache on by default
    eos=True,             # stop on <eos>
    stop_sequences=("END",),
)
# output.text          -> the full decoded text
# output.token_ids     -> prompt + generated ids
# output.finished_reason -> "max_len" | "eos" | "stop_string"
```

Streaming emits one decoded chunk per generated token via ``engine.stream(...)``
(same arguments as ``generate``).

## CLI Usage

One-shot completion:

```bash
python -m inference.generate \
    --checkpoint checkpoints/best.pt \
    --config configs/debug.yaml \
    --prompt "The capital of France is" \
    --max-tokens 50 --temperature 0.8 --top-k 50
```

Interactive chat (streaming):

```bash
python -m scripts.chat \
    --checkpoint checkpoints/best.pt \
    --config configs/debug.yaml
```

## Design Decisions

1. **Separate from training** — Inference loads checkpoints without importing
   the training pipeline (AGENTS.md Rule 10); sampling/engine live under
   ``inference/``, the interactive REPL under ``scripts/``.
2. **Composable strategies** — Temperature, top-k, and top-p combine in
   ``sample_next_token``; temperature 0.0 is greedy.
3. **KV cache is default on and verified** — cached stepwise logits must equal
   full recomputation (mask rows are sliced from the *end* of the causal
   matrix because cached queries sit at the newest positions); a prefill's own
   last-position logits sample the first generated token so the last prompt
   token is never fed twice. ``--no-cache`` disables it for debugging.
4. **Streaming is opt-in** — ``engine.stream`` yields chunks without buffering
   the whole continuation.
5. **First token from the prefill** — prefill logits already predict the token
   after the prompt, so the engine samples it directly instead of re-running
   the last prompt token (which would double-count it in the cache).

## Verification

- Greedy always returns the argmax; temperature 0 == greedy; top-k=1 forces
  the top id; tight top-p keeps only the dominant token.
- Stepwise cached decoding equals full recomputation to 1e-5 per logit.
- Cached and non-cached generation produce identical tokens/text (same seed).
- Long prompts are truncated to the newest ``context_length - max_new_tokens``
  tokens so generation always has room.
- 18 tests in ``tests/test_generation.py``; full suite: 82 passed.
