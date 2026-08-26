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

```python
from inference.engine import InferenceEngine

engine = InferenceEngine.from_checkpoint("checkpoints/best.pt")

output = engine.generate(
    prompt="The capital of France is",
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
)
```

## CLI Usage

```bash
python -m inference.generate \
    --checkpoint checkpoints/best.pt \
    --prompt "The capital of France is" \
    --max-tokens 50 \
    --temperature 0.8 \
    --top-k 50
```

## Design Decisions

1. **Separate from training** — Inference loads checkpoints without importing training pipeline
2. **Composable strategies** — Temperature, top-k, and top-p can be combined
3. **KV cache is optional** — Disabled for short prompts, enabled for long generation
4. **Streaming is opt-in** — For API/integration use cases
