# Architecture

AtlasLLM is a small dense decoder-only Transformer language model.

## For a beginner

A Transformer is a neural network arranged as a stack of repeated "blocks". Each block does two jobs:

1. **Look at the other words** (attention). The word at position *i* gathers helpful information from all words that came before it.
2. **Think locally** (feed-forward network). A small two-layer network transforms each word's representation on its own.

Every block output is added to its input (a "residual" connection), which lets information flow through many blocks without degrading. After the last block, a final layer turns each position into a score for every word in the vocabulary; the highest score is the model's next-word guess.

This model is **decoder-only**: it only ever reads the words to the left, never the words to the right. That property makes it a language model (predict the next word). It is **dense** (every block uses all of its parameters for every token) and **small** (13M parameters, trainable on a GTX 1070).

Related reading: the rest of this page is the technical description. See [index.md](index.md) for the full picture.

## Overview

```
Input IDs [B, T]
     ↓
Token Embedding [B, T, D]
     +
Position Embedding [B, T, D]
     ↓
Transformer Block × N
     ├── LayerNorm
     ├── Multi-Head Causal Self-Attention
     ├── Residual Connection
     ├── LayerNorm
     ├── Feed-Forward Network (GELU)
     └── Residual Connection
     ↓
Final LayerNorm [B, T, D]
     ↓
LM Head (Linear) [B, T, V]
     ↓
Vocabulary Logits
```

**B** = batch size, **T** = sequence length, **D** = model dimension, **V** = vocabulary size

## Default Configuration (AtlasLLM-Small)

| Parameter       | Value  | Description |
|-----------------|--------|-------------|
| vocab_size      | 16,000 | Subword vocabulary |
| context_length  | 256    | Maximum sequence length |
| d_model         | 256    | Hidden dimension |
| n_layers        | 6      | Transformer blocks |
| n_heads         | 8      | Attention heads |
| head_dim        | 32     | Per-head dimension (d_model / n_heads) |
| d_ff            | 1,024  | Feed-forward inner dimension |
| dropout         | 0.1    | Dropout rate |
| bias            | false  | No bias in linear layers |

## Parameter Count

~13M parameters (AtlasLLM-Small, measured 12,982,784). Breakdown: token
embedding 4.10M + LM head 4.10M (no weight tying) + 6 transformer blocks 4.72M
+ positional embeddings 0.07M. Peak GPU memory during a forward+backward step
with batch 2 × context 128: ~268 MB.

## Components

### Token Embedding

```python
nn.Embedding(vocab_size, d_model)  # 16000 → 256
```

Converts discrete token IDs into continuous vectors.

- **Input:** token IDs `[B, T]`
- **Output:** embeddings `[B, T, D]`

### Positional Embedding

```python
nn.Embedding(context_length, d_model)  # 256 → 256
```

Learned positional embeddings. Each position gets its own trainable vector.

- **Input:** position indices `[0, 1, 2, ..., T-1]`
- **Output:** position vectors `[B, T, D]`

The final input to the Transformer is:

```
x = token_embedding + positional_embedding
```

### Multi-Head Causal Self-Attention

Each attention head computes:

```
Q = X · W_q    # [B, T, D] → [B, H, T, head_dim]
K = X · W_k    # [B, T, D] → [B, H, T, head_dim]
V = X · W_v    # [B, T, D] → [B, H, T, head_dim]

Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

**Causal mask:** Token at position `i` can only attend to positions `0, 1, ..., i`.

```
Mask:
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

After attention, heads are concatenated and projected:

```
output = Concat(head_1, ..., head_H) · W_o  # [B, T, D]
```

### Feed-Forward Network

```
FFN(x) = GELU(x · W_1 + b_1) · W_2 + b_2
```

- **Input:** `[B, T, D]` (256)
- **Hidden:** `[B, T, d_ff]` (1024)
- **Output:** `[B, T, D]` (256)

The FFN expands by 4x, applies GELU, then projects back.

### Transformer Block (Pre-Normalization)

```python
# Pre-norm: normalize BEFORE attention/FFN, not after
x = x + attention(layer_norm_1(x))
x = x + ffn(layer_norm_2(x))
```

Pre-normalization is used because it provides more stable training gradients compared to post-normalization.

### Final LayerNorm + LM Head

```python
x = final_layer_norm(x)        # [B, T, D]
logits = lm_head(x)            # [B, T, V] - V=16000
```

The LM head is a linear projection from hidden dimension to vocabulary size.

Optionally, the LM head can share weights with the token embedding (`weight_tying`).

## Configuration Variants

| Variant | Layers | d_model | Heads | d_ff | ~Params | VRAM Est. |
|---------|--------|---------|-------|------|---------|-----------|
| Small   | 6      | 256     | 8     | 1024 | 13M (measured) | ~270 MB    |
| Medium  | 8      | 384     | 8     | 1536 | ~26M   | ~1.5 GB   |
| Large   | 12     | 512     | 8     | 2048 | ~85M   | ~3 GB     |

## Design Decisions

1. **Pre-normalization** - More stable gradients, standard in modern Transformers
2. **Learned positional embeddings** - Simpler than RoPE for initial implementation
3. **No bias** - Reduces parameter count, often slightly better performance
4. **GELU activation** - Smooth activation, standard in Transformers
5. **No weight tying (initially)** - Keeps the architecture simple; can be added later

## Limitations

- Maximum sequence length is fixed at configuration time
- No FlashAttention (GTX 1070 Pascal architecture does not support it efficiently)
- No gradient checkpointing (small model, not needed yet)
- No quantization (FP32 default; FP16 configurable via `dtype: float16`)

## Related documentation

- [index.md](index.md) - documentation entry point
- [training.md](training.md) - how the model is optimized and checkpointed
- [inference.md](inference.md) - how the trained model generates text
- [tokenizer.md](tokenizer.md) - how text becomes token IDs
- [dataset.md](dataset.md) - how token IDs become training batches
- [configs/small.yaml](../configs/small.yaml) - the canonical configuration
