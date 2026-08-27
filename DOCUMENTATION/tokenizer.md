# Tokenizer

AtlasLLM uses a subword tokenizer for converting text to token IDs and back.

## Overview

```
Raw Text: "The cat sat on the mat"
           ↓ encode()
Token IDs: [45, 192, 731, 84, 19, 552, 4]
           ↓ decode()
Raw Text: "The cat sat on the mat"
```

## Library

**Hugging Face Tokenizers** — chosen over SentencePiece for:

- Faster training
- Better Python API
- Active maintenance
- Rust-backed performance

## Vocabulary

| Setting | Value |
|---------|-------|
| Size | 16,000 tokens |
| Algorithm | BPE (Byte-Pair Encoding) |
| Special tokens | `<pad>`, `<unk>`, `<bos>`, `<eos>` |

### Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `<pad>` | 0 | Padding — fill shorter sequences to uniform length |
| `<unk>` | 1 | Unknown — fallback for unseen subwords |
| `<bos>` | 2 | Beginning of sequence — marks sequence start |
| `<eos>` | 3 | End of sequence — marks sequence end |

## Interface

```python
from tokenizer.tokenizer import AtlasTokenizer

tokenizer = AtlasTokenizer.from_pretrained("tokenizer/model")

# Encode text to token IDs
ids = tokenizer.encode("The cat sat on the mat")
# → [45, 192, 731, 84, 19, 552, 4]

# Decode token IDs back to text
text = tokenizer.decode(ids)
# → "The cat sat on the mat"

# With special tokens
ids = tokenizer.encode("Hello", add_bos=True, add_eos=True)
# → [2, 145, 3]  (bos + tokens + eos)
```

## Training

The tokenizer is trained on the target corpus:

```bash
python -m tokenizer.train_tokenizer --config configs/small.yaml
```

Training parameters (from the `tokenizer:` section of the YAML config):
- Vocabulary size: 16,000
- Training algorithm: BPE
- Minimum frequency: 2
- Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>` (ids 0-3)

The CLI cross-checks that `model.vocab_size >= tokenizer.vocab_size` and saves
`stats.json` (chars, tokens, tokens/char, top tokens) next to the model.

### Lossless design

AtlasLLM's tokenizer uses **no pretokenizer and no normalizer**: BPE merges run
over the raw character stream, so `encode`/`decode` is exactly lossless for
in-vocabulary text (including whitespace runs, tabs, newlines, and unicode).

A seed document containing every single-byte character (0x00-0xFF) is prepended
during training so the full byte-letter alphabet always exists. Characters
absent from the corpus and outside that alphabet map to `<unk>` (id 1) rather
than being silently dropped.

> Note: because the character alphabet is always kept, a corpus like WikiText
> (≈1,200 distinct characters) needs `vocab_size ≥ ~1,280`; the debug config
> uses 1,280, not 256.

## Saving and Loading

```python
# Save
tokenizer.save("tokenizer/model/")

# Load
tokenizer = AtlasTokenizer.from_pretrained("tokenizer/model/")
```

The tokenizer model is stored as:
```
tokenizer/model/
├── tokenizer.json
├── vocab.json
└── merges.txt
```

## Testing

Tokenizer tests verify:

1. **Encode → decode roundtrip** — `decode(encode(text)) == text` for in-vocabulary text
2. **Special token handling** — `<bos>`, `<eos>`, `<pad>`, `<unk>` have stable ids 0-3
3. **Vocabulary size** — matches `get_vocab_size()`; all ids in range
4. **Empty input** — handles gracefully
5. **Unicode handling** — processes accented and multi-byte characters correctly
6. **Unknown tokens** — unseen characters map to `<unk>`, never silently dropped

## Design Decisions

1. **16k vocabulary** — Large enough for reasonable coverage, small enough for manageable embedding tables
2. **BPE** — Well-understood, widely used, good balance of compression and flexibility
3. **No pretokenizer/normalizer** — guarantees lossless encode↔decode roundtrip and keeps text handling transparent (the data pipeline handles cleaning)
4. **Seeded single-byte alphabet** — all 256 byte letters always exist as base tokens, so ASCII/Latin-1 text never produces unknowns
5. **`<unk>` for genuinely unseen characters** — unknown multi-byte characters map to `<unk>` visibly, never silently dropped
6. **Stable interface** — `encode()` and `decode()` signatures remain stable even if internals change
