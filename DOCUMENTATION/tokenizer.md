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

Training parameters:
- Vocabulary size: 16,000
- Training algorithm: BPE
- Character coverage: 1.0
- Minimum frequency: 2

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

1. **Encode → decode roundtrip** — `decode(encode(text)) == text`
2. **Special token handling** — `<bos>`, `<eos>`, `<pad>`, `<unk>` work correctly
3. **Vocabulary size** — matches expected count
4. **Empty input** — handles gracefully
5. **Unicode handling** — processes multi-byte characters correctly

## Design Decisions

1. **16k vocabulary** — Large enough for reasonable coverage, small enough for manageable embedding tables
2. **BPE** — Well-understood, widely used, good balance of compression and flexibility
3. **No preprocessing** — The tokenizer handles tokenization; the data pipeline handles cleaning
4. **Stable interface** — `encode()` and `decode()` signatures remain stable even if internals change
