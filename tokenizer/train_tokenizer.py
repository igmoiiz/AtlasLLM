"""Train an AtlasLLM BPE tokenizer from a text corpus.

Usage:
    python -m tokenizer.train_tokenizer --config configs/small.yaml

The config file must contain a ``tokenizer`` section (vocab_size,
min_frequency, corpus_path) and a ``data.tokenizer_path`` for output.

Design notes:
- No pretokenizer/normalizer: BPE merges run over the raw character stream,
  so encode/decode is exactly lossless for in-vocabulary text.
- A seed line containing all 256 single-byte characters guarantees every byte
  letter exists as a base token, so any ASCII/Latin-1 text round-trips even if
  its characters never appeared in the corpus.
- Genuinely unseen multi-byte characters fall back to ``<unk>`` (visible
  failure at the id level, never a silent drop).
"""

import argparse
import json
from collections import Counter
from itertools import chain
from pathlib import Path
from time import perf_counter

import yaml
from tokenizers import Tokenizer as _HFTokenizer
from tokenizers.decoders import BPEDecoder
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from tokenizer.tokenizer import AtlasTokenizer
from tokenizer.vocabulary import SPECIAL_TOKENS, UNK

# Every single-byte character (0x00-0xFF). Seeded into training as a snowflake
# document so the byte-letter alphabet is complete even for rare characters.
_BYTE_ALPHABET = "".join(chr(i) for i in range(256))


def train_from_texts(
    texts,
    vocab_size: int,
    min_frequency: int,
) -> AtlasTokenizer:
    """Train a BPE tokenizer on an iterable of strings and wrap it."""
    hf_tokenizer = _HFTokenizer(BPE(unk_token=UNK))
    hf_tokenizer.decoder = BPEDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    hf_tokenizer.train_from_iterator(
        chain([_BYTE_ALPHABET], texts),
        trainer=trainer,
    )
    return AtlasTokenizer(hf_tokenizer)


def train_from_file(corpus_path: str | Path, vocab_size: int, min_frequency: int) -> AtlasTokenizer:
    """Train a tokenizer on a UTF-8 text file, one document per line."""
    corpus_path = Path(corpus_path)
    with open(corpus_path, encoding="utf-8") as f:
        return train_from_texts(f, vocab_size, min_frequency)


def compute_stats(tokenizer: AtlasTokenizer, corpus_path: Path) -> dict:
    """Encode the corpus once and summarize token distribution."""
    counter: Counter = Counter()
    total_chars = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            total_chars += len(line)
            counter.update(tokenizer.encode(line))
    total_tokens = sum(counter.values())
    id_to_token = {token_id: token for token, token_id in tokenizer.vocab.items()}
    top = [
        {"token": id_to_token[token_id], "id": token_id, "count": count}
        for token_id, count in counter.most_common(10)
    ]
    return {
        "corpus_chars": total_chars,
        "total_tokens": total_tokens,
        "tokens_per_char": round(total_tokens / max(total_chars, 1), 4),
        "vocab_size": tokenizer.vocab_size,
        "top_tokens": top,
    }


def train_tokenizer(
    corpus_path: Path,
    vocab_size: int,
    min_frequency: int,
) -> tuple[AtlasTokenizer, dict]:
    """Train a tokenizer, compute statistics, and return (tokenizer, stats)."""
    start = perf_counter()
    tokenizer = train_from_file(corpus_path, vocab_size, min_frequency)
    stats = compute_stats(tokenizer, corpus_path)
    stats["training_seconds"] = round(perf_counter() - start, 2)
    return tokenizer, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an AtlasLLM BPE tokenizer")
    parser.add_argument("--config", required=True, type=Path, help="Experiment YAML config")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tokenizer_config = config.get("tokenizer")
    if tokenizer_config is None:
        raise ValueError(f"Config {args.config} has no 'tokenizer' section")

    vocab_size = tokenizer_config["vocab_size"]
    min_frequency = tokenizer_config.get("min_frequency", 2)
    corpus_path = Path(tokenizer_config["corpus_path"])
    output_dir = Path(config["data"]["tokenizer_path"])

    model_vocab = config.get("model", {}).get("vocab_size")
    if model_vocab is not None and model_vocab < vocab_size:
        raise ValueError(
            f"model.vocab_size ({model_vocab}) < tokenizer.vocab_size ({vocab_size}); "
            "the LM embedding table must cover the full produced vocabulary"
        )

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    tokenizer, stats = train_tokenizer(corpus_path, vocab_size, min_frequency)

    if model_vocab is not None:
        if model_vocab < tokenizer.vocab_size:
            raise ValueError(
                f"Trained {tokenizer.vocab_size} tokens but model.vocab_size is only "
                f"{model_vocab}; raise model.vocab_size (e.g. to 1280 — a corpus "
                "like WikiText needs the full single-character alphabet plus merges)"
            )
        print(f"Vocabulary fits model: {tokenizer.vocab_size} <= {model_vocab}")

    tokenizer.save(output_dir)

    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Tokenized corpus:      {stats['corpus_chars']:,} chars")
    print(f"Total tokens:          {stats['total_tokens']:,}")
    print(f"Tokens per char:       {stats['tokens_per_char']}")
    print(f"Vocabulary size:       {stats['vocab_size']}")
    print(f"Special ids:           pad={tokenizer.pad_id} unk={tokenizer.unk_id} "
          f"bos={tokenizer.bos_id} eos={tokenizer.eos_id}")
    print("Top tokens:            " + ", ".join(
        f"{t['token']!r}:{t['count']}" for t in stats["top_tokens"]))
    print(f"Training time:         {stats['training_seconds']}s")
    print(f"Saved to:              {output_dir}")


if __name__ == "__main__":
    main()
