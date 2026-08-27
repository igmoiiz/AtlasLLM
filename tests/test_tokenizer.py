"""Tests for the AtlasLLM tokenizer.

The fixture trains a small BPE on an in-memory corpus so the suite has no
network or model dependencies.
"""

import pytest

from tokenizer.tokenizer import AtlasTokenizer
from tokenizer.train_tokenizer import train_from_texts
from tokenizer.vocabulary import BOS, EOS, PAD, UNK

# Chars used here (incl. latin-1 accents, CJK-adjacent multibyte, punctuation)
# are present in the training corpus so they become real tokens.
TRAIN_TEXTS = [
    "hello world",
    "Alpha = 0.5, beta?",
    "café ☕ and ümlauts éüöä",
    "quick brown fox",
    "numbers 1234567890",
    "punctuation ,.?!;:()[]{}",
    "quoting \"double\" and 'single' in text",
    "indented  and  double  spaces",
]

ROUNDTRIP_TEXTS = [
    "hello world",
    "café ☕ and ümlauts éüöä",
    "Alpha = 0.5, beta?",
    "quick brown fox",
    "numbers 1234567890",
    "punctuation ,.?!;:()[]{}",
    "quoting \"double\" and 'single' in text",
    "indented  and  double  spaces",
    "x",  # single chars are base tokens even if rare
    "~!@# $%^&*()",
]


@pytest.fixture(scope="module")
def tokenizer() -> AtlasTokenizer:
    return train_from_texts(TRAIN_TEXTS, vocab_size=800, min_frequency=2)


def test_vocab_size_and_special_ids(tokenizer: AtlasTokenizer) -> None:
    assert tokenizer.vocab_size == len(tokenizer.vocab)
    assert tokenizer.vocab_size > 256  # single-byte alphabet + merges
    assert tokenizer.pad_id == 0
    assert tokenizer.unk_id == 1
    assert tokenizer.bos_id == 2
    assert tokenizer.eos_id == 3
    assert PAD in tokenizer.vocab and UNK in tokenizer.vocab
    assert BOS in tokenizer.vocab and EOS in tokenizer.vocab


@pytest.mark.parametrize("text", ROUNDTRIP_TEXTS)
def test_roundtrip(tokenizer: AtlasTokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_ids_in_range(tokenizer: AtlasTokenizer) -> None:
    for text in ROUNDTRIP_TEXTS:
        for token_id in tokenizer.encode(text):
            assert 0 <= token_id < tokenizer.vocab_size


def test_bos_eos(tokenizer: AtlasTokenizer) -> None:
    ids = tokenizer.encode("hello world", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    # bos/eos are not in the "plain" encoding
    plain = tokenizer.encode("hello world")
    assert tokenizer.bos_id not in plain and tokenizer.eos_id not in plain


def test_empty_input(tokenizer: AtlasTokenizer) -> None:
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_unknown_token_uses_unk(tokenizer: AtlasTokenizer) -> None:
    # A character absent from the corpus and outside the single-byte alphabet
    # must map to <unk>, never be silently dropped.
    unknown_char = "你"
    ids = tokenizer.encode(unknown_char)
    assert ids == [tokenizer.unk_id]


def test_repeated_chars_are_lossless(tokenizer: AtlasTokenizer) -> None:
    text = "a" * 200
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_save_load_roundtrip(tokenizer: AtlasTokenizer, tmp_path) -> None:
    tokenizer.save(tmp_path)
    restored = AtlasTokenizer.from_pretrained(tmp_path)
    assert restored.vocab_size == tokenizer.vocab_size
    assert (tmp_path / "tokenizer.json").exists()
    assert (tmp_path / "vocab.json").exists()
    assert (tmp_path / "merges.txt").exists()
    for text in ROUNDTRIP_TEXTS:
        assert restored.decode(restored.encode(text)) == text


def test_from_pretrained_missing_dir(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        AtlasTokenizer.from_pretrained(tmp_path / "does-not-exist")
