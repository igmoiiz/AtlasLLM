"""Tests for the AtlasLLM dataset pipeline.

``TextDataset`` is tested against a hand-written uint16 token stream so the
LM-shift contract is verified exactly. Preprocessing tests tokenize small
text files with a tiny in-memory BPE tokenizer.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data_pipeline.dataset import TextDataset
from data_pipeline.preprocessing import build_processed_data, tokenize_to_bin
from tokenizer.tokenizer import AtlasTokenizer
from tokenizer.train_tokenizer import train_from_texts

TRAIN_TEXTS = [
    "hello world, this is a test",
    "the quick brown fox jumps",
    "numbers 1234567890",
    "accented éüö text",
    "punctuation ,.?;:()",
    "a b c 1 2 3",
    "second line \\n third line",
    "x y z word",
]


@pytest.fixture(scope="module")
def tokenizer() -> AtlasTokenizer:
    return train_from_texts(TRAIN_TEXTS, vocab_size=800, min_frequency=2)


def make_bin(tmp_path: Path, n: int, name: str = "data.bin") -> tuple[Path, np.ndarray]:
    """Write tokens 0..n-1 to a uint16 .bin file alongside a memmapped view."""
    path = tmp_path / name
    arr = np.arange(n, dtype=np.uint16)
    arr.tofile(path)
    return path, arr


def test_len_is_number_of_complete_windows(tmp_path: Path) -> None:
    bin_path, arr = make_bin(tmp_path, n=1000)
    ds = TextDataset(bin_path, context_length=64)
    assert len(ds) == (1000 - 1) // 64 == 15


def test_getitem_shapes(tmp_path: Path) -> None:
    bin_path, _ = make_bin(tmp_path, n=1000)
    ds = TextDataset(bin_path, context_length=64)
    sample = ds[0]
    assert set(sample) == {"input_ids", "targets"}
    assert sample["input_ids"].shape == (64,)
    assert sample["targets"].shape == (64,)
    assert sample["input_ids"].dtype == torch.long
    assert sample["targets"].dtype == torch.long


def test_getitem_first_window_is_shifted_stream(tmp_path: Path) -> None:
    bin_path, arr = make_bin(tmp_path, n=1000)
    ds = TextDataset(bin_path, context_length=64)
    sample = ds[0]
    assert sample["input_ids"].tolist() == arr[0:64].tolist()
    assert sample["targets"].tolist() == arr[1:65].tolist()


def test_getitem_arbitrary_window(tmp_path: Path) -> None:
    bin_path, arr = make_bin(tmp_path, n=1000)
    ds = TextDataset(bin_path, context_length=64)
    start = 3 * 64
    sample = ds[3]
    assert sample["input_ids"].tolist() == arr[start : start + 64].tolist()
    assert sample["targets"].tolist() == arr[start + 1 : start + 65].tolist()


def test_windows_are_contiguous_and_non_overlapping(tmp_path: Path) -> None:
    bin_path, arr = make_bin(tmp_path, n=1000)
    ds = TextDataset(bin_path, context_length=64)
    for i in range(len(ds)):
        start = i * 64
        assert ds[i]["input_ids"].tolist() == arr[start : start + 64].tolist()
        assert ds[i]["targets"].tolist() == arr[start + 1 : start + 65].tolist()


def test_context_length_one(tmp_path: Path) -> None:
    bin_path, arr = make_bin(tmp_path, n=10, name="ctx1.bin")
    ds = TextDataset(bin_path, context_length=1)
    assert len(ds) == 9
    assert ds[0]["input_ids"].tolist() == arr[0:1].tolist()
    assert ds[0]["targets"].tolist() == arr[1:2].tolist()


def test_empty_dataset_when_too_short(tmp_path: Path) -> None:
    bin_path, _ = make_bin(tmp_path, n=3, name="short.bin")
    ds = TextDataset(bin_path, context_length=3)
    assert len(ds) == 0
    tiny_path, _ = make_bin(tmp_path, n=1, name="tiny.bin")
    assert len(TextDataset(tiny_path, context_length=3)) == 0


def test_missing_bin_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TextDataset(tmp_path / "missing.bin", context_length=64)


def test_tokenize_to_bin_matches_encode(tmp_path: Path, tokenizer: AtlasTokenizer) -> None:
    text = TRAIN_TEXTS[0] + "\n" + TRAIN_TEXTS[3]
    text_path = tmp_path / "sample.txt"
    text_path.write_text(text, encoding="utf-8")
    bin_path = tmp_path / "sample.bin"
    result = tokenize_to_bin(tokenizer, text_path, bin_path)
    assert result == {"tokens": len(tokenizer.encode(text))}
    assert np.fromfile(bin_path, dtype=np.uint16).tolist() == tokenizer.encode(text)


def _config(tmp_path: Path) -> dict:
    tmp_path.joinpath("raw").mkdir(exist_ok=True)
    for name, rows in [("train.txt", TRAIN_TEXTS[:4]), ("valid.txt", TRAIN_TEXTS[4:]),
                       ("test.txt", ["unseen test line here"])]:
        tmp_path.joinpath("raw", name).write_text("\n".join(rows), encoding="utf-8")
    return {
        "model": {"context_length": 16},
        "data": {
            "tokenizer_path": str(tmp_path / "tok"),
            "train_path": str(tmp_path / "processed" / "train.bin"),
            "val_path": str(tmp_path / "processed" / "val.bin"),
            "test_path": str(tmp_path / "processed" / "test.bin"),
            "train_text": str(tmp_path / "raw" / "train.txt"),
            "val_text": str(tmp_path / "raw" / "valid.txt"),
            "test_text": str(tmp_path / "raw" / "test.txt"),
        },
    }


def test_build_processed_data_writes_bins_and_meta(
    tmp_path: Path, tokenizer: AtlasTokenizer
) -> None:
    config = _config(tmp_path)
    meta_path = tmp_path / "processed" / "meta.json"
    build_processed_data(config, tokenizer, meta_path=meta_path)

    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["vocab_size"] == tokenizer.vocab_size
    assert meta["context_length"] == 16
    assert meta["dtype"] == "uint16"
    assert set(meta["splits"]) == {"train", "val", "test"}

    for split in ("train", "val", "test"):
        info = meta["splits"][split]
        bin_path = Path(config["data"][f"{split}_path"])
        assert bin_path.exists()
        n = np.fromfile(bin_path, dtype=np.uint16).size
        assert n == info["tokens"]
        assert info["sequences"] == (n - 1) // meta["context_length"]


def test_build_processed_data_roundtrip_through_dataset(
    tmp_path: Path, tokenizer: AtlasTokenizer
) -> None:
    config = _config(tmp_path)
    build_processed_data(config, tokenizer, meta_path=tmp_path / "processed" / "meta.json")
    bin_path = Path(config["data"]["train_path"])
    ds = TextDataset(bin_path, context_length=16)
    n = np.fromfile(bin_path, dtype=np.uint16).size
    assert len(ds) == (n - 1) // 16
    for i in (0, len(ds) - 1):
        assert ds[i]["input_ids"].shape == (16,)
        assert ds[i]["targets"].shape == (16,)
