"""AtlasTokenizer — stable interface around a Hugging Face BPE tokenizer.

The Transformer never touches this class; tokenization stays strictly on the
text<->id boundary. Shapes: text is ``str`` in, ``list[int]`` out (encode) and
the reverse (decode).
"""

import json
from pathlib import Path

from tokenizers import Tokenizer as _HFTokenizer

from tokenizer.vocabulary import BOS, EOS, PAD, UNK


class AtlasTokenizer:
    """Thin wrapper exposing the project's stable tokenizer interface."""

    def __init__(self, tokenizer: _HFTokenizer) -> None:
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------ ids

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tokenizer.token_to_id(PAD)

    @property
    def unk_id(self) -> int:
        return self._tokenizer.token_to_id(UNK)

    @property
    def bos_id(self) -> int:
        return self._tokenizer.token_to_id(BOS)

    @property
    def eos_id(self) -> int:
        return self._tokenizer.token_to_id(EOS)

    @property
    def vocab(self) -> dict[str, int]:
        """Token -> id mapping for inspection and statistics."""
        return self._tokenizer.get_vocab()

    # ----------------------------------------------------------- interface

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Split ``text`` into token ids.

        Args:
            text: source string.
            add_bos: prepend the ``<bos>`` id.
            add_eos: append the ``<eos>`` id.

        Returns:
            List of integer token ids in [0, vocab_size).
        """
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Reconstruct text from token ids. Special ids are removed."""
        return self._tokenizer.decode(ids)

    # ------------------------------------------------------------- persist

    def save(self, directory: str | Path) -> None:
        """Write tokenizer.json (canonical) plus vocab.json and merges.txt.

        Files:
            directory/tokenizer.json — loadable by from_pretrained
            directory/vocab.json     — token -> id, for inspection
            directory/merges.txt     — BPE merges, one pair per line
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tokenizer_path = directory / "tokenizer.json"
        self._tokenizer.save(str(tokenizer_path))

        vocab_path = directory / "vocab.json"
        vocab_path.write_text(json.dumps(self.vocab, indent=2), encoding="utf-8")

        merges_path = directory / "merges.txt"
        with open(tokenizer_path, encoding="utf-8") as f:
            merges = json.load(f)["model"].get("merges", [])
        merges_path.write_text(
            "\n".join(" ".join(pair) for pair in merges),
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "AtlasTokenizer":
        """Load a tokenizer saved by :meth:`save` from ``directory``."""
        tokenizer_path = Path(directory) / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        return cls(_HFTokenizer.from_file(str(tokenizer_path)))
