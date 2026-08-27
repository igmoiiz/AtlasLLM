"""Token-stream dataset for causal language-model training.

Reads a ``uint16`` token file (see :mod:`data_pipeline.preprocessing`) as a
memory-mapped array and yields non-overlapping, contiguous input/target pairs
for the next-token prediction task:

    tokens:   t0  t1  t2  ...  tN
    window i: input  [t_{iT}      ... t_{(i+1)T-1}]
              target [t_{iT+1}    ... t_{(i+1)T}]

For a window of length ``T`` each sequence consumes ``T + 1`` tokens; the tail
remainder of the stream is not used in an epoch.
"""

from pathlib import Path

import numpy as np
import torch


class TextDataset(torch.utils.data.Dataset):
    """Slice a token stream into shifted ``(input_ids, targets)`` windows.

    Args:
        bin_path: Path to a ``uint16`` token file.
        context_length: Sequence length ``T`` returned per sample.

    Shapes:
        input_ids: [T]
        targets:   [T]
    """

    def __init__(self, bin_path: str | Path, context_length: int):
        bin_path = Path(bin_path)
        if not bin_path.is_file():
            raise FileNotFoundError(f"Token file not found: {bin_path}")
        self.bin_path = bin_path
        self.context_length = context_length
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        """Number of complete non-overlapping windows of length T+1."""
        return max(0, (len(self.data) - 1) // self.context_length)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.context_length
        chunk = np.asarray(self.data[start : start + self.context_length + 1], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(chunk[:-1]),
            "targets": torch.from_numpy(chunk[1:]),
        }
