"""Container for the per-layer key/value cache used during generation.

The cache owns one full ``[B, H, T, head_dim]`` K and V tensor per layer and
handles lifecycle only — tensor math lives in ``model.attention``. Layers
that have not been prefilled yet report ``None`` so the model treats them as
fresh.
"""

import torch


class KVCache:
    def __init__(self, n_layers: int):
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_layers = n_layers
        self.keys: list[torch.Tensor | None] = [None] * n_layers
        self.values: list[torch.Tensor | None] = [None] * n_layers

    def length(self) -> int:
        """Number of cached positions (tokens) per layer, 0 when empty."""
        first = self.keys[0]
        return first.size(2) if first is not None else 0

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store the full K/V tensors for one layer (replaces previous)."""
        self._check_layer(layer)
        self.keys[layer] = key
        self.values[layer] = value

    def update(self, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Store the (k, v) pair returned by each layer of a model forward."""
        if len(pairs) != self.n_layers:
            raise ValueError(f"expected {self.n_layers} pairs, got {len(pairs)}")
        for layer, (key, value) in enumerate(pairs):
            self.keys[layer] = key
            self.values[layer] = value

    def pairs(self) -> list[tuple[torch.Tensor | None, torch.Tensor | None]]:
        """One (k, v) pair per layer, ``(None, None)`` for never-seen layers."""
        return list(zip(self.keys, self.values))

    def reset(self) -> None:
        self.keys = [None] * self.n_layers
        self.values = [None] * self.n_layers

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} out of range [0, {self.n_layers})")
