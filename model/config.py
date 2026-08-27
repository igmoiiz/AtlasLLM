"""Model configuration — single source of truth for architecture hyperparameters.

All model modules receive a :class:`ModelConfig`; no architecture code reads
YAML or CLI arguments directly (AGENTS.md Rule 8, 12).
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Transformer architecture hyperparameters.

    Shapes: [B,T] -> [B,T,D] -> logits [B,T,V]
    ```

    Attributes:
        vocab_size: Vocabulary size V (LM head / embedding rows).
        context_length: Maximum sequence length T.
        d_model: Model/hidden dimension D.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads H (must divide d_model).
        d_ff: Feed-forward inner dimension.
        dropout: Dropout rate applied inside attention and FFN.
        bias: Whether linear layers learn a bias term.
    """

    vocab_size: int
    context_length: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float = 0.0
    bias: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must divide d_model ({self.d_model}) "
                "so each head gets an integer head_dim"
            )
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Build from a YAML ``model:`` section, rejecting unknown fields."""
        known = set(asdict(cls(1, 1, 1, 1, 1, 1)))
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown model config field(s): {sorted(unknown)}")
        return cls(**{k: v for k, v in data.items() if k in known})
