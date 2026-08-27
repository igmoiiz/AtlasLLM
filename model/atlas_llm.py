"""AtlasLLM — dense decoder-only Transformer language model.

    Tokens [B, T]
      -> TokenEmbedding + LearnedPositionalEmbedding    [B, T, D]
      -> TransformerBlock x n_layers                    [B, T, D]
      -> final LayerNorm                                [B, T, D]
      -> LM head (Linear)                               logits [B, T, V]

The model is pure: it receives token tensors and a config and returns tensor
logits. It contains no training loop, dataset, CLI, or loss logic (AGENTS.md
Rule 8); losses live in ``training.loss``.
"""

import torch
from torch import nn

from model.config import ModelConfig
from model.embeddings import TokenEmbedding
from model.normalization import LayerNorm
from model.positional_encoding import LearnedPositionalEmbedding
from model.transformer_block import TransformerBlock


class AtlasLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.pos_emb = LearnedPositionalEmbedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token ids [B, T] to vocabulary logits [B, T, V]."""
        b, t = input_ids.shape
        if t > self.config.context_length:
            raise ValueError(
                f"Sequence length {t} exceeds context_length {self.config.context_length}"
            )
        x = self.token_emb(input_ids) + self.pos_emb(t, input_ids.device)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
