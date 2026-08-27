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

    def forward(
        self, input_ids: torch.Tensor, past_key_values: list[tuple[torch.Tensor | None, torch.Tensor | None]] | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Map token ids [B, T] to vocabulary logits [B, T, V].

        With ``past_key_values`` (one (K, V) pair per layer, entries may be
        ``None`` to signal a fresh prefill) the model returns
        ``(logits, new_pairs)`` where new_pairs holds the *full* K/V tensors
        ``[B, H, P + T, head_dim]`` including the just-processed tokens,
        ready to be passed back on the next call. Without it the model keeps
        the pure ``logits`` behaviour used by training.
        """
        b, t = input_ids.shape
        if t > self.config.context_length:
            raise ValueError(
                f"Sequence length {t} exceeds context_length {self.config.context_length}"
            )
        if past_key_values is None:
            x = self.token_emb(input_ids) + self.pos_emb(t, input_ids.device)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            return self.lm_head(x)

        if len(past_key_values) != len(self.blocks):
            raise ValueError(
                f"Expected {len(self.blocks)} past key/value pairs, got {len(past_key_values)}"
            )
        first_key = past_key_values[0][0]
        offset = first_key.size(2) if first_key is not None else 0
        if offset + t > self.config.context_length:
            raise ValueError(
                f"Prefilled {offset} + new {t} exceeds context_length {self.config.context_length}"
            )
        x = self.token_emb(input_ids) + self.pos_emb(t, input_ids.device, offset=offset)
        new_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for block, (past_key, past_value) in zip(self.blocks, past_key_values):
            x, k, v = block.forward_with_cache(x, past_key, past_value)
            new_pairs.append((k, v))
        x = self.ln_f(x)
        return self.lm_head(x), new_pairs
