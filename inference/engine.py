"""Inference engine: prompt -> tokens -> logits -> sampling -> text.

The engine sequences the decoding loop

    prompt -> tokenizer -> model -> sample -> append -> EOS/max_tokens

over an :class:`AtlasLLM`. It optionally maintains a :class:`KVCache` so each
step sees only the newest token instead of re-reading the whole prompt; both
paths produce identical output (verified by test_generation). Generation is
model-pure: no training, checkpoint, or safety code lives here (AGENTS.md
Rules 10-11).
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from inference.kv_cache import KVCache
from inference.sampling import sample_next_token
from model.atlas_llm import AtlasLLM
from model.config import ModelConfig
from tokenizer.tokenizer import AtlasTokenizer


def _resolve_device(device: str | torch.device) -> torch.device:
    return torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


@dataclass(frozen=True)
class Generation:
    """Result of a single ``generate`` call."""

    text: str
    token_ids: list[int]
    finished_reason: str  # "max_len" | "eos" | "stop_string"


class InferenceEngine:
    def __init__(self, model: AtlasLLM, tokenizer: AtlasTokenizer, device: str | torch.device = "auto"):
        if tokenizer.vocab_size > model.config.vocab_size:
            raise ValueError(
                f"tokenizer vocab ({tokenizer.vocab_size}) exceeds model vocab "
                f"({model.config.vocab_size}); decode would produce out-of-range ids"
            )
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = _resolve_device(device)
        self.model.to(self.device)

    # ------------------------------------------------------------------ api

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        eos: bool = True,
        stop_sequences: tuple[str, ...] = (),
        use_cache: bool = True,
    ) -> Generation:
        """Complete ``prompt`` into a :class:`Generation`."""
        input_ids = self._prepare(prompt, max_new_tokens)
        ids = input_ids[0].tolist()
        for token in self._iter_tokens(input_ids, max_new_tokens, temperature, top_k, top_p, seed, eos, stop_sequences, use_cache):
            ids.append(token)
        return Generation(text=self.tokenizer.decode(ids), token_ids=ids, finished_reason=self._finish_reason(ids, eos, stop_sequences))

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        eos: bool = True,
        stop_sequences: tuple[str, ...] = (),
        use_cache: bool = True,
    ):
        """Same decoding loop, yielding one decoded text chunk per token."""
        input_ids = self._prepare(prompt, max_new_tokens)
        cumulative = input_ids[0].tolist()
        emitted = ""
        for token in self._iter_tokens(input_ids, max_new_tokens, temperature, top_k, top_p, seed, eos, stop_sequences, use_cache):
            cumulative.append(token)
            full = self.tokenizer.decode(cumulative)
            chunk = full[len(emitted):] if full.startswith(emitted) else full
            emitted = full
            yield chunk

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config: ModelConfig | dict | str | Path | None = None,
        tokenizer: AtlasTokenizer | None = None,
        device: str | torch.device = "auto",
    ) -> "InferenceEngine":
        """Load weights from a training checkpoint.

        ``config`` may be a ModelConfig, a ``model:`` dict, or a YAML path.
        When omitted the checkpoint's own stashed ``config`` is used (training
        checkpoints written before Stage 7 store an empty config and require
        an explicit ``config``).
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state")
        if state is None:
            raise ValueError(f"{checkpoint_path}: checkpoint has no 'model_state' key")
        cfg = cls._resolve_config(config, checkpoint)
        model = AtlasLLM(cfg)
        model.load_state_dict(state)

        tokenizer = cls._resolve_tokenizer(config, checkpoint, tokenizer)
        return cls(model, tokenizer, device)

    # --------------------------------------------------------------- internals

    @staticmethod
    def _resolve_config(config: ModelConfig | dict | str | Path | None, checkpoint: dict) -> ModelConfig:
        if isinstance(config, ModelConfig):
            return config
        if isinstance(config, (str, Path)):
            raw = yaml.safe_load(Path(config).read_text())
        elif isinstance(config, dict):
            raw = config
        elif config is None:
            raw = (checkpoint.get("config") or {}).get("model")
            if raw is None:
                raise ValueError("config is required: this checkpoint predates config stashing")
        else:
            raise ValueError(f"Cannot interpret config {config!r}")
        if isinstance(raw, dict) and "model" in raw:
            raw = raw["model"]
        return ModelConfig.from_dict(raw)

    @staticmethod
    def _resolve_tokenizer(config, checkpoint: dict, tokenizer: AtlasTokenizer | None) -> AtlasTokenizer:
        """Tokenizer from (a) explicit arg, (b) the config the engine was built
        from, (c) the checkpoint's stashed config — checkpoints written during
        Stage 6+ carry it, earlier ones need ``config`` to point at a run dir
        containing ``config.yaml``."""
        if tokenizer is not None:
            return tokenizer
        if isinstance(config, (str, Path)):
            source = yaml.safe_load(Path(config).read_text())
        elif isinstance(config, dict):
            source = config
        else:
            source = checkpoint.get("config") or {}
        tokenizer_path = (source.get("data", {}) or {}).get("tokenizer_path")
        if tokenizer_path is None:
            raise ValueError(
                "tokenizer is required: not passed and no data.tokenizer_path in the config or checkpoint"
            )
        return AtlasTokenizer.from_pretrained(tokenizer_path)

    def _prepare(self, prompt: str, max_new_tokens: int) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt)
        if not ids:
            raise ValueError("prompt must be non-empty")
        # Make room for the requested generation by keeping the newest tokens;
        # the oldest are dropped so the window never exceeds context_length.
        keep = self.model.config.context_length - max_new_tokens
        if keep < 1:
            keep = 1
        if len(ids) > keep:
            ids = ids[-keep:]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _iter_tokens(self, input_ids, max_new_tokens, temperature, top_k, top_p, seed, eos, stop_sequences, use_cache):
        model, ctx, tok = self.model, self.model.config.context_length, self.tokenizer
        prefix_len = input_ids.size(1)
        budget = min(max_new_tokens, ctx - prefix_len)
        if budget <= 0:
            return
        rng = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None

        # ``sequence`` grows into the full prompt + generated text (used for the
        # next-token window and for stop-sequence decoding); ``cache`` holds the
        # K/V of everything already processed.
        sequence = input_ids
        cache = KVCache(model.config.n_layers) if use_cache else None
        pending = None
        if use_cache:
            with torch.no_grad():
                logits, pairs = model(sequence, past_key_values=cache.pairs())
            cache.update(pairs)
            # The prefill logits already predict the token right after the
            # prompt, so it is sampled here instead of re-feeding the last
            # prompt token (which would duplicate it in the cache).
            pending = sample_next_token(logits[:, -1, :], temperature, top_k, top_p, rng)

        for _ in range(budget):
            with torch.no_grad():
                if pending is not None:
                    next_token = pending
                    pending = None
                elif use_cache:
                    logits, pairs = model(sequence[:, -1:], past_key_values=cache.pairs())
                    cache.update(pairs)
                    next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p, rng)
                else:
                    logits = model(sequence)
                    next_token = sample_next_token(logits[:, -1, :], temperature, top_k, top_p, rng)
            sequence = torch.cat((sequence, next_token.unsqueeze(0)), dim=-1)
            decoded = tok.decode(sequence[0].tolist())
            if eos and next_token.item() == tok.eos_id:
                yield next_token.item()
                return
            if stop_sequences and any(decoded.endswith(seq) for seq in stop_sequences):
                yield next_token.item()
                return
            yield next_token.item()

    def _finish_reason(self, ids: list[int], eos: bool, stop_sequences: tuple[str, ...]) -> str:
        if eos and ids[-1] == self.tokenizer.eos_id:
            return "eos"
        if stop_sequences and self.tokenizer.decode(ids).endswith(stop_sequences):
            return "stop_string"
        return "max_len"
