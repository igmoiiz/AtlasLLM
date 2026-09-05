"""Generation quality probes against a trained checkpoint.

Runs a fixed prompt suite through :class:`inference.engine.InferenceEngine`
and records each completion together with its finish reason and token count.
By keeping the prompts fixed across sessions the outputs are directly
comparable over training progress (the "same questions" protocol).
"""

from dataclasses import dataclass

from inference.engine import InferenceEngine


@dataclass(frozen=True)
class GenerationProbe:
    """Result of one fixed-prompt generation probe."""

    prompt: str
    text: str
    token_ids: list[int]
    finished_reason: str


def probe_generation(
    engine: InferenceEngine,
    prompts: list[str],
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    seed: int | None = None,
) -> list[GenerationProbe]:
    """Complete each prompt in ``prompts`` and return the probe records."""
    probes = []
    for prompt in prompts:
        gen = engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        probes.append(GenerationProbe(prompt=prompt, text=gen.text, token_ids=gen.token_ids, finished_reason=gen.finished_reason))
    return probes
