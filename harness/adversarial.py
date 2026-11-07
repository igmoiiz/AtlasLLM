"""Loop stress test: the same prompt across seeds, worst case wins.

A single sampled completion tells you almost nothing about looping. The failure
mode is stochastic: a checkpoint can produce clean prose nine times and one
0.4-repetition loop on the tenth seed, which is exactly what the v2 checkpoint
did on "In 1969, humans first". So the adversarial measurement is the *worst*
repetition score across seeds, not the mean of one run.

Measured on AtlasBase-v2 (10 seeds x 3 prompts, 200 tokens): 3/30 runs above
0.1 repetition without the n-gram ban (max 0.408), 0/30 with it (max 0.013).
"""

from evaluation.repetition import repetition_score
from harness import scoring

LOOP_THRESHOLD = 0.1


def stress_repetition(
    engine,
    probes,
    *,
    seeds=range(5),
    top_k: int = 50,
    top_p: float = 0.95,
    no_repeat_ngram_size: int = 4,
    threshold: float = LOOP_THRESHOLD,
) -> dict:
    """Generate every probe at every seed; report mean and worst repetition."""
    seeds = sorted(seeds)
    per_prompt = []
    for probe in probes:
        scores = []
        for seed in seeds:
            gen = engine.generate(
                probe.prompt,
                max_new_tokens=probe.max_new_tokens,
                temperature=probe.temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            scores.append(
                repetition_score(
                    scoring.completion_text(engine.tokenizer, gen.token_ids, gen.prompt_tokens)
                )
            )
        worst = max(scores)
        worst_seed = seeds[scores.index(worst)]
        per_prompt.append(
            {
                "id": probe.id,
                "prompt": probe.prompt,
                "mean_repetition": sum(scores) / len(scores),
                "worst_repetition": worst,
                "worst_seed": worst_seed,
                "looped": worst > threshold,
            }
        )
    n = len(per_prompt)
    return {
        "threshold": threshold,
        "seeds": seeds,
        "n_prompts": n,
        "loops": sum(int(row["looped"]) for row in per_prompt),
        "worst_repetition": max((row["worst_repetition"] for row in per_prompt), default=0.0),
        "mean_repetition": (sum(row["mean_repetition"] for row in per_prompt) / n) if n else 0.0,
        "per_prompt": per_prompt,
    }
