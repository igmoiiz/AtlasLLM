"""Stage-8 evaluation: perplexity, memorization gap, and generation probes.

Measurement layer for trained checkpoints. Depends on model/training/data
(one-way: nothing in model/training imports this package).

    eval = probe_generation + evaluate_memorization + evaluate_loss
"""

from evaluation.generation_eval import GenerationProbe, probe_generation
from evaluation.memorization import evaluate_memorization, memorize_gap
from evaluation.perplexity import evaluate_loss, perplexity

__all__ = [
    "GenerationProbe",
    "probe_generation",
    "evaluate_memorization",
    "memorize_gap",
    "evaluate_loss",
    "perplexity",
]
