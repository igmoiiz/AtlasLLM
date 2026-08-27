"""Inference: sampling, KV cache, and the generation engine.

``inference.generate`` (the ``python -m inference.generate`` entry point) is
deliberately NOT imported here so it can run as ``__main__`` without the
package being evaluated first (AGENTS.md Rule on no ``python -m pkg.sub``
when ``pkg/__init__`` imports that submodule).
"""

from inference.engine import Generation, InferenceEngine
from inference.kv_cache import KVCache
from inference.sampling import greedy, sample_next_token

__all__ = [
    "Generation",
    "InferenceEngine",
    "KVCache",
    "greedy",
    "sample_next_token",
]
