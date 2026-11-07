"""Combined guardrail pipeline: input filter -> model -> output filter.

    prompt
      -> InputGuardrail   blocked? refuse, the model is never called
      -> InferenceEngine  generate
      -> OutputGuardrail  blocked? replace the completion with the refusal
      -> GuardedResponse  text + both decisions (for display and audit)

The refusal text comes from the policy, so behavior is configurable per
deployment. ``complete`` is deliberately not a streaming API: output filtering
needs the whole response before anything is shown, and showing tokens you may
have to retract is not filtering. Chat runs the streamed path when safety is
off and this path when it is on.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from inference.engine import InferenceEngine
from safety.filters import Decision, InputGuardrail, OutputGuardrail
from safety.policy import Policy, load_policy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardedResponse:
    """Final text plus the decisions that produced it."""

    text: str
    input: Decision
    output: Decision | None
    refused: bool


class Guardrails:
    """Wrap an inference engine in the policy's input and output filters."""

    def __init__(self, engine: InferenceEngine, policy: Policy | None = None):
        self.engine = engine
        self.policy = policy if policy is not None else load_policy()
        self.input = InputGuardrail(self.policy)
        self.output = OutputGuardrail(self.policy)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        config=None,
        tokenizer=None,
        device: str = "auto",
        policy_path: str | Path | None = None,
    ) -> "Guardrails":
        engine = InferenceEngine.from_checkpoint(
            checkpoint, config=config, tokenizer=tokenizer, device=device
        )
        return cls(engine, load_policy(policy_path) if policy_path else None)

    def complete(self, prompt: str, **generate_kwargs) -> GuardedResponse:
        """Generate through both filters. Extra kwargs go to ``engine.generate``."""
        decision = self.input.check(prompt)
        if decision.blocked:
            return GuardedResponse(self.policy.refusal, decision, None, refused=True)
        generated = self.engine.generate(prompt, **generate_kwargs)
        output = self.output.check(generated.text)
        if output.blocked:
            return GuardedResponse(self.policy.refusal, decision, output, refused=True)
        return GuardedResponse(generated.text, decision, output, refused=False)
