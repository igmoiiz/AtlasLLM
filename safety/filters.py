"""Input and output filters: one rule match, applied on two sides of the model.

``InputGuardrail`` checks the user's request before the model sees it.
``OutputGuardrail`` checks the model's response before the user sees it. The
work is identical (match the policy's rules against text, return a decision);
only the rule list and the default differ, so both live here instead of in two
near-identical modules.

A decision is ``allow`` (``warn``) or ``block``:

    allow   nothing matched, or a rule said allow
    warn    a rule matched, the text is still used, the reason is logged
    block   a rule matched, the text is refused (output: replaced by refusal)
"""

import logging
from dataclasses import dataclass

from safety.policy import Policy, match

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Decision:
    """What the filter decided, and why."""

    direction: str  # "input" | "output"
    action: str  # "allow" | "warn" | "block"
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return self.action == "block"

    def to_dict(self) -> dict:
        return {"direction": self.direction, "action": self.action, "reason": self.reason}


class _Filter:
    """Shared matching logic; subclasses pick the rule list and the default."""

    direction = ""

    def __init__(self, policy: Policy):
        self.policy = policy

    def _rules(self):
        raise NotImplementedError

    def _default(self) -> str:
        raise NotImplementedError

    def check(self, text: str) -> Decision:
        rule = match(self._rules(), text)
        decision = Decision(
            self.direction,
            rule.action if rule else self._default(),
            rule.reason if rule else "",
        )
        if decision.action != "allow":
            logger.info(
                "safety %s action=%s reason=%r text=%r",
                decision.direction,
                decision.action,
                decision.reason,
                text[:120],
            )
        return decision


class InputGuardrail(_Filter):
    """Checks the user's request before generation."""

    direction = "input"

    def _rules(self):
        return self.policy.input_rules

    def _default(self) -> str:
        return self.policy.default_input


class OutputGuardrail(_Filter):
    """Checks the model's completion before it is returned."""

    direction = "output"

    def _rules(self):
        return self.policy.output_rules

    def _default(self) -> str:
        return self.policy.default_output
