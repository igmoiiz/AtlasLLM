"""Safety guardrails, external to the base model.

    policy       rules loaded from YAML (pattern, action, reason)
    filters      InputGuardrail and OutputGuardrail share one matcher
    guardrails   prompt -> input filter -> engine -> output filter -> response

The model stays a plain language model; every safety decision happens here
(AGENTS.md rule 11).
"""

from safety.filters import Decision, InputGuardrail, OutputGuardrail
from safety.guardrails import GuardedResponse, Guardrails
from safety.policy import DEFAULT_POLICY_PATH, Policy, Rule, load_policy, match

__all__ = [
    "Decision",
    "InputGuardrail",
    "OutputGuardrail",
    "GuardedResponse",
    "Guardrails",
    "DEFAULT_POLICY_PATH",
    "Policy",
    "Rule",
    "load_policy",
    "match",
]
