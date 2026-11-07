"""Safety policy: rules loaded from YAML, matched by regex, never hardcoded.

The policy is data, not code, so a rule can be tightened without touching the
model or the filters (AGENTS.md rule 11: safety stays outside the model).

    policy.yaml
        safety:
          input_rules:  [{pattern, action, reason}, ...]
          output_rules: [{pattern, action, reason}, ...]
          default_input:  allow
          default_output: allow
          refusal: "..."

Actions are ``allow``, ``warn``, and ``block``. ``warn`` means "let it through
but record why" - the design doc called that class "ambiguous"; with
rule matching there is no way to detect genuine ambiguity, so the honest
mechanism is an explicit warn rule. First matching rule wins, so order the
policy from most to least specific.

What this is not: a safety guarantee. Regex matching is bypassable, and a 48M
base model can say harmful things in ways no pattern anticipates. See
``DOCUMENTATION/safety.md``.
"""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ACTIONS = ("allow", "warn", "block")
DEFAULT_POLICY_PATH = Path(__file__).with_name("policy.yaml")


@lru_cache(maxsize=None)
def _compile(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


@dataclass(frozen=True)
class Rule:
    """One policy rule: a regex plus what to do when it matches."""

    pattern: str
    action: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"rule action must be one of {ACTIONS}, got {self.action!r}")
        _compile(self.pattern)  # fail at load, not at the first user prompt

    def matches(self, text: str) -> bool:
        return _compile(self.pattern).search(text) is not None


@dataclass(frozen=True)
class Policy:
    input_rules: tuple[Rule, ...]
    output_rules: tuple[Rule, ...]
    default_input: str = "allow"
    default_output: str = "allow"
    refusal: str = "I can't help with that request."

    def __post_init__(self) -> None:
        for name, default in (("default_input", self.default_input), ("default_output", self.default_output)):
            if default not in ACTIONS:
                raise ValueError(f"{name} must be one of {ACTIONS}, got {default!r}")


def match(rules: tuple[Rule, ...], text: str) -> Rule | None:
    """First rule matching ``text``, or ``None``."""
    for rule in rules:
        if rule.matches(text):
            return rule
    return None


def _parse_rules(raw_rules, where: str) -> tuple[Rule, ...]:
    if raw_rules is None:
        return ()
    if not isinstance(raw_rules, list):
        raise ValueError(f"{where} must be a list of rules, got {type(raw_rules).__name__}")
    rules = []
    for index, raw in enumerate(raw_rules):
        if not isinstance(raw, dict) or "pattern" not in raw:
            raise ValueError(f"{where}[{index}] needs at least a 'pattern' key")
        rules.append(
            Rule(pattern=str(raw["pattern"]), action=str(raw.get("action", "block")), reason=str(raw.get("reason", "")))
        )
    return tuple(rules)


def load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> Policy:
    """Load and validate a policy file.

    Accepts either a top-level ``safety:`` section or the section's contents
    directly, so the same file can sit inside a training config or stand alone.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: policy must be a YAML mapping")
    section = data.get("safety", data)
    if not isinstance(section, dict):
        raise ValueError(f"{path}: 'safety' must be a YAML mapping")
    return Policy(
        input_rules=_parse_rules(section.get("input_rules"), f"{path}: input_rules"),
        output_rules=_parse_rules(section.get("output_rules"), f"{path}: output_rules"),
        default_input=str(section.get("default_input", "allow")),
        default_output=str(section.get("default_output", "allow")),
        refusal=str(section.get("refusal", "I can't help with that request.")),
    )
