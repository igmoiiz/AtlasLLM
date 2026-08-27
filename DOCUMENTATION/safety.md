# Safety System (Design)

AtlasLLM guardrails - input and output filtering external to the base model.

> **Status: PLANNED.** The `safety/` package currently contains stub modules
> (`policy.py`, `input_filter.py`, `output_filter.py`, `guardrails.py`), all
> with no implementation. This page is the agreed design that implementation
> must follow. Nothing here should be described as working.

## For a beginner

Language models are tools, not judges. A small model trained only to predict text has no understanding of whether its words are harmful, and it can be tricked by wording it has never seen. The safety system is a separate wrapper around the model, like a gatekeeper at the door:

1. **Input gate** - reads the user's request before the model sees it, and lets it through, flags it, or refuses it.
2. **Model** - generates a response.
3. **Output gate** - checks the response before the user sees it, and returns it, warns, or blocks it.

The key idea is separation: the model itself stays a plain language model, and all safety decisions happen in dedicated layers around it. A gatekeeper that blocks clearly, rejects quietly, and is honest about its limits beats a system that silently pretends to be safe.

Anything below this line is the technical design. See [index.md](index.md) for the project overview.

## Architecture

```text
User Input
     ↓
Input Guardrail → classify request
     ↓
AtlasLLM Model → generate response
     ↓
Output Guardrail → check response
     ↓
Final Response
```

**Key principle:** Safety is external to the base model. The model is a language model. Safety is a separate system (AGENTS.md rule 11).

## Input Guardrail

Classifies incoming requests:

| Classification | Action |
|---------------|--------|
| allowed | Pass through to model |
| ambiguous | Pass through with logging |
| blocked | Refuse before reaching model |

### Classification Categories

- **Allowed:** Normal questions, creative writing, code, explanations
- **Ambiguous:** Potentially sensitive topics requiring context
- **Blocked:** Clearly harmful requests

### Planned Implementation

```python
class InputGuardrail:
    def __init__(self, policy_path):
        self.policy = load_policy(policy_path)

    def check(self, prompt: str) -> tuple[str, str]:
        """Returns (action, reason)."""
        # Rule-based matching
        # Keyword filtering
        # Pattern detection
        return ("allowed", "")
```

## Output Guardrail

Checks generated response before returning to user:

| Classification | Action |
|---------------|--------|
| safe | Return response |
| warning | Return response with logged warning |
| blocked | Replace with refusal message |

## Policy

Safety policies are defined in configuration, not hardcoded:

```yaml
safety:
  input_rules:
    - pattern: "harmful pattern"
      action: block
      reason: "matches blocked content"
  output_rules:
    - pattern: "sensitive output"
      action: warn
  default_input: allow
  default_output: allow
```

## Limitations

The guardrail system is **not** a safety guarantee:

- A small model can misunderstand context
- Pattern matching can be bypassed
- Novel harmful requests may not match existing rules
- The model may generate harmful content before output filtering catches it

**The system provides reasonable filtering, not absolute protection.**

## Logging

All safety decisions are logged:

```json
{
  "timestamp": "2026-01-01T00:00:00Z",
  "direction": "input",
  "prompt_preview": "The cap...",
  "action": "allowed",
  "reason": ""
}
```

## Testing

Safety tests will verify:

1. Allowed prompts pass through
2. Blocked prompts are caught
3. Ambiguous prompts are logged
4. Output filtering catches policy violations
5. Refusal messages are appropriate

These tests belong in `tests/test_safety.py` (currently empty).

## Design Decisions

1. **External to model** - The base model remains reusable for fine-tuning
2. **Policy-driven** - Rules are configurable, not hardcoded
3. **Explicit limitations** - No false safety claims
4. **Logged decisions** - Audit trail for testing and improvement

## Implementation checklist

- [ ] Implement `safety/policy.py` (policy loading and matching)
- [ ] Implement `safety/input_filter.py` (`InputGuardrail`)
- [ ] Implement `safety/output_filter.py` (`OutputGuardrail`)
- [ ] Implement `safety/guardrails.py` (combined pipeline = input + output)
- [ ] Wire guardrails into the chat CLI (`scripts/chat.py`)
- [ ] Write the tests listed under Testing above

## Related documentation

- [index.md](index.md) - documentation entry point
- [harness.md](harness.md) - where safety behavior gets tested at scale
- [inference.md](inference.md) - the model path the guardrails wrap
- [CONTEXT.md](../CONTEXT.md) - original safety requirements (sections 41-44)