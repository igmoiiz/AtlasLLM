# Safety System

AtlasLLM guardrails — input and output filtering external to the base model.

## Architecture

```
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

**Key principle:** Safety is external to the base model. The model is a language model. Safety is a separate system.

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

### Implementation

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

Safety tests verify:

1. Allowed prompts pass through
2. Blocked prompts are caught
3. Ambiguous prompts are logged
4. Output filtering catches policy violations
5. Refusal messages are appropriate

## Design Decisions

1. **External to model** — The base model remains reusable for fine-tuning
2. **Policy-driven** — Rules are configurable, not hardcoded
3. **Explicit limitations** — No false safety claims
4. **Logged decisions** — Audit trail for testing and improvement
