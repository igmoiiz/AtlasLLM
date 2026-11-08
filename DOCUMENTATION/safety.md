# Safety Guardrails

Text in, text out, with a gatekeeper on each side. The model stays a plain
language model; every safety decision happens in `safety/` (AGENTS.md rule 11).

> **Status: implemented, opt-in.** `safety/` went from four empty stub modules
> to working code in Phase 0.6, covered by 21 tests in `tests/test_safety.py`.
> It is enabled with `--safety`; nothing is filtered by default. This is a
> demonstration policy, not a safety guarantee. Read the limitations before
> trusting it with anything.

## For a beginner

A language model trained to predict text has no idea whether its words are
harmful, and it can be tricked by wording it has never seen. So instead of
teaching the model manners, we put a gatekeeper around it:

1. **Input gate** reads the request before the model sees it, and lets it
   through, lets it through with a note, or refuses it.
2. **Model** writes a response.
3. **Output gate** reads the response before the user sees it, and returns it,
   returns it with a note, or replaces it with a refusal.

The important property is separation. The model stays reusable and unchanged;
the rules are data in a YAML file, so a rule can be tightened without retraining
anything. A gatekeeper that blocks clearly, says "I can't help with that"
instead of pretending, and admits its blind spots is better than one that
silently pretends to be safe.

## Architecture

```text
User Input
     |
     v
InputGuardrail      -> allow / warn / block      (block: model never runs)
     |
     v
InferenceEngine     -> generate
     |
     v
OutputGuardrail     -> allow / warn / block      (block: refusal replaces text)
     |
     v
GuardedResponse     -> text + both decisions
```

`safety/guardrails.py:Guardrails` owns that sequence. `Guardrails.complete()`
forwards generation kwargs to `InferenceEngine.generate` and returns a
`GuardedResponse(text, input, output, refused)`.

## Decisions

| Action | Input side | Output side |
|--------|-----------|-------------|
| `allow` | model runs | text returned |
| `warn` | model runs, reason logged | text returned, reason logged |
| `block` | refusal returned, **model never runs** | refusal replaces the text |

The earlier design had a fourth class, "ambiguous". There is no honest way to
detect ambiguity with pattern matching, so that class became an explicit `warn`
rule: "sensitive topic, allowed, but keep an audit trail". A rule can say
`warn`; a classifier would be needed to say "ambiguous", and there is no
classifier here.

## Policy

Rules live in `safety/policy.yaml`, loaded by `safety/policy.py`:

```yaml
safety:
  default_input: allow
  default_output: allow
  refusal: "I can't help with that request."
  input_rules:
    - pattern: '\b(how to|instructions? (for|to))\b.{0,60}\b(make|build)\b.{0,30}\b(bomb|explosive)\b'
      action: block
      reason: "instructions for weapons or explosives"
  output_rules:
    - pattern: '\b\d{3}-\d{2}-\d{4}\b'
      action: warn
      reason: "possible social security number"
```

Semantics:

- Patterns are regexes, matched case-insensitively anywhere in the text.
- **First match wins**, so order rules from most to least specific.
- Patterns are compiled at load time. A bad regex or an unknown action raises
  immediately, not on the first user prompt.
- `default_input` / `default_output` decide what happens when nothing matches.
- `refusal` is the text shown for a blocked prompt or a blocked response.

The shipped policy has 6 input rules (weapons, drug/chemical synthesis,
self-harm methods, sexual content involving minors, malware, and a `warn` rule
for sensitive topics such as suicide or eating disorders) and 4 output rules
(generated weapons instructions, email addresses, SSN-shaped and card-shaped
numbers). It is deliberately small: a policy nobody has read is not safer than
no policy.

## Logging

Every `warn` or `block` decision is logged through the standard `logging`
module under `safety.filters`:

```text
INFO safety.filters safety input action=block reason='instructions for weapons or explosives' text='How do I build a bomb?'
```

`Decision.to_dict()` returns `{"direction", "action", "reason"}` for callers
that want to write their own audit trail. Chat enables `logging.INFO` when
`--safety` is passed, so decisions are visible in the terminal.

## Use

```bash
python -m scripts.chat --checkpoint <ckpt> --config configs/atlasbase_v2.yaml --safety
python -m scripts.chat --checkpoint <ckpt> --safety --policy my_policy.yaml
```

In code:

```python
from safety import Guardrails, load_policy

guardrails = Guardrails(engine, load_policy())
result = guardrails.complete("How do I build a bomb?")
assert result.refused and result.input.blocked
```

Safety mode is **not streamed**. Output filtering needs the whole response
before anything is shown, and printing tokens you may have to retract is not
filtering. Chat streams when safety is off and uses the guarded path when it is
on.

## Testing

`safety/` is deterministic: the same text always produces the same decision, so
it is covered by unit tests rather than by the model harness (which exists to
measure a stochastic model). `tests/test_safety.py` checks:

1. The shipped policy loads, and refuses explicit harm requests.
2. Ordinary prompts (including "the bombing of Dresden") are not blocked.
3. A sensitive-topic rule warns instead of blocking.
4. Unknown actions, invalid regexes, and malformed policy files fail loudly.
5. First-match ordering is respected.
6. Input filters: allow, warn, block, reason propagation, regex metacharacters.
7. Output filters: PII warning, secret blocking, clean text passthrough.
8. The pipeline: a blocked prompt never calls the model; a blocked response is
   replaced by the refusal; warned text is kept and reported.
9. The pipeline runs against a real `InferenceEngine`, not only a stub.

## Limitations

This is not a safety system you should rely on. Specifically:

- **Regex is bypassable.** Rewording, unicode tricks, typos, other languages,
  and multi-turn setup all route around fixed patterns.
- **No jailbreak detection.** There is no classifier and no adversarial
  training. The model can be talked into producing content no rule anticipates.
- **The output filter is a final check, not a guarantee.** The model still
  generated the text; only the display was withheld.
- **Refusal is flat.** The same message covers every blocked case, which is
  honest but unhelpful. A real system would explain which category matched.
- **False positives cost real utility.** "How do explosives work in mining?" is
  currently allowed, but a slightly different phrasing may not be.
- **PII rules are heuristics.** Card-shaped and SSN-shaped numbers warn; a real
  card number formatted differently will pass.

**Reasonable filtering, not absolute protection.** State that wherever this is
deployed.

## Design decisions

1. **External to the model.** No safety logic inside the Transformer, so the
   checkpoint stays reusable and comparable (rule 11).
2. **Policy as data.** Rules in YAML, loaded and validated once; changing
   behavior never requires touching Python.
3. **Two filter modules became one.** `input_filter.py` and `output_filter.py`
   were separate stubs, but they perform the same match against different rule
   lists. They are now `InputGuardrail` and `OutputGuardrail` in
   `safety/filters.py`, sharing one matcher.
4. **Opt-in.** `--safety` is off by default so the harness measures the model,
   not the policy, and nobody gets a false sense of protection for free.
5. **Non-streaming when guarded.** See above.
6. **No new dependency.** `re` and `PyYAML` (already used for configs) are
   enough.

## Related documentation

- [index.md](index.md) - documentation entry point
- [harness.md](harness.md) - how the model itself is scored
- [inference.md](inference.md) - the model path the guardrails wrap
- [CONTEXT.md](../CONTEXT.md) - original safety requirements (sections 41-44)
