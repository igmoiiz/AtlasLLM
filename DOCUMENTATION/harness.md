# Test Harness (Design)

AtlasLLM automated testing system for model evaluation.

> **Status: PLANNED.** The `harness/` package currently contains stub modules
> (`runner.py`, `scenarios.py`, `scoring.py`, `adversarial.py`, `reports.py`),
> all with no implementation. This page is the agreed design. The things that
> ARE working today are the 84 unit tests under `tests/` (see
> [training.md](training.md) and the test suite itself).

## For a beginner

How do you know a language model got better? You cannot just read its output once. The answer is a test harness: a fixed set of prompts, each with a clear expectation, run automatically against the model. A question about capitals must mention Paris; a grammar test must end with a period; a blocked safety request must be refused.

The same tests run against every model version, so a new training run is judged by the same ruler as the old one. "It got better" then means "more prompts passed, and nothing that used to pass now fails."

Anything below this line is the technical design. See [index.md](index.md) for the project overview.

## Overview

```
Scenario (test case)
     ↓
Harness Runner
     ↓
Model (inference)
     ↓
Guardrails (optional)
     ↓
Output
     ↓
Scorer (evaluate quality)
     ↓
Report
```

## Purpose

The harness provides automated, reproducible evaluation of AtlasLLM across multiple dimensions. Every model version is tested against the same suite.

## Test Categories

### Capability Tests

Basic model capabilities:

- Simple factual questions
- Text completion
- Sentence continuation
- Basic reasoning

### Language Tests

- Grammar correctness
- Coherence across sentences
- Vocabulary usage
- Punctuation

### Instruction Tests

- Follow simple instructions
- Format compliance
- Length control
- Style matching

### Reasoning Tests

- Simple logic
- Cause and effect
- Comparison
- Sequencing

### Robustness Tests

- Typo tolerance
- Capitalization changes
- Extra whitespace
- Rephrased prompts

### Safety Tests

- Allowed prompts pass
- Blocked prompts are refused
- Ambiguous prompts handled appropriately

### Adversarial Tests

- Prompt injection attempts
- Token manipulation
- Boundary testing
- Evasion attempts

### Regression Tests

- Known failures from previous versions
- Previously passing tests that must continue to pass

## Scenario Format

```yaml
- id: "cap_001"
  category: "capability"
  prompt: "The capital of France is"
  expected_pattern: "Paris"
  max_tokens: 10
  temperature: 0.0

- id: "robust_001"
  category: "robustness"
  prompt: "The capital of fRaNcE is"
  expected_pattern: "Paris"
  max_tokens: 10
  temperature: 0.0
```

## Scoring

Each test case produces:

| Field | Description |
|-------|-------------|
| passed | Boolean - did the output match expectations |
| output | Model's full response |
| latency_ms | Time to generate |
| tokens_generated | How many tokens |
| score | 0.0 to 1.0 - partial credit for fuzzy matching |

## Reports

```json
{
  "model": "checkpoints/run_20260101-120000/last.pt",
  "total_tests": 100,
  "passed": 78,
  "failed": 22,
  "by_category": {
    "capability": {"passed": 18, "total": 20},
    "robustness": {"passed": 15, "total": 20},
    "safety": {"passed": 20, "total": 20}
  },
  "regressions": [],
  "timestamp": "2026-01-01T00:00:00Z"
}
```

## Regression Testing

Compare across model versions:

```
v0.3 baseline: 78/100 passed
v0.4 results:   82/100 passed
New failures:    0 (no regressions)
New passes:      4 (improvements)
```

Any architecture change that breaks previously passing tests must be visible.

## Planned CLI Usage

```bash
python -m harness.runner --checkpoint checkpoints/run_20260101-120000/last.pt --suite all
python -m harness.runner --checkpoint checkpoints/run_20260101-120000/last.pt --suite safety
python -m harness.runner --checkpoint checkpoints/run_20260101-120000/last.pt --suite regression
```

## Design Decisions

1. **YAML-driven scenarios** - Easy to add new tests without code changes
2. **Fuzzy matching** - Not all responses need exact matches
3. **Per-version reports** - Every model version gets a snapshot
4. **Regression tracking** - Must not break what was working

## Implementation checklist

- [ ] Implement `harness/scenarios.py` (scenario loading from YAML)
- [ ] Implement `harness/scoring.py` (fuzzy matching and scoring)
- [ ] Implement `harness/runner.py` (runs scenarios against a checkpoint)
- [ ] Implement `harness/adversarial.py` (adversarial scenario generation)
- [ ] Implement `harness/reports.py` (reporting and regression comparison)
- [ ] Write the first scenario suite (start with capability + safety)

## Related documentation

- [index.md](index.md) - documentation entry point
- [safety.md](safety.md) - safety test scenarios feed into the harness
- [experiments.md](experiments.md) - comparing harness results across configs
- [inference.md](inference.md) - the engine the runner drives
- [CONTEXT.md](../CONTEXT.md) - original harness requirements (sections 45-51)