# Test Harness

AtlasLLM automated testing system for model evaluation.

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
| passed | Boolean — did the output match expectations |
| output | Model's full response |
| latency_ms | Time to generate |
| tokens_generated | How many tokens |
| score | 0.0 to 1.0 — partial credit for fuzzy matching |

## Reports

```json
{
  "model": "checkpoints/step_10000.pt",
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

## CLI Usage

```bash
python -m harness.runner --checkpoint checkpoints/best.pt --suite all
python -m harness.runner --checkpoint checkpoints/best.pt --suite safety
python -m harness.runner --checkpoint checkpoints/best.pt --suite regression
```

## Design Decisions

1. **YAML-driven scenarios** — Easy to add new tests without code changes
2. **Fuzzy matching** — Not all responses need exact matches
3. **Per-version reports** — Every model version gets a snapshot
4. **Regression tracking** — Must not break what was working
