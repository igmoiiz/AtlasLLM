# Test Harness

How AtlasLLM is judged. A fixed set of prompts, a fixed scoring rule, one JSON
file per checkpoint, and a gate that fails a new checkpoint when it got worse
than the last one on the things we actually measure.

> **Status: implemented and measured.** `harness/` went from five empty stub
> modules to working code in Phase 0.5. The numbers below come from
> `reports/data/AtlasBase-v2/harness.json`, `stress.json` and `stress_noban.json`,
> produced by the commands shown further down.

## For a beginner

You cannot tell whether a language model improved by reading its output once.
It is probabilistic: the same checkpoint can write clean prose nine times and
loop one phrase on the tenth seed. So you ask the same questions every time,
score the answers the same way, and compare the numbers.

Concretely, AtlasBase-v2 asked "The capital of France is" produced:

```
 taken by the British and British colonies in England (1516-173). The British
 were very much in Europe (176-134). Many of the major French colonies...
```

It never says Paris. That is a measurement, not an opinion: the probe has an
expected answer, the answer is absent, the probe fails. When v3 runs the same
probe and says Paris, that is visible progress. When v3 fails a probe v2
passed, that is a regression, and the gate stops it.

## What runs today

Three suites, 36 probes total, defined in `harness/scenarios.py`.

| Suite | Probes | What it is for |
|-------|--------|----------------|
| `scorecard` | 6 | The v1/v2 protocol: one prompt per failure mode (autocomplete, story, fact, echo) |
| `adversarial` | 10 | Long-form prompts that historically trigger repetition loops |
| `factual` | 20 | Cloze questions with a checkable answer (capitals, chemical symbols, arithmetic) |
| `all` | 36 | Everything |

Decoding is chosen per probe, on purpose:

| | Long-form / adversarial | Factual cloze |
|---|---|---|
| tokens | 200 | 24 |
| temperature | 0.8 | 0.0 (greedy) |
| top-k / top-p | 50 / 0.95 | ignored by greedy |
| no-repeat 4-gram ban | on (default 4) | bypassed by greedy |

Long-form probes want variety, so they sample. Knowledge probes want
determinism, so they are greedy: the same checkpoint gives the same answer
twice. Greedy decoding ignores the n-gram ban by design (see
[inference.md](inference.md)), which is why the factual probes loop when the
model is unsure and why their repetition number is reported separately.

## Scoring

Every completion is reduced to four numbers, computed from the text alone:

| Field | Meaning |
|-------|---------|
| `repetition` | Fraction of tokens inside an already-seen 4-gram. 0 = every window is new, 1 = the model is stuck |
| `prompt_echo` | Fraction of output words that also appear in the prompt |
| `on_topic` | Fraction of the prompt's words that reappear in the output |
| `passed` | `true`/`false` when the probe has an expected answer, `null` when it does not |

`passed` requires the answer as a **whole word**. Substring matching was tried
first and was too generous: the freezing-point probe "passed" because `0`
matched inside `30 minutes`. `repetition` reuses
`evaluation.repetition.repetition_score`, the same function the old scorecard
and adversarial script used, so pre-harness and post-harness numbers stay
comparable.

The transcript is the authoritative record. These numbers exist to compare
checkpoints, not to grade prose.

## Usage

Run a suite and write the report:

```bash
python -m harness.runner --checkpoint checkpoints/atlasbase_v2/run_20260915-201403/best.pt \
    --suite all --perplexity --out reports/data/AtlasBase-v2/harness.json
```

Compare a new checkpoint against a baseline (exit status 1 on regression):

```bash
python -m harness.runner --checkpoint <v3.pt> --suite all --perplexity \
    --baseline reports/data/AtlasBase-v2/harness.json --out reports/data/AtlasBase-v3/harness.json
```

Loop stress test: the same prompts across 10 seeds, worst case reported:

```bash
python -m harness.runner --checkpoint <ckpt> --suite adversarial --stress 10 --quiet \
    --out reports/data/<version>/stress.json
```

Useful flags: `--max-tokens` overrides every probe's budget for a fast smoke
run, `--quiet` drops the transcripts, `--no-perplexity` is the default,
`--device cpu` avoids the GPU.

## Report format

One JSON file per run. It is the artifact the docs cite, so it carries the
decoding settings next to the numbers:

```json
{
  "checkpoint": "...best.pt",
  "suite": "all",
  "decoding": {"top_k": 50, "top_p": 0.95, "seed": 42, "no_repeat_ngram_size": 4},
  "summary": {
    "n_probes": 36, "mean_repetition": 0.140, "mean_prompt_echo": 0.123,
    "checkable": 23, "passed": 3, "pass_rate": 0.130,
    "by_category": {"factual": {"total": 20, "checkable": 20, "passed": 3, "pass_rate": 0.15, "mean_repetition": 0.247}}
  },
  "perplexity": {"val_loss": 4.828, "val_perplexity": 125.0, "test_loss": 4.840, "test_perplexity": 126.5},
  "rows": [{"id": "fact_france", "category": "factual", "prompt": "...", "text": "...", "repetition": 0.0, "passed": false}]
}
```

`--stress` adds a `stress` block with per-prompt mean/worst repetition and the
seed that produced it. `--baseline` adds a `comparison` block with the gate
result.

## The regression gate

`harness/reports.py:compare` fails when either axis got worse:

| Axis | Fails when |
|------|-----------|
| Val perplexity | current > baseline x 1.02 (+2%) |
| Mean repetition | current > baseline + 0.10 |

Both are deltas against a previous checkpoint, never absolute targets. A 48M
base model will not reach "good" numbers, so the only question a harness can
answer honestly is "is this worse than what we had". Per-probe pass flips are
printed for information (`lost passes: fact_ocean`) but do not fail the gate:
with one sample per probe they are too noisy to gate on.

The repetition axis averages all probes, including the greedy factual ones
that loop by construction. Same ruler on both sides keeps the comparison fair;
the per-category breakdown is where the diagnostic lives (v2: adversarial
0.006, factual 0.247).

## Measured baseline: AtlasBase-v2

From `reports/data/AtlasBase-v2/harness.json` (36 probes, seed 42):

| Metric | Value |
|--------|-------|
| Val perplexity | 125.0 |
| Test perplexity | 126.5 |
| Answered | 3/23 (13%) |
| Mean repetition | 0.140 |
| Mean prompt echo | 0.123 |
| Mean on-topic | 0.404 |

Per category:

| Category | n | Pass | Mean repetition |
|----------|---|------|-----------------|
| adversarial | 10 | - | 0.006 |
| story | 2 | - | 0.003 |
| autocomplete | 2 | 0% | 0.000 |
| fact | 1 | 0% | 0.022 |
| echo | 1 | 0% | 0.000 |
| factual | 20 | 15% | 0.247 |

The three "passes" are `two` inside a degenerate `two equals...` loop, `cold`,
and `pacific`. None of them is a fact the model knew; they are keywords that
happened to appear. The honest summary is: v2 knows approximately none of the
20 cloze answers, cannot repeat a three-word phrase on request, and its greedy
short answers loop hard (`atre, its currency, its currency, its currency...`).

## Loop stress test: what the n-gram ban bought

Same checkpoint, same 10 adversarial prompts, 10 seeds each, 200 tokens
(`stress.json` = ban 4, `stress_noban.json` = ban 0):

| Configuration | Prompts over 0.1 repetition | Worst repetition | Mean |
|---------------|----------------------------|------------------|------|
| no-repeat 4-gram ban on | 1/10 | 0.118 | 0.007 |
| ban off | 8/10 | 0.896 | 0.073 |

Without the ban the worst prompt repeats 90% of its words (`adv_mars`, seed 5).
With it, one prompt still crosses 0.1 in its worst of 10 seeds (`adv_coin`,
0.118). That is expected: the ban is on token n-grams, the metric counts word
n-grams, and different tokenizations can produce the same words. The ban
reduces loops by roughly an order of magnitude; it does not remove them.

## Design decisions

1. **Probes are Python data, not YAML.** The earlier plan called for YAML
   scenario files. Typed dataclasses in `scenarios.py` are importable by tests,
   need no parser, and cannot drift from the scoring code. Revisit only if
   someone who does not write Python needs to add probes.
2. **One entry point.** `python -m harness.runner` replaced
   `scripts/scorecard.py` and `scripts/adversarial_eval.py`, which were two
   copies of the same loop with different prompt lists. The old numbers in
   `reports/data/AtlasBase-v1/` are still reproducible from git history.
3. **Reuse before rewriting.** Repetition scoring, perplexity, and the dataset
   loader come from `evaluation/` and `data_pipeline/`; the harness adds
   suites, scoring glue, and reporting, nothing else.
4. **Relative gate.** Absolute thresholds are a lie at this model size. See
   above.
5. **No hidden work.** `--perplexity` and `--stress` are explicit flags, so a
   plain run stays fast, and the JSON says whether they ran.

## Limitations

- Keyword matching cannot judge prose quality. `passed` is a lower bound on
  knowledge, and a generous one.
- One sample per probe in the main run. The stress test exists because that is
  not enough for loop detection.
- Latency is machine-specific; it is recorded for the same-machine comparison,
  not as a benchmark.
- No safety probes. The guardrails are deterministic, so they are unit-tested
  in `tests/test_safety.py` rather than scored as a stochastic suite (see
  [safety.md](safety.md)).
- No grammar, reasoning, or instruction-following scoring. Those tests would
  fail on a 48M base model regardless, and a suite that always returns zero
  measures nothing.

## Related documentation

- [index.md](index.md) - documentation entry point
- [experiments.md](experiments.md) - comparing checkpoints across versions
- [inference.md](inference.md) - the engine the runner drives
- [training.md](training.md) - how a checkpoint is produced
- [safety.md](safety.md) - guardrails that will feed a safety suite
