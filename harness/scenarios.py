"""Fixed probe suites for the harness.

Every checkpoint is measured with the same prompts, the same decoding settings,
and the same scoring (:mod:`harness.scoring`), so two runs compare 1:1. Suites
are plain Python data on purpose: typed, importable by tests, no YAML parser,
no file-format migration.

    scorecard   6 short prompts, one per failure mode (v1/v2 protocol)
    adversarial 10 long-form prompts that trigger repetition loops
    factual     20 cloze probes with a checkable answer
    all         36 probes

A probe's ``expected`` is a tuple of lowercase answer keywords. It is empty for
probes with no checkable answer (open-ended story continuation); those are
scored on the heuristic axes only and never count as pass/fail. Keyword matching
is deliberately lenient: a 48M base model rambles, and the question we ask is
"does the right answer ever appear", not "is this a good answer".
"""

from dataclasses import dataclass

# Long-form probes: sampled, long enough to expose loops.
LONG_FORM_TOKENS = 200
LONG_FORM_TEMPERATURE = 0.8

# Short answer probes: greedy, so the same checkpoint gives the same answer.
SHORT_FORM_TOKENS = 24
SHORT_FORM_TEMPERATURE = 0.0


@dataclass(frozen=True)
class Probe:
    """One prompt plus the decoding settings and answer check to use."""

    id: str
    category: str
    prompt: str
    expected: tuple[str, ...] = ()
    max_new_tokens: int = LONG_FORM_TOKENS
    temperature: float = LONG_FORM_TEMPERATURE


def long_form(id: str, category: str, prompt: str, expected: tuple[str, ...] = ()) -> Probe:
    return Probe(id, category, prompt, expected, LONG_FORM_TOKENS, LONG_FORM_TEMPERATURE)


def short_form(id: str, prompt: str, expected: tuple[str, ...]) -> Probe:
    return Probe(id, "factual", prompt, expected, SHORT_FORM_TOKENS, SHORT_FORM_TEMPERATURE)


SCORECARD = (
    long_form("sc_france", "autocomplete", "The capital of France is", ("paris",)),
    long_form("sc_village", "story", "Once upon a time, in a quiet village,"),
    long_form("sc_orbit", "autocomplete", "The Earth orbits the Sun because"),
    long_form("sc_mars", "fact", "List ten things you know about the planet Mars.", ("mars",)),
    long_form("sc_catdog", "story", "The difference between a cat and a dog is"),
    long_form(
        "sc_echo",
        "echo",
        "Please repeat the last three words I wrote exactly: 'one two three'.",
        ("one two three",),
    ),
)

ADVERSARIAL = (
    long_form("adv_france", "adversarial", "The capital of France is"),
    long_form("adv_story", "adversarial", "Once upon a time, in a land far away,"),
    long_form("adv_ai", "adversarial", "Artificial intelligence is"),
    long_form("adv_1969", "adversarial", "In 1969, humans first"),
    long_form("adv_orbit", "adversarial", "The Earth orbits the Sun because"),
    long_form("adv_engine", "adversarial", "Explain, step by step, how an internal combustion engine works."),
    long_form("adv_mars", "adversarial", "List ten things you know about the planet Mars."),
    long_form("adv_king", "adversarial", "The king sat on his throne and"),
    long_form("adv_coin", "adversarial", "If you flip a coin, the probability of heads is"),
    long_form(
        "adv_echo",
        "adversarial",
        "Please repeat the last three words I wrote exactly: 'one two three'.",
    ),
)

FACTUAL = (
    short_form("fact_france", "The capital of France is", ("paris",)),
    short_form("fact_japan", "The capital of Japan is", ("tokyo",)),
    short_form("fact_italy", "The capital of Italy is", ("rome",)),
    short_form("fact_germany", "The capital of Germany is", ("berlin",)),
    short_form("fact_spain", "The capital of Spain is", ("madrid",)),
    short_form("fact_water", "The chemical formula for water is", ("h2o", "h₂o")),
    short_form("fact_gold", "The chemical symbol for gold is", ("au",)),
    short_form("fact_jupiter", "The largest planet in the solar system is", ("jupiter",)),
    short_form("fact_mercury", "The closest planet to the Sun is", ("mercury",)),
    short_form("fact_romeo", "The author of Romeo and Juliet was", ("shakespeare",)),
    short_form("fact_president", "The first president of the United States was", ("washington",)),
    short_form("fact_one_plus_one", "One plus one equals", ("two", "2")),
    short_form("fact_two_times_three", "Two times three equals", ("six", "6")),
    short_form("fact_hot", "The opposite of hot is", ("cold",)),
    short_form("fact_up", "The opposite of up is", ("down",)),
    short_form("fact_ocean", "The largest ocean on Earth is the", ("pacific",)),
    short_form("fact_nile", "The longest river in Africa is the", ("nile",)),
    short_form("fact_freezing", "The freezing point of water in Celsius is", ("zero", "0")),
    short_form("fact_week", "The number of days in a week is", ("seven", "7")),
    short_form("fact_yen", "The currency of Japan is the", ("yen",)),
)

SUITES: dict[str, tuple[Probe, ...]] = {
    "scorecard": SCORECARD,
    "adversarial": ADVERSARIAL,
    "factual": FACTUAL,
    "all": SCORECARD + ADVERSARIAL + FACTUAL,
}


def suite(name: str) -> tuple[Probe, ...]:
    """Return the named suite; ``ValueError`` lists the valid names."""
    try:
        return SUITES[name]
    except KeyError:
        raise ValueError(f"unknown suite {name!r}; choose from {sorted(SUITES)}") from None
