"""Repetition metrics for generated text.

``repetition_score`` measures how much of a completion consists of tokens that
fall inside a previously-seen n-gram window.  This is the shared metric behind
the adversarial eval and the manual-test scorecard: both should measure the
same failure mode the same way (DRY).
"""


def repetition_score(text: str, n: int = 4) -> float:
    """Fraction of tokens that fall inside a previously-seen n-gram window.

    A value of 0.0 means every token is part of a novel n-gram; values near
    1.0 mean the model is looping a small set of phrases (the "Migrat" /
    "Dickens" failure mode).  Short texts are scored 0.0 because they have
    little room to loop.
    """
    tokens = text.split()
    if len(tokens) < n + 1:
        return 0.0
    seen: set[str] = set()
    repeated = 0
    for i in range(len(tokens) - n):
        window = " ".join(tokens[i : i + n])
        if window in seen:
            repeated += 1
        seen.add(window)
    return repeated / (len(tokens) - n)
