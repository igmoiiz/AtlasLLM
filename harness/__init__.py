"""Checkpoint harness: fixed probe suites, scoring, reports, regression gate.

    scenarios  what we ask, and with which decoding settings
    runner     drives a checkpoint over a suite (``python -m harness.runner``)
    scoring    turns one completion into comparable numbers
    adversarial  loop stress test across seeds
    reports    JSON payload, transcript, baseline comparison
"""
