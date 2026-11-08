# Test fixtures

Committed artifacts that tests need, in contrast to the git-ignored
`tokenizer/model/` and `checkpoints/` directories.

## `tokenizer/`

The 1280-token debug BPE tokenizer that the generation, harness, and safety
tests were written against (the same files as `tokenizer/model/debug/`, which
is git-ignored). It is committed here so a fresh clone, including CI, can run
the full suite without pretraining a tokenizer or downloading data.

The tokenizer is immutable for test purposes. If the tokenizer format changes,
regenerate it with `python -m tokenizer.train_tokenizer --config configs/debug.yaml`
and copy `tokenizer.json`, `vocab.json`, and `merges.txt` here.
