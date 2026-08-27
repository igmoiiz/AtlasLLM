"""Special tokens for the AtlasLLM vocabulary.

These four tokens are always the first entries in the trained vocabulary,
so their IDs are stable: <pad>=0, <unk>=1, <bos>=2, <eos>=3.
"""

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]
