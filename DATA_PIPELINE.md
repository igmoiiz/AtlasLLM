DATA_PIPELINE.md — AtlasLLM Automated Dataset Pipeline

1. Objective

AtlasLLM must have a fully automated data pipeline.

The user should not manually collect, clean, merge, tokenize, or prepare training data.

The pipeline should transform publicly available datasets into reproducible AtlasLLM training shards.

---

1. High-Level Pipeline

Configuration
      ↓
Dataset Registry
      ↓
Source Resolution
      ↓
Download / Streaming
      ↓
Raw Cache
      ↓
Schema Normalization
      ↓
Text Cleaning
      ↓
Language Filtering
      ↓
Quality Filtering
      ↓
Document Filtering
      ↓
Deduplication
      ↓
Dataset Sampling
      ↓
Dataset Mixing
      ↓
Deterministic Split
      ↓
Tokenizer Training
      ↓
Tokenization
      ↓
Sequence Packing
      ↓
Shard Generation
      ↓
Validation
      ↓
Manifest

---

1. Primary Dataset

Primary pretraining source:

FineWeb-Edu

The entire dataset must NOT be downloaded.

The pipeline must select controlled subsets.

The selected amount must be configurable.

---

1. Supporting Sources

Initial mixture:

FineWeb-Edu                 60%
Wikipedia / encyclopedia    15%
Public-domain books         15%
Open technical documents    10%

These values are defaults, not permanent truths.

The data configuration must make them adjustable.

---

1. Debug Dataset

TinyStories should be supported as a debug source.

Use it for:

pipeline tests
tokenizer tests
tiny-model training
overfitting
smoke tests

Do not treat TinyStories as the primary AtlasLLM pretraining corpus.

---

1. Dataset Tiers

Tiny

10M–50M tokens

Purpose:

debugging
pipeline validation
tiny-model training

---

Base

100M–300M tokens

Purpose:

first real AtlasLLM training

---

Extended

500M–1B+ tokens

Purpose:

scaling experiments

---

1. Dataset Registry

Create a central registry/configuration describing datasets.

Conceptually:

datasets:

  fineweb_edu:
    role: pretraining
    source: HuggingFace
    identifier: HuggingFaceFW/fineweb-edu
    enabled: true
    target_fraction: 0.60

  wikipedia:
    role: pretraining
    source: HuggingFace
    enabled: true
    target_fraction: 0.15

  public_domain_books:
    role: pretraining
    source: configured
    enabled: true
    target_fraction: 0.15

  technical_docs:
    role: pretraining
    source: configured
    enabled: true
    target_fraction: 0.10

The actual implementation must verify the correct dataset identifiers and available configurations instead of blindly assuming them.

---

1. Source Adapter Design

Dataset-specific loading logic should be isolated.

Conceptually:

Dataset Source
      ↓
Source Adapter
      ↓
Standard Document Record

Every source should eventually produce a normalized document representation such as:

document_id
text
source
source_dataset
language
metadata

Do not make the rest of the pipeline understand every dataset's unique schema.

---

1. Downloading

The pipeline should support:

download
stream
cache
resume
verify

For very large sources, prefer streaming or bounded downloads.

The pipeline must not unnecessarily download data that will never be used.

---

 1. Raw Storage

Example:

data/
    raw/
        fineweb_edu/
        wikipedia/
        books/
        technical_docs/

Raw files must be immutable.

---

 1. Normalization

Normalize each source into the common document representation.

Possible operations:

Unicode normalization
whitespace normalization
line-ending normalization
HTML artifact removal
control-character removal

Do not aggressively rewrite legitimate text.

The original raw source remains available.

---

 1. Document Filtering

Remove documents that are clearly unsuitable.

Examples:

empty documents
near-empty documents
malformed records
obviously corrupted records
extremely repetitive records

Every filter must be measurable.

Record:

input count
accepted count
rejected count
rejection reason

---

 1. Language Filtering

AtlasLLM's initial corpus is primarily English.

The pipeline should filter according to configured language requirements.

Language detection must not be assumed perfect.

The pipeline should record the detected language and filtering decision.

---

 1. Quality Filtering

FineWeb-Edu already provides educational-quality filtering.

Do not unnecessarily recreate an expensive educational-quality classifier over the entire corpus.

For supporting datasets, apply appropriate lightweight filtering.

Possible signals:

text length
alphabetic ratio
repetition
boilerplate
URL density
HTML remnants
garbage-character ratio

Quality thresholds must be configuration values.

---

 1. Deduplication

Deduplication is required.

At minimum:

exact document deduplication

Later, if computationally justified:

near-duplicate detection

Do not build an extremely expensive deduplication system before measuring the need.

---

 1. Sampling

The pipeline must support deterministic sampling.

For example:

seed = 42
target_tokens = configurable

Running the same configuration and dataset version should produce the same selection where deterministic processing is possible.

---

 1. Dataset Mixing

After filtering:

FineWeb-Edu
Wikipedia
Books
Technical Docs

are sampled according to the configured mixture.

Example:

60 : 15 : 15 : 10

The mixture should be based on actual usable data rather than blindly duplicating a source when it runs out.

---

 1. Splitting

Split before sequence construction.

Initial target:

90% train
5% validation
5% test

The split must be deterministic.

Prefer document-level splitting.

Do not split a document's individual token sequences randomly between train and test.

This reduces leakage.

---

 1. Tokenizer Training

The tokenizer should be trained on a representative sample of the training corpus.

Target vocabulary:

16,000 tokens

The tokenizer version must be recorded.

The tokenizer must be treated as an artifact.

Example:

artifacts/tokenizer/

---

 1. Tokenization

Pipeline:

Document
 ↓
Tokenizer
 ↓
Token IDs

The tokenized corpus must be generated without loading the entire corpus into RAM.

---

 1. Sequence Packing

Initial context length:

256

For each training example:

input  = token[t : t+256]
target = token[t+1 : t+257]

Efficient packing should minimize wasted tokens.

Padding should be avoided where possible for pretraining.

---

 1. Sharding

Tokenized training data should be divided into manageable shards.

Example conceptual structure:

data/tokenized/
    train/
        shard-00000
        shard-00001
        shard-00002

    validation/
        shard-00000

    test/
        shard-00000

The actual storage format should be chosen based on performance and simplicity.

Do not invent a complex database when simple sharded files are sufficient.

---

 1. Manifest

Every dataset build must create a manifest.

Example conceptual metadata:

dataset:
  name: AtlasBase
  version: 1

sources:

- name: fineweb_edu
    fraction: 0.60
    version: ...
- name: wikipedia
    fraction: 0.15
- name: books
    fraction: 0.15
- name: technical_docs
    fraction: 0.10

processing:
  language: en
  deduplication: exact
  seed: 42

tokenizer:
  vocabulary_size: 16000
  version: ...

splits:
  train: 0.90
  validation: 0.05
  test: 0.05

statistics:
  documents:
  tokens:
  bytes:

The actual manifest must contain real measured values.

---

 1. Data Versioning

Dataset changes must produce a new dataset version.

For example:

AtlasTiny-v1
AtlasBase-v1
AtlasBase-v2
AtlasExtended-v1

A training checkpoint must record exactly which dataset version produced it.

---

 1. Pipeline Commands

The agent should expose simple commands.

Conceptually:

python -m atlasllm.data download --config configs/data/atlas_base.yaml

python -m atlasllm.data process --config configs/data/atlas_base.yaml

python -m atlasllm.data build --config configs/data/atlas_base.yaml

python -m atlasllm.tokenizer train --config configs/data/atlas_base.yaml

python -m atlasllm.data validate --manifest data/manifests/AtlasBase-v1.yaml

The exact CLI design can differ, but the workflow must remain simple.

---

 1. One Command Build

Eventually there should be a single reproducible command equivalent to:

build AtlasBase

which executes:

resolve sources
→ acquire
→ process
→ filter
→ deduplicate
→ sample
→ split
→ train tokenizer
→ tokenize
→ pack
→ shard
→ validate
→ write manifest

The pipeline should also allow each stage to be run independently for debugging.

---

 1. Resume Support

If a long pipeline fails:

do not restart everything unnecessarily

The system should detect existing valid artifacts and resume from the last completed stage.

Do not silently reuse corrupted or incompatible artifacts.

---

 1. Validation

Before a dataset becomes trainable, validate:

document count
token count
split sizes
vocabulary compatibility
token ID ranges
sequence lengths
missing/corrupt shards
duplicate shards
manifest integrity

The validator must fail loudly when something is wrong.

---

 1. Dataset Statistics

Generate reports containing:

documents
tokens
characters
average document length
median document length
minimum/maximum length
source distribution
language distribution
filter rejection counts
duplicate counts
train/validation/test counts

These reports should be saved under:

reports/data/

---

 1. Cost and Hardware Awareness

The pipeline must not assume unlimited:

disk
RAM
bandwidth
CPU
GPU

Data processing should primarily use CPU.

GPU should only be involved where it genuinely provides value.

Tokenization and filtering should be designed to avoid unnecessary repeated passes over huge corpora.

---

 1. Reproducibility

A dataset build must be reproducible from:

dataset configuration
source versions
processing code version
tokenizer configuration
random seed

The manifest should contain enough information to identify exactly how the dataset was produced.

---

 1. Instruction Dataset

Instruction data is a separate pipeline.

Conceptually:

instruction source
 ↓
normalize
 ↓
quality filter
 ↓
format conversion
 ↓
train/validation split
 ↓
SFT-ready examples

It must not be merged into the pretraining corpus.

---

 1. Reasoning Dataset

Reasoning data is also separate.

Conceptually:

reasoning problems
 ↓
quality filtering
 ↓
format normalization
 ↓
verification/filtering
 ↓
reasoning training dataset

Potential structure:

problem
reasoning
answer

The reasoning dataset should be held out from evaluation datasets.

---

 1. Evaluation Dataset

Evaluation data must have:

role: evaluation

and must never be passed to:

pretraining
instruction training
reasoning training

---

 1. Data Pipeline Success Criteria

The data pipeline is complete when the agent can execute:

fresh machine
 ↓
install dependencies
 ↓
run configuration
 ↓
automatically acquire data
 ↓
process data
 ↓
build AtlasTiny
 ↓
train tokenizer
 ↓
tokenize
 ↓
create shards
 ↓
validate
 ↓
produce manifest

without manual corpus preparation.

---

 1. First Dataset Milestone

Do NOT begin by attempting to build AtlasExtended.

First prove:

TinyStories
→ automated acquisition
→ preprocessing
→ tokenizer
→ tokenization
→ sequence packing
→ shards
→ validation
→ tiny model
→ successful overfit

Then:

FineWeb-Edu sample
→ AtlasBase
→ real pretraining

This prevents debugging the model and a massive data pipeline simultaneously.

---

 1. Final Data Philosophy

The dataset is part of the experiment.

Never say:

«"The model performed better."»

without knowing:

what data
which version
how much data
which mixture
which tokenizer
which preprocessing

AtlasLLM should make the relationship between:

data
→ training
→ model
→ evaluation

observable and reproducible.
