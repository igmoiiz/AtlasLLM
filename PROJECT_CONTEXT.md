AtlasLLM — Project Context

Project: AtlasLLM
Type: Educational dense decoder-only Large Language Model
Framework: PyTorch
Primary Hardware: NVIDIA GTX 1070 8GB VRAM, Intel Xeon E3-1270 v3, 32GB DDR3 RAM
Primary OS: Ubuntu 24.04.3 LTS
Status: Pre-implementation / project initialization

---

1. Project Vision

AtlasLLM is a small language model built from scratch to understand how modern decoder-only Transformers actually work.

The goal is not to compete with GPT, Claude, Gemini, Llama, Qwen, or other frontier models.

The goal is to build a complete, reproducible LLM engineering pipeline:

Raw Internet/Open Data
        ↓
Automated Data Acquisition
        ↓
Data Cleaning
        ↓
Quality Filtering
        ↓
Deduplication
        ↓
Dataset Mixing
        ↓
Train / Validation / Test Split
        ↓
Tokenizer Training
        ↓
Tokenization
        ↓
Sequence Packing
        ↓
Dataset Validation
        ↓
AtlasLLM Pretraining
        ↓
Base Model
        ↓
Instruction Tuning
        ↓
AtlasLLM-Instruct
        ↓
Reasoning Training
        ↓
AtlasLLM-Reasoning
        ↓
Evaluation Harness
        ↓
Inference / Generation

Everything should be automated and reproducible.

---

1. What We Are Building

AtlasLLM is a dense decoder-only Transformer implemented directly in PyTorch.

The core model must not simply instantiate an existing GPT implementation from Hugging Face.

The model should expose and implement the important components itself:

- token embeddings
- positional representation
- multi-head causal self-attention
- causal masking
- feed-forward network
- residual connections
- normalization
- Transformer blocks
- final normalization
- language-model head
- autoregressive next-token prediction

The model should be understandable by reading the source code.

---

1. Why We Chose a Dense Transformer First

AtlasLLM is deliberately dense.

The next major architecture project is:

AtlasLLM
    ↓
AtlasMoE

AtlasMoE will eventually investigate:

- mixture-of-experts routing
- sparse expert activation
- router behavior
- expert specialization
- computational efficiency
- distributed training concepts
- visualization
- comparison against the dense AtlasLLM baseline

AtlasMoE must therefore not replace AtlasLLM.

AtlasLLM is the control/baseline model.

---

1. Current Hardware Constraint

Primary machine:

CPU: Intel Xeon E3-1270 v3
GPU: NVIDIA GTX 1070 8GB
RAM: 32GB DDR3 1600MHz
SSD: 512GB
HDD: 1TB
PSU: 400W
OS: Windows 10

The GTX 1070's 8GB VRAM is a hard engineering constraint.

The project must therefore prioritize:

- small model size
- gradient accumulation
- memory-aware batch sizing
- sequence-length control
- checkpointing
- CPU-side dataset preparation
- streaming/chunked dataset processing
- avoiding unnecessary GPU memory usage
- profiling before optimization

The project must not assume access to a large GPU cluster.

---

1. Initial Model

Initial baseline configuration:

vocab_size: 16000
context_length: 256

d_model: 256
n_layers: 6
n_heads: 8
d_ff: 1024

dropout: 0.1

learning_rate: 3e-4
weight_decay: configurable

optimizer: AdamW

scheduler:
  warmup: true
  decay: cosine

precision:
  initial: fp32
  later: fp16 experimentation

These are baseline configuration values, not immutable constants.

The configuration system must allow experiments without modifying model source code.

---

1. Debug Model

Before real training, a tiny model must exist for correctness testing.

Example:

vocab_size: 2048
context_length: 32

d_model: 64
n_layers: 2
n_heads: 4
d_ff: 256

The debug model exists to test:

- forward pass
- backward pass
- causal masking
- loss
- gradient flow
- checkpointing
- generation
- dataset loading
- overfitting

It should be possible to overfit a tiny dataset.

If the tiny model cannot overfit a tiny dataset, real pretraining must not begin.

---

1. Transformer Architecture

AtlasLLM uses a pre-normalization Transformer.

Conceptually:

Token IDs
    ↓
Token Embeddings
    +
Positional Representation
    ↓
Transformer Block × N
    ↓
Final LayerNorm
    ↓
Linear LM Head
    ↓
Vocabulary Logits

Each block:

x
 ↓
LayerNorm
 ↓
Multi-Head Causal Self-Attention
 ↓
Residual Add
 ↓
LayerNorm
 ↓
Feed Forward Network
 ↓
Residual Add

Mathematically:

x = x + Attention(LN(x))
x = x + MLP(LN(x))

---

1. Attention

For input:

X ∈ R^(B × T × D)

compute:

Q = XWq
K = XWk
V = XWv

Attention:

Attention(Q,K,V)
=

softmax(QKᵀ / √d_k)V

A causal mask must prevent position "t" from seeing future positions.

For example:

token 1 → token 1
token 2 → token 1,2
token 3 → token 1,2,3
token 4 → token 1,2,3,4

Causal masking is a mandatory correctness test.

---

1. Feed-Forward Network

Initial FFN:

256 → 1024 → 256

with GELU activation.

Later experiments may investigate alternatives such as SwiGLU, but the first implementation should remain simple.

---

 1. Training Objective

AtlasLLM is autoregressive.

Given:

The cat sat on

the model learns to predict:

the

Then:

The cat sat on the

predict:

mat

Training uses next-token cross entropy.

Conceptually:

input  = tokens[0:-1]
target = tokens[1:]

Loss:

CrossEntropy(logits, targets)

Padding tokens must be excluded from the loss when applicable.

---

 1. Training Strategy

Initial optimizer:

AdamW

Initial learning rate:

3e-4

Training should include:

- learning-rate warmup
- cosine decay
- gradient clipping
- gradient accumulation when necessary
- checkpointing
- validation evaluation
- metric logging
- reproducibility metadata

The system should automatically adapt the effective batch size around available VRAM where practical.

---

 1. Precision Strategy

First:

FP32

Reason:

Correctness comes first.

After the baseline works:

FP16 experiments

The GTX 1070 supports FP16 operations, but the project must measure whether they actually improve useful training throughput rather than assuming they will.

---

 1. Dataset Decision

The earlier idea of manually collecting a small corpus is abandoned.

The project will use an automated data acquisition pipeline.

Primary source:

FineWeb-Edu

Supporting sources:

- Wikipedia / encyclopedic text
- public-domain books
- openly licensed technical/educational documentation

Debugging source:

TinyStories

Reasoning/instruction datasets are separate from pretraining data.

---

 1. Why FineWeb-Edu

FineWeb-Edu provides a large educationally filtered web corpus.

We do NOT intend to download the entire corpus.

Instead, AtlasLLM will create controlled subsets appropriate for the available hardware.

The project should support configurable dataset tiers.

---

 1. Atlas Dataset Tiers

AtlasTiny

Purpose:

- pipeline testing
- tokenizer testing
- model debugging
- overfitting
- CI tests
- smoke tests

Approximate scale:

10M–50M tokens

---

AtlasBase

Purpose:

- first meaningful AtlasLLM pretraining run

Approximate scale:

100M–300M tokens

This is the primary experimental scale.

---

AtlasExtended

Purpose:

- scaling experiments
- studying data/model scaling
- longer training runs

Approximate scale:

500M–1B+ tokens

The system must not require AtlasExtended for the project to be considered complete.

---

 1. Dataset Mixture

Initial target mixture:

60% FineWeb-Edu
15% Wikipedia / encyclopedic material
15% public-domain books
10% open technical documentation

These percentages must be configurable.

Do not hard-code them throughout the codebase.

A single configuration should control the mixture.

---

 1. Important Dataset Separation

Pretraining data:

General language / knowledge corpus

Instruction data:

Question → answer / instruction → response

Reasoning data:

Problem → reasoning process → answer

Evaluation data:

NEVER used for training

The system must prevent accidental evaluation-data contamination.

---

 1. Training Stages

AtlasLLM is not one training operation.

The lifecycle is:

Stage 1
Pretraining
    ↓
AtlasLLM Base

then:

Stage 2
Instruction SFT
    ↓
AtlasLLM-Instruct

then:

Stage 3
Reasoning-focused training
    ↓
AtlasLLM-Reasoning

This allows us to compare what each training stage actually changes.

---

 1. Why Reasoning Is a Separate Stage

We want to determine experimentally whether a small model can gain useful reasoning behavior through training.

We are NOT pretending that a 256-dimensional, 6-layer model will become a frontier reasoning model.

The research question is:

«Does reasoning-focused training measurably improve performance on unseen reasoning problems compared with a similarly sized instruction-tuned model?»

Potential experiment:

Model A
Base model

vs.

Model B
Instruction tuned

vs.

Model C
Instruction + reasoning training

vs.

Model D
Instruction + reasoning + verification/quality filtering

---

 1. Tokenizer

Initial target:

16,000 vocabulary tokens

Special tokens should include appropriate:

BOS
EOS
UNK
PAD

The tokenizer must be trained from the selected corpus rather than blindly using a tokenizer designed for another model.

Possible initial implementation:

SentencePiece

or another clearly justified tokenizer implementation.

Only one baseline tokenizer should be maintained.

---

 1. Data Pipeline

The agent must build the data pipeline.

The user should NOT manually download or clean datasets.

Pipeline:

Dataset Configuration
        ↓
Source Resolver
        ↓
Hugging Face / Source Download
        ↓
Streaming / Chunked Processing
        ↓
Raw Dataset Cache
        ↓
Normalization
        ↓
Language Filtering
        ↓
Quality Filtering
        ↓
Document Filtering
        ↓
Deduplication
        ↓
Dataset Mixing
        ↓
Train / Validation / Test Split
        ↓
Tokenizer Training
        ↓
Tokenization
        ↓
Sequence Packing
        ↓
Dataset Validation
        ↓
Training Shards

---

 1. Data Provenance

Every generated dataset must have metadata.

At minimum:

dataset name
dataset version
source
source URL/reference
license
download date
language
document count
raw size
processed size
token count
filtering configuration
deduplication configuration
tokenizer version
dataset configuration version
train/validation/test split
processing software version

The dataset must be reproducible.

---

 1. Raw Data Must Be Immutable

The pipeline must distinguish:

raw/
processed/
tokenized/
splits/

Raw downloaded material must never be modified in-place.

Every transformation creates a new derived artifact.

---

 1. Data Processing Requirements

The pipeline should support:

- streaming datasets where practical
- chunked processing
- configurable sampling
- text normalization
- empty-document removal
- extremely-short-document removal
- malformed-record handling
- language filtering
- duplicate removal
- near-duplicate detection where practical
- source balancing
- deterministic sampling
- deterministic train/validation/test splitting
- tokenizer training
- tokenization
- sequence packing
- shard generation
- metadata generation

It must not load multi-hundred-million-token datasets entirely into RAM.

---

 1. Dataset Contamination

Evaluation datasets must never accidentally enter training.

The pipeline should maintain explicit dataset roles:

pretraining
instruction
reasoning
evaluation

Evaluation sources must be isolated.

Dataset manifests should record the exact sources used for each training run.

---

 1. Evaluation

Evaluation happens at multiple levels.

Training Metrics

- training loss
- validation loss
- test loss
- perplexity
- learning rate
- gradient norm
- throughput
- tokens/sec
- GPU memory usage
- checkpoint step

Perplexity:

PPL = exp(loss)

---

 1. Generalization Testing

The model must not only be tested on training examples.

Evaluation should include:

- unseen text
- held-out documents
- different topics
- different writing styles
- basic instructions
- reasoning problems
- format-following
- language behavior
- robustness

---

 1. Memorization Testing

The project should investigate memorization.

Compare:

training passages
near-duplicate passages
held-out passages

The goal is to understand whether the model is learning useful patterns or simply reproducing training material.

---

 1. Inference

Inference pipeline:

Prompt
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Model
 ↓
Logits
 ↓
Sampling
 ↓
Next Token
 ↓
Append
 ↓
Repeat

Initial decoding modes:

- greedy
- temperature
- top-k
- top-p

Later:

- repetition controls
- KV cache
- streaming generation

---

 1. KV Cache

KV caching should be implemented only after baseline generation works.

Measure:

generation latency
tokens/sec
GPU memory

Compare cached vs uncached generation.

---

 1. Safety Architecture

Safety should initially exist outside the base model.

Architecture:

User Input
    ↓
Input Guardrail
    ↓
AtlasLLM
    ↓
Output Guardrail
    ↓
Response

Do not contaminate the core Transformer with application-specific safety logic.

---

 1. Evaluation Harness

The project needs a reusable evaluation harness.

Architecture:

Scenario
    ↓
Harness Runner
    ↓
Input Guardrail
    ↓
Model
    ↓
Output Guardrail
    ↓
Scorer
    ↓
Report

Categories:

capability
language
instruction following
reasoning
format compliance
robustness
hallucination
safety
adversarial behavior
regression

Every meaningful discovered failure should become a regression test.

---

 1. Mandatory Tests

Before real training:

- tokenizer test
- dataset loading test
- tensor shape test
- forward-pass test
- backward-pass test
- causal-mask test
- loss test
- gradient-flow test
- checkpoint save/load test
- generation test
- tiny-dataset overfit test

---

 1. Reproducibility

Every meaningful experiment must record:

random seed
dataset version
dataset manifest
tokenizer version
model configuration
optimizer
scheduler
learning rate
batch size
gradient accumulation
context length
precision
software versions
hardware
training duration
checkpoint
metrics

---

 1. Configuration

There must be one source of truth for configuration.

Do not scatter:

d_model = 256

through multiple files.

Configuration should define:

- model
- training
- data
- tokenizer
- hardware
- logging
- checkpointing
- evaluation

Experiments should be represented through configuration files or explicit experiment definitions.

---

 1. Project Structure

The exact structure may evolve, but the conceptual separation should be:

AtlasLLM/
│
├── AGENTS.md
├── CONTEXT.md
├── DATA_PIPELINE.md
├── README.md
│
├── configs/
│   ├── model/
│   ├── data/
│   ├── training/
│   └── experiments/
│
├── atlasllm/
│   ├── model/
│   ├── data/
│   ├── tokenizer/
│   ├── training/
│   ├── inference/
│   ├── evaluation/
│   ├── safety/
│   └── utils/
│
├── scripts/
│   ├── data/
│   ├── tokenizer/
│   ├── training/
│   ├── evaluation/
│   └── inference/
│
├── tests/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── tokenized/
│   └── manifests/
│
├── checkpoints/
├── logs/
├── reports/
└── docs/

The agent may adjust this structure when implementation reveals a better minimal structure, but architectural responsibilities must remain separated.

---

 1. Development Order

Do not attempt to implement the entire system simultaneously.

Correct sequence:

1. Repository
2. Documentation
3. Configuration
4. Environment
5. Tokenizer pipeline
6. Dataset pipeline
7. Tiny dataset
8. Transformer implementation
9. Unit tests
10. Tiny overfit experiment
11. Training loop
12. Checkpointing
13. Real AtlasBase dataset
14. First pretraining run
15. Evaluation
16. Inference
17. Instruction SFT
18. Reasoning training
19. Evaluation harness
20. KV cache / performance work
21. Documentation of results

---

 1. What We Explicitly Do NOT Want

AtlasLLM is not:

- a GPT clone using Hugging Face's GPT implementation
- a frontier-scale model
- a billion-parameter local model
- a commercial chatbot
- an application-specific assistant
- an overengineered framework
- a collection of random notebooks
- a pile of duplicated utilities
- a model trained on manually collected random text

---

 1. Engineering Philosophy

The project should answer:

«"Can I explain every important part of this model and training system?"»

If the answer is no, the implementation is too abstract.

The project prioritizes:

Correctness
    >
Understanding
    >
Reproducibility
    >
Measurement
    >
Performance

Only optimize after the baseline is correct.

---

 1. Current Mission

The project is now ready to begin implementation.

The first objective is NOT:

«"Train the model."»

The first objective is:

«"Build a reproducible automated data + model engineering system capable of eventually training AtlasLLM."»

The first implementation milestone should therefore establish:

repository
+
configuration
+
automated dataset pipeline
+
tokenizer pipeline
+
tiny dataset
+
model skeleton
+
tests

Only after those pieces pass validation should real pretraining begin.
