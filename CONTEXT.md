# AtlasLLM
## Complete Project Execution Specification

**Project Type:** Dense Decoder-Only Transformer Language Model  
**Primary Framework:** PyTorch  
**Development Machine:** Main desktop PC  
**GPU:** NVIDIA GTX 1070 — 8 GB VRAM  
**CPU:** Intel Xeon E3-1270 v3  
**RAM:** 32 GB DDR3 1600 MHz  
**Storage:** 512 GB SSD + 1 TB HDD + 500 GB external storage  
**Operating System:** Ubuntu 24.04.3 LTS  
**Project Position:** First project in the Atlas AI/LLM series  
**Successor:** AtlasMoE

---

# 1. Project Definition

AtlasLLM is a small, dense, decoder-only Transformer language model implemented in PyTorch with the architecture, training pipeline, evaluation system, inference engine, safety layer, experiment framework, and deployment interface built as a complete engineering system.

The project is not intended to compete with commercial foundation models.

The primary objective is to understand and implement the complete lifecycle of a language model:

```text
Raw Data
    ↓
Data Cleaning
    ↓
Dataset Construction
    ↓
Tokenizer
    ↓
Tokenized Dataset
    ↓
Transformer Architecture
    ↓
Pretraining
    ↓
Validation
    ↓
Evaluation
    ↓
Instruction Tuning
    ↓
Safety / Guardrails
    ↓
Inference
    ↓
Harness
    ↓
Evaluation & Monitoring
```

AtlasLLM must be understandable at the implementation and mathematical level.

Every major subsystem should be independently inspectable.

---

# 2. Core Objectives

AtlasLLM has six primary objectives.

## 2.1 Transformer Understanding

Implement and understand:

- token embeddings
- positional representations
- self-attention
- causal masking
- multi-head attention
- feed-forward networks
- residual connections
- normalization
- Transformer blocks
- language-model heads

## 2.2 LLM Training

Build the complete training pipeline:

- dataset loading
- batching
- sequence packing
- forward propagation
- loss calculation
- backpropagation
- optimization
- learning-rate scheduling
- gradient clipping
- checkpointing
- validation
- experiment tracking

## 2.3 Generalization

The model must not merely memorize the training corpus.

Testing must explicitly measure:

- validation loss
- perplexity
- held-out text performance
- prompt completion
- instruction-following behavior
- repetition
- memorization
- out-of-distribution behavior

## 2.4 Inference

Implement generation from first principles:

- greedy decoding
- temperature
- top-k
- top-p
- repetition controls
- EOS handling
- streaming generation
- KV caching

## 2.5 Safety

Implement a lightweight safety architecture around the model.

The guardrail system must distinguish:

```text
Model capability
        ≠
Safety policy
```

The base model should remain a language model.

Safety behavior should be implemented through dedicated policy and filtering layers rather than pretending that a small pretrained model is inherently safe.

## 2.6 Engineering

The final project should resemble a miniature production LLM stack rather than a single training notebook.

---

# 3. Hardware Capability

The main PC:

```text
CPU: Intel Xeon E3-1270 v3
GPU: GTX 1070 8 GB
RAM: 32 GB
SSD: 512 GB
HDD: 1 TB
```

is suitable for:

- implementing the complete architecture
- CPU experiments
- small GPU training
- tokenizer training
- small-scale pretraining
- instruction tuning
- inference
- benchmarking
- experimentation
- visualization
- evaluation

It is not suitable for:

- training a modern 7B+ foundation model from scratch
- large-scale distributed pretraining
- large-context training at substantial batch sizes
- training frontier-scale models

AtlasLLM therefore deliberately targets a small model.

---

# 4. Initial Hardware-Aware Model

The first serious configuration:

```yaml
vocab_size: 16000
context_length: 256

d_model: 256
n_layers: 6
n_heads: 8
d_ff: 1024

dropout: 0.1

batch_size: 8
learning_rate: 3e-4
```

This configuration is deliberately conservative.

The architecture must remain configurable so larger experiments can be performed when VRAM permits.

---

# 5. Software Stack

## Core

```text
Python
PyTorch
CUDA
Git
GitHub
```

## Data

```text
datasets
numpy
pandas
```

## Tokenization

Primary:

```text
SentencePiece
```

or:

```text
Hugging Face Tokenizers
```

The tokenizer implementation should remain isolated from the model.

## Configuration

Use:

```text
YAML
```

with configuration files defining experiments.

## Experiment Tracking

Use:

```text
TensorBoard
```

for the initial implementation.

The training system should record:

- training loss
- validation loss
- learning rate
- gradient norm
- tokens processed
- throughput
- GPU memory
- checkpoint number

## Testing

Use:

```text
pytest
```

## Formatting / Quality

Use:

```text
ruff
black
```

where appropriate.

## Visualization

Use:

```text
matplotlib
```

for model and training analysis.

---

# 6. Repository Architecture

```text
AtlasLLM/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── debug.yaml
│   ├── small.yaml
│   ├── medium.yaml
│   └── experiments/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
│
├── tokenizer/
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── train_tokenizer.py
│   └── vocabulary.py
│
├── model/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── normalization.py
│   ├── attention.py
│   ├── feed_forward.py
│   ├── transformer_block.py
│   ├── positional_encoding.py
│   └── atlas_llm.py
│
├── data_pipeline/
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── packing.py
│   └── dataloader.py
│
├── training/
│   ├── loss.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── trainer.py
│   ├── checkpoint.py
│   └── train.py
│
├── evaluation/
│   ├── perplexity.py
│   ├── generation_eval.py
│   ├── memorization.py
│   ├── robustness.py
│   └── benchmarks.py
│
├── inference/
│   ├── generate.py
│   ├── sampling.py
│   ├── kv_cache.py
│   └── engine.py
│
├── safety/
│   ├── policy.py
│   ├── input_filter.py
│   ├── output_filter.py
│   └── guardrails.py
│
├── harness/
│   ├── runner.py
│   ├── scenarios.py
│   ├── scoring.py
│   ├── adversarial.py
│   └── reports.py
│
├── monitoring/
│   ├── metrics.py
│   ├── profiler.py
│   └── system.py
│
├── scripts/
│   ├── count_parameters.py
│   ├── inspect_dataset.py
│   ├── benchmark.py
│   └── chat.py
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_attention.py
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_generation.py
│   └── test_safety.py
│
├── notebooks/
│   ├── attention_visualization.ipynb
│   ├── training_analysis.ipynb
│   └── evaluation.ipynb
│
└── checkpoints/
```

---

# 7. Data Strategy

The dataset is one of the most important components.

The first model should use a relatively small, clean text corpus.

The data pipeline must contain:

```text
Collection
↓
Deduplication
↓
Cleaning
↓
Quality filtering
↓
Train/validation split
↓
Tokenization
↓
Sequence construction
↓
Storage
```

---

# 8. Data Sources

The project should prioritize legally usable and openly licensed/public-domain text.

Potential categories:

```text
public-domain books
Wikipedia-style encyclopedic text
open educational material
open documentation
open-source technical text
permissively licensed datasets
```

Dataset provenance must be recorded.

Create:

```text
data/README.md
```

containing:

```text
dataset name
source
license
download date
size
language
preprocessing
train/validation split
token count
```

The dataset must never become an undocumented collection of scraped material.

---

# 9. Data Cleaning

Cleaning pipeline:

```text
Raw document
    ↓
Unicode normalization
    ↓
Whitespace normalization
    ↓
Remove corrupted records
    ↓
Remove duplicates
    ↓
Remove extremely short records
    ↓
Quality filtering
    ↓
Final corpus
```

The pipeline should preserve meaningful punctuation and formatting.

Do not aggressively clean natural language into artificial text.

---

# 10. Train/Validation Split

The split must occur before sequence construction in order to reduce leakage.

Initial target:

```text
90% training
10% validation
```

Validation data must never participate in gradient updates.

A separate test set should eventually be created for final evaluation.

Target:

```text
90% train
5% validation
5% test
```

---

# 11. Tokenizer

AtlasLLM will initially use a subword tokenizer.

Target vocabulary:

```text
16,000 tokens
```

Tokenizer responsibilities:

```text
encode(text)
decode(tokens)
save()
load()
```

Special tokens:

```text
<BOS>
<EOS>
<UNK>
<PAD>
```

Tokenizer tests must verify:

```text
encode → decode
```

behavior.

---

# 12. Sequence Construction

For context length:

```text
256 tokens
```

a training sequence becomes:

```text
Input:
t0 t1 t2 t3 ... t254

Target:
t1 t2 t3 t4 ... t255
```

The model therefore learns next-token prediction.

---

# 13. Model Architecture

AtlasLLM is decoder-only.

```text
Token IDs
    ↓
Token Embedding
    +
Position Representation
    ↓
Transformer Block × N
    ↓
Final Normalization
    ↓
Language Model Head
    ↓
Vocabulary Logits
```

---

# 14. Embedding Layer

Implement:

```python
nn.Embedding(vocab_size, d_model)
```

The token embedding converts discrete token IDs into continuous vectors.

---

# 15. Positional Representation

Version 1:

```text
learned positional embeddings
```

Later experimental implementations:

```text
sinusoidal
RoPE
```

Only one positional system should be active in a given configuration.

---

# 16. Self-Attention

For each input representation:

```text
Q = XWq
K = XWk
V = XWv
```

Attention:

```text
Attention(Q,K,V)
=
softmax(QKᵀ / √d_k)V
```

AtlasLLM must use causal masking.

For a sequence:

```text
A B C D
```

the attention visibility must be:

```text
A → A
B → A B
C → A B C
D → A B C D
```

Future tokens must never influence the current prediction.

---

# 17. Multi-Head Attention

Initial configuration:

```text
d_model = 256
n_heads = 8
head_dim = 32
```

Each head independently computes attention.

The heads are concatenated and projected back to `d_model`.

---

# 18. Feed-Forward Network

Initial configuration:

```text
256 → 1024 → 256
```

Activation:

```text
GELU
```

Later experiment:

```text
SwiGLU
```

---

# 19. Normalization

Use LayerNorm.

The initial architecture uses pre-normalization:

```text
x = x + Attention(LayerNorm(x))

x = x + MLP(LayerNorm(x))
```

This structure should be explicitly documented because it affects training stability.

---

# 20. Transformer Block

Complete block:

```text
Input
  ↓
LayerNorm
  ↓
Multi-Head Causal Self-Attention
  ↓
Residual Addition
  ↓
LayerNorm
  ↓
Feed-Forward Network
  ↓
Residual Addition
  ↓
Output
```

---

# 21. Language Model Head

Final hidden states are projected into vocabulary logits:

```text
hidden_size → vocabulary_size
```

For:

```text
d_model = 256
vocab_size = 16000
```

the output dimension is:

```text
16000
```

Each position produces a probability distribution over possible next tokens after softmax.

---

# 22. Loss Function

Use autoregressive cross-entropy.

The model receives:

```text
The cat sat
```

and learns:

```text
cat
sat
on
```

respectively.

Padding positions, if present, must be excluded from loss calculation.

---

# 23. Optimizer

Initial optimizer:

```text
AdamW
```

Initial learning rate:

```text
3e-4
```

Weight decay should be configurable.

---

# 24. Learning-Rate Schedule

Implement:

```text
warmup
    ↓
decay
```

The exact schedule must remain configurable.

A common initial experiment:

```text
linear warmup
+
cosine decay
```

---

# 25. Gradient Management

Training must support:

```text
gradient clipping
gradient norm logging
optional gradient accumulation
```

Gradient clipping protects against unstable updates.

---

# 26. Mixed Precision

Development order:

```text
FP32
↓
verified training
↓
FP16 experiment
```

The first successful training run should be performed without adding unnecessary numerical complexity.

---

# 27. Checkpoint System

Each checkpoint should contain:

```text
model state
optimizer state
scheduler state
training step
epoch
configuration
random-state information
metrics
```

Example:

```text
checkpoints/
└── step_10000/
    ├── model.pt
    ├── optimizer.pt
    ├── scheduler.pt
    ├── config.yaml
    └── metrics.json
```

Training must be resumable.

---

# 28. Reproducibility

Every experiment should record:

```text
random seed
dataset version
tokenizer version
model configuration
PyTorch version
CUDA version
GPU
training steps
learning rate
batch size
context length
```

A training result without configuration metadata is not considered a valid experiment.

---

# 29. Experiment System

Experiments must be configuration-driven.

Example:

```text
configs/
├── small.yaml
├── medium.yaml
└── experiments/
    ├── depth.yaml
    ├── context.yaml
    ├── heads.yaml
    └── positional.yaml
```

This prevents manually changing source code between experiments.

---

# 30. Generalization Strategy

Generalization is a first-class project requirement.

AtlasLLM must be evaluated on data that was not used during training.

Measure:

```text
training loss
validation loss
test loss
perplexity
generation quality
repetition
memorization
out-of-distribution performance
```

A lower training loss alone does not demonstrate a better language model.

---

# 31. Memorization Testing

Create a memorization evaluation.

Test whether the model reproduces:

```text
training passages
near-duplicate passages
unseen passages
```

Compare generation behavior.

The evaluation should distinguish:

```text
learning language patterns
```

from:

```text
memorizing exact sequences
```

---

# 32. Overfitting Detection

Monitor:

```text
training loss
validation loss
```

Typical pattern:

```text
Training ↓
Validation ↓
```

is desirable.

Potential overfitting:

```text
Training ↓↓↓
Validation ↑
```

Checkpoints should preserve the best validation performance.

---

# 33. Perplexity

Calculate:

```text
PPL = exp(cross_entropy_loss)
```

Track perplexity on:

```text
validation set
test set
```

Perplexity must always be reported alongside dataset and tokenizer information because values are not directly comparable across incompatible tokenization schemes.

---

# 34. Inference Engine

Inference pipeline:

```text
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
Next token
 ↓
Append token
 ↓
Repeat
```

---

# 35. Generation Algorithms

Implement progressively.

## Greedy

```text
argmax(logits)
```

## Temperature

```text
logits / temperature
```

## Top-k

Restrict sampling to the K highest-probability tokens.

## Top-p

Restrict sampling to the smallest token set whose cumulative probability exceeds the selected probability threshold.

All parameters must be configurable.

---

# 36. Repetition Control

Implement configurable repetition controls.

Monitor:

```text
repeated tokens
repeated n-grams
sequence loops
```

Do not hide poor model behavior through aggressive sampling parameters.

Evaluation should include both raw and controlled generation.

---

# 37. KV Cache

Implement KV caching after the basic generation system works.

Without caching:

```text
previous tokens
→ recompute attention
```

With caching:

```text
previous K/V
+
new token
→ new computation
```

Measure:

```text
tokens/sec
latency/token
memory consumption
```

before and after KV caching.

---

# 38. Streaming

Inference should eventually support:

```text
token
token
token
token
...
```

rather than waiting for the entire response.

Streaming is an inference-engine feature, not a model feature.

---

# 39. Instruction Tuning

The pretrained model and assistant model should be treated as separate stages.

Pipeline:

```text
Pretraining
    ↓
Base AtlasLLM
    ↓
Instruction Dataset
    ↓
Supervised Fine-Tuning
    ↓
AtlasLLM-Instruct
```

The initial instruction-tuning dataset should remain small and controlled.

---

# 40. Instruction Dataset

Training examples should follow:

```json
{
  "instruction": "...",
  "response": "..."
}
```

Potential categories:

```text
question answering
summarization
classification
explanation
reasoning
coding
format following
refusal behavior
```

The instruction dataset must be separated from the pretraining corpus.

---

# 41. Guardrail Architecture

AtlasLLM should use layered safety.

```text
                 User Input
                     ↓
             Input Guardrail
                     ↓
              AtlasLLM-Instruct
                     ↓
             Output Guardrail
                     ↓
                  Response
```

The system should not assume that the language model itself provides reliable safety.

---

# 42. Input Guardrail

Input guardrails classify requests according to policy.

Categories can include:

```text
allowed
ambiguous
unsafe
```

The policy must be explicit and version-controlled.

The input guardrail should not modify ordinary prompts unnecessarily.

---

# 43. Output Guardrail

Generated output passes through an output safety check.

Potential actions:

```text
allow
block
replace with refusal
```

Safety decisions should be logged during testing.

---

# 44. Guardrail Limitations

The guardrail is not a guarantee of safety.

A small local model can:

- misunderstand requests
- produce hallucinations
- bypass weak filtering
- generate unsafe material
- fail to understand context

Therefore the project must distinguish:

```text
Safety mechanism
```

from:

```text
Safety guarantee
```

No absolute safety claim should be made.

---

# 45. Harness

The harness is one of the most important parts of the final project.

It provides an automated environment for testing AtlasLLM.

Architecture:

```text
Scenario
   ↓
Harness Runner
   ↓
Model
   ↓
Guardrails
   ↓
Output
   ↓
Scorer
   ↓
Report
```

---

# 46. Harness Categories

The harness should contain:

```text
basic capability tests
language tests
instruction tests
reasoning tests
format tests
robustness tests
hallucination tests
safety tests
adversarial tests
regression tests
```

---

# 47. Capability Harness

Examples:

```text
simple factual questions
basic arithmetic
text completion
classification
summarization
instruction following
structured output
```

These tests establish baseline capability.

---

# 48. Robustness Harness

Test:

```text
typos
capitalization changes
extra whitespace
different wording
irrelevant context
long prompts
short prompts
repeated prompts
```

The purpose is to determine whether behavior is stable under superficial changes.

---

# 49. Adversarial Harness

The adversarial harness attempts to identify failures in:

```text
prompt handling
instruction hierarchy
guardrails
format constraints
context handling
output filtering
```

Every discovered failure becomes a regression test.

---

# 50. Safety Harness

Safety tests should include:

```text
clearly allowed prompts
clearly disallowed prompts
ambiguous prompts
boundary cases
adversarial wording
encoded wording
multi-turn attempts
```

The objective is measurement, not claiming perfect protection.

---

# 51. Regression Testing

Every model release should be evaluated against the same test suite.

Example:

```text
AtlasLLM v0.3
       ↓
Harness
       ↓
baseline metrics

AtlasLLM v0.4
       ↓
Harness
       ↓
compare against baseline
```

An architectural improvement that improves training loss but breaks generation or safety behavior must be visible.

---

# 52. Evaluation Report

Every major model should produce:

```text
Model configuration
Parameter count
Dataset size
Token count
Training duration
Hardware
Training loss
Validation loss
Test loss
Perplexity
Generation benchmarks
Inference throughput
Latency
Memory consumption
Safety results
Harness results
Known failures
```

---

# 53. Benchmarking

Measure:

```text
parameters
VRAM
RAM
model load time
prompt processing speed
generation speed
tokens/sec
latency/token
```

Benchmark under consistent conditions.

---

# 54. Profiling

Use PyTorch profiling tools to determine where computation is spent.

Potential bottlenecks:

```text
attention
matrix multiplication
data loading
tokenization
GPU transfer
Python overhead
```

Optimization should be driven by measurements.

---

# 55. System Monitoring

During training monitor:

```text
GPU utilization
GPU memory
CPU utilization
RAM
disk usage
temperature
training throughput
```

The monitoring system should identify whether the bottleneck is:

```text
compute
memory
data loading
storage
```

---

# 56. Unit Testing

Before training:

```text
Tokenizer tests
Attention tests
Causal-mask tests
Shape tests
Gradient tests
Dataset tests
Generation tests
Guardrail tests
```

Critical mathematical properties must have tests.

For example:

```text
future tokens must not affect earlier-token predictions
```

must be explicitly tested.

---

# 57. Shape Contracts

Every module should have documented tensor shapes.

Example:

```text
Input:

[B, T]

Embedding:

[B, T, D]

Attention:

[B, T, D]

Logits:

[B, T, V]
```

where:

```text
B = batch
T = sequence length
D = model dimension
V = vocabulary size
```

Shape clarity prevents a large class of Transformer implementation errors.

---

# 58. Debug Configuration

Before GPU training, create:

```yaml
batch_size: 2
context_length: 32
d_model: 64
n_layers: 2
n_heads: 4
```

Train on a tiny dataset.

The model must successfully:

```text
forward
backward
update
checkpoint
reload
generate
```

before scaling.

---

# 59. Overfit-a-Tiny-Batch Test

This is a mandatory debugging test.

Take an extremely small dataset.

Train until the model nearly memorizes it.

If the model cannot overfit a tiny dataset, something is probably wrong with:

```text
data pipeline
loss
masking
optimizer
gradient flow
model implementation
```

This test should happen before serious pretraining.

---

# 60. Architecture Verification

Verify each component independently:

```text
Tokenizer
↓
Embedding
↓
Attention
↓
Causal masking
↓
MLP
↓
Transformer block
↓
Full model
```

Only after each layer works should they be composed.

---

# 61. Development Milestones

## Milestone 1 — Environment

Deliverables:

```text
Python environment
PyTorch
CUDA
Git repository
project structure
```

## Milestone 2 — Tokenizer

Deliverables:

```text
trained tokenizer
encode/decode
tests
vocabulary statistics
```

## Milestone 3 — Dataset

Deliverables:

```text
clean corpus
train/validation/test split
tokenized dataset
dataset statistics
```

## Milestone 4 — Transformer

Deliverables:

```text
embedding
attention
MLP
Transformer block
full AtlasLLM
```

## Milestone 5 — Debug Training

Deliverables:

```text
tiny dataset
overfit test
loss decreasing
checkpoint reload
```

## Milestone 6 — Real Training

Deliverables:

```text
pretraining
validation
TensorBoard
checkpoints
metrics
```

## Milestone 7 — Inference

Deliverables:

```text
generation
sampling
streaming
KV cache
```

## Milestone 8 — Evaluation

Deliverables:

```text
perplexity
benchmark suite
generalization tests
memorization tests
```

## Milestone 9 — Instruction Tuning

Deliverables:

```text
instruction dataset
SFT pipeline
AtlasLLM-Instruct
```

## Milestone 10 — Safety

Deliverables:

```text
input guardrail
output guardrail
safety policy
safety test suite
```

## Milestone 11 — Harness

Deliverables:

```text
automated scenarios
scoring
regression tests
adversarial tests
reports
```

## Milestone 12 — Release

Deliverables:

```text
AtlasLLM v1.0
documentation
model card
dataset documentation
architecture documentation
training report
benchmark report
known limitations
inference interface
```

---

# 62. AtlasLLM Version Structure

## v0.1

Transformer prototype.

## v0.2

Training framework.

## v0.3

First pretrained AtlasLLM.

## v0.4

Improved inference engine.

## v0.5

Instruction tuning.

## v0.6

Guardrails.

## v0.7

Evaluation harness.

## v0.8

Architecture experiments.

## v0.9

Optimization and profiling.

## v1.0

Complete documented system.

---

# 63. Research Experiments

After the baseline works, experiments should include:

```text
model depth
model width
number of heads
context length
vocabulary size
positional encoding
activation function
learning rate
batch size
weight decay
dropout
dataset size
dataset composition
```

Every experiment must have:

```text
hypothesis
configuration
result
interpretation
```

---

# 64. Critical Experiments

### Depth

```text
2
4
6
8
```

### Width

```text
128
256
384
512
```

### Context

```text
128
256
512
```

### Attention Heads

```text
4
8
```

### Positional Encoding

```text
learned
sinusoidal
RoPE
```

### Activation

```text
GELU
SwiGLU
```

---

# 65. What AtlasLLM Must Demonstrate

By the end of the project, the repository should demonstrate that the implementation can:

```text
train a tokenizer
prepare a corpus
construct causal language-modeling samples
implement Transformer attention
implement causal masking
train a Transformer
save checkpoints
resume training
evaluate perplexity
generate text
perform sampling
use KV caching
stream responses
fine-tune on instructions
apply safety policies
run automated evaluations
measure inference performance
detect regressions
```

---

# 66. What AtlasLLM Must Not Pretend To Be

AtlasLLM should not claim:

```text
GPT-level intelligence
human-level reasoning
general intelligence
perfect factual accuracy
perfect safety
production-grade safety
frontier-model performance
```

The project should explicitly document its limitations.

---

# 67. Final System Architecture

The completed system:

```text
                         ┌─────────────────┐
                         │   Data Sources   │
                         └────────┬────────┘
                                  ↓
                         ┌─────────────────┐
                         │ Data Processing │
                         └────────┬────────┘
                                  ↓
                         ┌─────────────────┐
                         │    Tokenizer    │
                         └────────┬────────┘
                                  ↓
                         ┌─────────────────┐
                         │ Tokenized Data  │
                         └────────┬────────┘
                                  ↓
                    ┌──────────────────────────┐
                    │       AtlasLLM           │
                    │                          │
                    │ Embeddings               │
                    │       ↓                  │
                    │ Transformer Blocks       │
                    │       ↓                  │
                    │ LM Head                  │
                    └────────────┬─────────────┘
                                 ↓
                    ┌──────────────────────────┐
                    │       Training           │
                    │                          │
                    │ Loss                     │
                    │ AdamW                    │
                    │ Scheduler                │
                    │ Checkpoints              │
                    └────────────┬─────────────┘
                                 ↓
                    ┌──────────────────────────┐
                    │      Evaluation          │
                    │                          │
                    │ Perplexity               │
                    │ Generalization           │
                    │ Memorization             │
                    │ Benchmarks               │
                    └────────────┬─────────────┘
                                 ↓
                    ┌──────────────────────────┐
                    │    Instruction Tuning    │
                    └────────────┬─────────────┘
                                 ↓
                    ┌──────────────────────────┐
                    │       Guardrails         │
                    │                          │
                    │ Input Policy             │
                    │ Output Policy            │
                    └────────────┬─────────────┘
                                 ↓
                    ┌──────────────────────────┐
                    │      Inference Engine    │
                    │                          │
                    │ Sampling                 │
                    │ KV Cache                 │
                    │ Streaming                │
                    └────────────┬─────────────┘
                                 ↓
                    ┌──────────────────────────┐
                    │       Harness            │
                    │                          │
                    │ Capability Tests         │
                    │ Safety Tests             │
                    │ Adversarial Tests        │
                    │ Regression Tests         │
                    └────────────┬─────────────┘
                                 ↓
                    ┌──────────────────────────┐
                    │     AtlasLLM Release     │
                    └──────────────────────────┘
```

---

# 68. Definition of Done

AtlasLLM is complete when the system can be cloned onto another machine and a developer can understand:

```text
where the data came from
how the data was cleaned
how the tokenizer works
how tokens become vectors
how attention works
how causal masking works
how Transformer blocks work
how the loss is calculated
how optimization occurs
how checkpoints are created
how training is evaluated
how generation works
how KV caching works
how instruction tuning works
how safety policies work
how the harness evaluates the model
how inference performance is measured
```

The project is successful when the resulting model is less important than the fact that every stage of its creation is understood, reproducible, measurable, and documented.

---

# 69. Relationship to AtlasMoE

AtlasLLM is the foundation for AtlasMoE.

AtlasLLM:

```text
Dense Transformer
        ↓
Every token
        ↓
Same FFN pathway
```

AtlasMoE will eventually introduce:

```text
Dense Transformer
        ↓
Router
        ↓
Expert 1
Expert 2
Expert 3
...
Expert N
        ↓
Selected experts
```

Therefore AtlasMoE should not begin until AtlasLLM's:

```text
attention
Transformer blocks
training
evaluation
inference
profiling
```

are understood and operational.

AtlasLLM establishes the dense baseline against which the MoE architecture can be measured.

---

# 70. Immediate Execution Order

The actual implementation order is:

```text
1. Create repository
2. Create Python environment
3. Verify PyTorch + CUDA
4. Build project structure
5. Implement configuration system
6. Implement tokenizer
7. Acquire and document dataset
8. Build preprocessing pipeline
9. Build dataset loader
10. Implement embeddings
11. Implement positional encoding
12. Implement causal self-attention
13. Implement multi-head attention
14. Implement feed-forward network
15. Implement Transformer block
16. Implement complete AtlasLLM
17. Implement loss
18. Implement optimizer
19. Implement training loop
20. Run tiny-batch overfit test
21. Run CPU/GPU sanity tests
22. Train first real model
23. Implement checkpointing
24. Implement validation
25. Implement TensorBoard logging
26. Train baseline
27. Implement inference
28. Implement sampling
29. Implement KV cache
30. Implement streaming
31. Build evaluation suite
32. Measure generalization
33. Measure memorization
34. Instruction-tune
35. Implement guardrails
36. Build safety harness
37. Build adversarial harness
38. Build regression harness
39. Profile and optimize
40. Document everything
41. Release AtlasLLM v1.0
```

**This is the execution blueprint. The first implementation target is not the full model. It is the environment + repository + configuration + tokenizer + dataset pipeline, followed by a mathematically verified Transformer implementation.**