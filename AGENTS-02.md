AGENTS.md — AtlasLLM Engineering Constitution

This file defines the rules for every AI coding agent working on AtlasLLM.

---

1. Read Before Doing Anything

Before modifying the repository, read:

AGENTS.md
CONTEXT.md
DATA_PIPELINE.md

Then inspect the existing repository.

Do not immediately start generating files.

Understand what already exists first.

---

1. Core Principle

You are an engineer working inside an existing project.

You are NOT a code generator.

Before creating anything:

1. inspect the repository
2. search for existing functionality
3. identify the source of truth
4. determine what actually needs to change
5. implement the smallest coherent solution
6. test it
7. inspect the result

---

1. No Duplicate Functionality

Never create a second implementation of something that already exists.

Before adding:

- utility
- class
- configuration
- dataset loader
- tokenizer wrapper
- checkpoint manager
- logger
- metric function
- training utility
- validation function

search the repository first.

One responsibility should have one authoritative implementation.

---

1. No Overengineering

Do not introduce architecture merely because it looks sophisticated.

Avoid:

- unnecessary factories
- unnecessary registries
- excessive abstractions
- unnecessary interfaces
- wrapper classes around trivial functions
- duplicate configuration systems
- unnecessary dependencies
- premature distributed infrastructure
- unnecessary microservices
- excessive framework usage

Do not create 1,000 lines of abstraction to solve a 20-line problem.

---

1. Core Model Must Remain Clean

The Transformer model must not depend on:

- dataset downloading
- Hugging Face dataset acquisition
- training orchestration
- evaluation harness
- safety guardrails
- UI
- experiment reporting
- application logic

The core model should primarily know about:

tensors
model configuration
forward pass
attention
Transformer blocks
loss-related outputs

---

1. Do Not Replace the Educational Core

Do not replace AtlasLLM's implementation with:

AutoModelForCausalLM
GPT2LMHeadModel
LlamaForCausalLM
other black-box pretrained Transformer implementations

External libraries may be used for infrastructure, datasets, tokenizers, serialization, logging, etc., when justified.

The actual AtlasLLM Transformer must remain understandable and implemented directly in PyTorch.

---

1. Configuration Has One Source of Truth

Do not duplicate model parameters across source files.

Bad:

# model.py

d_model = 256

and:

# trainer.py

d_model = 256

Good:

configuration
    ↓
model
trainer
evaluation
checkpoint metadata

All components receive configuration from the authoritative configuration system.

---

1. Dataset Acquisition Must Be Automated

The user will NOT manually gather the AtlasLLM corpus.

The agent must build scripts that can:

download
cache
validate
process
filter
deduplicate
mix
split
tokenize
pack
shard
validate

The pipeline must work from configuration.

Do not instruct the user to manually download hundreds of files.

---

1. Use Platforms Programmatically

Hugging Face and other appropriate dataset platforms should be accessed through code/scripts.

The implementation should support reproducible dataset acquisition.

Prefer official dataset APIs/libraries where appropriate.

Do not scrape random websites manually when an official machine-readable dataset exists.

---

 1. Dataset Licenses Matter

Every dataset source must record:

source
dataset identifier
version
license
retrieval date

Do not silently mix datasets with unknown licensing.

Do not claim a dataset is legally unrestricted without verifying its license.

---

 1. Raw Data Is Immutable

Never modify raw downloaded data in place.

Use:

raw
→ processed
→ tokenized
→ training shards

Every transformation must produce a derived artifact.

---

 1. Large Data Must Be Streamed or Chunked

Do not do:

dataset = load_everything_into_ram()

for large corpora.

Prefer:

- streaming
- chunked processing
- incremental writing
- sharded datasets
- bounded buffers

The machine has 32GB RAM.

---

 1. GPU Awareness

Primary GPU:

GTX 1070 8GB

Never assume:

- 16GB VRAM
- 24GB VRAM
- modern Tensor Core behavior
- large batch sizes

Training code must expose/configure:

batch size
gradient accumulation
context length
precision
checkpointing

---

 1. Correctness Before Performance

Order of priorities:

Correctness
↓
Tests
↓
Reproducibility
↓
Measurement
↓
Optimization

Do not introduce optimization before measuring the baseline.

---

 1. No Premature CUDA Engineering

Do not create:

- custom CUDA kernels
- custom fused kernels
- distributed training
- FlashAttention replacements
- complicated compilation infrastructure

unless a measured bottleneck justifies it.

The initial project should be simple PyTorch.

---

 1. Tensor Shapes Must Be Understandable

Important tensors should have documented shapes.

Example:

X:
[B, T, D]

Attention:

Q:
[B, H, T, Dh]

Attention scores:

[B, H, T, T]

Logits:

[B, T, V]

Do not hide important shape transformations behind unnecessary abstractions.

---

 1. Causal Mask Is Mandatory

The model must never allow information leakage from future tokens.

A dedicated test must prove:

position i cannot attend to position j
when j > i

---

 1. Tiny Overfit Test Is Mandatory

Before real pretraining:

tiny dataset
+
tiny model

must overfit.

This validates:

- forward
- loss
- gradients
- optimizer
- masking
- data pipeline
- labels
- model connectivity

If this fails, stop.

Do not compensate by increasing model size or training duration.

---

 1. Checkpointing

Checkpoints should preserve enough information to resume an experiment.

At minimum:

model state
optimizer state
scheduler state
training step
epoch
configuration
metrics

Where practical, preserve reproducibility state such as RNG state.

Checkpoint loading must have tests.

---

 1. Training Must Be Resumable

A training interruption should not require restarting from zero.

The trainer must support:

start new run
resume checkpoint
evaluate checkpoint

without duplicating training logic.

---

 1. Logging

Logs should answer:

What is training?
Which dataset?
Which configuration?
Which step?
What loss?
What learning rate?
How much GPU memory?
How fast?

Do not spam logs with meaningless information.

---

 1. Errors Must Be Visible

Never hide failures using:

except Exception:
    pass

Bad data should either:

- fail clearly, or
- be explicitly counted/skipped according to configured policy.

The pipeline must report what was discarded and why.

---

 1. Reproducibility

Meaningful experiments must record:

seed
dataset version
dataset manifest
tokenizer
model configuration
optimizer
scheduler
learning rate
batch size
gradient accumulation
context length
precision
hardware
software versions

---

 1. Tests Before Expansion

Before adding another major component:

implement
→ test
→ validate
→ continue

Do not build ten untested subsystems and debug them simultaneously.

---

 1. Documentation Must Match Reality

If code changes behavior, update the relevant documentation.

Never allow:

README says X
code actually does Y

Documentation must describe the actual system.

---

 1. Minimal Dependency Policy

Before adding a dependency, ask:

1. Is it actually necessary?
2. Does PyTorch already solve this?
3. Does an existing project dependency solve it?
4. Does it simplify the system enough to justify itself?

Do not add libraries because they are popular.

---

 1. External Libraries

Acceptable infrastructure dependencies may include libraries for:

- PyTorch
- datasets
- tokenization
- numerical processing
- experiment logging
- testing
- linting
- configuration

But the agent must not outsource the educational core of AtlasLLM.

---

 1. Hugging Face Usage

Hugging Face may be used for:

- dataset acquisition
- dataset streaming
- dataset metadata
- tokenizer infrastructure where selected
- publishing AtlasLLM artifacts later
- evaluation resources

It must not replace the AtlasLLM Transformer implementation.

---

 1. Git Rules

Use small, meaningful commits.

Examples:

initialize project configuration
implement tokenizer pipeline
add dataset provenance tracking
implement causal attention
add transformer tests
implement training loop
add checkpoint resume

Do not create enormous meaningless commits.

Never perform destructive git operations without explicit instruction.

Do not commit:

API keys
tokens
passwords
credentials
large raw datasets
private files

---

 1. AI-Generated Code

Every generated implementation must be:

read
understood
tested

Do not blindly accept generated code.

If an API is uncertain, verify it against authoritative documentation or the installed environment.

---

 1. No Fake Results

Never fabricate:

- benchmark scores
- training metrics
- dataset statistics
- throughput
- model performance
- evaluation results

If something has not been measured, say so.

---

 1. Experiment Discipline

Every experiment should have:

experiment ID
configuration
dataset version
checkpoint
metrics
notes

When comparing models, change one meaningful variable where possible.

---

 1. Evaluation Data Is Sacred

Evaluation data must remain outside training.

Never:

download benchmark
→ accidentally mix into pretraining

Evaluation datasets should have explicit provenance and role metadata.

---

 1. Instruction and Reasoning Are Separate

Do not merge instruction tuning and reasoning training into the base pretraining dataset.

Lifecycle:

AtlasLLM Base
    ↓
Instruction SFT
    ↓
AtlasLLM-Instruct
    ↓
Reasoning Training
    ↓
AtlasLLM-Reasoning

---

 1. Safety Is External

Do not hard-code application safety behavior into the Transformer.

Use:

Input Guardrail
→ Model
→ Output Guardrail

This allows the base model to remain an experimental ML artifact.

---

 1. Before Any Non-Trivial Change

The agent must internally determine:

What existing code is affected?
What new code is actually necessary?
Which configuration changes?
Which tests are required?
What could regress?

Then implement the smallest coherent change.

---

 1. Final Engineering Rule

The target is:

«The smallest coherent codebase that correctly implements the complete AtlasLLM lifecycle.»

Not the largest codebase.

Not the most sophisticated architecture.

Not the most abstractions.

Not the most dependencies.

The project succeeds when the system is:

understandable
reproducible
testable
measurable
and actually works.
