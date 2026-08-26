# AtlasLLM — Agent Engineering Constitution

## 1. Purpose

AtlasLLM is an educational and engineering project for implementing, training, evaluating, and serving a small dense decoder-only Transformer language model from the ground up.

The codebase must prioritize:

- correctness
- simplicity
- readability
- modularity
- reproducibility
- testability
- measurable performance
- maintainability
- architectural consistency

The project is not a competition to produce the largest amount of code.

A smaller correct implementation is preferable to a larger sophisticated implementation.

---

# 2. Absolute Rules

These rules are mandatory.

## Rule 1 — Understand Before Editing

Before modifying the repository, the agent must:

1. inspect the repository structure
2. read `AGENTS.md`
3. read `CONTEXT.md`
4. inspect the relevant existing modules
5. identify existing abstractions
6. determine whether the requested functionality already exists

Never immediately start writing code.

---

## Rule 2 — Never Duplicate Functionality

Before implementing a function, class, utility, helper, loader, validator, or configuration mechanism, search the repository for an existing implementation.

If equivalent functionality already exists:

- reuse it
- extend it
- refactor it if necessary

Do not create:

```text
foo.py
foo_utils.py
foo_helpers.py
foo_manager.py
foo_service.py
foo_handler.py
```

when one coherent module is sufficient.

One responsibility should have one authoritative implementation.

---

# 3. No Duplicate Code

The repository must have a strict DRY policy.

Do not duplicate:

- tensor-shape logic
- device selection
- configuration loading
- logging
- checkpoint loading
- checkpoint saving
- tokenization
- dataset loading
- model construction
- sampling logic
- evaluation logic
- validation logic
- seed initialization
- metrics calculation

If the same logic appears twice, determine whether it belongs in a shared abstraction.

Do not blindly abstract every repeated line.

Abstraction is justified when it improves structure rather than merely reducing line count.

---

# 4. No Overengineering

The simplest correct solution wins.

If a problem requires:

```text
10 lines
```

do not produce:

```text
1,000 lines
```

to make the architecture look sophisticated.

Do not introduce:

- unnecessary design patterns
- unnecessary factories
- unnecessary interfaces
- unnecessary dependency injection
- unnecessary microservices
- unnecessary abstraction layers
- unnecessary frameworks
- unnecessary configuration systems

AtlasLLM is a research/engineering project, not enterprise software cosplay.

---

# 5. No 10,000-Line Solutions

A component that should logically be small must remain small.

Examples:

```text
sampling.py
```

should not become a 2,000-line framework.

```text
attention.py
```

should not become a 1,500-line abstraction hierarchy.

```text
config.py
```

should not become a configuration platform.

Complexity must come from the actual problem, not from the implementation.

If a file becomes unusually large, the agent must determine whether:

- responsibilities are mixed
- code is duplicated
- abstractions are misplaced
- functionality should be separated

Do not split files merely to make line counts smaller.

---

# 6. One Source of Truth

Every important concept must have one canonical source.

Examples:

### Model configuration

One configuration system.

### Device selection

One device utility.

### Tokenizer

One tokenizer interface.

### Dataset

One authoritative dataset abstraction.

### Model construction

One model factory/build path if a factory is actually needed.

### Sampling

One sampling implementation.

### Checkpoints

One checkpoint manager.

### Metrics

One metrics implementation.

Do not maintain competing implementations of the same concept.

---

# 7. Architecture Before Code

The agent must preserve the architecture.

Current conceptual structure:

```text
data
  ↓
tokenizer
  ↓
dataset
  ↓
model
  ↓
training
  ↓
evaluation
  ↓
inference
  ↓
safety
  ↓
harness
```

Dependencies should generally flow downward through clear interfaces.

Do not allow arbitrary cross-imports.

For example:

```text
model/
```

must not depend on:

```text
evaluation/
```

or:

```text
safety/
```

The model must remain independently usable.

---

# 8. Model Purity

The Transformer implementation must remain independent of:

- CLI code
- training loops
- dataset code
- safety policies
- evaluation harnesses
- user interfaces
- logging dashboards

The model receives tensors and configuration.

It produces tensors.

This separation is mandatory.

---

# 9. Training Purity

Training code should orchestrate:

```text
data
→ model
→ loss
→ optimizer
→ scheduler
→ checkpoint
→ metrics
```

Training code must not contain:

- model architecture definitions
- tokenizer internals
- safety policy
- UI logic

---

# 10. Inference Purity

Inference should operate independently from training.

Inference must support loading a trained checkpoint without importing the training pipeline unnecessarily.

The inference engine should contain:

- generation
- sampling
- KV cache
- streaming

It should not contain training-specific logic.

---

# 11. Safety Separation

Safety must remain external to the base model.

Architecture:

```text
Input
  ↓
Input Guardrail
  ↓
Model
  ↓
Output Guardrail
  ↓
Response
```

Never insert application-specific safety logic directly into Transformer layers unless the experiment explicitly concerns model-level safety training.

The base model must remain reusable.

---

# 12. Configuration Rules

Do not hard-code experiment parameters throughout the repository.

Bad:

```python
layers = 6
```

inside multiple modules.

Preferred:

```text
configuration
      ↓
model
```

Configuration must control:

- vocabulary size
- context length
- hidden dimension
- number of layers
- number of heads
- feed-forward dimension
- dropout
- learning rate
- batch size
- training steps
- optimizer
- scheduler
- precision
- dataset paths
- checkpoint paths
- random seed

Do not create configuration fields until the system actually needs them.

---

# 13. Type Safety

Use Python type hints for public functions, classes, and important internal interfaces.

Prefer:

```python
def load_checkpoint(path: Path) -> Checkpoint:
```

over undocumented dynamically typed interfaces.

Do not add meaningless types simply to increase annotation density.

---

# 14. Tensor Shape Documentation

Tensor shapes must be explicit at architectural boundaries.

Canonical notation:

```text
B = batch size
T = sequence length
D = model dimension
H = attention heads
V = vocabulary size
```

Examples:

```text
Input IDs:
[B, T]

Embeddings:
[B, T, D]

Attention:
[B, H, T, head_dim]

Logits:
[B, T, V]
```

Shape assumptions must be documented and tested.

---

# 15. Mathematical Transparency

AtlasLLM is an educational project.

Do not hide core mathematics behind unnecessary libraries.

The following must remain understandable in the repository:

```text
Q = XWq
K = XWk
V = XWv

Attention(Q,K,V)
=
softmax(QKᵀ / √d_k)V
```

The causal mask must be explicit.

The loss must be understandable.

The Transformer block must be understandable.

Using PyTorch tensor operations is encouraged.

Replacing the architecture with a black-box Transformer library is not acceptable for the core implementation.

---

# 16. Dependency Rules

Every dependency must have a reason.

Before adding a package:

1. determine whether Python/PyTorch already provides the required functionality
2. determine whether an existing project dependency already provides it
3. determine whether the package materially improves the implementation

Do not add packages for trivial functionality.

Avoid dependency accumulation.

---

# 17. External Library Boundary

External libraries may be used for:

- tokenization
- dataset storage
- experiment tracking
- visualization
- testing
- hardware utilities

The core Transformer architecture should remain implemented directly in PyTorch.

---

# 18. Error Handling

Errors must fail clearly.

Bad:

```python
except Exception:
    pass
```

Never silently swallow exceptions.

Do not use broad exception handling unless there is a specific recovery strategy.

Errors should identify:

- what failed
- where it failed
- relevant configuration
- actionable cause when possible

---

# 19. Logging

Use structured, useful logging.

Do not spam the terminal with unnecessary messages.

Training logs should expose:

```text
step
loss
validation loss
learning rate
tokens/sec
GPU memory
gradient norm
```

Debug logs should be available without permanently polluting normal output.

---

# 20. Comments

Comments must explain:

- why something exists
- mathematical reasoning
- non-obvious implementation decisions
- hardware-specific workarounds
- important constraints

Do not write comments that merely translate code into English.

Bad:

```python
# Increment i
i += 1
```

Good:

```python
# Shift targets by one token so each position predicts the next token.
```

---

# 21. Documentation

Every major subsystem must have documentation explaining:

```text
purpose
inputs
outputs
dependencies
important design decisions
limitations
```

Documentation must remain synchronized with code.

---

# 22. Testing Before Expansion

Do not add major features to an unverified foundation.

Required order:

```text
implement
↓
test
↓
verify
↓
integrate
↓
benchmark
↓
expand
```

Not:

```text
implement everything
↓
hope it works
```

---

# 23. Mandatory Transformer Tests

The following must be tested:

- output tensor shapes
- causal masking
- attention dimensions
- gradient flow
- Transformer block forward pass
- full-model forward pass
- loss calculation
- checkpoint reload

The causal mask test is mandatory.

---

# 24. Tiny Overfit Test

Before real pretraining, the model must successfully overfit a tiny dataset.

This validates the complete path:

```text
dataset
→ tokenizer
→ model
→ loss
→ optimizer
→ backward pass
```

If this fails, do not proceed to large-scale training.

---

# 25. Reproducibility

Every experiment must record:

```text
random seed
model configuration
dataset version
tokenizer version
software versions
hardware
training parameters
```

Results without reproducibility metadata are incomplete.

---

# 26. Git Rules

Use small, meaningful commits.

Examples:

```text
feat: implement causal self-attention
feat: add tokenizer training pipeline
test: verify causal attention masking
fix: correct shifted language-model targets
perf: add KV cache to generation
docs: document training pipeline
```

Do not create commits such as:

```text
stuff
changes
final
final2
working
new
```

---

# 27. Git Safety

Never:

- overwrite unrelated user changes
- reset the repository without authorization
- delete branches
- rewrite history unnecessarily
- commit secrets
- commit datasets that should remain external
- commit large model checkpoints unless explicitly intended

---

# 28. Secrets

Never hard-code:

- API keys
- tokens
- passwords
- credentials

Use environment variables or local configuration excluded from Git.

---

# 29. Data Integrity

Never silently modify the source dataset.

Raw data:

```text
data/raw/
```

must remain immutable.

Processed data:

```text
data/processed/
```

must be reproducible from the raw data and preprocessing configuration.

---

# 30. Performance Rules

Do not optimize based on assumptions.

Measure first.

Optimization sequence:

```text
correctness
↓
profiling
↓
identify bottleneck
↓
optimize bottleneck
↓
benchmark
↓
verify correctness
```

Never sacrifice correctness for a hypothetical speed improvement.

---

# 31. GTX 1070 Constraint

The agent must account for the 8 GB VRAM limit.

Before increasing:

- batch size
- context length
- model width
- number of layers
- precision requirements

estimate memory implications.

Do not blindly configure a model that cannot fit.

---

# 32. CPU Fallback

AtlasLLM must support:

```text
CPU
CUDA
```

through a common device abstraction.

Code must not assume CUDA exists.

---

# 33. No Premature Optimization

Do not introduce:

- custom CUDA kernels
- distributed training
- FlashAttention dependencies
- quantization frameworks
- compilation systems
- complex serving frameworks

before the baseline model works.

Optimization is a later phase.

---

# 34. No Black-Box Replacement

The agent must not replace AtlasLLM's custom implementation with:

```text
Hugging Face GPT implementation
```

or another pretrained Transformer implementation.

External frameworks may assist with surrounding infrastructure, but the educational core must remain AtlasLLM's own implementation.

---

# 35. Refactoring Rules

Refactor when:

- code is duplicated
- responsibilities are mixed
- interfaces are unclear
- tests are difficult to write
- a module has become unnecessarily complex

Do not refactor merely because another coding style looks different.

---

# 36. Minimal Change Principle

When fixing a bug:

```text
change the smallest amount of code necessary
```

Do not rewrite unrelated modules.

A bug in sampling should not result in a rewrite of the model architecture.

---

# 37. No Hidden Behavior

Functions should do what their names imply.

Avoid functions that secretly:

```text
modify global state
download data
change configuration
initialize CUDA
write files
```

unless their purpose explicitly requires it.

---

# 38. No Global Mutable State

Avoid global mutable objects for:

- model
- tokenizer
- configuration
- dataset
- device
- training state

Pass dependencies explicitly.

---

# 39. Interface Stability

Once a subsystem interface is established, changes must be deliberate.

For example:

```python
tokenizer.encode(text)
tokenizer.decode(tokens)
```

should remain stable even if tokenizer internals change.

---

# 40. Agent Planning Requirement

Before implementing any non-trivial feature, the agent must produce an internal implementation plan containing:

```text
existing code affected
new code required
files to modify
tests required
possible regressions
```

The plan must remain proportional to the task.

A five-line change does not require a twenty-file architecture proposal.

---

# 41. Agent Inspection Requirement

Before creating a new file, verify that an existing file cannot logically contain the functionality.

Before creating a new class, verify that an existing class cannot reasonably own the responsibility.

Before creating a new utility, search for equivalent utilities.

---

# 42. No Duplicate Classes

Never create:

```text
DatasetLoader
TextDatasetLoader
LanguageDatasetLoader
TrainingDatasetLoader
```

when one coherent abstraction is enough.

Names must reflect genuine conceptual differences.

---

# 43. No Duplicate Configuration

Do not define the same parameter in:

```text
YAML
Python defaults
CLI defaults
environment variables
```

without a clear precedence system.

Configuration precedence must be documented.

---

# 44. CLI Rules

Command-line interfaces should remain thin.

CLI:

```text
parse arguments
load configuration
invoke application logic
```

CLI code should not contain model logic.

---

# 45. Notebook Rules

Notebooks are for:

- exploration
- visualization
- experiments
- analysis

Production logic must remain in Python modules.

Do not develop the entire model inside a notebook.

---

# 46. Generated Code Rules

AI-generated code is not automatically trusted.

Every generated implementation must be:

```text
read
understood
tested
integrated
```

Do not merge code merely because it executes.

---

# 47. Hallucination Prevention for Coding Agents

If the agent does not know:

- an API
- a library behavior
- a PyTorch detail
- a hardware limitation
- a dataset property

it must verify the fact from authoritative documentation or inspect the installed environment.

Never invent APIs.

---

# 48. Research Claims

Do not claim:

```text
faster
more efficient
better
state-of-the-art
memory efficient
more accurate
```

without measurements or evidence.

---

# 49. Experiment Integrity

Never modify experiment results manually.

Metrics must be generated by code.

Do not cherry-pick results without recording failed experiments when they materially affect conclusions.

---

# 50. Definition of Clean Code

AtlasLLM code is considered clean when:

- each module has a clear responsibility
- names communicate intent
- duplicate logic is absent
- abstractions are justified
- tests cover critical behavior
- tensor shapes are understandable
- configuration is centralized
- dependencies are controlled
- errors are visible
- code is shorter where simplicity permits
- complexity exists only where the problem requires it

---

# 51. Final Agent Principle

The agent must behave as an engineer working inside an existing system, not as a code generator filling an empty directory.

Before adding code:

```text
UNDERSTAND
```

Before duplicating:

```text
SEARCH
```

Before abstracting:

```text
JUSTIFY
```

Before optimizing:

```text
MEASURE
```

Before declaring success:

```text
TEST
```

Before changing architecture:

```text
DOCUMENT
```

The objective is not maximum code.

The objective is the smallest coherent codebase that correctly implements the required system.