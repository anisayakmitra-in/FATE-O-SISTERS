# MOIRA

MOIRA is an experimental sequence-model research project. The executable Phase 0 reference lives in `research/` and supports deterministic FP32 training and evaluation on CPU. The repository also contains the M1 data and capability foundation: a manifest-first source catalog, bounded record adapters, a capability loom, and a 33,032,562-parameter inspection surface.

The repository does not contain a trained checkpoint or a 3B model. It does not establish AGI, recursive self-improvement, benchmark gains, speedups, quantization, GPU execution, or production deployment. Each of those claims requires a later implementation and measured evidence.

## What works now

The Python reference contains:

- one MOIRA model with shared weights across Clotho, Lachesis, and Atropos attention modes;
- bidirectional Clotho visibility, sealed-source plus causal-target Lachesis visibility, and causal Atropos visibility;
- a Thread local-attention path with RoPE, a bounded recurrent Spool state, and Knot landmark attention;
- standalone typed capability-capsule validation for task, scope, metadata, tensor shape, slot types, and uncertainty, plus a canonical SHA-256 digest;
- Pulse state with 12 fast affect components and 6 slow affect components;
- an exact trainable-parameter ledger checked against the model;
- deterministic synthetic data, training, evaluation, and resume tests;
- a manifest-first M1 catalog for text, code, cybersecurity, vision-text,
  audio, robotics, graph, protein, time-series, diffusion, evaluation, and
  teacher references;
- bounded JSONL normalization with source revisions, licenses, split assignment, and record hashes;
- pure-PyTorch vision and feature capsule encoders for image, audio,
  cybersecurity telemetry, time-series, structured, graph, and robotics inputs;
- an M1 meta-device parameter check, equal-parameter component ablations, and a bounded in-memory JSONL training path;
- checkpointed capability-prefix training, resume, and no-update held-out evaluation;
- a native model-level agent layer with per-task role leases, registered tools,
  turn and effect budgets, request-bound permits, and receipts;
- checkpoint manifests, sidecars, compatibility checks, retention, and recovery; and
- a CPU command-line interface for preflight checks, training, checkpoint verification, evaluation, and deployment-profile descriptions.

Clotho, Lachesis, and Atropos are internal execution orientations. They do not have separate identities, conversations, permissions, memories, or parameter copies.

Pulse is bounded model state. Its affect components do not show that the model experiences emotions.

## Local Phase 0 run

Use Python 3.11 or newer with PyTorch 2.4 or newer and NumPy 1.26 or newer. Pytest 8 or newer is needed for the test suite.

From the repository root in PowerShell, expose the source package without building a wheel:

```powershell
$env:PYTHONPATH = (Resolve-Path "research/src").Path
```

Check the configuration, resource limits, parameter ledger, and run directory before training:

```powershell
python -m moira doctor --config research/configs/m0-smoke.toml --run-dir research/runs/m0-local --device cpu
```

Start a new lineage and stop after the first checkpoint:

```powershell
python -m moira train --config research/configs/m0-smoke.toml --run-dir research/runs/m0-local --device cpu --stop-at-step 2 --resume none
```

Verify and evaluate the compatible checkpoint selected from the run directory:

```powershell
python -m moira checkpoint verify --config research/configs/m0-smoke.toml --run-dir research/runs/m0-local
python -m moira evaluate --config research/configs/m0-smoke.toml --run-dir research/runs/m0-local --device cpu --batches 2
```

Resume that lineage to step 4:

```powershell
python -m moira train --config research/configs/m0-smoke.toml --run-dir research/runs/m0-local --device cpu --stop-at-step 4 --resume auto
```

`--stop-at-step` is an absolute global-step target. `--resume` is required and accepts `none`, `auto`, or an explicit checkpoint path.

Inspect the two planned deployment profiles:

```powershell
python -m moira deployment describe consumer
python -m moira deployment describe enterprise
```

The profile commands describe planned defaults. They do not switch the model, allocate hardware, or grant permissions in Phase 0.

## Inspect and exercise M1

Validate the source catalog and count the M1 model without downloading a dataset
or a teacher checkpoint:

```powershell
python -m moira dataset validate --catalog research/configs/m1-sources.toml
python -m moira m1 inspect
```

For a caller-owned text or code JSONL slice, run one bounded in-memory step:

```powershell
python -m moira m1 train-jsonl `
  --catalog research/configs/m1-sources.toml `
  --source zai-longcite-45k `
  --path C:\data\longcite-slice.jsonl `
  --steps 1 `
  --batch-size 1 `
  --device cpu
```

This text/code path can write and resume a bounded checkpoint; it remains a
small data-to-loss systems run. Long stable curves and precision comparisons
remain M1 exit gates. The capability path below adds bounded checkpoint resume
and held-out evaluation for vision and feature observations.

M2 uses the fixed 153,349,122-parameter GPU surface. Its checkpoint contains
FP32 model weights, FP32 AdamW state, RNG state, loss history, source manifest,
and data cursor:

```powershell
python -m moira m2 inspect
python -m moira m2 train-jsonl `
  --catalog research/configs/m1-sources.toml `
  --source zai-longcite-45k `
  --path C:\data\longcite-slice.jsonl `
  --steps 1 `
  --batch-size 1 `
  --device cuda `
  --checkpoint C:\runs\moira-m2\text.pt
python -m moira m2 train-jsonl `
  --catalog research/configs/m1-sources.toml `
  --source zai-longcite-45k `
  --path C:\data\longcite-slice.jsonl `
  --steps 2 `
  --batch-size 1 `
  --device cuda `
  --checkpoint C:\runs\moira-m2\text.pt `
  --resume
```

This command proves bounded training and resume. It does not satisfy the M2
long-stream, capsule-alignment, or validation-loss exit conditions by itself.

For a caller-owned capability slice, provide a CPU-readable `torch.save` tensor
of observations alongside the normalized JSONL rows. Vision uses
`[records, 3, height, width]`; audio, time-series, structured, and robotics
use `[records, time, features]`. The loom projects each observation into a
bounded prefix and the M1 checkpoint includes the core model, loom, optimizer,
loss history, cursor, and manifest:

```powershell
python -m moira m1 train-capabilities `
  --catalog research/configs/m1-sources.toml `
  --source chartqa `
  --path C:\data\chartqa-slice.jsonl `
  --observations C:\data\chartqa-images.pt `
  --modality vision `
  --steps 1 `
  --batch-size 1 `
  --checkpoint C:\runs\moira-m1\vision.pt `
  --device cuda
```

Pass `--resume` with the same manifest and checkpoint to continue at the next
sample cursor. Keep a separate, caller-owned JSONL/tensor slice for evaluation
and score it without changing the checkpoint:

```powershell
python -m moira m1 evaluate-capabilities `
  --catalog research/configs/m1-sources.toml `
  --source chartqa `
  --path C:\data\chartqa-eval.jsonl `
  --observations C:\data\chartqa-eval-images.pt `
  --modality vision `
  --checkpoint C:\runs\moira-m1\vision.pt `
  --split eval `
  --batch-size 1 `
  --device cuda
```

The evaluator loads and validates the exact model, loom, source, shape, and
scope manifest, reports token-weighted loss plus per-batch losses, and sets
`weights_updated` to `false`. The `--split` value records the held-out slice
identity; the JSONL and observation files themselves must be held out by the
caller. This remains bounded local training/evaluation; it does not fetch Hub
weights, execute robotics, or claim a trained 3B/AGI system.

Cybersecurity is a typed specialization beside the other capabilities. Advisory
or ATT&CK JSONL supplies the text objective, while a caller-owned finite
`[records, events, features]` tensor supplies log, flow, package, or detection
features:

```powershell
python -m moira m1 train-capabilities `
  --catalog research/configs/m1-sources.toml `
  --source github-advisory-database `
  --path C:\data\ghsa-train.jsonl `
  --observations C:\data\ghsa-features.pt `
  --modality cyber `
  --scope-id defensive-lab `
  --steps 1 `
  --checkpoint C:\runs\moira-cyber\advisories.pt `
  --device cuda
```

The specialization covers secure-code review, vulnerability triage,
remediation, ATT&CK mapping, detection engineering, threat-intelligence
synthesis, and incident analysis. It does not grant network access, credentials,
scanner control, exploit execution, or permission to act on a target.

## Native MOIRA Agent Layer

The whole MOIRA model is agentic. Any task-selected Clotho, Lachesis, or
Atropos orientation can serve as planner, analyst, executor, reviewer, or
evaluator. Agency is shared model infrastructure, not a fourth sister and not
the permanent responsibility of one sister. Inspect the executable contract
with:

```powershell
python -m moira agent inspect
```

The native layer records proposal turns, binds effect requests to their payload
with domain-separated SHA-256 digests, applies a deny-by-default policy, and
executes only registered tools under a bounded lease. Request-bound permits are
single-use. Approval-required actions pause until the operator approves the
exact request. Every attempted execution produces a receipt.

External harnesses are optional I/O hosts. They can restrict MOIRA but cannot
replace its cognition or expand its local authority. The reference action loop
can speak without being prompted again, ask the user a necessary question,
select a registered tool, consume its observation, and finish within a step
budget.

MOIRA now also has a native trainable action objective. It teaches the shared
decoder to emit the same exact `speak`, `ask_user`, `use_tool`, and `finish`
objects consumed by the runtime. Each row keeps its selected sister mode
embedding. A source/causal visibility boundary lets the prompt remain
bidirectional without exposing future action tokens. The objective adds no
separate policy model or action-head parameters: only canonical action bytes
receive loss. A 114,366-parameter local surface proves the data, loss,
exact-resume, held-out evaluation, and checkpoint contracts:

```powershell
python -m moira agent train-actions `
  --manifest research/tests/fixtures/agent-trajectories.manifest.json `
  --checkpoint research/runs/agent-actions.pt `
  --steps 1

python -m moira agent evaluate-actions `
  --manifest research/tests/fixtures/agent-trajectories-held-out.manifest.json `
  --training-manifest research/tests/fixtures/agent-trajectories.manifest.json `
  --checkpoint research/runs/agent-actions.pt `
  --max-records 4
```

Bare action JSONL files cannot enter the CLI. Their exact manifests declare the
split, source class, content digest, record count, license or consent review,
tenant scope, personal-data status, redistribution status, and deletion policy.
The evaluator also refuses identical files, reused record IDs, and repeated
model-visible prompts. It reports teacher-forced loss and token accuracy plus
greedy termination, schema, canonical JSON, exact-action, action-kind, and
tool-selection metrics. The two bundled four-record files are deterministic
schema fixtures, not useful agent training data or a benchmark. See
[`docs/guides/agent-action-training.md`](docs/guides/agent-action-training.md).

For an Ollama download, the weights provide cognition and structured action
selection. A model file cannot directly call the operating system. Real file,
terminal, browser, or API effects therefore use the small MOIRA companion
runtime, which connects the Ollama model to the same policy, approval, tool, and
receipt layer. A third-party harness may host that interface, but none is
required by MOIRA's architecture.

The first standalone command is executable now. It exposes only bounded file
reading and literal text search below one selected workspace:

```powershell
$env:PYTHONPATH = (Resolve-Path "research/src").Path
$moiraModel = "<published-moira-ollama-tag>"
python -m moira agent run-ollama `
  --model $moiraModel `
  --task "Inspect this repository and summarize its architecture." `
  --workspace (Resolve-Path ".") `
  --max-steps 8
```

Ollama must already be installed, running on `127.0.0.1:11434`, and have the
named MOIRA artifact pulled. This repository does not yet publish that trained
artifact. The command refuses remote Ollama URLs, absolute or escaping file
targets, links, and non-registered tools. By default it also refuses writes and
process execution. An explicit developer profile can add atomic writes and
predeclared commands:

```powershell
python -m moira agent run-ollama `
  --model $moiraModel `
  --task "Update the approved file and run its tests." `
  --workspace (Resolve-Path ".") `
  --allow-write `
  --allow-command git `
  --allow-command python
```

Every write and every individual command remains paused until the operator
approves its exact request digest. Commands receive an argv array directly;
there is no shell interpolation. See
[`docs/guides/ollama-agent.md`](docs/guides/ollama-agent.md) for the exact
runtime contract.

## Elastic capacity and custom bitplane precision

MOIRA now has a planning contract for one identity that can grow with
hardware-supported resources:

```text
parameters: 3B -> 9B -> 27B -> 81B -> 243B -> 729B -> ...
logical bits: int3 -> int6 -> int9 -> int27 -> int81 -> int243 -> ...
```

These are separate axes. A `3B` model is not a `3-bit` model, and `int81` is a
logical packed-bit budget rather than a claim that a GPU has native 81-bit
matrix multiplication. Consumer and Enterprise policies select different
logical precision per capability while keeping one MOIRA model identity:

```powershell
python -m moira elastic policy --profile consumer
python -m moira elastic policy --profile enterprise
python -m moira elastic plan `
  --current 3B `
  --target 243B `
  --profile enterprise `
  --runtime-bits int27
python -m moira deployment switch consumer `
  --state research/runs/deployment.json
```

The plan is non-allocating and records every growth step, target storage
estimate, capability precision policy, arithmetic dtype, and the requirement
for training and promotion. The deployment command atomically selects a local
policy; it does not convert weights, allocate a different model, or grant new
capabilities.

The executable proof path trains Linear and Embedding weights through an
`int3` fake-quantized forward view while AdamW updates retained FP32 masters,
then advances exactly one rung to `int6`:

```powershell
python -m moira elastic qat-smoke `
  --config research/configs/m0-smoke.toml `
  --initial-bits int3 `
  --promote-to int6 `
  --steps-per-stage 1 `
  --device cuda `
  --checkpoint research/runs/int3-to-int6.pt
```

This is weight-only quantization-aware training, not native packed INT3 matrix
multiplication. The executable QAT levels stop at `int27`: FP64 has 52 explicit
significand bits, so pretending to simulate `int81` numerics would be false.
`int81` and higher remain logical storage/policy levels. Turning an `int3`-only
artifact into a larger container cannot recover discarded information.
Large-field packed kernels and distributed training remain future work.

Successful commands write one compact JSON object to standard output. Expected argument, configuration, filesystem, and checkpoint failures write an error to standard error and exit with status 2. Keep the complete run directory for automatic recovery and resume; a checkpoint tensor file by itself is not the whole run state.

Run the research tests with pytest's cache disabled:

```powershell
python -m pytest research/tests -p no:cacheprovider
```

## Repository map

| Path | Purpose |
| --- | --- |
| `research/src/moira/` | Executable PyTorch CPU reference and CLI |
| `research/src/moira/elastic.py` | Unbounded capacity plans, custom bitplane views, and profile policies |
| `research/src/moira/qat.py` | Master-retained INT3/INT6/INT9/INT27 weight-QAT and continuation checkpoint |
| `research/src/moira/deployment.py` | Atomic local Consumer/Enterprise profile selection |
| `research/src/moira/agent_data.py` | Agent trajectory provenance, consent, license, split, and content-manifest gate |
| `research/tests/` | Model, mask, state, checkpoint, resume, and CLI tests |
| `research/configs/m0-smoke.toml` | Checked-in 24,478-parameter smoke configuration |
| `research/configs/m1-sources.toml` | M1 dataset, model, and evaluation source catalog |
| `docs/architecture/moira-v0.md` | Architecture, scope boundaries, and milestone gates |
| `docs/guides/kaggle-m0.md` | Cell-by-cell Kaggle workflow and artifact rules |
| `docs/guides/free-compute.md` | Current free compute routes and their limits |
| `docs/guides/m1-data.md` | M1 source, normalization, capability, and training workflow |
| `docs/guides/local-int3-qat.md` | RTX/Kaggle setup, INT3-to-INT6 run, and memory limits |
| `kaggle/` | Private Kaggle M2 FP32 train-save-resume script |
| `src/` and `Cargo.toml` | Initial Rust sketch kept for comparison; its forward path returns zero logits and is not the training reference |

## Research boundary

The architecture document records proposals for an exact 3B root model, governed threefold growth, consumer and enterprise placement, specialist capability fields, adapters, retrieval, quantization, and later GPU work. The current M1 work covers source contracts, capability boundaries, a fixed 33M parameter surface, equal-parameter ablations, bounded text training, and checkpointed capability-prefix training. It does not establish a long stable M1 curve, a 3B lineage, or any AGI claim.

For a hosted notebook run, follow the [Kaggle M0 guide](docs/guides/kaggle-m0.md). For the model contract and milestone criteria, read the [MOIRA v0 architecture](docs/architecture/moira-v0.md).

## License

MOIRA is available under the [MIT License](LICENSE).
