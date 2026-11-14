# MOIRA

MOIRA is an experimental sequence-model research project. The executable reference lives in `research/` and supports deterministic FP32 training and evaluation on CPU, bounded CUDA capacity probes, and master-retained INT2/INT4/INT8 quantization-aware training. The repository also contains the M1 data and capability foundation: a manifest-first source catalog, bounded record adapters, a capability loom, and an exact 33,032,562-parameter training surface.

The repository does not contain a released trained checkpoint or a 3B model. It does not establish AGI, recursive self-improvement, benchmark gains, production packed-integer kernels, general GPU speedups, or production deployment. Each of those claims requires a later implementation and measured evidence.

The current normative design is
[`MOIRA Offline Lifelong-Cognition Architecture v1`](docs/architecture/moira-lifelong-v1.md).
It separates executable proof foundations from the pending daemon, continual
training, distributed execution, and optimized-kernel milestones.

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
- deterministic lease-scoped agent context assembly with typed source, trust,
  classification, provenance, expiry, budget, and digest receipts;
- bounded deterministic candidate search with immutable lineage, failure-evidence,
  score-plus-novelty selection, and non-promoting receipts;
- a permit-gated candidate-work boundary that replays search evidence, confines
  accepted outputs to staging, enforces resource budgets, and hashes artifacts;
- a bounded PEFT LoRA candidate backend with content-addressed train and
  selection data, adapter-only safetensors output, and immutable base checks;
- checkpoint manifests, sidecars, compatibility checks, retention, and recovery; and
- a command-line interface for preflight checks, bounded CPU/CUDA research runs, checkpoint verification, evaluation, and deployment-profile selection.

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

Every turn now passes through `moira-context-v1`. The runtime converts the task
and prior observations into labelled fragments, admits only public or
active-lease-scoped material below the selected classification boundary,
rejects expired or ambiguous inputs, deduplicates only exact label-aware
evidence, and applies a conservative UTF-8 byte-token content budget. Required
task context fails closed; optional material that does not fit is recorded as
dropped rather than silently truncated. The receipt digest binds the selected
order, source, trust, classification, scope, assembly time, provenance, and
content digests. External priority is capped below runtime observations, and
the newest operational observation is required on its next turn.
External memory and retrieval remain untrusted until a verified memory store
and admission path exist.
See [`docs/guides/agent-context.md`](docs/guides/agent-context.md).

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
receipt layer. A third-party agent host may implement that interface, but none is
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
  --context-window-tokens 32768 `
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

## Root-relative capacity and standard precision

MOIRA plans threefold growth from an artifact's exact lineage root. The M1
research lineage is:

```text
33,032,562
-> 99,097,686
-> 297,293,058
-> 891,879,174
-> 2,675,637,522
-> 8,026,912,566
-> 24,080,737,698
-> 72,242,213,094
-> ...
```

An optional exact 3,000,000,000-parameter root forms a separate checkpoint
family. Parameter growth has no fixed architecture tier, but every concrete
child needs hardware, training, evaluation, qualification, and rollback. A
newly copied child is `MATERIALIZED_UNTRAINED`.

Numeric precision is a separate axis:

```text
integer weight storage: INT2 -> INT4 -> INT8
whole-model arithmetic: BF16 or FP16, with FP32 where required
verification: FP32, with FP64 restricted to bounded verifier paths
```

Inspect the policies and create a non-allocating M1 growth plan:

```powershell
python -m moira elastic policy --profile consumer --root 33032562
python -m moira elastic policy --profile enterprise --root 33032562
python -m moira elastic plan `
  --root 33032562 `
  --generations 2 `
  --profile consumer `
  --runtime-precision int4
python -m moira precision plan `
  --from int8 `
  --to fp16 `
  --master-retained
python -m moira precision plan `
  --from fp32 `
  --to fp64 `
  --scope verifier
python -m moira deployment switch enterprise `
  --state research/runs/deployment.json `
  --root-parameters 33032562
python -m moira deployment switch user `
  --state research/runs/deployment.json
```

The proof-scale QAT path keeps floating-point master weights while its forward
view uses INT2, INT4, or INT8:

```powershell
python -m moira elastic qat-smoke `
  --config research/configs/m0-smoke.toml `
  --initial-bits int2 `
  --promote-to int4 `
  --steps-per-stage 1 `
  --device cuda `
  --checkpoint research/runs/int2-to-int4.pt
```

For the exact M1 local profile, start with one measured INT8 stage before
raising the bounded step count:

```powershell
python -m moira elastic qat-smoke `
  --config research/configs/m1-local-int8.toml `
  --initial-bits int8 `
  --promote-to int8 `
  --steps-per-stage 1 `
  --device cuda `
  --checkpoint research/runs/m1-int8/qat.pt
```

This is an INT8 fake-quantized weight view with FP32 masters and gradients. It
is not a packed INT8 training kernel. The runner permits at most 32 steps per
stage; capacity must be measured on the target machine before increasing it.
Continue that dropout-free INT8 checkpoint without replacing its learned state:

```powershell
python -m moira elastic qat-smoke `
  --config research/configs/m1-local-int8.toml `
  --initial-bits int8 `
  --promote-to int8 `
  --steps-per-stage 32 `
  --device cuda `
  --checkpoint research/runs/m1-int8/qat.pt `
  --resume
```

Resume restores weights, optimizer state, and the persisted deterministic
synthetic-data cursor after verifying the exact training-configuration digest.
It is not advertised as exact replay because the proof checkpoint does not yet
persist every RNG and runtime state.

`user` is a command alias for the canonical `consumer` profile. Profile
switching changes persisted resource policy, not weights or model identity.
Each growth command plans at most 32 generations; a qualified child can become
the root of another command, so the lineage has no fixed final tier.

The planning commands do not allocate target weights or write an exported
artifact. The QAT command trains only the configured proof model. Converting a
lossy low-bit artifact to a wider container cannot reconstruct information
that was discarded; retained masters or new training are required for quality
recovery. FP64 remains a bounded verifier option, not a whole-model mode.

Successful commands write one compact JSON object to standard output. Expected argument, configuration, filesystem, and checkpoint failures write an error to standard error and exit with status 2. Keep the complete run directory for automatic recovery and resume; a checkpoint tensor file by itself is not the whole run state.

Run the research tests with pytest's cache disabled:

```powershell
python -m pytest research/tests -p no:cacheprovider
```

## Repository map

| Path | Purpose |
| --- | --- |
| `research/src/moira/` | Executable PyTorch CPU/CUDA research reference and CLI |
| `research/src/moira/elastic.py` | Exact root-relative growth plans, standard low-bit views, and profile policies |
| `research/src/moira/qat.py` | Master-retained INT2/INT4/INT8 weight QAT and continuation checkpoints |
| `research/src/moira/deployment.py` | Atomic local Consumer/Enterprise profile selection |
| `research/src/moira/context.py` | Deterministic scoped context admission, budgeting, and receipts |
| `research/src/moira/lifelong.py` | Typed consent, interaction, memory, perspective, wake, and cognition contracts |
| `research/src/moira/kernel.py` | Persistent five-module control-kernel state and receipt chain |
| `research/src/moira/core_extensions.py` | Affect, drive, truth-tone, triadic-growth, and architecture-candidate primitives |
| `research/src/moira/capabilities.py` | Capability-port ABI and evidence-backed qualification labels |
| `research/src/moira/candidate_search.py` | Bounded candidate archive and deterministic research selection |
| `research/src/moira/candidate_worker.py` | One-shot native-agent boundary for staged candidate work |
| `research/src/moira/lora_backend.py` | Bounded PEFT LoRA candidate trainer and selection evaluator |
| `research/src/moira/agent_data.py` | Agent trajectory provenance, consent, license, split, and content-manifest gate |
| `research/tests/` | Model, mask, state, checkpoint, resume, and CLI tests |
| `research/configs/m0-smoke.toml` | Checked-in 24,478-parameter smoke configuration |
| `research/configs/m1-local-int8.toml` | Exact 33,032,562-parameter local INT8-QAT profile |
| `research/configs/m1-sources.toml` | M1 dataset, model, and evaluation source catalog |
| `docs/architecture/moira-lifelong-v1.md` | Current normative lifelong-cognition architecture and milestone gates |
| `docs/architecture/moira-v0.md` | Superseded v0 design marker |
| `docs/guides/kaggle-m0.md` | Cell-by-cell Kaggle workflow and artifact rules |
| `docs/guides/free-compute.md` | Current free compute routes and their limits |
| `docs/guides/m1-data.md` | M1 source, normalization, capability, and training workflow |
| `docs/guides/candidate-search.md` | Candidate-search contract, selection algorithm, and promotion boundary |
| `docs/guides/candidate-worker.md` | Permit, staging, budget, artifact, and isolation contract for candidate work |
| `docs/guides/lora-candidates.md` | LoRA dataset, training, artifact, and evaluation contract |
| `kaggle/` | Private Kaggle M2 FP32 train-save-resume script |
| `src/main.rs` and `Cargo.toml` | Unified 100% Rust CLI (`infer`, `train [--push-hf]`, `colab-agent`, `colab-ctl`, `push-hf`) |
| `src/decision.rs` | 100% Rust 13,633-parameter `MoiraDecisionModel` (3-6-9 Clotho/Lachesis/Atropos) trainer & native `.safetensors` writer |
| `src/pnn.rs` | 100% Rust 3,201-parameter `TropicalPolicyStudent` Max-Plus $(\max, +)$ compiler & binary/JSON `.pnn` serializer |
| `src/colab.rs` | 100% Rust Antigravity ↔ Google Colab Remote Control Bridge (`colab-agent`, `colab-ctl`) & HF NDJSON Commit client |
| `src/engine.rs`, `src/memory.rs`, `src/data_sifter.rs` | Native Rust 3-6-9 Clotho/Lachesis/Atropos engine, WAL memory supervisor, and structural token sifter |

## 100% Rust Engine, Trainer & Colab Remote Control (`moira v0.2.0`)

```powershell
# Run unit tests & build release binary
cargo test
cargo build --release

# Train 13,633-param MOIRA 3-6-9 Teacher + 3,201-param Tropical .pnn Student and upload to Hugging Face Hub
cargo run --release -- train --push-hf

# Run on-device 3-6-9 inference + zero-multiplication Tropical .pnn policy check
cargo run --release -- infer

# Control Google Colab GPU remotely from Antigravity
cargo run --release -- colab-ctl status
cargo run --release -- colab-ctl exec "cargo run --release -- train --push-hf" --wait
```

## Research boundary

The current architecture uses exact root-relative threefold growth. Its shipped
research root is 33,032,562 parameters; a future exact
3,000,000,000-parameter root would be a separate lineage. The current M1 work
covers source contracts, capability boundaries, a fixed 33M inspection
surface, equal-parameter ablations, bounded text training, and checkpointed
capability-prefix training. It does not establish a long stable M1 curve, a
trained larger lineage, lifelong autonomy, or any AGI claim.

For a hosted notebook run, follow the [Kaggle M0 guide](docs/guides/kaggle-m0.md).
For model, control, learning, growth, and deployment contracts, read the
[MOIRA Offline Lifelong-Cognition Architecture v1](docs/architecture/moira-lifelong-v1.md).

## License

MOIRA is available under the [GNU General Public License v3.0 (GPL-3.0-or-later)](LICENSE).


## Colab GPU Trainer & Hugging Face Hub (`anisayakmitra-in/FATE-O-SISTERS`)

- **Hugging Face Model Repository:** [`https://huggingface.co/anisayakmitra-in/FATE-O-SISTERS`](https://huggingface.co/anisayakmitra-in/FATE-O-SISTERS)
- **Colab Trainer Script:** `scripts/train_moira_colab.py`
- **Colab Notebook:** `notebooks/train_moira_colab.ipynb`

`scripts/train_moira_colab.py` supports two scales (`--scale 3b` and `--scale m2`):

1. **Stage 1 (`--scale 3b` default):** Trains the **3.04B Joint Decision + Text Architecture** (`44 × (TSKBlock + SwiGLUFFN)` + `Mean+Max Knot` landmarks + `Pulse` 30D homeostatic state + `Moira369PulseHead`) in `bfloat16` with `AdamW8bit` and exports 4GB-sharded `safetensors` (`moira_3b/`).
2. **Stage 2:** Trains the standalone 13,633-parameter **3-6-9 `MoiraDecisionModel`**, distills the zero-multiplication 3,201-parameter **`TropicalPolicyStudent` (`model_decision.pnn`)** for Rust (`AGENT-PANDORA`), and pushes the bundle to `anisayakmitra-in/FATE-O-SISTERS`.
