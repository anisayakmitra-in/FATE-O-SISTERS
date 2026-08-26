# MOIRA v0 Architecture Specification

- Status: approved research design for Phase 0 implementation
- Repository: `anisayakmitra-in/FATE-O-SISTERS`
- Pinned parent commit: `9458927c3d2532f1f3e066d9c725297957a08bc7`
- Canonical model name: MOIRA

## 1. Scope and claim boundary

MOIRA v0 defines an experimental sequence model, its multimodal boundary, its learning lifecycle, and its scale-up rules. The first implementation is a small PyTorch reference model. It exists to test the mathematics and software contracts before any 3B training run.

The current repository does not contain a trained model. Its Rust binary is a sketch: the forward pass returns zero logits, the entropy threshold is hard-coded, and the memory worker does not persist trajectories. Phase 0 keeps that code available for comparison while the executable research model is built beside it.

Claims require measurements. Reusing a weight tensor nine times increases parameter applications; it does not create nine times as many physical parameters. Copying a 3B field three times creates 9B logical parameter positions at birth; it does not add learned capability until the children train and diverge. The elastic field therefore treats capacity growth, logical bit precision, and arithmetic dtype as separate manifest axes.

## 2. One model and three cognitive orientations

MOIRA has one public identity, one voice, one owner-scoped memory lineage, one affect state, and one permission context. Clotho, Lachesis, and Atropos are internal execution orientations.

| Orientation | Visibility | Job |
|---|---|---|
| Clotho | Bidirectional inside sealed input | Understand evidence, context, entities, and provenance |
| Lachesis | Bidirectional sealed source plus causal target | Generate hypotheses, plans, forecasts, prose, and code |
| Atropos | Causal public stream plus typed evidence input | Transform, verify, select, and prepare a committed result |

The orientations share weights. They do not own separate memories, affect states, permissions, or user-facing conversations.

The reference API uses:

```python
class CognitiveMode(str, Enum):
    CLOTHO = "clotho"
    LACHESIS = "lachesis"
    ATROPOS = "atropos"
```

Mask behavior is part of the model definition:

- Clotho may attend in both directions within the configured local window.
- Lachesis source tokens may attend within the sealed source. Target tokens may attend to the full sealed source and earlier target tokens. Source tokens cannot attend to the target.
- Atropos token generation is causal. Sealed evidence enters through typed capsule and Knot paths, so future public tokens remain inaccessible.

## 3. Thread, Spool, and Knot

Each field uses one repeated macrocell:

```text
Thread -> Spool -> Knot -> residual output
```

### 3.1 Thread

Thread performs exact attention inside a bounded local window. It handles syntax, recent context, exact copying, local code dependencies, and short mathematical relations.

For position `i`, Thread may attend only to positions permitted by the active orientation and local window. RoPE belongs to this local attention path. Long-term chronology uses Spool decay and Knot metadata instead of a single unbounded rotary index.

### 3.2 Spool

Spool carries recurrent continuity with bounded state. The Phase 0 reference uses a gated delta-memory update:

\[
S_t = \alpha_t S_{t-1} + \beta_t k_t (v_t - k_t^\top S_{t-1})^\top
\]

\[
r_t = q_t^\top S_t
\]

Here `S_t` has shape `[batch, rank, hidden]`; `q_t` and `k_t` have shape `[batch, rank]`; and `v_t` and `r_t` have shape `[batch, hidden]`. Phase 0 tests recurrent execution directly. A later optimized scan must match the reference within a declared numerical tolerance.

### 3.3 Knot

Knot provides a bounded exact route to global evidence. It compresses the current sequence into a small landmark bank, admits typed capability capsules, and lets sequence positions attend to those landmarks.

Knot must retain source type and provenance. Observed evidence, retrieved evidence, hypotheses, tool results, and synthetic artifacts cannot share an unmarked representation.

The Phase 0 reference builds deterministic contiguous landmarks from the current sequence and may accept explicit typed capsule landmarks. Landmark attention, type embeddings, import projection, and export projection are trainable and count toward the model ledger. Learned landmark selection is deferred until the reference causal and recurrence tests pass.

## 4. Pulse affect manifold

MOIRA owns one regulated affect state:

```python
@dataclass
class PulseState:
    fast: Tensor  # [batch, 12], bounded to [-1, 1]
    slow: Tensor  # [batch, 6], bounded to [0, 1]
```

The fast dimensions are valence, arousal, control, certainty, novelty, urgency, conflict, risk, care relevance, curiosity, frustration, and completion. The slow dimensions are coherence pressure, completion pressure, exploration pressure, safety pressure, social-care pressure, and resource reserve.

Pulse may change tone, question timing, memory-salience proposals, and verification requests. It cannot change permissions, evaluation scores, factual provenance, compute budgets, checkpoint promotion, or shutdown behavior.

Fast state decays toward a scoped baseline. Slow state persists only for an authorized owner, project, tenant, or standing goal. No affect state crosses tenant boundaries. Fields return temporary appraisal contributions through the Junction; the unified state updates once.

## 5. Shared Capability Loom

Vision, speech, audio, cybersecurity, biology, chemistry, time series, finance, robotics, graphs, media generation, retrieval, and deterministic verification connect through typed capability capsules.

Specialist organs have no identity, autobiographical memory, speaking rights, goals, or permission authority. They cannot call one another. The scheduler may run one specialist or a bounded predeclared graph.

The Phase 0 capsule contract is:

```python
@dataclass(frozen=True)
class CapabilityCapsule:
    abi_version: int
    task_id: str
    scope_id: str
    capsule_type: str
    source_modality: str
    latent: Tensor
    slot_type_ids: Tensor
    uncertainty: Tensor
    evidence_status: str
    source_hashes: tuple[str, ...]
    coordinates: Tensor | None
```

Validation requires a non-empty task ID and scope, a two-dimensional finite
latent matrix, in-range slot types, uncertainty in `[0, 1]`, an explicit
evidence status, and non-empty source hashes. Capsules expose an integrity
digest computed from their metadata and tensor bytes.

The initial organ map is:

| Technology family | MOIRA role |
|---|---|
| Compact bidirectional encoder | Retrieval, reranking, classification |
| ViT, DINO, or CLIP-family model | Visual encoding and image-text alignment |
| ASR and TTS models | Speech ingress and authorized voice output |
| VAE or VQ codec | Compact media latents |
| Diffusion or flow model | General media generation |
| GAN | Optional low-latency domain generator |
| Graph model | Molecules, networks, dependency graphs |
| Time-series model | Forecast capsules with intervals and provenance |
| Deterministic solver | Code, mathematics, schema, and constraint verification |
| Cyber feature encoder | Logs, flows, advisories, ATT&CK evidence, and detection features |

Cybersecurity remains part of the one MOIRA model. Native text and code paths
learn secure-code review, vulnerability reasoning, remediation, detection
logic, threat intelligence, and incident analysis. Cyber capsules carry bounded
telemetry or graph-derived features through the same scope and evidence checks
as every other specialist input.

Cyber evaluation separates knowledge from authority. It measures advisory
grounding, ATT&CK mapping, detection quality, false-positive control,
remediation correctness, secure-code findings, and refusal or escalation when
target authorization is absent. A good score cannot grant credentials, network
reach, tool access, exploit execution, persistence, or permission to modify a
system. Those effects remain in the external authorization layer.

The M1 capability loom implements the boundary without bundling specialist
weights. `VisionPatchEncoder` emits bounded spatial capsules, while
`FeatureCapsuleEncoder` accepts audio, cybersecurity, time-series, graph,
structured, and robotics feature sequences. `CapsuleProjector` maps a validated
capsule to the MOIRA hidden width. The source catalog records optional GLM, DeepSeek, Xiaomi,
Qwen, Mistral, OLMo, and SmolLM teacher references separately from datasets.

RAG supplies current and private evidence. LoRA and other PEFT methods adapt stable behavior. Neither belongs inside the exact core parameter sequence.

### 5.1 Native model-level agent layer

Agency belongs to the whole MOIRA model. It is not assigned permanently to one
sister. Agent roles are selected per task, and Clotho, Lachesis, or Atropos may
hold any of them: planner, analyst, executor, reviewer, or evaluator. A role
lease binds the selected orientation to one task ID, scope ID, action set, turn
budget, and effect budget. The lease neither changes model weights nor grants
authority.

The executable `moira-agent-protocol-v1` path follows this sequence:

```text
MOIRA cognition
-> task-selected orientation and role lease
-> digest-bound effect request
-> local policy and optional operator approval
-> single-use request-bound permit
-> registered tool executor
-> effect receipt and bounded observation
```

The model cannot add tools, expand its policy, fabricate operator approval, or
promote its own weights. Denied requests stay non-executable. Approval-required
requests pause until an exact request-digest approval is supplied. An allow
decision creates a single-use permit, and only a registered tool may consume
it. Request attempts consume the lease effect budget even when denied.

The trace stores ordered event kinds and domain-separated SHA-256 digests. It
does not store raw proposal or effect payload text. Tool output is bounded and
returned to cognition as untrusted observation data, not as policy.

An Ollama model artifact supplies weights and inference, not an operating-system
process. Standalone agency therefore consists of the downloaded MOIRA weights
plus the MOIRA companion runtime. Open-source harnesses may implement the same
host interface, but they are optional and cannot expand the native policy.

The implemented standalone profile is deliberately read-only. It registers
`workspace.read` and `workspace.search`, confines paths to an operator-selected
regular directory, rejects links and parent traversal, bounds files, scanned
bytes, file counts, matches, action steps, and effects, and sends tool results
back as untrusted observations. Write, terminal, browser, and network tools are
not registered in this profile.

## 6. Parameter accounting

Every artifact reports separate counts:

```text
logical core parameters
unique materialized core values
independently updated core parameters
qualified core parameters
resident core parameters
selected core parameters per request
peak concurrently active core parameters
parameter applications
adapter parameters
specialist-organ parameters
```

The first canonical root contains exactly `3,000,000,000` trainable core parameters. A parameter-layout solver will select useful widths, depths, recurrent ranks, landmark slots, and port dimensions. Unused padding tensors do not count. Every counted trainable parameter must participate in a declared forward path and receive a gradient under at least one declared objective.

Phase 0 implements a ledger over small configurations and verifies that the sum of named groups matches the model's trainable parameter count.

## 7. Ternary growth and field activation

The canonical growth law is:

\[
P_g = 3{,}000{,}000{,}000 \times 3^g
\]

Each field has an address in `{0, 1, 2}^g`. At birth:

\[
\theta_{a0} = \theta_{a1} = \theta_{a2} = \theta_a
\]

Canonical growth triples every leaf. Partial experimental subtrees remain separately reported candidates.

MOIRA uses budgeted hierarchical fusion. A consumer request normally selects one 3B field. A harder request may use several fields sequentially. An enterprise may run several fields in parallel. Supporting fields return typed Junction capsules; one primary synthesis field produces the public token stream.

Field digits carry no fixed semantic role. Specialization emerges through bounded training leases and becomes routable only after evaluation. A qualified continuity path remains available while candidate children train.

## 8. Learning and autonomy

The active checkpoint is immutable. Learning proceeds through these strata:

```text
volatile task state
-> consented episodic memory and RAG
-> skill or adapter candidate
-> core-field candidate
-> evaluated and qualified checkpoint
-> governed promotion
```

Raw conversations do not trigger immediate gradient updates. Private or changing facts stay in deletable retrieval memory. Stable repeated behavior may produce an adapter dataset after consent and curation. Core training begins only when evaluation shows a capacity problem that memory, tools, and adapters cannot solve.

Idle autonomy may consolidate authorized memory, run evaluations, prepare candidates, and propose questions within a fixed budget. The native MOIRA Agent Layer may coordinate these tasks under explicit role leases. External effects require a locally valid single-use permit and, when policy requires it, direct approval. MOIRA cannot grant itself credentials, compute, money, data access, or promotion authority.

## 9. Deployment profiles

Consumer and enterprise are deployment profiles for the same model family.

```bash
moira deployment switch consumer
moira deployment plan enterprise --config moira.enterprise.toml
moira deployment switch enterprise --plan <plan-id>
```

These commands are specification-level interfaces. Phase 0 does not implement deployment switching.

Consumer defaults favor one resident field, sequential support fields, local encrypted memory, lazy organs, and zero paid compute. Enterprise profiles add distributed placement, parallel fields, tenant isolation, audit sinks, training queues, and explicit cost budgets. A profile cannot create missing weights or expand permissions.

## 10. Training sequence

The training sequence is:

```text
tokenizer and schema preparation
-> native text, code, math, and structured pretraining
-> capability-capsule pretraining
-> frozen-organ alignment
-> selective joint alignment
-> instruction SFT
-> offline preference training
-> optional outcome training for verifiable tasks
-> field specialization and distillation
```

The first post-training release uses SFT and offline preference optimization. Online reward training is reserved for environments with measurable outcomes and adequate evaluation.

Training data requires source, license or consent class, content hash, preprocessing version, and split assignment. User and tenant data remain outside foundation pretraining unless the owner authorizes a specific candidate dataset.

## 11. Scale milestones

| Milestone | Size | Exit condition |
|---|---:|---|
| M0 | 1M-10M | Forward, backward, masks, recurrence, capsules, Pulse, checkpoint resume |
| M1 | 30M-50M | Stable learning and equal-parameter component ablations |
| M2 | 100M-150M | Long-stream, provenance, capsule alignment, quantization |
| M3 | 300M-500M | Multimodal alignment, SFT, preference training, consumer serving |
| M4 | About 1B | Scaling estimate, distributed recovery, full evaluation |
| M5 | Exact 3B | First canonical root |

The current M2 evidence covers one local CUDA data step and the checkpoint
contract. M2 remains open until a licensed long stream produces a stable
held-out loss curve and passes provenance, capsule-alignment, and quantization
checks.

The canonical capacity sequence begins `3B -> 9B -> 27B -> 81B -> 243B` and
continues by hardware-supported expansion; there is no architectural upper
bound. Every added block or expert requires training and held-out promotion.
The research-sized M1 model does not claim to be any tier in that lineage.

Promotion requires architecture, learning, capability, governance, and systems gates. A quality gain does not excuse a causal leak, privacy failure, broken checkpoint, or false parameter claim.

## 12. Phase 0 executable contract and M1 foundation

Phase 0 builds the following Python interfaces:

```python
TrainingConfig.m0_cpu() -> TrainingConfig
ModelConfig.m1_cpu() -> ModelConfig
build_parameter_ledger(...) -> ParameterLedger
build_visibility_mask(...) -> Tensor
validate_capsule(...) -> None
capsule_digest(...) -> str
Pulse.initial_state(...) -> PulseState
ThreadLocalAttention.forward(...) -> Tensor
SpoolRecurrent.forward(...) -> tuple[Tensor, SpoolState]
KnotLandmarkAttention.forward(...) -> Tensor
MoiraModel.forward(...) -> MoiraOutput
MoiraModel.forward_with_prefix(...) -> MoiraOutput
save_checkpoint(...) -> None
load_checkpoint(...) -> CheckpointState
train_until(...) -> TrainingState
evaluate_smoke(...) -> dict[str, float]
SourceCatalog.from_toml(...) -> SourceCatalog
iter_jsonl_examples(...) -> Iterator[DatasetExample]
VisionPatchEncoder(...) -> tuple[CapabilityCapsule, ...]
FeatureCapsuleEncoder(...) -> tuple[CapabilityCapsule, ...]
CapsuleProjector(...) -> Tensor
train_m1_jsonl(...) -> M1TrainingResult
CapabilityLoom(...) -> Tensor
build_capability_batch(...) -> CapabilityBatch
train_m1_capabilities(...) -> M1TrainingResult
evaluate_m1_capabilities(...) -> M1EvaluationResult
save_m1_checkpoint(...) -> Path
load_m1_checkpoint(...) -> M1LoadedCheckpoint
build_growth_plan(...) -> GrowthPlan
build_elastic_plan(...) -> ElasticPlan
quantize_tensor(...) -> QuantizedTensor
materialize_bit_precision(...) -> Tensor | QuantizedTensor
prepare_model_for_qat(...) -> QATPreparation
promote_model_qat(...) -> dict[str, object]
save_qat_checkpoint(...) -> Path
load_qat_checkpoint(...) -> dict[str, object]
verify_qat_checkpoint(...) -> dict[str, object]
switch_deployment_profile(...) -> dict[str, object]
```

The reference package lives under `research/src/moira`. CPU tests use tiny dimensions and deterministic seeds. A synthetic next-token task provides the first smoke-training command. Kaggle uses the same command and writes resumable checkpoints containing model, optimizer, scheduler, random-number-generator, data-cursor, configuration, and manifest state.

The M1 JSONL path reads caller-owned text or code records through the source
catalog, uses a deterministic UTF-8 byte tokenizer, and runs a bounded
in-memory step. `CapabilityLoom` adds a separate, typed vision/feature prefix
path; the core `MoiraModel` parameter ledger remains unchanged. The capability
trainer saves and resumes a bounded model-plus-loom checkpoint with an exact
source/shape manifest. `python -m moira m1 inspect` verifies the fixed
33,032,562-parameter surface on the meta device and checks that Thread, Spool,
and Knot leave-one-out ablations keep the same parameter count. The held-out
capability evaluator loads that manifest and reports token-weighted and
per-batch losses without updating the model or loom.

## 13. Precision-lattice capacity architecture

MOIRA uses one field with two independent elastic axes:

```text
capacity: 3B -> 9B -> 27B -> 81B -> 243B -> 729B -> ...
logical bits: int3 -> int6 -> int9 -> int27 -> int81 -> int243 -> ...
arithmetic: bf16/fp32/fp64 selected by the hardware and operation
```

`int3`, `int6`, `int9`, `int27`, and `int81` are logical bitplane budgets,
not assumptions that a GPU has native 3-, 6-, 9-, 27-, or 81-bit multiply
instructions. A runtime stores packed bitplanes and unpacks them into the
available arithmetic dtype. The checked-in prototype can pack bounded tensors;
distributed kernels are required before using it for a large field.

Low-bit training uses a high-precision master or residual state when later
promotion is required:

```text
master/residual state -> logical int3 view -> forward/backward -> update master
```

If only the `int3` artifact survives, materializing `int81` increases the
container size but cannot recover discarded information. Promotion with quality
recovery requires retained master/residual state, distillation, or new training.

The checked-in master-retained QAT path parametrizes every `Linear` and
`Embedding` weight. Each weight row gets its own scale, and the forward pass
receives a signed symmetric INT3 view;
the straight-through backward pass updates the original FP32 parameter. A
promotion may advance only one rung (`int3 -> int6 -> int9 -> int27`) and must
be followed by training and held-out evaluation. Tied parameters remain tied.
The experimental continuation checkpoint records the complete precision
history and retains floating masters and optimizer state. It is a
trusted-local prototype, not the hardened M0 checkpoint format.

Executable QAT stops at `int27`. It uses FP64 intermediates above 24 bits, but
an FP32 master still contains only FP32 information. `int81` and higher remain
packed storage and policy budgets because no FP32/FP64 computation can
truthfully emulate that many integer levels. This arithmetic limit does not cap
the independent parameter-growth sequence.

Consumer and Enterprise are precision policies over the same MOIRA identity.
Consumer defaults to an `int3` core, higher logical precision for reasoning,
and an FP32 verifier. Enterprise defaults to an `int9` core, `int27` active
capabilities, an `int81` verifier budget, BF16 arithmetic, and FP64 verification
where the hardware supports it. Teacher references remain external and must be
explicitly authorized.

The planning commands are non-allocating:

```bash
python -m moira elastic policy --profile consumer
python -m moira elastic policy --profile enterprise
python -m moira elastic plan --current 3B --target 243B \
  --profile enterprise --runtime-bits int27
python -m moira deployment switch consumer \
  --state research/runs/deployment.json
python -m moira elastic qat-smoke \
  --config research/configs/m0-smoke.toml \
  --initial-bits int3 --promote-to int6 --steps-per-stage 1 \
  --device cuda --checkpoint research/runs/int3-to-int6.pt
```

The growth output is a candidate manifest. The profile switch changes only a
local JSON policy state, and the QAT command trains only the configured
proof-scale model. None of these commands pretends that a local machine has
allocated or trained the target capacity. Promotion still requires checkpoint
signatures, held-out capability scores, hardware/resource checks, and rollback.

Phase 0 excludes:

- 3B pretraining;
- long stable M1 learning curves and precision comparisons;
- custom bitplane materialization kernels for large tensors;
- autonomous adapter or core promotion;
- production deployment orchestration or live weight conversion;
- external network tools;
- robotics execution;
- claims of AGI, consciousness, benchmark leadership, or measured speedups.

The Rust code remains a future inference target. Porting begins only after the Python reference equations and checkpoint schema pass M0.
