# MOIRA v0 Architecture Specification

- Status: approved research design for Phase 0 implementation
- Repository: `anisayakmitra-in/FATE-O-SISTERS`
- Pinned parent commit: `9458927c3d2532f1f3e066d9c725297957a08bc7`
- Canonical model name: MOIRA

## 1. Scope and claim boundary

MOIRA v0 defines an experimental sequence model, its multimodal boundary, its learning lifecycle, and its scale-up rules. The first implementation is a small PyTorch reference model. It exists to test the mathematics and software contracts before any 3B training run.

The current repository does not contain a trained model. Its Rust binary is a sketch: the forward pass returns zero logits, the entropy threshold is hard-coded, and the memory worker does not persist trajectories. Phase 0 keeps that code available for comparison while the executable research model is built beside it.

Claims require measurements. Reusing a weight tensor nine times increases parameter applications; it does not create nine times as many physical parameters. Copying a 3B field three times creates 9B logical parameter positions at birth; it does not add learned capability until the children train and diverge.

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

Vision, speech, audio, biology, chemistry, time series, finance, robotics, graphs, media generation, retrieval, and deterministic verification connect through typed capability capsules.

Specialist organs have no identity, autobiographical memory, speaking rights, goals, or permission authority. They cannot call one another. The scheduler may run one specialist or a bounded predeclared graph.

The Phase 0 capsule contract is:

```python
@dataclass(frozen=True)
class CapabilityCapsule:
    task_id: str
    capsule_type: CapsuleType
    source_modality: str
    payload: Tensor
    uncertainty: float
    evidence_status: EvidenceStatus
    source_hashes: tuple[str, ...]
    scope: str
```

Validation requires a non-empty task ID, a two-dimensional finite payload, uncertainty in `[0, 1]`, an explicit evidence status, and a non-empty scope. Capsules expose an integrity digest computed from their metadata and payload bytes.

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

RAG supplies current and private evidence. LoRA and other PEFT methods adapt stable behavior. Neither belongs inside the exact core parameter sequence.

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

Idle autonomy may consolidate authorized memory, run evaluations, prepare candidates, and propose questions within a fixed budget. External effects require a standing permit or direct approval. MOIRA cannot grant itself credentials, compute, money, data access, or promotion authority.

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

Smaller milestones are research models. They never enter the canonical `3B -> 9B -> 27B` lineage.

Promotion requires architecture, learning, capability, governance, and systems gates. A quality gain does not excuse a causal leak, privacy failure, broken checkpoint, or false parameter claim.

## 12. Phase 0 executable contract

Phase 0 builds the following Python interfaces:

```python
TrainingConfig.m0_cpu() -> TrainingConfig
build_parameter_ledger(...) -> ParameterLedger
build_visibility_mask(...) -> Tensor
validate_capsule(...) -> None
capsule_digest(...) -> str
Pulse.initial_state(...) -> PulseState
ThreadLocalAttention.forward(...) -> Tensor
SpoolRecurrent.forward(...) -> tuple[Tensor, SpoolState]
KnotLandmarkAttention.forward(...) -> Tensor
MoiraModel.forward(...) -> MoiraOutput
save_checkpoint(...) -> None
load_checkpoint(...) -> CheckpointState
train_until(...) -> TrainingState
evaluate_smoke(...) -> dict[str, float]
```

The reference package lives under `research/src/moira`. CPU tests use tiny dimensions and deterministic seeds. A synthetic next-token task provides the first smoke-training command. Kaggle uses the same command and writes resumable checkpoints containing model, optimizer, scheduler, random-number-generator, data-cursor, configuration, and manifest state.

Phase 0 excludes:

- 3B pretraining;
- canonical ternary materialization;
- autonomous adapter or core promotion;
- production deployment switching;
- external network tools;
- robotics execution;
- claims of AGI, consciousness, benchmark leadership, or measured speedups.

The Rust code remains a future inference target. Porting begins only after the Python reference equations and checkpoint schema pass M0.
