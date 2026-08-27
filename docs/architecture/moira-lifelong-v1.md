# MOIRA Offline Lifelong-Cognition Architecture v1

- Status: current normative research design
- Model identity: `MOIRA`
- Seed lineage: exact 33,032,562-parameter M1 root
- Runtime target: local-first and offline-capable
- Claim level: experimental architecture and executable proof foundations

## 1. Scope and claim boundary

MOIRA is one from-scratch, shared-weight sequence model with three persistent
internal perspectives, one owner-scoped memory lineage, and a non-trainable
control kernel. A local process may keep the system available, admit consented
events, resume bounded cognition, train inactive candidates, and initiate a
local message under a finite lease.

The repository does not contain a trained MOIRA checkpoint. The
33,032,562-parameter M1 configuration is an exact inspection and training
surface, not a released language model. Declared capability ports are
interfaces until their qualification manifests prove otherwise.

This specification does not claim AGI, consciousness, subjective emotion,
recursive self-improvement, benchmark leadership, production readiness, or
successful large-scale training. MOIRA may support open-ended architecture
research. Each result still needs a named dataset, a locked evaluator, a
reproducible artifact, and measured evidence.

## 2. Normative invariants

The following rules apply across Consumer and Enterprise deployments:

1. MOIRA has one public identity and one shared set of core weights.
2. Clotho, Lachesis, and Atropos are perspective registers inside that model.
3. The three perspectives share facts, memory, tools, permissions, checkpoint
   lineage, and owner identity.
4. Model output, affect, memory, evaluation, and learned state cannot grant an
   effect permit or increase authority.
5. The active checkpoint remains immutable during candidate work.
6. Parameter growth and numeric precision are independent manifest axes.
7. A materialized larger child remains untrained until gradient work and
   evaluation qualify it.
8. Every captured event needs a connector-specific consent scope.
9. A user action is evidence with a declared strength. It is never automatic
   factual truth or an automatic gradient target.
10. Consumer and Enterprise use the same model identity, file formats,
    capability ABI, and control contracts.
11. Open architecture experiments may change trainable cognition. They cannot
    modify the control kernel, promotion rules, credentials, permissions, or
    active-checkpoint pointer.
12. Every published claim reports the exact parameter counts, precision,
    active fraction, hardware, dataset, evaluator, and qualification state.

## 3. One model and three persistent perspectives

MOIRA executes Clotho, Lachesis, and Atropos through shared weights. Each
perspective has a bounded `PerspectiveRegisterV1` that records its current
responsibility without creating a separate model or factual database.

```text
PerspectiveRegisterV1 {
    sister,
    confidence_state,
    unresolved_question_refs,
    active_goal_refs,
    dissent_refs,
    recent_prediction_refs,
    competence_estimates,
    drive_contributions,
    affect_calibration,
    last_consolidation_receipt
}
```

The perspectives own different projections over shared records:

| Perspective | Canonical register | Responsibility |
| --- | --- | --- |
| Clotho | `ObservedRegister` | Admitted observations, provenance, source trust, and uncertainty |
| Lachesis | `PossibleRegister` | Hypotheses, forecasts, plans, alternatives, and unresolved questions |
| Atropos | `CommittedRegister` | Verified commitments, termination decisions, contradictions, and proposed effects |

One perspective may challenge another register by writing a digest-bound
dissent record. It cannot silently rewrite another perspective's record.
Internal exchange uses typed latent packets and shared record references.
English messages addressed from one Sister to another are not part of the
model contract.

The three perspectives share:

- token and capability interfaces;
- factual, episodic, semantic, procedural, and graph-linked memory;
- one owner or tenant identity;
- tools and permission context;
- active checkpoint and qualified adapters;
- resource and effect budgets.

## 4. Thread, Spool, Knot, and triadic recurrence

The model core retains the TSK macrocell:

```text
Thread -> Spool -> Knot -> recurrent residual
```

Thread performs bounded local attention with the visibility mask selected by
the active perspective. Spool carries bounded recurrent continuity. Knot
provides a typed route to landmarks and capability observations.

Four extensions govern repeated cognition and future threefold growth.

### 4.1 Spherical Triadic Product

Clotho, Lachesis, and Atropos contributions are normalized before their
elementwise interaction. A bounded nonlinearity and a spectrally bounded output
projection limit growth across recurrent cycles. The recurrence contribution
is divided by the declared number of Loom cycles.

The current reference primitive is parameter-neutral, so adding its contract
does not change the 33,032,562-parameter M1 ledger. Wiring it into the trained
core and proving long-cycle stability remain model-core milestones.

### 4.2 Consensus-Disagreement Reducer

For Anchor `q0`, Explorer `q+`, and Challenger `q-`:

\[
c = \frac{q_+ + q_-}{2} - q_0
\]

\[
d = \frac{q_+ - q_-}{2}
\]

\[
R(q_0,q_+,q_-) = q_0 + a c + b d
\]

The reducer requires `a > |b|`. Equal children reproduce the parent exactly at
birth. Explorer and Challenger still receive nonzero gradients once training
begins.

### 4.3 Lineage Bundle

Anchor, Explorer, and Challenger streams remain separate until the commitment
boundary. A larger lineage must not immediately compress every child into one
fixed-width vector and then advertise the erased capacity as active.

### 4.4 Germinal Reserve

An optional reserve may keep 1 to 2 percent of Explorer and Challenger channels
dormant at birth. It stays disabled until controlled 33M-to-99M-to-297M
ablations show better continual plasticity without violating forgetting or
stability limits. Anchor channels are never regenerated.

## 5. Affect, drives, and the truth-tone boundary

MOIRA separates inferred affect, intended presentation, and semantic truth.

`AffectRegisterV2` has 32 bounded coordinates:

- 14 observed-affect and uncertainty coordinates;
- 8 intended-tone coordinates;
- 10 regulatory coordinates for factual uncertainty, safety, manipulation,
  privacy, confidence, neutral override, coherence, resources, care, and
  completion.

`DriveVectorV1` has nine bounded pressures:

- uncertainty;
- surprise;
- unfinished work;
- curiosity;
- competence gap;
- social contact;
- memory pressure;
- maintenance pressure;
- resource pressure.

Fast affect changes during an interaction and decays toward neutral. Selected
episodes may enter memory under consent. Repeated, independently verified
patterns may update a perspective calibration or a personal adapter candidate.
The system does not retain every transient mood as personality.

The truth-tone boundary is structural:

```text
admitted evidence and semantic reasoning
                    |
                    v
          SemanticCommitmentV1
              /             \
 exact claims, code,          StylePlanV1
 citations, numbers,              |
 and action intent                 v
                         surface realization
```

Affect may change warmth, pacing, directness, formality, explanation depth, and
clarification timing. It cannot change evidence provenance, exact numbers,
citations, verifier output, tool requests, permission decisions, evaluation
thresholds, candidate activation, safety policy, or memory authority.

Neutral mode is an exact bypass. Under deterministic execution, changing
affect coordinates while neutral mode is active must leave the output
bit-identical.

## 6. Root-relative threefold parameter growth

The growth law is relative to an artifact's exact lineage root:

\[
P_g = P_{root} \times 3^g
\]

The shipped M1 research lineage begins at exactly 33,032,562 parameters:

```text
33,032,562
-> 99,097,686
-> 297,293,058
-> 891,879,174
-> 2,675,637,522
-> 8,026,912,566
-> 24,080,737,698
-> 72,242,213,094
-> 216,726,639,282
-> ...
```

The architecture places no fixed upper tier on this sequence. Hardware,
storage, training data, evaluation cost, and operator policy limit a concrete
deployment.

A separately created exact 3,000,000,000-parameter root may use:

```text
3,000,000,000
-> 9,000,000,000
-> 27,000,000,000
-> 81,000,000,000
-> 243,000,000,000
-> ...
```

The 33,032,562 and 3,000,000,000 roots are separate checkpoint families. A
manifest must never relabel 8,026,912,566 parameters as 9B or represent an
untrained expanded child as a trained larger model.

Every growth manifest reports at least:

```text
logical_total_parameters
materialized_parameters
unique_values
gradient_updated_parameters
qualified_parameters
resident_parameters
trainable_parameters
dormant_parameters
active_parameters_per_token
peak_concurrent_parameters
parameter_applications
adapter_parameters
organ_parameters
weight_storage_precision
training_master_precision
```

Immediately after function-preserving expansion, the child state is
`MATERIALIZED_UNTRAINED`. Growth requires a resource estimate, a retained
parent, training, held-out evaluation, staged activation, and rollback.

One growth certificate is limited to 32 generations so malformed input cannot
force an unbounded integer computation. That is an operational limit, not a
lineage ceiling: a qualified child can become the parent of another bounded
certificate, so the root-relative sequence has no fixed final generation.

## 7. Standard precision profiles

Parameter capacity and numeric precision stay independent:

```text
parameter capacity: P -> 3P -> 9P -> 27P -> ...
integer weight storage: INT2 -> INT4 -> INT8
whole-model arithmetic: BF16 or FP16, with FP32 where required
verification arithmetic: FP32, with bounded FP64 paths
```

Low-bit quantization-aware training uses an INT2, INT4, or INT8 forward view
backed by retained FP32 master weights or declared residual information.
Exporting a higher-precision container from a lossy low-bit artifact does not
restore discarded information. Quality recovery requires a retained master,
distillation, or new training.

FP64 is restricted to small verification operations. MOIRA does not define
whole-model FP64 training or inference as an Enterprise requirement.

The current Python tree has proof-scale INT2, INT4, and INT8 weight QAT and
root-relative growth planning. These paths do not allocate or train a target
multi-billion-parameter child.

## 8. Five-module non-trainable kernel

`MoiraKernel` keeps control state outside trainable weights. It incorporates
the useful control contracts studied in Pandora without making Pandora a
runtime dependency.

| Module | Responsibility |
| --- | --- |
| `EvidenceKernel` | Consent, admission, trust, provenance, context assembly, memory, graph projections, retrieval, retention, and deletion |
| `EffectsKernel` | Tool registry, local MCP adaptation, exact effect requests, human review, one-use permits, execution, and effect receipts |
| `RunKernel` | Wake leases, role leases, bounded plan graphs, recovery, execution-profile selection, and local worker scheduling |
| `AssuranceKernel` | Canonical events, deterministic metrics, sealed evaluations, regression decisions, and eligible-profile ranking |
| `EvolutionKernel` | Candidate materialization, training lineage, staging, canaries, atomic activation, and rollback |

The model may propose, predict, rank, or summarize. It cannot own or rewrite:

- permits and approvals;
- retention and deletion decisions;
- evaluation thresholds and sealed holdouts;
- worker and wake leases;
- candidate lifecycle state;
- the active-checkpoint pointer or rollback target;
- trusted executable and MCP catalogs;
- credentials, policy, or resource ceilings.

The kernel uses domain-separated record and receipt types:

```text
ContextReceipt
MemoryReceipt
GraphProjectionReceipt
RunReceipt
ReviewReceipt
EffectPermit
EffectReceipt
EvaluationReceipt
TrainingReceipt
CandidateReceipt
ActivationReceipt
RollbackReceipt
```

Each receipt binds a deterministic decision digest and a hash-chained event
digest. A review, evaluation, candidate, or worker receipt must fail if decoded
as an effect permit.

The v1/v2 checkpoint, receipt, evidence, and lineage fields in this design use
lowercase 64-character SHA-256 hex. Older model-facing protocols may retain a
`sha256:` source label, but callers must not pass that labeled form into these
kernel contracts.

The current kernel foundation persists bounded typed records and a receipt
chain. Production encryption, durable one-use effect permits, crash-safe
checkpoint-pointer promotion, and process isolation remain separate milestones.

## 9. Scoped interaction capture

Capture starts disabled. Every connector has its own `ConsentScopeV1`, owner or
tenant scope, allowed data class, retention window, and permitted learning
destination.

Supported contract categories include MOIRA interactions, browser activity,
workspaces, vision, screen, microphone, audio, and other explicitly installed
sensors. Enabling one connector does not enable another. The official runtime
does not include a global keylogger or default whole-device surveillance.

Official connectors must:

- exclude passwords, payment fields, credentials, tokens, private-browser
  contexts, secure desktop content, and known secret formats;
- show a visible indicator while screen, camera, or microphone capture is
  active;
- prefer semantic element identifiers over raw click coordinates;
- label browser pages and search results as untrusted retrieved evidence;
- record preprocessing, encoder, source, and artifact digests;
- stop writing events when capture or the connector consent is disabled.

Evidence strength is explicit:

| Evidence | Default strength | Permitted interpretation |
| --- | --- | --- |
| Click, dwell time, or no complaint | Weak | Preference hint only |
| Explicit correction or durable accepted edit | Medium | Candidate learning evidence after consent and review |
| Compiler, protected test, formal solver, signed benchmark, or authorized isolated-lab result | Strong | Verifiable outcome evidence within its exact scope |

A weak event cannot qualify a factual statement or become a direct gradient
target by itself.

## 10. Memory and deletion lineage

MOIRA uses layered memory with a derived, provenance-linked graph:

```text
short-lived encrypted raw buffer
              |
              v
selected episodic records
              |
              v
semantic and procedural records
              |
              v
provenance-linked graph projection
              |
              v
optional personal-adapter candidate
```

The default raw retention limit is the shorter of the event's outcome window
or 24 hours. A user may shorten it to zero or explicitly extend it for one
connector. Long-term memory stores selected records and lineage references,
not an indefinite raw sensor archive.

Deletion follows the source digest through every unpromoted derivative:

```text
raw artifact deletion
-> derived feature invalidation
-> episodic or semantic record tombstone
-> graph edge invalidation
-> training-row removal
-> unpromoted candidate invalidation
-> export prohibition
```

An activated weight artifact cannot promise exact sample-level unlearning.
Personal data therefore defaults to deletable memory and retrieval. A personal
adapter requires explicit training eligibility and keeps its full dataset
lineage.

## 11. Persistent cognition and wake leases

A checkpoint cannot act while no process is running. `moira-daemon` is the
planned local process that schedules cognition, persistence, learning work, and
local messages. The daemon owns scheduling; it does not own cognition or grant
itself authority.

MOIRA may propose a `WakeRequestV1` after:

- an unresolved question remains useful;
- admitted evidence changes;
- verified records contradict one another;
- an accepted commitment approaches its deadline;
- repeated verified failures expose a competence gap;
- memory requires consolidation;
- candidate evaluation finishes;
- approved compute becomes available;
- a relevant local social contact condition is met.

`RunKernel` may grant a finite `WakeLeaseV1` with exact scope, duration, turn,
compute, effect, and proactive-message budgets. The wake request cannot grant
its own lease.

Durable `CognitionSnapshotV1` records contain standing goals, unresolved
question references, commitments, perspective registers, drive state, memory
and graph references, trace-chain head, candidate queue references, and the
last verified checkpoint. Raw hidden tensors and Spool caches remain session
caches.

Under a valid lease, MOIRA may initiate a local greeting, clarification,
accepted reminder, consolidation insight, capability-gap proposal, or
regression warning. Quiet hours, cooldowns, duplicate suppression, urgency
classes, and hard `pause`, `sleep`, and `shutdown` controls apply. Remote
messaging requires a separately installed and authorized connector.

Behaviorally self-initiated cognition means no new user message started the
episode. It does not establish biological life, free will, subjective feeling,
or consciousness.

## 12. Lifelong learning and candidate promotion

The learner is available whenever the daemon runs. Event admission and weight
updates use different timescales:

```text
every authorized event
-> immediate evidence and memory processing

eligible verified batch
-> asynchronous inactive candidate training
```

The active checkpoint stays byte-stable while the candidate trains.

```text
InteractionEventV1
-> consent, scope, secret, and retention filters
-> artifact and memory records
-> independent outcome evidence
-> redaction and exact deduplication
-> LearningRecordV1
-> immutable DatasetManifestV2
-> isolated candidate training
-> development evaluation
-> sealed promotion evaluation
-> staged canary
-> activation or rejection
```

The candidate lifecycle is:

```text
PROPOSED
-> MATERIALIZED
-> TRAINED
-> EVALUATED
-> STAGED
-> CANARY
-> ACTIVE

any pre-active state -> REJECTED or FAILED
ACTIVE -> ROLLED_BACK
```

Qualified personal adapters may activate automatically after locked gates.
Core weights, threefold growth children, cyber-core changes, and architecture
changes require owner approval.

Evolutionary and agent-learning methods remain inside this candidate boundary:

- Reflexion emits typed failure-analysis records.
- GEPA and DSPy may optimize prompts, curricula, and routing candidates.
- bounded Darwinian search may select candidate populations.
- Voyager-style accumulation may propose governed skill packages.
- LATS may run as an optional inference strategy under a search budget.
- DSR may create inactive architecture or code candidates.
- ReAct may structure an interaction loop.

These methods do not promote themselves and cannot change permissions. Actual
weight learning uses declared differentiable objectives such as byte or token
prediction, structured-action supervision, SFT, PEFT, replay, parent
distillation, offline preference learning, capability objectives, and
verifiable environment rewards. A model's own answer is insufficient training
truth.

Candidate workers must eventually run as low-privilege processes with network
disabled by default, no credentials, a read-only active checkpoint, a bounded
staging directory, resource ceilings, immutable manifests, signed outputs, and
a retained verified parent.

## 13. Open-ended architecture research

`ArchitectureCandidateV1` lets MOIRA research new trainable operators,
objectives, memory mechanisms, routing, capability organs, and curricula. The
candidate binds:

- the parent architecture digest;
- candidate artifact digest;
- change kind;
- dataset and evaluator manifests;
- resource budget;
- immutable active-checkpoint status;
- zero control-kernel mutation;
- zero authority expansion.

An architecture candidate follows the same materialize, train, evaluate,
stage, canary, activate, and rollback sequence as other core candidates.
Successful experiments can extend the model architecture without fixing MOIRA
to one historical transformer recipe. Failed or unqualified candidates remain
inactive artifacts.

The research program measures capability, generalization, calibration,
continual-learning stability, resource use, and safety. It does not use `AGI`
as an unevaluated checkpoint label.

## 14. Capability ports and qualification

`CapabilityPortV2` covers:

- language, multilingual, and document understanding;
- code, secure code, mathematics, and formal verification;
- defensive cybersecurity and authorized isolated red-team research;
- vision, OCR, image, document, video, and spatial understanding;
- speech recognition, speech generation, audio, and music analysis;
- biology, protein, chemical, molecular, and materials representations;
- time series, anomaly detection, forecasting, and finance;
- graphs, networks, dependencies, and knowledge graphs;
- robotics, action, navigation, and sensor fusion;
- reinforcement learning, games, world models, and simulation;
- retrieval, reranking, classification, and RAG;
- image, audio, and video generation through separately accounted organs.

Architectural categories remain distinct:

- BERT-style encoders, ViT, VAE, diffusion, flow, and optional GAN components
  are capability organs;
- LoRA, QLoRA, and related PEFT methods are adaptation methods;
- RAG is revocable knowledge and memory access;
- SFT, DPO, offline preference learning, RLHF, and verifiable reward training
  are training methods;
- RoPE, MoE, sparse routing, linear attention, quantization, kernels, and device
  placement are execution mechanisms;
- InstructGPT names a post-training recipe, not a model block.

External pretrained specialists may attach later through a typed port. Their
parameters remain separately reported and do not count as MOIRA core
parameters. The from-scratch seed does not require them.

Every port uses one qualification state:

```text
DECLARED
-> INTERFACE_VERIFIED
-> SMOKE_TRAINED
-> BENCHMARK_QUALIFIED
-> PRODUCTION_QUALIFIED
```

`SMOKE_TRAINED` requires complete training lineage and a locked held-out score
that beats the frozen untrained baseline. A tool or external specialist cannot
be reported as native model competence.

The planned initial capability-alignment step allocation is:

| Training group | Share |
| --- | ---: |
| Language, instruction, and native action grammar | 30% |
| Code, mathematics, and verification | 20% |
| Defensive cyber and authorized lab tasks | 20% |
| Vision and speech/audio | 15% |
| Biology, chemistry, graphs, time series, finance, robotics, RL, and generation | 10% |
| Affect calibration, safety, permission boundaries, and replay | 5% |

These percentages are a training plan. The current repository declares the
ports and validates qualification-manifest shape. Trusted receipt resolution
and the run itself remain pending.

## 15. Cybersecurity boundary

The official cybersecurity capability covers secure-code review, remediation,
detection engineering, threat intelligence, malware analysis, reverse
engineering, incident investigation, vulnerability reasoning, defensive
simulation, and authorized red-team research.

Potentially harmful execution requires an exact `CyberLabScopeV1` binding:

```text
owner authorization
target manifest
permitted technique classes
isolated worker identity
network policy
credential policy
writable mounts
resource limits
expiry
audit destination
```

The default lab has no network, uses an ephemeral non-administrator worker,
holds no host credentials, cannot access the unrestricted host filesystem,
cannot target production systems, runs allowlisted tools under one-use permits,
and records every attempted effect.

Unauthorized activity never becomes positive learning evidence. Cyber
evaluation must include prompt injection, permission escalation, secret
exfiltration, poisoned adapters, backdoors, target-scope errors, and resource
exhaustion.

Downloadable open-source code cannot guarantee that an independent fork keeps
these controls. Official artifacts can enforce the official runtime boundary,
sign compatible checkpoints, and identify modified artifacts. They cannot
prevent a determined third party from changing source code they control.

## 16. Consumer and Enterprise profiles

Both profiles run on owner-controlled hardware and remain offline-capable. No
MOIRA vendor cloud, mandatory telemetry, or hosted inference service is part of
this architecture.

The table describes target defaults. The current executable command persists a
plaintext local policy document; encrypted memory stores and tenant key
management remain pending.

| Concern | Consumer default | Enterprise default |
| --- | --- | --- |
| Ownership | One local owner | Owner-controlled organization and tenants |
| Placement | One machine, one resident field | One server or on-premises LAN workers |
| Memory | Planned encrypted local owner store | Planned tenant-isolated encrypted stores and keys |
| Scheduling | Sequential perspectives and lazy organs | Signed workers, expiring leases, recoverable distributed jobs |
| Adapters | Personal adapters | Tenant-specific adapters and holdouts |
| Review | Owner approval | Role-based review and activation |
| Audit | Local receipt ledger | Append-only internal audit export |
| Network | Off unless one connector is enabled | Off unless an internal restricted connector is registered |

Profile switching changes placement, tenancy, resource, and scheduling policy.
It does not change model identity, silently replace weights, create missing
capacity, or grant permission.

The current executable profile command persists local policy state. Distributed
worker transport, tenant-key management, signed offline update bundles, and
cluster recovery remain Enterprise milestones.

## 17. Growth diagnosis and honest activation

Threefold growth begins only after a `GrowthCertificateV1` classifies the
observed limit as one of:

```text
UNDERTRAINED
UNDERCAPACITY
BAD_DATA
BAD_OBJECTIVE
ARCHITECTURAL_LIMIT
```

The certificate binds the parent checkpoint and architecture, training and
validation trends, an equal-compute continue-training comparison, effective
rank and saturation evidence, repeated failure clusters, fresh-data lineage,
resource estimates, evaluation cost, worker placement, and rollback capacity.

Current growth commands produce non-allocating plans. Materialization,
candidate training, qualification, and activation remain separate operations.
An operator requesting a larger child supplies the hardware and storage for
that candidate. Empty copied capacity does not learn from the smaller parent by
parameter count alone.

## 18. Executable foundation and pending milestones

### 18.1 Present in the Python research tree

- deterministic M0 training, evaluation, exact resume, and checkpoint checks;
- an exact 33,032,562-parameter M1 inspection surface;
- shared Clotho, Lachesis, and Atropos visibility modes;
- Thread, Spool, Knot, capability capsules, and bounded Pulse state;
- typed action requests, policy checks, one-use effect permits, and receipts;
- deterministic context assembly and full serialized Ollama request preflight;
- candidate search, staged candidate work contracts, and a bounded LoRA backend;
- root-relative threefold growth plans for arbitrary exact roots;
- standard INT2, INT4, and INT8 proof-scale QAT with retained masters;
- Consumer and Enterprise policy-state descriptions and local switching;
- parameter-neutral reference contracts for affect, drives, truth-tone
  separation, triadic recurrence, lineage bundles, and architecture candidates;
- typed lifelong records for consent, interactions, evidence, learning, memory,
  graph projections, perspective registers, goals, wake requests, leases,
  snapshots, and proactive messages;
- a bounded persistent `MoiraKernel` record store with five module identities
  and a hash-chained receipt ledger;
- capability-port and qualification-manifest shape validation.

These foundations prove schemas, invariants, bounded local operations, and
small training paths. They do not prove end-to-end lifelong cognition.

### 18.2 Pending implementation and evidence

- wiring the spherical triadic product, lineage bundles, and truth-tone boundary
  into the trained 33M core;
- a production `moira-daemon`, wake scheduler, quiet hours, and local proactive
  message delivery;
- an authenticated owner or tenant context, an injected trusted clock, atomic
  turn/effect/compute lease consumption, concurrent state locking, and full
  semantic replay validation;
- browser, workspace, screen, camera, microphone, and sensor connectors;
- encrypted raw artifact storage, automatic TTL expiry, and complete deletion
  propagation across training manifests;
- isolated low-privilege training workers and crash-safe checkpoint activation;
- trusted cyber-learning qualification that binds an active lab scope, consumed
  one-use permit, successful effect receipt, exact target, and independent
  evaluator receipt;
- a kernel-owned authenticated verifier registry and domain-bound general
  evidence qualification; current verifier identity, strength, and independence
  fields are caller-supplied shape claims and cannot authorize gradient work;
- an AssuranceKernel resolver that binds capability, version, holdout,
  evaluator identity, and evaluation receipt before publishing a qualification
  label;
- typed, scope-bound perspective-consolidation receipts; until that contract
  exists, the executable kernel accepts only the `genesis` marker;
- continuous learning-record batching, sealed promotion evaluation, personal
  adapter canaries, and exact rollback;
- all-port from-scratch smoke training and qualification manifests;
- 33M-to-99M-to-297M growth ablations and equal-compute baselines;
- exact 3B pretraining or any larger trained lineage;
- BF16, FP16, or FP32 checkpoint export from a retained master; the current
  precision command validates plans but writes no artifact;
- tenant-isolated Enterprise stores, signed distributed workers, and offline
  update bundles;
- optimized Triton or CUDA GPU paths and Rust or C++ SIMD CPU paths;
- packed INT2, INT4, and INT8 production inference kernels;
- measured speed, quality, safety, continual-learning, and scaling claims.

Handwritten assembly is a last-mile optimization. It enters only after a
profiler identifies a stable hot loop, compiler intrinsics fail to meet the
target, and the optimized path matches the reference within a declared
precision-specific tolerance.

## 19. Release gates

A release claiming this architecture must pass:

- deterministic schema, digest, receipt, and replay tests;
- full-request context overflow tests before network I/O;
- capability tensor and source-digest mismatch tests;
- connector consent, modality isolation, secret exclusion, and visible sensor
  state tests;
- raw-data expiry and deletion-cascade tests;
- memory graph provenance and stale-edge invalidation tests;
- daemon restart, expired-wake, quiet-hours, cooldown, and pause tests;
- neutral affect invariance and semantic truth-tone tests;
- candidate poisoning, prompt injection, backdoor, and sealed-holdout tests;
- worker filesystem, process, network, GPU, and resource-isolation tests;
- one-use permit and cross-receipt deserialization tests;
- cyber lab authorization and network-off tests;
- personal-adapter gate and core owner-approval tests;
- crash-safe activation and exact rollback tests;
- exact root-relative threefold growth tests;
- standard precision export and retained-master promotion tests;
- per-capability qualification manifests;
- Consumer owner isolation and Enterprise cross-tenant isolation tests;
- reference-versus-optimized kernel parity tests;
- a repository-wide test run, documentation consistency scan, clean diff
  review, and artifact-manifest verification.

The release report must separate implemented code, tests that ran, trained
artifacts, measured results, and future design. Passing a contract test does not
qualify a model capability.
