# MOIRA candidate worker

The candidate worker connects deterministic candidate search to an isolated
training or mutation backend. It runs through MOIRA's native agent kernel. It
does not introduce another authority system and it cannot promote its output.

## Execution path

```text
current candidate archive
-> replay parent-selection receipt
-> replay failure-selection receipt
-> rebuild exact candidate-work request
-> issue request to candidate worker
-> request native candidate_training:run effect
-> local policy and optional operator approval
-> HMAC-backed one-use permit
-> registered backend in a new staging directory
-> hash and validate output
-> non-promoting candidate record and receipts
```

`prepare_candidate_work` binds the selected parent records, target failure
evidence, active-checkpoint digest, training dataset manifest, trainer
manifest, candidate kind, lineage, iteration, expiry, and resource budget.
`CandidateWorker.issue` then replays both search receipts against the current
archive and reconstructs the request. A stale population, modified receipt, or
altered selection-derived field is rejected before it can become executable.
The dataset and trainer manifests remain issuer-selected inputs and are bound
by the native effect permit once the request is submitted.

The native tool action is exactly:

```text
candidate_training:run
```

The request contains no command, executable, arguments, or arbitrary tool
name. An application supplies a pre-registered Python backend when it creates
the worker. The native agent policy must allow the exact action, and it may
require operator approval. The agent kernel binds the canonical request bytes
to its effect request and consumes the permit before invoking the backend.
Submitting the same effect twice is rejected before a second permit or handler
call.

## Budgets

Each request has positive upper bounds for:

- optimizer or mutation steps;
- processed samples;
- processed tokens;
- elapsed wall time; and
- combined output bytes.

The request expiry cannot exceed its wall-time budget or seven days. The
worker rejects a backend report that exceeds the step, sample, or token limits.
It measures wall time around the backend and totals the actual output file
sizes itself.

These are admission and result gates. An in-process backend can consume excess
CPU, GPU memory, or RAM before returning. Hard enforcement of compute, memory,
network, and accelerator limits requires a container, job scheduler, cgroup,
Windows job object, or remote worker with equivalent controls.

## Staging and artifacts

Every accepted request gets a new directory named by its request digest below
the configured staging root. The backend may return only four distinct
relative files from that directory:

- the candidate artifact;
- its creation manifest;
- its selection-evaluation manifest; and
- its metrics artifact.

Absolute paths, `..` traversal, missing files, non-regular files, path escapes,
symlinks, junctions, and hard links are rejected. The worker reads and hashes
every output rather than trusting digests reported by the backend. The
resulting `CandidateRecord`
contains those hashes, the verified parent lineage, and the selection-only
evaluation result. Adding that record to `CandidateSearch` is a separate
explicit operation.

The worker hashes the configured active checkpoint immediately before and
after backend execution. A mismatch fails the work result. Hash comparison is
detection, not prevention: production workers must mount the active checkpoint
read-only and write only to staging. A malicious in-process backend could
temporarily alter and restore a writable file between the two checks.

## Authority boundary

There are two receipts with different roles:

- `AgentEffectReceipt` proves that the native permit was consumed and records
  whether the registered tool call succeeded.
- `CandidateWorkReceipt` binds the request, actual file hashes, measured wall
  time, output size, and candidate-record digest.

Neither receipt can authorize another effect or promote the candidate. The
candidate still requires selection validation, a sealed promotion holdout,
provenance and policy review, artifact signing, canary evidence, and a separate
rollback-capable promotion transaction.

## Current limitations

The v1 worker is a single-process local validation boundary. Its issued
requests and results are held in memory, and failed staging directories are
retained for inspection. The registered backend is trusted application code;
without an OS sandbox it can still write outside the staging directory. The
worker validates only the files it accepts as candidate outputs. It does not
include a LoRA, full-checkpoint, prompt, routing, or
architecture mutation backend. It does not persist work leases, recover a job
after process failure, distribute work to Kaggle or enterprise nodes, validate
manifest schemas, sign artifacts, or activate weights.

The next implementation should add one concrete bounded backend—LoRA is the
smallest useful target—while preserving the same request, native permit,
staging, content-addressing, evaluation, and promotion boundaries.
