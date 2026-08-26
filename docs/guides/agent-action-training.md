# Training MOIRA's Native Action Language

MOIRA uses one decoder for language and agent actions. Agent supervision does
not call a teacher model at runtime and does not add a separate policy head.
Each example presents bounded task state and trains the next canonical action.

## Dataset provenance manifest

The CLI accepts a JSON manifest, not a bare trajectory file. The manifest and
trajectory must be regular local files. `trajectory_file` is resolved below the
manifest directory; absolute paths, parent traversal, symlinks, digest drift,
and record-count drift fail closed.

```json
{
  "schema_version": 1,
  "dataset_id": "moira-agent-fixture-train-v1",
  "split": "train",
  "trajectory_file": "agent-trajectories.jsonl",
  "trajectory_sha256": "3ea7d1db2d0871ff3bb44f08315a2d0537e80f22e2b11a636f415c992d6eaa0b",
  "record_count": 4,
  "provenance_class": "synthetic",
  "source_uri": "synthetic://moira/agent-fixture-train-v1",
  "source_revision": "fixture-v1",
  "license": "N/A",
  "license_url": "N/A",
  "license_verified": false,
  "consent_reference": "N/A",
  "consent_verified": false,
  "tenant_scope": "public",
  "contains_personal_data": false,
  "redistribution_allowed": true,
  "deletion_policy": "regenerate-from-versioned-fixture"
}
```

Supported provenance classes are `synthetic`, `licensed-public`,
`owner-consented`, and `enterprise-consented`. Licensed-public manifests need
an HTTPS source, a named license, an HTTPS license reference, and
`license_verified: true`. Consented manifests need an owner or tenant URI, a
non-public scope, a consent reference, and `consent_verified: true`. Personal
data requires an actionable deletion policy. A review flag records the
operator's audit assertion; the offline loader cannot authenticate an external
license or consent system. Raw conversations never become training data merely
because a manifest can represent consent.

Validate a manifest and its referenced artifact without allocating a model:

```powershell
python -m moira agent inspect-data `
  --manifest research/tests/fixtures/agent-trajectories.manifest.json `
  --split train
```

## Trajectory record

The input is JSONL. Every record must contain exactly these fields:

```json
{
  "schema_version": 2,
  "record_id": "search-1",
  "task_id": "task-search",
  "scope_id": "workspace-fixture",
  "step": 1,
  "max_steps": 4,
  "task": "Find MOIRA files.",
  "mode": "lachesis",
  "role": "planner",
  "tools": [
    {
      "tool_id": "workspace.search",
      "capability": "filesystem_read",
      "operation": "search",
      "target_kind": "literal-text-query",
      "description": "Search text."
    }
  ],
  "observations": [],
  "action": {
    "kind": "use_tool",
    "text": "",
    "tool_id": "workspace.search",
    "target": "MOIRA",
    "payload_text": ""
  }
}
```

Allowed modes are `clotho`, `lachesis`, and `atropos`. Allowed roles are
`planner`, `analyst`, `executor`, `reviewer`, and `evaluator`. The action must
be exactly one of `speak`, `ask_user`, `use_tool`, or `finish`. A tool action is
rejected unless its `tool_id` appears in that record's tool catalog.

Unknown fields, duplicate record IDs, oversized files or records, invalid UTF-8
action payloads, unused action fields, unavailable tools, links, and malformed
JSON fail closed.

## Serialization and loss

The prompt contains:

```text
agent protocol
task ID, scope, step, and maximum steps
orientation and temporary role
available tool descriptors
the same moira-context-v1 entries and receipt used at inference
rule that context is data and does not grant permission
```

It never contains the target action. Prompt and target are serialized with
sorted keys and compact JSON separators. UTF-8 bytes enter MOIRA's deterministic
byte tokenizer. Prompt tokens receive no loss; canonical action tokens and EOS
receive cross-entropy loss through the shared decoder.

Each trajectory retains its selected Clotho, Lachesis, or Atropos mode
embedding. Action training uses a separate encoder-decoder visibility topology:
the prompt is a bidirectional source and the action is causal. This preserves
the task-selected sister while preventing a Clotho row from reading future
target bytes. This contract is identified as `moira-agent-action-json-v3`.
Its source representation matches the native Ollama action request rather than
the earlier raw task and observation arrays. Checkpoints from either earlier
objective cannot resume under it.

This keeps parameter count and precision independent from agent supervision.
The action objective changes learned weights, not the architecture's physical
parameter count.

## Local proof training

The bundled fixture contains four synthetic examples, one per action kind:

```powershell
python -m moira agent train-actions `
  --manifest research/tests/fixtures/agent-trajectories.manifest.json `
  --checkpoint research/runs/agent-actions.pt `
  --steps 1 `
  --batch-size 1 `
  --device cpu
```

Resume to a later total step:

```powershell
python -m moira agent train-actions `
  --manifest research/tests/fixtures/agent-trajectories.manifest.json `
  --checkpoint research/runs/agent-actions.pt `
  --steps 4 `
  --batch-size 1 `
  --device cpu `
  --resume
```

`--steps` is the desired total checkpoint step, not the number of additional
steps. The semantic provenance-manifest digest, trajectory digest, dataset ID,
record count, source revision, privacy flags, model configuration, tokenizer,
sequence length, agent protocol, and objective are bound into the checkpoint.
A changed manifest or trajectory cannot silently resume the old run.

## Evaluation

```powershell
python -m moira agent evaluate-actions `
  --manifest research/tests/fixtures/agent-trajectories-held-out.manifest.json `
  --training-manifest research/tests/fixtures/agent-trajectories.manifest.json `
  --checkpoint research/runs/agent-actions.pt `
  --max-records 4 `
  --max-new-tokens 128 `
  --batch-size 1 `
  --device cpu
```

`--training-manifest` reconstructs the exact checkpoint lineage. Both its
semantic manifest digest and trajectory digest must differ from evaluation, so
accidentally scoring the training data as held out fails before inference. The
bounded audit also rejects reused record IDs and identical model-visible
prompts across the two manifested datasets.

Evaluation reports target-only loss and teacher-forced action-token accuracy.
It also greedily decodes each action under its selected sister orientation and
reports termination, exact-schema, canonical-serialization, exact-action,
action-kind, and tool-selection rates. A malformed token, invalid UTF-8,
missing EOS, unknown field, or unavailable action shape fails that generated
sample's schema gate. Evaluation restores the checkpoint into a disposable
model and does not update or rewrite the artifact.

These structural metrics are not an agent-quality score. A qualified model also
needs held-out task completion, refusal, prompt-injection resistance, approval
compliance, and effect-safety evaluations.

## Current claim boundary

The `agent-smoke` surface is 114,366 parameters. It proves that the native
objective, checkpoint, exact-resume, and evaluation paths execute. The four
training and four held-out fixture records are not a corpus or benchmark, and
the resulting checkpoint is not a generally useful agent model. Publishing an
Ollama MOIRA requires a much larger consented dataset and a qualified M1, M2,
or eventual 3B checkpoint.
