# Training MOIRA's Native Action Language

MOIRA uses one decoder for language and agent actions. Agent supervision does
not call a teacher model at runtime and does not add a separate policy head.
Each example presents bounded task state and trains the next canonical action.

## Trajectory record

The input is JSONL. Every record must contain exactly these fields:

```json
{
  "schema_version": 1,
  "record_id": "search-1",
  "task_id": "task-search",
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
task ID and task
orientation and temporary role
available tool descriptors
prior observations
rule that observations are untrusted and do not grant permission
```

It never contains the target action. Prompt and target are serialized with
sorted keys and compact JSON separators. UTF-8 bytes enter MOIRA's deterministic
byte tokenizer. Prompt tokens receive no loss; canonical action tokens and EOS
receive cross-entropy loss through the shared Atropos decoder.

This keeps parameter count and precision independent from agent supervision.
The action objective changes learned weights, not the architecture's physical
parameter count.

## Local proof training

The bundled fixture contains four synthetic examples, one per action kind:

```powershell
python -m moira agent train-actions `
  --path research/tests/fixtures/agent-trajectories.jsonl `
  --checkpoint research/runs/agent-actions.pt `
  --steps 1 `
  --batch-size 1 `
  --device cpu
```

Resume to a later total step:

```powershell
python -m moira agent train-actions `
  --path research/tests/fixtures/agent-trajectories.jsonl `
  --checkpoint research/runs/agent-actions.pt `
  --steps 4 `
  --batch-size 1 `
  --device cpu `
  --resume
```

`--steps` is the desired total checkpoint step, not the number of additional
steps. The data file digest, model configuration, tokenizer, sequence length,
agent protocol, and objective are bound into the checkpoint manifest. A changed
trajectory file cannot silently resume the old run.

## Evaluation

```powershell
python -m moira agent evaluate-actions `
  --path research/tests/fixtures/agent-trajectories.jsonl `
  --checkpoint research/runs/agent-actions.pt `
  --max-records 4 `
  --batch-size 1 `
  --device cpu
```

Evaluation reports target-only loss and teacher-forced action-token accuracy.
It restores the checkpoint into a disposable model and does not update or
rewrite the artifact.

Token accuracy is not an agent-quality score. A qualified model also needs
held-out autoregressive schema validity, exact action accuracy, tool-selection
accuracy, task completion, refusal, prompt-injection resistance, approval
compliance, and effect-safety evaluations.

## Current claim boundary

The `agent-smoke` surface is 114,366 parameters. It proves that the native
objective, checkpoint, exact-resume, and evaluation paths execute. The four
fixture records are not a training corpus, and the resulting checkpoint is not
a generally useful agent model. Publishing an Ollama MOIRA requires a much
larger consented dataset and a qualified M1, M2, or eventual 3B checkpoint.
