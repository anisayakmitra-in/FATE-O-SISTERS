# Running MOIRA with Ollama

This path runs MOIRA's native agent loop against weights served by a local
Ollama process. It does not invoke another agent or require an external harness.

## Current prerequisites

- Python 3.11 or newer;
- the `research/` package and its dependencies;
- Ollama listening on `http://127.0.0.1:11434`; and
- a pulled MOIRA model artifact that supports the structured action schema.

The repository does not yet contain or publish trained MOIRA weights. Until an
artifact is published, the command can be tested against the mocked Ollama API
used by the test suite but cannot run genuine MOIRA inference.

## Run

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path "research/src").Path
$moiraModel = "<published-moira-ollama-tag>"
python -m moira agent run-ollama `
  --model $moiraModel `
  --task "Find the files that define MOIRA's parameter ledger and explain them." `
  --workspace (Resolve-Path ".") `
  --mode lachesis `
  --role planner `
  --max-steps 8 `
  --max-effects 4
```

An installed wheel also exposes the equivalent `moira` command through the
package's console entry point.

MOIRA's speech is written to standard error so standard output remains a final
machine-readable JSON result. Questions and exact-request approval prompts are
interactive.

## Read-only boundary

The current standalone policy registers two tools:

| Tool | Target | Limit |
|---|---|---|
| `workspace.read` | Relative regular-file path | 1 MiB per file |
| `workspace.search` | Literal UTF-8 text query | 2,000 files, 16 MiB, 200 matches |

The selected workspace is resolved before the run. Absolute paths, `..`
traversal, links and Windows reparse points are rejected. Search does not follow
linked directories. The runtime does not expose write, delete, shell, browser,
credential, network, package-management, or model-promotion tools.

Each model action consumes one turn. Each tool request consumes one effect,
including requests denied by policy. The default step budget is eight. The
default effect budget is the smaller of four and the step budget. A requested
effect budget cannot exceed the step budget.

## Ollama boundary

The adapter sends a non-streaming request to `/api/chat` with temperature zero
and a strict JSON schema. The model must return exactly one of `speak`,
`ask_user`, `use_tool`, or `finish`. Tool results are labeled untrusted data in
the next action context.

Only loopback Ollama URLs are accepted. Remote inference needs a separate
authenticated transport design; it is not enabled by this command.
