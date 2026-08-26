# Agent context assembly

MOIRA's native agent loop assembles every cognition turn through
`moira-context-v1`. This is a deterministic runtime contract. It does not
summarize with another model, retrieve data by itself, persist memory, or grant
authority.

## Operating sequence

For each turn, the runtime:

1. reads the active role lease and verifies the task ID, orientation, role, and
   step budget;
2. creates one required task fragment bound to the lease scope;
3. labels prior user, model, runtime, and tool observations, with the newest
   observation required on its next turn;
4. combines optional caller-provided user, memory, and retrieval fragments;
5. rejects duplicate fragment IDs;
6. sorts required fragments first, then priority and stable fragment ID;
7. drops expired, wrong-scope, over-classified, exactly label-duplicate, and
   over-budget optional fragments;
8. fails if required context is rejected or cannot fit;
9. returns admitted entries plus a digest receipt; and
10. sends that assembly to cognition. Tool output remains untrusted data.

`public` fragments can enter any scope. All other fragments must match the
active lease's `scope_id` exactly. The classification order is `public`,
`private`, then `restricted`; a turn admits only values at or below its declared
boundary.

## Fragment contract

Every `ContextFragment` declares:

- a unique bounded ID;
- source: task, user, memory, retrieval, tool, model, or runtime;
- trust: trusted, user-supplied, or untrusted;
- classification and scope;
- bounded UTF-8 content;
- provenance;
- priority;
- optional integer expiry; and
- whether the fragment is required.

The public runtime API reserves `runtime.*` IDs. External fragments must be
optional, cannot claim trusted status, may use only user, memory, or retrieval
sources, and have a maximum priority of 60. Current runtime observations use
priority 80; the newest is required at priority 90. Runtime-created tool and
model observations therefore cannot be forged or crowded out through that API.
A later memory subsystem needs a separate verified admission path before it can
issue trusted memory.

## Deduplication and receipts

Content equality alone is insufficient for deduplication. The assembler uses an
exact label-aware key containing source, trust, classification, scope, content
digest, and provenance. Therefore a retrieved statement and an approved owner-memory
statement remain separate evidence even when their text is identical.

Each included entry has both a content digest and a labelled-fragment digest.
The receipt records included IDs and fragment digests, typed drop records,
budget use, task, scope, assembly time, and an assembly digest. The assembly digest binds
all admitted labels, provenance, ordering, token costs, and drop decisions. It
does not treat any fragment as permission or policy. The runtime reads its
injected clock on every turn, so expiry changes during a run are effective.

## Budget boundary

The current token cost is the UTF-8 byte length of fragment content. That is
exact for MOIRA's byte tokenizer and conservative for common subword
tokenizers. Optional fragments are never partially truncated to fit because a
partial tool result or evidence passage can change meaning.

This is not yet a whole-request context-window guarantee. The system prompt,
tool catalog, JSON field names, entry metadata, and receipt also consume
tokens. A later inference preflight must tokenize the complete provider request
against the selected model's actual context limit.

## Current limits

The implementation does not yet provide:

- durable external memory;
- context caching;
- retrieval or reranking;
- semantic compression;
- context-rot recovery;
- provider-specific full-request tokenization; or
- persistent audit storage.

Those are separate milestones. The current layer establishes the admission,
scope, trust, provenance, budget, and receipt boundary they must use.
