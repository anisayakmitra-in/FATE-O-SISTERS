# MOIRA candidate search

MOIRA's candidate search ranks immutable research artifacts. It does not mutate
or train a model, execute candidate code, authorize an effect, activate an
artifact, or promote a checkpoint.

The design takes one useful idea from population-based program and prompt
search: performance alone is a poor parent selector because a strong candidate
can dominate every generation. MOIRA combines a bounded selection score with a
novelty term based on the number of recorded children. It also selects related
failure evidence so a future mutation worker can target observed weaknesses.
The implementation is native and has no runtime dependency on Darwinian
Evolver or a hosted model API.

## Artifact and evidence contract

Each candidate records:

- a candidate and lineage ID;
- a kind: prompt, adapter, routing policy, curriculum, architecture
  configuration, or core checkpoint;
- SHA-256 digests for the immutable artifact and its creation manifest;
- ordered parent IDs and a monotonically advancing iteration; and
- an evaluation manifest, metric digest, bounded selection score, viability
  result, mutation-verification result, and referenced failure IDs.

Failure evidence records a bounded weight, category, dataset ID, and evidence
digest. The archive rejects duplicate candidate IDs, duplicate artifact
digests, missing parents, cross-lineage parents, non-advancing generations,
unknown failure evidence, and a second root.

The archive is bounded to 4,096 candidates and 16,384 failure records. A parent
request can draw at most 64 candidates. A failure request returns at most 64
records from one selected category and never repeats a record within the batch.

## Parent selection

Eligible parents must be viable, mutation-verified, carry failure evidence,
and precede the requested iteration. Candidate IDs provide the stable ordering.
For candidate `i`, the selector computes:

```text
performance_i = sigmoid(sharpness * (score_i - percentile_midpoint))
novelty_i = 1 / (1 + novelty_weight * child_count_i)
weight_i = performance_i * novelty_i
```

The default midpoint is the 75th percentile of eligible scores. Selection is
with replacement, but the temporary child count increases after each draw so
repeated picks become less likely. SHA-256 domain-separated digests over the
population, seed, iteration, ordered candidates, and weights supply each
deterministic draw. The receipt binds every draw and the final selection.

Failure selection first draws one category by aggregate evidence weight, then
draws records within that category by their individual weights without
replacement. Its receipt binds the candidate record, population, seed,
eligible evidence, and selected evidence.

The current v1 algorithm is deterministic within the supported Python runtime.
It uses IEEE-754 floating-point sigmoid and percentile calculations; it does
not claim bit-identical reproduction in another language or math library.

## Evaluation separation

MOIRA needs three distinct data roles:

1. Mutation feedback tells a worker what failed and may be used to generate a
   candidate.
2. Selection validation scores candidates and may influence parent selection.
3. A sealed promotion holdout is not visible to mutation or selection and may
   be opened only by the later governed promotion path.

The executable candidate evaluation is explicitly marked `selection_only`, and
its digest records that it is not a sealed promotion holdout. A good selection
score therefore cannot activate weights or expand permissions.

## Current boundary

`CandidateSearch` is an in-memory, single-process research component. It does
not persist archives, schedule workers, run mutations, train weights, manage
distributed leases, sign artifacts, perform canaries, or switch active model
pointers. Those are separate later milestones.

A future end-to-end path should preserve this order:

```text
immutable active checkpoint
-> authorized research lease
-> mutation feedback and selected parents
-> isolated candidate worker
-> immutable candidate artifact
-> selection validation
-> sealed holdout and policy review
-> signed promotion transaction
-> canary and rollback-capable activation
```

The candidate-search receipt is evidence for the first half of that pipeline.
It is never an execution permit or promotion approval.

The next boundary is implemented by `CandidateWorker`; see
[candidate-worker.md](candidate-worker.md). Candidate search still does not
schedule or execute work itself.
