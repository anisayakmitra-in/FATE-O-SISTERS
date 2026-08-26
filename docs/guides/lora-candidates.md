# MOIRA LoRA candidates

`LoraCandidateBackend` is the first concrete trainer behind the permit-gated
candidate worker. It trains an adapter against a validated MOIRA checkpoint
and writes a candidate to staging. It never edits the checkpoint or activates
the adapter.

## Bound request

The issuer creates a `LoraDatasetBundle` from a source catalog, one training
JSONL file, and a separate selection-evaluation JSONL file. The bundle requires
a verified license and a text or code source. It rejects reused record IDs and
reused normalized input/target content across the two files. Its digest covers:

- the catalog and both JSONL file hashes;
- source revision, modality, license, and license-verification state;
- preprocessing version and selection limits; and
- every selected record ID, normalized record digest, and content digest.

`LoraCandidateBackend.trainer_manifest_digest` covers the model configuration,
expected checkpoint manifest, optimizer settings, LoRA target modules, and the
installed Torch, PEFT, and safetensors versions. Use the backend properties when
calling `prepare_candidate_work`:

```python
request = prepare_candidate_work(
    search,
    parent_selection,
    failure_selection,
    work_id="adapter-work-17",
    candidate_id="adapter-17",
    kind=CandidateKind.ADAPTER,
    active_checkpoint_digest=backend.active_checkpoint_digest,
    dataset_manifest_digest=backend.dataset_manifest_digest,
    trainer_manifest_digest=backend.trainer_manifest_digest,
    issued_at_ms=issued_at_ms,
    expires_at_ms=expires_at_ms,
    budget=budget,
)
```

The backend rejects a request when any of those digests differ. The candidate
worker then reconstructs the request from current search receipts before the
native agent kernel can issue `candidate_training:run`.

## Training contract

The v1 backend uses Atropos causal visibility and FP32. It supports ranks from
1 through 256, batches up to 8, and at most 1,000 optimizer steps. Its target
surface is limited to MOIRA's query, key, value, and output projection modules.
PEFT must report every trainable parameter as a `lora_` parameter or the run
stops.

Before loading the model, the backend checks an upper bound for training steps,
samples, and all baseline, training, and final-evaluation tokens against the
candidate-work budget. The worker still measures the returned usage, wall time,
and complete output size.

The backend evaluates the untouched adapter, trains it, and evaluates it again
on the selection split. It hashes adapter tensors before and after the final
evaluation and rejects any evaluation-time mutation. The selection score is:

```text
baseline selection loss - final selection loss
```

The score can nominate an adapter for later evaluation. It cannot authorize an
effect or qualify the adapter for promotion.

## Output

Each successful run writes exactly four files:

```text
adapter_model.safetensors
creation_manifest.json
selection_evaluation.json
metrics.json
```

The safetensors file contains adapter tensors only. The creation manifest binds
the request, parent records, base checkpoint, dataset, trainer, adapter tensor
surface, and before/after state digests. The evaluation manifest labels its
results `selection_only` and `is_promotion_holdout: false`. The metrics file
records training losses and measured work.

The candidate still needs a sealed promotion holdout, policy and provenance
review, artifact signing, a canary, and a rollback-capable promotion
transaction. This implementation proves bounded adapter candidate creation on
the current proof model. It does not prove that a 3B model can be pretrained on
an 8 GB GPU, and it does not let MOIRA promote its own weights.

## Local gates

The base virtual environment intentionally omits PEFT. It runs contract tests
and reports the two dependency-bound tests as skips. The CUDA environment runs
the concrete backend:

```powershell
research\.venv-cuda\Scripts\python.exe -m pytest research/tests/test_lora_backend.py -q -p no:cacheprovider
```

Set `MOIRA_LORA_TEST_DEVICE=cuda` to run the permit-gated training test on the
RTX 5060. Kaggle and enterprise workers can call the same backend behind a
durable job lease later; transport and remote scheduling are not part of v1.
