# Run the MOIRA M0 smoke model on Kaggle

This guide runs the deterministic Phase 0 model on Kaggle CPU. The checked-in
configuration has 24,478 trainable parameters and uses indexed synthetic data.
It tests the model, checkpoint, verification, evaluation, and exact-resume code
paths. The 1M to 10M M0 research milestone and the exact 3B root model remain
later work.

Phase 0 runs in FP32 on CPU. LoRA, PEFT, 8-bit weights, MoE routing, RAG, RLHF,
real-corpus ingestion, GPU execution, recursive training, deployment switching,
and 3x parameter growth remain unimplemented. The reported loss is a code-path
check on fixed synthetic samples, without a benchmark or capability claim.

## Prepare the notebook

Add a copy of this repository to the notebook as a Kaggle input. Paste each
block below into its own notebook cell. Every block starts a Bash cell and
reloads the saved paths, so it does not depend on shell state from an earlier
cell.

Set `SOURCE_REPO` to the input directory that contains
`research/pyproject.toml`. Kaggle mounts inputs read-only, so copy the source
into `/kaggle/working` first.

```bash
%%bash
set -euo pipefail

SOURCE_REPO=/kaggle/input/fate-o-sisters
REPO=/kaggle/working/fate-o-sisters-source
ENV_FILE=/kaggle/working/moira-m0-env.sh

test -f "$SOURCE_REPO/research/pyproject.toml"
mkdir -p "$REPO"
cp -a "$SOURCE_REPO/." "$REPO/"

printf '%s\n' \
  'export REPO=/kaggle/working/fate-o-sisters-source' \
  'export CONFIG=$REPO/research/configs/m0-smoke.toml' \
  'export RUN_DIR=/kaggle/working/moira-m0' \
  'export PYTHONPATH=$REPO/research/src${PYTHONPATH:+:$PYTHONPATH}' \
  'cd "$REPO"' \
  > "$ENV_FILE"

source "$ENV_FILE"
python --version
python -c "import moira, numpy, torch; print('numpy', numpy.__version__); print('torch', torch.__version__)"
```

The package requires Python 3.11 or newer. `PYTHONPATH` runs the copied source
directly, with the notebook's installed PyTorch and NumPy builds. It avoids a
PEP 517 build step and any build-backend download. If the import check fails,
choose a Kaggle image that already supplies compatible versions before creating
a run you expect to resume later.

## Inspect the deployment contracts

MOIRA has one model identity. Consumer and enterprise are future deployment
profiles for that identity.

```bash
%%bash
set -euo pipefail
source /kaggle/working/moira-m0-env.sh

python -m moira deployment describe consumer
python -m moira deployment describe enterprise
```

These commands print static descriptions. They do not change the model or the
notebook, and Phase 0 has no deployment switch command.

## Check the local run boundary

```bash
%%bash
set -euo pipefail
source /kaggle/working/moira-m0-env.sh

python -m moira doctor \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --device cpu
```

Doctor parses the strict TOML schema, applies the deterministic CPU settings,
builds the model, checks the 24,478-parameter ledger, constructs the synthetic
stream, and confirms that the run directory is writable. It does not perform a
training step.

## Train to an absolute step

```bash
%%bash
set -euo pipefail
source /kaggle/working/moira-m0-env.sh

python -m moira train \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --device cpu \
  --stop-at-step 10 \
  --resume none
```

`--stop-at-step` is an absolute global-step target. `--resume none` creates a
new lineage. It never searches the run directory for an older checkpoint. The
command prints one JSON object containing the resolved checkpoint path, lineage
ID, global step, samples seen, tokens seen, and steps run by this invocation.

## Verify and evaluate

Verify the latest compatible checkpoint and its full resumable state:

```bash
%%bash
set -euo pipefail
source /kaggle/working/moira-m0-env.sh

python -m moira checkpoint verify \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR"
```

You can also name the exact checkpoint produced at step 10:

```bash
%%bash
set -euo pipefail
source /kaggle/working/moira-m0-env.sh

CHECKPOINT="$RUN_DIR/checkpoints/step-000000010.pt"
python -m moira checkpoint verify \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT"
```

Run the fixed synthetic evaluation split:

```bash
%%bash
set -euo pipefail
source /kaggle/working/moira-m0-env.sh

python -m moira evaluate \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --device cpu \
  --batches 2
```

## Continue the run

Choose one resume policy. Automatic recovery selects the latest compatible
checkpoint in `RUN_DIR`:

```bash
%%bash
set -euo pipefail
source /kaggle/working/moira-m0-env.sh

python -m moira train \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --device cpu \
  --stop-at-step 20 \
  --resume auto
```

An explicit path pins the starting checkpoint:

```bash
%%bash
set -euo pipefail
source /kaggle/working/moira-m0-env.sh

CHECKPOINT="$RUN_DIR/checkpoints/step-000000010.pt"
python -m moira train \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --device cpu \
  --stop-at-step 20 \
  --resume "$CHECKPOINT"
```

Run only one of those commands after the step-10 run. Both target global step
20, which is the end of the schedule in `m0-smoke.toml`.

## Save the complete run

`/kaggle/working` disappears with the session unless its files are saved as a
notebook output or downloaded. Archive the whole run directory:

```bash
%%bash
set -euo pipefail
source /kaggle/working/moira-m0-env.sh

tar -C /kaggle/working \
  -czf /kaggle/working/moira-m0-run.tar.gz \
  moira-m0
```

Keep the `.pt` files, adjacent `.json` sidecars, `run.json`, `latest.json`, and
any recovery metadata together. Exact checkpoint verification needs the `.pt`
file and its sidecar. Explicit resume and evaluation also read the compatible
`run.json` in the checkpoint directory. Automatic recovery additionally uses
`latest.json` and recovery metadata.

Checkpoint verification binds the model, data, optimization, and
reproducibility sections, plus package source hashes, Python, PyTorch, NumPy,
platform, device, and deterministic runtime settings. Logging frequency,
checkpoint frequency, and retention are operational run settings rather than
semantic checkpoint identity. Restore the same repository revision and
environment with the archived run. A later Kaggle image with different versions
or platform details can fail closed; that rejection protects exact resume rather
than silently changing the run.
