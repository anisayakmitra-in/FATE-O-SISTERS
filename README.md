# MOIRA

MOIRA is an experimental sequence-model research project. The executable Phase 0 reference lives in `research/` and supports deterministic FP32 training and evaluation on CPU. The checked-in smoke configuration has 24,478 trainable parameters and trains on indexed synthetic samples.

The repository does not contain a trained checkpoint or a 3B model. It does not establish AGI, recursive self-improvement, benchmark gains, speedups, quantization, GPU execution, or production deployment. Each of those claims requires a later implementation and measured evidence.

## What works now

The Python reference contains:

- one MOIRA model with shared weights across Clotho, Lachesis, and Atropos attention modes;
- bidirectional Clotho visibility, sealed-source plus causal-target Lachesis visibility, and causal Atropos visibility;
- a Thread local-attention path with RoPE, a bounded recurrent Spool state, and Knot landmark attention;
- standalone typed capability-capsule validation for task, scope, metadata, tensor shape, slot types, and uncertainty, plus a canonical SHA-256 digest;
- Pulse state with 12 fast affect components and 6 slow affect components;
- an exact trainable-parameter ledger checked against the model;
- deterministic synthetic data, training, evaluation, and resume tests;
- checkpoint manifests, sidecars, compatibility checks, retention, and recovery; and
- a CPU command-line interface for preflight checks, training, checkpoint verification, evaluation, and deployment-profile descriptions.

Clotho, Lachesis, and Atropos are internal execution orientations. They do not have separate identities, conversations, permissions, memories, or parameter copies.

Pulse is bounded model state. Its affect components do not show that the model experiences emotions.

## Local Phase 0 run

Use Python 3.11 or newer with PyTorch 2.4 or newer and NumPy 1.26 or newer. Pytest 8 or newer is needed for the test suite.

From the repository root in PowerShell, expose the source package without building a wheel:

```powershell
$env:PYTHONPATH = (Resolve-Path "research/src").Path
```

Check the configuration, resource limits, parameter ledger, and run directory before training:

```powershell
python -m moira doctor --config research/configs/m0-smoke.toml --run-dir research/runs/m0-local --device cpu
```

Start a new lineage and stop after the first checkpoint:

```powershell
python -m moira train --config research/configs/m0-smoke.toml --run-dir research/runs/m0-local --device cpu --stop-at-step 2 --resume none
```

Verify and evaluate the compatible checkpoint selected from the run directory:

```powershell
python -m moira checkpoint verify --config research/configs/m0-smoke.toml --run-dir research/runs/m0-local
python -m moira evaluate --config research/configs/m0-smoke.toml --run-dir research/runs/m0-local --device cpu --batches 2
```

Resume that lineage to step 4:

```powershell
python -m moira train --config research/configs/m0-smoke.toml --run-dir research/runs/m0-local --device cpu --stop-at-step 4 --resume auto
```

`--stop-at-step` is an absolute global-step target. `--resume` is required and accepts `none`, `auto`, or an explicit checkpoint path.

Inspect the two planned deployment profiles:

```powershell
python -m moira deployment describe consumer
python -m moira deployment describe enterprise
```

The profile commands describe planned defaults. They do not switch the model, allocate hardware, or grant permissions in Phase 0.

Successful commands write one compact JSON object to standard output. Expected argument, configuration, filesystem, and checkpoint failures write an error to standard error and exit with status 2. Keep the complete run directory for automatic recovery and resume; a checkpoint tensor file by itself is not the whole run state.

Run the research tests with pytest's cache disabled:

```powershell
python -m pytest research/tests -p no:cacheprovider
```

## Repository map

| Path | Purpose |
| --- | --- |
| `research/src/moira/` | Executable PyTorch CPU reference and CLI |
| `research/tests/` | Model, mask, state, checkpoint, resume, and CLI tests |
| `research/configs/m0-smoke.toml` | Checked-in 24,478-parameter smoke configuration |
| `docs/architecture/moira-v0.md` | Architecture, scope boundaries, and milestone gates |
| `docs/guides/kaggle-m0.md` | Cell-by-cell Kaggle workflow and artifact rules |
| `src/` and `Cargo.toml` | Initial Rust sketch kept for comparison; its forward path returns zero logits and is not the training reference |

## Research boundary

The architecture document records proposals for an exact 3B root model, governed threefold growth, consumer and enterprise placement, specialist capability fields, adapters, retrieval, quantization, and later GPU work. None of those proposals are implemented by the current smoke model. The next model-size milestone is a measured 1M to 10M parameter reference, followed by the evaluation gates in the architecture document.

For a hosted notebook run, follow the [Kaggle M0 guide](docs/guides/kaggle-m0.md). For the model contract and milestone criteria, read the [MOIRA v0 architecture](docs/architecture/moira-v0.md).

## License

MOIRA is available under the [MIT License](LICENSE).
