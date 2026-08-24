# MOIRA Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-testable PyTorch reference for MOIRA's Thread, Spool, Knot, orientation masks, Pulse state, capability capsules, parameter ledger, exact checkpoint resume, and synthetic smoke training.

**Architecture:** The Python research package lives beside the existing Rust sketch. It implements the equations in `docs/architecture/moira-v0.md` with small deterministic configurations. The Rust code remains a future inference target until parity fixtures exist.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pytest, TOML configuration, JSON manifests, SHA-256 checkpoint sidecars, Rust/Candle preserved as the future serving path.

**Spec:** `docs/architecture/moira-v0.md`

## Global Constraints

- Work from parent commit `9458927c3d2532f1f3e066d9c725297957a08bc7`. Push only reviewed commits to `origin/main`, as directed by the repository owner.
- Keep the current Rust files in place; do not port or delete them in Phase 0.
- Run every production-code change through a red, green, refactor cycle.
- CPU tests must require no network, private data, external service, or GPU.
- Keep model state explicit. Do not store Spool, Pulse, tenant, or relationship state in module globals.
- Treat `True` as allowed in every internal visibility mask; convert conventions only at the PyTorch operation boundary.
- Count each unique trainable parameter object once. Record aliases for tied weights.
- Every trainable parameter must receive a gradient in the representative reachability test.
- Checkpoints are accepted only from a trusted local or owner-provided path.
- No remote push, paid job, model publication, canonical growth, or long training run belongs in Phase 0.
- Documentation must separate implemented behavior, approved design, and untested research claims.

---

## File map

| File | Responsibility |
|---|---|
| `research/pyproject.toml` | Python package, test dependencies, pytest settings |
| `research/configs/m0-smoke.toml` | Deterministic CPU smoke configuration |
| `research/src/moira/config.py` | Model and training configuration validation |
| `research/src/moira/ledger.py` | Unique parameter accounting and subsystem classification |
| `research/src/moira/masks.py` | Clotho, Lachesis, and Atropos visibility rules |
| `research/src/moira/capsule.py` | Typed capability capsule validation and digest |
| `research/src/moira/pulse.py` | Unified bounded fast and slow affect state |
| `research/src/moira/model.py` | Thread, Spool, Knot, TSK block, and `MoiraModel` |
| `research/src/moira/synthetic.py` | Index-addressed deterministic token stream |
| `research/src/moira/checkpoint.py` | Atomic checkpoint, sidecar, pointer, and resume validation |
| `research/src/moira/train.py` | Optimizer-step loop, smoke evaluation, and exact resume |
| `research/src/moira/__main__.py` | `doctor`, `train`, `evaluate`, and `checkpoint verify` CLI |
| `research/tests/` | Focused unit and integration tests |
| `docs/guides/kaggle-m0.md` | Local and Kaggle commands with persistence limits |
| `README.md` | Honest project status, implemented surface, and entry points |

---

### Task 1: Research package, configuration, and parameter ledger

**Files:**
- Create: `research/pyproject.toml`
- Create: `research/src/moira/__init__.py`
- Create: `research/src/moira/config.py`
- Create: `research/src/moira/ledger.py`
- Create: `research/tests/test_config.py`
- Create: `research/tests/test_ledger.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `ModelConfig`, `TrainingConfig`, `ParameterSubsystem`, `ParameterLedger`, and `build_parameter_ledger()`.
- Consumes: no earlier Phase 0 task.

- [ ] **Step 1: Add the isolated package definition and test settings**

```toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "moira-research"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["torch>=2.4", "numpy>=1.26"]

[project.optional-dependencies]
test = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

Add `.venv/`, `__pycache__/`, `.pytest_cache/`, `research/runs/`, `*.pt`, and `*.pt.tmp` to `.gitignore`.

- [ ] **Step 2: Install a local CPU test environment**

Run:

```powershell
uv venv research/.venv
uv pip install --python research/.venv/Scripts/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python research/.venv/Scripts/python -e "research[test]"
```

Expected: the editable package imports with the CPU PyTorch wheel. This environment validates code; Kaggle uses its preinstalled CUDA build with `pip install -e . --no-deps`.

- [ ] **Step 3: Write failing configuration tests**

```python
def test_m0_config_is_valid_and_head_aligned() -> None:
    config = ModelConfig.m0_cpu()
    assert config.d_model % config.n_heads == 0
    assert config.head_dim == config.d_model // config.n_heads


def test_config_rejects_misaligned_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(vocab_size=32, d_model=15, n_layers=1, n_heads=2,
                    ffn_dim=32, thread_radius=2, spool_rank=4,
                    knot_slots=2, capsule_width=8, max_seq_len=16)
```

- [ ] **Step 4: Run the tests and verify the missing-module failure**

Run:

```powershell
research/.venv/Scripts/python -m pytest research/tests/test_config.py -q
```

Expected: FAIL because `moira.config.ModelConfig` does not exist.

- [ ] **Step 5: Implement validated immutable configuration**

```python
@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn_dim: int
    thread_radius: int
    spool_rank: int
    knot_slots: int
    capsule_width: int
    max_seq_len: int
    dropout: float = 0.0
    tie_embeddings: bool = True
    expected_trainable_parameters: int | None = None

    def __post_init__(self) -> None:
        positive = (self.vocab_size, self.d_model, self.n_layers,
                    self.n_heads, self.ffn_dim, self.thread_radius,
                    self.spool_rank, self.knot_slots, self.capsule_width,
                    self.max_seq_len)
        if any(value <= 0 for value in positive):
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.ffn_dim < self.d_model:
            raise ValueError("ffn_dim must be at least d_model")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def m0_cpu(cls) -> "ModelConfig":
        return cls(64, 32, 2, 4, 64, 3, 8, 4, 16, 32)
```

- [ ] **Step 6: Verify configuration tests pass**

Run the command from Step 4. Expected: PASS.

- [ ] **Step 7: Write failing ledger tests**

```python
class TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Embedding(8, 4)
        self.output = nn.Linear(4, 8, bias=False)
        self.output.weight = self.token.weight


def test_ledger_counts_tied_storage_once() -> None:
    ledger = build_parameter_ledger(
        TiedModel(),
        prefix_rules=(("token", ParameterSubsystem.TOKEN_IO),
                      ("output", ParameterSubsystem.TOKEN_IO)),
    )
    assert ledger.total_trainable == 32
    assert ledger.entries[0].aliases == ("output.weight", "token.weight")
```

- [ ] **Step 8: Run the ledger test and verify failure**

Run:

```powershell
research/.venv/Scripts/python -m pytest research/tests/test_ledger.py -q
```

Expected: FAIL because the ledger API does not exist.

- [ ] **Step 9: Implement unique parameter accounting**

Enumerate `model.named_parameters(remove_duplicate=False)`, group entries by `id(parameter)`, choose the longest matching prefix rule, reject an unclassified name, reject alias classifications that disagree, and compare the observed count with `expected_trainable_parameters` when supplied. `ParameterLedger.assert_matches()` recomputes the unique object count and raises on drift.

- [ ] **Step 10: Run both focused test files**

Run:

```powershell
research/.venv/Scripts/python -m pytest research/tests/test_config.py research/tests/test_ledger.py -q
```

Expected: PASS.

- [ ] **Step 11: Commit Task 1**

```bash
git add .gitignore research/pyproject.toml research/src/moira research/tests/test_config.py research/tests/test_ledger.py
git commit -m "feat: add MOIRA research config and parameter ledger"
```

---

### Task 2: Exact cognitive-orientation masks

**Files:**
- Create: `research/src/moira/masks.py`
- Create: `research/tests/test_masks.py`

**Interfaces:**
- Produces: `CognitiveMode` and `build_visibility_mask(mode, valid_tokens, source_lengths=None)`.
- Consumes: PyTorch only.

- [ ] **Step 1: Write exact failing truth-table tests**

```python
@pytest.mark.parametrize(
    ("mode", "source_lengths", "expected"),
    [
        (CognitiveMode.CLOTHO, None,
         [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
        (CognitiveMode.LACHESIS, torch.tensor([2]),
         [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]),
        (CognitiveMode.ATROPOS, None,
         [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]),
    ],
)
def test_orientation_truth_tables(mode, source_lengths, expected) -> None:
    valid = torch.ones((1, 4), dtype=torch.bool)
    actual = build_visibility_mask(mode, valid, source_lengths)
    assert torch.equal(actual[0], torch.tensor(expected, dtype=torch.bool))
```

- [ ] **Step 2: Verify the tests fail for the missing mask module**

Run:

```powershell
research/.venv/Scripts/python -m pytest research/tests/test_masks.py -q
```

Expected: FAIL because `moira.masks` does not exist.

- [ ] **Step 3: Implement the three visibility regimes**

```python
def build_visibility_mask(mode, valid_tokens, source_lengths=None):
    if valid_tokens.ndim != 2 or valid_tokens.dtype is not torch.bool:
        raise ValueError("valid_tokens must be bool [batch, sequence]")
    batch, length = valid_tokens.shape
    query = torch.arange(length, device=valid_tokens.device).view(1, length, 1)
    key = torch.arange(length, device=valid_tokens.device).view(1, 1, length)
    if mode is CognitiveMode.CLOTHO:
        allowed = torch.ones((batch, length, length), dtype=torch.bool,
                             device=valid_tokens.device)
    elif mode is CognitiveMode.ATROPOS:
        allowed = key <= query
        allowed = allowed.expand(batch, -1, -1)
    else:
        if source_lengths is None or source_lengths.shape != (batch,):
            raise ValueError("source_lengths must be int64 [batch] for lachesis")
        source = source_lengths.view(batch, 1, 1)
        query_is_source = query < source
        key_is_source = key < source
        allowed = (query_is_source & key_is_source) | (~query_is_source & (key_is_source | (key <= query)))
    return allowed & valid_tokens[:, :, None] & valid_tokens[:, None, :]
```

- [ ] **Step 4: Add padding and invalid-input tests, then run the file**

Expected: padded query rows and padded key columns are false; Lachesis without a valid source length raises `ValueError`; all tests PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add research/src/moira/masks.py research/tests/test_masks.py
git commit -m "feat: define MOIRA cognitive visibility masks"
```

---

### Task 3: Capability capsules and unified Pulse state

**Files:**
- Create: `research/src/moira/capsule.py`
- Create: `research/src/moira/pulse.py`
- Create: `research/tests/test_capsule.py`
- Create: `research/tests/test_pulse.py`

**Interfaces:**
- Produces: `CapabilityCapsule`, `validate_capsule()`, `capsule_digest()`, `PulseState`, and `Pulse`.
- Consumes: `ModelConfig.capsule_width` in later tasks.

- [ ] **Step 1: Write failing capsule validation and digest tests**

```python
def test_capsule_fails_closed_for_scope_and_nonfinite_payload() -> None:
    capsule = make_capsule()
    validate_capsule(capsule, expected_task_id="task-1",
                     allowed_scope_ids={"owner:anis"}, expected_width=8,
                     n_slot_types=4)
    with pytest.raises(ValueError, match="scope"):
        validate_capsule(capsule, expected_task_id="task-1",
                         allowed_scope_ids={"tenant:other"}, expected_width=8,
                         n_slot_types=4)
    bad = replace(capsule, latent=torch.tensor([[float("nan")] * 8]))
    with pytest.raises(ValueError, match="finite"):
        bad.validate(expected_width=8, n_slot_types=4)


def test_capsule_digest_tracks_header_and_payload() -> None:
    capsule = make_capsule()
    assert capsule_digest(capsule) == capsule_digest(replace(capsule, latent=capsule.latent.clone()))
    changed = replace(capsule, latent=capsule.latent + 1)
    assert capsule_digest(capsule) != capsule_digest(changed)
```

- [ ] **Step 2: Run capsule tests and verify the missing-module failure**

- [ ] **Step 3: Implement the frozen header, tensor validation, and canonical SHA-256 digest**

Canonicalize header metadata with sorted compact JSON. Add each tensor's name, dtype, shape, and contiguous CPU bytes to the digest. Reject an unknown ABI, empty task or scope, task mismatch, foreign scope, non-floating or nonfinite latent, invalid slot IDs, invalid uncertainty, or coordinate shape mismatch.

- [ ] **Step 4: Run capsule tests and verify PASS**

- [ ] **Step 5: Write failing Pulse bounds and masking tests**

```python
def test_pulse_remains_bounded_and_masked_rows_do_not_change() -> None:
    pulse = Pulse(d_model=8)
    state = pulse.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    appraisal = torch.full((2, 12), 1000.0)
    for _ in range(100):
        next_state = pulse.step(state, appraisal, update_mask=torch.tensor([True, False]))
        assert torch.equal(next_state.fast[1], state.fast[1])
        assert torch.equal(next_state.slow[1], state.slow[1])
        state = next_state
    assert torch.all(state.fast.abs() <= 1)
    assert torch.all((state.slow >= 0) & (state.slow <= 1))
```

- [ ] **Step 6: Run Pulse tests and verify failure**

- [ ] **Step 7: Implement one bounded Pulse module**

Use fixed `FAST_AXES` and `SLOW_AXES` tuples. `Pulse.appraise()` returns `tanh(linear(summary))`. `Pulse.step()` applies configured decay, adds bounded projections, clamps fast to `[-1, 1]`, clamps slow to `[0, 1]`, and preserves rows whose update mask is false. `Pulse.modulate()` applies bounded scale and bias projections to `[batch, sequence, hidden]`.

- [ ] **Step 8: Run both focused files**

Run:

```powershell
research/.venv/Scripts/python -m pytest research/tests/test_capsule.py research/tests/test_pulse.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit Task 3**

```bash
git add research/src/moira/capsule.py research/src/moira/pulse.py research/tests/test_capsule.py research/tests/test_pulse.py
git commit -m "feat: add typed capsules and unified Pulse state"
```

---

### Task 4: CPU reference Thread, Spool, Knot, and MOIRA model

**Files:**
- Create: `research/src/moira/model.py`
- Create: `research/tests/test_model.py`

**Interfaces:**
- Produces: `SpoolState`, `PulseState`, `MoiraState`, `MoiraOutput`, `ThreadLocalAttention`, `SpoolRecurrent`, `KnotLandmarkAttention`, `TSKBlock`, and `MoiraModel`.
- Consumes: `ModelConfig`, `CognitiveMode`, `build_visibility_mask()`, `Pulse`, and `build_parameter_ledger()`.

- [ ] **Step 1: Write a failing strict-locality Thread test**

Construct a `ThreadLocalAttention` with identity test projections, perturb one token outside the configured radius, and assert the selected query output is unchanged. Perturb an allowed neighbor and assert the output changes. Pass an all-false query row and assert the output is finite zero.

- [ ] **Step 2: Run the focused test and verify failure**

- [ ] **Step 3: Implement Thread with local-window intersection and safe masked softmax**

Project Q, K, and V, reshape to heads, intersect the orientation mask with `abs(query-key) <= radius`, set blocked scores to the dtype minimum, zero all-blocked rows after softmax, combine heads, and apply the output projection. RoPE applies to Q and K before scoring.

- [ ] **Step 4: Verify the Thread test passes**

- [ ] **Step 5: Write a failing Spool chunk-equivalence test**

```python
def test_spool_full_sequence_matches_chunked_execution() -> None:
    torch.manual_seed(7)
    layer = SpoolRecurrent(d_model=8, rank=3)
    x = torch.randn(2, 9, 8)
    full, full_state = layer(x)
    first, state = layer(x[:, :4])
    second, state = layer(x[:, 4:7], state)
    third, state = layer(x[:, 7:], state)
    assert torch.allclose(full, torch.cat((first, second, third), dim=1), atol=1e-6)
    assert torch.allclose(full_state.memory, state.memory, atol=1e-6)
```

- [ ] **Step 6: Implement the gated delta-memory Spool equation**

`SpoolState.memory` has shape `[batch, rank, hidden]`. Each step computes Q, K, V, decay, and write gates; reads `q @ state`; updates with the outer product of `k` and the value prediction error; and keeps state unchanged for invalid tokens.

- [ ] **Step 7: Run the Spool test and verify PASS**

- [ ] **Step 8: Write failing Knot visibility and finite-output tests**

Use explicit landmarks with source-end positions. Perturb a future landmark and assert an earlier Atropos output is unchanged. Verify that all-masked rows return finite zero. Verify that changing the landmark type ID changes a visible output.

- [ ] **Step 9: Implement fixed-capacity typed landmark attention**

Knot adds a learned type embedding, projects query/key/value tensors, applies the supplied `[batch, sequence, landmark]` allowed mask, and uses the same safe masked-softmax rule as Thread. Learned landmark selection remains outside Phase 0.

- [ ] **Step 10: Write failing end-to-end leakage and gradient-reachability tests**

```python
def test_atropos_future_token_cannot_change_prior_logits() -> None:
    torch.manual_seed(11)
    model = MoiraModel(ModelConfig.m0_cpu()).eval()
    left = torch.tensor([[1, 2, 3, 4, 5]])
    right = torch.tensor([[1, 2, 3, 4, 9]])
    a = model(left, mode=CognitiveMode.ATROPOS).logits
    b = model(right, mode=CognitiveMode.ATROPOS).logits
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-6)


def test_every_trainable_parameter_is_reachable() -> None:
    model = MoiraModel(ModelConfig.m0_cpu())
    output = model(torch.tensor([[1, 2, 3, 4]]), mode=CognitiveMode.ATROPOS)
    (output.logits.square().mean() + output.appraisal.square().mean()
     + output.state.pulse.slow.mean()).backward()
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    assert missing == []
```

- [ ] **Step 11: Implement `TSKBlock` and `MoiraModel` with one Pulse state**

The model embeds tokens and orientation, runs the configured TSK blocks, computes one masked sequence summary, updates one Pulse state, applies Pulse modulation, normalizes, and emits tied or untied logits. `MoiraState` contains one Spool state per block and exactly one Pulse state.

- [ ] **Step 12: Run the complete model test file**

Run:

```powershell
research/.venv/Scripts/python -m pytest research/tests/test_model.py -q
```

Expected: PASS with finite outputs and no future leakage.

- [ ] **Step 13: Commit Task 4**

```bash
git add research/src/moira/model.py research/tests/test_model.py
git commit -m "feat: implement CPU Thread Spool Knot reference"
```

---

### Task 5: Deterministic synthetic stream and exact checkpoint resume

**Files:**
- Create: `research/src/moira/synthetic.py`
- Create: `research/src/moira/checkpoint.py`
- Create: `research/src/moira/train.py`
- Create: `research/configs/m0-smoke.toml`
- Create: `research/tests/test_synthetic.py`
- Create: `research/tests/test_checkpoint.py`
- Create: `research/tests/test_resume.py`

**Interfaces:**
- Produces: `SyntheticStream.batch()`, `DataCursor`, `save_checkpoint()`, `load_checkpoint()`, `train_until()`, and `evaluate_smoke()`.
- Consumes: `MoiraModel`, `ModelConfig`, and `TrainingConfig`.

- [ ] **Step 1: Write failing index-addressed stream tests**

```python
def test_synthetic_sample_is_a_function_of_seed_split_and_index() -> None:
    stream = SyntheticStream(seed=13, vocab_size=64, sequence_length=16)
    first = stream.sample(split=0, index=9)
    second = stream.sample(split=0, index=9)
    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(first.targets, second.targets)
    assert not torch.equal(first.input_ids, stream.sample(split=1, index=9).input_ids)
```

- [ ] **Step 2: Verify stream tests fail, then implement SplitMix64-derived sample seeds**

Each sample builds a local `torch.Generator`, creates a deterministic repeated-pattern sequence, and returns next-token inputs and targets. `batch(cursor, batch_size)` returns the stacked batch plus a cursor with updated sample and token counts. It does not mutate global RNG state.

- [ ] **Step 3: Run stream tests and verify PASS**

- [ ] **Step 4: Write failing checkpoint integrity tests**

Tests require a `.pt` file, JSON sidecar, and atomic `latest.json`; verify SHA-256; corrupt one byte and require rejection; verify a failed temporary write leaves the prior pointer unchanged.

- [ ] **Step 5: Implement atomic checkpoint writes and strict manifest hashes**

Write to a temporary file in the destination directory, flush, call `os.fsync`, replace the final path atomically, reopen and load it, write the sidecar atomically, then replace `latest.json`. Store model, optimizer, scheduler, optional scaler, Python/NumPy/Torch RNG, cursor, configuration, source revision, ledger, counters, and last metrics. Restore RNG only after constructors and state dictionaries are loaded.

- [ ] **Step 6: Run checkpoint tests and verify PASS**

- [ ] **Step 7: Write the failing uninterrupted-versus-resume test**

Train run A through eight optimizer steps. Train run B through four steps, save, construct fresh model and optimizer objects, resume, and finish through eight. Require exact CPU equality for model tensors, optimizer tensors, scheduler state, cursor, counters, next batch, and post-resume loss values.

- [ ] **Step 8: Implement the optimizer-boundary training loop**

The order is zero gradients, read indexed batch, forward, backward, clip, optimizer step, scheduler step, increment counters, and checkpoint. Phase 0 never saves partial gradient accumulation. Evaluation uses a separate split and never changes the training cursor or RNG.

- [ ] **Step 9: Run the three focused files**

Run:

```powershell
research/.venv/Scripts/python -m pytest research/tests/test_synthetic.py research/tests/test_checkpoint.py research/tests/test_resume.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit Task 5**

```bash
git add research/configs research/src/moira/synthetic.py research/src/moira/checkpoint.py research/src/moira/train.py research/tests/test_synthetic.py research/tests/test_checkpoint.py research/tests/test_resume.py
git commit -m "feat: add deterministic smoke training and exact resume"
```

---

### Task 6: CLI and Kaggle workflow

**Files:**
- Create: `research/src/moira/__main__.py`
- Create: `research/tests/test_cli.py`
- Create: `docs/guides/kaggle-m0.md`

**Interfaces:**
- Produces: `python -m moira doctor`, `train`, `evaluate`, and `checkpoint verify`.
- Consumes: configuration, training, evaluation, and checkpoint APIs from Tasks 1 through 5.

- [ ] **Step 1: Write the failing CLI smoke test**

From a temporary run directory, invoke two CPU training steps, verify the checkpoint, resume through four steps, and run smoke evaluation. Assert zero exit status, a four-step pointer, finite loss, and no network access.

- [ ] **Step 2: Run the CLI test and verify failure**

Run:

```powershell
research/.venv/Scripts/python -m pytest research/tests/test_cli.py -q
```

Expected: FAIL because `moira.__main__` does not expose the commands.

- [ ] **Step 3: Implement the four CLI commands with `argparse`**

`doctor` reports Python, PyTorch, CUDA availability, requested device validity, writable output path, model parameter count, ledger digest, and estimated checkpoint bytes. It fails when CUDA is requested but unavailable. `train` supports `--config`, `--run-dir`, `--device`, `--stop-at-step`, `--resume`, and `--session-minutes`. `evaluate` selects a checkpoint and fixed split. `checkpoint verify` validates hash, schema, and manifests.

- [ ] **Step 4: Run the CLI test and verify PASS**

- [ ] **Step 5: Write the Kaggle guide**

Document the exact flow:

```bash
cp -a /kaggle/input/fate-o-sisters/. /kaggle/working/fate-o-sisters/
cd /kaggle/working/fate-o-sisters/research
python -m pip install -e . --no-deps
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m moira doctor --device cuda
python -m moira train --config configs/m0-smoke.toml \
  --run-dir /kaggle/working/moira-runs/m0 --device cuda \
  --resume auto --session-minutes 600
python -m moira checkpoint verify --run-dir /kaggle/working/moira-runs/m0
```

State that `/kaggle/working` is session storage, GPU type and quota can change, the run directory must be saved or downloaded, and the trainer never uploads or promotes a checkpoint.

- [ ] **Step 6: Commit Task 6**

```bash
git add research/src/moira/__main__.py research/tests/test_cli.py docs/guides/kaggle-m0.md
git commit -m "feat: add local and Kaggle smoke-training CLI"
```

---

### Task 7: Replace unsupported documentation claims and verify the branch

**Files:**
- Modify: `README.md`
- Create: `docs/current-status.md`
- Modify: `docs/architecture/moira-v0.md`
- Modify: `docs/superpowers/plans/2026-08-24-moira-phase-0.md`

**Interfaces:**
- Produces: a repository entry page that distinguishes implemented, proposed, and measured behavior.
- Consumes: the commands and test results from Tasks 1 through 6.

- [ ] **Step 1: Rewrite the README around current evidence**

The first screen must state:

```markdown
# MOIRA

MOIRA is an experimental model architecture under active development. This repository currently contains a small PyTorch research implementation and an earlier Rust/Candle sketch. It does not contain a pretrained 3B checkpoint.
```

Link the architecture specification, status page, Phase 0 plan, and Kaggle guide. List only commands that pass in the current checkout. Remove the unsupported 2B-to-18B claim, `2.9x` speedup, continuous in-place learning, zero-allocation claim, working MCP claim, and unimplemented public Rust API.

- [ ] **Step 2: Record implemented and unimplemented status**

`docs/current-status.md` must name the pinned baseline, current branch, implemented Python modules, preserved Rust limitations, verified test commands, blocked Cargo baseline, and Phase 0 exclusions.

- [ ] **Step 3: Run the no-ai-slop documentation checks**

Search the three user-facing documents for banned filler, unsupported importance claims, rhetorical openers, fake quotations, decorative emphasis, repeated summaries, and excessive em dashes. Revise any match while keeping architecture terminology and exact commands intact.

- [ ] **Step 4: Run focused and full Python verification**

```powershell
research/.venv/Scripts/python -m pytest research/tests -q
research/.venv/Scripts/python -m moira doctor --device cpu
research/.venv/Scripts/python -m moira train --config research/configs/m0-smoke.toml --run-dir research/runs/final-smoke --device cpu --stop-at-step 4 --resume none
research/.venv/Scripts/python -m moira checkpoint verify --run-dir research/runs/final-smoke
research/.venv/Scripts/python -m moira evaluate --config research/configs/m0-smoke.toml --checkpoint research/runs/final-smoke/checkpoints/latest.json --device cpu
```

Expected: every command succeeds; losses are finite; the checkpoint hash and manifests validate.

- [ ] **Step 5: Re-run the Rust baseline without changing its result**

Run:

```powershell
cargo test
```

Record success or the exact dependency-fetch/build failure. Do not describe a network failure as a source-code failure.

- [ ] **Step 6: Review parameter and claim integrity**

Confirm that the observed unique trainable count matches the ledger, every trainable parameter receives a gradient, recurrent applications are not counted as parameters, and the README reports the checkpoint as M0 research output rather than 3B MOIRA.

- [ ] **Step 7: Review repository state**

```bash
git status --short --branch
git diff --check
git diff --stat 9458927c3d2532f1f3e066d9c725297957a08bc7...HEAD
```

Expected: only Phase 0 files are changed, no checkpoint or virtual-environment artifacts are tracked, and `git diff --check` produces no output.

- [ ] **Step 8: Commit Task 7**

```bash
git add README.md docs
git commit -m "docs: define MOIRA Phase 0 architecture and status"
```

## Plan self-review

- Spec coverage: each Phase 0 interface in `docs/architecture/moira-v0.md` maps to Tasks 1 through 6; Task 7 covers claim and status accuracy.
- File boundaries: configuration, masks, typed state, model computation, data, checkpointing, training, and CLI have separate owners.
- Type consistency: the plan uses `ModelConfig`, `CognitiveMode`, `CapabilityCapsule`, `PulseState`, `MoiraState`, `MoiraOutput`, and `ParameterLedger` consistently.
- Test order: every production module begins with a failing behavior test and records the expected reason.
- Scope: deployment switching, 3B training, canonical growth, organ downloads, distributed execution, and external actions remain outside Phase 0.
