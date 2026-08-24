import hashlib
import json
import random
import shutil
import uuid
import zipfile
from copy import deepcopy
from pathlib import Path
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
import torch

import moira.checkpoint as checkpoint_module
from moira.checkpoint import load_checkpoint, save_checkpoint, verify_checkpoint
from moira.config import ModelConfig
from moira.model import MoiraModel
from moira.synthetic import SYNTHETIC_CURSOR_KIND, DataCursor


@pytest.fixture
def run_dir() -> Iterator[Path]:
    path = Path("research/runs") / f"task5-checkpoint-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        ffn_dim=16,
        thread_radius=1,
        spool_rank=2,
        knot_slots=2,
        capsule_width=4,
        max_seq_len=4,
        dropout=0.2,
    )


def _components() -> tuple[
    MoiraModel,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    model = MoiraModel(_model_config())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.01, foreach=False, fused=False
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / 2)
    )
    return model, optimizer, scheduler


def _manifests() -> dict[str, dict[str, object]]:
    config = _model_config()
    return {
        "model": {
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "ffn_dim": config.ffn_dim,
            "thread_radius": config.thread_radius,
            "spool_rank": config.spool_rank,
            "knot_slots": config.knot_slots,
            "capsule_width": config.capsule_width,
            "max_seq_len": config.max_seq_len,
            "dropout": config.dropout,
            "tie_embeddings": config.tie_embeddings,
        },
        "data": {
            "kind": SYNTHETIC_CURSOR_KIND,
            "stream_version": 1,
            "seed": 11,
            "train_split": 0,
            "eval_split": 1,
            "batch_size": 2,
            "sequence_length": 4,
            "vocab_size": 16,
        },
        "optimization": {
            "optimizer": "adamw",
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 1,
            "schedule": "cosine-with-warmup",
            "warmup_steps": 2,
            "schedule_total_steps": 10,
            "precision": "fp32",
            "clip_grad_norm": 1.0,
        },
        "reproducibility": {"seed": 23, "deterministic": "strict"},
        "source": {
            "model.py": "1" * 64,
            "train.py": "2" * 64,
            "synthetic.py": "3" * 64,
            "checkpoint.py": "4" * 64,
        },
        "runtime": {
            "python": "3.test",
            "pytorch": "2.test",
            "numpy": "2.test",
            "platform": "test-platform",
            "device": "cpu",
            "cuda": None,
            "cudnn": None,
            "deterministic_algorithms": True,
        },
    }


def _save(
    run_dir: Path,
    *,
    step: int = 4,
    keep: int = 2,
    lineage_id: str = "lineage-test",
    manifests: dict[str, dict[str, object]] | None = None,
    provenance: dict[str, object] | None = None,
):
    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)
    model, optimizer, scheduler = _components()
    selected_manifests = _manifests() if manifests is None else manifests
    batch_size = int(selected_manifests["data"]["batch_size"])
    sequence_length = int(selected_manifests["data"]["sequence_length"])
    samples_seen = step * batch_size
    return save_checkpoint(
        run_dir,
        lineage_id=lineage_id,
        global_step=step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        data_cursor=DataCursor(
            SYNTHETIC_CURSOR_KIND,
            samples_seen,
            samples_seen,
            samples_seen * sequence_length,
        ),
        semantic_manifests=selected_manifests,
        provenance=(
            {
                "git": {
                    "available": True,
                    "revision": "abc123",
                    "dirty": False,
                }
            }
            if provenance is None
            else provenance
        ),
        parameter_ledger=model.parameter_ledger(),
        counters={"optimizer_steps": step, "batches_seen": step},
        last_metrics={"loss": 1.25, "learning_rate": 3e-4},
        keep_last_checkpoints=keep,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rewrite_sidecar(record, **updates: object) -> None:
    sidecar = json.loads(record.sidecar_path.read_text(encoding="utf-8"))
    sidecar.update(updates)
    record.sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True), encoding="utf-8"
    )


def _rewrite_payload(record, mutate) -> None:
    payload = torch.load(record.checkpoint_path, map_location="cpu", weights_only=True)
    mutate(payload)
    torch.save(payload, record.checkpoint_path)
    _rewrite_sidecar(
        record,
        sha256=_sha256(record.checkpoint_path),
        byte_size=record.checkpoint_path.stat().st_size,
    )


def _copy_candidate(record, checkpoint_dir: Path, name: str) -> tuple[Path, Path]:
    target = checkpoint_dir / name
    sidecar_target = target.with_suffix(".json")
    shutil.copyfile(record.checkpoint_path, target)
    sidecar = json.loads(record.sidecar_path.read_text(encoding="utf-8"))
    sidecar["checkpoint"] = target.name
    sidecar_target.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    return target, sidecar_target


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        assert isinstance(right, Sequence)
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def _rng_snapshot() -> tuple[object, tuple[object, ...], torch.Tensor]:
    numpy_state = np.random.get_state()
    return (
        random.getstate(),
        (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        torch.get_rng_state().clone(),
    )


def _assert_rng_equal(
    left: tuple[object, tuple[object, ...], torch.Tensor],
    right: tuple[object, tuple[object, ...], torch.Tensor],
) -> None:
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    assert np.array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


def test_checkpoint_is_weights_only_loadable_complete_and_pointer_committed(
    run_dir: Path,
) -> None:
    record = _save(run_dir)

    payload = torch.load(
        record.checkpoint_path, map_location="cpu", weights_only=True
    )
    required = {
        "schema_version",
        "lineage_id",
        "global_step",
        "micro_step",
        "samples_seen",
        "tokens_seen",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "data_cursor",
        "semantic_manifests",
        "manifest_hashes",
        "provenance",
        "parameter_ledger",
        "counters",
        "last_metrics",
    }
    assert set(payload) == required
    assert payload["schema_version"] == 2
    assert payload["lineage_id"] == "lineage-test"
    assert payload["global_step"] == 4
    assert payload["micro_step"] == 0
    assert payload["samples_seen"] == 8
    assert payload["tokens_seen"] == 32
    assert set(payload["rng_state"]) == {"python", "numpy", "torch_cpu"}
    assert isinstance(payload["rng_state"]["python"]["state"], torch.Tensor)
    assert isinstance(payload["rng_state"]["numpy"]["keys"], torch.Tensor)

    sidecar = json.loads(record.sidecar_path.read_text(encoding="utf-8"))
    assert set(sidecar) == checkpoint_module._SIDECAR_KEYS
    assert sidecar["sha256"] == _sha256(record.checkpoint_path)
    assert sidecar["byte_size"] == record.checkpoint_path.stat().st_size
    assert sidecar["step"] == 4
    assert sidecar["lineage_id"] == "lineage-test"
    assert sidecar["manifest_hashes"] == payload["manifest_hashes"]

    pointer = json.loads(record.latest_path.read_text(encoding="utf-8"))
    assert set(pointer) == checkpoint_module._POINTER_KEYS
    assert pointer["checkpoint"] == record.checkpoint_path.name
    assert pointer["sidecar"] == record.sidecar_path.name
    assert pointer["sha256"] == sidecar["sha256"]
    assert pointer["byte_size"] == sidecar["byte_size"]
    verified = verify_checkpoint(
        record.checkpoint_path,
        expected_manifests=_manifests(),
        expected_lineage_id="lineage-test",
    )
    assert verified["global_step"] == 4


def test_load_restores_component_states_and_returns_the_cursor(run_dir: Path) -> None:
    record = _save(run_dir)
    payload = torch.load(record.checkpoint_path, map_location="cpu", weights_only=True)
    model, optimizer, scheduler = _components()

    loaded = load_checkpoint(
        record.checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
    )

    assert loaded.global_step == 4
    assert loaded.lineage_id == "lineage-test"
    assert loaded.data_cursor == DataCursor(SYNTHETIC_CURSOR_KIND, 8, 8, 32)
    assert loaded.last_metrics == {"loss": 1.25, "learning_rate": 3e-4}
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, payload["model_state"][name])
    assert optimizer.state_dict() == payload["optimizer_state"]
    assert scheduler.state_dict() == payload["scheduler_state"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.__setitem__("schema_version", 3),
        lambda payload: payload.__setitem__("micro_step", 1),
        lambda payload: payload["data_cursor"].__setitem__("tokens_seen", 31),
        lambda payload: payload.__setitem__("lineage_id", "other-lineage"),
        lambda payload: payload["model_state"].pop(next(iter(payload["model_state"]))),
    ],
    ids=("schema", "micro-step", "cursor", "lineage", "model-state"),
)
def test_deep_payload_mismatches_are_rejected(run_dir: Path, mutation) -> None:
    record = _save(run_dir)
    _rewrite_payload(record, mutation)

    with pytest.raises(ValueError):
        verify_checkpoint(record.checkpoint_path)


@pytest.mark.parametrize("field", ["sha256", "byte_size"])
def test_wrong_sidecar_integrity_is_rejected(run_dir: Path, field: str) -> None:
    record = _save(run_dir)
    sidecar = json.loads(record.sidecar_path.read_text(encoding="utf-8"))
    replacement = "0" * 64 if field == "sha256" else sidecar[field] + 1
    _rewrite_sidecar(record, **{field: replacement})

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        verify_checkpoint(record.checkpoint_path)


def test_truncated_or_byte_modified_explicit_checkpoint_never_falls_back(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    original = record.checkpoint_path.read_bytes()
    record.checkpoint_path.write_bytes(original[: len(original) // 2])

    model, optimizer, scheduler = _components()
    with pytest.raises(ValueError):
        load_checkpoint(
            record.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
        )


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        (
            "optimization",
            lambda manifest: manifest.__setitem__("schedule_total_steps", 11),
        ),
        (
            "source",
            lambda manifest: manifest.__setitem__("train.py", "f" * 64),
        ),
        ("runtime", lambda manifest: manifest.__setitem__("pytorch", "changed")),
    ],
)
def test_changed_strict_manifest_is_rejected(
    run_dir: Path, section: str, mutate
) -> None:
    record = _save(run_dir)
    changed = deepcopy(_manifests())
    mutate(changed[section])

    with pytest.raises(ValueError, match="manifest"):
        verify_checkpoint(record.checkpoint_path, expected_manifests=changed)


def test_checkpoint_provenance_has_an_explicit_non_strict_schema(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    _rewrite_payload(
        record,
        lambda payload: payload["provenance"].__setitem__("unexpected", True),
    )

    with pytest.raises(ValueError, match="provenance"):
        verify_checkpoint(record.checkpoint_path)


def test_failed_temporary_save_keeps_the_previous_pointer_and_checkpoint(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _save(run_dir, step=1)
    pointer_before = first.latest_path.read_bytes()
    model, optimizer, scheduler = _components()

    def fail_save(*args: object, **kwargs: object) -> None:
        raise OSError("injected save failure")

    monkeypatch.setattr(checkpoint_module.torch, "save", fail_save)
    with pytest.raises(OSError, match="injected"):
        save_checkpoint(
            run_dir,
            lineage_id="lineage-test",
            global_step=2,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            data_cursor=DataCursor(SYNTHETIC_CURSOR_KIND, 4, 4, 16),
            semantic_manifests=_manifests(),
            provenance={
                "git": {
                    "available": True,
                    "revision": "abc123",
                    "dirty": False,
                }
            },
            parameter_ledger=model.parameter_ledger(),
            counters={"optimizer_steps": 2, "batches_seen": 2},
            last_metrics={"loss": 1.0},
            keep_last_checkpoints=2,
        )

    assert first.latest_path.read_bytes() == pointer_before
    assert verify_checkpoint(first.checkpoint_path)["global_step"] == 1
    assert not (run_dir / "checkpoints" / "step-000000002.pt").exists()


def test_auto_resume_scans_only_when_the_pointer_is_damaged(run_dir: Path) -> None:
    first = _save(run_dir, step=1)
    second = _save(run_dir, step=2)
    second.latest_path.write_text("not-json", encoding="utf-8")
    model, optimizer, scheduler = _components()

    loaded = load_checkpoint(
        run_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
        auto=True,
    )

    assert loaded.checkpoint_path.name == "step-000000002.pt"
    assert first.checkpoint_path.exists()


def test_auto_resume_does_not_hide_corruption_behind_a_valid_pointer(
    run_dir: Path,
) -> None:
    first = _save(run_dir, step=1)
    second = _save(run_dir, step=2)
    second.checkpoint_path.write_bytes(b"corrupt")
    model, optimizer, scheduler = _components()

    with pytest.raises(ValueError):
        load_checkpoint(
            run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
            auto=True,
        )
    assert first.checkpoint_path.exists()


def test_pruning_removes_only_older_validated_checkpoints(run_dir: Path) -> None:
    first = _save(run_dir, step=1, keep=2)
    second = _save(run_dir, step=2, keep=2)
    third = _save(run_dir, step=3, keep=2)

    assert not first.checkpoint_path.exists()
    assert not first.sidecar_path.exists()
    assert verify_checkpoint(second.checkpoint_path)["global_step"] == 2
    assert verify_checkpoint(third.checkpoint_path)["global_step"] == 3


@pytest.mark.parametrize(
    "boundary",
    ("canonical", "reopen", "hash", "sidecar", "pointer"),
)
def test_same_step_retry_survives_each_publication_boundary(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    first = _save(run_dir, step=1, keep=3)
    pointer_before = first.latest_path.read_bytes()
    original_link = checkpoint_module.os.link
    original_load = checkpoint_module._load_payload
    original_sha256 = checkpoint_module._sha256
    original_write_json = checkpoint_module._atomic_write_json
    injected = False

    def link_then_fail(source: object, target: object, *args: object, **kwargs: object):
        nonlocal injected
        result = original_link(source, target, *args, **kwargs)
        target_path = Path(target)
        is_boundary = (
            boundary == "canonical" and target_path.suffix == ".pt"
        ) or (
            boundary == "sidecar"
            and target_path.name == "step-000000002.json"
        )
        if is_boundary and not injected:
            injected = True
            raise OSError(f"injected {boundary} failure")
        return result

    def load_then_fail(path: Path):
        nonlocal injected
        if boundary == "reopen" and not injected:
            injected = True
            raise OSError("injected reopen failure")
        return original_load(path)

    def hash_then_fail(path: Path) -> str:
        nonlocal injected
        if boundary == "hash" and not injected:
            injected = True
            raise OSError("injected hash failure")
        return original_sha256(path)

    def pointer_then_fail(path: Path, value: object) -> None:
        nonlocal injected
        if boundary == "pointer" and path.name == "latest.json" and not injected:
            injected = True
            raise OSError("injected pointer failure")
        original_write_json(path, value)

    with monkeypatch.context() as context:
        context.setattr(checkpoint_module.os, "link", link_then_fail)
        context.setattr(checkpoint_module, "_load_payload", load_then_fail)
        context.setattr(checkpoint_module, "_sha256", hash_then_fail)
        context.setattr(checkpoint_module, "_atomic_write_json", pointer_then_fail)
        with pytest.raises(OSError, match=f"injected {boundary}"):
            _save(run_dir, step=2, keep=3)

    assert injected
    assert first.latest_path.read_bytes() == pointer_before
    assert verify_checkpoint(first.checkpoint_path)["global_step"] == 1

    retried = _save(run_dir, step=2, keep=3)

    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 2
    assert json.loads(retried.latest_path.read_text(encoding="utf-8"))["step"] == 2


def test_atomic_claim_never_overwrites_a_concurrent_winner(run_dir: Path) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    staged_a = checkpoint_dir / ".a.tmp"
    staged_b = checkpoint_dir / ".b.tmp"
    canonical = checkpoint_dir / "step-000000001.pt"
    staged_a.write_bytes(b"writer-a")
    staged_b.write_bytes(b"writer-b")
    barrier = Barrier(2)

    def claim(staged: Path) -> bool:
        barrier.wait()
        return checkpoint_module._claim_file(staged, canonical)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(claim, (staged_a, staged_b)))

    assert sorted(results) == [False, True]
    assert canonical.read_bytes() in {b"writer-a", b"writer-b"}


def test_retry_never_adopts_a_different_same_step_payload(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save(run_dir, step=1, keep=3)
    original_write_json = checkpoint_module._atomic_write_json

    def fail_before_pointer(path: Path, value: object) -> None:
        if path.name == "latest.json":
            raise OSError("injected pointer failure")
        original_write_json(path, value)

    with monkeypatch.context() as context:
        context.setattr(checkpoint_module, "_atomic_write_json", fail_before_pointer)
        with pytest.raises(OSError, match="pointer"):
            _save(run_dir, step=2, keep=3)

    model, optimizer, scheduler = _components()
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="different|committed"):
        save_checkpoint(
            run_dir,
            lineage_id="lineage-test",
            global_step=2,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            data_cursor=DataCursor(SYNTHETIC_CURSOR_KIND, 4, 4, 16),
            semantic_manifests=_manifests(),
            provenance={
                "git": {
                    "available": True,
                    "revision": "abc123",
                    "dirty": False,
                }
            },
            parameter_ledger=model.parameter_ledger(),
            counters={"optimizer_steps": 2, "batches_seen": 2},
            last_metrics={"loss": 1.0},
            keep_last_checkpoints=3,
        )


def test_retry_reclaims_an_invalid_unreferenced_same_step_artifact(
    run_dir: Path,
) -> None:
    first = _save(run_dir, step=1, keep=3)
    checkpoint = first.checkpoint_path.parent / "step-000000002.pt"
    sidecar = checkpoint.with_suffix(".json")
    checkpoint.write_bytes(b"invalid")
    sidecar.write_text("{}", encoding="utf-8")

    retried = _save(run_dir, step=2, keep=3)

    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 2


def test_retry_repairs_only_an_invalid_unreferenced_sidecar(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save(run_dir, step=1, keep=3)
    original_write_json = checkpoint_module._atomic_write_json

    def fail_before_pointer(path: Path, value: object) -> None:
        if path.name == "latest.json":
            raise OSError("injected pointer failure")
        original_write_json(path, value)

    with monkeypatch.context() as context:
        context.setattr(checkpoint_module, "_atomic_write_json", fail_before_pointer)
        with pytest.raises(OSError, match="pointer"):
            _save(run_dir, step=2, keep=3)
    sidecar = run_dir / "checkpoints" / "step-000000002.json"
    sidecar.write_text("{}", encoding="utf-8")

    retried = _save(run_dir, step=2, keep=3)

    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 2


def test_same_step_retry_ignores_changed_non_strict_git_context(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save(run_dir, step=1, keep=3)
    original_write_json = checkpoint_module._atomic_write_json

    def fail_before_pointer(path: Path, value: object) -> None:
        if path.name == "latest.json":
            raise OSError("injected pointer failure")
        original_write_json(path, value)

    with monkeypatch.context() as context:
        context.setattr(checkpoint_module, "_atomic_write_json", fail_before_pointer)
        with pytest.raises(OSError, match="pointer"):
            _save(run_dir, step=2, keep=3)

    retried = _save(
        run_dir,
        step=2,
        keep=3,
        provenance={
            "git": {"available": True, "revision": "changed", "dirty": True}
        },
    )

    assert verify_checkpoint(retried.checkpoint_path)["provenance"]["git"] == {
        "available": True,
        "revision": "abc123",
        "dirty": False,
    }


def test_retry_never_reclaims_a_pointer_referenced_invalid_artifact(
    run_dir: Path,
) -> None:
    record = _save(run_dir, step=1)
    record.checkpoint_path.write_bytes(b"invalid referenced bytes")
    artifact_before = record.checkpoint_path.read_bytes()
    pointer_before = record.latest_path.read_bytes()

    with pytest.raises(ValueError):
        _save(run_dir, step=1)

    assert record.checkpoint_path.read_bytes() == artifact_before
    assert record.latest_path.read_bytes() == pointer_before


def _malformed_optimizer(payload: dict[str, object]) -> None:
    payload["optimizer_state"]["param_groups"][0]["params"].append(999)


def _malformed_scheduler(payload: dict[str, object]) -> None:
    payload["scheduler_state"]["base_lrs"].append(9.0)


def _malformed_scheduler_scalar(payload: dict[str, object]) -> None:
    payload["scheduler_state"]["last_epoch"] = "not-an-integer"


def _malformed_optimizer_group(payload: dict[str, object]) -> None:
    payload["optimizer_state"]["param_groups"][0]["betas"] = "invalid"


def _malformed_python_rng(payload: dict[str, object]) -> None:
    state = payload["rng_state"]["python"]["state"]
    payload["rng_state"]["python"]["state"] = state[:-1]


def _nonfinite_optimizer(payload: dict[str, object]) -> None:
    optimizer_state = payload["optimizer_state"]
    parameter = optimizer_state["param_groups"][0]["params"][0]
    optimizer_state["state"][parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.tensor([float("nan")]),
        "exp_avg_sq": torch.tensor([0.0]),
    }


def _meta_optimizer(payload: dict[str, object]) -> None:
    optimizer_state = payload["optimizer_state"]
    parameter = optimizer_state["param_groups"][0]["params"][0]
    optimizer_state["state"][parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.empty(1, device="meta"),
        "exp_avg_sq": torch.tensor([0.0]),
    }


def _unsupported_optimizer_dtype(payload: dict[str, object]) -> None:
    optimizer_state = payload["optimizer_state"]
    parameter = optimizer_state["param_groups"][0]["params"][0]
    first_parameter = next(iter(payload["model_state"].values()))
    optimizer_state["state"][parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros(first_parameter.shape, dtype=torch.uint16),
        "exp_avg_sq": torch.zeros(first_parameter.shape, dtype=torch.uint16),
    }


@pytest.mark.parametrize(
    "mutation",
    (
        _malformed_optimizer,
        _malformed_scheduler,
        _malformed_scheduler_scalar,
        _malformed_optimizer_group,
        _malformed_python_rng,
        _nonfinite_optimizer,
        _meta_optimizer,
        _unsupported_optimizer_dtype,
    ),
    ids=(
        "optimizer",
        "scheduler-shape",
        "scheduler-scalar",
        "optimizer-group",
        "rng",
        "nonfinite",
        "meta",
        "unsupported-dtype",
    ),
)
def test_failed_load_never_mutates_live_components_or_global_rng(
    run_dir: Path,
    mutation,
) -> None:
    record = _save(run_dir)
    _rewrite_payload(record, mutation)
    model, optimizer, scheduler = _components()
    with torch.no_grad():
        next(model.parameters()).add_(0.75)
    before = (
        deepcopy(model.state_dict()),
        deepcopy(optimizer.state_dict()),
        deepcopy(scheduler.state_dict()),
    )
    rng_before = _rng_snapshot()

    with pytest.raises(ValueError):
        load_checkpoint(
            record.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
        )

    _assert_nested_equal(before[0], model.state_dict())
    _assert_nested_equal(before[1], optimizer.state_dict())
    _assert_nested_equal(before[2], scheduler.state_dict())
    _assert_rng_equal(rng_before, _rng_snapshot())


def test_injected_component_load_failure_is_preflighted_off_live_objects(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    with torch.no_grad():
        next(model.parameters()).add_(0.75)

    def fail_load(state: object) -> None:
        raise RuntimeError("injected component failure")

    monkeypatch.setattr(scheduler, "load_state_dict", fail_load)
    before = (
        deepcopy(model.state_dict()),
        deepcopy(optimizer.state_dict()),
        deepcopy(scheduler.state_dict()),
    )
    rng_before = _rng_snapshot()
    with pytest.raises(RuntimeError, match="injected component"):
        load_checkpoint(
            record.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
        )

    _assert_nested_equal(before[0], model.state_dict())
    _assert_nested_equal(before[1], optimizer.state_dict())
    _assert_nested_equal(before[2], scheduler.state_dict())
    _assert_rng_equal(rng_before, _rng_snapshot())


def test_phase_zero_checkpoint_rejects_scaler_and_omits_cuda_rng(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    payload = torch.load(record.checkpoint_path, map_location="cpu", weights_only=True)

    assert payload["scaler_state"] is None
    assert set(payload["rng_state"]) == {"python", "numpy", "torch_cpu"}

    class FakeScaler:
        def state_dict(self) -> dict[str, float]:
            return {"scale": 2.0}

    model, optimizer, scheduler = _components()
    with pytest.raises(ValueError, match="Phase 0|scaler"):
        save_checkpoint(
            run_dir / "scaler",
            lineage_id="lineage-test",
            global_step=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=FakeScaler(),
            data_cursor=DataCursor(SYNTHETIC_CURSOR_KIND, 2, 2, 8),
            semantic_manifests=_manifests(),
            provenance={
                "git": {
                    "available": True,
                    "revision": "abc123",
                    "dirty": False,
                }
            },
            parameter_ledger=model.parameter_ledger(),
            counters={"optimizer_steps": 1, "batches_seen": 1},
            last_metrics={"loss": 1.0},
            keep_last_checkpoints=2,
        )


def test_run_metadata_durably_binds_lineage_and_strict_manifests(
    run_dir: Path,
) -> None:
    record = _save(run_dir, step=1)
    metadata_path = record.checkpoint_path.parent / "run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = verify_checkpoint(record.checkpoint_path)

    assert metadata == {
        "schema_version": 2,
        "lineage_id": "lineage-test",
        "manifest_hashes": payload["manifest_hashes"],
    }
    with pytest.raises(ValueError, match="lineage|run metadata"):
        _save(run_dir, step=2, lineage_id="foreign-lineage")
    changed = deepcopy(_manifests())
    changed["source"]["train.py"] = "f" * 64
    with pytest.raises(ValueError, match="run metadata"):
        _save(run_dir, step=2, manifests=changed)
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
    assert not (record.checkpoint_path.parent / "step-000000002.pt").exists()


def test_pointer_field_mismatch_recovers_by_scanning_the_bound_lineage(
    run_dir: Path,
) -> None:
    first = _save(run_dir, step=1, keep=3)
    second = _save(run_dir, step=2, keep=3)
    pointer = json.loads(second.latest_path.read_text(encoding="utf-8"))
    pointer["checkpoint"] = first.checkpoint_path.name
    pointer["sidecar"] = first.sidecar_path.name
    second.latest_path.write_text(json.dumps(pointer), encoding="utf-8")
    model, optimizer, scheduler = _components()

    loaded = load_checkpoint(
        run_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
        expected_lineage_id="lineage-test",
        auto=True,
    )

    assert loaded.checkpoint_path.name == "step-000000002.pt"


@pytest.mark.parametrize("metadata_state", ("missing", "corrupt"))
def test_auto_recovery_fails_closed_without_trusted_run_metadata(
    run_dir: Path,
    metadata_state: str,
) -> None:
    record = _save(run_dir, step=1)
    metadata_path = record.checkpoint_path.parent / "run.json"
    record.latest_path.write_text("not-json", encoding="utf-8")
    if metadata_state == "missing":
        metadata_path.unlink(missing_ok=True)
    else:
        metadata_path.write_text("not-json", encoding="utf-8")
    model, optimizer, scheduler = _components()

    with pytest.raises(ValueError, match="run metadata"):
        load_checkpoint(
            run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
            auto=True,
        )


def test_recovery_filters_higher_foreign_and_incompatible_candidates(
    run_dir: Path,
) -> None:
    first = _save(run_dir, step=1, keep=4)
    second = _save(run_dir, step=2, keep=4)
    checkpoint_dir = second.checkpoint_path.parent
    foreign = _save(
        run_dir / "foreign", step=9, lineage_id="foreign-lineage", keep=4
    )
    changed = deepcopy(_manifests())
    changed["source"]["train.py"] = "f" * 64
    incompatible = _save(
        run_dir / "incompatible", step=8, manifests=changed, keep=4
    )
    _copy_candidate(foreign, checkpoint_dir, "step-000000009.pt")
    _copy_candidate(incompatible, checkpoint_dir, "step-000000008.pt")
    second.latest_path.write_text("not-json", encoding="utf-8")
    model, optimizer, scheduler = _components()

    loaded = load_checkpoint(
        run_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
        expected_lineage_id="lineage-test",
        auto=True,
    )

    assert loaded.checkpoint_path == second.checkpoint_path
    assert first.checkpoint_path.exists()


def test_recovery_rejects_filename_payload_step_mismatch(run_dir: Path) -> None:
    first = _save(run_dir, step=1, keep=3)
    second = _save(run_dir, step=2, keep=3)
    _copy_candidate(first, second.checkpoint_path.parent, "step-000000009.pt")
    second.latest_path.write_text("not-json", encoding="utf-8")
    model, optimizer, scheduler = _components()

    loaded = load_checkpoint(
        run_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
        auto=True,
    )

    assert loaded.checkpoint_path == second.checkpoint_path


def test_retention_counts_only_active_compatible_pairs(run_dir: Path) -> None:
    first = _save(run_dir, step=1, keep=4)
    second = _save(run_dir, step=2, keep=4)
    checkpoint_dir = second.checkpoint_path.parent
    foreign = _save(
        run_dir / "foreign-retention",
        step=9,
        lineage_id="foreign-lineage",
        keep=4,
    )
    changed = deepcopy(_manifests())
    changed["source"]["train.py"] = "f" * 64
    incompatible = _save(
        run_dir / "incompatible-retention",
        step=8,
        manifests=changed,
        keep=4,
    )
    foreign_copy, foreign_sidecar = _copy_candidate(
        foreign, checkpoint_dir, "step-000000009.pt"
    )
    incompatible_copy, incompatible_sidecar = _copy_candidate(
        incompatible, checkpoint_dir, "step-000000008.pt"
    )
    invalid = checkpoint_dir / "step-000000007.pt"
    invalid.write_bytes(b"invalid")
    invalid_sidecar = invalid.with_suffix(".json")
    invalid_sidecar.write_text("{}", encoding="utf-8")

    third = _save(run_dir, step=3, keep=2)

    assert not first.checkpoint_path.exists()
    assert not first.sidecar_path.exists()
    assert verify_checkpoint(second.checkpoint_path)["global_step"] == 2
    assert verify_checkpoint(third.checkpoint_path)["global_step"] == 3
    for path in (
        foreign_copy,
        foreign_sidecar,
        incompatible_copy,
        incompatible_sidecar,
        invalid,
        invalid_sidecar,
    ):
        assert path.exists()


def test_retention_never_deletes_the_current_pointer_target(run_dir: Path) -> None:
    first = _save(run_dir, step=1, keep=3)
    _save(run_dir, step=2, keep=3)
    _save(run_dir, step=3, keep=3)
    sidecar = json.loads(first.sidecar_path.read_text(encoding="utf-8"))
    pointer = dict(sidecar)
    pointer["sidecar"] = first.sidecar_path.name
    first.latest_path.write_text(json.dumps(pointer), encoding="utf-8")

    retried = _save(run_dir, step=1, keep=1)

    assert retried.checkpoint_path == first.checkpoint_path
    assert verify_checkpoint(first.checkpoint_path)["global_step"] == 1


def test_retention_tombstone_recovers_after_pair_deletion_failure(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _save(run_dir, step=1, keep=3)
    _save(run_dir, step=2, keep=3)
    original_link = checkpoint_module.os.link
    injected = False

    def fail_between_pair_operations(
        source: object, target: object, *args: object, **kwargs: object
    ) -> object:
        nonlocal injected
        if Path(target).name == ".step-000000001.pt.pruning" and not injected:
            injected = True
            raise OSError("injected pair deletion failure")
        return original_link(source, target, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(checkpoint_module.os, "link", fail_between_pair_operations)
        with pytest.raises(OSError, match="pair deletion"):
            _save(run_dir, step=3, keep=2)

    assert injected
    assert not first.sidecar_path.exists()
    assert first.checkpoint_path.exists()
    assert (first.sidecar_path.parent / ".step-000000001.json.pruning").exists()

    retried = _save(run_dir, step=3, keep=2)

    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 3
    assert not first.checkpoint_path.exists()
    assert not (first.sidecar_path.parent / ".step-000000001.json.pruning").exists()


def test_retention_marker_finishes_cleanup_after_tombstone_unlink_failure(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _save(run_dir, step=1, keep=3)
    _save(run_dir, step=2, keep=3)
    original_unlink = Path.unlink
    injected = False

    def fail_last_tombstone_unlink(path: Path, missing_ok: bool = False) -> None:
        nonlocal injected
        if path.name == ".step-000000001.pt.pruning" and not injected:
            injected = True
            raise OSError("injected tombstone unlink failure")
        original_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as context:
        context.setattr(Path, "unlink", fail_last_tombstone_unlink)
        with pytest.raises(OSError, match="tombstone unlink"):
            _save(run_dir, step=3, keep=2)

    assert injected
    assert not first.checkpoint_path.exists()
    assert not first.sidecar_path.exists()
    assert (first.checkpoint_path.parent / ".step-000000001.pt.pruning").exists()

    retried = _save(run_dir, step=3, keep=2)

    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 3
    assert not (first.checkpoint_path.parent / ".step-000000001.pt.pruning").exists()


@pytest.mark.parametrize(
    "limit_name",
    (
        "PHASE0_MAX_CHECKPOINT_BYTES",
        "PHASE0_MAX_ARCHIVE_MEMBERS",
        "PHASE0_MAX_ARCHIVE_UNCOMPRESSED_BYTES",
    ),
)
def test_artifact_budgets_reject_before_torch_load(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
) -> None:
    record = _save(run_dir)
    monkeypatch.setattr(checkpoint_module, limit_name, 1, raising=False)
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="Phase 0|archive|budget|size"):
        checkpoint_module._load_payload(record.checkpoint_path)

    assert not called


def test_compressed_archive_is_rejected_before_torch_load(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = run_dir / "step-000000001.pt"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("archive/data.pkl", b"not a torch payload")
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="archive|compression"):
        checkpoint_module._load_payload(path)

    assert not called


def test_control_json_budget_is_enforced_before_decode(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)
    monkeypatch.setattr(
        checkpoint_module, "PHASE0_MAX_CONTROL_JSON_BYTES", 1, raising=False
    )

    with pytest.raises(ValueError, match="JSON|size|budget"):
        checkpoint_module._read_json(
            record.checkpoint_path.parent / "run.json", "run metadata"
        )


@pytest.mark.parametrize("target_kind", ("artifact", "directory", "control"))
def test_symlinked_checkpoint_paths_fail_closed(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    record = _save(run_dir)
    checkpoint_dir = record.checkpoint_path.parent
    target = {
        "artifact": record.checkpoint_path,
        "directory": checkpoint_dir,
        "control": checkpoint_dir / "run.json",
    }[target_kind]
    original_is_symlink = Path.is_symlink

    def reported_symlink(path: Path) -> bool:
        return path == target or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", reported_symlink)
    model, optimizer, scheduler = _components()
    if target_kind == "artifact":
        operation = lambda: verify_checkpoint(record.checkpoint_path)
    elif target_kind == "control":
        operation = lambda: load_checkpoint(
            run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
            auto=True,
        )
    else:
        operation = lambda: _save(run_dir, step=2)

    with pytest.raises(ValueError, match="symlink"):
        operation()
