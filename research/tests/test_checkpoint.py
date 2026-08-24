import hashlib
import json
import random
import shutil
import uuid
from copy import deepcopy
from pathlib import Path
from collections.abc import Iterator

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
            "git_revision": "abc123",
            "files": {
                "model.py": "1" * 64,
                "train.py": "2" * 64,
                "synthetic.py": "3" * 64,
                "checkpoint.py": "4" * 64,
            },
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
):
    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)
    model, optimizer, scheduler = _components()
    batch_size = int(_manifests()["data"]["batch_size"])
    sequence_length = int(_manifests()["data"]["sequence_length"])
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
        semantic_manifests=_manifests(),
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
        "parameter_ledger",
        "counters",
        "last_metrics",
    }
    assert required <= payload.keys()
    assert payload["schema_version"] == 1
    assert payload["lineage_id"] == "lineage-test"
    assert payload["global_step"] == 4
    assert payload["micro_step"] == 0
    assert payload["samples_seen"] == 8
    assert payload["tokens_seen"] == 32
    assert set(payload["rng_state"]) == {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda_all",
    }
    assert isinstance(payload["rng_state"]["python"]["state"], torch.Tensor)
    assert isinstance(payload["rng_state"]["numpy"]["keys"], torch.Tensor)

    sidecar = json.loads(record.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["sha256"] == _sha256(record.checkpoint_path)
    assert sidecar["byte_size"] == record.checkpoint_path.stat().st_size
    assert sidecar["step"] == 4
    assert sidecar["lineage_id"] == "lineage-test"
    assert sidecar["manifest_hashes"] == payload["manifest_hashes"]

    pointer = json.loads(record.latest_path.read_text(encoding="utf-8"))
    assert pointer["checkpoint"] == record.checkpoint_path.name
    assert pointer["sidecar"] == record.sidecar_path.name
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
        lambda payload: payload.__setitem__("schema_version", 2),
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
            lambda manifest: manifest["files"].__setitem__("train.py", "f" * 64),
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
