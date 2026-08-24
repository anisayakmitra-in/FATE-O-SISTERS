"""Phase-0 CPU correctness checkpoints; this is not a 3B-scalable loader."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import random
import re
import stat
import struct
import tempfile
import zipfile
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import Lock, RLock, local
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from .ledger import ParameterLedger, ParameterSubsystem
from .model import MoiraModel
from .synthetic import DataCursor


CHECKPOINT_SCHEMA_VERSION = 2
PHASE0_MAX_CHECKPOINT_BYTES = 256 * 1024 * 1024
PHASE0_MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024 * 1024
PHASE0_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
PHASE0_MAX_ARCHIVE_MEMBERS = 8_192
PHASE0_MAX_CONTROL_JSON_BYTES = 1 * 1024 * 1024
PHASE0_MAX_DIRECTORY_ENTRIES = 16_384
PHASE0_MAX_STATE_DEPTH = 32
PHASE0_MAX_STATE_ENTRIES = 100_000
PHASE0_MAX_CONTAINER_ENTRIES = 16_384
PHASE0_MAX_TENSORS = 8_192
PHASE0_MAX_TENSOR_RANK = 8
PHASE0_MAX_TENSOR_STORAGE_BYTES = 256 * 1024 * 1024
PHASE0_MAX_TENSOR_DIMENSION = 1_000_000
PHASE0_MAX_TENSOR_NUMEL = 64 * 1024 * 1024
PHASE0_MAX_TENSOR_LOGICAL_BYTES = 256 * 1024 * 1024
PHASE0_MAX_TOTAL_TENSOR_LOGICAL_BYTES = 256 * 1024 * 1024
PHASE0_MAX_STRING_BYTES = 64 * 1024
PHASE0_MAX_CENTRAL_DIRECTORY_BYTES = 16 * 1024 * 1024
PHASE0_MAX_SCAN_CANDIDATES = 64
PHASE0_MAX_SCAN_BYTES = 3 * (
    PHASE0_MAX_CHECKPOINT_BYTES + PHASE0_MAX_CONTROL_JSON_BYTES
)
PHASE0_INACTIVE_PARAMETER_ALIASES = frozenset(
    {
        "pulse.appraisal_projection.weight",
        "pulse.appraisal_projection.bias",
        "pulse.fast_to_slow.weight",
        "pulse.fast_to_slow.bias",
    }
)
_PHASE0_TENSOR_DTYPES = frozenset(
    {
        torch.bool,
        torch.uint8,
        torch.int64,
        torch.float32,
    }
)
_STEP_NAME = re.compile(r"step-([0-9]{9})\.pt\Z", re.ASCII)
_SIDECAR_NAME = re.compile(r"step-[0-9]{9}\.json\Z", re.ASCII)
_PRUNING_MARKER_NAME = re.compile(
    r"\.step-([0-9]{9})\.pruning\.json\Z", re.ASCII
)
_INVALID_TOMBSTONE_NAME = re.compile(
    r"\.(step-[0-9]{9}\.(?:pt|json))\.invalid\Z", re.ASCII
)
_SIDECAR_KEYS = {
    "schema_version",
    "checkpoint",
    "sha256",
    "byte_size",
    "step",
    "lineage_id",
    "manifest_hashes",
}
_POINTER_KEYS = _SIDECAR_KEYS | {"sidecar"}
_RUN_METADATA_KEYS = {"schema_version", "lineage_id", "manifest_hashes"}
_SEMANTIC_MANIFEST_NAMES = (
    "model",
    "data",
    "optimization",
    "reproducibility",
    "source",
    "runtime",
)
_REQUIRED_PAYLOAD_KEYS = {
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
_RUN_LOCKS_GUARD = Lock()
_RUN_LOCKS: dict[str, RLock] = {}
_RUN_LOCK_LOCAL = local()


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    checkpoint_path: Path
    sidecar_path: Path
    latest_path: Path
    step: int
    lineage_id: str


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    checkpoint_path: Path
    lineage_id: str
    global_step: int
    data_cursor: DataCursor
    counters: dict[str, int]
    last_metrics: dict[str, float]


@contextmanager
def checkpoint_run_lock(
    run_dir: str | os.PathLike[str],
) -> Iterator[Path]:
    """Hold the permanent per-run Phase-0 writer lock, reentrantly in one thread."""
    with _checkpoint_directory_lock(Path(run_dir) / "checkpoints") as checkpoint_dir:
        yield checkpoint_dir


@contextmanager
def _checkpoint_directory_lock(checkpoint_dir: Path) -> Iterator[Path]:
    _reject_symlink_hops(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _validate_checkpoint_dir(checkpoint_dir)
    key = str(checkpoint_dir.resolve(strict=True))
    with _RUN_LOCKS_GUARD:
        process_lock = _RUN_LOCKS.setdefault(key, RLock())
    with process_lock:
        held = getattr(_RUN_LOCK_LOCAL, "held", {})
        entry = held.get(key)
        if entry is not None:
            held[key] = (entry[0] + 1, entry[1])
            _RUN_LOCK_LOCAL.held = held
            try:
                yield checkpoint_dir
            finally:
                depth, descriptor = held[key]
                held[key] = (depth - 1, descriptor)
            return

        descriptor = _acquire_run_lock(checkpoint_dir)
        held[key] = (1, descriptor)
        _RUN_LOCK_LOCAL.held = held
        try:
            yield checkpoint_dir
        finally:
            del held[key]
            _release_run_lock(descriptor)


def _acquire_run_lock(checkpoint_dir: Path) -> int:
    lock_path = checkpoint_dir / ".run.lock"
    _reject_symlink_hops(lock_path)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise ValueError("checkpoint writer lock cannot be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        named = lock_path.lstat()
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(opened, named):
            raise ValueError("checkpoint writer lock is not a stable regular file")
        if opened.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
            _fsync_directory(checkpoint_dir)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        os.close(descriptor)
        if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            raise ValueError("checkpoint writer lock is held by another process") from error
        raise
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _release_run_lock(descriptor: int) -> None:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def save_checkpoint(
    run_dir: str | os.PathLike[str],
    *,
    lineage_id: str,
    global_step: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: object | None,
    data_cursor: DataCursor,
    semantic_manifests: Mapping[str, Mapping[str, object]],
    provenance: Mapping[str, object],
    parameter_ledger: ParameterLedger,
    counters: Mapping[str, int],
    last_metrics: Mapping[str, float],
    keep_last_checkpoints: int,
) -> CheckpointRecord:
    """Commit one immutable optimizer-boundary checkpoint and then its pointer."""
    _require_non_empty_string("lineage_id", lineage_id)
    _require_positive_integer("global_step", global_step)
    if global_step > 999_999_999:
        raise ValueError("global_step is outside the Phase 0 canonical range")
    _require_positive_integer("keep_last_checkpoints", keep_last_checkpoints)
    if not isinstance(model, MoiraModel):
        raise ValueError("Phase 0 checkpoints require a MoiraModel")
    if type(optimizer) is not AdamW:
        raise ValueError("Phase 0 checkpoints require AdamW")
    if type(scheduler) is not LambdaLR:
        raise ValueError("Phase 0 checkpoints require LambdaLR")
    if not isinstance(data_cursor, DataCursor):
        raise ValueError("data_cursor must be a DataCursor")
    if scaler is not None:
        raise ValueError("Phase 0 checkpoints require scaler=None")
    if not isinstance(parameter_ledger, ParameterLedger):
        raise ValueError("parameter_ledger must be a ParameterLedger")
    if parameter_ledger._model is not model:
        raise ValueError("parameter ledger must belong to the checkpoint model")
    parameter_ledger.assert_matches()

    manifests = _normalize_manifests(semantic_manifests)
    manifest_hashes = _manifest_hashes(manifests)
    normalized_provenance = _normalize_provenance(provenance)
    model_state = dict(model.state_dict())
    ledger_evidence = _ledger_evidence(parameter_ledger, model_state)
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "lineage_id": lineage_id,
        "global_step": global_step,
        "micro_step": 0,
        "samples_seen": data_cursor.samples_seen,
        "tokens_seen": data_cursor.tokens_seen,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": None,
        "rng_state": _capture_rng_state(),
        "data_cursor": {
            "kind": data_cursor.kind,
            "next_sample_index": data_cursor.next_sample_index,
            "samples_seen": data_cursor.samples_seen,
            "tokens_seen": data_cursor.tokens_seen,
        },
        "semantic_manifests": manifests,
        "manifest_hashes": manifest_hashes,
        "provenance": normalized_provenance,
        "parameter_ledger": ledger_evidence,
        "counters": _normalize_counters(counters),
        "last_metrics": _normalize_metrics(last_metrics),
    }
    _validate_payload(payload, expected_manifests=manifests)
    _validate_phase0_component_binding(model, optimizer, scheduler, payload, manifests)

    with checkpoint_run_lock(run_dir) as checkpoint_dir:
        return _commit_checkpoint(
            checkpoint_dir,
            lineage_id=lineage_id,
            global_step=global_step,
            payload=payload,
            manifests=manifests,
            manifest_hashes=manifest_hashes,
            keep_last_checkpoints=keep_last_checkpoints,
        )


def _commit_checkpoint(
    checkpoint_dir: Path,
    *,
    lineage_id: str,
    global_step: int,
    payload: Mapping[str, object],
    manifests: Mapping[str, Mapping[str, object]],
    manifest_hashes: Mapping[str, str],
    keep_last_checkpoints: int,
) -> CheckpointRecord:
    run_metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "lineage_id": lineage_id,
        "manifest_hashes": manifest_hashes,
    }
    _bind_run_metadata(checkpoint_dir, run_metadata)
    checkpoint_path = checkpoint_dir / f"step-{global_step:09d}.pt"
    sidecar_path = checkpoint_path.with_suffix(".json")
    latest_path = checkpoint_dir / "latest.json"
    _validate_previous_pointer(latest_path, lineage_id, global_step)

    staged_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            dir=checkpoint_dir,
            delete=False,
        ) as temporary:
            staged_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        reopened = _load_payload(staged_path)
        _validate_payload(reopened, expected_manifests=manifests)
        if not _values_equal(payload, reopened):
            raise ValueError("staged checkpoint did not reopen exactly")
        staged_size = staged_path.stat().st_size
        staged_checksum = _sha256(staged_path)
        claimed = _claim_file(staged_path, checkpoint_path)
        if not claimed:
            try:
                existing = _load_payload(checkpoint_path)
                _validate_payload(existing, expected_manifests=manifests)
            except ValueError:
                _reclaim_unreferenced(
                    (sidecar_path, checkpoint_path), latest_path, checkpoint_path.name
                )
                if not _claim_file(staged_path, checkpoint_path):
                    raise ValueError("checkpoint step was claimed concurrently")
            else:
                if not _same_checkpoint_state(payload, existing):
                    raise ValueError("checkpoint step has different committed state")
        else:
            if checkpoint_path.stat().st_size != staged_size:
                raise ValueError("published checkpoint size changed")
            if _sha256(checkpoint_path) != staged_checksum:
                raise ValueError("published checkpoint checksum changed")
    finally:
        if staged_path is not None:
            try:
                staged_path.unlink(missing_ok=True)
            except OSError:
                pass

    byte_size = checkpoint_path.stat().st_size
    checksum = _sha256(checkpoint_path)
    reopened = _load_payload(checkpoint_path)
    _validate_payload(reopened, expected_manifests=manifests)
    sidecar = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint": checkpoint_path.name,
        "sha256": checksum,
        "byte_size": byte_size,
        "step": global_step,
        "lineage_id": lineage_id,
        "manifest_hashes": manifest_hashes,
    }
    if sidecar_path.exists():
        try:
            existing_sidecar = _read_json(sidecar_path, "existing checkpoint sidecar")
        except ValueError:
            _reclaim_unreferenced(
                (sidecar_path,), latest_path, checkpoint_path.name
            )
        else:
            if set(existing_sidecar) != _SIDECAR_KEYS:
                _reclaim_unreferenced(
                    (sidecar_path,), latest_path, checkpoint_path.name
                )
            elif existing_sidecar != sidecar:
                raise ValueError("checkpoint sidecar has different committed content")
    _publish_json_once(sidecar_path, sidecar)
    reopened_sidecar = _read_json(sidecar_path, "checkpoint sidecar")
    _validate_sidecar_integrity(reopened_sidecar, checkpoint_path)
    _validate_sidecar_binding(reopened_sidecar, checkpoint_path, reopened)

    pointer = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint": checkpoint_path.name,
        "sidecar": sidecar_path.name,
        "step": global_step,
        "lineage_id": lineage_id,
        "manifest_hashes": manifest_hashes,
        "sha256": checksum,
        "byte_size": byte_size,
    }
    existing_pointer = None
    if latest_path.exists():
        existing_pointer = _read_json(latest_path, "latest pointer")
    if existing_pointer != pointer:
        _atomic_write_json(latest_path, pointer)
    if _read_json(latest_path, "latest pointer") != pointer:
        raise ValueError("latest pointer did not reopen exactly")

    record = CheckpointRecord(
        checkpoint_path=checkpoint_path,
        sidecar_path=sidecar_path,
        latest_path=latest_path,
        step=global_step,
        lineage_id=lineage_id,
    )
    _prune_validated(checkpoint_dir, keep_last_checkpoints)
    return record


def verify_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    expected_manifests: Mapping[str, Mapping[str, object]] | None = None,
    expected_lineage_id: str | None = None,
) -> dict[str, Any]:
    """Verify the immutable artifact, sidecar, schema, and semantic evidence."""
    path = Path(checkpoint_path)
    _validate_regular_file(path, "checkpoint artifact")
    sidecar_path = path.with_suffix(".json")
    sidecar = _read_json(sidecar_path, "checkpoint sidecar")
    _validate_sidecar_integrity(sidecar, path)
    payload = _load_payload(path)
    _validate_payload(payload, expected_manifests=expected_manifests)
    _validate_sidecar_binding(sidecar, path, payload)
    if expected_lineage_id is not None:
        _require_non_empty_string("expected_lineage_id", expected_lineage_id)
        if payload["lineage_id"] != expected_lineage_id:
            raise ValueError("checkpoint lineage mismatch")
    return payload


def load_checkpoint(
    source: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: object | None,
    expected_manifests: Mapping[str, Mapping[str, object]],
    expected_lineage_id: str | None = None,
    auto: bool = False,
) -> LoadedCheckpoint:
    source_path = Path(source)
    checkpoint_dir = source_path / "checkpoints" if auto else source_path.parent
    with _checkpoint_directory_lock(checkpoint_dir):
        return _load_checkpoint_locked(
            source_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_manifests=expected_manifests,
            expected_lineage_id=expected_lineage_id,
            auto=auto,
        )


def _load_checkpoint_locked(
    source: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: object | None,
    expected_manifests: Mapping[str, Mapping[str, object]],
    expected_lineage_id: str | None,
    auto: bool,
) -> LoadedCheckpoint:
    """Strictly restore components and restore process RNG state last."""
    normalized_expected = _normalize_manifests(expected_manifests)
    checkpoint_path, payload, _ = _resolve_checkpoint(
        Path(source),
        auto=auto,
        expected_manifests=normalized_expected,
        expected_lineage_id=expected_lineage_id,
    )
    _validate_model_compatibility(model, payload)
    if scaler is not None:
        raise ValueError("Phase 0 checkpoints require scaler=None")
    _validate_phase0_component_binding(
        model, optimizer, scheduler, payload, normalized_expected
    )
    _preflight_components(model, optimizer, scheduler, payload)
    cursor = _cursor_from_payload(payload)
    counters = dict(payload["counters"])
    last_metrics = {
        str(name): float(value) for name, value in payload["last_metrics"].items()
    }
    component_state = (
        deepcopy(model.state_dict()),
        deepcopy(optimizer.state_dict()),
        deepcopy(scheduler.state_dict()),
    )
    rng_before = _capture_rng_state()
    try:
        model.load_state_dict(payload["model_state"], strict=True)
        scheduler.load_state_dict(payload["scheduler_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        _restore_rng_state(payload["rng_state"])
    except Exception:
        nn.Module.load_state_dict(model, component_state[0], strict=True)
        LRScheduler.load_state_dict(scheduler, component_state[2])
        Optimizer.load_state_dict(optimizer, component_state[1])
        _restore_rng_state(rng_before)
        raise
    return LoadedCheckpoint(
        checkpoint_path=checkpoint_path,
        lineage_id=payload["lineage_id"],
        global_step=payload["global_step"],
        data_cursor=cursor,
        counters=counters,
        last_metrics=last_metrics,
    )


def _normalize_manifests(
    manifests: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if not isinstance(manifests, Mapping) or set(manifests) != set(
        _SEMANTIC_MANIFEST_NAMES
    ):
        raise ValueError("semantic manifests must contain the approved sections")
    normalized: dict[str, dict[str, object]] = {}
    for name in _SEMANTIC_MANIFEST_NAMES:
        value = manifests[name]
        if not isinstance(value, Mapping):
            raise ValueError(f"semantic manifest {name} must be a mapping")
        try:
            normalized[name] = json.loads(_canonical_json(value).decode("utf-8"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"semantic manifest {name} is not JSON-compatible") from error
    source = normalized["source"]
    if not source or any(not isinstance(path, str) or not path for path in source):
        raise ValueError("strict source manifest must contain package file hashes")
    for digest in source.values():
        _validate_sha256(digest, "strict source manifest")
    return normalized


def _manifest_hashes(manifests: Mapping[str, Mapping[str, object]]) -> dict[str, str]:
    return {
        name: hashlib.sha256(_canonical_json(manifests[name])).hexdigest()
        for name in _SEMANTIC_MANIFEST_NAMES
    }


def _validate_manifest_hashes(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_SEMANTIC_MANIFEST_NAMES):
        raise ValueError("strict manifest hash schema is invalid")
    for digest in value.values():
        _validate_sha256(digest, "manifest")


def _validate_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} sha256 is invalid")


def _normalize_provenance(provenance: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint provenance must be a mapping")
    try:
        normalized = json.loads(_canonical_json(provenance).decode("utf-8"))
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint provenance is not JSON-compatible") from error
    if not isinstance(normalized, dict):
        raise ValueError("checkpoint provenance must be a mapping")
    git = normalized.get("git")
    if (
        set(normalized) != {"git"}
        or not isinstance(git, dict)
        or set(git) != {"available", "revision", "dirty"}
        or not isinstance(git["available"], bool)
    ):
        raise ValueError("checkpoint provenance schema is invalid")
    if git["available"]:
        if (
            not isinstance(git["revision"], str)
            or not git["revision"]
            or not isinstance(git["dirty"], bool)
        ):
            raise ValueError("checkpoint Git provenance is invalid")
    elif git["revision"] is not None or git["dirty"] is not None:
        raise ValueError("unavailable Git provenance must contain null context")
    return normalized


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _capture_rng_state() -> dict[str, object]:
    python_version, python_values, python_gauss = random.getstate()
    numpy_name, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached = (
        np.random.get_state()
    )
    return {
        "python": {
            "version": python_version,
            "state": torch.tensor(python_values, dtype=torch.int64),
            "gauss": python_gauss,
        },
        "numpy": {
            "bit_generator": numpy_name,
            "keys": torch.from_numpy(numpy_keys.astype(np.int64, copy=True)),
            "position": int(numpy_position),
            "has_gauss": int(numpy_has_gauss),
            "cached_gaussian": float(numpy_cached),
        },
        "torch_cpu": torch.get_rng_state().clone(),
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    python_state = state["python"]
    assert isinstance(python_state, Mapping)
    python_values = python_state["state"]
    assert isinstance(python_values, Tensor)
    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(value) for value in python_values.tolist()),
            python_state["gauss"],
        )
    )
    numpy_state = state["numpy"]
    assert isinstance(numpy_state, Mapping)
    numpy_keys = numpy_state["keys"]
    assert isinstance(numpy_keys, Tensor)
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_keys.numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch_cpu = state["torch_cpu"]
    assert isinstance(torch_cpu, Tensor)
    torch.set_rng_state(torch_cpu)


def _validate_payload(
    payload: object,
    *,
    expected_manifests: Mapping[str, Mapping[str, object]] | None,
) -> None:
    if not isinstance(payload, dict) or set(payload) != _REQUIRED_PAYLOAD_KEYS:
        raise ValueError("checkpoint payload schema is invalid")
    _validate_state_tree(payload)
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version is invalid")
    _require_non_empty_string("lineage_id", payload["lineage_id"])
    _require_positive_integer("global_step", payload["global_step"])
    if payload["global_step"] > 999_999_999:
        raise ValueError("checkpoint step is outside the canonical range")
    if payload["micro_step"] != 0:
        raise ValueError("checkpoint micro step must be zero")
    _require_non_negative_integer("samples_seen", payload["samples_seen"])
    _require_non_negative_integer("tokens_seen", payload["tokens_seen"])

    manifests = _normalize_manifests(payload["semantic_manifests"])
    observed_hashes = _manifest_hashes(manifests)
    if payload["manifest_hashes"] != observed_hashes:
        raise ValueError("checkpoint manifest hashes do not match their content")
    _normalize_provenance(payload["provenance"])
    if expected_manifests is not None:
        normalized_expected = _normalize_manifests(expected_manifests)
        if observed_hashes != _manifest_hashes(normalized_expected):
            raise ValueError("checkpoint strict manifest mismatch")

    cursor = _cursor_from_payload(payload)
    if (
        cursor.samples_seen != payload["samples_seen"]
        or cursor.tokens_seen != payload["tokens_seen"]
    ):
        raise ValueError("checkpoint cursor mismatch")
    _validate_training_counters(payload, manifests, cursor)
    _validate_model_state_and_ledger(payload)
    if not isinstance(payload["optimizer_state"], dict):
        raise ValueError("optimizer state must be a mapping")
    if not isinstance(payload["scheduler_state"], dict):
        raise ValueError("scheduler state must be a mapping")
    if payload["scaler_state"] is not None:
        raise ValueError("Phase 0 checkpoint scaler state must be null")
    metrics = _normalize_metrics(payload["last_metrics"])
    _validate_phase0_saved_progress(payload, manifests, metrics)
    _validate_rng_state(payload["rng_state"])


def _validate_training_counters(
    payload: Mapping[str, object],
    manifests: Mapping[str, Mapping[str, object]],
    cursor: DataCursor,
) -> None:
    counters = _normalize_counters(payload["counters"])
    step = int(payload["global_step"])
    if counters.get("optimizer_steps") != step or counters.get("batches_seen") != step:
        raise ValueError("checkpoint counters do not match the global step")
    data = manifests["data"]
    optimization = manifests["optimization"]
    batch_size = data.get("batch_size")
    sequence_length = data.get("sequence_length")
    accumulation = optimization.get("gradient_accumulation_steps")
    _require_positive_integer("manifest batch_size", batch_size)
    _require_positive_integer("manifest sequence_length", sequence_length)
    _require_positive_integer("manifest gradient accumulation", accumulation)
    expected_samples = step * int(batch_size) * int(accumulation)
    if cursor.samples_seen != expected_samples:
        raise ValueError("checkpoint cursor sample counter is inconsistent")
    if cursor.tokens_seen != expected_samples * int(sequence_length):
        raise ValueError("checkpoint cursor token counter is inconsistent")


def _cursor_from_payload(payload: Mapping[str, object]) -> DataCursor:
    value = payload.get("data_cursor")
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint data cursor must be a mapping")
    try:
        return DataCursor(
            kind=value["kind"],
            next_sample_index=value["next_sample_index"],
            samples_seen=value["samples_seen"],
            tokens_seen=value["tokens_seen"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint cursor mismatch") from error


def _ledger_evidence(
    ledger: ParameterLedger, model_state: Mapping[str, Tensor]
) -> dict[str, object]:
    return {
        "total_trainable": ledger.total_trainable,
        "entries": [
            {
                "aliases": list(entry.aliases),
                "subsystem": entry.subsystem.value,
                "trainable_parameters": entry.trainable_parameters,
            }
            for entry in ledger.entries
        ],
        "model_state_schema": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
            }
            for name, tensor in sorted(model_state.items())
        },
    }


def _validate_model_state_and_ledger(payload: Mapping[str, object]) -> None:
    state = payload["model_state"]
    evidence = payload["parameter_ledger"]
    if (
        not isinstance(state, dict)
        or not state
        or not isinstance(evidence, dict)
        or set(evidence) != {"total_trainable", "entries", "model_state_schema"}
    ):
        raise ValueError("model state and parameter ledger must be mappings")
    schema = evidence.get("model_state_schema")
    entries = evidence.get("entries")
    total = evidence.get("total_trainable")
    subsystems = {subsystem.value for subsystem in ParameterSubsystem}
    valid_entries = isinstance(entries, list) and bool(entries) and all(
        isinstance(entry, dict)
        and set(entry) == {"aliases", "subsystem", "trainable_parameters"}
        and isinstance(entry.get("aliases"), list)
        and bool(entry["aliases"])
        and all(isinstance(alias, str) and alias for alias in entry["aliases"])
        and entry["aliases"] == sorted(set(entry["aliases"]))
        and entry.get("subsystem") in subsystems
        and _is_positive_integer(entry.get("trainable_parameters"))
        for entry in entries or []
    )
    aliases = [
        alias
        for entry in entries or []
        if isinstance(entry, dict)
        for alias in entry.get("aliases", [])
    ]
    if (
        not isinstance(schema, dict)
        or set(schema) != set(state)
        or not _is_positive_integer(total)
        or not valid_entries
        or len(aliases) != len(set(aliases))
        or sum(entry["trainable_parameters"] for entry in entries) != total
    ):
        raise ValueError("model-state mismatch with parameter ledger evidence")
    for name, tensor in state.items():
        expected = schema[name]
        if (
            not isinstance(name, str)
            or not isinstance(tensor, Tensor)
            or not isinstance(expected, dict)
            or set(expected) != {"shape", "dtype", "numel"}
            or expected.get("shape") != list(tensor.shape)
            or expected.get("dtype") != str(tensor.dtype)
            or expected.get("numel") != tensor.numel()
        ):
            raise ValueError("model-state mismatch with parameter ledger evidence")


def _validate_model_compatibility(model: nn.Module, payload: Mapping[str, object]) -> None:
    if not isinstance(model, MoiraModel):
        raise ValueError("Phase 0 checkpoints require a MoiraModel")
    expected = model.state_dict()
    observed = payload["model_state"]
    if not isinstance(observed, Mapping) or set(expected) != set(observed):
        raise ValueError("model-state mismatch with the constructed model")
    for name, tensor in expected.items():
        saved = observed[name]
        if tensor.device.type != "cpu" or not isinstance(saved, Tensor) or (
            saved.shape != tensor.shape or saved.dtype != tensor.dtype
        ):
            raise ValueError("model-state mismatch with the constructed model")
    actual_ledger = _ledger_evidence(model.parameter_ledger(), expected)
    if not _values_equal(actual_ledger, payload["parameter_ledger"]):
        raise ValueError("parameter ledger does not match the constructed model")


def _validate_phase0_saved_progress(
    payload: Mapping[str, object],
    manifests: Mapping[str, Mapping[str, object]],
    metrics: Mapping[str, float],
) -> None:
    step = int(payload["global_step"])
    expected_lr, base_lr, weight_decay = _phase0_learning_rates(manifests, step)
    optimizer_state = payload["optimizer_state"]
    scheduler_state = payload["scheduler_state"]
    if (
        not isinstance(optimizer_state, dict)
        or set(optimizer_state) != {"state", "param_groups"}
        or not isinstance(optimizer_state["param_groups"], list)
        or len(optimizer_state["param_groups"]) != 1
        or not isinstance(optimizer_state["state"], dict)
        or not optimizer_state["state"]
    ):
        raise ValueError("Phase 0 AdamW optimizer state is incomplete")
    group = optimizer_state["param_groups"][0]
    expected_group_keys = {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
        "initial_lr",
        "params",
    }
    if not isinstance(group, dict) or set(group) != expected_group_keys:
        raise ValueError("Phase 0 AdamW parameter group schema is invalid")
    parameters = group["params"]
    state = optimizer_state["state"]
    if (
        not isinstance(parameters, list)
        or not parameters
        or len(parameters) != len(set(parameters))
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in parameters)
        or not set(state) <= set(parameters)
        or type(group["betas"]) is not tuple
        or group["betas"] != (0.9, 0.999)
        or type(group["eps"]) is not float
        or group["eps"] != 1e-8
        or group["amsgrad"] is not False
        or group["foreach"] is not False
        or group["fused"] is not False
        or group["maximize"] is not False
        or group["capturable"] is not False
        or group["differentiable"] is not False
        or group["decoupled_weight_decay"] is not True
        or type(group["initial_lr"]) is not float
        or type(group["weight_decay"]) is not float
        or type(group["lr"]) is not float
        or not _same_float(group["initial_lr"], base_lr)
        or not _same_float(group["weight_decay"], weight_decay)
        or not _same_float(group["lr"], expected_lr)
    ):
        raise ValueError("Phase 0 AdamW parameter group is incompatible")
    for parameter_state in state.values():
        if not isinstance(parameter_state, dict) or set(parameter_state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise ValueError("Phase 0 AdamW parameter state schema is invalid")
        saved_step = parameter_state["step"]
        exp_avg = parameter_state["exp_avg"]
        exp_avg_sq = parameter_state["exp_avg_sq"]
        if (
            not isinstance(saved_step, Tensor)
            or saved_step.device.type != "cpu"
            or saved_step.dtype is not torch.float32
            or saved_step.layout is not torch.strided
            or saved_step.ndim != 0
            or float(saved_step.item()) != step
            or not isinstance(exp_avg, Tensor)
            or not isinstance(exp_avg_sq, Tensor)
            or exp_avg.shape != exp_avg_sq.shape
            or exp_avg.dtype is not torch.float32
            or exp_avg_sq.dtype is not torch.float32
            or exp_avg.device.type != "cpu"
            or exp_avg_sq.device.type != "cpu"
            or exp_avg.layout is not torch.strided
            or exp_avg_sq.layout is not torch.strided
        ):
            raise ValueError("Phase 0 AdamW progress is incompatible")

    scheduler_keys = {
        "base_lrs",
        "last_epoch",
        "_step_count",
        "_is_initial",
        "_get_lr_called_within_step",
        "_last_lr",
        "lr_lambdas",
    }
    if not isinstance(scheduler_state, dict) or set(scheduler_state) != scheduler_keys:
        raise ValueError("Phase 0 LambdaLR state schema is invalid")
    if (
        type(scheduler_state["last_epoch"]) is not int
        or type(scheduler_state["_step_count"]) is not int
        or scheduler_state["last_epoch"] != step
        or scheduler_state["_step_count"] != step + 1
        or scheduler_state["_is_initial"] is not False
        or scheduler_state["_get_lr_called_within_step"] is not False
        or scheduler_state["lr_lambdas"] != [None]
        or not _single_matching_float(scheduler_state["base_lrs"], base_lr)
        or not _single_matching_float(scheduler_state["_last_lr"], expected_lr)
        or not _same_float(metrics.get("learning_rate"), expected_lr)
    ):
        raise ValueError("Phase 0 LambdaLR progress is inconsistent with global_step")


def _validate_phase0_component_binding(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    payload: Mapping[str, object],
    manifests: Mapping[str, Mapping[str, object]],
) -> None:
    if not isinstance(model, MoiraModel) or type(optimizer) is not AdamW:
        raise ValueError("Phase 0 requires MoiraModel with AdamW")
    if type(scheduler) is not LambdaLR or scheduler.optimizer is not optimizer:
        raise ValueError("Phase 0 requires an optimizer-bound LambdaLR")
    if len(optimizer.param_groups) != 1:
        raise ValueError("Phase 0 AdamW requires one parameter group")
    actual_parameters = list(optimizer.param_groups[0]["params"])
    model_parameters = list(model.parameters())
    if [id(item) for item in actual_parameters] != [id(item) for item in model_parameters]:
        raise ValueError("Phase 0 AdamW parameters are not bound to the model")
    runtime_group = optimizer.param_groups[0]
    if (
        runtime_group.get("betas") != (0.9, 0.999)
        or runtime_group.get("eps") != 1e-8
        or runtime_group.get("foreach") is not False
        or runtime_group.get("fused") is not False
        or runtime_group.get("amsgrad") is not False
        or runtime_group.get("maximize") is not False
        or runtime_group.get("capturable") is not False
        or runtime_group.get("differentiable") is not False
        or runtime_group.get("decoupled_weight_decay") is not True
    ):
        raise ValueError("Phase 0 AdamW runtime options are incompatible")

    optimizer_state = payload["optimizer_state"]
    saved_group = optimizer_state["param_groups"][0]
    live_group = optimizer.state_dict()["param_groups"][0]
    if saved_group["params"] != live_group["params"]:
        raise ValueError("saved AdamW parameters do not bind to the constructed model")
    aliases_by_parameter: dict[int, set[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases_by_parameter.setdefault(id(parameter), set()).add(name)
    all_aliases = set().union(*aliases_by_parameter.values())
    if not PHASE0_INACTIVE_PARAMETER_ALIASES <= all_aliases:
        raise ValueError("Phase 0 inactive parameter alias contract drifted")
    expected_active = {
        saved_id
        for saved_id, parameter in zip(saved_group["params"], actual_parameters)
        if aliases_by_parameter[id(parameter)].isdisjoint(
            PHASE0_INACTIVE_PARAMETER_ALIASES
        )
    }
    saved_state = optimizer_state["state"]
    if set(saved_state) != expected_active:
        raise ValueError("Phase 0 AdamW active parameter coverage is incomplete")
    for saved_id, parameter in zip(saved_group["params"], actual_parameters):
        if saved_id not in expected_active:
            continue
        parameter_state = saved_state[saved_id]
        for name in ("exp_avg", "exp_avg_sq"):
            value = parameter_state[name]
            if (
                value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or value.device.type != "cpu"
                or value.layout is not torch.strided
            ):
                raise ValueError("AdamW tensor state does not match its model parameter")

    expected_lr, _, _ = _phase0_learning_rates(manifests, int(payload["global_step"]))
    if not _same_float(saved_group["lr"], expected_lr):
        raise ValueError("AdamW learning rate is inconsistent with the manifest")


def _phase0_learning_rates(
    manifests: Mapping[str, Mapping[str, object]], step: int
) -> tuple[float, float, float]:
    optimization = manifests["optimization"]
    base_lr = optimization.get("learning_rate")
    weight_decay = optimization.get("weight_decay")
    warmup = optimization.get("warmup_steps")
    total = optimization.get("schedule_total_steps")
    if (
        optimization.get("optimizer") != "adamw"
        or optimization.get("schedule") != "cosine-with-warmup"
        or optimization.get("precision") != "fp32"
        or optimization.get("gradient_accumulation_steps") != 1
        or not _is_finite_number(base_lr, positive=True)
        or not _is_finite_number(weight_decay, non_negative=True)
        or not isinstance(warmup, int)
        or isinstance(warmup, bool)
        or warmup < 0
        or not _is_positive_integer(total)
        or step > total
    ):
        raise ValueError("Phase 0 optimization manifest is incompatible")
    if warmup and step < warmup:
        factor = float(step + 1) / warmup
    else:
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(max(progress, 0.0), 1.0)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * factor, float(base_lr), float(weight_decay)


def _single_matching_float(value: object, expected: float) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 1
        and type(value[0]) is float
        and _same_float(value[0], expected)
    )


def _same_float(value: object, expected: float) -> bool:
    return _is_finite_number(value) and math.isclose(
        float(value), expected, rel_tol=0.0, abs_tol=1e-15
    )


def _is_finite_number(
    value: object, *, positive: bool = False, non_negative: bool = False
) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    number = float(value)
    return (
        math.isfinite(number)
        and (not positive or number > 0.0)
        and (not non_negative or number >= 0.0)
    )


def _validate_state_tree(value: object) -> None:
    stack = [(value, 0)]
    entries = tensors = tensor_storage_bytes = tensor_logical_bytes = 0
    storages: set[tuple[int, int]] = set()
    containers: set[int] = set()
    while stack:
        current, depth = stack.pop()
        entries += 1
        if entries > PHASE0_MAX_STATE_ENTRIES or depth > PHASE0_MAX_STATE_DEPTH:
            raise ValueError("checkpoint state exceeds the Phase 0 structure budget")
        if isinstance(current, Tensor):
            tensors += 1
            logical_bytes = current.numel() * current.element_size()
            if (
                tensors > PHASE0_MAX_TENSORS
                or current.ndim > PHASE0_MAX_TENSOR_RANK
                or any(size > PHASE0_MAX_TENSOR_DIMENSION for size in current.shape)
                or current.numel() > PHASE0_MAX_TENSOR_NUMEL
                or logical_bytes > PHASE0_MAX_TENSOR_LOGICAL_BYTES
                or current.dtype not in _PHASE0_TENSOR_DTYPES
                or current.device.type != "cpu"
                or current.layout is not torch.strided
                or any(
                    size > 1 and stride == 0
                    for size, stride in zip(current.shape, current.stride())
                )
            ):
                raise ValueError("checkpoint tensor state is outside Phase 0 bounds")
            tensor_logical_bytes += logical_bytes
            if tensor_logical_bytes > PHASE0_MAX_TOTAL_TENSOR_LOGICAL_BYTES:
                raise ValueError("checkpoint tensor logical state exceeds Phase 0 bounds")
            storage = current.untyped_storage()
            storage_key = (storage.data_ptr(), storage.nbytes())
            if storage_key not in storages:
                storages.add(storage_key)
                tensor_storage_bytes += storage.nbytes()
            if tensor_storage_bytes > PHASE0_MAX_TENSOR_STORAGE_BYTES:
                raise ValueError("checkpoint tensor storage exceeds Phase 0 bounds")
            if current.is_floating_point() and not bool(torch.isfinite(current).all()):
                raise ValueError("checkpoint tensor state must be finite")
        elif type(current) in (dict, list, tuple):
            if id(current) in containers:
                raise ValueError("checkpoint state contains a repeated container")
            containers.add(id(current))
            if len(current) > PHASE0_MAX_CONTAINER_ENTRIES:
                raise ValueError("checkpoint container exceeds the Phase 0 entry budget")
            if isinstance(current, dict):
                for key, item in current.items():
                    if not isinstance(key, (str, int)) or isinstance(key, bool):
                        raise ValueError(
                            "checkpoint mapping keys must be strings or integers"
                        )
                    if isinstance(key, str) and len(key.encode("utf-8")) > PHASE0_MAX_STRING_BYTES:
                        raise ValueError("checkpoint mapping key exceeds the string budget")
                    stack.append((item, depth + 1))
            else:
                stack.extend((item, depth + 1) for item in current)
        elif isinstance(current, str):
            if len(current.encode("utf-8")) > PHASE0_MAX_STRING_BYTES:
                raise ValueError("checkpoint string exceeds the Phase 0 budget")
        elif isinstance(current, float):
            if not math.isfinite(current):
                raise ValueError("checkpoint floating state must be finite")
        elif current is not None and not isinstance(current, (int, bool)):
            raise ValueError("checkpoint state contains an unsupported value")


def _preflight_components(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    payload: Mapping[str, object],
) -> None:
    cloned_model, cloned_optimizer, cloned_scheduler = deepcopy(
        (model, optimizer, scheduler)
    )
    cloned_model.load_state_dict(payload["model_state"], strict=True)
    cloned_scheduler.load_state_dict(payload["scheduler_state"])
    cloned_optimizer.load_state_dict(payload["optimizer_state"])


def _validate_rng_state(state: object) -> None:
    if not isinstance(state, Mapping) or set(state) != {
        "python",
        "numpy",
        "torch_cpu",
    }:
        raise ValueError("RNG state is incomplete")
    python_state = state["python"]
    numpy_state = state["numpy"]
    if (
        not isinstance(python_state, Mapping)
        or set(python_state) != {"version", "state", "gauss"}
        or python_state.get("version") != 3
        or not _is_vector(python_state.get("state"), torch.int64, length=625)
        or not (
            python_state.get("gauss") is None
            or isinstance(python_state.get("gauss"), float)
            and math.isfinite(python_state["gauss"])
        )
    ):
        raise ValueError("Python RNG state is invalid")
    if (
        not isinstance(numpy_state, Mapping)
        or set(numpy_state)
        != {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"}
        or numpy_state.get("bit_generator") != "MT19937"
        or not _is_vector(numpy_state.get("keys"), torch.int64, length=624)
        or not isinstance(numpy_state.get("position"), int)
        or isinstance(numpy_state.get("position"), bool)
        or not 0 <= numpy_state["position"] <= 624
        or numpy_state.get("has_gauss") not in (0, 1)
        or not isinstance(numpy_state.get("cached_gaussian"), float)
        or not math.isfinite(numpy_state["cached_gaussian"])
    ):
        raise ValueError("NumPy RNG state is invalid")
    torch_cpu = state["torch_cpu"]
    expected_torch_length = torch.Generator(device="cpu").get_state().numel()
    if not _is_vector(torch_cpu, torch.uint8, length=expected_torch_length):
        raise ValueError("Torch CPU RNG state is invalid")
    python_values = python_state["state"]
    numpy_keys = numpy_state["keys"]
    assert isinstance(python_values, Tensor)
    assert isinstance(numpy_keys, Tensor)
    if (
        not bool(((python_values[:-1] >= 0) & (python_values[:-1] <= 0xFFFFFFFF)).all())
        or not 0 <= int(python_values[-1]) <= 624
        or not bool(((numpy_keys >= 0) & (numpy_keys <= 0xFFFFFFFF)).all())
    ):
        raise ValueError("RNG integer state is out of range")
    try:
        python_probe = random.Random()
        python_probe.setstate(
            (
                3,
                tuple(int(value) for value in python_values.tolist()),
                python_state["gauss"],
            )
        )
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(
            (
                "MT19937",
                numpy_keys.numpy().astype(np.uint32, copy=True),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch_probe = torch.Generator(device="cpu")
        torch_probe.set_state(torch_cpu)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("RNG state cannot be restored locally") from error


def _normalize_counters(counters: object) -> dict[str, int]:
    if not isinstance(counters, Mapping):
        raise ValueError("counters must be a mapping")
    normalized: dict[str, int] = {}
    for name, value in counters.items():
        if not isinstance(name, str) or not name:
            raise ValueError("counter names must be non-empty strings")
        _require_non_negative_integer(f"counter {name}", value)
        normalized[name] = int(value)
    if not {"optimizer_steps", "batches_seen"} <= normalized.keys():
        raise ValueError("checkpoint counters are incomplete")
    return normalized


def _normalize_metrics(metrics: object) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise ValueError("last_metrics must be a mapping")
    normalized: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be non-empty strings")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError("metric values must be finite numbers")
        normalized[name] = float(value)
    return normalized


def _validate_sidecar_integrity(sidecar: Mapping[str, object], path: Path) -> None:
    if set(sidecar) != _SIDECAR_KEYS:
        raise ValueError("checkpoint sidecar schema is invalid")
    if sidecar.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint sidecar schema is invalid")
    if sidecar.get("checkpoint") != path.name:
        raise ValueError("checkpoint sidecar path mismatch")
    _require_positive_integer("checkpoint sidecar byte_size", sidecar.get("byte_size"))
    _validate_sha256(sidecar.get("sha256"), "checkpoint sidecar")
    if sidecar["byte_size"] > PHASE0_MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint exceeds the Phase 0 artifact size budget")
    if sidecar.get("byte_size") != path.stat().st_size:
        raise ValueError("checkpoint byte size mismatch")
    if sidecar.get("sha256") != _sha256(path):
        raise ValueError("checkpoint sha256 mismatch")


def _validate_sidecar_binding(
    sidecar: Mapping[str, object], path: Path, payload: Mapping[str, object]
) -> None:
    filename_step = _canonical_step(path)
    if sidecar.get("step") != filename_step:
        raise ValueError("checkpoint sidecar filename step mismatch")
    if sidecar.get("step") != payload["global_step"]:
        raise ValueError("checkpoint sidecar step mismatch")
    if sidecar.get("lineage_id") != payload["lineage_id"]:
        raise ValueError("checkpoint lineage mismatch")
    if sidecar.get("manifest_hashes") != payload["manifest_hashes"]:
        raise ValueError("checkpoint sidecar manifest mismatch")


def _preflight_archive(path: Path) -> None:
    _validate_regular_file(path, "checkpoint artifact")
    size = path.stat().st_size
    if not 0 < size <= PHASE0_MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint exceeds the Phase 0 artifact size budget")
    expected_members = _preflight_zip_eocd(path, size)
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("checkpoint artifact is not a valid ZIP archive") from error
    if len(members) != expected_members:
        raise ValueError("checkpoint archive member budget is invalid")
    names: set[str] = set()
    uncompressed = 0
    for member in members:
        name = member.filename
        parsed = PurePosixPath(name)
        if (
            name in names
            or not name
            or "\\" in name
            or parsed.is_absolute()
            or not parsed.parts
            or ":" in parsed.parts[0]
            or any(part in ("", ".", "..") for part in parsed.parts)
            or len(name.encode("utf-8")) > PHASE0_MAX_STRING_BYTES
            or member.flag_bits & 0x1
            or member.compress_type != zipfile.ZIP_STORED
            or member.file_size > PHASE0_MAX_ARCHIVE_MEMBER_BYTES
            or stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK
        ):
            raise ValueError("checkpoint archive structure or compression is invalid")
        names.add(name)
        uncompressed += member.file_size
        if uncompressed > PHASE0_MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ValueError("checkpoint archive exceeds the uncompressed budget")


def _preflight_zip_eocd(path: Path, size: int) -> int:
    tail_size = min(size, 22 + 65_535 + 76)
    try:
        with path.open("rb") as handle:
            handle.seek(size - tail_size)
            tail = handle.read(tail_size)
    except OSError as error:
        raise ValueError("checkpoint ZIP trailer cannot be read") from error
    signature = b"PK\x05\x06"
    relative_offset = tail.rfind(signature)
    if relative_offset < 0 or len(tail) - relative_offset < 22:
        raise ValueError("checkpoint ZIP EOCD is missing")
    eocd_offset = size - tail_size + relative_offset
    (
        _,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        comment_size,
    ) = struct.unpack_from("<4s4H2LH", tail, relative_offset)
    if (
        eocd_offset + 22 + comment_size != size
        or disk_number != 0
        or central_disk != 0
        or disk_entries != total_entries
        or total_entries in (0, 0xFFFF)
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
        or total_entries > PHASE0_MAX_ARCHIVE_MEMBERS
        or central_size > PHASE0_MAX_CENTRAL_DIRECTORY_BYTES
        or central_offset + central_size > eocd_offset
    ):
        raise ValueError("checkpoint ZIP central-directory budget or layout is invalid")
    central_end = central_offset + central_size
    if central_end != eocd_offset:
        zip64_size = struct.calcsize("<4sQ2H2L4Q")
        locator_size = struct.calcsize("<4sLQL")
        if central_end + zip64_size + locator_size != eocd_offset:
            raise ValueError("checkpoint ZIP has an inconsistent trailing record")
        zip64_relative = central_end - (size - tail_size)
        locator_relative = zip64_relative + zip64_size
        if zip64_relative < 0:
            raise ValueError(
                "checkpoint ZIP64 compatibility trailer exceeds the tail budget"
            )
        zip64 = struct.unpack_from("<4sQ2H2L4Q", tail, zip64_relative)
        locator = struct.unpack_from("<4sLQL", tail, locator_relative)
        if (
            zip64[0] != b"PK\x06\x06"
            or zip64[1] != 44
            or zip64[4] != 0
            or zip64[5] != 0
            or zip64[6] != total_entries
            or zip64[7] != total_entries
            or zip64[8] != central_size
            or zip64[9] != central_offset
            or locator[0] != b"PK\x06\x07"
            or locator[1] != 0
            or locator[2] != central_end
            or locator[3] != 1
        ):
            raise ValueError("checkpoint ZIP64 compatibility trailer is inconsistent")

    try:
        with path.open("rb") as handle:
            handle.seek(central_offset)
            central = handle.read(central_size)
    except OSError as error:
        raise ValueError("checkpoint ZIP central directory cannot be read") from error
    position = entries = 0
    while position < len(central):
        if len(central) - position < 46 or central[position : position + 4] != b"PK\x01\x02":
            raise ValueError("checkpoint ZIP central directory is invalid")
        name_size, extra_size, entry_comment_size = struct.unpack_from(
            "<3H", central, position + 28
        )
        position += 46 + name_size + extra_size + entry_comment_size
        entries += 1
        if position > len(central) or entries > total_entries:
            raise ValueError("checkpoint ZIP central-directory entry count is invalid")
    if position != len(central) or entries != total_entries:
        raise ValueError("checkpoint ZIP central-directory entry count is invalid")
    return total_entries


def _validate_regular_file(path: Path, label: str) -> None:
    _reject_symlink_hops(path)
    try:
        status = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} cannot be read") from error
    if path.is_symlink() or not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")


def _validate_checkpoint_dir(path: Path) -> None:
    _reject_symlink_hops(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("checkpoint directory must be a non-symlink directory")


def _reject_symlink_hops(path: Path) -> None:
    for candidate in (path, *path.parents):
        try:
            status = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError("checkpoint path components cannot be inspected") from error
        is_junction = getattr(candidate, "is_junction", None)
        if (
            stat.S_ISLNK(status.st_mode)
            or bool(
                getattr(status, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            )
            or callable(is_junction)
            and is_junction()
        ):
            raise ValueError(
                "checkpoint paths may not contain symlink, junction, or reparse hops"
            )


def _directory_entries(path: Path) -> list[Path]:
    _validate_checkpoint_dir(path)
    entries: list[Path] = []
    for entry in path.iterdir():
        entries.append(entry)
        if len(entries) > PHASE0_MAX_DIRECTORY_ENTRIES:
            raise ValueError("checkpoint directory exceeds the Phase 0 entry budget")
    return entries


def _load_payload(path: Path) -> dict[str, Any]:
    def cpu_only(storage: object, location: str) -> object:
        if location != "cpu":
            raise ValueError("Phase 0 checkpoints may contain only CPU tensors")
        return storage

    _preflight_archive(path)
    try:
        payload = torch.load(path, map_location=cpu_only, weights_only=True)
    except Exception as error:
        raise ValueError("checkpoint artifact cannot be loaded safely") from error
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    return payload


def _resolve_checkpoint(
    source: Path,
    *,
    auto: bool,
    expected_manifests: Mapping[str, Mapping[str, object]],
    expected_lineage_id: str | None,
) -> tuple[Path, dict[str, Any], str]:
    checkpoint_dir = source / "checkpoints" if auto else source.parent
    _validate_checkpoint_dir(checkpoint_dir)
    run_metadata = _read_run_metadata(checkpoint_dir)
    expected_hashes = _manifest_hashes(expected_manifests)
    if run_metadata["manifest_hashes"] != expected_hashes:
        raise ValueError("run metadata strict manifest mismatch")
    lineage_id = run_metadata["lineage_id"]
    if expected_lineage_id is not None:
        _require_non_empty_string("expected_lineage_id", expected_lineage_id)
        if lineage_id != expected_lineage_id:
            raise ValueError("run metadata lineage mismatch")
    if not auto:
        payload = verify_checkpoint(
            source,
            expected_manifests=expected_manifests,
            expected_lineage_id=lineage_id,
        )
        return source, payload, lineage_id

    pointer_path = checkpoint_dir / "latest.json"
    if os.path.lexists(pointer_path):
        _validate_regular_file(pointer_path, "latest pointer")
    try:
        pointer = _read_json(pointer_path, "latest pointer")
        _validate_pointer_document(pointer, run_metadata)
    except ValueError:
        checkpoint_path, payload = _scan_compatible_checkpoints(
            checkpoint_dir,
            expected_manifests=expected_manifests,
            lineage_id=lineage_id,
        )
        return checkpoint_path, payload, lineage_id
    _recover_pruning(checkpoint_dir, pointer, run_metadata)
    _recover_invalid_tombstones(checkpoint_dir, pointer, run_metadata)
    checkpoint_path = checkpoint_dir / pointer["checkpoint"]
    _require_contained(checkpoint_dir, checkpoint_path)
    payload = verify_checkpoint(
        checkpoint_path,
        expected_manifests=expected_manifests,
        expected_lineage_id=lineage_id,
    )
    _validate_pointer_target(pointer, checkpoint_path, payload)
    return checkpoint_path, payload, lineage_id


def _validate_previous_pointer(path: Path, lineage_id: str, step: int) -> None:
    run_metadata = _read_run_metadata(path.parent)
    if not os.path.lexists(path):
        _recover_invalid_tombstones(path.parent, None, run_metadata)
        return
    pointer = _read_json(path, "latest pointer")
    _validate_pointer_document(pointer, run_metadata)
    _recover_pruning(path.parent, pointer, run_metadata)
    _recover_invalid_tombstones(path.parent, pointer, run_metadata)
    checkpoint_path = path.parent / pointer["checkpoint"]
    payload = verify_checkpoint(
        checkpoint_path,
        expected_lineage_id=lineage_id,
    )
    _validate_pointer_target(pointer, checkpoint_path, payload)
    if pointer["step"] > step:
        raise ValueError("new checkpoint step must advance the lineage")


def _bind_run_metadata(
    checkpoint_dir: Path, expected: Mapping[str, object]
) -> None:
    path = checkpoint_dir / "run.json"
    if not path.exists():
        ambiguous = any(
            item.name != ".run.lock" for item in _directory_entries(checkpoint_dir)
        )
        if ambiguous:
            raise ValueError("run metadata is missing from a non-empty checkpoint run")
        _publish_json_once(path, expected)
    observed = _read_run_metadata(checkpoint_dir)
    if observed != expected:
        raise ValueError("run metadata conflicts with this checkpoint run")


def _read_run_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    metadata = _read_json(checkpoint_dir / "run.json", "run metadata")
    if set(metadata) != _RUN_METADATA_KEYS:
        raise ValueError("run metadata schema is invalid")
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("run metadata schema is invalid")
    _require_non_empty_string("run metadata lineage", metadata.get("lineage_id"))
    _validate_manifest_hashes(metadata.get("manifest_hashes"))
    return metadata


def _validate_pointer_document(
    pointer: Mapping[str, object], run_metadata: Mapping[str, object]
) -> None:
    if set(pointer) != _POINTER_KEYS:
        raise ValueError("latest pointer schema is invalid")
    if pointer.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("latest pointer schema is invalid")
    step = pointer.get("step")
    _require_positive_integer("latest pointer step", step)
    if int(step) > 999_999_999:
        raise ValueError("latest pointer step is outside the canonical range")
    expected_checkpoint = f"step-{int(step):09d}.pt"
    if (
        pointer.get("checkpoint") != expected_checkpoint
        or pointer.get("sidecar") != Path(expected_checkpoint).with_suffix(".json").name
        or pointer.get("lineage_id") != run_metadata["lineage_id"]
        or pointer.get("manifest_hashes") != run_metadata["manifest_hashes"]
    ):
        raise ValueError("latest pointer binding is invalid")
    _validate_sha256(pointer.get("sha256"), "latest pointer")
    _require_positive_integer("latest pointer byte_size", pointer.get("byte_size"))


def _validate_pointer_target(
    pointer: Mapping[str, object], path: Path, payload: Mapping[str, object]
) -> None:
    sidecar = _read_json(path.with_suffix(".json"), "checkpoint sidecar")
    if any(pointer[name] != sidecar[name] for name in _SIDECAR_KEYS):
        raise ValueError("latest pointer does not match its sidecar")
    if (
        pointer["step"] != payload["global_step"]
        or pointer["lineage_id"] != payload["lineage_id"]
        or pointer["manifest_hashes"] != payload["manifest_hashes"]
    ):
        raise ValueError("latest pointer does not match its checkpoint")


def _scan_compatible_checkpoints(
    checkpoint_dir: Path,
    *,
    expected_manifests: Mapping[str, Mapping[str, object]],
    lineage_id: str,
) -> tuple[Path, dict[str, Any]]:
    candidates = [
        (step, path)
        for step, path in _bounded_checkpoint_candidates(checkpoint_dir)
        if not _has_pruning_material(path)
    ]
    for _, candidate in sorted(candidates, reverse=True):
        try:
            _require_contained(checkpoint_dir, candidate)
            payload = verify_checkpoint(
                candidate,
                expected_manifests=expected_manifests,
                expected_lineage_id=lineage_id,
            )
        except ValueError:
            continue
        return candidate, payload
    raise ValueError("no valid immutable checkpoint matches the trusted run metadata")


def _bounded_checkpoint_candidates(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    candidates: list[tuple[int, Path]] = []
    total_bytes = 0
    for path in _directory_entries(checkpoint_dir):
        match = _STEP_NAME.fullmatch(path.name)
        if match is None:
            continue
        candidates.append((int(match.group(1)), path))
        try:
            total_bytes += path.lstat().st_size
            sidecar = path.with_suffix(".json")
            if sidecar.exists() or sidecar.is_symlink():
                total_bytes += sidecar.lstat().st_size
        except OSError as error:
            raise ValueError("checkpoint candidate cannot be inspected") from error
        if (
            len(candidates) > PHASE0_MAX_SCAN_CANDIDATES
            or total_bytes > PHASE0_MAX_SCAN_BYTES
        ):
            raise ValueError("checkpoint candidate scan exceeds the Phase 0 budget")
    return candidates


def _canonical_step(path: Path) -> int:
    match = _STEP_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError("checkpoint filename is not canonical")
    step = int(match.group(1))
    if step <= 0:
        raise ValueError("checkpoint step must be positive")
    return step


def _require_contained(directory: Path, path: Path) -> None:
    _reject_symlink_hops(path)
    try:
        if path.resolve(strict=True).parent != directory.resolve(strict=True):
            raise ValueError("checkpoint path escapes its run directory")
    except OSError as error:
        raise ValueError("checkpoint path cannot be resolved safely") from error


def _has_pruning_material(checkpoint_path: Path) -> bool:
    return any(
        path.exists() or path.is_symlink()
        for path in (
            checkpoint_path.parent / f".{checkpoint_path.name}.pruning",
            checkpoint_path.parent
            / f".{checkpoint_path.with_suffix('.json').name}.pruning",
            checkpoint_path.parent / f".{checkpoint_path.stem}.pruning.json",
        )
    )


def _claim_file(staged: Path, canonical: Path) -> bool:
    """Atomically claim a canonical name without replacing another writer."""
    _validate_checkpoint_dir(canonical.parent)
    _validate_regular_file(staged, "staged checkpoint file")
    try:
        os.link(staged, canonical)
    except FileExistsError:
        return False
    _fsync_directory(canonical.parent)
    return True


def _recover_invalid_tombstones(
    checkpoint_dir: Path,
    pointer: Mapping[str, object] | None,
    run_metadata: Mapping[str, object],
) -> None:
    tombstones = [
        (match.group(1), path)
        for path in _directory_entries(checkpoint_dir)
        if (match := _INVALID_TOMBSTONE_NAME.fullmatch(path.name)) is not None
    ]
    pointer_names = (
        {str(pointer["checkpoint"]), str(pointer["sidecar"])}
        if pointer is not None
        else set()
    )
    referenced: list[Path] = []
    for canonical_name, tombstone in sorted(tombstones):
        canonical = checkpoint_dir / canonical_name
        _validate_regular_file(tombstone, "invalid checkpoint tombstone")
        if canonical_name in pointer_names:
            if canonical.exists():
                _validate_regular_file(canonical, "pointer-referenced checkpoint material")
                try:
                    same_file = os.path.samefile(canonical, tombstone)
                except OSError as error:
                    raise ValueError("pointer invalid tombstone cannot be reconciled") from error
                if not same_file:
                    raise ValueError(
                        "pointer invalid tombstone conflicts with its canonical file"
                    )
            elif not _claim_file(tombstone, canonical):
                raise ValueError("pointer invalid tombstone restoration raced")
            referenced.append(tombstone)
            continue

        if canonical.exists():
            _validate_regular_file(canonical, "unreferenced checkpoint material")
            try:
                same_file = os.path.samefile(canonical, tombstone)
            except OSError as error:
                raise ValueError("invalid checkpoint tombstone cannot be reconciled") from error
            if same_file:
                canonical.unlink(missing_ok=True)
                _fsync_directory(checkpoint_dir)
        tombstone.unlink(missing_ok=True)
        _fsync_directory(checkpoint_dir)

    if referenced:
        assert pointer is not None
        checkpoint_path = checkpoint_dir / str(pointer["checkpoint"])
        payload = _verify_against_run(checkpoint_path, run_metadata)
        _validate_pointer_target(pointer, checkpoint_path, payload)
        for tombstone in referenced:
            tombstone.unlink(missing_ok=True)
            _fsync_directory(checkpoint_dir)


def _reclaim_unreferenced(
    paths: tuple[Path, ...], latest_path: Path, checkpoint_name: str
) -> None:
    if latest_path.exists():
        pointer = _read_json(latest_path, "latest pointer")
        if pointer.get("checkpoint") == checkpoint_name:
            raise ValueError("a pointer-referenced checkpoint cannot be reclaimed")
    for path in paths:
        tombstone = path.parent / f".{path.name}.invalid"
        if tombstone.exists():
            _validate_regular_file(tombstone, "invalid checkpoint tombstone")
            if path.exists():
                _validate_regular_file(path, "invalid unreferenced checkpoint material")
                try:
                    same_file = os.path.samefile(path, tombstone)
                except OSError as error:
                    raise ValueError("invalid checkpoint reclamation raced") from error
                if same_file:
                    path.unlink(missing_ok=True)
                    _fsync_directory(path.parent)
            tombstone.unlink(missing_ok=True)
            _fsync_directory(path.parent)
        if not path.exists():
            continue
        _validate_regular_file(path, "invalid unreferenced checkpoint material")
        if not _claim_file(path, tombstone):
            raise ValueError("invalid checkpoint reclamation raced")
        path.unlink(missing_ok=True)
        _fsync_directory(path.parent)
        tombstone.unlink(missing_ok=True)
        _fsync_directory(path.parent)


@contextmanager
def _staged_json(path: Path, value: Mapping[str, object]) -> Iterator[Path]:
    encoded = _canonical_json(value) + b"\n"
    staged_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            staged_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        if _read_json(staged_path, "staged JSON") != value:
            raise ValueError("staged JSON did not reopen exactly")
        yield staged_path
    finally:
        if staged_path is not None:
            try:
                staged_path.unlink(missing_ok=True)
            except OSError:
                pass


def _publish_json_once(path: Path, value: Mapping[str, object]) -> None:
    with _staged_json(path, value) as staged_path:
        if not _claim_file(staged_path, path):
            if _read_json(path, "existing JSON") != value:
                raise ValueError(f"{path.name} has different committed content")


def _values_equal(left: object, right: object) -> bool:
    if isinstance(left, Tensor):
        return (
            isinstance(right, Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and left.device == right.device
            and left.layout == right.layout
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping):
        return (
            isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(_values_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _same_checkpoint_state(
    left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    return left.keys() == right.keys() and all(
        name == "provenance" or _values_equal(left[name], right[name])
        for name in left
    )


def _prune_validated(checkpoint_dir: Path, keep: int) -> None:
    run_metadata = _read_run_metadata(checkpoint_dir)
    pointer = _read_json(checkpoint_dir / "latest.json", "latest pointer")
    _validate_pointer_document(pointer, run_metadata)
    _recover_pruning(checkpoint_dir, pointer, run_metadata)
    _recover_invalid_tombstones(checkpoint_dir, pointer, run_metadata)
    pointer_path = checkpoint_dir / pointer["checkpoint"]
    pointer_payload = _verify_against_run(pointer_path, run_metadata)
    _validate_pointer_target(pointer, pointer_path, pointer_payload)

    compatible: list[tuple[int, Path]] = []
    for step, candidate in _bounded_checkpoint_candidates(checkpoint_dir):
        if _has_pruning_material(candidate):
            continue
        try:
            _verify_against_run(candidate, run_metadata)
        except ValueError:
            continue
        compatible.append((step, candidate))
    compatible.sort(reverse=True)
    retained = {path.name for _, path in compatible[:keep]}
    retained.add(pointer["checkpoint"])
    for _, candidate in compatible:
        if candidate.name not in retained:
            _prune_pair(candidate)


def _verify_against_run(
    candidate: Path, run_metadata: Mapping[str, object]
) -> dict[str, Any]:
    payload = verify_checkpoint(
        candidate,
        expected_lineage_id=run_metadata["lineage_id"],
    )
    if payload["manifest_hashes"] != run_metadata["manifest_hashes"]:
        raise ValueError("checkpoint is incompatible with the trusted run metadata")
    return payload


def _prune_pair(checkpoint_path: Path) -> None:
    sidecar_path = checkpoint_path.with_suffix(".json")
    run_metadata = _read_run_metadata(checkpoint_path.parent)
    marker = checkpoint_path.parent / f".{checkpoint_path.stem}.pruning.json"
    marker_value = {
        **run_metadata,
        "checkpoint": checkpoint_path.name,
        "sidecar": sidecar_path.name,
    }
    sidecar_tombstone = sidecar_path.parent / f".{sidecar_path.name}.pruning"
    checkpoint_tombstone = (
        checkpoint_path.parent / f".{checkpoint_path.name}.pruning"
    )
    _publish_json_once(marker, marker_value)
    _claim_file(sidecar_path, sidecar_tombstone)
    sidecar_path.unlink(missing_ok=True)
    _fsync_directory(checkpoint_path.parent)
    _claim_file(checkpoint_path, checkpoint_tombstone)
    checkpoint_path.unlink(missing_ok=True)
    _fsync_directory(checkpoint_path.parent)
    sidecar_tombstone.unlink(missing_ok=True)
    checkpoint_tombstone.unlink(missing_ok=True)
    _fsync_directory(checkpoint_path.parent)
    marker.unlink(missing_ok=True)
    _fsync_directory(checkpoint_path.parent)


def _recover_pruning(
    checkpoint_dir: Path,
    pointer: Mapping[str, object],
    run_metadata: Mapping[str, object],
) -> None:
    markers = [
        (match.group(1), path)
        for path in _directory_entries(checkpoint_dir)
        if (match := _PRUNING_MARKER_NAME.fullmatch(path.name)) is not None
    ]
    for step, marker in sorted(markers):
        checkpoint_path = checkpoint_dir / f"step-{step}.pt"
        sidecar_path = checkpoint_path.with_suffix(".json")
        checkpoint_tombstone = checkpoint_dir / f".{checkpoint_path.name}.pruning"
        sidecar_tombstone = checkpoint_dir / f".{sidecar_path.name}.pruning"
        marker_value = _read_json(marker, "checkpoint pruning marker")
        expected_marker = {
            **run_metadata,
            "checkpoint": checkpoint_path.name,
            "sidecar": sidecar_path.name,
        }
        if marker_value != expected_marker:
            raise ValueError("checkpoint pruning marker is incompatible")
        if checkpoint_path.name == pointer["checkpoint"]:
            _restore_tombstone(checkpoint_path, checkpoint_tombstone)
            _restore_tombstone(sidecar_path, sidecar_tombstone)
            payload = _verify_against_run(checkpoint_path, run_metadata)
            _validate_pointer_target(pointer, checkpoint_path, payload)
        else:
            sidecar_path.unlink(missing_ok=True)
            checkpoint_path.unlink(missing_ok=True)
        sidecar_tombstone.unlink(missing_ok=True)
        checkpoint_tombstone.unlink(missing_ok=True)
        _fsync_directory(checkpoint_dir)
        marker.unlink(missing_ok=True)
        _fsync_directory(checkpoint_dir)


def _restore_tombstone(canonical: Path, tombstone: Path) -> None:
    if tombstone.exists():
        _validate_regular_file(tombstone, "checkpoint recovery tombstone")
        if canonical.exists():
            _validate_regular_file(canonical, "checkpoint recovery target")
            try:
                same_file = os.path.samefile(canonical, tombstone)
            except OSError as error:
                raise ValueError("checkpoint tombstone cannot be reconciled") from error
            if not same_file:
                raise ValueError("checkpoint tombstone conflicts with its canonical file")
        elif not _claim_file(tombstone, canonical):
            raise ValueError("checkpoint tombstone restoration raced")
    elif not canonical.exists():
        raise ValueError("checkpoint tombstone recovery is missing required material")


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    with _staged_json(path, value) as staged_path:
        os.replace(staged_path, path)
        _fsync_directory(path.parent)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _validate_regular_file(path, label)
    try:
        size = path.stat().st_size
        if not 0 < size <= PHASE0_MAX_CONTROL_JSON_BYTES:
            raise ValueError(f"{label} exceeds the Phase 0 JSON size budget")
        encoded = path.read_bytes()
        if len(encoded) != size:
            raise ValueError(f"{label} changed while being read")
        value = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    _validate_state_tree(value)
    return value


def _sha256(path: Path) -> str:
    _validate_regular_file(path, "checkpoint file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        if os.name != "nt":
            raise
        return
    try:
        os.fsync(descriptor)
    except OSError:
        if os.name != "nt":
            raise
    finally:
        os.close(descriptor)


def _require_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_non_negative_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_integer(name: str, value: object) -> None:
    if not _is_positive_integer(value):
        raise ValueError(f"{name} must be a positive integer")


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_vector(value: object, dtype: torch.dtype, *, length: int) -> bool:
    return (
        isinstance(value, Tensor)
        and value.device.type == "cpu"
        and value.dtype is dtype
        and value.ndim == 1
        and value.numel() == length
    )
