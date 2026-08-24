"""Atomic, weights-only-compatible checkpoints for deterministic MOIRA runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import stat
import tempfile
import zipfile
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .ledger import ParameterLedger
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
PHASE0_MAX_STRING_BYTES = 64 * 1024
_PHASE0_TENSOR_DTYPES = frozenset(
    {
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    }
)
_STEP_NAME = re.compile(r"step-([0-9]{9})\.pt\Z", re.ASCII)
_SIDECAR_NAME = re.compile(r"step-[0-9]{9}\.json\Z", re.ASCII)
_PRUNING_MARKER_NAME = re.compile(
    r"\.step-([0-9]{9})\.pruning\.json\Z", re.ASCII
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
    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module")
    if not isinstance(optimizer, Optimizer):
        raise ValueError("optimizer must be an Optimizer")
    if not isinstance(scheduler, LRScheduler):
        raise ValueError("scheduler must be an LRScheduler")
    if not isinstance(data_cursor, DataCursor):
        raise ValueError("data_cursor must be a DataCursor")
    if not isinstance(parameter_ledger, ParameterLedger):
        raise ValueError("parameter_ledger must be a ParameterLedger")
    parameter_ledger.assert_matches()

    manifests = _normalize_manifests(semantic_manifests)
    manifest_hashes = _manifest_hashes(manifests)
    normalized_provenance = _normalize_provenance(provenance)
    model_state = dict(model.state_dict())
    ledger_evidence = _ledger_evidence(parameter_ledger, model_state)
    if scaler is not None:
        raise ValueError("Phase 0 checkpoints require scaler=None")
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

    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _validate_checkpoint_dir(checkpoint_dir)
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
    _validate_sidecar(reopened_sidecar, checkpoint_path, reopened)

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
    payload = _load_payload(path)
    _validate_payload(payload, expected_manifests=expected_manifests)
    _validate_sidecar(sidecar, path, payload)
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
    """Strictly restore components and restore process RNG state last."""
    normalized_expected = _normalize_manifests(expected_manifests)
    checkpoint_path = _resolve_checkpoint(
        Path(source),
        auto=auto,
        expected_manifests=normalized_expected,
        expected_lineage_id=expected_lineage_id,
    )
    payload = verify_checkpoint(
        checkpoint_path,
        expected_manifests=expected_manifests,
        expected_lineage_id=expected_lineage_id,
    )
    _validate_model_compatibility(model, payload)
    if scaler is not None:
        raise ValueError("Phase 0 checkpoints require scaler=None")
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
    _validate_rng_state(payload["rng_state"])
    _normalize_metrics(payload["last_metrics"])


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
    if not isinstance(state, dict) or not state or not isinstance(evidence, dict):
        raise ValueError("model state and parameter ledger must be mappings")
    schema = evidence.get("model_state_schema")
    entries = evidence.get("entries")
    total = evidence.get("total_trainable")
    valid_entries = isinstance(entries, list) and bool(entries) and all(
        isinstance(entry, dict)
        and isinstance(entry.get("aliases"), list)
        and bool(entry["aliases"])
        and all(isinstance(alias, str) and alias for alias in entry["aliases"])
        and _is_positive_integer(entry.get("trainable_parameters"))
        for entry in entries or []
    )
    if (
        not isinstance(schema, dict)
        or set(schema) != set(state)
        or not _is_positive_integer(total)
        or not valid_entries
        or sum(entry["trainable_parameters"] for entry in entries) != total
    ):
        raise ValueError("model-state mismatch with parameter ledger evidence")
    for name, tensor in state.items():
        expected = schema[name]
        if (
            not isinstance(name, str)
            or not isinstance(tensor, Tensor)
            or not isinstance(expected, dict)
            or expected.get("shape") != list(tensor.shape)
            or expected.get("dtype") != str(tensor.dtype)
            or expected.get("numel") != tensor.numel()
        ):
            raise ValueError("model-state mismatch with parameter ledger evidence")


def _validate_model_compatibility(model: nn.Module, payload: Mapping[str, object]) -> None:
    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module")
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


def _validate_state_tree(value: object) -> None:
    stack = [(value, 0)]
    entries = tensors = tensor_bytes = 0
    storages: set[tuple[int, int]] = set()
    containers: set[int] = set()
    while stack:
        current, depth = stack.pop()
        entries += 1
        if entries > PHASE0_MAX_STATE_ENTRIES or depth > PHASE0_MAX_STATE_DEPTH:
            raise ValueError("checkpoint state exceeds the Phase 0 structure budget")
        if isinstance(current, Tensor):
            tensors += 1
            if current.device.type == "cpu" and current.layout is torch.strided:
                storage = current.untyped_storage()
                storage_key = (storage.data_ptr(), storage.nbytes())
                if storage_key not in storages:
                    storages.add(storage_key)
                    tensor_bytes += storage.nbytes()
            if (
                tensors > PHASE0_MAX_TENSORS
                or tensor_bytes > PHASE0_MAX_TENSOR_STORAGE_BYTES
                or current.ndim > PHASE0_MAX_TENSOR_RANK
                or current.dtype not in _PHASE0_TENSOR_DTYPES
                or current.device.type != "cpu"
                or current.layout is not torch.strided
                or (
                    (current.is_floating_point() or current.is_complex())
                    and not bool(torch.isfinite(current).all())
                )
            ):
                raise ValueError("checkpoint tensor state is outside Phase 0 bounds")
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
        elif isinstance(current, complex):
            if not math.isfinite(current.real) or not math.isfinite(current.imag):
                raise ValueError("checkpoint complex state must be finite")
        elif current is not None and not isinstance(current, (int, bool)):
            raise ValueError("checkpoint state contains an unsupported value")


def _preflight_components(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    payload: Mapping[str, object],
) -> None:
    _validate_optimizer_schema(optimizer.state_dict(), payload["optimizer_state"])
    _validate_component_schema(
        scheduler.state_dict(), payload["scheduler_state"], "scheduler"
    )
    cloned_model, cloned_optimizer, cloned_scheduler = deepcopy(
        (model, optimizer, scheduler)
    )
    cloned_model.load_state_dict(payload["model_state"], strict=True)
    cloned_scheduler.load_state_dict(payload["scheduler_state"])
    cloned_optimizer.load_state_dict(payload["optimizer_state"])
    for parameter, state in cloned_optimizer.state.items():
        if not isinstance(parameter, Tensor) or not isinstance(state, Mapping):
            raise ValueError("optimizer state is incompatible with its parameters")
        for name, state_value in state.items():
            if isinstance(state_value, Tensor) and name != "step":
                if state_value.shape != parameter.shape:
                    raise ValueError("optimizer tensor state has an incompatible shape")


def _validate_optimizer_schema(expected: object, observed: object) -> None:
    if (
        not isinstance(expected, Mapping)
        or not isinstance(observed, Mapping)
        or set(expected) != {"state", "param_groups"}
        or set(observed) != {"state", "param_groups"}
    ):
        raise ValueError("optimizer state schema is invalid")
    expected_groups = expected["param_groups"]
    observed_groups = observed["param_groups"]
    if (
        not isinstance(expected_groups, list)
        or not isinstance(observed_groups, list)
        or len(expected_groups) != len(observed_groups)
    ):
        raise ValueError("optimizer parameter groups are incompatible")
    saved_parameters: set[int] = set()
    for expected_group, observed_group in zip(expected_groups, observed_groups):
        if (
            not isinstance(expected_group, Mapping)
            or not isinstance(observed_group, Mapping)
            or set(expected_group) != set(observed_group)
            or not isinstance(observed_group.get("params"), list)
            or len(expected_group["params"]) != len(observed_group["params"])
            or not all(
                isinstance(item, int) and not isinstance(item, bool)
                for item in observed_group["params"]
            )
        ):
            raise ValueError("optimizer parameter groups are incompatible")
        for key in expected_group:
            if key != "params":
                _validate_component_schema(
                    expected_group[key], observed_group[key], "optimizer"
                )
        saved_parameters.update(observed_group["params"])
    observed_state = observed["state"]
    if not isinstance(observed_state, Mapping) or not set(observed_state) <= saved_parameters:
        raise ValueError("optimizer state references unknown parameters")


def _validate_component_schema(expected: object, observed: object, label: str) -> None:
    if isinstance(expected, Mapping):
        expected = {key: value for key, value in expected.items() if not callable(value)}
        if not isinstance(observed, Mapping) or set(expected) != set(observed):
            raise ValueError(f"{label} state schema is incompatible")
        for key in expected:
            _validate_component_schema(expected[key], observed[key], label)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(observed, type(expected)) or len(expected) != len(observed):
            raise ValueError(f"{label} state schema is incompatible")
        for left, right in zip(expected, observed):
            _validate_component_schema(left, right, label)
    elif type(expected) is not type(observed):
        raise ValueError(f"{label} state schema is incompatible")


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
    if sidecar.get("byte_size") != path.stat().st_size:
        raise ValueError("checkpoint byte size mismatch")
    if sidecar.get("sha256") != _sha256(path):
        raise ValueError("checkpoint sha256 mismatch")


def _validate_sidecar(
    sidecar: Mapping[str, object], path: Path, payload: Mapping[str, object]
) -> None:
    _validate_sidecar_integrity(sidecar, path)
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
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError("checkpoint artifact is not a valid ZIP archive") from error
    if not members or len(members) > PHASE0_MAX_ARCHIVE_MEMBERS:
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
        if candidate.is_symlink():
            raise ValueError("checkpoint paths may not contain symlink hops")


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
) -> Path:
    checkpoint_dir = (
        source / "checkpoints"
        if auto and source.name != "checkpoints"
        else source if auto else source.parent
    )
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
        return source

    pointer_path = checkpoint_dir / "latest.json"
    try:
        pointer = _read_json(pointer_path, "latest pointer")
        _validate_pointer_document(pointer, run_metadata)
    except ValueError:
        return _scan_compatible_checkpoints(
            checkpoint_dir,
            expected_manifests=expected_manifests,
            lineage_id=lineage_id,
        )
    checkpoint_path = checkpoint_dir / pointer["checkpoint"]
    _require_contained(checkpoint_dir, checkpoint_path)
    payload = verify_checkpoint(
        checkpoint_path,
        expected_manifests=expected_manifests,
        expected_lineage_id=lineage_id,
    )
    _validate_pointer_target(pointer, checkpoint_path, payload)
    return checkpoint_path


def _validate_previous_pointer(path: Path, lineage_id: str, step: int) -> None:
    if not path.exists():
        return
    pointer = _read_json(path, "latest pointer")
    run_metadata = _read_run_metadata(path.parent)
    _validate_pointer_document(pointer, run_metadata)
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
            item.name == "latest.json"
            or _STEP_NAME.fullmatch(item.name)
            or _SIDECAR_NAME.fullmatch(item.name)
            for item in _directory_entries(checkpoint_dir)
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
) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in _directory_entries(checkpoint_dir):
        match = _STEP_NAME.fullmatch(path.name)
        if match is not None and not _has_pruning_material(path):
            candidates.append((int(match.group(1)), path))
    for _, candidate in sorted(candidates, reverse=True):
        try:
            _require_contained(checkpoint_dir, candidate)
            verify_checkpoint(
                candidate,
                expected_manifests=expected_manifests,
                expected_lineage_id=lineage_id,
            )
        except ValueError:
            continue
        return candidate
    raise ValueError("no valid immutable checkpoint matches the trusted run metadata")


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


def _reclaim_unreferenced(
    paths: tuple[Path, ...], latest_path: Path, checkpoint_name: str
) -> None:
    if latest_path.exists():
        pointer = _read_json(latest_path, "latest pointer")
        if pointer.get("checkpoint") == checkpoint_name:
            raise ValueError("a pointer-referenced checkpoint cannot be reclaimed")
    for path in paths:
        if not path.exists():
            continue
        _validate_regular_file(path, "invalid unreferenced checkpoint material")
        tombstone = path.parent / f".{path.name}.invalid"
        if not _claim_file(path, tombstone):
            try:
                same_file = os.path.samefile(path, tombstone)
            except OSError as error:
                raise ValueError("invalid checkpoint reclamation raced") from error
            if not same_file:
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
        return isinstance(right, Tensor) and torch.equal(left, right)
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
    pointer_path = checkpoint_dir / pointer["checkpoint"]
    pointer_payload = _verify_against_run(pointer_path, run_metadata)
    _validate_pointer_target(pointer, pointer_path, pointer_payload)
    _recover_pruning(checkpoint_dir, pointer["checkpoint"], run_metadata)

    compatible: list[tuple[int, Path]] = []
    for candidate in _directory_entries(checkpoint_dir):
        match = _STEP_NAME.fullmatch(candidate.name)
        if match is None or _has_pruning_material(candidate):
            continue
        try:
            _verify_against_run(candidate, run_metadata)
        except ValueError:
            continue
        compatible.append((int(match.group(1)), candidate))
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
    marker.unlink(missing_ok=True)
    _fsync_directory(checkpoint_path.parent)


def _recover_pruning(
    checkpoint_dir: Path,
    pointer_target: str,
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
        if checkpoint_path.name == pointer_target:
            if not checkpoint_path.exists() and checkpoint_tombstone.exists():
                _claim_file(checkpoint_tombstone, checkpoint_path)
            if not sidecar_path.exists() and sidecar_tombstone.exists():
                _claim_file(sidecar_tombstone, sidecar_path)
            _verify_against_run(checkpoint_path, run_metadata)
        else:
            sidecar_path.unlink(missing_ok=True)
            checkpoint_path.unlink(missing_ok=True)
        sidecar_tombstone.unlink(missing_ok=True)
        checkpoint_tombstone.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        _fsync_directory(checkpoint_dir)


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
