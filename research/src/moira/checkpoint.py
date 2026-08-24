"""Atomic, weights-only-compatible checkpoints for deterministic MOIRA runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .ledger import ParameterLedger
from .synthetic import DataCursor


CHECKPOINT_SCHEMA_VERSION = 1
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
    parameter_ledger: ParameterLedger,
    counters: Mapping[str, int],
    last_metrics: Mapping[str, float],
    keep_last_checkpoints: int,
) -> CheckpointRecord:
    """Commit one immutable optimizer-boundary checkpoint and then its pointer."""
    _require_non_empty_string("lineage_id", lineage_id)
    _require_positive_integer("global_step", global_step)
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
    model_state = dict(model.state_dict())
    ledger_evidence = _ledger_evidence(parameter_ledger, model_state)
    if scaler is not None and not hasattr(scaler, "state_dict"):
        raise ValueError("scaler must expose state_dict")
    scaler_state = None if scaler is None else scaler.state_dict()
    if scaler_state is not None and not isinstance(scaler_state, dict):
        raise ValueError("scaler state must be a mapping")
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
        "scaler_state": scaler_state,
        "rng_state": _capture_rng_state(),
        "data_cursor": {
            "kind": data_cursor.kind,
            "next_sample_index": data_cursor.next_sample_index,
            "samples_seen": data_cursor.samples_seen,
            "tokens_seen": data_cursor.tokens_seen,
        },
        "semantic_manifests": manifests,
        "manifest_hashes": manifest_hashes,
        "parameter_ledger": ledger_evidence,
        "counters": _normalize_counters(counters),
        "last_metrics": _normalize_metrics(last_metrics),
    }
    _validate_payload(payload, expected_manifests=manifests)

    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"step-{global_step:09d}.pt"
    sidecar_path = checkpoint_path.with_suffix(".json")
    latest_path = checkpoint_dir / "latest.json"
    if checkpoint_path.exists() or sidecar_path.exists():
        raise ValueError("checkpoint step is already committed")
    _validate_previous_pointer(latest_path, lineage_id, global_step)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            dir=checkpoint_dir,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, checkpoint_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
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
    _atomic_write_json(sidecar_path, sidecar)
    reopened_sidecar = _read_json(sidecar_path, "checkpoint sidecar")
    _validate_sidecar(reopened_sidecar, checkpoint_path, reopened)

    pointer = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint": checkpoint_path.name,
        "sidecar": sidecar_path.name,
        "step": global_step,
        "lineage_id": lineage_id,
        "manifest_hashes": manifest_hashes,
    }
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
    if path.suffix != ".pt" or not path.is_file():
        raise ValueError("checkpoint must be an existing .pt artifact")
    sidecar_path = path.with_suffix(".json")
    sidecar = _read_json(sidecar_path, "checkpoint sidecar")
    _validate_sidecar_integrity(sidecar, path)
    payload = _load_payload(path)
    normalized_expected = (
        None if expected_manifests is None else _normalize_manifests(expected_manifests)
    )
    _validate_payload(payload, expected_manifests=normalized_expected)
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
    checkpoint_path = _resolve_checkpoint(Path(source), auto=auto)
    payload = verify_checkpoint(
        checkpoint_path,
        expected_manifests=expected_manifests,
        expected_lineage_id=expected_lineage_id,
    )
    _validate_model_compatibility(model, payload)
    model.load_state_dict(payload["model_state"], strict=True)
    scheduler.load_state_dict(payload["scheduler_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    saved_scaler = payload["scaler_state"]
    if saved_scaler is None:
        if scaler is not None:
            raise ValueError("checkpoint has no scaler state")
    else:
        if scaler is None or not hasattr(scaler, "load_state_dict"):
            raise ValueError("checkpoint requires a compatible scaler")
        scaler.load_state_dict(saved_scaler)

    cursor = _cursor_from_payload(payload)
    counters = dict(payload["counters"])
    last_metrics = {
        str(name): float(value) for name, value in payload["last_metrics"].items()
    }
    _restore_rng_state(payload["rng_state"])
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
    return normalized


def _manifest_hashes(manifests: Mapping[str, Mapping[str, object]]) -> dict[str, str]:
    return {
        name: hashlib.sha256(_canonical_json(manifests[name])).hexdigest()
        for name in _SEMANTIC_MANIFEST_NAMES
    }


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
        "torch_cuda_all": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    _validate_rng_state(state)
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
    cuda_states = state["torch_cuda_all"]
    assert isinstance(cuda_states, list)
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA RNG state does not match the current hardware")
        torch.cuda.set_rng_state_all(cuda_states)


def _validate_payload(
    payload: object,
    *,
    expected_manifests: Mapping[str, Mapping[str, object]] | None,
) -> None:
    if not isinstance(payload, dict) or not _REQUIRED_PAYLOAD_KEYS <= payload.keys():
        raise ValueError("checkpoint payload is missing required keys")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema version is invalid")
    _require_non_empty_string("lineage_id", payload["lineage_id"])
    _require_positive_integer("global_step", payload["global_step"])
    if payload["micro_step"] != 0:
        raise ValueError("checkpoint micro step must be zero")
    _require_non_negative_integer("samples_seen", payload["samples_seen"])
    _require_non_negative_integer("tokens_seen", payload["tokens_seen"])

    manifests = _normalize_manifests(payload["semantic_manifests"])
    observed_hashes = _manifest_hashes(manifests)
    if payload["manifest_hashes"] != observed_hashes:
        raise ValueError("checkpoint manifest hashes do not match their content")
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
    if payload["scaler_state"] is not None and not isinstance(
        payload["scaler_state"], dict
    ):
        raise ValueError("scaler state must be a mapping or null")
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
            or tensor.device.type == "meta"
            or not isinstance(expected, dict)
            or expected.get("shape") != list(tensor.shape)
            or expected.get("dtype") != str(tensor.dtype)
            or expected.get("numel") != tensor.numel()
            or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
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
        if not isinstance(saved, Tensor) or (
            saved.shape != tensor.shape or saved.dtype != tensor.dtype
        ):
            raise ValueError("model-state mismatch with the constructed model")


def _validate_rng_state(state: object) -> None:
    if not isinstance(state, Mapping) or set(state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda_all",
    }:
        raise ValueError("RNG state is incomplete")
    python_state = state["python"]
    numpy_state = state["numpy"]
    if (
        not isinstance(python_state, Mapping)
        or set(python_state) != {"version", "state", "gauss"}
        or not _is_vector(python_state.get("state"), torch.int64)
    ):
        raise ValueError("Python RNG state is invalid")
    if (
        not isinstance(numpy_state, Mapping)
        or set(numpy_state)
        != {"bit_generator", "keys", "position", "has_gauss", "cached_gaussian"}
        or not _is_vector(numpy_state.get("keys"), torch.int64)
    ):
        raise ValueError("NumPy RNG state is invalid")
    torch_cpu = state["torch_cpu"]
    cuda_states = state["torch_cuda_all"]
    if not isinstance(torch_cpu, Tensor) or torch_cpu.dtype is not torch.uint8:
        raise ValueError("Torch CPU RNG state is invalid")
    if not isinstance(cuda_states, list) or not all(
        isinstance(value, Tensor) and value.dtype is torch.uint8
        for value in cuda_states
    ):
        raise ValueError("Torch CUDA RNG states are invalid")


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
    if sidecar.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint sidecar schema is invalid")
    if sidecar.get("checkpoint") != path.name:
        raise ValueError("checkpoint sidecar path mismatch")
    if sidecar.get("byte_size") != path.stat().st_size:
        raise ValueError("checkpoint byte size mismatch")
    if sidecar.get("sha256") != _sha256(path):
        raise ValueError("checkpoint sha256 mismatch")


def _validate_sidecar(
    sidecar: Mapping[str, object], path: Path, payload: Mapping[str, object]
) -> None:
    _validate_sidecar_integrity(sidecar, path)
    if sidecar.get("step") != payload["global_step"]:
        raise ValueError("checkpoint sidecar step mismatch")
    if sidecar.get("lineage_id") != payload["lineage_id"]:
        raise ValueError("checkpoint lineage mismatch")
    if sidecar.get("manifest_hashes") != payload["manifest_hashes"]:
        raise ValueError("checkpoint sidecar manifest mismatch")


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("checkpoint artifact cannot be loaded safely") from error
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    return payload


def _resolve_checkpoint(source: Path, *, auto: bool) -> Path:
    if not auto:
        return source
    checkpoint_dir = source / "checkpoints" if source.name != "checkpoints" else source
    pointer_path = checkpoint_dir / "latest.json"
    try:
        pointer = _read_json(pointer_path, "latest pointer")
        checkpoint_name = pointer.get("checkpoint")
        sidecar_name = pointer.get("sidecar")
        if (
            pointer.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or not isinstance(checkpoint_name, str)
            or Path(checkpoint_name).name != checkpoint_name
            or not isinstance(sidecar_name, str)
            or Path(sidecar_name).name != sidecar_name
            or Path(checkpoint_name).with_suffix(".json").name != sidecar_name
        ):
            raise ValueError("latest pointer is invalid")
        checkpoint_path = checkpoint_dir / checkpoint_name
    except ValueError:
        for candidate in sorted(checkpoint_dir.glob("step-*.pt"), reverse=True):
            try:
                verify_checkpoint(candidate)
            except ValueError:
                continue
            return candidate
        raise ValueError("no valid immutable checkpoint is available") from None
    return checkpoint_path


def _validate_previous_pointer(path: Path, lineage_id: str, step: int) -> None:
    if not path.exists():
        return
    pointer = _read_json(path, "latest pointer")
    if pointer.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("existing latest pointer schema is invalid")
    if pointer.get("lineage_id") != lineage_id:
        raise ValueError("cannot mix checkpoint lineages in one run directory")
    previous_step = pointer.get("step")
    _require_positive_integer("existing checkpoint step", previous_step)
    if int(previous_step) >= step:
        raise ValueError("new checkpoint step must advance the lineage")


def _prune_validated(checkpoint_dir: Path, keep: int) -> None:
    candidates = sorted(checkpoint_dir.glob("step-*.pt"), reverse=True)
    for candidate in candidates[keep:]:
        try:
            verify_checkpoint(candidate)
        except ValueError:
            continue
        try:
            candidate.unlink()
            candidate.with_suffix(".json").unlink()
        except OSError:
            continue


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    encoded = _canonical_json(value) + b"\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _is_vector(value: object, dtype: torch.dtype) -> bool:
    return isinstance(value, Tensor) and value.dtype is dtype and value.ndim == 1
