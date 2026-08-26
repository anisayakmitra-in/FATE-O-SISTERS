"""Portable, bounded checkpoints for the M1 and M2 data trainers.

The existing checkpoint module is deliberately strict and Phase-0-specific.
M1 uses a separate small contract so a capability loom can be saved alongside
the core model without weakening the verified M0 checkpoint format.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import os
import pickle
from pathlib import Path
import tempfile
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .config import ModelConfig
from .model import MoiraModel


M1_CHECKPOINT_SCHEMA_VERSION = 1
M1_MAX_CHECKPOINT_BYTES = 1 * 1024 * 1024 * 1024
M2_MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
M2_REFERENCE_PARAMETERS = 153_349_122
_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "lineage_id",
        "global_step",
        "sample_cursor",
        "model_config",
        "model_state",
        "loom_state",
        "optimizer_state",
        "rng_state",
        "losses",
        "manifest",
    }
)


@dataclass(frozen=True, slots=True)
class M1LoadedCheckpoint:
    checkpoint_path: Path
    lineage_id: str
    global_step: int
    sample_cursor: int
    losses: tuple[float, ...]
    manifest: dict[str, Any]


def save_m1_checkpoint(
    path: str | os.PathLike[str],
    *,
    lineage_id: str,
    global_step: int,
    sample_cursor: int,
    model: MoiraModel,
    optimizer: Optimizer,
    loom: nn.Module | None,
    losses: tuple[float, ...],
    manifest: Mapping[str, object],
) -> Path:
    """Atomically save one M1 model/optimizer/loom state at an optimizer boundary."""
    destination = Path(path)
    _validate_destination(destination)
    _validate_identity(lineage_id, global_step, sample_cursor, losses, manifest)
    if not isinstance(model, MoiraModel):
        raise ValueError("M1 checkpoints require a MoiraModel")
    if not isinstance(optimizer, Optimizer):
        raise ValueError("M1 checkpoints require a torch optimizer")
    if loom is not None and not isinstance(loom, nn.Module):
        raise ValueError("loom must be a torch module when present")
    normalized_manifest = _normalize_manifest(manifest)
    payload: dict[str, Any] = {
        "schema_version": M1_CHECKPOINT_SCHEMA_VERSION,
        "lineage_id": lineage_id,
        "global_step": global_step,
        "sample_cursor": sample_cursor,
        "model_config": asdict(model.config),
        "model_state": _cpu_tree(model.state_dict()),
        "loom_state": None if loom is None else _cpu_tree(loom.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "rng_state": torch.get_rng_state().cpu(),
        "losses": list(losses),
        "manifest": normalized_manifest,
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size > checkpoint_byte_limit(model):
            raise ValueError("checkpoint exceeds its model-surface artifact limit")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_m1_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: MoiraModel,
    optimizer: Optimizer,
    loom: nn.Module | None,
    expected_manifest: Mapping[str, object] | None = None,
) -> M1LoadedCheckpoint:
    """Load and validate an M1 checkpoint, restoring model, loom, optimizer, and RNG."""
    source = Path(path)
    _validate_source(source, max_bytes=checkpoint_byte_limit(model))
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise ValueError("M1 checkpoint cannot be loaded safely") from error
    _validate_payload(payload, model=model, loom=loom, expected_manifest=expected_manifest)
    assert isinstance(payload, Mapping)
    model.load_state_dict(payload["model_state"], strict=True)
    if loom is not None:
        loom_state = payload["loom_state"]
        assert isinstance(loom_state, Mapping)
        loom.load_state_dict(loom_state, strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    rng_state = payload["rng_state"]
    assert isinstance(rng_state, Tensor)
    torch.set_rng_state(rng_state)
    return M1LoadedCheckpoint(
        checkpoint_path=source,
        lineage_id=payload["lineage_id"],
        global_step=payload["global_step"],
        sample_cursor=payload["sample_cursor"],
        losses=tuple(payload["losses"]),
        manifest=dict(payload["manifest"]),
    )


def checkpoint_manifest_hash(manifest: Mapping[str, object]) -> str:
    """Return a stable digest suitable for a caller's run metadata."""
    normalized = _normalize_manifest(manifest)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_destination(path: Path) -> None:
    if not path.name or path.name in {".", ".."}:
        raise ValueError("M1 checkpoint path must name a file")
    if path.exists() and path.is_symlink():
        raise ValueError("M1 checkpoint path may not be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)


def _validate_source(path: Path, *, max_bytes: int) -> None:
    try:
        status = path.stat()
    except OSError as error:
        raise ValueError("M1 checkpoint path cannot be inspected") from error
    if not path.is_file() or path.is_symlink():
        raise ValueError("M1 checkpoint must be a regular file")
    if status.st_size <= 0 or status.st_size > max_bytes:
        raise ValueError("M1 checkpoint exceeds the bounded artifact limit")


def checkpoint_byte_limit(model: MoiraModel) -> int:
    """Return the fail-closed artifact envelope for a supported model surface."""
    parameters = model.parameter_ledger().total_trainable
    if parameters <= 50_000_000:
        return M1_MAX_CHECKPOINT_BYTES
    if (
        parameters == M2_REFERENCE_PARAMETERS
        and model.config == ModelConfig.m2_gpu()
    ):
        return M2_MAX_CHECKPOINT_BYTES
    raise ValueError("checkpointing is not enabled for this model surface")


def _validate_identity(
    lineage_id: str,
    global_step: int,
    sample_cursor: int,
    losses: tuple[float, ...],
    manifest: Mapping[str, object],
) -> None:
    if not isinstance(lineage_id, str) or not lineage_id.strip():
        raise ValueError("lineage_id must be non-empty")
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("global_step must be positive")
    if type(sample_cursor) is not int or sample_cursor < 0:
        raise ValueError("sample_cursor must be non-negative")
    if not isinstance(losses, tuple) or len(losses) > 100_000:
        raise ValueError("losses must be a bounded tuple")
    if any(not isinstance(value, (float, int)) or not torch.isfinite(torch.tensor(float(value))) for value in losses):
        raise ValueError("losses must contain finite numbers")
    _normalize_manifest(manifest)


def _normalize_manifest(manifest: Mapping[str, object]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise ValueError("M1 checkpoint manifest must be a mapping")
    try:
        normalized = json.loads(
            json.dumps(
                dict(manifest),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("M1 checkpoint manifest must be JSON-compatible") from error
    if not isinstance(normalized, dict):
        raise ValueError("M1 checkpoint manifest must be an object")
    return normalized


def _validate_payload(
    payload: object,
    *,
    model: MoiraModel,
    loom: nn.Module | None,
    expected_manifest: Mapping[str, object] | None,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _CHECKPOINT_KEYS:
        raise ValueError("M1 checkpoint schema is invalid")
    if payload.get("schema_version") != M1_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("M1 checkpoint schema version is invalid")
    if payload.get("model_config") != asdict(model.config):
        raise ValueError("M1 checkpoint model config does not match")
    _validate_identity(
        payload.get("lineage_id"),
        payload.get("global_step"),
        payload.get("sample_cursor"),
        tuple(payload.get("losses", ())),
        payload.get("manifest"),
    )
    if not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("M1 checkpoint model state is invalid")
    _validate_state_mapping(payload["model_state"], model.state_dict(), "model")
    if loom is None:
        if payload.get("loom_state") is not None:
            raise ValueError("M1 checkpoint contains an unexpected loom")
    elif not isinstance(payload.get("loom_state"), Mapping):
        raise ValueError("M1 checkpoint is missing the capability loom")
    elif isinstance(payload.get("loom_state"), Mapping):
        _validate_state_mapping(payload["loom_state"], loom.state_dict(), "loom")
    if not isinstance(payload.get("optimizer_state"), Mapping):
        raise ValueError("M1 checkpoint optimizer state is invalid")
    rng_state = payload.get("rng_state")
    if (
        not isinstance(rng_state, Tensor)
        or rng_state.dtype is not torch.uint8
        or rng_state.ndim != 1
        or rng_state.numel() != torch.get_rng_state().numel()
    ):
        raise ValueError("M1 checkpoint RNG state is invalid")
    if expected_manifest is not None and dict(payload["manifest"]) != _normalize_manifest(expected_manifest):
        raise ValueError("M1 checkpoint manifest does not match")


def _validate_state_mapping(
    observed: Mapping[object, object], expected: Mapping[str, Tensor], label: str
) -> None:
    if set(observed) != set(expected):
        raise ValueError(f"M1 checkpoint {label} state keys do not match")
    for name, reference in expected.items():
        value = observed[name]
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or value.dtype != reference.dtype
            or value.shape != reference.shape
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"M1 checkpoint {label} state tensor is invalid")


def _cpu_tree(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value
