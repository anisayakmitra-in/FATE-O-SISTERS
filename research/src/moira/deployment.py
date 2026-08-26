"""Persistent Consumer/Enterprise profile state for one MOIRA identity."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from .elastic import (
    BIT_PRECISION_LADDER,
    CANONICAL_PARAMETER_LADDER,
    PrecisionPolicy,
    format_parameter_count,
)


DEPLOYMENT_STATE_SCHEMA_VERSION = 1
_MAX_DEPLOYMENT_STATE_BYTES = 64 * 1024


def deployment_state(profile: str) -> dict[str, object]:
    """Return the canonical local policy state without touching model weights."""
    policy = PrecisionPolicy.for_profile(profile)
    return {
        "schema_version": DEPLOYMENT_STATE_SCHEMA_VERSION,
        "model": "MOIRA",
        "profile": profile,
        "profile_changes_model_identity": False,
        "precision_policy": policy.to_dict(),
        "parameter_growth_ladder": [
            format_parameter_count(value) for value in CANONICAL_PARAMETER_LADDER
        ],
        "bit_precision_ladder": [f"int{value}" for value in BIT_PRECISION_LADDER],
        "unbounded_parameter_growth": True,
        "state_only_no_weight_conversion": True,
    }


def load_deployment_state(path: str | Path) -> dict[str, Any]:
    """Load and validate a bounded deployment state document."""
    state_path = Path(path)
    try:
        status = state_path.lstat()
    except OSError as error:
        raise ValueError("deployment state cannot be read") from error
    if state_path.is_symlink() or not stat.S_ISREG(status.st_mode):
        raise ValueError("deployment state must be a regular non-symlink file")
    if not 0 < status.st_size <= _MAX_DEPLOYMENT_STATE_BYTES:
        raise ValueError("deployment state exceeds its JSON size budget")
    try:
        value = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("deployment state cannot be read") from error
    if not isinstance(value, dict):
        raise ValueError("deployment state must be a JSON object")
    profile = value.get("profile")
    if not isinstance(profile, str) or value != deployment_state(profile):
        raise ValueError("deployment state does not match the canonical schema")
    return value


def switch_deployment_profile(
    profile: str,
    path: str | Path,
) -> dict[str, object]:
    """Atomically persist a local policy selection and report the transition."""
    state_path = Path(path)
    state = deployment_state(profile)
    parent = state_path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("deployment state parent must be a non-symlink directory")

    previous_profile: str | None = None
    if os.path.lexists(state_path):
        previous_profile = str(load_deployment_state(state_path)["profile"])

    encoded = (json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{state_path.name}.",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, state_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    if load_deployment_state(state_path) != state:
        raise ValueError("deployment state did not reopen exactly")
    return {
        **state,
        "previous_profile": previous_profile,
        "state_path": str(state_path.resolve()),
        "switch_applied": True,
        "weights_converted": False,
    }


__all__ = [
    "DEPLOYMENT_STATE_SCHEMA_VERSION",
    "deployment_state",
    "load_deployment_state",
    "switch_deployment_profile",
]
