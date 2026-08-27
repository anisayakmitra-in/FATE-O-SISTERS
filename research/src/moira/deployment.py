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
    COMPUTE_DTYPE_LADDER,
    DEFAULT_PARAMETER_ROOT,
    PARAMETER_GROWTH_MULTIPLIER,
    PARAMETER_GROWTH_PREVIEW_STEPS,
    PrecisionPolicy,
    VERIFICATION_DTYPE_LADDER,
    format_bit_precision,
    format_parameter_count,
    parameter_growth_ladder,
    parse_parameter_count,
)


DEPLOYMENT_STATE_SCHEMA_VERSION = 2
_MAX_DEPLOYMENT_STATE_BYTES = 64 * 1024


def deployment_state(
    profile: str,
    root_parameters: int | str = DEFAULT_PARAMETER_ROOT,
) -> dict[str, object]:
    """Return the canonical local policy state without touching model weights."""
    policy = PrecisionPolicy.for_profile(profile)
    root_count = parse_parameter_count(root_parameters)
    return {
        "schema_version": DEPLOYMENT_STATE_SCHEMA_VERSION,
        "model": "MOIRA",
        "profile": policy.profile,
        "profile_changes_model_identity": False,
        "precision_policy": policy.to_dict(),
        "parameter_root": format_parameter_count(root_count),
        "parameter_root_parameters": root_count,
        "parameter_growth_policy": {
            "multiplier": PARAMETER_GROWTH_MULTIPLIER,
            "preview_steps": PARAMETER_GROWTH_PREVIEW_STEPS,
            "root_relative": True,
            "exact_integer_counts": True,
        },
        "parameter_growth_ladder": [
            format_parameter_count(value)
            for value in parameter_growth_ladder(root_count)
        ],
        "bit_precision_ladder": [
            format_bit_precision(value) for value in BIT_PRECISION_LADDER
        ],
        "whole_model_compute_dtypes": list(COMPUTE_DTYPE_LADDER),
        "verification_compute_dtypes": list(VERIFICATION_DTYPE_LADDER),
        "fp64_scope": "bounded-verification-arithmetic-only",
        "parameter_count_is_separate_from_precision": True,
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
    root_parameters = value.get("parameter_root_parameters")
    if (
        not isinstance(profile, str)
        or type(root_parameters) is not int
        or value != deployment_state(profile, root_parameters)
    ):
        raise ValueError("deployment state does not match the canonical schema")
    return value


def switch_deployment_profile(
    profile: str,
    path: str | Path,
    *,
    root_parameters: int | str | None = None,
) -> dict[str, object]:
    """Atomically persist a local policy selection and report the transition."""
    state_path = Path(path)
    parent = state_path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("deployment state parent must be a non-symlink directory")

    previous_profile: str | None = None
    selected_root = (
        DEFAULT_PARAMETER_ROOT
        if root_parameters is None
        else parse_parameter_count(root_parameters)
    )
    if os.path.lexists(state_path):
        previous_state = load_deployment_state(state_path)
        previous_profile = str(previous_state["profile"])
        previous_root = int(previous_state["parameter_root_parameters"])
        if root_parameters is not None and selected_root != previous_root:
            raise ValueError("deployment profile switch cannot change parameter root")
        selected_root = previous_root
    state = deployment_state(profile, selected_root)

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
