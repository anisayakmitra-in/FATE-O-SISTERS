"""Fail-closed provenance manifests for MOIRA agent trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any, Mapping


AGENT_DATASET_MANIFEST_VERSION = 1
MAX_AGENT_DATASET_MANIFEST_BYTES = 64 * 1024
MAX_AGENT_TRAJECTORY_BYTES = 128 * 1024 * 1024
MAX_AGENT_DATASET_RECORDS = 100_000
_DATASET_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = frozenset({"train", "evaluation"})
_PROVENANCE_CLASSES = frozenset(
    {
        "synthetic",
        "licensed-public",
        "owner-consented",
        "enterprise-consented",
    }
)
_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "split",
        "trajectory_file",
        "trajectory_sha256",
        "record_count",
        "provenance_class",
        "source_uri",
        "source_revision",
        "license",
        "license_url",
        "license_verified",
        "consent_reference",
        "consent_verified",
        "tenant_scope",
        "contains_personal_data",
        "redistribution_allowed",
        "deletion_policy",
    }
)


@dataclass(frozen=True, slots=True)
class AgentDatasetManifest:
    schema_version: int
    dataset_id: str
    split: str
    trajectory_file: str
    trajectory_sha256: str
    record_count: int
    provenance_class: str
    source_uri: str
    source_revision: str
    license: str
    license_url: str
    license_verified: bool
    consent_reference: str
    consent_verified: bool
    tenant_scope: str
    contains_personal_data: bool
    redistribution_allowed: bool
    deletion_policy: str
    manifest_path: Path
    trajectory_path: Path
    manifest_sha256: str

    def checkpoint_fields(self) -> dict[str, object]:
        """Return immutable provenance fields bound into a training checkpoint."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_split": self.split,
            "dataset_manifest_sha256": self.manifest_sha256,
            "dataset_sha256": self.trajectory_sha256,
            "dataset_record_count": self.record_count,
            "provenance_class": self.provenance_class,
            "source_revision": self.source_revision,
            "license_verified": self.license_verified,
            "consent_verified": self.consent_verified,
            "tenant_scope": self.tenant_scope,
            "contains_personal_data": self.contains_personal_data,
            "redistribution_allowed": self.redistribution_allowed,
        }


def load_agent_dataset_manifest(
    path: str | Path,
    *,
    expected_split: str | None = None,
) -> AgentDatasetManifest:
    """Load one exact JSON manifest and verify its local trajectory artifact."""
    source = Path(path)
    raw_bytes = _read_regular_file(
        source,
        maximum=MAX_AGENT_DATASET_MANIFEST_BYTES,
        label="agent dataset manifest",
    )
    try:
        value = json.loads(raw_bytes, object_pairs_hook=_unique_json_object)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError("agent dataset manifest is not valid JSON") from error
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        raise ValueError("agent dataset manifest fields do not match the exact schema")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != AGENT_DATASET_MANIFEST_VERSION
    ):
        raise ValueError("agent dataset manifest schema version is invalid")

    dataset_id = _text("dataset_id", value["dataset_id"], 128)
    if not _DATASET_ID.fullmatch(dataset_id):
        raise ValueError("agent dataset_id must be a stable lowercase identifier")
    split = _text("split", value["split"], 32)
    if split not in _SPLITS:
        raise ValueError("agent dataset split must be train or evaluation")
    if expected_split is not None and split != expected_split:
        raise ValueError("agent dataset manifest has the wrong split")
    trajectory_file = _text("trajectory_file", value["trajectory_file"], 4_096)
    trajectory_sha256 = _text(
        "trajectory_sha256", value["trajectory_sha256"], 64
    )
    if not _SHA256.fullmatch(trajectory_sha256):
        raise ValueError("agent trajectory_sha256 must be a lowercase SHA-256 digest")
    record_count = value["record_count"]
    if (
        type(record_count) is not int
        or not 1 <= record_count <= MAX_AGENT_DATASET_RECORDS
    ):
        raise ValueError("agent dataset record_count is outside its bounded range")
    provenance_class = _text(
        "provenance_class", value["provenance_class"], 64
    )
    if provenance_class not in _PROVENANCE_CLASSES:
        raise ValueError("agent dataset provenance_class is unsupported")

    fields = {
        name: _text(name, value[name], maximum)
        for name, maximum in (
            ("source_uri", 2_048),
            ("source_revision", 256),
            ("license", 256),
            ("license_url", 2_048),
            ("consent_reference", 512),
            ("tenant_scope", 256),
            ("deletion_policy", 512),
        )
    }
    contains_personal_data = value["contains_personal_data"]
    redistribution_allowed = value["redistribution_allowed"]
    license_verified = value["license_verified"]
    consent_verified = value["consent_verified"]
    if any(
        type(item) is not bool
        for item in (
            contains_personal_data,
            redistribution_allowed,
            license_verified,
            consent_verified,
        )
    ):
        raise ValueError("agent dataset review and privacy fields must be booleans")
    _validate_provenance(
        provenance_class=provenance_class,
        contains_personal_data=contains_personal_data,
        redistribution_allowed=redistribution_allowed,
        license_verified=license_verified,
        consent_verified=consent_verified,
        **fields,
    )

    trajectory_path = _resolve_trajectory(source, trajectory_file)
    trajectory_bytes = _read_regular_file(
        trajectory_path,
        maximum=MAX_AGENT_TRAJECTORY_BYTES,
        label="agent trajectory file",
    )
    observed_sha256 = hashlib.sha256(trajectory_bytes).hexdigest()
    if observed_sha256 != trajectory_sha256:
        raise ValueError("agent trajectory file does not match its manifest digest")
    observed_records = sum(bool(line.strip()) for line in trajectory_bytes.splitlines())
    if observed_records != record_count:
        raise ValueError("agent trajectory record count does not match its manifest")

    canonical = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return AgentDatasetManifest(
        schema_version=AGENT_DATASET_MANIFEST_VERSION,
        dataset_id=dataset_id,
        split=split,
        trajectory_file=trajectory_file,
        trajectory_sha256=trajectory_sha256,
        record_count=record_count,
        provenance_class=provenance_class,
        source_uri=fields["source_uri"],
        source_revision=fields["source_revision"],
        license=fields["license"],
        license_url=fields["license_url"],
        license_verified=license_verified,
        consent_reference=fields["consent_reference"],
        consent_verified=consent_verified,
        tenant_scope=fields["tenant_scope"],
        contains_personal_data=contains_personal_data,
        redistribution_allowed=redistribution_allowed,
        deletion_policy=fields["deletion_policy"],
        manifest_path=source.resolve(),
        trajectory_path=trajectory_path,
        manifest_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def _validate_provenance(
    *,
    provenance_class: str,
    source_uri: str,
    source_revision: str,
    license: str,
    license_url: str,
    license_verified: bool,
    consent_reference: str,
    consent_verified: bool,
    tenant_scope: str,
    contains_personal_data: bool,
    redistribution_allowed: bool,
    deletion_policy: str,
) -> None:
    del source_revision, redistribution_allowed
    if deletion_policy.casefold() in {"n/a", "none"}:
        raise ValueError("agent datasets require an actionable deletion policy")
    if provenance_class == "synthetic":
        if (
            not source_uri.startswith("synthetic://")
            or license != "N/A"
            or license_url != "N/A"
            or license_verified
            or consent_reference != "N/A"
            or consent_verified
            or tenant_scope != "public"
            or contains_personal_data
        ):
            raise ValueError("synthetic agent provenance fields are inconsistent")
        return
    if provenance_class == "licensed-public":
        if (
            not source_uri.startswith("https://")
            or license == "N/A"
            or not license_url.startswith("https://")
            or not license_verified
            or consent_reference != "N/A"
            or consent_verified
            or tenant_scope != "public"
        ):
            raise ValueError("licensed agent provenance fields are inconsistent")
        return
    expected_scheme = (
        "owner://" if provenance_class == "owner-consented" else "tenant://"
    )
    if (
        not source_uri.startswith(expected_scheme)
        or consent_reference == "N/A"
        or not consent_verified
        or license_verified
        or tenant_scope == "public"
    ):
        raise ValueError("consented agent provenance fields are inconsistent")


def _resolve_trajectory(manifest_path: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("agent trajectory_file must stay below the manifest directory")
    root = manifest_path.parent.resolve()
    candidate = manifest_path.parent / relative
    if candidate.is_symlink():
        raise ValueError("agent trajectory file may not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError("agent trajectory file is unavailable") from error
    if not resolved.is_relative_to(root):
        raise ValueError("agent trajectory_file escapes the manifest directory")
    return resolved


def _read_regular_file(path: Path, *, maximum: int, label: str) -> bytes:
    try:
        status = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if status.st_size <= 0 or status.st_size > maximum:
        raise ValueError(f"{label} exceeds its bounded size")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{label} could not be read") from error
    if len(data) != status.st_size:
        raise ValueError(f"{label} changed while it was read")
    return data


def _text(name: str, value: Any, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValueError(f"agent dataset {name} must be bounded non-empty text")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("JSON object keys must be unique")
        value[key] = item
    return value


__all__ = [
    "AGENT_DATASET_MANIFEST_VERSION",
    "MAX_AGENT_DATASET_MANIFEST_BYTES",
    "MAX_AGENT_DATASET_RECORDS",
    "AgentDatasetManifest",
    "load_agent_dataset_manifest",
]
