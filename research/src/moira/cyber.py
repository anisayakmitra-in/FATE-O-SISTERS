"""Authorization contracts for defensive and isolated MOIRA cyber research."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath

from .lifelong import EvidenceOutcome


class CyberNetworkPolicy(str, Enum):
    OFF = "off"
    TARGET_ALLOWLIST_ONLY = "target-allowlist-only"


class CyberCredentialPolicy(str, Enum):
    NONE = "none"
    SYNTHETIC_LAB_ONLY = "synthetic-lab-only"


@dataclass(frozen=True, slots=True)
class CyberResourceLimitsV1:
    cpu_seconds: int
    wall_seconds: int
    peak_memory_bytes: int
    writable_bytes: int
    process_count: int

    def validate(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.cpu_seconds,
                self.wall_seconds,
                self.peak_memory_bytes,
                self.writable_bytes,
                self.process_count,
            )
        ):
            raise ValueError("cyber resource limits must be positive integers")


@dataclass(frozen=True, slots=True)
class CyberLabScopeV1:
    """One expiring owner-authorized lab scope; it is not an effect permit."""

    scope_id: str
    owner_authorization_digest: str
    target_manifest_digest: str
    permitted_techniques: tuple[str, ...]
    isolated_worker_identity: str
    network_policy: CyberNetworkPolicy
    credential_policy: CyberCredentialPolicy
    writable_mounts: tuple[str, ...]
    resource_limits: CyberResourceLimitsV1
    expires_at_ms: int
    audit_destination: str
    production_targets_allowed: bool = False
    administrator_identity_allowed: bool = False
    host_credentials_allowed: bool = False
    one_use_effect_permits_required: bool = True

    def validate(self, *, now_ms: int) -> None:
        if type(now_ms) is not int or now_ms < 0:
            raise ValueError("now_ms must be a non-negative integer")
        for name, value in (
            ("scope_id", self.scope_id),
            ("isolated_worker_identity", self.isolated_worker_identity),
            ("audit_destination", self.audit_destination),
        ):
            _nonempty(name, value)
        _digest("owner_authorization_digest", self.owner_authorization_digest)
        _digest("target_manifest_digest", self.target_manifest_digest)
        if not isinstance(self.permitted_techniques, tuple) or not self.permitted_techniques:
            raise ValueError("permitted_techniques must be a unique nonempty tuple")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.permitted_techniques
        ) or len(set(self.permitted_techniques)) != len(self.permitted_techniques):
            raise ValueError("permitted_techniques must be a unique nonempty tuple")
        if not isinstance(self.network_policy, CyberNetworkPolicy):
            raise ValueError("network_policy must be a CyberNetworkPolicy")
        if not isinstance(self.credential_policy, CyberCredentialPolicy):
            raise ValueError("credential_policy must be a CyberCredentialPolicy")
        if not isinstance(self.resource_limits, CyberResourceLimitsV1):
            raise ValueError("resource_limits must be CyberResourceLimitsV1")
        self.resource_limits.validate()
        if type(self.expires_at_ms) is not int or self.expires_at_ms <= now_ms:
            raise ValueError("cyber lab scope is expired")
        _validate_mounts(self.writable_mounts)
        if self.production_targets_allowed is not False:
            raise ValueError("production targets are forbidden in the official cyber lab")
        if self.administrator_identity_allowed is not False:
            raise ValueError("administrator identity is forbidden in the cyber lab")
        if self.host_credentials_allowed is not False:
            raise ValueError("host credentials are forbidden in the cyber lab")
        if self.one_use_effect_permits_required is not True:
            raise ValueError("cyber lab effects require one-use permits")


@dataclass(frozen=True, slots=True)
class CyberLearningEvidenceV1:
    """Shape-only claim; trusted qualification must resolve its bound receipts."""

    lab_scope_id: str
    effect_receipt_digest: str
    evaluator_digest: str
    outcome: EvidenceOutcome
    independent: bool
    authorized_target: bool

    def validate_positive(self) -> None:
        _nonempty("lab_scope_id", self.lab_scope_id)
        _digest("effect_receipt_digest", self.effect_receipt_digest)
        _digest("evaluator_digest", self.evaluator_digest)
        if not isinstance(self.outcome, EvidenceOutcome):
            raise ValueError("cyber learning outcome must be typed evidence")
        if self.outcome is not EvidenceOutcome.PASS:
            raise ValueError("positive cyber learning requires a verified pass")
        if self.independent is not True:
            raise ValueError("positive cyber learning requires independent evidence")
        if self.authorized_target is not True:
            raise ValueError("unauthorized activity cannot become positive training evidence")


def _validate_mounts(mounts: object) -> None:
    if not isinstance(mounts, tuple):
        raise ValueError("writable_mounts must be a unique tuple")
    for value in mounts:
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or "\\" in value
            or ":" in value
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("writable mounts must be normalized relative paths")
        if any(part in {"", ".", ".."} for part in value.split("/")):
            raise ValueError("writable mounts must be normalized relative paths")
        path = PurePosixPath(value)
        if path.is_absolute() or path.as_posix() != value:
            raise ValueError("writable mounts must be normalized relative paths")
    if len(set(mounts)) != len(mounts):
        raise ValueError("writable_mounts must be a unique tuple")


def _nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _digest(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


__all__ = [
    "CyberCredentialPolicy",
    "CyberLabScopeV1",
    "CyberLearningEvidenceV1",
    "CyberNetworkPolicy",
    "CyberResourceLimitsV1",
]
