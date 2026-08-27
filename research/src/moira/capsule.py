"""Typed capability capsules and their in-memory integrity evidence."""

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Collection

import torch
from torch import Tensor


CAPSULE_ABI_VERSION = 1
CAPABILITY_OBSERVATION_ABI_VERSION = 2


class CapabilityEvidenceClass(str, Enum):
    UNTRUSTED_RETRIEVAL = "untrusted-retrieval"
    WEAK_BEHAVIOR = "weak-behavior"
    EXPLICIT_CORRECTION = "explicit-correction"
    DURABLE_ACCEPTED_EDIT = "durable-accepted-edit"
    INDEPENDENT_VERIFICATION = "independent-verification"


@dataclass(frozen=True, slots=True)
class CapabilityResourceCostV1:
    cpu_time_ms: int
    gpu_time_ms: int
    peak_memory_bytes: int

    def validate(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (self.cpu_time_ms, self.gpu_time_ms, self.peak_memory_bytes)
        ):
            raise ValueError("resource cost values must be non-negative integers")


@dataclass(frozen=True, slots=True)
class CapabilityObservationV2:
    """A source-bound capability observation; receipt references grant no authority."""

    abi_version: int
    capability_id: str
    capability_version: str
    task_id: str
    scope_id: str
    owner_or_tenant: str
    modality: str
    evidence_class: CapabilityEvidenceClass
    raw_artifact_digest: str
    feature_digest: str
    preprocessing_digest: str
    encoder_digest: str
    latent: Tensor
    uncertainty: Tensor | None
    coordinates: Tensor | None
    resource_cost: CapabilityResourceCostV1
    permission_receipt_ref: str

    def validate(
        self,
        *,
        expected_width: int,
        feature_tensor: Tensor,
        expected_raw_artifact_digest: str,
    ) -> None:
        if self.abi_version != CAPABILITY_OBSERVATION_ABI_VERSION:
            raise ValueError("unsupported capability observation ABI version")
        for name, value in (
            ("capability_id", self.capability_id),
            ("capability_version", self.capability_version),
            ("task_id", self.task_id),
            ("scope_id", self.scope_id),
            ("owner_or_tenant", self.owner_or_tenant),
            ("modality", self.modality),
            ("permission_receipt_ref", self.permission_receipt_ref),
        ):
            _require_nonempty(name, value)
        if not isinstance(self.evidence_class, CapabilityEvidenceClass):
            raise ValueError("evidence_class must be a CapabilityEvidenceClass")
        for name, value in (
            ("raw_artifact_digest", self.raw_artifact_digest),
            ("feature_digest", self.feature_digest),
            ("preprocessing_digest", self.preprocessing_digest),
            ("encoder_digest", self.encoder_digest),
        ):
            _require_sha256(name, value)
        if type(expected_width) is not int or expected_width <= 0:
            raise ValueError("expected_width must be positive")
        _require_floating_matrix("latent", self.latent)
        slots, width = self.latent.shape
        if slots <= 0 or width != expected_width:
            raise ValueError("latent shape does not match expected_width")
        if self.uncertainty is not None:
            _require_floating_vector("uncertainty", self.uncertainty, slots)
            if torch.any((self.uncertainty < 0) | (self.uncertainty > 1)):
                raise ValueError("uncertainty must be in [0, 1] or explicitly unknown")
            _require_same_device(self.latent, self.uncertainty)
        if self.coordinates is not None:
            _require_floating_matrix("coordinates", self.coordinates)
            if self.coordinates.shape[0] != slots or self.coordinates.shape[1] <= 0:
                raise ValueError("coordinates slots must match latent slots")
            _require_same_device(self.latent, self.coordinates)
        if not isinstance(self.resource_cost, CapabilityResourceCostV1):
            raise ValueError("resource_cost must be CapabilityResourceCostV1")
        self.resource_cost.validate()
        if tensor_content_digest(feature_tensor) != self.feature_digest:
            raise ValueError("feature tensor digest does not match the observation")
        _require_sha256("expected_raw_artifact_digest", expected_raw_artifact_digest)
        if expected_raw_artifact_digest != self.raw_artifact_digest:
            raise ValueError("raw artifact digest does not match the observation")

    @property
    def observation_digest(self) -> str:
        digest = sha256()
        header = {
            "abi_version": self.abi_version,
            "capability_id": self.capability_id,
            "capability_version": self.capability_version,
            "encoder_digest": self.encoder_digest,
            "evidence_class": (
                self.evidence_class.value
                if isinstance(self.evidence_class, CapabilityEvidenceClass)
                else self.evidence_class
            ),
            "feature_digest": self.feature_digest,
            "modality": self.modality,
            "owner_or_tenant": self.owner_or_tenant,
            "permission_receipt_ref": self.permission_receipt_ref,
            "preprocessing_digest": self.preprocessing_digest,
            "raw_artifact_digest": self.raw_artifact_digest,
            "resource_cost": {
                "cpu_time_ms": self.resource_cost.cpu_time_ms,
                "gpu_time_ms": self.resource_cost.gpu_time_ms,
                "peak_memory_bytes": self.resource_cost.peak_memory_bytes,
            },
            "scope_id": self.scope_id,
            "task_id": self.task_id,
            "uncertainty": "unknown" if self.uncertainty is None else "calibrated",
        }
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        for name in ("latent", "uncertainty", "coordinates"):
            tensor = getattr(self, name)
            if tensor is not None:
                _update_tensor(digest, name, tensor)
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CapabilityCapsule:
    abi_version: int
    task_id: str
    scope_id: str
    capsule_type: str
    source_modality: str
    evidence_status: str
    source_hashes: tuple[str, ...]
    latent: Tensor
    slot_type_ids: Tensor
    uncertainty: Tensor
    coordinates: Tensor | None = None

    def validate(
        self,
        *,
        expected_width: int,
        n_slot_types: int,
        expected_task_id: str | None = None,
        allowed_scope_ids: Collection[str] | None = None,
    ) -> None:
        """Reject malformed capsules before they enter a model boundary."""
        if self.abi_version != CAPSULE_ABI_VERSION:
            raise ValueError("unsupported capsule ABI version")
        _require_nonempty("task_id", self.task_id)
        _require_nonempty("scope_id", self.scope_id)
        _require_nonempty("capsule_type", self.capsule_type)
        _require_nonempty("source_modality", self.source_modality)
        _require_nonempty("evidence_status", self.evidence_status)
        if not isinstance(self.source_hashes, tuple) or not self.source_hashes:
            raise ValueError("source_hashes must be a nonempty tuple")
        if any(not isinstance(value, str) or not value.strip() for value in self.source_hashes):
            raise ValueError("source_hashes must contain nonempty strings")
        if not isinstance(expected_width, int) or expected_width <= 0:
            raise ValueError("expected_width must be positive")
        if not isinstance(n_slot_types, int) or n_slot_types <= 0:
            raise ValueError("n_slot_types must be positive")
        if expected_task_id is not None and self.task_id != expected_task_id:
            raise ValueError("task_id does not match the expected task")
        if allowed_scope_ids is not None:
            _validate_allowed_scope_ids(allowed_scope_ids)
            if self.scope_id not in allowed_scope_ids:
                raise ValueError("scope_id is not allowed")

        _require_floating_matrix("latent", self.latent)
        slots, width = self.latent.shape
        if slots <= 0:
            raise ValueError("latent slots must be nonempty")
        if width != expected_width:
            raise ValueError("latent width does not match expected_width")
        _require_int64_vector("slot_type_ids", self.slot_type_ids, slots)
        if torch.any((self.slot_type_ids < 0) | (self.slot_type_ids >= n_slot_types)):
            raise ValueError("slot_type_ids must be within the declared type range")
        _require_floating_vector("uncertainty", self.uncertainty, slots)
        if torch.any((self.uncertainty < 0) | (self.uncertainty > 1)):
            raise ValueError("uncertainty must be in [0, 1]")
        _require_same_device(self.latent, self.slot_type_ids, self.uncertainty)
        if self.coordinates is not None:
            _require_floating_matrix("coordinates", self.coordinates)
            if self.coordinates.shape[0] != slots or self.coordinates.shape[1] <= 0:
                raise ValueError("coordinates slots must match latent slots")
            _require_same_device(self.latent, self.coordinates)


def validate_capsule(
    capsule: CapabilityCapsule,
    *,
    expected_task_id: str,
    allowed_scope_ids: Collection[str],
    expected_width: int,
    n_slot_types: int,
) -> None:
    """Validate a capsule against its receiving task and scope boundary."""
    if not isinstance(capsule, CapabilityCapsule):
        raise ValueError("capsule must be a CapabilityCapsule")
    _require_nonempty("expected_task_id", expected_task_id)
    _validate_allowed_scope_ids(allowed_scope_ids)
    capsule.validate(
        expected_task_id=expected_task_id,
        allowed_scope_ids=allowed_scope_ids,
        expected_width=expected_width,
        n_slot_types=n_slot_types,
    )


def capsule_digest(capsule: CapabilityCapsule) -> str:
    """Return canonical SHA-256 evidence for a capsule's current memory values."""
    if not isinstance(capsule, CapabilityCapsule):
        raise ValueError("capsule must be a CapabilityCapsule")
    digest = sha256()
    digest.update(_header_bytes(capsule))
    for name in ("latent", "slot_type_ids", "uncertainty", "coordinates"):
        tensor = getattr(capsule, name)
        if tensor is not None:
            _update_tensor(digest, name, tensor)
    return digest.hexdigest()


def tensor_content_digest(tensor: Tensor) -> str:
    """Hash one concrete tensor with its dtype and shape as source evidence."""
    if (
        not isinstance(tensor, Tensor)
        or tensor.device.type == "meta"
        or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
    ):
        raise ValueError("tensor must be concrete and finite")
    digest = sha256()
    _update_tensor(digest, "feature", tensor)
    return digest.hexdigest()


def _require_nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _require_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_allowed_scope_ids(allowed_scope_ids: Collection[str]) -> None:
    if isinstance(allowed_scope_ids, (str, bytes)) or not allowed_scope_ids:
        raise ValueError("allowed_scope_ids must be a nonempty collection")
    if any(not isinstance(scope_id, str) or not scope_id.strip() for scope_id in allowed_scope_ids):
        raise ValueError("allowed_scope_ids must contain nonempty strings")


def _require_tensor(name: str, value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.device.type == "meta":
        raise ValueError(f"{name} must have a concrete device")
    return value


def _require_floating_matrix(name: str, value: object) -> None:
    tensor = _require_tensor(name, value)
    if not tensor.is_floating_point() or tensor.ndim != 2:
        raise ValueError(f"{name} must be a floating [slots, width] tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")


def _require_floating_vector(name: str, value: object, slots: int) -> None:
    tensor = _require_tensor(name, value)
    if not tensor.is_floating_point() or tensor.shape != (slots,):
        raise ValueError(f"{name} must be a floating [slots] tensor")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")


def _require_int64_vector(name: str, value: object, slots: int) -> None:
    tensor = _require_tensor(name, value)
    if tensor.dtype != torch.int64 or tensor.shape != (slots,):
        raise ValueError(f"{name} must be int64 [slots]")


def _require_same_device(reference: Tensor, *tensors: Tensor) -> None:
    if any(tensor.device != reference.device for tensor in tensors):
        raise ValueError("capsule tensors must use the same device")


def _header_bytes(capsule: CapabilityCapsule) -> bytes:
    header = {
        "abi_version": capsule.abi_version,
        "capsule_type": capsule.capsule_type,
        "evidence_status": capsule.evidence_status,
        "scope_id": capsule.scope_id,
        "source_hashes": capsule.source_hashes,
        "source_modality": capsule.source_modality,
        "task_id": capsule.task_id,
    }
    return json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _update_tensor(digest: object, name: str, tensor: Tensor) -> None:
    if tensor.device.type == "meta":
        raise ValueError(f"{name} must have a concrete device")
    descriptor = json.dumps(
        {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(descriptor)  # type: ignore[union-attr]
    payload = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    digest.update(payload)  # type: ignore[union-attr]
