"""Pure-PyTorch capability encoders for the M1 capability loom.

The loom converts bounded local observations into validated capability
capsules.  It deliberately does not download or wrap third-party model
weights; external checkpoints remain optional teacher references in the source
catalog.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .capsule import CapabilityCapsule


_MAX_CAPSULE_SLOTS = 256
_CAPSULE_TYPES = 4


class VisionPatchEncoder(nn.Module):
    """Encode RGB images into a bounded sequence of spatial capsules."""

    def __init__(
        self,
        *,
        capsule_width: int,
        patch_size: int = 16,
        max_slots: int = 64,
    ) -> None:
        super().__init__()
        _validate_positive("capsule_width", capsule_width)
        _validate_positive("patch_size", patch_size)
        _validate_slots(max_slots)
        self.capsule_width = capsule_width
        self.patch_size = patch_size
        self.max_slots = max_slots
        self.projection = nn.Linear(3 * patch_size * patch_size, capsule_width)
        self.norm = nn.LayerNorm(capsule_width)

    def forward(
        self,
        images: Tensor,
        *,
        task_id: str,
        scope_id: str,
        source_hashes: Sequence[str] | None = None,
    ) -> tuple[CapabilityCapsule, ...]:
        batch, _, coordinates = self._validate_images(images)
        if source_hashes is None:
            hashes = tuple(_tensor_hash(images[index]) for index in range(batch))
        else:
            hashes = _validate_source_hashes(source_hashes, batch)
        patch_features = F.unfold(
            images,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        patch_features, coordinates = _bound_slots(
            patch_features, coordinates, self.max_slots
        )
        latent = self.norm(self.projection(patch_features))
        return tuple(
            _make_capsule(
                latent=latent[index],
                coordinates=coordinates,
                task_id=task_id,
                scope_id=scope_id,
                source_modality="vision",
                source_hash=hashes[index],
                capsule_type="vision-patch",
            )
            for index in range(batch)
        )

    def _validate_images(self, images: object) -> tuple[int, int, Tensor]:
        if (
            not isinstance(images, Tensor)
            or not images.is_floating_point()
            or images.ndim != 4
            or images.shape[1] != 3
            or images.device.type == "meta"
            or not bool(torch.isfinite(images).all())
        ):
            raise ValueError("images must be finite floating [batch, 3, height, width]")
        batch, _, height, width = images.shape
        if batch <= 0 or height < self.patch_size or width < self.patch_size:
            raise ValueError("images must contain a positive batch and one full patch")
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("image height and width must be divisible by patch_size")
        rows = height // self.patch_size
        columns = width // self.patch_size
        patch_count = rows * columns
        if patch_count <= 0:
            raise ValueError("images must produce at least one patch")
        y = (torch.arange(rows, device=images.device, dtype=images.dtype) + 0.5) / rows
        x = (torch.arange(columns, device=images.device, dtype=images.dtype) + 0.5) / columns
        coordinates = torch.cartesian_prod(y, x).reshape(patch_count, 2)
        return batch, patch_count, coordinates


class FeatureCapsuleEncoder(nn.Module):
    """Encode audio, time-series, graph, or robotics features into capsules."""

    def __init__(
        self,
        *,
        feature_dim: int,
        capsule_width: int,
        source_modality: str,
        max_slots: int = 64,
    ) -> None:
        super().__init__()
        _validate_positive("feature_dim", feature_dim)
        _validate_positive("capsule_width", capsule_width)
        _validate_slots(max_slots)
        if not isinstance(source_modality, str) or not source_modality.strip():
            raise ValueError("source_modality must be non-empty")
        self.feature_dim = feature_dim
        self.capsule_width = capsule_width
        self.source_modality = source_modality.strip()
        self.max_slots = max_slots
        self.projection = nn.Linear(feature_dim, capsule_width)
        self.norm = nn.LayerNorm(capsule_width)

    def forward(
        self,
        features: Tensor,
        *,
        task_id: str,
        scope_id: str,
        source_hashes: Sequence[str] | None = None,
    ) -> tuple[CapabilityCapsule, ...]:
        batch, _, coordinates = _validate_features(
            features, self.feature_dim
        )
        if source_hashes is None:
            hashes = tuple(_tensor_hash(features[index]) for index in range(batch))
        else:
            hashes = _validate_source_hashes(source_hashes, batch)
        bounded, coordinates = _bound_slots(
            features, coordinates, self.max_slots
        )
        latent = self.norm(self.projection(bounded))
        return tuple(
            _make_capsule(
                latent=latent[index],
                coordinates=coordinates,
                task_id=task_id,
                scope_id=scope_id,
                source_modality=self.source_modality,
                source_hash=hashes[index],
                capsule_type="feature-sequence",
            )
            for index in range(batch)
        )


class CapsuleProjector(nn.Module):
    """Project a validated capsule into a model hidden width."""

    def __init__(self, *, capsule_width: int, d_model: int) -> None:
        super().__init__()
        _validate_positive("capsule_width", capsule_width)
        _validate_positive("d_model", d_model)
        self.capsule_width = capsule_width
        self.d_model = d_model
        self.projection = nn.Linear(capsule_width, d_model)

    def forward(
        self,
        capsule: CapabilityCapsule,
        *,
        task_id: str | None = None,
        allowed_scope_ids: Sequence[str] | None = None,
    ) -> Tensor:
        capsule.validate(
            expected_width=self.capsule_width,
            n_slot_types=_CAPSULE_TYPES,
            expected_task_id=task_id,
            allowed_scope_ids=allowed_scope_ids,
        )
        projected = self.projection(capsule.latent)
        if not bool(torch.isfinite(projected).all()):
            raise ValueError("projected capsule contains non-finite values")
        return projected


def _make_capsule(
    *,
    latent: Tensor,
    coordinates: Tensor,
    task_id: str,
    scope_id: str,
    source_modality: str,
    source_hash: str,
    capsule_type: str,
) -> CapabilityCapsule:
    capsule = CapabilityCapsule(
        abi_version=1,
        task_id=task_id,
        scope_id=scope_id,
        capsule_type=capsule_type,
        source_modality=source_modality,
        evidence_status="observed",
        source_hashes=(source_hash,),
        latent=latent,
        slot_type_ids=torch.zeros(
            latent.shape[0], dtype=torch.int64, device=latent.device
        ),
        uncertainty=torch.full(
            (latent.shape[0],), 0.25, dtype=latent.dtype, device=latent.device
        ),
        coordinates=coordinates,
    )
    capsule.validate(
        expected_width=latent.shape[1],
        n_slot_types=_CAPSULE_TYPES,
    )
    return capsule


def _validate_features(features: object, feature_dim: int) -> tuple[int, int, Tensor]:
    if (
        not isinstance(features, Tensor)
        or not features.is_floating_point()
        or features.ndim != 3
        or features.shape[2] != feature_dim
        or features.device.type == "meta"
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("features must be finite floating [batch, sequence, feature]")
    batch, length, _ = features.shape
    if batch <= 0 or length <= 0:
        raise ValueError("features must have positive batch and sequence")
    coordinates = (
        (torch.arange(length, device=features.device, dtype=features.dtype) + 0.5)
        / length
    ).unsqueeze(-1)
    return batch, length, coordinates


def _bound_slots(
    values: Tensor,
    coordinates: Tensor,
    max_slots: int,
) -> tuple[Tensor, Tensor]:
    length = values.shape[1]
    if length <= max_slots:
        return values, coordinates
    groups = []
    coordinate_groups = []
    for index in range(max_slots):
        start = index * length // max_slots
        end = (index + 1) * length // max_slots
        groups.append(values[:, start:end].mean(dim=1))
        coordinate_groups.append(coordinates[start:end].mean(dim=0))
    return torch.stack(groups, dim=1), torch.stack(coordinate_groups, dim=0)


def _validate_source_hashes(values: Sequence[str], batch: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or len(values) != batch:
        raise ValueError("source_hashes must contain one hash per batch item")
    hashes = tuple(value for value in values if isinstance(value, str) and value.strip())
    if len(hashes) != batch:
        raise ValueError("source_hashes must contain non-empty strings")
    return hashes


def _tensor_hash(tensor: Tensor) -> str:
    payload = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    descriptor = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode("utf-8")
    return "sha256:" + sha256(descriptor + payload).hexdigest()


def _validate_positive(name: str, value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_slots(value: int) -> None:
    _validate_positive("max_slots", value)
    if value > _MAX_CAPSULE_SLOTS:
        raise ValueError("max_slots exceeds the bounded capsule capacity")
