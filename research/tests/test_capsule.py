from dataclasses import replace

import pytest
import torch

from moira.capsule import CapabilityCapsule, capsule_digest, validate_capsule


def make_capsule(**changes: object) -> CapabilityCapsule:
    """Builds a literal valid v1 capsule for contract tests."""
    values: dict[str, object] = {
        "abi_version": 1,
        "task_id": "task-1",
        "scope_id": "owner:anis",
        "capsule_type": "retrieval",
        "source_modality": "text",
        "evidence_status": "observed",
        "source_hashes": ("sha256:source-a",),
        "latent": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "slot_type_ids": torch.tensor([0, 2], dtype=torch.int64),
        "uncertainty": torch.tensor([0.0, 1.0]),
        "coordinates": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
    }
    values.update(changes)
    return CapabilityCapsule(**values)  # type: ignore[arg-type]


def test_capsule_accepts_a_complete_v1_record() -> None:
    """Catches rejecting a complete capsule that matches every ABI boundary."""
    capsule = make_capsule()

    validate_capsule(
        capsule,
        expected_task_id="task-1",
        allowed_scope_ids={"owner:anis"},
        expected_width=3,
        n_slot_types=3,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("abi_version", 2, "ABI"),
        ("task_id", "", "task_id"),
        ("scope_id", "", "scope"),
        ("capsule_type", "", "capsule_type"),
        ("source_modality", "", "source_modality"),
        ("evidence_status", "", "evidence_status"),
        ("source_hashes", (), "source_hashes"),
    ],
)
def test_capsule_rejects_missing_or_unknown_header_values(
    field: str, value: object, message: str
) -> None:
    """Catches accepting an ABI/header/provenance field without an explicit value."""
    with pytest.raises(ValueError, match=message):
        make_capsule(**{field: value}).validate(expected_width=3, n_slot_types=3)


def test_capsule_rejects_task_and_scope_outside_the_call_contract() -> None:
    """Catches a capsule crossing a task or owner-scope boundary."""
    capsule = make_capsule()

    with pytest.raises(ValueError, match="task_id"):
        validate_capsule(
            capsule,
            expected_task_id="task-2",
            allowed_scope_ids={"owner:anis"},
            expected_width=3,
            n_slot_types=3,
        )
    with pytest.raises(ValueError, match="scope"):
        validate_capsule(
            capsule,
            expected_task_id="task-1",
            allowed_scope_ids={"tenant:other"},
            expected_width=3,
            n_slot_types=3,
        )


@pytest.mark.parametrize(
    "allowed_scope_ids", ["prefix-owner:anis-suffix", b"owner:anis"]
)
def test_capsule_rejects_string_like_scope_allowlists(
    allowed_scope_ids: object,
) -> None:
    """Catches substring membership allowing a scope through a scalar allowlist."""
    with pytest.raises(ValueError, match="allowed_scope_ids"):
        validate_capsule(
            make_capsule(),
            expected_task_id="task-1",
            allowed_scope_ids=allowed_scope_ids,  # type: ignore[arg-type]
            expected_width=3,
            n_slot_types=3,
        )


@pytest.mark.parametrize(
    "allowed_scope_ids", ["prefix-owner:anis-suffix", b"owner:anis"]
)
def test_capsule_direct_validation_rejects_string_like_scope_allowlists(
    allowed_scope_ids: object,
) -> None:
    """Catches direct validation bypassing the wrapper's scope allowlist guard."""
    with pytest.raises(ValueError, match="allowed_scope_ids"):
        make_capsule().validate(
            expected_width=3,
            n_slot_types=3,
            allowed_scope_ids=allowed_scope_ids,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "latent",
    [
        torch.ones((2, 3), dtype=torch.int64),
        torch.ones((2, 3, 1)),
        torch.tensor([[float("nan"), 0.0, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor([[float("inf"), 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ],
)
def test_capsule_rejects_malformed_latent(latent: torch.Tensor) -> None:
    """Catches invalid payload dtype, rank, or finite-value validation."""
    with pytest.raises(ValueError, match="latent"):
        make_capsule(latent=latent).validate(expected_width=3, n_slot_types=3)


@pytest.mark.parametrize(
    "slot_type_ids",
    [
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([[0, 1]], dtype=torch.int64),
        torch.tensor([0, 3], dtype=torch.int64),
        torch.tensor([-1, 1], dtype=torch.int64),
    ],
)
def test_capsule_rejects_invalid_slot_types(slot_type_ids: torch.Tensor) -> None:
    """Catches slot IDs with the wrong type, shape, or declared type range."""
    with pytest.raises(ValueError, match="slot_type_ids"):
        make_capsule(slot_type_ids=slot_type_ids).validate(expected_width=3, n_slot_types=3)


@pytest.mark.parametrize(
    "uncertainty",
    [
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([float("nan"), 0.5]),
        torch.tensor([-0.1, 0.5]),
        torch.tensor([0.5, 1.1]),
    ],
)
def test_capsule_rejects_invalid_uncertainty(uncertainty: torch.Tensor) -> None:
    """Catches uncertainty that cannot represent finite per-slot confidence."""
    with pytest.raises(ValueError, match="uncertainty"):
        make_capsule(uncertainty=uncertainty).validate(expected_width=3, n_slot_types=3)


@pytest.mark.parametrize(
    "coordinates",
    [
        torch.ones((2, 2), dtype=torch.int64),
        torch.ones((2,)),
        torch.ones((3, 2)),
        torch.tensor([[0.0, float("nan")], [1.0, 2.0]]),
    ],
)
def test_capsule_rejects_invalid_coordinates(coordinates: torch.Tensor) -> None:
    """Catches coordinate metadata that cannot align with capsule slots."""
    with pytest.raises(ValueError, match="coordinates"):
        make_capsule(coordinates=coordinates).validate(expected_width=3, n_slot_types=3)


def test_capsule_rejects_tensor_shape_width_and_device_mismatches() -> None:
    """Catches tensors that cannot form one same-device slotted payload."""
    with pytest.raises(ValueError, match="expected_width"):
        make_capsule().validate(expected_width=4, n_slot_types=3)
    with pytest.raises(ValueError, match="slots"):
        make_capsule(uncertainty=torch.tensor([0.5])).validate(
            expected_width=3, n_slot_types=3
        )
    with pytest.raises(ValueError, match="device"):
        make_capsule(
            slot_type_ids=torch.empty((2,), dtype=torch.int64, device="meta")
        ).validate(expected_width=3, n_slot_types=3)


def test_capsule_rejects_zero_slots() -> None:
    """Catches an empty capsule that carries no typed capability slots."""
    with pytest.raises(ValueError, match="slots"):
        make_capsule(
            latent=torch.empty((0, 3)),
            slot_type_ids=torch.empty((0,), dtype=torch.int64),
            uncertainty=torch.empty((0,)),
            coordinates=None,
        ).validate(expected_width=3, n_slot_types=3)


def test_capsule_rejects_empty_coordinate_dimensions() -> None:
    """Catches coordinate metadata with no declared spatial dimensions."""
    with pytest.raises(ValueError, match="coordinates"):
        make_capsule(coordinates=torch.empty((2, 0))).validate(
            expected_width=3, n_slot_types=3
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("abi_version", 2),
        ("task_id", "task-2"),
        ("scope_id", "owner:other"),
        ("capsule_type", "verification"),
        ("source_modality", "audio"),
        ("evidence_status", "derived"),
        ("source_hashes", ("sha256:source-b",)),
        ("latent", torch.tensor([[7.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        ("slot_type_ids", torch.tensor([1, 2], dtype=torch.int64)),
        ("uncertainty", torch.tensor([0.5, 1.0])),
        ("coordinates", torch.tensor([[7.0, 1.0], [2.0, 3.0]])),
    ],
)
def test_capsule_digest_changes_for_every_header_and_tensor_field(
    field: str, value: object
) -> None:
    """Catches a canonical digest omitting any v1 capsule field."""
    capsule = make_capsule()

    assert capsule_digest(capsule) != capsule_digest(replace(capsule, **{field: value}))


def test_capsule_digest_is_equal_for_noncontiguous_equal_values_and_bfloat16() -> None:
    """Catches hashing tensor storage layout rather than canonical tensor bytes."""
    capsule = make_capsule()
    noncontiguous = capsule.latent.transpose(0, 1).contiguous().transpose(0, 1)
    bfloat = make_capsule(
        latent=capsule.latent.to(torch.bfloat16),
        uncertainty=capsule.uncertainty.to(torch.bfloat16),
        coordinates=capsule.coordinates.to(torch.bfloat16),
    )

    assert not noncontiguous.is_contiguous()
    assert capsule_digest(capsule) == capsule_digest(
        replace(capsule, latent=noncontiguous)
    )
    assert capsule_digest(bfloat) == capsule_digest(
        replace(bfloat, latent=bfloat.latent.clone())
    )


def test_capsule_digest_includes_tensor_dtype_and_shape_descriptors() -> None:
    """Catches equal storage bytes masking a changed tensor dtype or shape."""
    capsule = make_capsule()
    changed_dtype = capsule.latent.view(torch.int32)
    changed_shape = capsule.latent.reshape(3, 2)

    assert changed_dtype.element_size() == capsule.latent.element_size()
    assert changed_dtype.numel() == capsule.latent.numel()
    assert changed_shape.numel() == capsule.latent.numel()
    assert capsule_digest(capsule) != capsule_digest(
        replace(capsule, latent=changed_dtype)
    )
    assert capsule_digest(capsule) != capsule_digest(
        replace(capsule, latent=changed_shape)
    )


def test_capsule_digest_does_not_mutate_or_bump_input_tensor_versions() -> None:
    """Catches integrity calculation changing caller-owned payload tensors."""
    capsule = make_capsule(
        latent=torch.arange(6.0).reshape(3, 2).transpose(0, 1),
        slot_type_ids=torch.tensor([0, 1], dtype=torch.int64),
        uncertainty=torch.tensor([0.2, 0.8]),
        coordinates=torch.tensor([[1.0], [2.0]]),
    )
    tensors = (
        capsule.latent,
        capsule.slot_type_ids,
        capsule.uncertainty,
        capsule.coordinates,
    )
    originals = tuple(tensor.clone() for tensor in tensors)
    versions = tuple(tensor._version for tensor in tensors)

    capsule_digest(capsule)

    assert tuple(tensor._version for tensor in tensors) == versions
    assert all(torch.equal(tensor, original) for tensor, original in zip(tensors, originals))
