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


def test_capsule_digest_is_stable_for_equal_values_and_changes_for_header_or_payload() -> None:
    """Catches a digest that ignores metadata or canonical payload bytes."""
    capsule = make_capsule()

    assert capsule_digest(capsule) == capsule_digest(
        replace(capsule, latent=capsule.latent.clone())
    )
    assert capsule_digest(capsule) != capsule_digest(replace(capsule, task_id="task-2"))
    assert capsule_digest(capsule) != capsule_digest(
        replace(capsule, latent=capsule.latent + 1)
    )
