import pytest
import torch

from moira.loom import CapsuleProjector, FeatureCapsuleEncoder, VisionPatchEncoder


def test_vision_encoder_emits_bounded_valid_capsules_and_gradients() -> None:
    encoder = VisionPatchEncoder(capsule_width=8, patch_size=2, max_slots=3)
    images = torch.rand(2, 3, 4, 4)

    capsules = encoder(
        images,
        task_id="vision-task",
        scope_id="owner:test",
        source_hashes=("sha256:image-a", "sha256:image-b"),
    )

    assert len(capsules) == 2
    assert capsules[0].latent.shape == (3, 8)
    assert capsules[0].coordinates is not None
    assert capsules[0].coordinates.shape == (3, 2)
    capsules[0].validate(expected_width=8, n_slot_types=4)
    capsules[0].latent.square().mean().backward()
    assert encoder.projection.weight.grad is not None


def test_feature_encoder_and_projector_cover_audio_time_series_and_robotics() -> None:
    encoder = FeatureCapsuleEncoder(
        feature_dim=3,
        capsule_width=8,
        source_modality="time-series",
        max_slots=3,
    )
    features = torch.randn(1, 5, 3)
    capsule = encoder(
        features,
        task_id="sequence-task",
        scope_id="tenant:test",
    )[0]
    projector = CapsuleProjector(capsule_width=8, d_model=12)
    projected = projector(
        capsule,
        task_id="sequence-task",
        allowed_scope_ids=("tenant:test",),
    )

    assert capsule.source_modality == "time-series"
    assert capsule.latent.shape == (3, 8)
    assert projected.shape == (3, 12)
    assert torch.isfinite(projected).all()


def test_loom_rejects_unbounded_or_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="divisible"):
        VisionPatchEncoder(capsule_width=8, patch_size=3)(
            torch.rand(1, 3, 4, 4),
            task_id="task",
            scope_id="scope",
        )
    encoder = FeatureCapsuleEncoder(
        feature_dim=2,
        capsule_width=4,
        source_modality="audio",
    )
    with pytest.raises(ValueError, match="one hash"):
        encoder(
            torch.rand(2, 4, 2),
            task_id="task",
            scope_id="scope",
            source_hashes=("sha256:only-one",),
        )
