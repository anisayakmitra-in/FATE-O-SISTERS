from dataclasses import replace

import pytest

from moira.capabilities import (
    CAPABILITY_PORTS,
    INITIAL_CAPABILITY_STEP_ALLOCATION,
    CapabilityImplementation,
    CapabilityQualification,
    CapabilityQualificationManifestV1,
    declared_capability_manifests,
    declared_capability_ports,
    validate_capability_step_allocation,
)


_DIGEST = "a" * 64


def test_every_declared_capability_port_starts_honestly_untrained() -> None:
    manifests = declared_capability_manifests()
    ports = declared_capability_ports()

    assert len(CAPABILITY_PORTS) == 12
    assert len(ports) == 12
    assert {manifest.capability_id for manifest in manifests} == set(CAPABILITY_PORTS)
    assert {port.capability_id for port in ports} == set(CAPABILITY_PORTS)
    for port in ports:
        port.validate()
    for manifest in manifests:
        manifest.validate()
        assert manifest.state is CapabilityQualification.DECLARED
        assert manifest.native_competence_claimed is False


def test_initial_step_allocation_is_exact_and_complete() -> None:
    validate_capability_step_allocation(INITIAL_CAPABILITY_STEP_ALLOCATION)
    assert sum(INITIAL_CAPABILITY_STEP_ALLOCATION.values()) == 100

    with pytest.raises(ValueError, match="six fixed groups"):
        validate_capability_step_allocation(
            {**INITIAL_CAPABILITY_STEP_ALLOCATION, "extra": 1}
        )


def test_smoke_training_requires_locked_evidence_and_baseline_improvement() -> None:
    base = CapabilityQualificationManifestV1(
        capability_id="defensive-cyber",
        capability_version="0.1.0",
        state=CapabilityQualification.SMOKE_TRAINED,
        implementation=CapabilityImplementation.NATIVE_CORE,
        core_parameters=33_032_562,
        organ_parameters=0,
        training_lineage_digest=_DIGEST,
        frozen_baseline_digest=_DIGEST,
        heldout_manifest_digest=_DIGEST,
        evaluator_digest=_DIGEST,
        baseline_score=0.1,
        candidate_score=0.2,
        native_competence_claimed=True,
    )

    base.validate()
    with pytest.raises(ValueError, match="beating"):
        replace(base, candidate_score=0.1).validate()
    with pytest.raises(ValueError, match="lineage"):
        replace(base, evaluator_digest=None).validate()
    with pytest.raises(ValueError, match="untrained capability"):
        replace(
            base,
            state=CapabilityQualification.DECLARED,
            training_lineage_digest=None,
            frozen_baseline_digest=None,
            heldout_manifest_digest=None,
            evaluator_digest=None,
            baseline_score=None,
            candidate_score=None,
        ).validate()


def test_higher_qualification_labels_require_their_own_evidence() -> None:
    smoke = CapabilityQualificationManifestV1(
        capability_id="defensive-cyber",
        capability_version="0.1.0",
        state=CapabilityQualification.SMOKE_TRAINED,
        implementation=CapabilityImplementation.NATIVE_CORE,
        core_parameters=33_032_562,
        organ_parameters=0,
        training_lineage_digest=_DIGEST,
        frozen_baseline_digest=_DIGEST,
        heldout_manifest_digest=_DIGEST,
        evaluator_digest=_DIGEST,
        baseline_score=0.1,
        candidate_score=0.2,
    )

    with pytest.raises(ValueError, match="benchmark_receipt_digest"):
        replace(
            smoke, state=CapabilityQualification.BENCHMARK_QUALIFIED
        ).validate()
    benchmark = replace(
        smoke,
        state=CapabilityQualification.BENCHMARK_QUALIFIED,
        benchmark_receipt_digest=_DIGEST,
    )
    benchmark.validate()
    with pytest.raises(ValueError, match="production_receipt_digest"):
        replace(
            benchmark, state=CapabilityQualification.PRODUCTION_QUALIFIED
        ).validate()
    replace(
        benchmark,
        state=CapabilityQualification.PRODUCTION_QUALIFIED,
        production_receipt_digest=_DIGEST,
    ).validate()


def test_external_organ_or_tool_cannot_be_labeled_native_competence() -> None:
    manifest = declared_capability_manifests()[0]
    with pytest.raises(ValueError, match="not native competence"):
        replace(
            manifest,
            implementation=CapabilityImplementation.EXTERNAL_SPECIALIST,
            native_competence_claimed=True,
        ).validate()
