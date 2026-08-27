from dataclasses import replace

import pytest

from moira.growth import (
    GrowthArtifactState,
    GrowthCandidateManifestV1,
    GrowthCertificateV1,
    GrowthDiagnosis,
    ParameterReportV1,
)


_DIGEST = "a" * 64


def _certificate(diagnosis: GrowthDiagnosis) -> GrowthCertificateV1:
    return GrowthCertificateV1(
        parent_parameters=33_032_562,
        generations=2,
        diagnosis=diagnosis,
        parent_checkpoint_digest=_DIGEST,
        architecture_digest=_DIGEST,
        training_trend_digest=_DIGEST,
        validation_trend_digest=_DIGEST,
        equal_compute_parent_digest=_DIGEST,
        failure_cluster_digest=_DIGEST,
        fresh_data_manifest_digest=_DIGEST,
        worker_placement_digest=_DIGEST,
        rollback_checkpoint_digest=_DIGEST,
        fresh_data_records=1_000,
        effective_rank_ratio=0.95,
        saturation_ratio=0.9,
        optimizer_bytes=10,
        activation_bytes=20,
        checkpoint_bytes=30,
        evaluation_cost_units=40,
    )


def _report(total: int) -> ParameterReportV1:
    return ParameterReportV1(
        logical_total_parameters=total,
        materialized_parameters=total,
        unique_values=33_032_562,
        gradient_updated_parameters=0,
        qualified_parameters=0,
        resident_parameters=33_032_562,
        trainable_parameters=total,
        dormant_parameters=total - 33_032_562,
        active_parameters_per_token=33_032_562,
        peak_concurrent_parameters=33_032_562,
        parameter_applications=total,
        adapter_parameters=0,
        organ_parameters=0,
        weight_storage_precision="int4",
        training_master_precision="fp32",
    )


def test_growth_certificate_computes_exact_root_relative_child() -> None:
    certificate = _certificate(GrowthDiagnosis.UNDERCAPACITY)

    assert certificate.child_parameters == 297_293_058
    assert certificate.materialization_eligible is True
    assert len(certificate.certificate_digest) == 64


@pytest.mark.parametrize("value", (True, False))
def test_growth_certificate_rejects_boolean_ratios(value: bool) -> None:
    with pytest.raises(ValueError, match="ratios"):
        replace(_certificate(GrowthDiagnosis.UNDERCAPACITY), effective_rank_ratio=value).validate()


def test_generation_limit_is_per_certificate_not_a_lineage_cap() -> None:
    first = replace(_certificate(GrowthDiagnosis.UNDERCAPACITY), generations=32)
    first.validate()
    second = replace(first, parent_parameters=first.child_parameters)
    second.validate()

    assert second.child_parameters == 33_032_562 * 3**64
    with pytest.raises(ValueError, match="operational limit"):
        replace(first, generations=33).validate()


@pytest.mark.parametrize(
    "diagnosis",
    (
        GrowthDiagnosis.UNDERTRAINED,
        GrowthDiagnosis.BAD_DATA,
        GrowthDiagnosis.BAD_OBJECTIVE,
    ),
)
def test_noncapacity_diagnosis_does_not_authorize_materialization(
    diagnosis: GrowthDiagnosis,
) -> None:
    assert _certificate(diagnosis).materialization_eligible is False


def test_materialized_child_cannot_claim_untrained_capacity_as_learned() -> None:
    certificate = _certificate(GrowthDiagnosis.UNDERCAPACITY)
    manifest = GrowthCandidateManifestV1(
        candidate_id="growth:1",
        certificate_digest=certificate.certificate_digest,
        parent_checkpoint_digest=_DIGEST,
        child_parameters=certificate.child_parameters,
        state=GrowthArtifactState.MATERIALIZED_UNTRAINED,
        report=_report(certificate.child_parameters),
    )
    manifest.validate()

    with pytest.raises(ValueError, match="cannot claim trained"):
        replace(
            manifest,
            report=replace(manifest.report, gradient_updated_parameters=1),
        ).validate()


def test_trained_and_evaluated_growth_states_require_measured_capacity() -> None:
    certificate = _certificate(GrowthDiagnosis.UNDERCAPACITY)
    manifest = GrowthCandidateManifestV1(
        candidate_id="growth:measured",
        certificate_digest=certificate.certificate_digest,
        parent_checkpoint_digest=_DIGEST,
        child_parameters=certificate.child_parameters,
        state=GrowthArtifactState.TRAINED,
        report=_report(certificate.child_parameters),
    )

    with pytest.raises(ValueError, match="gradient-updated"):
        manifest.validate()
    with pytest.raises(ValueError, match="qualified parameters"):
        replace(
            manifest,
            state=GrowthArtifactState.EVALUATED,
            report=replace(manifest.report, gradient_updated_parameters=1),
        ).validate()
    with pytest.raises(ValueError, match="unqualified parameters"):
        replace(
            manifest,
            state=GrowthArtifactState.ACTIVE,
            report=replace(
                manifest.report,
                gradient_updated_parameters=1,
                qualified_parameters=1,
            ),
        ).validate()


def test_parameter_report_rejects_impossible_active_or_materialized_counts() -> None:
    report = _report(99_097_686)
    with pytest.raises(ValueError, match="materialization invariants"):
        replace(
            report, active_parameters_per_token=report.resident_parameters + 1
        ).validate()
    with pytest.raises(ValueError, match="materialization invariants"):
        replace(
            report,
            parameter_applications=report.active_parameters_per_token - 1,
        ).validate()
