from dataclasses import replace

import pytest
import torch

from moira.core_extensions import (
    AFFECT_OBSERVED_AXES,
    AFFECT_REGULATORY_AXES,
    AFFECT_TONE_AXES,
    DRIVE_AXES,
    AffectRegisterV2,
    ArchitectureCandidateV1,
    ArchitectureChangeKind,
    ConsensusDisagreementReducer,
    DriveVectorV1,
    GerminalReservePolicyV1,
    LineageBundle,
    SemanticCommitmentV1,
    SphericalTriadicProduct,
    StylePlanV1,
    TruthToneEnvelopeV1,
)


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64
_DIGEST_D = "d" * 64
_DIGEST_E = "e" * 64


def test_affect_v2_and_drive_v1_have_the_declared_bounded_dimensions() -> None:
    affect = AffectRegisterV2.neutral_state(
        2, device=torch.device("cpu"), dtype=torch.float32
    )
    drive = DriveVectorV1(torch.zeros((2, len(DRIVE_AXES))))

    affect.validate()
    drive.validate()

    assert len(AFFECT_OBSERVED_AXES) == 14
    assert len(AFFECT_TONE_AXES) == 8
    assert len(AFFECT_REGULATORY_AXES) == 10
    assert len(DRIVE_AXES) == 9
    assert affect.flattened().shape == (2, 32)
    assert affect.neutral is True


@pytest.mark.parametrize(
    "register",
    [
        AffectRegisterV2(
            torch.zeros((1, 13)), torch.zeros((1, 8)), torch.zeros((1, 10))
        ),
        AffectRegisterV2(
            torch.zeros((1, 14)), torch.full((1, 8), 1.1), torch.zeros((1, 10))
        ),
        AffectRegisterV2(
            torch.zeros((2, 14)), torch.zeros((1, 8)), torch.zeros((2, 10))
        ),
    ],
)
def test_affect_v2_rejects_malformed_or_unbounded_registers(
    register: AffectRegisterV2,
) -> None:
    with pytest.raises(ValueError, match="affect|observed|tone|batch"):
        register.validate()


def test_truth_tone_firewall_keeps_semantics_invariant_under_style_changes() -> None:
    commitment = SemanticCommitmentV1(
        evidence_digest=_DIGEST_A,
        claims=("The verified result is fixed.",),
        exact_numbers=("33032562",),
        citations=("artifact:held-out-evaluation",),
        code_blocks=("return verified_result",),
        action_intent="report-only",
    )
    neutral = TruthToneEnvelopeV1(
        commitment,
        StylePlanV1((0.0,) * 8, neutral=True),
    )
    warm = TruthToneEnvelopeV1(
        commitment,
        StylePlanV1((0.9, -0.2, 0.1, 0.4, 0.6, 0.8, -0.1, -0.4)),
    )

    assert neutral.semantic_digest == warm.semantic_digest
    assert neutral.semantic_digest == commitment.decision_digest


def test_spherical_triadic_product_is_parameter_neutral_bounded_and_cycle_scaled() -> None:
    torch.manual_seed(7)
    layer = SphericalTriadicProduct()
    inputs = tuple(torch.randn((3, 5, 16)) for _ in range(3))

    once = layer(*inputs, loom_cycles=1)
    eight = layer(*inputs, loom_cycles=8)

    assert sum(parameter.numel() for parameter in layer.parameters()) == 0
    assert layer.projection_spectral_bound == 1.0
    assert torch.all(torch.isfinite(once))
    assert torch.all(torch.abs(once) <= 1.0)
    assert torch.allclose(eight, once / 8.0)


def test_consensus_disagreement_birth_is_exact_and_both_new_branches_learn() -> None:
    parent = torch.randn((2, 3, 5))
    anchor = parent.detach().clone().requires_grad_(True)
    explorer = parent.detach().clone().requires_grad_(True)
    challenger = parent.detach().clone().requires_grad_(True)
    reducer = ConsensusDisagreementReducer(consensus_gain=0.5, dissent_gain=0.25)

    result = reducer(LineageBundle(anchor, explorer, challenger))
    assert torch.equal(result, parent)

    result.sum().backward()
    assert explorer.grad is not None and torch.all(explorer.grad != 0)
    assert challenger.grad is not None and torch.all(challenger.grad != 0)


def test_consensus_must_dominate_dissent() -> None:
    with pytest.raises(ValueError, match="greater"):
        ConsensusDisagreementReducer(consensus_gain=0.2, dissent_gain=0.2)


def test_germinal_reserve_stays_off_until_all_three_growth_ablations_exist() -> None:
    GerminalReservePolicyV1().validate()
    with pytest.raises(ValueError, match="ablation"):
        GerminalReservePolicyV1(
            enabled=True, ablation_generations=(1, 2)
        ).validate()
    GerminalReservePolicyV1(enabled=True, ablation_generations=(1, 2, 3)).validate()


def _candidate() -> ArchitectureCandidateV1:
    return ArchitectureCandidateV1(
        candidate_id="operator-experiment-1",
        parent_architecture_digest=_DIGEST_A,
        candidate_artifact_digest=_DIGEST_B,
        change_kind=ArchitectureChangeKind.MODEL_OPERATOR,
        dataset_manifest_digest=_DIGEST_C,
        evaluator_manifest_digest=_DIGEST_D,
        resource_budget_digest=_DIGEST_E,
    )


def test_open_ended_candidate_accepts_new_model_operator_but_not_new_authority() -> None:
    candidate = _candidate()
    candidate.validate()
    assert len(candidate.proposal_digest) == 64

    with pytest.raises(ValueError, match="authority"):
        replace(candidate, authority_expansion=True).validate()
    with pytest.raises(ValueError, match="control kernel"):
        replace(candidate, control_kernel_mutation=True).validate()
    with pytest.raises(ValueError, match="active checkpoint"):
        replace(candidate, active_checkpoint_mutation=True).validate()
