import pytest
import torch

from moira.elastic import (
    CAPABILITY_AREAS,
    ComputeDType,
    MAX_GROWTH_GENERATIONS_PER_PLAN,
    PrecisionPolicy,
    VerificationDType,
    build_elastic_plan,
    build_growth_plan,
    format_bit_precision,
    materialize_bit_precision,
    next_bit_precision,
    parameter_count_after_generations,
    parameter_growth_ladder,
    parse_bit_precision,
    parse_parameter_count,
    plan_precision_transition,
    quantize_tensor,
)


def test_standard_integer_precision_is_not_a_parameter_ladder() -> None:
    assert [next_bit_precision(value) for value in (2, 4)] == [4, 8]
    with pytest.raises(ValueError, match="highest supported"):
        next_bit_precision("int8")
    assert parse_bit_precision("INT4") == 4
    assert format_bit_precision("8-bits") == "int8"
    with pytest.raises(ValueError, match="int2, int4, or int8"):
        parse_bit_precision("int3")
    assert parse_parameter_count("81B") == 81_000_000_000


def test_growth_plan_continues_past_81b_without_an_upper_cap() -> None:
    plan = build_growth_plan("3B", "243B")

    assert [step.target_parameters for step in plan.steps] == [
        9_000_000_000,
        27_000_000_000,
        81_000_000_000,
        243_000_000_000,
    ]
    assert plan.requires_training is True
    assert plan.to_dict()["unbounded_growth"] is True


def test_growth_plan_generates_every_x3_tier_beyond_the_preview_ladder() -> None:
    plan = build_growth_plan("81B", "729B")

    assert [step.target_parameters for step in plan.steps] == [
        243_000_000_000,
        729_000_000_000,
    ]


def test_growth_plan_is_exact_and_relative_to_an_arbitrary_root() -> None:
    root = 33_032_562
    plan = build_growth_plan("33,032,562", root * 27)

    assert parameter_growth_ladder(root, steps=3) == (
        33_032_562,
        99_097_686,
        297_293_058,
        891_879_174,
    )
    assert [step.target_parameters for step in plan.steps] == [
        99_097_686,
        297_293_058,
        891_879_174,
    ]
    assert plan.to_dict()["root"] == "33,032,562"
    assert plan.to_dict()["root_parameters"] == root
    assert plan.to_dict()["root_relative"] is True


def test_growth_plan_rejects_a_non_x3_target() -> None:
    with pytest.raises(ValueError, match="exact x3 steps"):
        build_growth_plan("3B", "6B")


def test_generation_growth_is_exact_bounded_and_repeatable() -> None:
    target = parameter_count_after_generations("9B", 5)
    plan = build_growth_plan("9B", target)

    assert target == 2_187_000_000_000
    assert plan.target_parameters == target
    assert plan.to_dict()["generations"] == 5
    assert plan.to_dict()["per_plan_generation_limit"] == (
        MAX_GROWTH_GENERATIONS_PER_PLAN
    )
    assert plan.to_dict()["repeatable_from_qualified_child"] is True
    with pytest.raises(ValueError, match="between 1 and 32"):
        parameter_count_after_generations("9B", 33)
    with pytest.raises(ValueError, match="cannot exceed 32 generations"):
        build_growth_plan("9B", 9_000_000_000 * 3**33)


@pytest.mark.parametrize("bits", (2, 4, 8))
def test_packed_standard_integer_views_round_trip_shape_and_finiteness(
    bits: int,
) -> None:
    source = torch.tensor([-1.0, -0.25, 0.0, 0.5, 1.0], dtype=torch.float32)

    view = quantize_tensor(source, bits)
    restored = view.dequantize(dtype=torch.float32)

    assert view.bits == bits
    assert view.original_shape == (5,)
    assert restored.shape == source.shape
    assert torch.isfinite(restored).all()
    assert view.packed_values.numel() == (source.numel() * bits + 7) // 8
    assert torch.all(restored[1:] >= restored[:-1])
    assert materialize_bit_precision(view, bits) is view


def test_consumer_and_enterprise_policies_switch_bits_and_compute_dtype() -> None:
    consumer = PrecisionPolicy.for_profile("consumer")
    enterprise = PrecisionPolicy.for_profile("enterprise")

    assert set(consumer.capabilities) == set(CAPABILITY_AREAS)
    assert consumer.core_bits == 4
    assert consumer.capabilities["mathematics-reasoning"] == 8
    assert consumer.capabilities["cybersecurity"] == 8
    assert consumer.cold_expert_bits == 2
    assert consumer.verifier_bits == 8
    assert consumer.compute_dtype.value == "bf16"
    assert consumer.compute_fallback_dtype.value == "fp16"
    assert consumer.master_weight_dtype.value == "fp32"
    assert enterprise.core_bits == 8
    assert enterprise.capabilities["mathematics-reasoning"] == 8
    assert enterprise.capabilities["cybersecurity"] == 8
    assert enterprise.verifier_bits == 8
    assert enterprise.compute_dtype.value == "bf16"
    assert enterprise.compute_fallback_dtype.value == "fp16"
    assert enterprise.verifier_compute_dtype.value == "fp64"
    assert enterprise.to_dict()["whole_model_fp64_allowed"] is False
    assert enterprise.to_dict()["fp64_scope"] == "bounded-verification-arithmetic-only"
    with pytest.raises(ValueError, match="bf16, fp16, or fp32"):
        ComputeDType.parse("fp64")
    assert VerificationDType.parse("fp64").torch_dtype is torch.float64


def test_user_profile_alias_canonicalizes_to_consumer() -> None:
    assert PrecisionPolicy.for_profile("user") == PrecisionPolicy.for_profile(
        "consumer"
    )
    assert PrecisionPolicy.for_profile("user").profile == "consumer"


def test_precision_transition_plan_separates_reexport_from_lossy_widening() -> None:
    master_backed = plan_precision_transition(
        "int8", "fp16", master_retained=True
    ).to_dict()
    lossy = plan_precision_transition("int8", "fp32").to_dict()
    float_widening = plan_precision_transition("fp16", "fp32").to_dict()
    float_master = plan_precision_transition(
        "fp16", "fp32", master_retained=True
    ).to_dict()
    downconversion = plan_precision_transition("fp32", "fp16").to_dict()
    verifier = plan_precision_transition(
        "fp32", "fp64", scope="verifier"
    ).to_dict()

    assert master_backed["uses_retained_master"] is True
    assert master_backed["representation_widening_only"] is False
    assert master_backed["artifact_exported"] is False
    assert lossy["representation_widening_only"] is True
    assert lossy["requires_new_training_for_quality_recovery"] is True
    assert lossy["reconstructs_discarded_information"] is False
    assert float_widening["representation_widening_only"] is True
    assert float_widening["requires_new_training_for_quality_recovery"] is True
    assert float_widening[
        "requires_higher_precision_source_for_information_recovery"
    ] is True
    assert float_master["uses_retained_master"] is True
    assert float_master["requires_new_training_for_quality_recovery"] is False
    assert downconversion["discards_information"] is True
    assert verifier["scope"] == "verifier"
    assert verifier["target_axis"] == "bounded-verification-arithmetic"
    assert verifier["representation_widening_only"] is True
    assert verifier["requires_new_training_for_quality_recovery"] is False
    assert verifier["whole_model_fp64_allowed"] is False
    with pytest.raises(ValueError, match="only for bounded verifier"):
        plan_precision_transition("fp32", "fp64")


def test_elastic_plan_keeps_runtime_bits_independent_from_growth() -> None:
    plan = build_elastic_plan("3B", "9B", profile="consumer", runtime_bits="int4")
    payload = plan.to_dict()

    assert payload["current_parameters"] == "3B"
    assert payload["target_parameters"] == "9B"
    assert payload["runtime_bits"] == "int4"
    assert payload["target_weight_storage_lower_bound_bytes"] == 4_500_000_000
    assert payload["fp32_master_weights_verified"] is False
    assert payload["qat_master_weight_policy"] == "retain-fp32-if-qat-is-materialized"
    assert "quantization-scales" in payload["weight_storage_estimate_excludes"]
    assert payload["growth"]["parameter_count_is_separate_from_precision"] is True
    assert payload["status"] == "plan-only-no-target-allocation"
