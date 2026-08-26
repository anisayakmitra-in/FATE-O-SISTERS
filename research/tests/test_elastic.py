import pytest
import torch

from moira.elastic import (
    CAPABILITY_AREAS,
    PrecisionPolicy,
    build_elastic_plan,
    build_growth_plan,
    format_bit_precision,
    materialize_bit_precision,
    next_bit_precision,
    parse_bit_precision,
    parse_parameter_count,
    quantize_tensor,
)


def test_custom_bit_precision_ladder_is_not_a_parameter_ladder() -> None:
    assert [next_bit_precision(value) for value in (3, 6, 9, 27, 81)] == [
        6,
        9,
        27,
        81,
        243,
    ]
    assert parse_bit_precision("int27") == 27
    assert format_bit_precision("81-bits") == "int81"
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


def test_growth_plan_rejects_a_non_x3_target() -> None:
    with pytest.raises(ValueError, match="exact x3 steps"):
        build_growth_plan("3B", "6B")


@pytest.mark.parametrize("bits", (3, 6, 9, 27, 81))
def test_packed_custom_bitplane_views_round_trip_shape_and_finiteness(bits: int) -> None:
    source = torch.tensor([-1.0, -0.25, 0.0, 0.5, 1.0], dtype=torch.float32)

    view = quantize_tensor(source, bits)
    restored = view.dequantize(dtype=torch.float32)

    assert view.bits == bits
    assert view.original_shape == (5,)
    assert restored.shape == source.shape
    assert torch.isfinite(restored).all()
    assert materialize_bit_precision(view, bits) is view


def test_consumer_and_enterprise_policies_switch_bits_and_compute_dtype() -> None:
    consumer = PrecisionPolicy.for_profile("consumer")
    enterprise = PrecisionPolicy.for_profile("enterprise")

    assert set(consumer.capabilities) == set(CAPABILITY_AREAS)
    assert consumer.core_bits == 3
    assert consumer.capabilities["mathematics-reasoning"] == 9
    assert consumer.verifier_bits == 9
    assert consumer.compute_dtype.value == "bf16"
    assert enterprise.core_bits == 9
    assert enterprise.capabilities["mathematics-reasoning"] == 27
    assert enterprise.verifier_bits == 81
    assert enterprise.verifier_compute_dtype.value == "fp64"


def test_elastic_plan_keeps_runtime_bits_independent_from_growth() -> None:
    plan = build_elastic_plan("3B", "9B", profile="consumer", runtime_bits="int6")
    payload = plan.to_dict()

    assert payload["current_parameters"] == "3B"
    assert payload["target_parameters"] == "9B"
    assert payload["runtime_bits"] == "int6"
    assert payload["growth"]["parameter_count_is_separate_from_precision"] is True
    assert payload["status"] == "plan-only-no-target-allocation"
