"""Root-relative parameter growth and standard precision contracts for MOIRA.

MOIRA has two independent scaling axes:

* parameter capacity: exact, root-relative x3 tiers; and
* integer weight precision: INT2, INT4, or INT8.

Integer weight storage is separate from BF16/FP16/FP32 whole-model arithmetic.
FP64 is reserved for a bounded verification-arithmetic path and is never a
whole-model storage, training, or inference policy. This module produces
bounded plans and small tensor storage views; it never allocates a
multi-billion-parameter model.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
import math
import re
from typing import Any, Mapping

import torch
from torch import Tensor


DEFAULT_PARAMETER_ROOT = 3_000_000_000
PARAMETER_GROWTH_MULTIPLIER = 3
PARAMETER_GROWTH_PREVIEW_STEPS = 5
MAX_GROWTH_GENERATIONS_PER_PLAN = 32
CANONICAL_PARAMETER_LADDER = tuple(
    DEFAULT_PARAMETER_ROOT * PARAMETER_GROWTH_MULTIPLIER**step
    for step in range(PARAMETER_GROWTH_PREVIEW_STEPS + 1)
)
BIT_PRECISION_LADDER = (2, 4, 8)
COMPUTE_DTYPE_LADDER = ("bf16", "fp16", "fp32")
VERIFICATION_DTYPE_LADDER = ("fp32", "fp64")
CAPABILITY_AREAS = (
    "mathematics-reasoning",
    "speech-audio",
    "vision",
    "cybersecurity",
    "graph-molecular",
    "time-series",
    "moe-experts",
    "diffusion-generation",
)
_PARAMETER_RE = re.compile(
    r"^\s*((?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?)\s*([kKmMbB]?)\s*$"
)
_BIT_RE = re.compile(
    r"^\s*(?:int)?(\d+)\s*(?:-?bits?|b)?\s*$", re.IGNORECASE
)
_MAX_PACKABLE_VALUES = 1_000_000


class ComputeDType(str, Enum):
    """Allowed whole-model training and inference arithmetic dtype."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"

    @classmethod
    def parse(cls, value: str | "ComputeDType") -> "ComputeDType":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError("compute dtype must be bf16, fp16, or fp32")
        try:
            return cls(value.strip().lower())
        except ValueError as error:
            raise ValueError("compute dtype must be bf16, fp16, or fp32") from error

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            ComputeDType.BF16: torch.bfloat16,
            ComputeDType.FP16: torch.float16,
            ComputeDType.FP32: torch.float32,
        }[self]


class VerificationDType(str, Enum):
    """Arithmetic allowed only for bounded verifier computations."""

    FP32 = "fp32"
    FP64 = "fp64"

    @classmethod
    def parse(
        cls, value: str | "VerificationDType"
    ) -> "VerificationDType":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError("verification dtype must be fp32 or fp64")
        try:
            return cls(value.strip().lower())
        except ValueError as error:
            raise ValueError("verification dtype must be fp32 or fp64") from error

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            VerificationDType.FP32: torch.float32,
            VerificationDType.FP64: torch.float64,
        }[self]


def parse_parameter_count(value: int | str) -> int:
    """Parse an exact parameter count such as ``3B`` or ``27000000000``."""
    if type(value) is int:
        if value <= 0:
            raise ValueError("parameter count must be positive")
        return value
    if not isinstance(value, str):
        raise ValueError("parameter count must be an integer or a string")
    match = _PARAMETER_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            "parameter count must look like 3B, 33,032,562, or an integer"
        )
    try:
        number = Decimal(match.group(1).replace(",", ""))
        multiplier = {
            "": Decimal(1),
            "k": Decimal(1_000),
            "m": Decimal(1_000_000),
            "b": Decimal(1_000_000_000),
        }[match.group(2).lower()]
        result = number * multiplier
    except (InvalidOperation, KeyError) as error:
        raise ValueError("parameter count is not numeric") from error
    if result != result.to_integral_value() or result <= 0:
        raise ValueError("parameter count must resolve to a positive integer")
    return int(result)


def format_parameter_count(value: int | str) -> str:
    """Format a parameter count using the largest exact common unit."""
    count = parse_parameter_count(value)
    for suffix, divisor in (
        ("B", 1_000_000_000),
        ("M", 1_000_000),
        ("K", 1_000),
    ):
        if count % divisor == 0:
            return f"{count // divisor}{suffix}"
    return f"{count:,}"


def parse_bit_precision(value: int | str) -> int:
    """Parse one supported integer weight format: INT2, INT4, or INT8."""
    if type(value) is int:
        bits = value
    elif isinstance(value, str):
        match = _BIT_RE.fullmatch(value)
        if match is None:
            raise ValueError("integer weight precision must be int2, int4, or int8")
        bits = int(match.group(1))
    else:
        raise ValueError("integer weight precision must be int2, int4, or int8")
    if bits not in BIT_PRECISION_LADDER:
        raise ValueError("integer weight precision must be int2, int4, or int8")
    return bits


def format_bit_precision(value: int | str) -> str:
    return f"int{parse_bit_precision(value)}"


def next_bit_precision(value: int | str) -> int:
    """Return the next standard integer weight format."""
    bits = parse_bit_precision(value)
    for candidate in BIT_PRECISION_LADDER:
        if candidate > bits:
            return candidate
    raise ValueError("int8 is the highest supported integer weight precision")


def parameter_growth_ladder(
    root: int | str,
    *,
    steps: int = PARAMETER_GROWTH_PREVIEW_STEPS,
) -> tuple[int, ...]:
    """Return an exact x3 preview whose first tier is the supplied root."""
    if (
        type(steps) is not int
        or not 0 <= steps <= MAX_GROWTH_GENERATIONS_PER_PLAN
    ):
        raise ValueError(
            "growth preview steps must be between 0 and "
            f"{MAX_GROWTH_GENERATIONS_PER_PLAN}"
        )
    root_count = parse_parameter_count(root)
    return tuple(
        root_count * PARAMETER_GROWTH_MULTIPLIER**step
        for step in range(steps + 1)
    )


def parameter_count_after_generations(
    current: int | str,
    generations: int,
) -> int:
    """Return ``current * 3**generations`` for one bounded planning command."""
    if (
        type(generations) is not int
        or not 1 <= generations <= MAX_GROWTH_GENERATIONS_PER_PLAN
    ):
        raise ValueError(
            "growth generations must be between 1 and "
            f"{MAX_GROWTH_GENERATIONS_PER_PLAN}"
        )
    return parse_parameter_count(current) * PARAMETER_GROWTH_MULTIPLIER**generations


@dataclass(frozen=True, slots=True)
class GrowthStep:
    source_parameters: int
    target_parameters: int
    added_parameters: int
    initialization: str = "inherit-and-train"

    def __post_init__(self) -> None:
        if self.source_parameters <= 0 or self.target_parameters <= 0:
            raise ValueError("growth step parameter counts must be positive")
        if self.target_parameters <= self.source_parameters:
            raise ValueError("growth step target must exceed its source")
        if self.added_parameters != self.target_parameters - self.source_parameters:
            raise ValueError("growth step added parameter count is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "source": format_parameter_count(self.source_parameters),
            "target": format_parameter_count(self.target_parameters),
            "source_parameters": self.source_parameters,
            "target_parameters": self.target_parameters,
            "added_parameters": self.added_parameters,
            "initialization": self.initialization,
            "requires_training": True,
        }


@dataclass(frozen=True, slots=True)
class GrowthPlan:
    current_parameters: int
    target_parameters: int
    steps: tuple[GrowthStep, ...]

    @property
    def requires_training(self) -> bool:
        return self.current_parameters != self.target_parameters

    def to_dict(self) -> dict[str, object]:
        return {
            "root": format_parameter_count(self.current_parameters),
            "root_parameters": self.current_parameters,
            "current": format_parameter_count(self.current_parameters),
            "target": format_parameter_count(self.target_parameters),
            "current_parameters": self.current_parameters,
            "target_parameters": self.target_parameters,
            "steps": [step.to_dict() for step in self.steps],
            "requires_training": self.requires_training,
            "parameter_count_is_separate_from_precision": True,
            "root_relative": True,
            "growth_multiplier": PARAMETER_GROWTH_MULTIPLIER,
            "generations": len(self.steps),
            "per_plan_generation_limit": MAX_GROWTH_GENERATIONS_PER_PLAN,
            "repeatable_from_qualified_child": True,
            "unbounded_growth": True,
        }


def build_growth_plan(
    current: int | str,
    target: int | str,
) -> GrowthPlan:
    """Build an exact x3 plan relative to the supplied root/current model."""
    current_count = parse_parameter_count(current)
    target_count = parse_parameter_count(target)
    if target_count < current_count:
        raise ValueError("growth target cannot be smaller than the current model")
    if current_count == target_count:
        return GrowthPlan(current_count, target_count, ())

    steps: list[GrowthStep] = []
    source = current_count
    while source < target_count:
        if len(steps) >= MAX_GROWTH_GENERATIONS_PER_PLAN:
            raise ValueError(
                "one growth plan cannot exceed "
                f"{MAX_GROWTH_GENERATIONS_PER_PLAN} generations; plan again from "
                "the qualified child"
            )
        milestone = source * PARAMETER_GROWTH_MULTIPLIER
        if milestone > target_count:
            raise ValueError(
                "growth target must be reachable from the current model by exact x3 steps"
            )
        steps.append(
            GrowthStep(
                source_parameters=source,
                target_parameters=milestone,
                added_parameters=milestone - source,
            )
        )
        source = milestone
    return GrowthPlan(current_count, target_count, tuple(steps))


def estimate_weight_storage_bytes(parameters: int | str, bits: int | str) -> int:
    """Return a lower-bound packed-weight estimate, excluding scales/optimizer."""
    count = parse_parameter_count(parameters)
    level = parse_bit_precision(bits)
    return math.ceil(count * level / 8)


def _pack_unsigned(values: list[int], bits: int) -> Tensor:
    accumulator = 0
    buffered = 0
    packed = bytearray()
    for value in values:
        accumulator |= value << buffered
        buffered += bits
        while buffered >= 8:
            packed.append(accumulator & 0xFF)
            accumulator >>= 8
            buffered -= 8
    if buffered:
        packed.append(accumulator & 0xFF)
    return torch.tensor(list(packed), dtype=torch.uint8)


def _unpack_unsigned(packed: Tensor, count: int, bits: int) -> list[int]:
    accumulator = 0
    buffered = 0
    values: list[int] = []
    byte_values = packed.detach().cpu().tolist()
    byte_index = 0
    mask = (1 << bits) - 1
    while len(values) < count:
        while buffered < bits:
            if byte_index >= len(byte_values):
                raise ValueError("packed bitplane data ended before its value count")
            accumulator |= int(byte_values[byte_index]) << buffered
            byte_index += 1
            buffered += 8
        values.append(accumulator & mask)
        accumulator >>= bits
        buffered -= bits
    return values


@dataclass(frozen=True, slots=True)
class QuantizedTensor:
    """A portable packed bitplane storage view."""

    packed_values: Tensor
    scale: float
    zero_point: int
    bits: int
    original_shape: tuple[int, ...]
    value_count: int

    def __post_init__(self) -> None:
        if self.bits <= 0:
            raise ValueError("quantized bit width must be positive")
        if not isinstance(self.packed_values, Tensor):
            raise ValueError("packed values must be a tensor")
        if self.packed_values.dtype is not torch.uint8:
            raise ValueError("packed values must use uint8 storage")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("quantized scale must be finite and positive")
        if self.value_count <= 0 or math.prod(self.original_shape) != self.value_count:
            raise ValueError("quantized shape metadata does not match value count")

    def dequantize(self, *, dtype: torch.dtype = torch.float64) -> Tensor:
        """Materialize the bitplane view; this cannot recover discarded detail."""
        values = _unpack_unsigned(self.packed_values, self.value_count, self.bits)
        result = (
            torch.tensor([float(value) for value in values], dtype=torch.float64)
            - float(self.zero_point)
        ) * self.scale
        return result.reshape(self.original_shape).to(dtype=dtype)

    def to_dict(self) -> dict[str, object]:
        return {
            "bits": self.bits,
            "shape": list(self.original_shape),
            "value_count": self.value_count,
            "packed_bytes": int(self.packed_values.numel()),
            "scale": self.scale,
            "zero_point": self.zero_point,
            "storage_dtype": "uint8-packed-bitplanes",
        }


def quantize_tensor(tensor: Tensor, bits: int | str) -> QuantizedTensor:
    """Create a bounded affine INT2/INT4/INT8 view of a finite tensor."""
    level = parse_bit_precision(bits)
    if (
        not isinstance(tensor, Tensor)
        or not tensor.is_floating_point()
        or tensor.numel() <= 0
        or tensor.numel() > _MAX_PACKABLE_VALUES
        or not bool(torch.isfinite(tensor).all())
    ):
        raise ValueError("tensor must be a bounded, non-empty finite floating tensor")
    source = tensor.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    qmax = (1 << level) - 1
    minimum = float(source.amin().item())
    maximum = float(source.amax().item())
    span = maximum - minimum
    scale = span / qmax if span else max(abs(minimum) / qmax, 2.0**-52)
    zero_point = round(-minimum / scale)
    zero_point = max(0, min(qmax, zero_point))
    values = []
    for value in source.tolist():
        quantized = round(float(value) / scale + zero_point)
        values.append(max(0, min(qmax, quantized)))
    return QuantizedTensor(
        packed_values=_pack_unsigned(values, level),
        scale=scale,
        zero_point=zero_point,
        bits=level,
        original_shape=tuple(tensor.shape),
        value_count=tensor.numel(),
    )


def materialize_bit_precision(
    tensor: Tensor | QuantizedTensor,
    bits: int | str,
) -> Tensor | QuantizedTensor:
    """Materialize a packed logical bit view or preserve an existing view."""
    level = parse_bit_precision(bits)
    if isinstance(tensor, QuantizedTensor) and tensor.bits == level:
        return tensor
    if isinstance(tensor, QuantizedTensor):
        tensor = tensor.dequantize()
    if not isinstance(tensor, Tensor):
        raise ValueError("materialize_bit_precision requires a tensor or bitplane view")
    return quantize_tensor(tensor, level)


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """Integer weight storage plus explicitly separate arithmetic policy."""

    profile: str
    core_bits: int
    capabilities: Mapping[str, int]
    active_expert_bits: int
    cold_expert_bits: int
    verifier_bits: int
    compute_dtype: ComputeDType
    compute_fallback_dtype: ComputeDType
    master_weight_dtype: ComputeDType
    verifier_compute_dtype: VerificationDType

    def __post_init__(self) -> None:
        if self.profile not in {"consumer", "enterprise"}:
            raise ValueError("profile must be consumer or enterprise")
        normalized = {
            str(area): parse_bit_precision(bits)
            for area, bits in self.capabilities.items()
        }
        if set(normalized) != set(CAPABILITY_AREAS):
            raise ValueError("precision policy must cover every capability area")
        object.__setattr__(self, "core_bits", parse_bit_precision(self.core_bits))
        object.__setattr__(
            self, "active_expert_bits", parse_bit_precision(self.active_expert_bits)
        )
        object.__setattr__(
            self, "cold_expert_bits", parse_bit_precision(self.cold_expert_bits)
        )
        object.__setattr__(self, "verifier_bits", parse_bit_precision(self.verifier_bits))
        object.__setattr__(self, "compute_dtype", ComputeDType.parse(self.compute_dtype))
        object.__setattr__(
            self,
            "compute_fallback_dtype",
            ComputeDType.parse(self.compute_fallback_dtype),
        )
        object.__setattr__(
            self,
            "master_weight_dtype",
            ComputeDType.parse(self.master_weight_dtype),
        )
        if self.master_weight_dtype is not ComputeDType.FP32:
            raise ValueError("master weight dtype must be fp32")
        object.__setattr__(
            self,
            "verifier_compute_dtype",
            VerificationDType.parse(self.verifier_compute_dtype),
        )
        object.__setattr__(self, "capabilities", normalized)

    @classmethod
    def for_profile(cls, profile: str) -> "PrecisionPolicy":
        if profile == "user":
            profile = "consumer"
        if profile == "consumer":
            return cls(
                profile=profile,
                core_bits=4,
                capabilities={
                    "mathematics-reasoning": 8,
                    "speech-audio": 4,
                    "vision": 4,
                    "cybersecurity": 8,
                    "graph-molecular": 4,
                    "time-series": 4,
                    "moe-experts": 4,
                    "diffusion-generation": 4,
                },
                active_expert_bits=4,
                cold_expert_bits=2,
                verifier_bits=8,
                compute_dtype=ComputeDType.BF16,
                compute_fallback_dtype=ComputeDType.FP16,
                master_weight_dtype=ComputeDType.FP32,
                verifier_compute_dtype=VerificationDType.FP32,
            )
        if profile == "enterprise":
            return cls(
                profile=profile,
                core_bits=8,
                capabilities={area: 8 for area in CAPABILITY_AREAS},
                active_expert_bits=8,
                cold_expert_bits=4,
                verifier_bits=8,
                compute_dtype=ComputeDType.BF16,
                compute_fallback_dtype=ComputeDType.FP16,
                master_weight_dtype=ComputeDType.FP32,
                verifier_compute_dtype=VerificationDType.FP64,
            )
        raise ValueError("profile must be user, consumer, or enterprise")

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "core_bits": format_bit_precision(self.core_bits),
            "capabilities": {
                area: format_bit_precision(self.capabilities[area])
                for area in CAPABILITY_AREAS
            },
            "active_expert_bits": format_bit_precision(self.active_expert_bits),
            "cold_expert_bits": format_bit_precision(self.cold_expert_bits),
            "verifier_bits": format_bit_precision(self.verifier_bits),
            "integer_weight_precisions": [
                format_bit_precision(value) for value in BIT_PRECISION_LADDER
            ],
            "compute_dtype": self.compute_dtype.value,
            "compute_fallback_dtype": self.compute_fallback_dtype.value,
            "master_weight_dtype": self.master_weight_dtype.value,
            "whole_model_compute_dtypes": list(COMPUTE_DTYPE_LADDER),
            "verifier_compute_dtype": self.verifier_compute_dtype.value,
            "verification_compute_dtypes": list(VERIFICATION_DTYPE_LADDER),
            "fp64_scope": "bounded-verification-arithmetic-only",
            "whole_model_fp64_allowed": False,
            "parameter_count_is_separate_from_precision": True,
            "teacher_weights": "external-and-explicitly-authorized",
        }


def _normalize_precision_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(
            "precision must be int2, int4, int8, bf16, fp16, fp32, or fp64"
        )
    normalized = value.strip().lower()
    if normalized in COMPUTE_DTYPE_LADDER or normalized == "fp64":
        return normalized
    try:
        return format_bit_precision(normalized)
    except ValueError as error:
        raise ValueError(
            "precision must be int2, int4, int8, bf16, fp16, fp32, or fp64"
        ) from error


@dataclass(frozen=True, slots=True)
class PrecisionTransitionPlan:
    """Describe a precision transition without writing or converting an artifact."""

    source_precision: str
    target_precision: str
    master_retained: bool = False
    scope: str = "whole-model"

    def __post_init__(self) -> None:
        source = _normalize_precision_name(self.source_precision)
        target = _normalize_precision_name(self.target_precision)
        if type(self.master_retained) is not bool:
            raise ValueError("master_retained must be a boolean")
        if self.scope not in {"whole-model", "verifier"}:
            raise ValueError("precision scope must be whole-model or verifier")
        if self.scope == "whole-model" and (source == "fp64" or target == "fp64"):
            raise ValueError("fp64 is allowed only for bounded verifier arithmetic")
        if self.scope == "verifier" and target not in VERIFICATION_DTYPE_LADDER:
            raise ValueError("verifier precision target must be fp32 or fp64")
        object.__setattr__(self, "source_precision", source)
        object.__setattr__(self, "target_precision", target)

    @staticmethod
    def _axis(precision: str) -> str:
        if precision.startswith("int"):
            return "integer-weight-storage"
        if precision == "fp64":
            return "bounded-verification-arithmetic"
        return "whole-model-compute-or-export"

    @staticmethod
    def _width(precision: str) -> int:
        if precision.startswith("int"):
            return int(precision[3:])
        return {"bf16": 16, "fp16": 16, "fp32": 32, "fp64": 64}[precision]

    def to_dict(self) -> dict[str, object]:
        source_width = self._width(self.source_precision)
        target_width = self._width(self.target_precision)
        representation_widening = target_width > source_width
        uses_retained_master = self.master_retained and representation_widening
        lossy_container_widening = representation_widening and not uses_retained_master
        downconversion = target_width < source_width
        same_width_format_change = (
            source_width == target_width
            and self.source_precision != self.target_precision
        )
        return {
            "contract": "MOIRA-precision-transition-plan-v1",
            "source_precision": self.source_precision,
            "source_axis": self._axis(self.source_precision),
            "target_precision": self.target_precision,
            "target_axis": self._axis(self.target_precision),
            "scope": self.scope,
            "master_retained": self.master_retained,
            "uses_retained_master": uses_retained_master,
            "representation_widening_only": lossy_container_widening,
            "requires_higher_precision_source_for_information_recovery": (
                lossy_container_widening
            ),
            "requires_new_training_for_quality_recovery": (
                lossy_container_widening and self.scope == "whole-model"
            ),
            "discards_information": downconversion,
            "same_width_format_change": same_width_format_change,
            "requires_evaluation": self.source_precision != self.target_precision,
            "reconstructs_discarded_information": False,
            "restores_lost_quantization_information": False,
            "whole_model_fp64_allowed": False,
            "plan_only": True,
            "artifact_exported": False,
            "status": "plan-only-no-artifact-written",
        }


def plan_precision_transition(
    source: str,
    target: str,
    *,
    master_retained: bool = False,
    scope: str = "whole-model",
) -> PrecisionTransitionPlan:
    """Build an honest, non-mutating precision transition plan."""
    return PrecisionTransitionPlan(
        source_precision=source,
        target_precision=target,
        master_retained=master_retained,
        scope=scope,
    )


@dataclass(frozen=True, slots=True)
class ElasticPlan:
    """A non-allocating growth and precision plan for a MOIRA lineage."""

    growth: GrowthPlan
    policy: PrecisionPolicy
    runtime_bits: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_bits", parse_bit_precision(self.runtime_bits))

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": "MOIRA-root-relative-standard-precision-v2",
            "current_parameters": format_parameter_count(self.growth.current_parameters),
            "target_parameters": format_parameter_count(self.growth.target_parameters),
            "growth": self.growth.to_dict(),
            "precision": self.policy.to_dict(),
            "runtime_bits": format_bit_precision(self.runtime_bits),
            "target_weight_storage_lower_bound_bytes": estimate_weight_storage_bytes(
                self.growth.target_parameters, self.runtime_bits
            ),
            "weight_storage_estimate_excludes": [
                "quantization-scales",
                "tensor-metadata",
                "optimizer-state",
                "gradients",
                "activations",
                "checkpoint-overhead",
            ],
            "qat_master_weight_policy": "retain-fp32-if-qat-is-materialized",
            "fp32_master_weights_verified": False,
            "parameter_count_is_separate_from_precision": True,
            "status": "plan-only-no-target-allocation",
        }


def build_elastic_plan(
    current: int | str,
    target: int | str,
    *,
    profile: str,
    runtime_bits: int | str | None = None,
) -> ElasticPlan:
    """Combine unbounded parameter growth with a deployment bit policy."""
    policy = PrecisionPolicy.for_profile(profile)
    return ElasticPlan(
        growth=build_growth_plan(current, target),
        policy=policy,
        runtime_bits=policy.core_bits if runtime_bits is None else runtime_bits,
    )


__all__ = [
    "BIT_PRECISION_LADDER",
    "CANONICAL_PARAMETER_LADDER",
    "CAPABILITY_AREAS",
    "COMPUTE_DTYPE_LADDER",
    "ComputeDType",
    "DEFAULT_PARAMETER_ROOT",
    "ElasticPlan",
    "GrowthPlan",
    "GrowthStep",
    "MAX_GROWTH_GENERATIONS_PER_PLAN",
    "PARAMETER_GROWTH_MULTIPLIER",
    "PARAMETER_GROWTH_PREVIEW_STEPS",
    "PrecisionPolicy",
    "PrecisionTransitionPlan",
    "QuantizedTensor",
    "VERIFICATION_DTYPE_LADDER",
    "VerificationDType",
    "build_elastic_plan",
    "build_growth_plan",
    "estimate_weight_storage_bytes",
    "format_bit_precision",
    "format_parameter_count",
    "materialize_bit_precision",
    "next_bit_precision",
    "parameter_growth_ladder",
    "parameter_count_after_generations",
    "parse_bit_precision",
    "parse_parameter_count",
    "plan_precision_transition",
    "quantize_tensor",
]
