"""Precision-elastic parameter growth contracts for the MOIRA field.

MOIRA has two independent scaling axes:

* parameter capacity: 3B, 9B, 27B, 81B, and future x3-generated tiers; and
* logical weight precision: 3-bit, 6-bit, 9-bit, 27-bit, 81-bit, and future
  generated levels.

The logical bit widths are bitplane budgets, not promises of native hardware
instructions. A runtime may pack them and dequantize into the arithmetic dtype
available on the target accelerator. This module produces bounded plans and
small tensor storage views; it never allocates a multi-billion-parameter model.
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


CANONICAL_PARAMETER_LADDER = (
    3_000_000_000,
    9_000_000_000,
    27_000_000_000,
    81_000_000_000,
    243_000_000_000,
    729_000_000_000,
)
BIT_PRECISION_LADDER = (3, 6, 9, 27, 81)
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
_PARAMETER_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmMbB]?)\s*$")
_BIT_RE = re.compile(
    r"^\s*(?:int)?(\d+)\s*(?:-?bits?|b)?\s*$", re.IGNORECASE
)
_MAX_PACKABLE_VALUES = 1_000_000


class ComputeDType(str, Enum):
    """Arithmetic dtype used after a logical bitplane view is unpacked."""

    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    FP64 = "fp64"

    @classmethod
    def parse(cls, value: str | "ComputeDType") -> "ComputeDType":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise ValueError("compute dtype must be fp16, bf16, fp32, or fp64")
        try:
            return cls(value.strip().lower())
        except ValueError as error:
            raise ValueError("compute dtype must be fp16, bf16, fp32, or fp64") from error

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            ComputeDType.FP16: torch.float16,
            ComputeDType.BF16: torch.bfloat16,
            ComputeDType.FP32: torch.float32,
            ComputeDType.FP64: torch.float64,
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
        raise ValueError("parameter count must look like 3B, 27B, or an integer")
    try:
        number = Decimal(match.group(1))
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
    return str(count)


def parse_bit_precision(value: int | str) -> int:
    """Parse a logical level such as ``int3``, ``3bit``, or ``81-bits``."""
    if type(value) is int:
        bits = value
    elif isinstance(value, str):
        match = _BIT_RE.fullmatch(value)
        if match is None:
            raise ValueError("bit precision must look like int3, int27, or int81")
        bits = int(match.group(1))
    else:
        raise ValueError("bit precision must be an integer or a bit string")
    if bits <= 0:
        raise ValueError("bit precision must be positive")
    return bits


def format_bit_precision(value: int | str) -> str:
    return f"int{parse_bit_precision(value)}"


def next_bit_precision(value: int | str) -> int:
    """Return the next custom level, growing by three beyond the seed ladder."""
    bits = parse_bit_precision(value)
    for candidate in BIT_PRECISION_LADDER:
        if candidate > bits:
            return candidate
    return bits * 3


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
            "current": format_parameter_count(self.current_parameters),
            "target": format_parameter_count(self.target_parameters),
            "current_parameters": self.current_parameters,
            "target_parameters": self.target_parameters,
            "steps": [step.to_dict() for step in self.steps],
            "requires_training": self.requires_training,
            "parameter_count_is_separate_from_precision": True,
            "unbounded_growth": True,
        }


def build_growth_plan(
    current: int | str,
    target: int | str,
) -> GrowthPlan:
    """Build an exact x3, non-allocating expansion plan with no hard upper tier."""
    current_count = parse_parameter_count(current)
    target_count = parse_parameter_count(target)
    if target_count < current_count:
        raise ValueError("growth target cannot be smaller than the current model")
    if current_count == target_count:
        return GrowthPlan(current_count, target_count, ())

    steps: list[GrowthStep] = []
    source = current_count
    while source < target_count:
        milestone = source * 3
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
    """Create a bounded affine 3/6/9/27/81-bit view of a finite tensor."""
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
    """Capability-specific logical bits plus the arithmetic backend."""

    profile: str
    core_bits: int
    capabilities: Mapping[str, int]
    active_expert_bits: int
    cold_expert_bits: int
    verifier_bits: int
    compute_dtype: ComputeDType
    verifier_compute_dtype: ComputeDType

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
            "verifier_compute_dtype",
            ComputeDType.parse(self.verifier_compute_dtype),
        )
        object.__setattr__(self, "capabilities", normalized)

    @classmethod
    def for_profile(cls, profile: str) -> "PrecisionPolicy":
        if profile == "consumer":
            return cls(
                profile=profile,
                core_bits=3,
                capabilities={
                    "mathematics-reasoning": 9,
                    "speech-audio": 6,
                    "vision": 3,
                    "cybersecurity": 9,
                    "graph-molecular": 6,
                    "time-series": 6,
                    "moe-experts": 3,
                    "diffusion-generation": 6,
                },
                active_expert_bits=6,
                cold_expert_bits=3,
                verifier_bits=9,
                compute_dtype=ComputeDType.BF16,
                verifier_compute_dtype=ComputeDType.FP32,
            )
        if profile == "enterprise":
            return cls(
                profile=profile,
                core_bits=9,
                capabilities={area: 27 for area in CAPABILITY_AREAS},
                active_expert_bits=27,
                cold_expert_bits=9,
                verifier_bits=81,
                compute_dtype=ComputeDType.BF16,
                verifier_compute_dtype=ComputeDType.FP64,
            )
        raise ValueError("profile must be consumer or enterprise")

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
            "compute_dtype": self.compute_dtype.value,
            "verifier_compute_dtype": self.verifier_compute_dtype.value,
            "teacher_weights": "external-and-explicitly-authorized",
        }


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
            "architecture": "MOIRA-precision-lattice-v1",
            "current_parameters": format_parameter_count(self.growth.current_parameters),
            "target_parameters": format_parameter_count(self.growth.target_parameters),
            "growth": self.growth.to_dict(),
            "precision": self.policy.to_dict(),
            "runtime_bits": format_bit_precision(self.runtime_bits),
            "target_weight_storage_bytes": estimate_weight_storage_bytes(
                self.growth.target_parameters, self.runtime_bits
            ),
            "master_weights_required_for_high_precision_promotion": True,
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
        runtime_bits=runtime_bits or policy.core_bits,
    )


__all__ = [
    "BIT_PRECISION_LADDER",
    "CANONICAL_PARAMETER_LADDER",
    "CAPABILITY_AREAS",
    "ComputeDType",
    "ElasticPlan",
    "GrowthPlan",
    "GrowthStep",
    "PrecisionPolicy",
    "QuantizedTensor",
    "build_elastic_plan",
    "build_growth_plan",
    "estimate_weight_storage_bytes",
    "format_bit_precision",
    "format_parameter_count",
    "materialize_bit_precision",
    "next_bit_precision",
    "parse_bit_precision",
    "parse_parameter_count",
    "quantize_tensor",
]
