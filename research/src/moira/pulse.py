"""Unified bounded affect state for the MOIRA reference model."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


FAST_AXES = (
    "valence",
    "arousal",
    "control",
    "certainty",
    "novelty",
    "urgency",
    "conflict",
    "risk",
    "care_relevance",
    "curiosity",
    "frustration",
    "completion",
)
SLOW_AXES = (
    "coherence_pressure",
    "completion_pressure",
    "exploration_pressure",
    "safety_pressure",
    "social_care_pressure",
    "resource_reserve",
)


@dataclass(frozen=True, slots=True)
class PulseState:
    fast: Tensor
    slow: Tensor


class Pulse(nn.Module):
    """Maintain one differentiable fast/slow affect state per batch row."""

    def __init__(
        self, d_model: int, *, fast_decay: float = 0.9, slow_decay: float = 0.99
    ) -> None:
        super().__init__()
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be positive")
        _require_decay("fast_decay", fast_decay)
        _require_decay("slow_decay", slow_decay)
        self.d_model = d_model
        self.fast_decay = fast_decay
        self.slow_decay = slow_decay
        self.appraisal_projection = nn.Linear(d_model, len(FAST_AXES))
        self.fast_to_slow = nn.Linear(len(FAST_AXES), len(SLOW_AXES))
        self.modulation_scale = nn.Linear(len(FAST_AXES) + len(SLOW_AXES), d_model)
        self.modulation_bias = nn.Linear(len(FAST_AXES) + len(SLOW_AXES), d_model)

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> PulseState:
        """Create a zero state within the declared fast and slow bounds."""
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not isinstance(device, torch.device):
            raise ValueError("device must be a torch.device")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("dtype must be floating")
        parameter = next(self.parameters())
        if device != parameter.device or dtype != parameter.dtype:
            raise ValueError("initial state must match Pulse device and dtype")
        return PulseState(
            fast=torch.zeros((batch_size, len(FAST_AXES)), device=device, dtype=dtype),
            slow=torch.zeros((batch_size, len(SLOW_AXES)), device=device, dtype=dtype),
        )

    def appraise(self, summary: Tensor) -> Tensor:
        """Map a sequence summary to bounded fast appraisal dimensions."""
        _require_model_tensor("summary", summary, (None, self.d_model), self)
        return torch.tanh(self.appraisal_projection(summary))

    def step(self, state: PulseState, appraisal: Tensor, *, update_mask: Tensor) -> PulseState:
        """Apply one bounded affect update while preserving masked rows exactly."""
        batch_size = _validate_state(state, self)
        _require_model_tensor("appraisal", appraisal, (batch_size, len(FAST_AXES)), self)
        if appraisal.device != state.fast.device or appraisal.dtype != state.fast.dtype:
            raise ValueError("appraisal must match state device and dtype")
        if (
            not isinstance(update_mask, Tensor)
            or update_mask.dtype != torch.bool
            or update_mask.shape != (batch_size,)
            or update_mask.device != state.fast.device
        ):
            raise ValueError("update_mask must be bool [batch] on the state device")

        bounded_appraisal = torch.tanh(appraisal)
        fast = torch.clamp(
            self.fast_decay * state.fast + (1.0 - self.fast_decay) * bounded_appraisal,
            min=-1.0,
            max=1.0,
        )
        slow_target = torch.sigmoid(self.fast_to_slow(fast))
        slow = torch.clamp(
            self.slow_decay * state.slow + (1.0 - self.slow_decay) * slow_target,
            min=0.0,
            max=1.0,
        )
        update = update_mask[:, None]
        return PulseState(
            fast=torch.where(update, fast, state.fast),
            slow=torch.where(update, slow, state.slow),
        )

    def modulate(self, hidden: Tensor, state: PulseState) -> Tensor:
        """Apply bounded per-channel scale and bias without changing hidden shape."""
        batch_size = _validate_state(state, self)
        _require_model_tensor("hidden", hidden, (batch_size, None, self.d_model), self)
        if hidden.device != state.fast.device or hidden.dtype != state.fast.dtype:
            raise ValueError("hidden must match state device and dtype")
        affect = torch.cat((state.fast, state.slow), dim=-1)
        scale = torch.tanh(self.modulation_scale(affect)).unsqueeze(1)
        bias = torch.tanh(self.modulation_bias(affect)).unsqueeze(1)
        return hidden * (1.0 + scale) + bias


def _require_decay(name: str, value: float) -> None:
    if not isinstance(value, (float, int)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _require_model_tensor(
    name: str, tensor: object, shape: tuple[int | None, ...], module: nn.Module
) -> None:
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point() or tensor.shape.__len__() != len(shape):
        raise ValueError(f"{name} has an invalid shape or dtype")
    if any(expected is not None and actual != expected for actual, expected in zip(tensor.shape, shape)):
        raise ValueError(f"{name} has an invalid shape or dtype")
    if tensor.device.type == "meta" or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    parameter = next(module.parameters())
    if tensor.device != parameter.device or tensor.dtype != parameter.dtype:
        raise ValueError(f"{name} must match Pulse device and dtype")


def _validate_state(state: object, module: nn.Module) -> int:
    if not isinstance(state, PulseState):
        raise ValueError("state must be a PulseState")
    fast, slow = state.fast, state.slow
    if (
        not isinstance(fast, Tensor)
        or not isinstance(slow, Tensor)
        or not fast.is_floating_point()
        or not slow.is_floating_point()
        or fast.ndim != 2
        or slow.ndim != 2
        or fast.shape[1] != len(FAST_AXES)
        or slow.shape != (fast.shape[0], len(SLOW_AXES))
        or fast.device != slow.device
        or fast.dtype != slow.dtype
        or fast.device.type == "meta"
        or not bool(torch.isfinite(fast).all())
        or not bool(torch.isfinite(slow).all())
        or bool(torch.any((fast < -1) | (fast > 1)))
        or bool(torch.any((slow < 0) | (slow > 1)))
    ):
        raise ValueError("state must contain bounded fast [batch, 12] and slow [batch, 6]")
    parameter = next(module.parameters())
    if fast.device != parameter.device or fast.dtype != parameter.dtype:
        raise ValueError("state must match Pulse device and dtype")
    return fast.shape[0]
