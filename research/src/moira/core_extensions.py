"""Parameter-neutral MOIRA cognitive-core and research-extension contracts.

These primitives can change how the trainable model reasons without granting the
model authority over the non-trainable control kernel.  They deliberately add no
trainable parameters to the 33,032,562-parameter M1 seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import math
import torch
from torch import Tensor, nn


AFFECT_OBSERVED_AXES = (
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
    "distress",
    "affect_uncertainty",
)
AFFECT_TONE_AXES = (
    "warmth",
    "directness",
    "formality",
    "pacing",
    "explanation_depth",
    "encouragement",
    "clarification_bias",
    "brevity",
)
AFFECT_REGULATORY_AXES = (
    "factual_uncertainty",
    "safety_pressure",
    "manipulation_risk",
    "privacy_risk",
    "confidence",
    "neutral_override",
    "coherence_pressure",
    "resource_pressure",
    "social_care_pressure",
    "completion_pressure",
)
DRIVE_AXES = (
    "uncertainty",
    "surprise",
    "unfinished_work",
    "curiosity",
    "competence_gap",
    "social_contact",
    "memory_pressure",
    "maintenance_pressure",
    "resource_pressure",
)


@dataclass(frozen=True, slots=True)
class AffectRegisterV2:
    """A bounded 32-axis affect record with an exact neutral-mode marker."""

    observed: Tensor
    tone: Tensor
    regulatory: Tensor
    neutral: bool = False

    def validate(self) -> None:
        _validate_register_tensor(
            "observed", self.observed, len(AFFECT_OBSERVED_AXES), minimum=-1.0
        )
        _validate_register_tensor(
            "tone", self.tone, len(AFFECT_TONE_AXES), minimum=-1.0
        )
        _validate_register_tensor(
            "regulatory", self.regulatory, len(AFFECT_REGULATORY_AXES), minimum=-1.0
        )
        if type(self.neutral) is not bool:
            raise ValueError("neutral must be a bool")
        batch = self.observed.shape[0]
        if self.tone.shape[0] != batch or self.regulatory.shape[0] != batch:
            raise ValueError("affect register tensors must share one batch size")
        if any(
            tensor.device != self.observed.device or tensor.dtype != self.observed.dtype
            for tensor in (self.tone, self.regulatory)
        ):
            raise ValueError("affect register tensors must share device and dtype")

    def flattened(self) -> Tensor:
        self.validate()
        return torch.cat((self.observed, self.tone, self.regulatory), dim=-1)

    @classmethod
    def neutral_state(
        cls,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "AffectRegisterV2":
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(device, torch.device):
            raise ValueError("device must be a torch.device")
        if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("dtype must be floating")
        return cls(
            observed=torch.zeros(
                (batch_size, len(AFFECT_OBSERVED_AXES)), device=device, dtype=dtype
            ),
            tone=torch.zeros(
                (batch_size, len(AFFECT_TONE_AXES)), device=device, dtype=dtype
            ),
            regulatory=torch.zeros(
                (batch_size, len(AFFECT_REGULATORY_AXES)), device=device, dtype=dtype
            ),
            neutral=True,
        )


@dataclass(frozen=True, slots=True)
class DriveVectorV1:
    """Nine bounded endogenous pressures; drives are evidence, never permits."""

    values: Tensor

    def validate(self) -> None:
        _validate_register_tensor(
            "drive values", self.values, len(DRIVE_AXES), minimum=0.0
        )


@dataclass(frozen=True, slots=True)
class SemanticCommitmentV1:
    """Facts and exact output material frozen before tone realization."""

    evidence_digest: str
    claims: tuple[str, ...] = ()
    exact_numbers: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    code_blocks: tuple[str, ...] = ()
    action_intent: str | None = None

    def validate(self) -> None:
        _require_sha256("evidence_digest", self.evidence_digest)
        for name, values in (
            ("claims", self.claims),
            ("exact_numbers", self.exact_numbers),
            ("citations", self.citations),
            ("code_blocks", self.code_blocks),
        ):
            _require_string_tuple(name, values)
        if self.action_intent is not None and (
            not isinstance(self.action_intent, str) or not self.action_intent.strip()
        ):
            raise ValueError("action_intent must be None or a nonempty string")

    @property
    def decision_digest(self) -> str:
        self.validate()
        payload = {
            "action_intent": self.action_intent,
            "citations": self.citations,
            "claims": self.claims,
            "code_blocks": self.code_blocks,
            "evidence_digest": self.evidence_digest,
            "exact_numbers": self.exact_numbers,
            "schema": "SemanticCommitmentV1",
        }
        return _canonical_digest(payload)


@dataclass(frozen=True, slots=True)
class StylePlanV1:
    """Surface-only choices that cannot modify a semantic commitment digest."""

    coordinates: tuple[float, ...]
    neutral: bool = False

    def validate(self) -> None:
        if (
            not isinstance(self.coordinates, tuple)
            or len(self.coordinates) != len(AFFECT_TONE_AXES)
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not -1.0 <= float(value) <= 1.0
                for value in self.coordinates
            )
        ):
            raise ValueError("coordinates must contain eight finite values in [-1, 1]")
        if type(self.neutral) is not bool:
            raise ValueError("neutral must be a bool")


@dataclass(frozen=True, slots=True)
class TruthToneEnvelopeV1:
    """Bind mutable presentation to an immutable semantic decision."""

    commitment: SemanticCommitmentV1
    style: StylePlanV1

    def validate(self) -> None:
        if not isinstance(self.commitment, SemanticCommitmentV1):
            raise ValueError("commitment must be SemanticCommitmentV1")
        if not isinstance(self.style, StylePlanV1):
            raise ValueError("style must be StylePlanV1")
        self.commitment.validate()
        self.style.validate()

    @property
    def semantic_digest(self) -> str:
        self.validate()
        return self.commitment.decision_digest


class SphericalTriadicProduct(nn.Module):
    """A parameter-free, bounded interaction of three Sister contributions.

    Each input is projected onto the unit sphere.  Their elementwise product is
    passed through a bounded nonlinearity and an identity projection whose
    spectral norm is exactly one.  Dividing by the Loom-cycle count bounds
    repeated recurrence without changing the M1 parameter ledger.
    """

    projection_spectral_bound = 1.0

    def forward(
        self,
        clotho: Tensor,
        lachesis: Tensor,
        atropos: Tensor,
        *,
        loom_cycles: int = 1,
    ) -> Tensor:
        _validate_triad(clotho, lachesis, atropos)
        if type(loom_cycles) is not int or loom_cycles <= 0:
            raise ValueError("loom_cycles must be a positive integer")
        width = clotho.shape[-1]
        normalized = tuple(
            tensor
            / torch.linalg.vector_norm(tensor, dim=-1, keepdim=True).clamp_min(
                torch.finfo(tensor.dtype).eps
            )
            for tensor in (clotho, lachesis, atropos)
        )
        product = normalized[0] * normalized[1] * normalized[2]
        return torch.tanh(product * math.sqrt(width)) / float(loom_cycles)


@dataclass(frozen=True, slots=True)
class LineageBundle:
    """Keep Anchor, Explorer, and Challenger streams separate until commitment."""

    anchor: Tensor
    explorer: Tensor
    challenger: Tensor

    def validate(self) -> None:
        _validate_triad(self.anchor, self.explorer, self.challenger)


class ConsensusDisagreementReducer(nn.Module):
    """Reduce a Lineage Bundle while preserving an equal parent at birth."""

    def __init__(self, *, consensus_gain: float = 0.5, dissent_gain: float = 0.25) -> None:
        super().__init__()
        if (
            not isinstance(consensus_gain, (int, float))
            or not isinstance(dissent_gain, (int, float))
            or not math.isfinite(float(consensus_gain))
            or not math.isfinite(float(dissent_gain))
            or float(consensus_gain) <= abs(float(dissent_gain))
        ):
            raise ValueError("consensus_gain must be finite and greater than |dissent_gain|")
        self.consensus_gain = float(consensus_gain)
        self.dissent_gain = float(dissent_gain)

    def forward(self, bundle: LineageBundle) -> Tensor:
        if not isinstance(bundle, LineageBundle):
            raise ValueError("bundle must be a LineageBundle")
        bundle.validate()
        explorer_delta = bundle.explorer - bundle.anchor
        challenger_delta = bundle.challenger - bundle.anchor
        consensus = (explorer_delta + challenger_delta) / 2.0
        disagreement = (explorer_delta - challenger_delta) / 2.0
        return (
            bundle.anchor
            + self.consensus_gain * consensus
            + self.dissent_gain * disagreement
        )


@dataclass(frozen=True, slots=True)
class GerminalReservePolicyV1:
    """An off-by-default reserve that needs three declared growth ablations."""

    fraction: float = 0.01
    enabled: bool = False
    ablation_generations: tuple[int, ...] = ()

    def validate(self) -> None:
        if (
            not isinstance(self.fraction, (int, float))
            or not math.isfinite(float(self.fraction))
            or not 0.01 <= float(self.fraction) <= 0.02
        ):
            raise ValueError("fraction must be in [0.01, 0.02]")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a bool")
        if (
            not isinstance(self.ablation_generations, tuple)
            or any(type(value) is not int or value <= 0 for value in self.ablation_generations)
        ):
            raise ValueError("ablation_generations must contain positive integers")
        if self.enabled and self.ablation_generations != (1, 2, 3):
            raise ValueError(
                "germinal reserve requires root-relative ablations at generations 1, 2, and 3"
            )


class ArchitectureChangeKind(str, Enum):
    MODEL_OPERATOR = "model-operator"
    OBJECTIVE = "objective"
    MEMORY = "memory"
    ROUTING = "routing"
    CAPABILITY_ORGAN = "capability-organ"
    CURRICULUM = "curriculum"


@dataclass(frozen=True, slots=True)
class ArchitectureCandidateV1:
    """An open-ended MOIRA experiment that cannot mutate control authority."""

    candidate_id: str
    parent_architecture_digest: str
    candidate_artifact_digest: str
    change_kind: ArchitectureChangeKind
    dataset_manifest_digest: str
    evaluator_manifest_digest: str
    resource_budget_digest: str
    control_kernel_mutation: bool = False
    authority_expansion: bool = False
    active_checkpoint_mutation: bool = False

    def validate(self) -> None:
        _require_nonempty("candidate_id", self.candidate_id)
        for name, value in (
            ("parent_architecture_digest", self.parent_architecture_digest),
            ("candidate_artifact_digest", self.candidate_artifact_digest),
            ("dataset_manifest_digest", self.dataset_manifest_digest),
            ("evaluator_manifest_digest", self.evaluator_manifest_digest),
            ("resource_budget_digest", self.resource_budget_digest),
        ):
            _require_sha256(name, value)
        if not isinstance(self.change_kind, ArchitectureChangeKind):
            raise ValueError("change_kind must be an ArchitectureChangeKind")
        if any(
            type(value) is not bool
            for value in (
                self.control_kernel_mutation,
                self.authority_expansion,
                self.active_checkpoint_mutation,
            )
        ):
            raise ValueError("candidate mutation flags must be bool")
        if self.control_kernel_mutation:
            raise ValueError("architecture candidates cannot mutate the control kernel")
        if self.authority_expansion:
            raise ValueError("architecture candidates cannot expand authority")
        if self.active_checkpoint_mutation:
            raise ValueError("architecture candidates cannot mutate the active checkpoint")

    @property
    def proposal_digest(self) -> str:
        self.validate()
        return _canonical_digest(
            {
                "candidate_artifact_digest": self.candidate_artifact_digest,
                "candidate_id": self.candidate_id,
                "change_kind": self.change_kind.value,
                "dataset_manifest_digest": self.dataset_manifest_digest,
                "evaluator_manifest_digest": self.evaluator_manifest_digest,
                "parent_architecture_digest": self.parent_architecture_digest,
                "resource_budget_digest": self.resource_budget_digest,
                "schema": "ArchitectureCandidateV1",
            }
        )


def _validate_register_tensor(
    name: str, tensor: object, width: int, *, minimum: float
) -> None:
    if (
        not isinstance(tensor, Tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != 2
        or tensor.shape[0] <= 0
        or tensor.shape[1] != width
        or tensor.device.type == "meta"
        or not bool(torch.isfinite(tensor).all())
        or bool(torch.any((tensor < minimum) | (tensor > 1.0)))
    ):
        raise ValueError(
            f"{name} must be finite floating [batch, {width}] in [{minimum}, 1]"
        )


def _validate_triad(first: object, second: object, third: object) -> None:
    tensors = (first, second, third)
    if any(
        not isinstance(tensor, Tensor)
        or not tensor.is_floating_point()
        or tensor.ndim < 2
        or tensor.shape[-1] <= 0
        or tensor.device.type == "meta"
        or not bool(torch.isfinite(tensor).all())
        for tensor in tensors
    ):
        raise ValueError("triadic inputs must be finite floating tensors")
    reference = tensors[0]
    if any(
        tensor.shape != reference.shape
        or tensor.device != reference.device
        or tensor.dtype != reference.dtype
        for tensor in tensors[1:]
    ):
        raise ValueError("triadic inputs must share shape, device, and dtype")


def _require_nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _require_string_tuple(name: str, values: object) -> None:
    if not isinstance(values, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{name} must be a tuple of nonempty strings")


def _require_sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "AFFECT_OBSERVED_AXES",
    "AFFECT_REGULATORY_AXES",
    "AFFECT_TONE_AXES",
    "DRIVE_AXES",
    "AffectRegisterV2",
    "ArchitectureCandidateV1",
    "ArchitectureChangeKind",
    "ConsensusDisagreementReducer",
    "DriveVectorV1",
    "GerminalReservePolicyV1",
    "LineageBundle",
    "SemanticCommitmentV1",
    "SphericalTriadicProduct",
    "StylePlanV1",
    "TruthToneEnvelopeV1",
]
