"""Typed MOIRA capability ports and evidence-backed qualification labels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import math


CAPABILITY_PORT_ABI_VERSION = 2


class CapabilityQualification(str, Enum):
    DECLARED = "DECLARED"
    INTERFACE_VERIFIED = "INTERFACE_VERIFIED"
    SMOKE_TRAINED = "SMOKE_TRAINED"
    BENCHMARK_QUALIFIED = "BENCHMARK_QUALIFIED"
    PRODUCTION_QUALIFIED = "PRODUCTION_QUALIFIED"


class CapabilityImplementation(str, Enum):
    NATIVE_CORE = "native-core"
    RANDOM_ORGAN = "random-organ"
    EXTERNAL_SPECIALIST = "external-specialist"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class CapabilityPortV2:
    abi_version: int
    capability_id: str
    capability_version: str
    input_modalities: tuple[str, ...]
    output_contract: str
    contract_digest: str

    def validate(self) -> None:
        if self.abi_version != CAPABILITY_PORT_ABI_VERSION:
            raise ValueError("unsupported capability port ABI version")
        for name, value in (
            ("capability_id", self.capability_id),
            ("capability_version", self.capability_version),
            ("output_contract", self.output_contract),
        ):
            _nonempty(name, value)
        if (
            not isinstance(self.input_modalities, tuple)
            or not self.input_modalities
            or len(set(self.input_modalities)) != len(self.input_modalities)
            or any(not isinstance(value, str) or not value.strip() for value in self.input_modalities)
        ):
            raise ValueError("input_modalities must be a unique nonempty tuple")
        _digest("contract_digest", self.contract_digest)


@dataclass(frozen=True, slots=True)
class CapabilityQualificationManifestV1:
    """Qualification-claim shape pending trusted AssuranceKernel resolution."""

    capability_id: str
    capability_version: str
    state: CapabilityQualification
    implementation: CapabilityImplementation
    core_parameters: int
    organ_parameters: int
    training_lineage_digest: str | None = None
    frozen_baseline_digest: str | None = None
    heldout_manifest_digest: str | None = None
    evaluator_digest: str | None = None
    benchmark_receipt_digest: str | None = None
    production_receipt_digest: str | None = None
    baseline_score: float | None = None
    candidate_score: float | None = None
    native_competence_claimed: bool = False

    def validate(self) -> None:
        _nonempty("capability_id", self.capability_id)
        _nonempty("capability_version", self.capability_version)
        if not isinstance(self.state, CapabilityQualification):
            raise ValueError("state must be a CapabilityQualification")
        if not isinstance(self.implementation, CapabilityImplementation):
            raise ValueError("implementation must be a CapabilityImplementation")
        if any(
            type(value) is not int or value < 0
            for value in (self.core_parameters, self.organ_parameters)
        ):
            raise ValueError("parameter counts must be non-negative integers")
        if type(self.native_competence_claimed) is not bool:
            raise ValueError("native_competence_claimed must be a bool")
        if self.native_competence_claimed and self.implementation in {
            CapabilityImplementation.EXTERNAL_SPECIALIST,
            CapabilityImplementation.TOOL,
        }:
            raise ValueError("external specialists and tools are not native competence")

        ranked = list(CapabilityQualification).index(self.state)
        smoke_rank = list(CapabilityQualification).index(
            CapabilityQualification.SMOKE_TRAINED
        )
        benchmark_rank = list(CapabilityQualification).index(
            CapabilityQualification.BENCHMARK_QUALIFIED
        )
        production_rank = list(CapabilityQualification).index(
            CapabilityQualification.PRODUCTION_QUALIFIED
        )
        if self.native_competence_claimed and ranked < smoke_rank:
            raise ValueError("untrained capability cannot claim native competence")
        evidence = (
            self.training_lineage_digest,
            self.frozen_baseline_digest,
            self.heldout_manifest_digest,
            self.evaluator_digest,
        )
        if ranked >= smoke_rank:
            if any(value is None for value in evidence):
                raise ValueError("trained qualification requires complete lineage evidence")
            for name, value in zip(
                (
                    "training_lineage_digest",
                    "frozen_baseline_digest",
                    "heldout_manifest_digest",
                    "evaluator_digest",
                ),
                evidence,
            ):
                _digest(name, value)
            if (
                not _finite_score(self.baseline_score)
                or not _finite_score(self.candidate_score)
                or float(self.candidate_score) <= float(self.baseline_score)
            ):
                raise ValueError("SMOKE_TRAINED requires beating the frozen baseline")
        elif any(value is not None for value in evidence) or any(
            value is not None for value in (self.baseline_score, self.candidate_score)
        ):
            raise ValueError("untrained qualification cannot carry training evidence")
        if ranked >= benchmark_rank:
            _digest("benchmark_receipt_digest", self.benchmark_receipt_digest)
        elif self.benchmark_receipt_digest is not None:
            raise ValueError("pre-benchmark qualification cannot carry benchmark evidence")
        if ranked >= production_rank:
            _digest("production_receipt_digest", self.production_receipt_digest)
        elif self.production_receipt_digest is not None:
            raise ValueError("pre-production qualification cannot carry production evidence")


_CAPABILITY_PORT_SPECS = (
    ("language-document", ("text", "document"), "semantic-commitment-v1"),
    ("code-math-verification", ("text", "code", "proof"), "verified-artifact-v1"),
    ("defensive-cyber", ("text", "code", "telemetry", "graph"), "cyber-advisory-v1"),
    ("vision-ocr-video-spatial", ("image", "video", "document"), "capability-observation-v2"),
    ("speech-audio-music", ("speech", "audio", "music"), "capability-observation-v2"),
    ("biology-chemistry-materials", ("sequence", "molecule", "graph"), "capability-observation-v2"),
    ("time-series-finance", ("time-series", "table"), "forecast-with-uncertainty-v1"),
    ("graph-network", ("graph", "network"), "graph-projection-v1"),
    ("robotics-action-navigation", ("sensor", "state", "video"), "bounded-action-plan-v1"),
    ("rl-games-world-model", ("state", "action", "reward"), "trajectory-evaluation-v1"),
    ("retrieval-reranking-rag", ("text", "document", "query"), "provenance-ranked-evidence-v1"),
    ("image-audio-video-generation", ("text", "image", "audio", "video"), "generated-artifact-v1"),
)
CAPABILITY_PORTS = tuple(spec[0] for spec in _CAPABILITY_PORT_SPECS)


INITIAL_CAPABILITY_STEP_ALLOCATION = {
    "language-instruction-action": 30,
    "code-math-verification": 20,
    "defensive-cyber-lab": 20,
    "vision-speech-audio": 15,
    "long-tail-capability-ports": 10,
    "affect-safety-permission-replay": 5,
}


def validate_capability_step_allocation(allocation: dict[str, int]) -> None:
    if (
        not isinstance(allocation, dict)
        or set(allocation) != set(INITIAL_CAPABILITY_STEP_ALLOCATION)
        or any(type(value) is not int or value <= 0 for value in allocation.values())
        or sum(allocation.values()) != 100
    ):
        raise ValueError("capability step allocation must contain the six fixed groups totaling 100")


def declared_capability_manifests() -> tuple[CapabilityQualificationManifestV1, ...]:
    """Return honest initial labels; implementation or training must advance them."""
    return tuple(
        CapabilityQualificationManifestV1(
            capability_id=capability_id,
            capability_version="0.1.0",
            state=CapabilityQualification.DECLARED,
            implementation=CapabilityImplementation.NATIVE_CORE,
            core_parameters=0,
            organ_parameters=0,
        )
        for capability_id in CAPABILITY_PORTS
    )


def declared_capability_ports() -> tuple[CapabilityPortV2, ...]:
    """Return every launch interface without implying that it has been trained."""
    ports = []
    for capability_id, modalities, output_contract in _CAPABILITY_PORT_SPECS:
        material = "|".join((capability_id, *modalities, output_contract)).encode("utf-8")
        ports.append(
            CapabilityPortV2(
                abi_version=CAPABILITY_PORT_ABI_VERSION,
                capability_id=capability_id,
                capability_version="0.1.0",
                input_modalities=modalities,
                output_contract=output_contract,
                contract_digest=sha256(material).hexdigest(),
            )
        )
    return tuple(ports)


def _finite_score(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _digest(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


__all__ = [
    "CAPABILITY_PORT_ABI_VERSION",
    "CAPABILITY_PORTS",
    "INITIAL_CAPABILITY_STEP_ALLOCATION",
    "CapabilityImplementation",
    "CapabilityPortV2",
    "CapabilityQualification",
    "CapabilityQualificationManifestV1",
    "declared_capability_manifests",
    "declared_capability_ports",
    "validate_capability_step_allocation",
]
