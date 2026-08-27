"""Evidence contracts for honest root-relative MOIRA growth decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
import math

from .elastic import PARAMETER_GROWTH_MULTIPLIER


MAX_GROWTH_GENERATIONS_PER_CERTIFICATE = 32


class GrowthDiagnosis(str, Enum):
    UNDERTRAINED = "UNDERTRAINED"
    UNDERCAPACITY = "UNDERCAPACITY"
    BAD_DATA = "BAD_DATA"
    BAD_OBJECTIVE = "BAD_OBJECTIVE"
    ARCHITECTURAL_LIMIT = "ARCHITECTURAL_LIMIT"


class GrowthArtifactState(str, Enum):
    MATERIALIZED_UNTRAINED = "MATERIALIZED_UNTRAINED"
    TRAINED = "TRAINED"
    EVALUATED = "EVALUATED"
    STAGED = "STAGED"
    CANARY = "CANARY"
    ACTIVE = "ACTIVE"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


@dataclass(frozen=True, slots=True)
class GrowthCertificateV1:
    parent_parameters: int
    generations: int
    diagnosis: GrowthDiagnosis
    parent_checkpoint_digest: str
    architecture_digest: str
    training_trend_digest: str
    validation_trend_digest: str
    equal_compute_parent_digest: str
    failure_cluster_digest: str
    fresh_data_manifest_digest: str
    worker_placement_digest: str
    rollback_checkpoint_digest: str
    fresh_data_records: int
    effective_rank_ratio: float
    saturation_ratio: float
    optimizer_bytes: int
    activation_bytes: int
    checkpoint_bytes: int
    evaluation_cost_units: int

    def validate(self) -> None:
        if type(self.parent_parameters) is not int or self.parent_parameters <= 0:
            raise ValueError("parent_parameters must be a positive integer")
        if (
            type(self.generations) is not int
            or not 1 <= self.generations <= MAX_GROWTH_GENERATIONS_PER_CERTIFICATE
        ):
            raise ValueError(
                "generations exceed the per-certificate operational limit"
            )
        if not isinstance(self.diagnosis, GrowthDiagnosis):
            raise ValueError("diagnosis must be a GrowthDiagnosis")
        for name, value in (
            ("parent_checkpoint_digest", self.parent_checkpoint_digest),
            ("architecture_digest", self.architecture_digest),
            ("training_trend_digest", self.training_trend_digest),
            ("validation_trend_digest", self.validation_trend_digest),
            ("equal_compute_parent_digest", self.equal_compute_parent_digest),
            ("failure_cluster_digest", self.failure_cluster_digest),
            ("fresh_data_manifest_digest", self.fresh_data_manifest_digest),
            ("worker_placement_digest", self.worker_placement_digest),
            ("rollback_checkpoint_digest", self.rollback_checkpoint_digest),
        ):
            _digest(name, value)
        if any(
            type(value) is not int or value < 0
            for value in (
                self.fresh_data_records,
                self.optimizer_bytes,
                self.activation_bytes,
                self.checkpoint_bytes,
                self.evaluation_cost_units,
            )
        ):
            raise ValueError("growth counts and estimates must be non-negative integers")
        if any(
            type(value) not in {int, float}
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in (self.effective_rank_ratio, self.saturation_ratio)
        ):
            raise ValueError("growth rank and saturation ratios must be in [0, 1]")

    @property
    def child_parameters(self) -> int:
        self.validate()
        return self.parent_parameters * PARAMETER_GROWTH_MULTIPLIER**self.generations

    @property
    def materialization_eligible(self) -> bool:
        self.validate()
        return self.diagnosis in {
            GrowthDiagnosis.UNDERCAPACITY,
            GrowthDiagnosis.ARCHITECTURAL_LIMIT,
        }

    @property
    def certificate_digest(self) -> str:
        self.validate()
        payload = {
            **asdict(self),
            "diagnosis": self.diagnosis.value,
            "schema": "GrowthCertificateV1",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ParameterReportV1:
    logical_total_parameters: int
    materialized_parameters: int
    unique_values: int
    gradient_updated_parameters: int
    qualified_parameters: int
    resident_parameters: int
    trainable_parameters: int
    dormant_parameters: int
    active_parameters_per_token: int
    peak_concurrent_parameters: int
    parameter_applications: int
    adapter_parameters: int
    organ_parameters: int
    weight_storage_precision: str
    training_master_precision: str

    def validate(self) -> None:
        counts = (
            self.logical_total_parameters,
            self.materialized_parameters,
            self.unique_values,
            self.gradient_updated_parameters,
            self.qualified_parameters,
            self.resident_parameters,
            self.trainable_parameters,
            self.dormant_parameters,
            self.active_parameters_per_token,
            self.peak_concurrent_parameters,
            self.parameter_applications,
            self.adapter_parameters,
            self.organ_parameters,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("parameter report counts must be non-negative integers")
        if self.logical_total_parameters <= 0:
            raise ValueError("logical_total_parameters must be positive")
        if not (
            self.unique_values <= self.materialized_parameters <= self.logical_total_parameters
            and self.gradient_updated_parameters <= self.materialized_parameters
            and self.qualified_parameters <= self.materialized_parameters
            and self.resident_parameters <= self.materialized_parameters
            and self.trainable_parameters <= self.materialized_parameters
            and self.dormant_parameters <= self.materialized_parameters
            and self.active_parameters_per_token <= self.resident_parameters
            and self.peak_concurrent_parameters <= self.resident_parameters
            and self.parameter_applications >= self.active_parameters_per_token
        ):
            raise ValueError("parameter report counts violate materialization invariants")
        if self.weight_storage_precision not in {"int2", "int4", "int8", "bf16", "fp16", "fp32"}:
            raise ValueError("unsupported weight_storage_precision")
        if self.training_master_precision != "fp32":
            raise ValueError("training_master_precision must be fp32")


@dataclass(frozen=True, slots=True)
class GrowthCandidateManifestV1:
    candidate_id: str
    certificate_digest: str
    parent_checkpoint_digest: str
    child_parameters: int
    state: GrowthArtifactState
    report: ParameterReportV1

    def validate(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError("candidate_id must be a nonempty string")
        _digest("certificate_digest", self.certificate_digest)
        _digest("parent_checkpoint_digest", self.parent_checkpoint_digest)
        if type(self.child_parameters) is not int or self.child_parameters <= 0:
            raise ValueError("child_parameters must be a positive integer")
        if not isinstance(self.state, GrowthArtifactState):
            raise ValueError("state must be a GrowthArtifactState")
        if not isinstance(self.report, ParameterReportV1):
            raise ValueError("report must be ParameterReportV1")
        self.report.validate()
        if self.report.logical_total_parameters != self.child_parameters:
            raise ValueError("growth candidate report does not match child parameters")
        if self.state is GrowthArtifactState.MATERIALIZED_UNTRAINED and (
            self.report.gradient_updated_parameters != 0
            or self.report.qualified_parameters != 0
        ):
            raise ValueError("materialized-untrained growth cannot claim trained parameters")
        trained_states = {
            GrowthArtifactState.TRAINED,
            GrowthArtifactState.EVALUATED,
            GrowthArtifactState.STAGED,
            GrowthArtifactState.CANARY,
            GrowthArtifactState.ACTIVE,
        }
        if (
            self.state in trained_states
            and self.report.gradient_updated_parameters <= 0
        ):
            raise ValueError("trained growth state requires gradient-updated parameters")
        qualified_states = {
            GrowthArtifactState.EVALUATED,
            GrowthArtifactState.STAGED,
            GrowthArtifactState.CANARY,
            GrowthArtifactState.ACTIVE,
        }
        if self.state in qualified_states and self.report.qualified_parameters <= 0:
            raise ValueError("evaluated growth state requires qualified parameters")
        if (
            self.state is GrowthArtifactState.ACTIVE
            and self.report.active_parameters_per_token
            > self.report.qualified_parameters
        ):
            raise ValueError("active growth cannot use unqualified parameters")


def _digest(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


__all__ = [
    "GrowthArtifactState",
    "GrowthCandidateManifestV1",
    "GrowthCertificateV1",
    "GrowthDiagnosis",
    "MAX_GROWTH_GENERATIONS_PER_CERTIFICATE",
    "ParameterReportV1",
]
