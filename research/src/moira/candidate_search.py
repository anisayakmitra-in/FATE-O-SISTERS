"""Bounded deterministic candidate search for MOIRA research artifacts.

This module selects immutable, already-evaluated candidates and mutation
failure evidence. It does not train, execute, promote, activate, or replace a
model artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import sys
from typing import Iterable


CANDIDATE_SEARCH_PROTOCOL = "moira-candidate-search-v1"
CANDIDATE_SELECTION_POLICY = "moira-score-novelty-selection-v1"
FAILURE_SELECTION_POLICY = "moira-weighted-failure-batch-v1"
MAX_CANDIDATES = 4_096
MAX_FAILURE_EVIDENCE = 16_384
MAX_PARENTS_PER_CANDIDATE = 16
MAX_SELECTION_DRAWS = 64
MAX_FAILURE_BATCH = 64
_MAX_SCORE = 1_000_000_000.0
_MAX_WEIGHT = 1_000_000.0
_MAX_SEED = (1 << 63) - 1


class CandidateKind(str, Enum):
    PROMPT = "prompt"
    ADAPTER = "adapter"
    ROUTING_POLICY = "routing_policy"
    CURRICULUM = "curriculum"
    ARCHITECTURE_CONFIG = "architecture_config"
    CORE_CHECKPOINT = "core_checkpoint"


@dataclass(frozen=True, slots=True)
class FailureEvidence:
    failure_id: str
    category: str
    dataset_id: str
    evidence_digest: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        _bounded_text("failure_id", self.failure_id, 256)
        _bounded_text("failure category", self.category, 128)
        _bounded_text("failure dataset_id", self.dataset_id, 256)
        _sha256("failure evidence_digest", self.evidence_digest)
        _bounded_float("failure weight", self.weight, minimum=0.0, maximum=_MAX_WEIGHT)
        if self.weight == 0.0:
            raise ValueError("failure weight must be positive")

    @property
    def record_digest(self) -> str:
        return _digest(
            "moira-failure-evidence-v1",
            {
                "failure_id": self.failure_id,
                "category": self.category,
                "dataset_id": self.dataset_id,
                "evidence_digest": self.evidence_digest,
                "weight": _float_text(self.weight),
            },
        )


@dataclass(frozen=True, slots=True)
class CandidateArtifact:
    candidate_id: str
    lineage_id: str
    kind: CandidateKind
    artifact_digest: str
    creation_manifest_digest: str
    parent_ids: tuple[str, ...]
    iteration: int

    def __post_init__(self) -> None:
        _bounded_text("candidate_id", self.candidate_id, 256)
        _bounded_text("candidate lineage_id", self.lineage_id, 256)
        if not isinstance(self.kind, CandidateKind):
            raise ValueError("candidate kind is invalid")
        _sha256("candidate artifact_digest", self.artifact_digest)
        _sha256("candidate creation_manifest_digest", self.creation_manifest_digest)
        if (
            not isinstance(self.parent_ids, tuple)
            or len(self.parent_ids) > MAX_PARENTS_PER_CANDIDATE
            or any(not isinstance(item, str) for item in self.parent_ids)
        ):
            raise ValueError("candidate parent IDs are outside their bounded contract")
        for parent_id in self.parent_ids:
            _bounded_text("candidate parent_id", parent_id, 256)
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError("candidate parent IDs must be unique")
        if self.candidate_id in self.parent_ids:
            raise ValueError("candidate cannot be its own parent")
        if type(self.iteration) is not int or self.iteration < 0:
            raise ValueError("candidate iteration must be a non-negative integer")
        if self.iteration == 0 and self.parent_ids:
            raise ValueError("root candidate cannot declare parents")
        if self.iteration > 0 and not self.parent_ids:
            raise ValueError("non-root candidate must declare at least one parent")

    @property
    def record_digest(self) -> str:
        return _digest(
            "moira-candidate-artifact-v1",
            {
                "candidate_id": self.candidate_id,
                "lineage_id": self.lineage_id,
                "kind": self.kind.value,
                "artifact_digest": self.artifact_digest,
                "creation_manifest_digest": self.creation_manifest_digest,
                "parent_ids": list(self.parent_ids),
                "iteration": self.iteration,
            },
        )


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    evaluation_manifest_digest: str
    metric_digest: str
    selection_score: float
    viable: bool
    mutation_verified: bool
    failure_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _sha256("candidate evaluation_manifest_digest", self.evaluation_manifest_digest)
        _sha256("candidate metric_digest", self.metric_digest)
        _bounded_float(
            "candidate selection_score",
            self.selection_score,
            minimum=-_MAX_SCORE,
            maximum=_MAX_SCORE,
        )
        if type(self.viable) is not bool or type(self.mutation_verified) is not bool:
            raise ValueError("candidate evaluation flags must be boolean")
        if (
            not isinstance(self.failure_ids, tuple)
            or len(self.failure_ids) > MAX_FAILURE_EVIDENCE
            or any(not isinstance(item, str) for item in self.failure_ids)
        ):
            raise ValueError("candidate failure IDs are outside their bounded contract")
        for failure_id in self.failure_ids:
            _bounded_text("candidate failure_id", failure_id, 256)
        if len(set(self.failure_ids)) != len(self.failure_ids):
            raise ValueError("candidate failure IDs must be unique")

    @property
    def record_digest(self) -> str:
        return _digest(
            "moira-candidate-evaluation-v1",
            {
                "evaluation_manifest_digest": self.evaluation_manifest_digest,
                "metric_digest": self.metric_digest,
                "selection_score": _float_text(self.selection_score),
                "viable": self.viable,
                "mutation_verified": self.mutation_verified,
                "failure_ids": list(self.failure_ids),
                "evaluation_role": "selection_only",
                "sealed_promotion_holdout": False,
            },
        )


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    artifact: CandidateArtifact
    evaluation: CandidateEvaluation

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, CandidateArtifact):
            raise ValueError("candidate record artifact is invalid")
        if not isinstance(self.evaluation, CandidateEvaluation):
            raise ValueError("candidate record evaluation is invalid")

    @property
    def record_digest(self) -> str:
        return _digest(
            "moira-candidate-record-v1",
            {
                "artifact_digest": self.artifact.record_digest,
                "evaluation_digest": self.evaluation.record_digest,
            },
        )


@dataclass(frozen=True, slots=True)
class CandidateDraw:
    index: int
    selected_id: str
    eligible_ids: tuple[str, ...]
    weights: tuple[str, ...]
    draw_digest: str


@dataclass(frozen=True, slots=True)
class CandidateSelectionReceipt:
    protocol: str
    policy: str
    iteration: int
    population_digest: str
    seed: int
    sharpness: str
    midpoint_percentile: str
    novelty_weight: str
    draws: tuple[CandidateDraw, ...]
    selected_ids: tuple[str, ...]
    selection_digest: str
    can_authorize: bool = False
    can_promote: bool = False


@dataclass(frozen=True, slots=True)
class FailureSelectionReceipt:
    protocol: str
    policy: str
    candidate_id: str
    candidate_digest: str
    population_digest: str
    seed: int
    category: str
    eligible_failure_ids: tuple[str, ...]
    selected_failure_ids: tuple[str, ...]
    selection_digest: str
    can_authorize: bool = False
    can_promote: bool = False


class CandidateSearch:
    """In-memory bounded archive and deterministic research selector."""

    def __init__(self, failures: Iterable[FailureEvidence] = ()) -> None:
        self._failures: dict[str, FailureEvidence] = {}
        self._candidates: dict[str, CandidateRecord] = {}
        self._artifact_ids: dict[str, str] = {}
        try:
            initial_failures = tuple(failures)
        except TypeError as error:
            raise ValueError("failure evidence must be iterable") from error
        for failure in initial_failures:
            self.add_failure(failure)

    def add_failure(self, failure: FailureEvidence) -> None:
        if not isinstance(failure, FailureEvidence):
            raise ValueError("failure evidence must be a FailureEvidence value")
        if len(self._failures) >= MAX_FAILURE_EVIDENCE:
            raise ValueError("failure evidence archive is full")
        if failure.failure_id in self._failures:
            raise ValueError("failure evidence ID already exists")
        self._failures[failure.failure_id] = failure

    def add_candidate(self, record: CandidateRecord) -> None:
        if not isinstance(record, CandidateRecord):
            raise ValueError("candidate must be a CandidateRecord")
        if len(self._candidates) >= MAX_CANDIDATES:
            raise ValueError("candidate archive is full")
        artifact = record.artifact
        if artifact.candidate_id in self._candidates:
            raise ValueError("candidate ID already exists")
        if artifact.artifact_digest in self._artifact_ids:
            raise ValueError("candidate artifact already exists")
        missing_failures = set(record.evaluation.failure_ids) - self._failures.keys()
        if missing_failures:
            raise ValueError("candidate evaluation references unknown failure evidence")
        if not self._candidates:
            if artifact.iteration != 0 or artifact.parent_ids:
                raise ValueError("first candidate must be the root")
        else:
            if artifact.iteration == 0:
                raise ValueError("candidate archive can contain only one root")
            parents = []
            for parent_id in artifact.parent_ids:
                parent = self._candidates.get(parent_id)
                if parent is None:
                    raise ValueError("candidate parent is unknown")
                parents.append(parent)
            if any(
                parent.artifact.lineage_id != artifact.lineage_id
                for parent in parents
            ):
                raise ValueError("candidate parent belongs to another lineage")
            if artifact.iteration <= max(parent.artifact.iteration for parent in parents):
                raise ValueError("candidate iteration must advance every parent")
        self._candidates[artifact.candidate_id] = record
        self._artifact_ids[artifact.artifact_digest] = artifact.candidate_id

    @property
    def candidates(self) -> tuple[CandidateRecord, ...]:
        return tuple(self._candidates[key] for key in sorted(self._candidates))

    @property
    def failures(self) -> tuple[FailureEvidence, ...]:
        return tuple(self._failures[key] for key in sorted(self._failures))

    @property
    def population_digest(self) -> str:
        return _digest(
            "moira-candidate-population-v1",
            {
                "protocol": CANDIDATE_SEARCH_PROTOCOL,
                "candidates": [item.record_digest for item in self.candidates],
                "failures": [item.record_digest for item in self.failures],
            },
        )

    def select_parents(
        self,
        *,
        count: int,
        seed: int,
        iteration: int,
        sharpness: float = 10.0,
        midpoint_percentile: float = 75.0,
        novelty_weight: float = 1.0,
    ) -> CandidateSelectionReceipt:
        _bounded_count("selection count", count, MAX_SELECTION_DRAWS)
        _seed(seed)
        if type(iteration) is not int or iteration <= 0:
            raise ValueError("selection iteration must be a positive integer")
        _bounded_float("selection sharpness", sharpness, minimum=0.0, maximum=100.0)
        if sharpness == 0.0:
            raise ValueError("selection sharpness must be positive")
        _bounded_float(
            "selection midpoint_percentile",
            midpoint_percentile,
            minimum=0.0,
            maximum=100.0,
        )
        _bounded_float(
            "selection novelty_weight",
            novelty_weight,
            minimum=0.0,
            maximum=_MAX_WEIGHT,
        )
        eligible = tuple(
            record
            for record in self.candidates
            if record.evaluation.viable
            and record.evaluation.mutation_verified
            and record.evaluation.failure_ids
            and record.artifact.iteration < iteration
        )
        if not eligible:
            raise ValueError("candidate archive has no eligible mutation parents")
        midpoint = _percentile(
            tuple(record.evaluation.selection_score for record in eligible),
            midpoint_percentile,
        )
        child_counts = self._child_counts()
        temporary_counts = {
            record.artifact.candidate_id: float(child_counts[record.artifact.candidate_id])
            for record in eligible
        }
        population_digest = self.population_digest
        draws: list[CandidateDraw] = []
        selected: list[str] = []
        for index in range(count):
            weights = tuple(
                _selection_weight(
                    record.evaluation.selection_score,
                    midpoint=midpoint,
                    sharpness=sharpness,
                    novelty_weight=novelty_weight,
                    child_count=temporary_counts[record.artifact.candidate_id],
                )
                for record in eligible
            )
            draw_digest = _digest(
                "moira-candidate-draw-v1",
                {
                    "population_digest": population_digest,
                    "seed": seed,
                    "iteration": iteration,
                    "index": index,
                    "eligible_ids": [item.artifact.candidate_id for item in eligible],
                    "weights": [_float_text(item) for item in weights],
                },
            )
            chosen = eligible[_weighted_index(weights, draw_digest)]
            chosen_id = chosen.artifact.candidate_id
            selected.append(chosen_id)
            temporary_counts[chosen_id] += 1.0
            draws.append(
                CandidateDraw(
                    index=index,
                    selected_id=chosen_id,
                    eligible_ids=tuple(item.artifact.candidate_id for item in eligible),
                    weights=tuple(_float_text(item) for item in weights),
                    draw_digest=draw_digest,
                )
            )
        receipt_value = {
            "protocol": CANDIDATE_SEARCH_PROTOCOL,
            "policy": CANDIDATE_SELECTION_POLICY,
            "iteration": iteration,
            "population_digest": population_digest,
            "seed": seed,
            "sharpness": _float_text(sharpness),
            "midpoint_percentile": _float_text(midpoint_percentile),
            "novelty_weight": _float_text(novelty_weight),
            "draws": [
                {
                    "index": item.index,
                    "selected_id": item.selected_id,
                    "eligible_ids": list(item.eligible_ids),
                    "weights": list(item.weights),
                    "draw_digest": item.draw_digest,
                }
                for item in draws
            ],
            "selected_ids": selected,
            "can_authorize": False,
            "can_promote": False,
        }
        return CandidateSelectionReceipt(
            protocol=CANDIDATE_SEARCH_PROTOCOL,
            policy=CANDIDATE_SELECTION_POLICY,
            iteration=iteration,
            population_digest=population_digest,
            seed=seed,
            sharpness=_float_text(sharpness),
            midpoint_percentile=_float_text(midpoint_percentile),
            novelty_weight=_float_text(novelty_weight),
            draws=tuple(draws),
            selected_ids=tuple(selected),
            selection_digest=_digest(
                "moira-candidate-selection-receipt-v1", receipt_value
            ),
        )

    def select_failures(
        self,
        candidate_id: str,
        *,
        count: int,
        seed: int,
    ) -> FailureSelectionReceipt:
        _bounded_text("failure selection candidate_id", candidate_id, 256)
        _bounded_count("failure selection count", count, MAX_FAILURE_BATCH)
        _seed(seed)
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise ValueError("failure selection candidate is unknown")
        failures = tuple(
            self._failures[failure_id]
            for failure_id in candidate.evaluation.failure_ids
        )
        if not failures:
            raise ValueError("candidate has no mutation failure evidence")
        categories = sorted({item.category for item in failures})
        category_weights = tuple(
            sum(item.weight for item in failures if item.category == category)
            for category in categories
        )
        population_digest = self.population_digest
        category_draw = _digest(
            "moira-failure-category-draw-v1",
            {
                "population_digest": population_digest,
                "candidate_digest": candidate.record_digest,
                "seed": seed,
                "categories": categories,
                "weights": [_float_text(item) for item in category_weights],
            },
        )
        category = categories[_weighted_index(category_weights, category_draw)]
        eligible = sorted(
            (item for item in failures if item.category == category),
            key=lambda item: item.failure_id,
        )
        selected: list[str] = []
        remaining = list(eligible)
        effective_count = min(count, len(remaining))
        for index in range(effective_count):
            draw_digest = _digest(
                "moira-failure-evidence-draw-v1",
                {
                    "population_digest": population_digest,
                    "candidate_digest": candidate.record_digest,
                    "seed": seed,
                    "category": category,
                    "index": index,
                    "remaining_ids": [item.failure_id for item in remaining],
                    "weights": [_float_text(item.weight) for item in remaining],
                },
            )
            selected_index = _weighted_index(
                tuple(item.weight for item in remaining), draw_digest
            )
            selected.append(remaining.pop(selected_index).failure_id)
        receipt_value = {
            "protocol": CANDIDATE_SEARCH_PROTOCOL,
            "policy": FAILURE_SELECTION_POLICY,
            "candidate_id": candidate_id,
            "candidate_digest": candidate.record_digest,
            "population_digest": population_digest,
            "seed": seed,
            "category": category,
            "eligible_failure_ids": [item.failure_id for item in eligible],
            "selected_failure_ids": selected,
            "can_authorize": False,
            "can_promote": False,
        }
        return FailureSelectionReceipt(
            protocol=CANDIDATE_SEARCH_PROTOCOL,
            policy=FAILURE_SELECTION_POLICY,
            candidate_id=candidate_id,
            candidate_digest=candidate.record_digest,
            population_digest=population_digest,
            seed=seed,
            category=category,
            eligible_failure_ids=tuple(item.failure_id for item in eligible),
            selected_failure_ids=tuple(selected),
            selection_digest=_digest(
                "moira-failure-selection-receipt-v1", receipt_value
            ),
        )

    def verify_parent_selection(
        self, receipt: CandidateSelectionReceipt
    ) -> None:
        if not isinstance(receipt, CandidateSelectionReceipt):
            raise ValueError("candidate selection receipt is invalid")
        try:
            expected = self.select_parents(
                count=len(receipt.draws),
                seed=receipt.seed,
                iteration=receipt.iteration,
                sharpness=float(receipt.sharpness),
                midpoint_percentile=float(receipt.midpoint_percentile),
                novelty_weight=float(receipt.novelty_weight),
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("candidate selection receipt cannot be replayed") from error
        if receipt != expected:
            raise ValueError("candidate selection receipt does not match the archive")

    def verify_failure_selection(self, receipt: FailureSelectionReceipt) -> None:
        if not isinstance(receipt, FailureSelectionReceipt):
            raise ValueError("failure selection receipt is invalid")
        try:
            expected = self.select_failures(
                receipt.candidate_id,
                count=len(receipt.selected_failure_ids),
                seed=receipt.seed,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("failure selection receipt cannot be replayed") from error
        if receipt != expected:
            raise ValueError("failure selection receipt does not match the archive")

    def _child_counts(self) -> dict[str, int]:
        counts = {candidate_id: 0 for candidate_id in self._candidates}
        for record in self._candidates.values():
            for parent_id in record.artifact.parent_ids:
                counts[parent_id] += 1
        return counts


def candidate_search_contract() -> dict[str, object]:
    return {
        "protocol": CANDIDATE_SEARCH_PROTOCOL,
        "selection_policy": CANDIDATE_SELECTION_POLICY,
        "failure_selection_policy": FAILURE_SELECTION_POLICY,
        "maximum_candidates": MAX_CANDIDATES,
        "maximum_failure_evidence": MAX_FAILURE_EVIDENCE,
        "maximum_selection_draws": MAX_SELECTION_DRAWS,
        "selection_evaluation_is_promotion_holdout": False,
        "executes_mutations": False,
        "trains_weights": False,
        "activates_candidates": False,
        "can_authorize": False,
        "can_promote": False,
    }


def _selection_weight(
    score: float,
    *,
    midpoint: float,
    sharpness: float,
    novelty_weight: float,
    child_count: float,
) -> float:
    value = sharpness * (score - midpoint)
    if value >= 0:
        performance = 1.0 / (1.0 + math.exp(-value))
    else:
        exponential = math.exp(value)
        performance = exponential / (1.0 + exponential)
    novelty = 1.0 / (1.0 + novelty_weight * child_count)
    return max(performance * novelty, sys.float_info.min)


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _weighted_index(weights: tuple[float, ...], draw_digest: str) -> int:
    if not weights or any(not math.isfinite(item) or item <= 0.0 for item in weights):
        raise ValueError("selection weights must be finite and positive")
    total = sum(weights)
    point = (int(draw_digest[:16], 16) / float(1 << 64)) * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if point < cumulative:
            return index
    return len(weights) - 1


def _bounded_text(name: str, value: object, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValueError(f"{name} must be bounded non-empty text")


def _sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _bounded_float(name: str, value: object, *, minimum: float, maximum: float) -> None:
    if (
        not isinstance(value, float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} is outside its bounded finite range")


def _bounded_count(name: str, value: object, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} is outside its bounded range")


def _seed(value: object) -> None:
    if type(value) is not int or not 0 <= value <= _MAX_SEED:
        raise ValueError("selection seed is outside its bounded range")


def _float_text(value: float) -> str:
    return format(value, ".17g")


def _digest(domain: str, value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(domain.encode("ascii") + b"\x00" + encoded).hexdigest()


__all__ = [
    "CANDIDATE_SEARCH_PROTOCOL",
    "CANDIDATE_SELECTION_POLICY",
    "FAILURE_SELECTION_POLICY",
    "MAX_CANDIDATES",
    "MAX_FAILURE_EVIDENCE",
    "MAX_SELECTION_DRAWS",
    "CandidateArtifact",
    "CandidateDraw",
    "CandidateEvaluation",
    "CandidateKind",
    "CandidateRecord",
    "CandidateSearch",
    "CandidateSelectionReceipt",
    "FailureEvidence",
    "FailureSelectionReceipt",
    "candidate_search_contract",
]
