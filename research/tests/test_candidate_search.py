from __future__ import annotations

import hashlib
import math

import pytest

from moira.candidate_search import (
    CANDIDATE_SEARCH_PROTOCOL,
    CandidateArtifact,
    CandidateEvaluation,
    CandidateKind,
    CandidateRecord,
    CandidateSearch,
    FailureEvidence,
    candidate_search_contract,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _failure(
    failure_id: str,
    *,
    category: str = "reasoning",
    weight: float = 1.0,
) -> FailureEvidence:
    return FailureEvidence(
        failure_id=failure_id,
        category=category,
        dataset_id="selection-validation-v1",
        evidence_digest=_sha(f"evidence:{failure_id}"),
        weight=weight,
    )


def _record(
    candidate_id: str,
    *,
    parents: tuple[str, ...] = (),
    iteration: int = 0,
    score: float = 0.0,
    failures: tuple[str, ...] = ("failure-1",),
    viable: bool = True,
    verified: bool = True,
    lineage: str = "lineage-1",
    artifact_suffix: str | None = None,
) -> CandidateRecord:
    suffix = artifact_suffix or candidate_id
    return CandidateRecord(
        artifact=CandidateArtifact(
            candidate_id=candidate_id,
            lineage_id=lineage,
            kind=CandidateKind.ADAPTER,
            artifact_digest=_sha(f"artifact:{suffix}"),
            creation_manifest_digest=_sha(f"creation:{candidate_id}"),
            parent_ids=parents,
            iteration=iteration,
        ),
        evaluation=CandidateEvaluation(
            evaluation_manifest_digest=_sha(f"evaluation:{candidate_id}"),
            metric_digest=_sha(f"metric:{candidate_id}"),
            selection_score=score,
            viable=viable,
            mutation_verified=verified,
            failure_ids=failures,
        ),
    )


def _archive(*, reverse_siblings: bool = False) -> CandidateSearch:
    search = CandidateSearch(
        (
            _failure("failure-1", weight=3.0),
            _failure("failure-2", category="code", weight=2.0),
            _failure("failure-3", category="code", weight=1.0),
        )
    )
    search.add_candidate(
        _record("root", verified=False, failures=("failure-1",))
    )
    siblings = (
        _record(
            "candidate-high",
            parents=("root",),
            iteration=1,
            score=0.9,
            failures=("failure-1", "failure-2", "failure-3"),
        ),
        _record(
            "candidate-low",
            parents=("root",),
            iteration=1,
            score=0.1,
            failures=("failure-1", "failure-2"),
        ),
    )
    for record in reversed(siblings) if reverse_siblings else siblings:
        search.add_candidate(record)
    return search


def test_parent_selection_is_deterministic_and_order_independent() -> None:
    first = _archive()
    second = _archive(reverse_siblings=True)

    first_receipt = first.select_parents(count=8, seed=71, iteration=2)
    repeated = first.select_parents(count=8, seed=71, iteration=2)
    reordered = second.select_parents(count=8, seed=71, iteration=2)

    assert first.population_digest == second.population_digest
    assert first_receipt == repeated == reordered
    assert first_receipt.protocol == CANDIDATE_SEARCH_PROTOCOL
    assert first_receipt.can_authorize is False
    assert first_receipt.can_promote is False
    assert len(first_receipt.draws) == 8


def test_parent_selection_uses_score_and_novelty_weights() -> None:
    receipt = _archive().select_parents(count=3, seed=9, iteration=2)
    first = receipt.draws[0]
    weights = dict(zip(first.eligible_ids, map(float, first.weights), strict=True))

    assert weights["candidate-high"] > weights["candidate-low"]
    selected = receipt.selected_ids[0]
    selected_index = first.eligible_ids.index(selected)
    second = receipt.draws[1]
    assert float(second.weights[selected_index]) < float(first.weights[selected_index])


def test_failure_selection_is_one_category_without_replacement() -> None:
    search = _archive()

    receipt = search.select_failures("candidate-high", count=10, seed=17)
    repeated = search.select_failures("candidate-high", count=10, seed=17)

    assert receipt == repeated
    assert receipt.can_authorize is False
    assert receipt.can_promote is False
    assert len(receipt.selected_failure_ids) == len(set(receipt.selected_failure_ids))
    known = {item.failure_id: item for item in search.failures}
    assert {known[item].category for item in receipt.selected_failure_ids} == {
        receipt.category
    }
    assert set(receipt.selected_failure_ids) == set(receipt.eligible_failure_ids)


def test_receipts_bind_seed_and_population() -> None:
    search = _archive()
    baseline = search.select_parents(count=4, seed=3, iteration=2)
    other_seed = search.select_parents(count=4, seed=4, iteration=2)
    search.add_candidate(
        _record(
            "candidate-third",
            parents=("root",),
            iteration=1,
            score=0.5,
        )
    )
    expanded = search.select_parents(count=4, seed=3, iteration=2)

    assert baseline.selection_digest != other_seed.selection_digest
    assert baseline.population_digest != expanded.population_digest
    assert baseline.selection_digest != expanded.selection_digest


def test_only_viable_verified_candidates_with_failure_evidence_are_eligible() -> None:
    search = CandidateSearch((_failure("failure-1"),))
    search.add_candidate(_record("root", verified=False))
    search.add_candidate(
        _record("not-viable", parents=("root",), iteration=1, viable=False)
    )
    search.add_candidate(
        _record("not-verified", parents=("root",), iteration=1, verified=False)
    )
    search.add_candidate(
        _record("no-failures", parents=("root",), iteration=1, failures=())
    )

    with pytest.raises(ValueError, match="no eligible"):
        search.select_parents(count=1, seed=1, iteration=2)


@pytest.mark.parametrize(
    ("record", "message"),
    (
        (_record("not-root", parents=("missing",), iteration=1), "first candidate"),
        (_record("second-root"), "only one root"),
    ),
)
def test_archive_rejects_invalid_root_structure(
    record: CandidateRecord, message: str
) -> None:
    search = CandidateSearch((_failure("failure-1"),))
    if record.artifact.candidate_id == "second-root":
        search.add_candidate(_record("root"))
    with pytest.raises(ValueError, match=message):
        search.add_candidate(record)


def test_archive_rejects_unknown_parent_failure_duplicate_and_lineage() -> None:
    search = CandidateSearch((_failure("failure-1"),))
    search.add_candidate(_record("root"))

    with pytest.raises(ValueError, match="parent is unknown"):
        search.add_candidate(
            _record("unknown-parent", parents=("missing",), iteration=1)
        )
    with pytest.raises(ValueError, match="another lineage"):
        search.add_candidate(
            _record(
                "other-lineage",
                parents=("root",),
                iteration=1,
                lineage="lineage-2",
            )
        )
    with pytest.raises(ValueError, match="unknown failure"):
        search.add_candidate(
            _record(
                "unknown-failure",
                parents=("root",),
                iteration=1,
                failures=("missing",),
            )
        )
    with pytest.raises(ValueError, match="artifact already exists"):
        search.add_candidate(
            _record(
                "duplicate-artifact",
                parents=("root",),
                iteration=1,
                artifact_suffix="root",
            )
        )


def test_candidate_values_fail_closed() -> None:
    with pytest.raises(ValueError, match="selection_score"):
        CandidateEvaluation(
            evaluation_manifest_digest=_sha("evaluation"),
            metric_digest=_sha("metric"),
            selection_score=math.nan,
            viable=True,
            mutation_verified=True,
            failure_ids=(),
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        FailureEvidence(
            failure_id="failure",
            category="code",
            dataset_id="dataset",
            evidence_digest="not-a-digest",
        )
    with pytest.raises(ValueError, match="must be unique"):
        CandidateArtifact(
            candidate_id="candidate",
            lineage_id="lineage",
            kind=CandidateKind.ADAPTER,
            artifact_digest=_sha("artifact"),
            creation_manifest_digest=_sha("manifest"),
            parent_ids=("parent", "parent"),
            iteration=1,
        )


def test_contract_separates_search_from_training_authority_and_promotion() -> None:
    contract = candidate_search_contract()

    assert contract["selection_evaluation_is_promotion_holdout"] is False
    assert contract["executes_mutations"] is False
    assert contract["trains_weights"] is False
    assert contract["activates_candidates"] is False
    assert contract["can_authorize"] is False
    assert contract["can_promote"] is False
