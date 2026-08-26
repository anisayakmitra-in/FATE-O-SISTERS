from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import pytest

from moira.agent import AgentPolicy, AgentRole, MoiraAgentKernel
from moira.candidate_search import (
    CandidateArtifact,
    CandidateEvaluation,
    CandidateKind,
    CandidateRecord,
    CandidateSearch,
    FailureEvidence,
)
from moira.candidate_worker import (
    CANDIDATE_WORK_TOOL_ID,
    CandidateWorkBudget,
    CandidateWorkUsage,
    CandidateWorker,
    CandidateWorkerOutput,
    candidate_worker_contract,
    prepare_candidate_work,
)
from moira.masks import CognitiveMode


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def candidate_root() -> Iterator[Path]:
    root = (
        REPOSITORY_ROOT
        / "research"
        / "runs"
        / f"candidate-worker-{uuid.uuid4().hex}"
    )
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _sha(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _search() -> CandidateSearch:
    failure = FailureEvidence(
        failure_id="reasoning-1",
        category="reasoning",
        dataset_id="selection-v1",
        evidence_digest=_sha("failure-evidence"),
    )
    search = CandidateSearch((failure,))
    search.add_candidate(
        CandidateRecord(
            artifact=CandidateArtifact(
                candidate_id="root",
                lineage_id="lineage-1",
                kind=CandidateKind.CORE_CHECKPOINT,
                artifact_digest=_sha("root-artifact"),
                creation_manifest_digest=_sha("root-creation"),
                parent_ids=(),
                iteration=0,
            ),
            evaluation=CandidateEvaluation(
                evaluation_manifest_digest=_sha("root-evaluation"),
                metric_digest=_sha("root-metric"),
                selection_score=0.5,
                viable=True,
                mutation_verified=True,
                failure_ids=("reasoning-1",),
            ),
        )
    )
    return search


def _request(
    candidate_root: Path,
    *,
    now: int = 100,
    max_steps: int = 4,
):
    active = candidate_root / "active.ckpt"
    active.write_bytes(b"immutable active checkpoint")
    search = _search()
    parents = search.select_parents(count=1, seed=7, iteration=1)
    failures = search.select_failures("root", count=1, seed=11)
    request = prepare_candidate_work(
        search,
        parents,
        failures,
        work_id="work-1",
        candidate_id="candidate-1",
        kind=CandidateKind.ADAPTER,
        active_checkpoint_digest=_sha(active.read_bytes()),
        dataset_manifest_digest=_sha("training-dataset"),
        trainer_manifest_digest=_sha("trainer-config"),
        issued_at_ms=now,
        expires_at_ms=now + 700,
        budget=CandidateWorkBudget(
            max_steps=max_steps,
            max_samples=8,
            max_tokens=128,
            max_wall_ms=700,
            max_output_bytes=4_096,
        ),
    )
    return search, parents, failures, request, active


def _issue(worker: CandidateWorker, prepared) -> None:
    search, parents, failures, request, _ = prepared
    worker.issue(
        request,
        search=search,
        parent_selection=parents,
        failure_selection=failures,
    )


def _output(workspace: Path, *, steps: int = 2) -> CandidateWorkerOutput:
    (workspace / "candidate.bin").write_bytes(b"new candidate weights")
    (workspace / "creation.json").write_text(
        '{"trainer":"fixture"}', encoding="utf-8"
    )
    (workspace / "evaluation.json").write_text(
        '{"split":"selection"}', encoding="utf-8"
    )
    (workspace / "metrics.json").write_text(
        '{"loss":0.25}', encoding="utf-8"
    )
    return CandidateWorkerOutput(
        artifact_file="candidate.bin",
        creation_manifest_file="creation.json",
        evaluation_manifest_file="evaluation.json",
        metric_file="metrics.json",
        selection_score=0.75,
        viable=True,
        mutation_verified=True,
        remaining_failure_ids=(),
        usage=CandidateWorkUsage(steps=steps, samples=4, tokens=64),
    )


def _kernel(worker: CandidateWorker, *, max_effects: int = 2) -> MoiraAgentKernel:
    kernel = MoiraAgentKernel(
        AgentPolicy(
            policy_version=1,
            allowed_actions=frozenset({"candidate_training:run"}),
        ),
        permit_secret=b"candidate-worker-test-secret-32b" * 2,
    )
    kernel.register_tool(worker.tool())
    kernel.open_lease(
        lease_id="candidate-lease",
        task_id="candidate-task",
        scope_id="lineage-1",
        mode=CognitiveMode.ATROPOS,
        role=AgentRole.EXECUTOR,
        max_turns=1,
        max_effects=max_effects,
        allowed_actions=("candidate_training:run",),
    )
    kernel.record_turn("candidate-lease", "Run the bounded candidate job")
    return kernel


def _execute(kernel: MoiraAgentKernel, request):
    outcome = kernel.request_tool(
        "candidate-lease",
        tool_id=CANDIDATE_WORK_TOOL_ID,
        target=request.work_id,
        payload=request.payload,
    )
    assert outcome.authorized is not None
    return kernel.execute_effect(outcome.authorized, CANDIDATE_WORK_TOOL_ID)


def test_permit_gated_worker_produces_content_addressed_candidate(
    candidate_root: Path,
) -> None:
    prepared = _request(candidate_root)
    search, _, _, request, active = prepared
    calls: list[str] = []

    def backend(work, workspace: Path) -> CandidateWorkerOutput:
        calls.append(work.request_digest)
        return _output(workspace)

    worker = CandidateWorker(
        backend,
        active_checkpoint=active,
        staging_root=candidate_root / "staging",
        clock_ms=lambda: 100,
    )
    _issue(worker, prepared)
    kernel = _kernel(worker)

    execution = _execute(kernel, request)
    receipt = worker.result(request.work_id)

    assert execution.receipt.outcome == "succeeded"
    assert execution.error_code is None
    assert json.loads(execution.output)["result_digest"] == receipt.result_digest
    assert calls == [request.request_digest]
    assert receipt.artifact_digest == _sha(b"new candidate weights")
    assert receipt.candidate.artifact.parent_ids == ("root",)
    assert receipt.candidate.artifact.iteration == 1
    assert receipt.can_authorize is False
    assert receipt.can_promote is False
    assert active.read_bytes() == b"immutable active checkpoint"

    search.add_candidate(receipt.candidate)
    assert tuple(item.artifact.candidate_id for item in search.candidates) == (
        "candidate-1",
        "root",
    )


def test_work_request_is_one_shot_even_with_a_second_agent_permit(
    candidate_root: Path,
) -> None:
    prepared = _request(candidate_root)
    _, _, _, request, active = prepared
    calls = 0

    def backend(work, workspace: Path) -> CandidateWorkerOutput:
        nonlocal calls
        calls += 1
        return _output(workspace)

    worker = CandidateWorker(
        backend,
        active_checkpoint=active,
        staging_root=candidate_root / "staging",
        clock_ms=lambda: 100,
    )
    _issue(worker, prepared)
    kernel = _kernel(worker)

    assert _execute(kernel, request).receipt.outcome == "succeeded"
    with pytest.raises(ValueError, match="already submitted"):
        _execute(kernel, request)
    assert calls == 1


def test_worker_rejects_payload_not_bound_to_issued_request(
    candidate_root: Path,
) -> None:
    prepared = _request(candidate_root)
    _, _, _, request, active = prepared
    called = False

    def backend(work, workspace: Path) -> CandidateWorkerOutput:
        nonlocal called
        called = True
        return _output(workspace)

    worker = CandidateWorker(
        backend,
        active_checkpoint=active,
        staging_root=candidate_root / "staging",
        clock_ms=lambda: 100,
    )
    _issue(worker, prepared)
    kernel = _kernel(worker, max_effects=1)
    outcome = kernel.request_tool(
        "candidate-lease",
        tool_id=CANDIDATE_WORK_TOOL_ID,
        target=request.work_id,
        payload=b"{}",
    )
    assert outcome.authorized is not None

    execution = kernel.execute_effect(outcome.authorized, CANDIDATE_WORK_TOOL_ID)

    assert execution.receipt.outcome == "failed"
    assert execution.error_code == "ValueError"
    assert called is False


def test_worker_rejects_budget_overrun(candidate_root: Path) -> None:
    prepared = _request(candidate_root, max_steps=1)
    _, _, _, request, active = prepared
    worker = CandidateWorker(
        lambda work, workspace: _output(workspace, steps=2),
        active_checkpoint=active,
        staging_root=candidate_root / "staging",
        clock_ms=lambda: 100,
    )
    _issue(worker, prepared)

    execution = _execute(_kernel(worker, max_effects=1), request)

    assert execution.receipt.outcome == "failed"
    assert execution.error_code == "ValueError"
    with pytest.raises(ValueError, match="unavailable"):
        worker.result(request.work_id)


def test_worker_detects_active_checkpoint_change(candidate_root: Path) -> None:
    prepared = _request(candidate_root)
    _, _, _, request, active = prepared

    def backend(work, workspace: Path) -> CandidateWorkerOutput:
        active.write_bytes(b"mutated active checkpoint")
        return _output(workspace)

    worker = CandidateWorker(
        backend,
        active_checkpoint=active,
        staging_root=candidate_root / "staging",
        clock_ms=lambda: 100,
    )
    _issue(worker, prepared)

    execution = _execute(_kernel(worker, max_effects=1), request)

    assert execution.receipt.outcome == "failed"
    assert execution.error_code == "RuntimeError"
    with pytest.raises(ValueError, match="unavailable"):
        worker.result(request.work_id)


def test_worker_rejects_hard_linked_candidate_output(candidate_root: Path) -> None:
    prepared = _request(candidate_root)
    _, _, _, request, active = prepared
    outside = candidate_root / "outside.bin"
    outside.write_bytes(b"outside candidate")
    probe = candidate_root / "hard-link-probe.bin"
    try:
        os.link(outside, probe)
    except OSError as error:
        pytest.skip(f"hard links are unavailable: {error}")
    finally:
        probe.unlink(missing_ok=True)

    def backend(work, workspace: Path) -> CandidateWorkerOutput:
        output = _output(workspace)
        (workspace / output.artifact_file).unlink()
        os.link(outside, workspace / output.artifact_file)
        return output

    worker = CandidateWorker(
        backend,
        active_checkpoint=active,
        staging_root=candidate_root / "staging",
        clock_ms=lambda: 100,
    )
    _issue(worker, prepared)

    execution = _execute(_kernel(worker, max_effects=1), request)

    assert execution.receipt.outcome == "failed"
    assert execution.error_code == "ValueError"
    with pytest.raises(ValueError, match="unavailable"):
        worker.result(request.work_id)


def test_expired_request_never_calls_backend(candidate_root: Path) -> None:
    prepared = _request(candidate_root)
    _, _, _, request, active = prepared
    now = [100]
    called = False

    def backend(work, workspace: Path) -> CandidateWorkerOutput:
        nonlocal called
        called = True
        return _output(workspace)

    worker = CandidateWorker(
        backend,
        active_checkpoint=active,
        staging_root=candidate_root / "staging",
        clock_ms=lambda: now[0],
    )
    _issue(worker, prepared)
    now[0] = request.expires_at_ms

    execution = _execute(_kernel(worker, max_effects=1), request)

    assert execution.receipt.outcome == "failed"
    assert called is False


def test_request_expiring_during_backend_does_not_produce_result(
    candidate_root: Path,
) -> None:
    prepared = _request(candidate_root)
    _, _, _, request, active = prepared
    times = iter((100, 100, request.expires_at_ms))
    worker = CandidateWorker(
        lambda work, workspace: _output(workspace),
        active_checkpoint=active,
        staging_root=candidate_root / "staging",
        clock_ms=lambda: next(times),
    )
    _issue(worker, prepared)

    execution = _execute(_kernel(worker, max_effects=1), request)

    assert execution.receipt.outcome == "failed"
    assert execution.error_code == "ValueError"
    with pytest.raises(ValueError, match="unavailable"):
        worker.result(request.work_id)


def test_worker_issue_reconstructs_request_from_selection_evidence(
    candidate_root: Path,
) -> None:
    prepared = _request(candidate_root)
    search, parents, failures, request, active = prepared
    worker = CandidateWorker(
        lambda work, workspace: _output(workspace),
        active_checkpoint=active,
        staging_root=candidate_root / "staging",
        clock_ms=lambda: 100,
    )
    tampered = replace(
        request,
        parent_selection_digest=_sha("different-selection"),
    )

    with pytest.raises(ValueError, match="does not match selection evidence"):
        worker.issue(
            tampered,
            search=search,
            parent_selection=parents,
            failure_selection=failures,
        )


def test_search_rejects_tampered_selection_receipts() -> None:
    search = _search()
    parents = search.select_parents(count=1, seed=7, iteration=1)
    failures = search.select_failures("root", count=1, seed=11)

    with pytest.raises(ValueError, match="does not match"):
        search.verify_parent_selection(
            replace(parents, selection_digest=_sha("tampered-parent"))
        )
    with pytest.raises(ValueError, match="does not match"):
        search.verify_failure_selection(
            replace(failures, selection_digest=_sha("tampered-failure"))
        )


def test_candidate_output_paths_and_contract_fail_closed() -> None:
    with pytest.raises(ValueError, match="relative file path"):
        CandidateWorkerOutput(
            artifact_file="../candidate.bin",
            creation_manifest_file="creation.json",
            evaluation_manifest_file="evaluation.json",
            metric_file="metrics.json",
            selection_score=0.0,
            viable=True,
            mutation_verified=True,
            remaining_failure_ids=(),
            usage=CandidateWorkUsage(steps=0, samples=0, tokens=0),
        )
    contract = candidate_worker_contract()
    assert contract["requires_native_one_use_effect_permit"] is True
    assert contract["accepts_commands_from_request"] is False
    assert contract["os_read_only_mount_required_for_prevention"] is True
    assert contract["can_authorize"] is False
    assert contract["can_promote"] is False
