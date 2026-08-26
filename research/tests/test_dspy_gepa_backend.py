from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
import hashlib
import json
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
    CandidateWorkRequest,
    CandidateWorker,
    prepare_candidate_work,
)
from moira.dspy_gepa_backend import (
    DSPY_GEPA_BACKEND_PROTOCOL,
    DspyGepaProfile,
    DspyGepaPromptBackend,
    DspyGepaRun,
    PromptDatasetBundle,
    PromptRecord,
    PromptSelection,
    dspy_gepa_backend_contract,
)
from moira.masks import CognitiveMode


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def dspy_root() -> Iterator[Path]:
    root = REPOSITORY_ROOT / "research" / "runs" / f"dspy-gepa-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _sha(value: str | bytes) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(encoded).hexdigest()


def _profile() -> DspyGepaProfile:
    return DspyGepaProfile(
        dspy_version="3.0.0",
        provider_id="offline-test-runner",
        model_id="fixture-model-revision-1",
        metric_version="fixture-metric-v1",
        program_schema="fixture-program-v1",
        seed=17,
        max_optimizer_steps=4,
    )


def _dataset() -> PromptDatasetBundle:
    return PromptDatasetBundle(
        dataset_id="fixture-dspy-gepa-v1",
        optimizer_train=(
            PromptRecord(
                record_id="optimizer-1",
                input_text="Summarize this evidence.",
                expected_text="A grounded summary.",
                feedback="Cite the supplied evidence.",
            ),
        ),
        selection_evaluation=(
            PromptRecord(
                record_id="selection-1",
                input_text="Explain the result.",
                expected_text="A precise explanation.",
            ),
        ),
    )


def _backend(parent: Path, calls: dict[str, object]) -> DspyGepaPromptBackend:
    def runner(
        parent_program: str,
        records: tuple[PromptRecord, ...],
        profile: DspyGepaProfile,
    ) -> DspyGepaRun:
        calls["runner_parent"] = parent_program
        calls["runner_ids"] = tuple(record.record_id for record in records)
        calls["runner_profile"] = profile
        return DspyGepaRun(
            program='{"instruction":"cite supplied evidence"}',
            trace_digest=_sha("optimizer-trace"),
            steps=2,
            samples=3,
            tokens=24,
        )

    def evaluator(
        program: str,
        records: tuple[PromptRecord, ...],
        profile: DspyGepaProfile,
    ) -> PromptSelection:
        calls["evaluator_program"] = program
        calls["evaluator_ids"] = tuple(record.record_id for record in records)
        calls["evaluator_profile"] = profile
        return PromptSelection(
            score=0.75,
            viable=True,
            mutation_verified=True,
            remaining_failure_ids=(),
            feedback_digest=_sha("selection-feedback"),
        )

    return DspyGepaPromptBackend(
        parent_program=parent,
        dataset=_dataset(),
        profile=_profile(),
        runner=runner,
        evaluator=evaluator,
    )


def _direct_request(backend: DspyGepaPromptBackend) -> CandidateWorkRequest:
    return CandidateWorkRequest(
        work_id="dspy-work-1",
        candidate_id="dspy-candidate-1",
        lineage_id="dspy-lineage-1",
        kind=CandidateKind.PROMPT,
        iteration=1,
        parent_ids=("dspy-parent-1",),
        parent_record_digests=(_sha("parent-record"),),
        target_failure_ids=("prompt-failure-1",),
        parent_selection_digest=_sha("parent-selection"),
        failure_selection_digest=_sha("failure-selection"),
        active_checkpoint_digest=backend.active_checkpoint_digest,
        dataset_manifest_digest=backend.dataset_manifest_digest,
        trainer_manifest_digest=backend.trainer_manifest_digest,
        issued_at_ms=100,
        expires_at_ms=1_000,
        budget=CandidateWorkBudget(
            max_steps=4,
            max_samples=8,
            max_tokens=128,
            max_wall_ms=900,
            max_output_bytes=16_384,
        ),
    )


def _search(active_digest: str) -> CandidateSearch:
    failure = FailureEvidence(
        failure_id="prompt-failure-1",
        category="prompt-grounding",
        dataset_id="selection-prompt-v1",
        evidence_digest=_sha("failure-evidence"),
    )
    search = CandidateSearch((failure,))
    search.add_candidate(
        CandidateRecord(
            artifact=CandidateArtifact(
                candidate_id="dspy-root",
                lineage_id="dspy-lineage-1",
                kind=CandidateKind.PROMPT,
                artifact_digest=active_digest,
                creation_manifest_digest=_sha("root-creation"),
                parent_ids=(),
                iteration=0,
            ),
            evaluation=CandidateEvaluation(
                evaluation_manifest_digest=_sha("root-evaluation"),
                metric_digest=_sha("root-metric"),
                selection_score=0.0,
                viable=True,
                mutation_verified=True,
                failure_ids=("prompt-failure-1",),
            ),
        )
    )
    return search


def _kernel(worker: CandidateWorker) -> MoiraAgentKernel:
    kernel = MoiraAgentKernel(
        AgentPolicy(
            policy_version=1,
            allowed_actions=frozenset({"candidate_training:run"}),
        ),
        permit_secret=b"dspy-gepa-backend-test-secret" * 2,
    )
    kernel.register_tool(worker.tool())
    kernel.open_lease(
        lease_id="dspy-lease",
        task_id="dspy-task",
        scope_id="dspy-lineage-1",
        mode=CognitiveMode.ATROPOS,
        role=AgentRole.EXECUTOR,
        max_turns=1,
        max_effects=1,
        allowed_actions=("candidate_training:run",),
    )
    kernel.record_turn("dspy-lease", "Stage one bounded DSPy GEPA prompt candidate")
    return kernel


def test_dspy_gepa_contract_and_dataset_lanes_are_narrow() -> None:
    contract = dspy_gepa_backend_contract()
    assert contract["protocol"] == DSPY_GEPA_BACKEND_PROTOCOL
    assert contract["candidate_kind"] == CandidateKind.PROMPT.value
    assert contract["trains_model_weights"] is False
    assert contract["accepts_network_or_tool_config_from_request"] is False
    assert contract["is_promotion_holdout"] is False
    assert contract["can_authorize"] is False
    assert contract["can_promote"] is False

    duplicate = PromptRecord(
        record_id="selection-copy",
        input_text="Summarize this evidence.",
        expected_text="A grounded summary.",
        feedback="Cite the supplied evidence.",
    )
    with pytest.raises(ValueError, match="content must be disjoint"):
        PromptDatasetBundle(
            dataset_id="overlap",
            optimizer_train=(_dataset().optimizer_train[0],),
            selection_evaluation=(duplicate,),
        )


def test_dspy_gepa_backend_keeps_optimizer_and_selection_data_separate(
    dspy_root: Path,
) -> None:
    parent = dspy_root / "parent-program.json"
    parent.write_text('{"instruction":"be grounded"}\n', encoding="utf-8")
    calls: dict[str, object] = {}
    backend = _backend(parent, calls)
    request = _direct_request(backend)
    workspace = dspy_root / "workspace"
    workspace.mkdir()

    output = backend(request, workspace)

    assert calls["runner_parent"] == '{"instruction":"be grounded"}\n'
    assert calls["runner_ids"] == ("optimizer-1",)
    assert calls["evaluator_ids"] == ("selection-1",)
    assert calls["runner_profile"] == calls["evaluator_profile"] == _profile()
    assert {path.name for path in workspace.iterdir()} == {
        "program.json",
        "creation_manifest.json",
        "selection_evaluation.json",
        "metrics.json",
    }
    creation = json.loads((workspace / output.creation_manifest_file).read_text("utf-8"))
    evaluation = json.loads(
        (workspace / output.evaluation_manifest_file).read_text("utf-8")
    )
    assert creation["optimizer_train_manifest"]["record_ids"] == ["optimizer-1"]
    assert evaluation["selection_manifest"]["record_ids"] == ["selection-1"]
    assert evaluation["is_selection_only"] is True
    assert evaluation["is_promotion_holdout"] is False
    assert creation["can_authorize"] is False
    assert creation["can_promote"] is False


def test_dspy_gepa_backend_rejects_wrong_kind_and_parent_mutation(
    dspy_root: Path,
) -> None:
    parent = dspy_root / "parent-program.json"
    parent.write_text('{"instruction":"be grounded"}\n', encoding="utf-8")
    backend = _backend(parent, {})
    request = _direct_request(backend)
    workspace = dspy_root / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="prompt candidates only"):
        backend(replace(request, kind=CandidateKind.ADAPTER), workspace)

    parent.write_text('{"instruction":"mutated"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind the parent program"):
        backend(request, workspace)


def test_permit_bound_dspy_gepa_candidate_stays_staged(dspy_root: Path) -> None:
    parent = dspy_root / "parent-program.json"
    parent.write_text('{"instruction":"be grounded"}\n', encoding="utf-8")
    parent_before = parent.read_bytes()
    calls: dict[str, object] = {}
    backend = _backend(parent, calls)
    search = _search(backend.active_checkpoint_digest)
    parents = search.select_parents(count=1, seed=7, iteration=1)
    failures = search.select_failures("dspy-root", count=1, seed=11)
    request = prepare_candidate_work(
        search,
        parents,
        failures,
        work_id="dspy-work-1",
        candidate_id="dspy-candidate-1",
        kind=CandidateKind.PROMPT,
        active_checkpoint_digest=backend.active_checkpoint_digest,
        dataset_manifest_digest=backend.dataset_manifest_digest,
        trainer_manifest_digest=backend.trainer_manifest_digest,
        issued_at_ms=100,
        expires_at_ms=1_000,
        budget=CandidateWorkBudget(
            max_steps=4,
            max_samples=8,
            max_tokens=128,
            max_wall_ms=900,
            max_output_bytes=16_384,
        ),
    )
    worker = CandidateWorker(
        backend,
        active_checkpoint=parent,
        staging_root=dspy_root / "staging",
        clock_ms=lambda: 100,
    )
    worker.issue(
        request,
        search=search,
        parent_selection=parents,
        failure_selection=failures,
    )
    kernel = _kernel(worker)
    outcome = kernel.request_tool(
        "dspy-lease",
        tool_id=CANDIDATE_WORK_TOOL_ID,
        target=request.work_id,
        payload=request.payload,
    )
    assert outcome.authorized is not None
    execution = kernel.execute_effect(outcome.authorized, CANDIDATE_WORK_TOOL_ID)
    receipt = worker.result(request.work_id)

    assert execution.receipt.outcome == "succeeded"
    assert parent.read_bytes() == parent_before
    assert receipt.candidate.artifact.kind is CandidateKind.PROMPT
    assert receipt.can_authorize is False
    assert receipt.can_promote is False
    workspace = dspy_root / "staging" / request.request_digest
    assert {path.name for path in workspace.iterdir()} == {
        "program.json",
        "creation_manifest.json",
        "selection_evaluation.json",
        "metrics.json",
    }
