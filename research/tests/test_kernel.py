from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Iterator
import uuid

import pytest

from moira.kernel import (
    KERNEL_MODULE_NAMES,
    MoiraKernel,
    ReceiptDomain,
    ReceiptEntryV1,
)
from moira.lifelong import (
    ArtifactEnvelopeV1,
    CognitionSnapshotV1,
    ConsentScopeV1,
    ConsentUse,
    EventTrust,
    EvidenceOutcome,
    EvidenceReceiptV1,
    EvidenceStrength,
    GoalStatus,
    GraphProjectionV1,
    InteractionEventV1,
    InteractionKind,
    LearningDestination,
    LearningRecordV1,
    MemoryLayer,
    MemoryRecordV1,
    MessageState,
    PerspectiveRegisterV1,
    ProactiveMessageV1,
    QuestionCandidateV1,
    QuestionState,
    Sister,
    StandingGoalV1,
    WakeLeaseV1,
    WakeRequestV1,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture
def kernel_path() -> Iterator[Path]:
    path = Path("research") / f".kernel-state-{uuid.uuid4().hex}.json"
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _register(
    *,
    questions: tuple[str, ...] = (),
    goals: tuple[str, ...] = (),
    receipt: str = "genesis",
    updated_at_ms: int = 1,
) -> PerspectiveRegisterV1:
    return PerspectiveRegisterV1(
        register_id="perspective-clotho",
        owner_id="owner-local",
        scope_id="project-local",
        sister=Sister.CLOTHO,
        confidence_state=0.5,
        unresolved_question_refs=questions,
        active_goal_refs=goals,
        dissent_refs=(),
        recent_prediction_refs=(),
        competence_estimates=(("code", 0.5),),
        drive_contributions=(("curiosity", 0.25),),
        affect_calibration=(("risk", 0.0),),
        last_consolidation_receipt=receipt,
        updated_at_ms=updated_at_ms,
    )


def _consent() -> ConsentScopeV1:
    return ConsentScopeV1(
        consent_id="consent-1",
        owner_id="owner-local",
        scope_id="project-local",
        allowed_event_kinds=(InteractionKind.CLICK, InteractionKind.EDIT),
        allowed_uses=(ConsentUse.MEMORY, ConsentUse.PERSONAL_TRAINING),
        granted_at_ms=1,
        expires_at_ms=1_000,
        capture_allowed=True,
    )


def _event(
    event_id: str,
    kind: InteractionKind,
    *,
    payload: str,
    artifact_id: str | None = None,
    training_eligible: bool = False,
) -> InteractionEventV1:
    return InteractionEventV1(
        event_id=event_id,
        session_id="session-1",
        task_id="task-1",
        scope_id="project-local",
        perspective_id="perspective-clotho",
        consent_id="consent-1",
        kind=kind,
        trust=EventTrust.USER_SUPPLIED,
        occurred_at_ms=10,
        payload_digest=_sha(payload),
        artifact_id=artifact_id,
        training_eligible=training_eligible,
    )


def _evidence(evidence_id: str, event_id: str, *, issued_at_ms: int = 20) -> EvidenceReceiptV1:
    return EvidenceReceiptV1(
        evidence_id=evidence_id,
        scope_id="project-local",
        verifier_id="pytest",
        verifier_version="8",
        event_ids=(event_id,),
        outcome=EvidenceOutcome.PASS,
        strength=EvidenceStrength.STRONG,
        artifact_digest=_sha(f"result:{event_id}"),
        issued_at_ms=issued_at_ms,
        independent=True,
    )


def test_kernel_has_exactly_five_closed_non_trainable_identities(
    kernel_path: Path,
) -> None:
    kernel = MoiraKernel(kernel_path)

    assert tuple(item.name for item in kernel.module_identities) == KERNEL_MODULE_NAMES
    assert len(kernel.module_identities) == 5
    assert all(item.trainable_parameters == 0 for item in kernel.module_identities)
    assert all(item.trainable is False for item in kernel.module_identities)
    assert all(item.model_mutable is False for item in kernel.module_identities)
    assert kernel.capture_enabled is False

    invalid = {
        "index": 0,
        "domain": "arbitrary-domain",
        "record_id": "record-1",
        "record_digest": _sha("record"),
        "previous_digest": "genesis",
        "receipt_digest": _sha("receipt"),
    }
    with pytest.raises(ValueError, match="not a valid ReceiptDomain"):
        ReceiptEntryV1.from_mapping(invalid)
    with pytest.raises(ValueError, match="invalid schema"):
        ReceiptEntryV1.from_mapping({**invalid, "permit_id": "forbidden"})


def test_perspective_updates_fail_closed_without_typed_consolidation_receipts(
    kernel_path: Path,
) -> None:
    kernel = MoiraKernel(kernel_path)
    kernel.configure_capture(True)

    with pytest.raises(ValueError, match="not implemented"):
        kernel.register_perspectives(_register(receipt=kernel.receipt_head))


def test_capture_is_explicit_and_clicks_cannot_enter_training(
    kernel_path: Path,
) -> None:
    path = kernel_path
    kernel = MoiraKernel(path)
    kernel.register_perspectives(_register())
    kernel.grant_consent(_consent())
    click = _event("click-1", InteractionKind.CLICK, payload="click")

    with pytest.raises(ValueError, match="capture is disabled"):
        kernel.capture_interaction(click)

    kernel.configure_capture(True)
    kernel.capture_interaction(click)
    kernel.record_evidence(_evidence("click-evidence", "click-1"))
    click_memory_provenance = kernel.learning_provenance_digest(("click-1",), ())
    kernel.record_learning(
        LearningRecordV1(
            learning_id="click-memory",
            scope_id="project-local",
            event_ids=("click-1",),
            evidence_ids=(),
            destination=LearningDestination.MEMORY_ONLY,
            provenance_digest=click_memory_provenance,
            created_at_ms=11,
            training_eligible=False,
        )
    )
    click_provenance = kernel.learning_provenance_digest(
        ("click-1",), ("click-evidence",)
    )
    with pytest.raises(ValueError, match="weak interactions"):
        kernel.record_learning(
            LearningRecordV1(
                learning_id="click-learning",
                scope_id="project-local",
                event_ids=("click-1",),
                evidence_ids=("click-evidence",),
                destination=LearningDestination.PERSONAL_ADAPTER,
                provenance_digest=click_provenance,
                created_at_ms=30,
                training_eligible=True,
            )
        )

    artifact = ArtifactEnvelopeV1(
        artifact_id="edit-artifact",
        owner_id="owner-local",
        scope_id="project-local",
        consent_id="consent-1",
        interaction_kind=InteractionKind.EDIT,
        media_type="text/plain",
        content_digest=_sha("edit"),
        byte_count=4,
        provenance_refs=("editor-local",),
        created_at_ms=9,
        expires_at_ms=900,
        encrypted=True,
    )
    kernel.store_artifact(artifact)
    edit = _event(
        "edit-1",
        InteractionKind.EDIT,
        payload="edit",
        artifact_id="edit-artifact",
        training_eligible=True,
    )
    kernel.capture_interaction(edit)
    kernel.record_evidence(_evidence("edit-evidence", "edit-1"))

    mixed_provenance = kernel.learning_provenance_digest(
        ("click-1", "edit-1"), ("click-evidence", "edit-evidence")
    )
    with pytest.raises(ValueError, match="weak interactions"):
        kernel.record_learning(
            LearningRecordV1(
                learning_id="mixed-learning",
                scope_id="project-local",
                event_ids=("click-1", "edit-1"),
                evidence_ids=("click-evidence", "edit-evidence"),
                destination=LearningDestination.PERSONAL_ADAPTER,
                provenance_digest=mixed_provenance,
                created_at_ms=30,
                training_eligible=True,
            )
        )

    unverified_edit = _event(
        "edit-unverified",
        InteractionKind.EDIT,
        payload="edit-unverified",
        training_eligible=True,
    )
    kernel.capture_interaction(unverified_edit)
    partially_verified_provenance = kernel.learning_provenance_digest(
        ("edit-1", "edit-unverified"), ("edit-evidence",)
    )
    with pytest.raises(ValueError, match="each training-eligible interaction"):
        kernel.record_learning(
            LearningRecordV1(
                learning_id="unverified-learning",
                scope_id="project-local",
                event_ids=("edit-1", "edit-unverified"),
                evidence_ids=("edit-evidence",),
                destination=LearningDestination.PERSONAL_ADAPTER,
                provenance_digest=partially_verified_provenance,
                created_at_ms=30,
                training_eligible=True,
            )
        )

    provenance = kernel.learning_provenance_digest(
        ("edit-1",), ("edit-evidence",)
    )
    with pytest.raises(ValueError, match="cannot precede"):
        kernel.record_learning(
            LearningRecordV1(
                learning_id="backdated-learning",
                scope_id="project-local",
                event_ids=("edit-1",),
                evidence_ids=("edit-evidence",),
                destination=LearningDestination.PERSONAL_ADAPTER,
                provenance_digest=provenance,
                created_at_ms=19,
                training_eligible=True,
            )
        )
    kernel.record_learning(
        LearningRecordV1(
            learning_id="edit-learning",
            scope_id="project-local",
            event_ids=("edit-1",),
            evidence_ids=("edit-evidence",),
            destination=LearningDestination.PERSONAL_ADAPTER,
            provenance_digest=provenance,
            created_at_ms=30,
            training_eligible=True,
        )
    )

    before = path.read_bytes()
    digest = kernel.state_digest
    reopened = MoiraKernel(path)

    assert reopened.capture_enabled is True
    assert reopened.artifacts == (artifact,)
    assert reopened.state_digest == digest
    assert path.read_bytes() == before


def test_capture_policy_rolls_back_if_persistence_fails(
    kernel_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel = MoiraKernel(kernel_path)
    before = kernel_path.read_bytes()

    def fail_persist() -> None:
        raise OSError("simulated persistence failure")

    monkeypatch.setattr(kernel, "_persist", fail_persist)
    with pytest.raises(OSError, match="simulated persistence failure"):
        kernel.configure_capture(True)

    assert kernel.capture_enabled is False
    assert kernel.revision == 0
    assert kernel.receipt_head == "genesis"
    assert kernel_path.read_bytes() == before


def test_state_rejects_receipts_without_an_implemented_typed_record(
    kernel_path: Path,
) -> None:
    MoiraKernel(kernel_path)
    state = json.loads(kernel_path.read_text(encoding="utf-8"))
    state["receipts"] = [
        ReceiptEntryV1.create(
            index=0,
            domain=ReceiptDomain.ACTIVATION,
            record_id="activation-1",
            record_mapping={"candidate": "candidate-1"},
            previous_digest="genesis",
        ).mapping()
    ]
    kernel_path.write_text(
        json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="domain is not implemented"):
        MoiraKernel(kernel_path)


def test_memory_deletion_tombstones_memory_and_graph_descendants(
    kernel_path: Path,
) -> None:
    path = kernel_path
    kernel = MoiraKernel(path)
    root = MemoryRecordV1(
        memory_id="memory-root",
        scope_id="project-local",
        layer=MemoryLayer.SESSION,
        content_digest=_sha("root"),
        provenance_learning_ids=(),
        parent_memory_ids=(),
        created_at_ms=1,
    )
    child = MemoryRecordV1(
        memory_id="memory-child",
        scope_id="project-local",
        layer=MemoryLayer.EPISODIC,
        content_digest=_sha("child"),
        provenance_learning_ids=(),
        parent_memory_ids=("memory-root",),
        created_at_ms=2,
    )
    grandchild = MemoryRecordV1(
        memory_id="memory-grandchild",
        scope_id="project-local",
        layer=MemoryLayer.SEMANTIC,
        content_digest=_sha("grandchild"),
        provenance_learning_ids=(),
        parent_memory_ids=("memory-child",),
        created_at_ms=3,
    )
    for memory in (root, child, grandchild):
        kernel.store_memory(memory)

    first = GraphProjectionV1(
        projection_id="projection-root",
        scope_id="project-local",
        relation="derived-from",
        source_memory_ids=("memory-root",),
        parent_projection_ids=(),
        projection_digest=kernel.graph_projection_digest(
            "derived-from", ("memory-root",), ()
        ),
        created_at_ms=4,
    )
    kernel.project_graph(first)
    second = GraphProjectionV1(
        projection_id="projection-child",
        scope_id="project-local",
        relation="refines",
        source_memory_ids=("memory-child",),
        parent_projection_ids=("projection-root",),
        projection_digest=kernel.graph_projection_digest(
            "refines", ("memory-child",), ("projection-root",)
        ),
        created_at_ms=5,
    )
    kernel.project_graph(second)
    third = GraphProjectionV1(
        projection_id="projection-grandchild",
        scope_id="project-local",
        relation="summarizes",
        source_memory_ids=(),
        parent_projection_ids=("projection-child",),
        projection_digest=kernel.graph_projection_digest(
            "summarizes", (), ("projection-child",)
        ),
        created_at_ms=6,
    )
    kernel.project_graph(third)

    deletion = kernel.tombstone_memory("memory-root", now_ms=10)

    assert deletion["memory_ids"] == (
        "memory-child",
        "memory-grandchild",
        "memory-root",
    )
    assert deletion["projection_ids"] == (
        "projection-child",
        "projection-grandchild",
        "projection-root",
    )
    assert all(not memory.live for memory in kernel.memories)
    assert all(not projection.live for projection in kernel.graph_projections)
    reopened = MoiraKernel(path)
    assert reopened.memories == kernel.memories
    assert reopened.graph_projections == kernel.graph_projections


def test_memory_tombstone_rolls_back_if_persistence_fails(
    kernel_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kernel = MoiraKernel(kernel_path)
    kernel.store_memory(
        MemoryRecordV1(
            memory_id="memory-root",
            scope_id="project-local",
            layer=MemoryLayer.SESSION,
            content_digest=_sha("root"),
            provenance_learning_ids=(),
            parent_memory_ids=(),
            created_at_ms=1,
        )
    )
    before = kernel_path.read_bytes()
    revision = kernel.revision
    receipt_head = kernel.receipt_head

    def fail_persist() -> None:
        raise OSError("simulated persistence failure")

    monkeypatch.setattr(kernel, "_persist", fail_persist)
    with pytest.raises(OSError, match="simulated persistence failure"):
        kernel.tombstone_memory("memory-root", now_ms=2)

    assert kernel.memories[0].live is True
    assert kernel.revision == revision
    assert kernel.receipt_head == receipt_head
    assert kernel_path.read_bytes() == before


def test_wake_leases_expire_and_persistent_cognition_stays_bounded(
    kernel_path: Path,
) -> None:
    path = kernel_path
    kernel = MoiraKernel(path)
    kernel.register_perspectives(_register())
    goal = StandingGoalV1(
        goal_id="goal-1",
        scope_id="project-local",
        perspective_id="perspective-clotho",
        statement="Review verified local evidence.",
        status=GoalStatus.ACTIVE,
        allowed_actions=("workspace:read",),
        max_turns_per_wake=4,
        max_effects_per_wake=1,
        max_compute_ms_per_wake=1_000,
        max_proactive_messages_per_wake=1,
        created_at_ms=2,
    )
    kernel.add_standing_goal(goal)
    question = QuestionCandidateV1(
        question_id="question-1",
        scope_id="project-local",
        perspective_register_id="perspective-clotho",
        goal_id="goal-1",
        prompt_digest=_sha("question"),
        evidence_refs=(),
        priority=60,
        state=QuestionState.CANDIDATE,
        created_at_ms=3,
        expires_at_ms=100,
    )
    kernel.add_question_candidate(question)
    kernel.register_perspectives(
        _register(
            questions=("question-1",),
            goals=("goal-1",),
            updated_at_ms=4,
        )
    )

    expired_request = WakeRequestV1(
        request_id="request-expired",
        goal_id="goal-1",
        scope_id="project-local",
        requested_by="scheduler",
        reason="expired test",
        requested_at_ms=10,
        expires_at_ms=20,
        max_turns=4,
        max_effects=1,
        max_compute_ms=1_000,
        max_proactive_messages=1,
        allowed_actions=("workspace:read",),
    )
    kernel.request_wake(expired_request)
    with pytest.raises(ValueError, match="expired"):
        kernel.issue_wake_lease(
            WakeLeaseV1(
                lease_id="lease-expired",
                request_id="request-expired",
                goal_id="goal-1",
                scope_id="project-local",
                issued_by="owner-policy",
                issued_at_ms=20,
                expires_at_ms=21,
                max_turns=4,
                max_effects=1,
                max_compute_ms=1_000,
                max_proactive_messages=1,
                allowed_actions=("workspace:read",),
            ),
            now_ms=20,
        )

    request = WakeRequestV1(
        request_id="request-1",
        goal_id="goal-1",
        scope_id="project-local",
        requested_by="scheduler",
        reason="scheduled review",
        requested_at_ms=30,
        expires_at_ms=60,
        max_turns=4,
        max_effects=1,
        max_compute_ms=1_000,
        max_proactive_messages=1,
        allowed_actions=("workspace:read",),
    )
    kernel.request_wake(request)
    lease = WakeLeaseV1(
        lease_id="lease-1",
        request_id="request-1",
        goal_id="goal-1",
        scope_id="project-local",
        issued_by="owner-policy",
        issued_at_ms=31,
        expires_at_ms=50,
        max_turns=4,
        max_effects=1,
        max_compute_ms=1_000,
        max_proactive_messages=1,
        allowed_actions=("workspace:read",),
    )
    kernel.issue_wake_lease(lease, now_ms=31)
    snapshot = CognitionSnapshotV1(
        snapshot_id="snapshot-1",
        scope_id="project-local",
        perspective_register_ids=("perspective-clotho",),
        goal_ids=("goal-1",),
        unresolved_question_refs=("question-1",),
        commitment_refs=(),
        drive_state=(("curiosity", 0.25),),
        memory_ids=(),
        graph_projection_ids=(),
        candidate_queue_refs=(),
        wake_lease_id="lease-1",
        context_digest=_sha("context"),
        trace_chain_head=_sha("trace"),
        ledger_head=kernel.receipt_head,
        last_verified_checkpoint=_sha("checkpoint"),
        created_at_ms=40,
    )
    kernel.save_cognition_snapshot(snapshot)
    message = ProactiveMessageV1(
        message_id="message-1",
        goal_id="goal-1",
        wake_lease_id="lease-1",
        scope_id="project-local",
        content_digest=_sha("message"),
        state=MessageState.STAGED,
        created_at_ms=41,
        requires_user_attention=True,
    )
    kernel.record_proactive_message(message)
    with pytest.raises(ValueError, match="wake lease budget"):
        kernel.record_proactive_message(
            ProactiveMessageV1(
                message_id="message-2",
                goal_id="goal-1",
                wake_lease_id="lease-1",
                scope_id="project-local",
                content_digest=_sha("message-2"),
                state=MessageState.STAGED,
                created_at_ms=42,
                requires_user_attention=False,
            )
        )

    with pytest.raises(ValueError, match="expired"):
        kernel.require_active_lease("lease-1", now_ms=50)

    reopened = MoiraKernel(path)
    assert reopened.question_candidates == (question,)
    prefixes = {receipt.domain.value.split(".", 1)[0] for receipt in reopened.receipts}
    assert prefixes == {
        "EffectsKernel",
        "EvidenceKernel",
        "RunKernel",
    }
    assert ReceiptDomain.EFFECT_OUTCOME.value.endswith("effect-outcome-receipt.v1")
    assert all("permit" not in receipt.mapping() for receipt in reopened.receipts)
