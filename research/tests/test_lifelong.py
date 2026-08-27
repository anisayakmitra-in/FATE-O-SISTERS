from __future__ import annotations

import hashlib

import pytest

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


def _contracts() -> tuple[object, ...]:
    return (
        PerspectiveRegisterV1(
            register_id="perspective-clotho",
            owner_id="owner-local",
            scope_id="project-local",
            sister=Sister.CLOTHO,
            confidence_state=0.75,
            unresolved_question_refs=("question-1",),
            active_goal_refs=("goal-1",),
            dissent_refs=("dissent-1",),
            recent_prediction_refs=("prediction-1",),
            competence_estimates=(("code", 0.8),),
            drive_contributions=(("curiosity", 0.25),),
            affect_calibration=(("risk", -0.1),),
            last_consolidation_receipt=_sha("consolidation"),
            updated_at_ms=10,
        ),
        ConsentScopeV1(
            consent_id="consent-1",
            owner_id="owner-local",
            scope_id="project-local",
            allowed_event_kinds=(InteractionKind.EDIT,),
            allowed_uses=(ConsentUse.MEMORY, ConsentUse.PERSONAL_TRAINING),
            granted_at_ms=1,
            expires_at_ms=100,
            capture_allowed=True,
        ),
        ArtifactEnvelopeV1(
            artifact_id="artifact-1",
            owner_id="owner-local",
            scope_id="project-local",
            consent_id="consent-1",
            interaction_kind=InteractionKind.EDIT,
            media_type="text/plain",
            content_digest=_sha("artifact"),
            byte_count=8,
            provenance_refs=("editor-local",),
            created_at_ms=10,
            expires_at_ms=90,
            encrypted=True,
        ),
        InteractionEventV1(
            event_id="event-1",
            session_id="session-1",
            task_id="task-1",
            scope_id="project-local",
            perspective_id="perspective-clotho",
            consent_id="consent-1",
            kind=InteractionKind.EDIT,
            trust=EventTrust.USER_SUPPLIED,
            occurred_at_ms=10,
            payload_digest=_sha("artifact"),
            artifact_id="artifact-1",
            training_eligible=True,
        ),
        EvidenceReceiptV1(
            evidence_id="evidence-1",
            scope_id="project-local",
            verifier_id="pytest",
            verifier_version="8",
            event_ids=("event-1",),
            outcome=EvidenceOutcome.PASS,
            strength=EvidenceStrength.STRONG,
            artifact_digest=_sha("test-result"),
            issued_at_ms=20,
            independent=True,
        ),
        LearningRecordV1(
            learning_id="learning-1",
            scope_id="project-local",
            event_ids=("event-1",),
            evidence_ids=("evidence-1",),
            destination=LearningDestination.PERSONAL_ADAPTER,
            provenance_digest=_sha("provenance"),
            created_at_ms=30,
            training_eligible=True,
        ),
        MemoryRecordV1(
            memory_id="memory-1",
            scope_id="project-local",
            layer=MemoryLayer.EPISODIC,
            content_digest=_sha("memory"),
            provenance_learning_ids=("learning-1",),
            parent_memory_ids=(),
            created_at_ms=40,
        ),
        GraphProjectionV1(
            projection_id="projection-1",
            scope_id="project-local",
            relation="supports",
            source_memory_ids=("memory-1",),
            parent_projection_ids=(),
            projection_digest=_sha("projection"),
            created_at_ms=50,
        ),
        StandingGoalV1(
            goal_id="goal-1",
            scope_id="project-local",
            perspective_id="perspective-clotho",
            statement="Consolidate verified local coding evidence.",
            status=GoalStatus.ACTIVE,
            allowed_actions=("workspace:read",),
            max_turns_per_wake=4,
            max_effects_per_wake=1,
            max_compute_ms_per_wake=1_000,
            max_proactive_messages_per_wake=1,
            created_at_ms=1,
            next_wake_at_ms=60,
        ),
        QuestionCandidateV1(
            question_id="question-1",
            scope_id="project-local",
            perspective_register_id="perspective-clotho",
            goal_id="goal-1",
            prompt_digest=_sha("question"),
            evidence_refs=("evidence-1",),
            priority=70,
            state=QuestionState.CANDIDATE,
            created_at_ms=50,
            expires_at_ms=90,
        ),
        WakeRequestV1(
            request_id="wake-request-1",
            goal_id="goal-1",
            scope_id="project-local",
            requested_by="offline-scheduler",
            reason="scheduled consolidation",
            requested_at_ms=60,
            expires_at_ms=80,
            max_turns=4,
            max_effects=1,
            max_compute_ms=1_000,
            max_proactive_messages=1,
            allowed_actions=("workspace:read",),
        ),
        WakeLeaseV1(
            lease_id="wake-lease-1",
            request_id="wake-request-1",
            goal_id="goal-1",
            scope_id="project-local",
            issued_by="owner-policy",
            issued_at_ms=61,
            expires_at_ms=79,
            max_turns=4,
            max_effects=1,
            max_compute_ms=1_000,
            max_proactive_messages=1,
            allowed_actions=("workspace:read",),
        ),
        CognitionSnapshotV1(
            snapshot_id="snapshot-1",
            scope_id="project-local",
            perspective_register_ids=("perspective-clotho",),
            goal_ids=("goal-1",),
            unresolved_question_refs=("question-1",),
            commitment_refs=("commitment-1",),
            drive_state=(("curiosity", 0.25),),
            memory_ids=("memory-1",),
            graph_projection_ids=("projection-1",),
            candidate_queue_refs=("candidate-1",),
            wake_lease_id="wake-lease-1",
            context_digest=_sha("context"),
            trace_chain_head=_sha("trace"),
            ledger_head=_sha("ledger"),
            last_verified_checkpoint=_sha("checkpoint"),
            created_at_ms=70,
        ),
        ProactiveMessageV1(
            message_id="message-1",
            goal_id="goal-1",
            wake_lease_id="wake-lease-1",
            scope_id="project-local",
            content_digest=_sha("message"),
            state=MessageState.STAGED,
            created_at_ms=70,
            requires_user_attention=True,
        ),
    )


@pytest.mark.parametrize("contract", _contracts())
def test_lifelong_contract_mapping_round_trips_exactly(contract: object) -> None:
    restored = type(contract).from_mapping(contract.mapping())

    assert restored == contract
    assert restored.digest == contract.digest
    assert len(contract.digest) == 64


def test_perspective_register_is_one_sister_responsibility_view() -> None:
    register = _contracts()[0]

    assert register.sister is Sister.CLOTHO
    assert register.competence_estimates == (("code", 0.8),)
    assert not hasattr(register, "perspective_ids")


def test_click_cannot_claim_training_eligibility() -> None:
    with pytest.raises(ValueError, match="click events"):
        InteractionEventV1(
            event_id="click-1",
            session_id="session-1",
            task_id="task-1",
            scope_id="project-local",
            perspective_id="perspective-clotho",
            consent_id="consent-1",
            kind=InteractionKind.CLICK,
            trust=EventTrust.USER_SUPPLIED,
            occurred_at_ms=10,
            payload_digest=_sha("click"),
            training_eligible=True,
        )


def test_capture_consent_is_disabled_by_default() -> None:
    consent = ConsentScopeV1(
        consent_id="consent-default",
        owner_id="owner-local",
        scope_id="project-local",
        allowed_event_kinds=(InteractionKind.USER_TEXT,),
        allowed_uses=(ConsentUse.MEMORY,),
        granted_at_ms=1,
        expires_at_ms=10,
    )

    assert consent.capture_allowed is False
    assert consent.allows_event(InteractionKind.USER_TEXT, 2) is False
