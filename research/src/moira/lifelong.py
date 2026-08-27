"""Typed, non-neural contracts for MOIRA's offline lifelong control plane."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields as dataclass_fields, replace
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, ClassVar, Mapping, Self


LIFELONG_CONTRACT_VERSION = 1
MAX_CONTRACT_TEXT_BYTES = 64 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTION = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_.-]+$")


class InteractionKind(str, Enum):
    USER_TEXT = "user_text"
    MODEL_TEXT = "model_text"
    ACCEPT = "accept"
    REJECT = "reject"
    EDIT = "edit"
    REVERT = "revert"
    CLICK = "click"
    SEARCH_QUERY = "search_query"
    SEARCH_CHOICE = "search_choice"
    TOOL_RESULT = "tool_result"
    VISION_FRAME = "vision_frame"
    AUDIO_SEGMENT = "audio_segment"


class Sister(str, Enum):
    CLOTHO = "clotho"
    LACHESIS = "lachesis"
    ATROPOS = "atropos"


class EventTrust(str, Enum):
    TRUSTED_RUNTIME = "trusted_runtime"
    USER_SUPPLIED = "user_supplied"
    UNTRUSTED = "untrusted"


class ConsentUse(str, Enum):
    MEMORY = "memory"
    PERSONAL_TRAINING = "personal_training"
    PROJECT_TRAINING = "project_training"
    EXPORT = "export"


class EvidenceOutcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INDETERMINATE = "indeterminate"


class EvidenceStrength(str, Enum):
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"


class LearningDestination(str, Enum):
    MEMORY_ONLY = "memory_only"
    PERSONAL_ADAPTER = "personal_adapter"
    PROJECT_ADAPTER = "project_adapter"


class MemoryLayer(str, Enum):
    SESSION = "session"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class MessageState(str, Enum):
    STAGED = "staged"
    DELIVERED = "delivered"
    DISMISSED = "dismissed"


class QuestionState(str, Enum):
    CANDIDATE = "candidate"
    ASKED = "asked"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


@dataclass(frozen=True, slots=True)
class _ContractV1:
    CONTRACT: ClassVar[str]

    def mapping(self) -> dict[str, object]:
        return {"contract": self.CONTRACT, **asdict(self)}

    @property
    def digest(self) -> str:
        return domain_digest(f"moira.contract.{self.CONTRACT}", self.mapping())


@dataclass(frozen=True, slots=True)
class PerspectiveRegisterV1(_ContractV1):
    CONTRACT: ClassVar[str] = "PerspectiveRegisterV1"

    register_id: str
    owner_id: str
    scope_id: str
    sister: Sister
    confidence_state: float
    unresolved_question_refs: tuple[str, ...]
    active_goal_refs: tuple[str, ...]
    dissent_refs: tuple[str, ...]
    recent_prediction_refs: tuple[str, ...]
    competence_estimates: tuple[tuple[str, float], ...]
    drive_contributions: tuple[tuple[str, float], ...]
    affect_calibration: tuple[tuple[str, float], ...]
    last_consolidation_receipt: str
    updated_at_ms: int

    def __post_init__(self) -> None:
        _text("perspective register ID", self.register_id, 256)
        _text("perspective owner ID", self.owner_id, 256)
        _text("perspective scope ID", self.scope_id, 256)
        if not isinstance(self.sister, Sister):
            raise ValueError("perspective sister is invalid")
        _bounded_float("perspective confidence", self.confidence_state, 0.0, 1.0)
        _unique_text_tuple(
            "unresolved question refs",
            self.unresolved_question_refs,
            256,
            allow_empty=True,
        )
        _unique_text_tuple(
            "active goal refs", self.active_goal_refs, 128, allow_empty=True
        )
        _unique_text_tuple("dissent refs", self.dissent_refs, 256, allow_empty=True)
        _unique_text_tuple(
            "recent prediction refs",
            self.recent_prediction_refs,
            256,
            allow_empty=True,
        )
        _bounded_score_pairs(
            "competence estimates", self.competence_estimates, 0.0, 1.0
        )
        _bounded_score_pairs(
            "drive contributions", self.drive_contributions, -1.0, 1.0
        )
        _bounded_score_pairs(
            "affect calibration", self.affect_calibration, -1.0, 1.0
        )
        if self.last_consolidation_receipt != "genesis":
            _digest(
                "last consolidation receipt", self.last_consolidation_receipt
            )
        _time("perspective update time", self.updated_at_ms)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        fields = _payload(value, cls.CONTRACT, cls)
        return cls(
            register_id=fields["register_id"],
            owner_id=fields["owner_id"],
            scope_id=fields["scope_id"],
            sister=Sister(fields["sister"]),
            confidence_state=fields["confidence_state"],
            unresolved_question_refs=_strings(fields["unresolved_question_refs"]),
            active_goal_refs=_strings(fields["active_goal_refs"]),
            dissent_refs=_strings(fields["dissent_refs"]),
            recent_prediction_refs=_strings(fields["recent_prediction_refs"]),
            competence_estimates=_score_pairs(fields["competence_estimates"]),
            drive_contributions=_score_pairs(fields["drive_contributions"]),
            affect_calibration=_score_pairs(fields["affect_calibration"]),
            last_consolidation_receipt=fields["last_consolidation_receipt"],
            updated_at_ms=fields["updated_at_ms"],
        )


@dataclass(frozen=True, slots=True)
class ConsentScopeV1(_ContractV1):
    CONTRACT: ClassVar[str] = "ConsentScopeV1"

    consent_id: str
    owner_id: str
    scope_id: str
    allowed_event_kinds: tuple[InteractionKind, ...]
    allowed_uses: tuple[ConsentUse, ...]
    granted_at_ms: int
    expires_at_ms: int
    capture_allowed: bool = False
    revoked_at_ms: int | None = None

    def __post_init__(self) -> None:
        _text("consent ID", self.consent_id, 256)
        _text("consent owner ID", self.owner_id, 256)
        _text("consent scope ID", self.scope_id, 256)
        _unique_enum_tuple(
            "allowed event kinds", self.allowed_event_kinds, InteractionKind, 32
        )
        _unique_enum_tuple("allowed consent uses", self.allowed_uses, ConsentUse, 8)
        _time("consent grant time", self.granted_at_ms)
        _time("consent expiry time", self.expires_at_ms)
        if self.expires_at_ms <= self.granted_at_ms:
            raise ValueError("consent expiry must follow its grant")
        if type(self.capture_allowed) is not bool:
            raise ValueError("capture_allowed must be boolean")
        if self.revoked_at_ms is not None:
            _time("consent revocation time", self.revoked_at_ms)
            if self.revoked_at_ms < self.granted_at_ms:
                raise ValueError("consent revocation cannot precede its grant")

    def is_active(self, now_ms: int) -> bool:
        _time("consent check time", now_ms)
        return (
            self.granted_at_ms <= now_ms < self.expires_at_ms
            and (self.revoked_at_ms is None or now_ms < self.revoked_at_ms)
        )

    def allows_event(self, kind: InteractionKind, now_ms: int) -> bool:
        return self.capture_allowed and self.is_active(now_ms) and kind in self.allowed_event_kinds

    def allows_use(self, use: ConsentUse, now_ms: int) -> bool:
        return self.is_active(now_ms) and use in self.allowed_uses

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        fields = _payload(value, cls.CONTRACT, cls)
        return cls(
            consent_id=fields["consent_id"],
            owner_id=fields["owner_id"],
            scope_id=fields["scope_id"],
            allowed_event_kinds=_enums(fields["allowed_event_kinds"], InteractionKind),
            allowed_uses=_enums(fields["allowed_uses"], ConsentUse),
            granted_at_ms=fields["granted_at_ms"],
            expires_at_ms=fields["expires_at_ms"],
            capture_allowed=fields["capture_allowed"],
            revoked_at_ms=fields["revoked_at_ms"],
        )


@dataclass(frozen=True, slots=True)
class ArtifactEnvelopeV1(_ContractV1):
    """Digest-only metadata for one consent-bound local artifact."""

    CONTRACT: ClassVar[str] = "ArtifactEnvelopeV1"

    artifact_id: str
    owner_id: str
    scope_id: str
    consent_id: str
    interaction_kind: InteractionKind
    media_type: str
    content_digest: str
    byte_count: int
    provenance_refs: tuple[str, ...]
    created_at_ms: int
    expires_at_ms: int
    encrypted: bool
    tombstoned_at_ms: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("artifact ID", self.artifact_id),
            ("artifact owner ID", self.owner_id),
            ("artifact scope ID", self.scope_id),
            ("artifact consent ID", self.consent_id),
            ("artifact media type", self.media_type),
        ):
            _text(name, value, 256)
        if not isinstance(self.interaction_kind, InteractionKind):
            raise ValueError("artifact interaction kind is invalid")
        _digest("artifact content digest", self.content_digest)
        _positive_int("artifact byte count", self.byte_count)
        _unique_text_tuple(
            "artifact provenance refs", self.provenance_refs, 256, allow_empty=True
        )
        _time("artifact creation time", self.created_at_ms)
        _time("artifact expiry time", self.expires_at_ms)
        if self.expires_at_ms <= self.created_at_ms:
            raise ValueError("artifact expiry must follow creation")
        if type(self.encrypted) is not bool:
            raise ValueError("artifact encrypted flag must be boolean")
        if self.tombstoned_at_ms is not None:
            _time("artifact tombstone time", self.tombstoned_at_ms)
            if self.tombstoned_at_ms < self.created_at_ms:
                raise ValueError("artifact tombstone cannot precede creation")

    @property
    def live(self) -> bool:
        return self.tombstoned_at_ms is None

    def tombstone(self, now_ms: int) -> Self:
        _time("artifact tombstone time", now_ms)
        if now_ms < self.created_at_ms:
            raise ValueError("artifact tombstone cannot precede creation")
        if self.tombstoned_at_ms is not None:
            return self
        return replace(self, tombstoned_at_ms=now_ms)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        fields = _payload(value, cls.CONTRACT, cls)
        return cls(
            artifact_id=fields["artifact_id"],
            owner_id=fields["owner_id"],
            scope_id=fields["scope_id"],
            consent_id=fields["consent_id"],
            interaction_kind=InteractionKind(fields["interaction_kind"]),
            media_type=fields["media_type"],
            content_digest=fields["content_digest"],
            byte_count=fields["byte_count"],
            provenance_refs=_strings(fields["provenance_refs"]),
            created_at_ms=fields["created_at_ms"],
            expires_at_ms=fields["expires_at_ms"],
            encrypted=fields["encrypted"],
            tombstoned_at_ms=fields["tombstoned_at_ms"],
        )


@dataclass(frozen=True, slots=True)
class InteractionEventV1(_ContractV1):
    CONTRACT: ClassVar[str] = "InteractionEventV1"

    event_id: str
    session_id: str
    task_id: str
    scope_id: str
    perspective_id: str
    consent_id: str
    kind: InteractionKind
    trust: EventTrust
    occurred_at_ms: int
    payload_digest: str
    artifact_id: str | None = None
    parent_event_ids: tuple[str, ...] = ()
    training_eligible: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("interaction event ID", self.event_id),
            ("interaction session ID", self.session_id),
            ("interaction task ID", self.task_id),
            ("interaction scope ID", self.scope_id),
            ("interaction perspective ID", self.perspective_id),
            ("interaction consent ID", self.consent_id),
        ):
            _text(name, value, 256)
        if not isinstance(self.kind, InteractionKind):
            raise ValueError("interaction kind is invalid")
        if not isinstance(self.trust, EventTrust):
            raise ValueError("interaction trust is invalid")
        _time("interaction occurrence time", self.occurred_at_ms)
        _digest("interaction payload digest", self.payload_digest)
        if self.artifact_id is not None:
            _text("interaction artifact ID", self.artifact_id, 256)
        _unique_text_tuple("parent event IDs", self.parent_event_ids, 64, allow_empty=True)
        if type(self.training_eligible) is not bool:
            raise ValueError("training_eligible must be boolean")
        if self.kind is InteractionKind.CLICK and self.training_eligible:
            raise ValueError("weak click events cannot be training-eligible")

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        fields = _payload(value, cls.CONTRACT, cls)
        return cls(
            event_id=fields["event_id"],
            session_id=fields["session_id"],
            task_id=fields["task_id"],
            scope_id=fields["scope_id"],
            perspective_id=fields["perspective_id"],
            consent_id=fields["consent_id"],
            kind=InteractionKind(fields["kind"]),
            trust=EventTrust(fields["trust"]),
            occurred_at_ms=fields["occurred_at_ms"],
            payload_digest=fields["payload_digest"],
            artifact_id=fields["artifact_id"],
            parent_event_ids=_strings(fields["parent_event_ids"]),
            training_eligible=fields["training_eligible"],
        )


@dataclass(frozen=True, slots=True)
class EvidenceReceiptV1(_ContractV1):
    """Caller-supplied evidence shape, not authenticated qualification authority."""

    CONTRACT: ClassVar[str] = "EvidenceReceiptV1"

    evidence_id: str
    scope_id: str
    verifier_id: str
    verifier_version: str
    event_ids: tuple[str, ...]
    outcome: EvidenceOutcome
    strength: EvidenceStrength
    artifact_digest: str
    issued_at_ms: int
    independent: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence ID", self.evidence_id),
            ("evidence scope ID", self.scope_id),
            ("evidence verifier ID", self.verifier_id),
            ("evidence verifier version", self.verifier_version),
        ):
            _text(name, value, 256)
        _unique_text_tuple("evidence event IDs", self.event_ids, 128)
        if not isinstance(self.outcome, EvidenceOutcome):
            raise ValueError("evidence outcome is invalid")
        if not isinstance(self.strength, EvidenceStrength):
            raise ValueError("evidence strength is invalid")
        _digest("evidence artifact digest", self.artifact_digest)
        _time("evidence issue time", self.issued_at_ms)
        if type(self.independent) is not bool:
            raise ValueError("evidence independent flag must be boolean")

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        fields = _payload(value, cls.CONTRACT, cls)
        return cls(
            evidence_id=fields["evidence_id"],
            scope_id=fields["scope_id"],
            verifier_id=fields["verifier_id"],
            verifier_version=fields["verifier_version"],
            event_ids=_strings(fields["event_ids"]),
            outcome=EvidenceOutcome(fields["outcome"]),
            strength=EvidenceStrength(fields["strength"]),
            artifact_digest=fields["artifact_digest"],
            issued_at_ms=fields["issued_at_ms"],
            independent=fields["independent"],
        )


@dataclass(frozen=True, slots=True)
class LearningRecordV1(_ContractV1):
    CONTRACT: ClassVar[str] = "LearningRecordV1"

    learning_id: str
    scope_id: str
    event_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    destination: LearningDestination
    provenance_digest: str
    created_at_ms: int
    training_eligible: bool = False

    def __post_init__(self) -> None:
        _text("learning record ID", self.learning_id, 256)
        _text("learning scope ID", self.scope_id, 256)
        _unique_text_tuple("learning event IDs", self.event_ids, 128)
        _unique_text_tuple(
            "learning evidence IDs", self.evidence_ids, 128, allow_empty=True
        )
        if not isinstance(self.destination, LearningDestination):
            raise ValueError("learning destination is invalid")
        _digest("learning provenance digest", self.provenance_digest)
        _time("learning creation time", self.created_at_ms)
        if type(self.training_eligible) is not bool:
            raise ValueError("training_eligible must be boolean")
        if self.training_eligible and not self.evidence_ids:
            raise ValueError("training-eligible learning requires evidence")

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        fields = _payload(value, cls.CONTRACT, cls)
        return cls(
            learning_id=fields["learning_id"],
            scope_id=fields["scope_id"],
            event_ids=_strings(fields["event_ids"]),
            evidence_ids=_strings(fields["evidence_ids"]),
            destination=LearningDestination(fields["destination"]),
            provenance_digest=fields["provenance_digest"],
            created_at_ms=fields["created_at_ms"],
            training_eligible=fields["training_eligible"],
        )


@dataclass(frozen=True, slots=True)
class MemoryRecordV1(_ContractV1):
    CONTRACT: ClassVar[str] = "MemoryRecordV1"

    memory_id: str
    scope_id: str
    layer: MemoryLayer
    content_digest: str
    provenance_learning_ids: tuple[str, ...]
    parent_memory_ids: tuple[str, ...]
    created_at_ms: int
    tombstoned_at_ms: int | None = None

    def __post_init__(self) -> None:
        _text("memory ID", self.memory_id, 256)
        _text("memory scope ID", self.scope_id, 256)
        if not isinstance(self.layer, MemoryLayer):
            raise ValueError("memory layer is invalid")
        _digest("memory content digest", self.content_digest)
        _unique_text_tuple(
            "memory provenance learning IDs",
            self.provenance_learning_ids,
            128,
            allow_empty=True,
        )
        _unique_text_tuple(
            "parent memory IDs", self.parent_memory_ids, 128, allow_empty=True
        )
        if self.memory_id in self.parent_memory_ids:
            raise ValueError("memory cannot be its own parent")
        _time("memory creation time", self.created_at_ms)
        if self.tombstoned_at_ms is not None:
            _time("memory tombstone time", self.tombstoned_at_ms)
            if self.tombstoned_at_ms < self.created_at_ms:
                raise ValueError("memory tombstone cannot precede creation")

    @property
    def live(self) -> bool:
        return self.tombstoned_at_ms is None

    def tombstone(self, now_ms: int) -> Self:
        _time("memory tombstone time", now_ms)
        if now_ms < self.created_at_ms:
            raise ValueError("memory tombstone cannot precede creation")
        if self.tombstoned_at_ms is not None:
            return self
        return replace(self, tombstoned_at_ms=now_ms)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        fields = _payload(value, cls.CONTRACT, cls)
        return cls(
            memory_id=fields["memory_id"],
            scope_id=fields["scope_id"],
            layer=MemoryLayer(fields["layer"]),
            content_digest=fields["content_digest"],
            provenance_learning_ids=_strings(fields["provenance_learning_ids"]),
            parent_memory_ids=_strings(fields["parent_memory_ids"]),
            created_at_ms=fields["created_at_ms"],
            tombstoned_at_ms=fields["tombstoned_at_ms"],
        )


@dataclass(frozen=True, slots=True)
class GraphProjectionV1(_ContractV1):
    CONTRACT: ClassVar[str] = "GraphProjectionV1"

    projection_id: str
    scope_id: str
    relation: str
    source_memory_ids: tuple[str, ...]
    parent_projection_ids: tuple[str, ...]
    projection_digest: str
    created_at_ms: int
    tombstoned_at_ms: int | None = None

    def __post_init__(self) -> None:
        _text("graph projection ID", self.projection_id, 256)
        _text("graph scope ID", self.scope_id, 256)
        _text("graph relation", self.relation, 256)
        _unique_text_tuple(
            "graph source memory IDs", self.source_memory_ids, 128, allow_empty=True
        )
        _unique_text_tuple(
            "parent graph projection IDs",
            self.parent_projection_ids,
            128,
            allow_empty=True,
        )
        if not self.source_memory_ids and not self.parent_projection_ids:
            raise ValueError("graph projection requires provenance parents")
        if self.projection_id in self.parent_projection_ids:
            raise ValueError("graph projection cannot be its own parent")
        _digest("graph projection digest", self.projection_digest)
        _time("graph projection creation time", self.created_at_ms)
        if self.tombstoned_at_ms is not None:
            _time("graph projection tombstone time", self.tombstoned_at_ms)
            if self.tombstoned_at_ms < self.created_at_ms:
                raise ValueError("graph tombstone cannot precede creation")

    @property
    def live(self) -> bool:
        return self.tombstoned_at_ms is None

    def tombstone(self, now_ms: int) -> Self:
        _time("graph projection tombstone time", now_ms)
        if now_ms < self.created_at_ms:
            raise ValueError("graph tombstone cannot precede creation")
        if self.tombstoned_at_ms is not None:
            return self
        return replace(self, tombstoned_at_ms=now_ms)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        fields = _payload(value, cls.CONTRACT, cls)
        return cls(
            projection_id=fields["projection_id"],
            scope_id=fields["scope_id"],
            relation=fields["relation"],
            source_memory_ids=_strings(fields["source_memory_ids"]),
            parent_projection_ids=_strings(fields["parent_projection_ids"]),
            projection_digest=fields["projection_digest"],
            created_at_ms=fields["created_at_ms"],
            tombstoned_at_ms=fields["tombstoned_at_ms"],
        )


@dataclass(frozen=True, slots=True)
class StandingGoalV1(_ContractV1):
    CONTRACT: ClassVar[str] = "StandingGoalV1"

    goal_id: str
    scope_id: str
    perspective_id: str
    statement: str
    status: GoalStatus
    allowed_actions: tuple[str, ...]
    max_turns_per_wake: int
    max_effects_per_wake: int
    max_compute_ms_per_wake: int
    max_proactive_messages_per_wake: int
    created_at_ms: int
    next_wake_at_ms: int | None = None

    def __post_init__(self) -> None:
        _text("standing goal ID", self.goal_id, 256)
        _text("standing goal scope ID", self.scope_id, 256)
        _text("standing goal perspective ID", self.perspective_id, 256)
        _text("standing goal statement", self.statement, MAX_CONTRACT_TEXT_BYTES)
        if not isinstance(self.status, GoalStatus):
            raise ValueError("standing goal status is invalid")
        _actions(self.allowed_actions)
        _positive_int("standing goal turn budget", self.max_turns_per_wake)
        _nonnegative_int("standing goal effect budget", self.max_effects_per_wake)
        _positive_int("standing goal compute budget", self.max_compute_ms_per_wake)
        _nonnegative_int(
            "standing goal proactive-message budget",
            self.max_proactive_messages_per_wake,
        )
        _time("standing goal creation time", self.created_at_ms)
        if self.next_wake_at_ms is not None:
            _time("standing goal next wake time", self.next_wake_at_ms)

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        fields = _payload(value, cls.CONTRACT, cls)
        return cls(
            goal_id=fields["goal_id"],
            scope_id=fields["scope_id"],
            perspective_id=fields["perspective_id"],
            statement=fields["statement"],
            status=GoalStatus(fields["status"]),
            allowed_actions=_strings(fields["allowed_actions"]),
            max_turns_per_wake=fields["max_turns_per_wake"],
            max_effects_per_wake=fields["max_effects_per_wake"],
            max_compute_ms_per_wake=fields["max_compute_ms_per_wake"],
            max_proactive_messages_per_wake=fields[
                "max_proactive_messages_per_wake"
            ],
            created_at_ms=fields["created_at_ms"],
            next_wake_at_ms=fields["next_wake_at_ms"],
        )


@dataclass(frozen=True, slots=True)
class QuestionCandidateV1(_ContractV1):
    """One bounded proactive question proposal without raw prompt content."""

    CONTRACT: ClassVar[str] = "QuestionCandidateV1"

    question_id: str
    scope_id: str
    perspective_register_id: str
    goal_id: str
    prompt_digest: str
    evidence_refs: tuple[str, ...]
    priority: int
    state: QuestionState
    created_at_ms: int
    expires_at_ms: int
    asked_at_ms: int | None = None
    resolved_at_ms: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("question candidate ID", self.question_id),
            ("question scope ID", self.scope_id),
            ("question perspective register ID", self.perspective_register_id),
            ("question goal ID", self.goal_id),
        ):
            _text(name, value, 256)
        _digest("question prompt digest", self.prompt_digest)
        _unique_text_tuple(
            "question evidence refs", self.evidence_refs, 128, allow_empty=True
        )
        if type(self.priority) is not int or not 0 <= self.priority <= 100:
            raise ValueError("question priority must be between 0 and 100")
        if not isinstance(self.state, QuestionState):
            raise ValueError("question state is invalid")
        _time("question creation time", self.created_at_ms)
        _time("question expiry time", self.expires_at_ms)
        if self.expires_at_ms <= self.created_at_ms:
            raise ValueError("question expiry must follow creation")
        for name, value in (
            ("question asked time", self.asked_at_ms),
            ("question resolved time", self.resolved_at_ms),
        ):
            if value is not None:
                _time(name, value)
                if value < self.created_at_ms:
                    raise ValueError(f"{name} cannot precede creation")
        if self.state is QuestionState.CANDIDATE and (
            self.asked_at_ms is not None or self.resolved_at_ms is not None
        ):
            raise ValueError("candidate question cannot claim an outcome time")
        if self.state is QuestionState.ASKED and self.asked_at_ms is None:
            raise ValueError("asked question requires asked_at_ms")
        if self.state in {QuestionState.RESOLVED, QuestionState.DISMISSED} and (
            self.resolved_at_ms is None
        ):
            raise ValueError("closed question requires resolved_at_ms")

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        fields = _payload(value, cls.CONTRACT, cls)
        return cls(
            question_id=fields["question_id"],
            scope_id=fields["scope_id"],
            perspective_register_id=fields["perspective_register_id"],
            goal_id=fields["goal_id"],
            prompt_digest=fields["prompt_digest"],
            evidence_refs=_strings(fields["evidence_refs"]),
            priority=fields["priority"],
            state=QuestionState(fields["state"]),
            created_at_ms=fields["created_at_ms"],
            expires_at_ms=fields["expires_at_ms"],
            asked_at_ms=fields["asked_at_ms"],
            resolved_at_ms=fields["resolved_at_ms"],
        )


@dataclass(frozen=True, slots=True)
class WakeRequestV1(_ContractV1):
    CONTRACT: ClassVar[str] = "WakeRequestV1"

    request_id: str
    goal_id: str
    scope_id: str
    requested_by: str
    reason: str
    requested_at_ms: int
    expires_at_ms: int
    max_turns: int
    max_effects: int
    max_compute_ms: int
    max_proactive_messages: int
    allowed_actions: tuple[str, ...]
    can_self_authorize: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("wake request ID", self.request_id),
            ("wake request goal ID", self.goal_id),
            ("wake request scope ID", self.scope_id),
            ("wake requester", self.requested_by),
        ):
            _text(name, value, 256)
        _text("wake reason", self.reason, 4_096)
        _time("wake request time", self.requested_at_ms)
        _time("wake request expiry", self.expires_at_ms)
        if self.expires_at_ms <= self.requested_at_ms:
            raise ValueError("wake request expiry must follow its request")
        _positive_int("wake request turn budget", self.max_turns)
        _nonnegative_int("wake request effect budget", self.max_effects)
        _positive_int("wake request compute budget", self.max_compute_ms)
        _nonnegative_int(
            "wake request proactive-message budget", self.max_proactive_messages
        )
        _actions(self.allowed_actions)
        if self.can_self_authorize is not False:
            raise ValueError("wake requests cannot self-authorize")

    def require_active(self, now_ms: int) -> None:
        _time("wake request check time", now_ms)
        if now_ms < self.requested_at_ms or now_ms >= self.expires_at_ms:
            raise ValueError("wake request is expired or not yet active")

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        fields = _payload(value, cls.CONTRACT, cls)
        return cls(
            request_id=fields["request_id"],
            goal_id=fields["goal_id"],
            scope_id=fields["scope_id"],
            requested_by=fields["requested_by"],
            reason=fields["reason"],
            requested_at_ms=fields["requested_at_ms"],
            expires_at_ms=fields["expires_at_ms"],
            max_turns=fields["max_turns"],
            max_effects=fields["max_effects"],
            max_compute_ms=fields["max_compute_ms"],
            max_proactive_messages=fields["max_proactive_messages"],
            allowed_actions=_strings(fields["allowed_actions"]),
            can_self_authorize=fields["can_self_authorize"],
        )


@dataclass(frozen=True, slots=True)
class WakeLeaseV1(_ContractV1):
    CONTRACT: ClassVar[str] = "WakeLeaseV1"

    lease_id: str
    request_id: str
    goal_id: str
    scope_id: str
    issued_by: str
    issued_at_ms: int
    expires_at_ms: int
    max_turns: int
    max_effects: int
    max_compute_ms: int
    max_proactive_messages: int
    allowed_actions: tuple[str, ...]
    can_self_extend: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("wake lease ID", self.lease_id),
            ("wake lease request ID", self.request_id),
            ("wake lease goal ID", self.goal_id),
            ("wake lease scope ID", self.scope_id),
            ("wake lease issuer", self.issued_by),
        ):
            _text(name, value, 256)
        _time("wake lease issue time", self.issued_at_ms)
        _time("wake lease expiry", self.expires_at_ms)
        if self.expires_at_ms <= self.issued_at_ms:
            raise ValueError("wake lease expiry must follow issue")
        _positive_int("wake lease turn budget", self.max_turns)
        _nonnegative_int("wake lease effect budget", self.max_effects)
        _positive_int("wake lease compute budget", self.max_compute_ms)
        _nonnegative_int(
            "wake lease proactive-message budget", self.max_proactive_messages
        )
        _actions(self.allowed_actions)
        if self.can_self_extend is not False:
            raise ValueError("wake leases cannot self-extend")

    def require_active(self, now_ms: int) -> None:
        _time("wake lease check time", now_ms)
        if now_ms < self.issued_at_ms or now_ms >= self.expires_at_ms:
            raise ValueError("wake lease is expired or not yet active")

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        fields = _payload(value, cls.CONTRACT, cls)
        return cls(
            lease_id=fields["lease_id"],
            request_id=fields["request_id"],
            goal_id=fields["goal_id"],
            scope_id=fields["scope_id"],
            issued_by=fields["issued_by"],
            issued_at_ms=fields["issued_at_ms"],
            expires_at_ms=fields["expires_at_ms"],
            max_turns=fields["max_turns"],
            max_effects=fields["max_effects"],
            max_compute_ms=fields["max_compute_ms"],
            max_proactive_messages=fields["max_proactive_messages"],
            allowed_actions=_strings(fields["allowed_actions"]),
            can_self_extend=fields["can_self_extend"],
        )


@dataclass(frozen=True, slots=True)
class CognitionSnapshotV1(_ContractV1):
    CONTRACT: ClassVar[str] = "CognitionSnapshotV1"

    snapshot_id: str
    scope_id: str
    perspective_register_ids: tuple[str, ...]
    goal_ids: tuple[str, ...]
    unresolved_question_refs: tuple[str, ...]
    commitment_refs: tuple[str, ...]
    drive_state: tuple[tuple[str, float], ...]
    memory_ids: tuple[str, ...]
    graph_projection_ids: tuple[str, ...]
    candidate_queue_refs: tuple[str, ...]
    wake_lease_id: str
    context_digest: str
    trace_chain_head: str
    ledger_head: str
    last_verified_checkpoint: str
    created_at_ms: int
    hidden_state_persisted: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("cognition snapshot ID", self.snapshot_id),
            ("cognition snapshot scope ID", self.scope_id),
            ("cognition wake lease ID", self.wake_lease_id),
        ):
            _text(name, value, 256)
        _unique_text_tuple(
            "cognition perspective register IDs",
            self.perspective_register_ids,
            3,
        )
        _unique_text_tuple("cognition goal IDs", self.goal_ids, 128, allow_empty=True)
        _unique_text_tuple(
            "cognition unresolved question refs",
            self.unresolved_question_refs,
            256,
            allow_empty=True,
        )
        _unique_text_tuple(
            "cognition commitment refs",
            self.commitment_refs,
            256,
            allow_empty=True,
        )
        _bounded_score_pairs("cognition drive state", self.drive_state, 0.0, 1.0)
        _unique_text_tuple(
            "cognition memory IDs", self.memory_ids, 1_024, allow_empty=True
        )
        _unique_text_tuple(
            "cognition graph projection IDs",
            self.graph_projection_ids,
            1_024,
            allow_empty=True,
        )
        _unique_text_tuple(
            "cognition candidate queue refs",
            self.candidate_queue_refs,
            1_024,
            allow_empty=True,
        )
        _digest("cognition context digest", self.context_digest)
        if self.trace_chain_head != "genesis":
            _digest("cognition trace-chain head", self.trace_chain_head)
        if self.ledger_head != "genesis":
            _digest("cognition ledger head", self.ledger_head)
        _digest(
            "cognition last verified checkpoint", self.last_verified_checkpoint
        )
        _time("cognition snapshot creation time", self.created_at_ms)
        if self.hidden_state_persisted is not False:
            raise ValueError("cognition snapshots cannot persist hidden neural state")

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        fields = _payload(value, cls.CONTRACT, cls)
        return cls(
            snapshot_id=fields["snapshot_id"],
            scope_id=fields["scope_id"],
            perspective_register_ids=_strings(fields["perspective_register_ids"]),
            goal_ids=_strings(fields["goal_ids"]),
            unresolved_question_refs=_strings(
                fields["unresolved_question_refs"]
            ),
            commitment_refs=_strings(fields["commitment_refs"]),
            drive_state=_score_pairs(fields["drive_state"]),
            memory_ids=_strings(fields["memory_ids"]),
            graph_projection_ids=_strings(fields["graph_projection_ids"]),
            candidate_queue_refs=_strings(fields["candidate_queue_refs"]),
            wake_lease_id=fields["wake_lease_id"],
            context_digest=fields["context_digest"],
            trace_chain_head=fields["trace_chain_head"],
            ledger_head=fields["ledger_head"],
            last_verified_checkpoint=fields["last_verified_checkpoint"],
            created_at_ms=fields["created_at_ms"],
            hidden_state_persisted=fields["hidden_state_persisted"],
        )


@dataclass(frozen=True, slots=True)
class ProactiveMessageV1(_ContractV1):
    CONTRACT: ClassVar[str] = "ProactiveMessageV1"

    message_id: str
    goal_id: str
    wake_lease_id: str
    scope_id: str
    content_digest: str
    state: MessageState
    created_at_ms: int
    requires_user_attention: bool
    can_authorize_effects: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("proactive message ID", self.message_id),
            ("proactive goal ID", self.goal_id),
            ("proactive wake lease ID", self.wake_lease_id),
            ("proactive scope ID", self.scope_id),
        ):
            _text(name, value, 256)
        _digest("proactive message content digest", self.content_digest)
        if not isinstance(self.state, MessageState):
            raise ValueError("proactive message state is invalid")
        _time("proactive message creation time", self.created_at_ms)
        if type(self.requires_user_attention) is not bool:
            raise ValueError("requires_user_attention must be boolean")
        if self.can_authorize_effects is not False:
            raise ValueError("proactive messages cannot authorize effects")

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        fields = _payload(value, cls.CONTRACT, cls)
        return cls(
            message_id=fields["message_id"],
            goal_id=fields["goal_id"],
            wake_lease_id=fields["wake_lease_id"],
            scope_id=fields["scope_id"],
            content_digest=fields["content_digest"],
            state=MessageState(fields["state"]),
            created_at_ms=fields["created_at_ms"],
            requires_user_attention=fields["requires_user_attention"],
            can_authorize_effects=fields["can_authorize_effects"],
        )


def domain_digest(domain: str, value: Mapping[str, object]) -> str:
    """Return a canonical SHA-256 digest separated from every other domain."""
    _text("digest domain", domain, 256)
    try:
        domain_bytes = domain.encode("ascii")
        payload = json.dumps(
            dict(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (UnicodeEncodeError, TypeError, ValueError, RecursionError) as error:
        raise ValueError("digest value must be canonical JSON") from error
    return hashlib.sha256(domain_bytes + b"\x00" + payload).hexdigest()


def _payload(value: object, contract: str, cls: type[object]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{contract} must be a mapping")
    expected = {field.name for field in dataclass_fields(cls) if field.init}
    if set(value) != {"contract", *expected} or value.get("contract") != contract:
        raise ValueError(f"{contract} fields do not match its schema")
    return {name: value[name] for name in expected}


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("contract sequence must be a list or tuple")
    return tuple(value)


def _enums(value: object, enum_type: type[Enum]) -> tuple[Any, ...]:
    return tuple(enum_type(item) for item in _strings(value))


def _score_pairs(value: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("score pairs must be a list or tuple")
    pairs: list[tuple[str, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("score pair must contain a name and value")
        pairs.append((item[0], item[1]))
    return tuple(pairs)


def _text(name: str, value: object, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValueError(f"{name} must be bounded non-empty text")


def _digest(name: str, value: object) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _time(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _positive_int(name: str, value: object) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(name: str, value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _bounded_float(name: str, value: object, minimum: float, maximum: float) -> None:
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")


def _bounded_score_pairs(
    name: str,
    value: object,
    minimum: float,
    maximum: float,
) -> None:
    if not isinstance(value, tuple) or len(value) > 128:
        raise ValueError(f"{name} must be a bounded tuple")
    keys: list[str] = []
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{name} entries must be name/value tuples")
        key, score = item
        _text(name, key, 256)
        _bounded_float(name, score, minimum, maximum)
        keys.append(key)
    if len(set(keys)) != len(keys) or keys != sorted(keys):
        raise ValueError(f"{name} keys must be unique and sorted")


def _unique_text_tuple(
    name: str,
    value: object,
    maximum: int,
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(value, tuple) or len(value) > maximum or (not value and not allow_empty):
        raise ValueError(f"{name} must be a bounded tuple")
    for item in value:
        _text(name, item, 256)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must be unique")


def _unique_enum_tuple(
    name: str,
    value: object,
    enum_type: type[Enum],
    maximum: int,
) -> None:
    if (
        not isinstance(value, tuple)
        or not value
        or len(value) > maximum
        or any(not isinstance(item, enum_type) for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be a bounded unique enum tuple")


def _actions(value: object) -> None:
    _unique_text_tuple("allowed actions", value, 128, allow_empty=True)
    if any(not _ACTION.fullmatch(action) for action in value):
        raise ValueError("allowed actions must use capability:operation names")


__all__ = [
    "ArtifactEnvelopeV1",
    "CognitionSnapshotV1",
    "ConsentScopeV1",
    "ConsentUse",
    "EventTrust",
    "EvidenceOutcome",
    "EvidenceReceiptV1",
    "EvidenceStrength",
    "GoalStatus",
    "GraphProjectionV1",
    "InteractionEventV1",
    "InteractionKind",
    "LearningDestination",
    "LearningRecordV1",
    "LIFELONG_CONTRACT_VERSION",
    "MemoryLayer",
    "MemoryRecordV1",
    "MessageState",
    "PerspectiveRegisterV1",
    "ProactiveMessageV1",
    "QuestionCandidateV1",
    "QuestionState",
    "Sister",
    "StandingGoalV1",
    "WakeLeaseV1",
    "WakeRequestV1",
    "domain_digest",
]
