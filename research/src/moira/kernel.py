"""Persistent offline control facade for MOIRA's non-trainable kernels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Mapping, TypeVar

from .lifelong import (
    ArtifactEnvelopeV1,
    CognitionSnapshotV1,
    ConsentScopeV1,
    ConsentUse,
    EvidenceOutcome,
    EvidenceReceiptV1,
    EvidenceStrength,
    GoalStatus,
    GraphProjectionV1,
    InteractionEventV1,
    LearningDestination,
    LearningRecordV1,
    MemoryLayer,
    MemoryRecordV1,
    PerspectiveRegisterV1,
    ProactiveMessageV1,
    QuestionCandidateV1,
    StandingGoalV1,
    WakeLeaseV1,
    WakeRequestV1,
    domain_digest,
)


MOIRA_KERNEL_PROTOCOL = "moira-offline-kernel-v1"
MOIRA_KERNEL_STATE_VERSION = 1
MAX_KERNEL_STATE_BYTES = 16 * 1024 * 1024
KERNEL_MODULE_NAMES = (
    "EvidenceKernel",
    "EffectsKernel",
    "RunKernel",
    "AssuranceKernel",
    "EvolutionKernel",
)

_GENESIS = "genesis"
_STATE_FIELDS = {
    "protocol",
    "schema_version",
    "revision",
    "capture_enabled",
    "modules",
    "perspective_registers",
    "consents",
    "artifacts",
    "interactions",
    "evidence",
    "learning_records",
    "memories",
    "graph_projections",
    "standing_goals",
    "question_candidates",
    "wake_requests",
    "wake_leases",
    "cognition_snapshots",
    "proactive_messages",
    "receipts",
}

class ReceiptDomain(str, Enum):
    """Closed audit domains; these are receipts, never effect permits."""

    CAPTURE_POLICY = "AssuranceKernel.capture-policy.v1"
    PERSPECTIVE = "EvidenceKernel.perspective-register.v1"
    CONSENT = "AssuranceKernel.consent-scope.v1"
    ARTIFACT = "EvidenceKernel.artifact-envelope.v1"
    INTERACTION = "EvidenceKernel.interaction-event.v1"
    CONTEXT = "EvidenceKernel.context-receipt.v1"
    EVIDENCE = "AssuranceKernel.evidence-receipt.v1"
    REVIEW = "AssuranceKernel.review-receipt.v1"
    EFFECT_OUTCOME = "EffectsKernel.effect-outcome-receipt.v1"
    LEARNING = "EvolutionKernel.learning-record.v1"
    MEMORY = "EvidenceKernel.memory-record.v1"
    GRAPH_PROJECTION = "EvidenceKernel.graph-projection.v1"
    GOAL = "RunKernel.standing-goal.v1"
    QUESTION = "RunKernel.question-candidate.v1"
    WAKE_REQUEST = "RunKernel.wake-request.v1"
    WAKE_LEASE = "RunKernel.wake-lease.v1"
    RUN = "RunKernel.run-receipt.v1"
    SNAPSHOT = "RunKernel.cognition-snapshot.v1"
    PROACTIVE_MESSAGE = "EffectsKernel.proactive-message.v1"
    EVALUATION = "AssuranceKernel.evaluation-receipt.v1"
    TRAINING = "EvolutionKernel.training-receipt.v1"
    CANDIDATE = "EvolutionKernel.candidate-receipt.v1"
    ACTIVATION = "AssuranceKernel.activation-receipt.v1"
    ROLLBACK = "AssuranceKernel.rollback-receipt.v1"


_DOMAIN_CAPTURE = ReceiptDomain.CAPTURE_POLICY
_DOMAIN_PERSPECTIVE = ReceiptDomain.PERSPECTIVE
_DOMAIN_CONSENT = ReceiptDomain.CONSENT
_DOMAIN_ARTIFACT = ReceiptDomain.ARTIFACT
_DOMAIN_INTERACTION = ReceiptDomain.INTERACTION
_DOMAIN_EVIDENCE = ReceiptDomain.EVIDENCE
_DOMAIN_LEARNING = ReceiptDomain.LEARNING
_DOMAIN_MEMORY = ReceiptDomain.MEMORY
_DOMAIN_PROJECTION = ReceiptDomain.GRAPH_PROJECTION
_DOMAIN_GOAL = ReceiptDomain.GOAL
_DOMAIN_QUESTION = ReceiptDomain.QUESTION
_DOMAIN_WAKE_REQUEST = ReceiptDomain.WAKE_REQUEST
_DOMAIN_WAKE_LEASE = ReceiptDomain.WAKE_LEASE
_DOMAIN_SNAPSHOT = ReceiptDomain.SNAPSHOT
_DOMAIN_MESSAGE = ReceiptDomain.PROACTIVE_MESSAGE
_IMPLEMENTED_RECEIPT_DOMAINS = frozenset(
    {
        _DOMAIN_CAPTURE,
        _DOMAIN_PERSPECTIVE,
        _DOMAIN_CONSENT,
        _DOMAIN_ARTIFACT,
        _DOMAIN_INTERACTION,
        _DOMAIN_EVIDENCE,
        _DOMAIN_LEARNING,
        _DOMAIN_MEMORY,
        _DOMAIN_PROJECTION,
        _DOMAIN_GOAL,
        _DOMAIN_QUESTION,
        _DOMAIN_WAKE_REQUEST,
        _DOMAIN_WAKE_LEASE,
        _DOMAIN_SNAPSHOT,
        _DOMAIN_MESSAGE,
    }
)

_LAYER_RANK = {
    MemoryLayer.SESSION: 0,
    MemoryLayer.EPISODIC: 1,
    MemoryLayer.SEMANTIC: 2,
    MemoryLayer.PROCEDURAL: 3,
}


@dataclass(frozen=True, slots=True)
class KernelModuleIdentityV1:
    name: str
    trainable_parameters: int = 0
    trainable: bool = False
    model_mutable: bool = False

    def __post_init__(self) -> None:
        if self.name not in KERNEL_MODULE_NAMES:
            raise ValueError("unknown MOIRA kernel module identity")
        if (
            self.trainable_parameters != 0
            or self.trainable is not False
            or self.model_mutable is not False
        ):
            raise ValueError("MOIRA kernel module identities must be non-trainable")

    def mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: object) -> "KernelModuleIdentityV1":
        if not isinstance(value, Mapping) or set(value) != {
            "name",
            "trainable_parameters",
            "trainable",
            "model_mutable",
        }:
            raise ValueError("kernel module identity has an invalid schema")
        return cls(
            name=value["name"],
            trainable_parameters=value["trainable_parameters"],
            trainable=value["trainable"],
            model_mutable=value["model_mutable"],
        )


KERNEL_MODULE_IDENTITIES = tuple(
    KernelModuleIdentityV1(name) for name in KERNEL_MODULE_NAMES
)


@dataclass(frozen=True, slots=True)
class ReceiptEntryV1:
    index: int
    domain: ReceiptDomain
    record_id: str
    record_digest: str
    previous_digest: str
    receipt_digest: str

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("receipt index must be a non-negative integer")
        if not isinstance(self.domain, ReceiptDomain):
            raise ValueError("receipt domain is outside the closed receipt schema")
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError("receipt record ID must be non-empty")
        _sha256("receipt record digest", self.record_digest)
        if self.previous_digest != _GENESIS:
            _sha256("receipt previous digest", self.previous_digest)
        _sha256("receipt digest", self.receipt_digest)

    def material(self) -> dict[str, object]:
        return {
            "index": self.index,
            "record_id": self.record_id,
            "record_digest": self.record_digest,
            "previous_digest": self.previous_digest,
        }

    def mapping(self) -> dict[str, object]:
        return {
            **self.material(),
            "domain": self.domain.value,
            "receipt_digest": self.receipt_digest,
        }

    def verify(self) -> None:
        expected = domain_digest(
            f"moira.kernel.receipt.{self.domain.value}", self.material()
        )
        if self.receipt_digest != expected:
            raise ValueError("kernel receipt digest does not match its material")

    @classmethod
    def create(
        cls,
        *,
        index: int,
        domain: ReceiptDomain,
        record_id: str,
        record_mapping: Mapping[str, object],
        previous_digest: str,
    ) -> "ReceiptEntryV1":
        if not isinstance(domain, ReceiptDomain):
            raise ValueError("receipt domain is outside the closed receipt schema")
        record_digest = domain_digest(f"{domain.value}.record", record_mapping)
        material = {
            "index": index,
            "record_id": record_id,
            "record_digest": record_digest,
            "previous_digest": previous_digest,
        }
        return cls(
            index=index,
            domain=domain,
            record_id=record_id,
            record_digest=record_digest,
            previous_digest=previous_digest,
            receipt_digest=domain_digest(
                f"moira.kernel.receipt.{domain.value}", material
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> "ReceiptEntryV1":
        if not isinstance(value, Mapping) or set(value) != {
            "index",
            "domain",
            "record_id",
            "record_digest",
            "previous_digest",
            "receipt_digest",
        }:
            raise ValueError("kernel receipt has an invalid schema")
        return cls(
            index=value["index"],
            domain=ReceiptDomain(value["domain"]),
            record_id=value["record_id"],
            record_digest=value["record_digest"],
            previous_digest=value["previous_digest"],
            receipt_digest=value["receipt_digest"],
        )


T = TypeVar("T")


class MoiraKernel:
    """One deterministic facade over MOIRA's five offline control kernels."""

    def __init__(self, state_path: str | Path) -> None:
        self.state_path = Path(state_path)
        _validate_state_parent(self.state_path)
        self._revision = 0
        self._capture_enabled = False
        self._perspective_registers: dict[str, PerspectiveRegisterV1] = {}
        self._consents: dict[str, ConsentScopeV1] = {}
        self._artifacts: dict[str, ArtifactEnvelopeV1] = {}
        self._interactions: dict[str, InteractionEventV1] = {}
        self._evidence: dict[str, EvidenceReceiptV1] = {}
        self._learning_records: dict[str, LearningRecordV1] = {}
        self._memories: dict[str, MemoryRecordV1] = {}
        self._graph_projections: dict[str, GraphProjectionV1] = {}
        self._standing_goals: dict[str, StandingGoalV1] = {}
        self._question_candidates: dict[str, QuestionCandidateV1] = {}
        self._wake_requests: dict[str, WakeRequestV1] = {}
        self._wake_leases: dict[str, WakeLeaseV1] = {}
        self._cognition_snapshots: dict[str, CognitionSnapshotV1] = {}
        self._proactive_messages: dict[str, ProactiveMessageV1] = {}
        self._receipts: list[ReceiptEntryV1] = []
        if os.path.lexists(self.state_path):
            self._load()
        else:
            self._persist()

    @property
    def module_identities(self) -> tuple[KernelModuleIdentityV1, ...]:
        return KERNEL_MODULE_IDENTITIES

    @property
    def capture_enabled(self) -> bool:
        return self._capture_enabled

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def receipts(self) -> tuple[ReceiptEntryV1, ...]:
        return tuple(self._receipts)

    @property
    def receipt_head(self) -> str:
        return self._receipts[-1].receipt_digest if self._receipts else _GENESIS

    @property
    def state_digest(self) -> str:
        return domain_digest(MOIRA_KERNEL_PROTOCOL, self._state_mapping())

    @property
    def memories(self) -> tuple[MemoryRecordV1, ...]:
        return tuple(self._memories[key] for key in sorted(self._memories))

    @property
    def artifacts(self) -> tuple[ArtifactEnvelopeV1, ...]:
        return tuple(self._artifacts[key] for key in sorted(self._artifacts))

    @property
    def question_candidates(self) -> tuple[QuestionCandidateV1, ...]:
        return tuple(
            self._question_candidates[key]
            for key in sorted(self._question_candidates)
        )

    @property
    def graph_projections(self) -> tuple[GraphProjectionV1, ...]:
        return tuple(
            self._graph_projections[key] for key in sorted(self._graph_projections)
        )

    def configure_capture(self, enabled: bool) -> ReceiptEntryV1:
        if type(enabled) is not bool:
            raise ValueError("capture policy must be boolean")
        previous = self._capture_enabled
        self._capture_enabled = enabled
        receipt = self._append_receipt(
            _DOMAIN_CAPTURE,
            "capture-policy",
            {"capture_enabled": enabled},
        )
        try:
            self._commit()
        except Exception:
            self._capture_enabled = previous
            self._receipts.pop()
            raise
        return receipt

    def register_perspectives(
        self, register: PerspectiveRegisterV1
    ) -> ReceiptEntryV1:
        _require_type(register, PerspectiveRegisterV1)
        if register.last_consolidation_receipt != _GENESIS:
            raise ValueError(
                "typed perspective consolidation receipts are not implemented"
            )
        self._require_records(
            self._standing_goals,
            register.active_goal_refs,
            "active standing goal",
            allow_empty=True,
        )
        self._require_records(
            self._question_candidates,
            register.unresolved_question_refs,
            "unresolved question",
            allow_empty=True,
        )
        current = self._perspective_registers.get(register.register_id)
        if current is not None:
            if (
                current.owner_id != register.owner_id
                or current.scope_id != register.scope_id
                or current.sister is not register.sister
                or register.updated_at_ms <= current.updated_at_ms
            ):
                raise ValueError("perspective update changes identity or is not newer")
            return self._replace_record(
                self._perspective_registers,
                register.register_id,
                register,
                _DOMAIN_PERSPECTIVE,
            )
        for existing in self._perspective_registers.values():
            if (
                existing.owner_id == register.owner_id
                and existing.scope_id == register.scope_id
                and existing.sister is register.sister
            ):
                raise ValueError("sister perspective already exists in the scope")
        return self._insert(
            self._perspective_registers,
            register.register_id,
            register,
            _DOMAIN_PERSPECTIVE,
        )

    def grant_consent(self, consent: ConsentScopeV1) -> ReceiptEntryV1:
        _require_type(consent, ConsentScopeV1)
        if not any(
            register.owner_id == consent.owner_id
            and register.scope_id == consent.scope_id
            for register in self._perspective_registers.values()
        ):
            raise ValueError("consent owner and scope have no perspective register")
        return self._insert(
            self._consents,
            consent.consent_id,
            consent,
            _DOMAIN_CONSENT,
        )

    def store_artifact(self, artifact: ArtifactEnvelopeV1) -> ReceiptEntryV1:
        _require_type(artifact, ArtifactEnvelopeV1)
        if not self._capture_enabled:
            raise ValueError("artifact capture is disabled")
        consent = self._consents.get(artifact.consent_id)
        if (
            consent is None
            or consent.owner_id != artifact.owner_id
            or consent.scope_id != artifact.scope_id
            or not consent.allows_event(
                artifact.interaction_kind, artifact.created_at_ms
            )
            or artifact.expires_at_ms > consent.expires_at_ms
        ):
            raise ValueError("artifact has no matching active consent scope")
        return self._insert(
            self._artifacts,
            artifact.artifact_id,
            artifact,
            _DOMAIN_ARTIFACT,
        )

    def capture_interaction(self, event: InteractionEventV1) -> ReceiptEntryV1:
        _require_type(event, InteractionEventV1)
        if not self._capture_enabled:
            raise ValueError("interaction capture is disabled")
        consent = self._consents.get(event.consent_id)
        if consent is None or consent.scope_id != event.scope_id:
            raise ValueError("interaction has no matching consent scope")
        owner_id = self._perspective_owner(event.scope_id, event.perspective_id)
        if owner_id != consent.owner_id:
            raise ValueError("interaction perspective does not match consent owner")
        if not consent.allows_event(event.kind, event.occurred_at_ms):
            raise ValueError("interaction kind is not actively consented")
        if event.artifact_id is not None:
            artifact = self._artifacts.get(event.artifact_id)
            if (
                artifact is None
                or not artifact.live
                or artifact.scope_id != event.scope_id
                or artifact.consent_id != event.consent_id
                or artifact.interaction_kind is not event.kind
                or artifact.content_digest != event.payload_digest
                or event.occurred_at_ms >= artifact.expires_at_ms
            ):
                raise ValueError("interaction artifact does not match its envelope")
        return self._insert(
            self._interactions,
            event.event_id,
            event,
            _DOMAIN_INTERACTION,
        )

    def record_evidence(self, evidence: EvidenceReceiptV1) -> ReceiptEntryV1:
        _require_type(evidence, EvidenceReceiptV1)
        events = self._require_records(
            self._interactions, evidence.event_ids, "interaction event"
        )
        if any(event.scope_id != evidence.scope_id for event in events):
            raise ValueError("evidence crosses an interaction scope")
        if any(event.occurred_at_ms > evidence.issued_at_ms for event in events):
            raise ValueError("evidence cannot precede its interactions")
        return self._insert(
            self._evidence,
            evidence.evidence_id,
            evidence,
            _DOMAIN_EVIDENCE,
        )

    def learning_provenance_digest(
        self, event_ids: tuple[str, ...], evidence_ids: tuple[str, ...]
    ) -> str:
        events = self._require_records(self._interactions, event_ids, "interaction event")
        evidence = self._require_records(
            self._evidence, evidence_ids, "evidence", allow_empty=True
        )
        return domain_digest(
            "EvolutionKernel.learning-provenance.v1",
            {
                "events": [item.digest for item in events],
                "evidence": [item.digest for item in evidence],
            },
        )

    def record_learning(self, record: LearningRecordV1) -> ReceiptEntryV1:
        """Store a candidate record; its evidence fields grant no training authority."""
        _require_type(record, LearningRecordV1)
        events = self._require_records(
            self._interactions, record.event_ids, "interaction event"
        )
        evidence = self._require_records(
            self._evidence,
            record.evidence_ids,
            "evidence",
            allow_empty=not record.training_eligible,
        )
        if any(event.scope_id != record.scope_id for event in events) or any(
            item.scope_id != record.scope_id for item in evidence
        ):
            raise ValueError("learning record crosses a scope boundary")
        if record.provenance_digest != self.learning_provenance_digest(
            record.event_ids, record.evidence_ids
        ):
            raise ValueError("learning provenance does not match its records")
        event_ids = set(record.event_ids)
        if any(not set(item.event_ids).issubset(event_ids) for item in evidence):
            raise ValueError("learning evidence covers events outside its batch")
        if any(event.occurred_at_ms > record.created_at_ms for event in events) or any(
            item.issued_at_ms > record.created_at_ms for item in evidence
        ):
            raise ValueError("learning record cannot precede its events or evidence")
        required_use = _learning_use(record.destination)
        for event in events:
            consent = self._consents[event.consent_id]
            if not consent.allows_use(required_use, record.created_at_ms):
                raise ValueError("learning use is not actively consented")
        if record.training_eligible:
            if not all(event.training_eligible for event in events):
                raise ValueError("weak interactions cannot make learning trainable")
            unverified_event_ids = tuple(
                event.event_id
                for event in events
                if not any(
                    event.event_id in item.event_ids
                    and item.outcome is EvidenceOutcome.PASS
                    and item.strength is EvidenceStrength.STRONG
                    and item.independent
                    for item in evidence
                )
            )
            if unverified_event_ids:
                raise ValueError(
                    "each training-eligible interaction requires independent strong evidence"
                )
        return self._insert(
            self._learning_records,
            record.learning_id,
            record,
            _DOMAIN_LEARNING,
        )

    def store_memory(self, memory: MemoryRecordV1) -> ReceiptEntryV1:
        _require_type(memory, MemoryRecordV1)
        learning = self._require_records(
            self._learning_records,
            memory.provenance_learning_ids,
            "learning record",
            allow_empty=True,
        )
        parents = self._require_records(
            self._memories,
            memory.parent_memory_ids,
            "parent memory",
            allow_empty=True,
        )
        if any(item.scope_id != memory.scope_id for item in (*learning, *parents)):
            raise ValueError("memory crosses a provenance scope")
        if any(not parent.live for parent in parents):
            raise ValueError("memory cannot descend from a tombstone")
        if any(_LAYER_RANK[parent.layer] > _LAYER_RANK[memory.layer] for parent in parents):
            raise ValueError("memory cannot descend into a lower layer")
        return self._insert(
            self._memories,
            memory.memory_id,
            memory,
            _DOMAIN_MEMORY,
        )

    def graph_projection_digest(
        self,
        relation: str,
        source_memory_ids: tuple[str, ...],
        parent_projection_ids: tuple[str, ...],
    ) -> str:
        memories = self._require_records(
            self._memories, source_memory_ids, "source memory", allow_empty=True
        )
        projections = self._require_records(
            self._graph_projections,
            parent_projection_ids,
            "parent graph projection",
            allow_empty=True,
        )
        return domain_digest(
            "EvidenceKernel.graph-provenance.v1",
            {
                "relation": relation,
                "memories": [item.digest for item in memories],
                "projections": [item.digest for item in projections],
            },
        )

    def project_graph(self, projection: GraphProjectionV1) -> ReceiptEntryV1:
        _require_type(projection, GraphProjectionV1)
        memories = self._require_records(
            self._memories,
            projection.source_memory_ids,
            "source memory",
            allow_empty=True,
        )
        parents = self._require_records(
            self._graph_projections,
            projection.parent_projection_ids,
            "parent graph projection",
            allow_empty=True,
        )
        if any(item.scope_id != projection.scope_id for item in (*memories, *parents)):
            raise ValueError("graph projection crosses a provenance scope")
        if any(not item.live for item in (*memories, *parents)):
            raise ValueError("graph projection cannot descend from a tombstone")
        expected = self.graph_projection_digest(
            projection.relation,
            projection.source_memory_ids,
            projection.parent_projection_ids,
        )
        if projection.projection_digest != expected:
            raise ValueError("graph projection digest does not match its provenance")
        return self._insert(
            self._graph_projections,
            projection.projection_id,
            projection,
            _DOMAIN_PROJECTION,
        )

    def tombstone_memory(self, memory_id: str, *, now_ms: int) -> dict[str, object]:
        root = self._memories.get(memory_id)
        if root is None:
            raise ValueError("unknown memory")
        root.tombstone(now_ms)
        memory_ids = {memory_id}
        changed = True
        while changed:
            changed = False
            for memory in self._memories.values():
                if (
                    memory.memory_id not in memory_ids
                    and set(memory.parent_memory_ids) & memory_ids
                ):
                    memory_ids.add(memory.memory_id)
                    changed = True

        projection_ids = {
            projection.projection_id
            for projection in self._graph_projections.values()
            if set(projection.source_memory_ids) & memory_ids
        }
        changed = True
        while changed:
            changed = False
            for projection in self._graph_projections.values():
                if (
                    projection.projection_id not in projection_ids
                    and set(projection.parent_projection_ids) & projection_ids
                ):
                    projection_ids.add(projection.projection_id)
                    changed = True

        memory_updates = {
            identifier: self._memories[identifier].tombstone(now_ms)
            for identifier in memory_ids
        }
        projection_updates = {
            identifier: self._graph_projections[identifier].tombstone(now_ms)
            for identifier in projection_ids
        }
        previous_memories = {
            identifier: self._memories[identifier] for identifier in memory_updates
        }
        previous_projections = {
            identifier: self._graph_projections[identifier]
            for identifier in projection_updates
        }
        receipt_count = len(self._receipts)
        try:
            for identifier in sorted(memory_updates):
                current = self._memories[identifier]
                updated = memory_updates[identifier]
                if updated != current:
                    self._memories[identifier] = updated
                    self._append_receipt(
                        _DOMAIN_MEMORY, updated.memory_id, updated.mapping()
                    )
            for identifier in sorted(projection_updates):
                current = self._graph_projections[identifier]
                updated = projection_updates[identifier]
                if updated != current:
                    self._graph_projections[identifier] = updated
                    self._append_receipt(
                        _DOMAIN_PROJECTION, updated.projection_id, updated.mapping()
                    )
            self._commit()
        except Exception:
            self._memories.update(previous_memories)
            self._graph_projections.update(previous_projections)
            del self._receipts[receipt_count:]
            raise
        return {
            "memory_ids": tuple(sorted(memory_ids)),
            "projection_ids": tuple(sorted(projection_ids)),
            "ledger_head": self.receipt_head,
        }

    def add_standing_goal(self, goal: StandingGoalV1) -> ReceiptEntryV1:
        _require_type(goal, StandingGoalV1)
        self._perspective_owner(goal.scope_id, goal.perspective_id)
        return self._insert(
            self._standing_goals,
            goal.goal_id,
            goal,
            _DOMAIN_GOAL,
        )

    def add_question_candidate(
        self, question: QuestionCandidateV1
    ) -> ReceiptEntryV1:
        _require_type(question, QuestionCandidateV1)
        register = self._perspective_registers.get(
            question.perspective_register_id
        )
        goal = self._standing_goals.get(question.goal_id)
        if (
            register is None
            or goal is None
            or register.scope_id != question.scope_id
            or goal.scope_id != question.scope_id
            or goal.perspective_id != register.register_id
        ):
            raise ValueError("question candidate crosses its perspective or goal")
        evidence = self._require_records(
            self._evidence,
            question.evidence_refs,
            "question evidence",
            allow_empty=True,
        )
        if any(item.scope_id != question.scope_id for item in evidence):
            raise ValueError("question candidate crosses an evidence scope")
        return self._insert(
            self._question_candidates,
            question.question_id,
            question,
            _DOMAIN_QUESTION,
        )

    def request_wake(self, request: WakeRequestV1) -> ReceiptEntryV1:
        _require_type(request, WakeRequestV1)
        goal = self._standing_goals.get(request.goal_id)
        if goal is None or goal.scope_id != request.scope_id:
            raise ValueError("wake request has no matching standing goal")
        if goal.status is not GoalStatus.ACTIVE:
            raise ValueError("wake request requires an active standing goal")
        if (
            request.max_turns > goal.max_turns_per_wake
            or request.max_effects > goal.max_effects_per_wake
            or request.max_compute_ms > goal.max_compute_ms_per_wake
            or request.max_proactive_messages
            > goal.max_proactive_messages_per_wake
            or not set(request.allowed_actions).issubset(goal.allowed_actions)
        ):
            raise ValueError("wake request exceeds its standing goal")
        return self._insert(
            self._wake_requests,
            request.request_id,
            request,
            _DOMAIN_WAKE_REQUEST,
        )

    def issue_wake_lease(
        self, lease: WakeLeaseV1, *, now_ms: int
    ) -> ReceiptEntryV1:
        _require_type(lease, WakeLeaseV1)
        request = self._wake_requests.get(lease.request_id)
        if request is None:
            raise ValueError("wake lease has no matching request")
        request.require_active(now_ms)
        if any(item.request_id == request.request_id for item in self._wake_leases.values()):
            raise ValueError("wake request already has a lease")
        if (
            lease.goal_id != request.goal_id
            or lease.scope_id != request.scope_id
            or lease.issued_at_ms < request.requested_at_ms
            or lease.issued_at_ms > now_ms
            or lease.expires_at_ms > request.expires_at_ms
            or lease.max_turns > request.max_turns
            or lease.max_effects > request.max_effects
            or lease.max_compute_ms > request.max_compute_ms
            or lease.max_proactive_messages > request.max_proactive_messages
            or not set(lease.allowed_actions).issubset(request.allowed_actions)
        ):
            raise ValueError("wake lease exceeds or mismatches its request")
        lease.require_active(now_ms)
        return self._insert(
            self._wake_leases,
            lease.lease_id,
            lease,
            _DOMAIN_WAKE_LEASE,
        )

    def require_active_lease(self, lease_id: str, *, now_ms: int) -> WakeLeaseV1:
        lease = self._wake_leases.get(lease_id)
        if lease is None:
            raise ValueError("unknown wake lease")
        lease.require_active(now_ms)
        return lease

    def save_cognition_snapshot(
        self, snapshot: CognitionSnapshotV1
    ) -> ReceiptEntryV1:
        _require_type(snapshot, CognitionSnapshotV1)
        lease = self.require_active_lease(
            snapshot.wake_lease_id, now_ms=snapshot.created_at_ms
        )
        registers = self._require_records(
            self._perspective_registers,
            snapshot.perspective_register_ids,
            "perspective register",
        )
        goals = self._require_records(
            self._standing_goals, snapshot.goal_ids, "standing goal", allow_empty=True
        )
        memories = self._require_records(
            self._memories, snapshot.memory_ids, "memory", allow_empty=True
        )
        questions = self._require_records(
            self._question_candidates,
            snapshot.unresolved_question_refs,
            "unresolved question",
            allow_empty=True,
        )
        projections = self._require_records(
            self._graph_projections,
            snapshot.graph_projection_ids,
            "graph projection",
            allow_empty=True,
        )
        if (
            lease.scope_id != snapshot.scope_id
            or any(
                item.scope_id != snapshot.scope_id
                for item in (*registers, *goals, *questions, *memories, *projections)
            )
            or any(not item.live for item in memories)
            or any(not item.live for item in projections)
        ):
            raise ValueError("snapshot crosses its active scope or includes tombstones")
        if snapshot.ledger_head != self.receipt_head:
            raise ValueError("snapshot does not bind the current receipt head")
        return self._insert(
            self._cognition_snapshots,
            snapshot.snapshot_id,
            snapshot,
            _DOMAIN_SNAPSHOT,
        )

    def record_proactive_message(
        self, message: ProactiveMessageV1
    ) -> ReceiptEntryV1:
        _require_type(message, ProactiveMessageV1)
        lease = self.require_active_lease(
            message.wake_lease_id, now_ms=message.created_at_ms
        )
        goal = self._standing_goals.get(message.goal_id)
        if (
            goal is None
            or lease.goal_id != message.goal_id
            or goal.scope_id != message.scope_id
            or lease.scope_id != message.scope_id
        ):
            raise ValueError("proactive message crosses its goal or lease scope")
        message_count = sum(
            item.wake_lease_id == lease.lease_id
            for item in self._proactive_messages.values()
        )
        if message_count >= lease.max_proactive_messages:
            raise ValueError("proactive message exceeds its wake lease budget")
        return self._insert(
            self._proactive_messages,
            message.message_id,
            message,
            _DOMAIN_MESSAGE,
        )

    def _perspective_owner(self, scope_id: str, perspective_id: str) -> str:
        owners = {
            register.owner_id
            for register in self._perspective_registers.values()
            if register.scope_id == scope_id
            and perspective_id == register.register_id
        }
        if len(owners) != 1:
            raise ValueError("perspective is unknown or ambiguous in the scope")
        return owners.pop()

    @staticmethod
    def _require_records(
        records: Mapping[str, T],
        identifiers: tuple[str, ...],
        label: str,
        *,
        allow_empty: bool = False,
    ) -> tuple[T, ...]:
        if not identifiers and not allow_empty:
            raise ValueError(f"{label} references cannot be empty")
        try:
            return tuple(records[identifier] for identifier in identifiers)
        except KeyError as error:
            raise ValueError(f"unknown {label}: {error.args[0]}") from error

    def _insert(
        self,
        records: dict[str, T],
        identifier: str,
        value: T,
        domain: ReceiptDomain,
    ) -> ReceiptEntryV1:
        if identifier in records:
            raise ValueError("record ID already exists")
        mapping = value.mapping()
        records[identifier] = value
        receipt = self._append_receipt(domain, identifier, mapping)
        try:
            self._commit()
        except Exception:
            del records[identifier]
            self._receipts.pop()
            raise
        return receipt

    def _replace_record(
        self,
        records: dict[str, T],
        identifier: str,
        value: T,
        domain: ReceiptDomain,
    ) -> ReceiptEntryV1:
        previous = records[identifier]
        records[identifier] = value
        receipt = self._append_receipt(domain, identifier, value.mapping())
        try:
            self._commit()
        except Exception:
            records[identifier] = previous
            self._receipts.pop()
            raise
        return receipt

    def _append_receipt(
        self,
        domain: ReceiptDomain,
        record_id: str,
        mapping: Mapping[str, object],
    ) -> ReceiptEntryV1:
        receipt = ReceiptEntryV1.create(
            index=len(self._receipts),
            domain=domain,
            record_id=record_id,
            record_mapping=mapping,
            previous_digest=self.receipt_head,
        )
        self._receipts.append(receipt)
        return receipt

    def _commit(self) -> None:
        self._revision += 1
        try:
            self._persist()
        except Exception:
            self._revision -= 1
            raise

    def _state_mapping(self) -> dict[str, object]:
        return {
            "protocol": MOIRA_KERNEL_PROTOCOL,
            "schema_version": MOIRA_KERNEL_STATE_VERSION,
            "revision": self._revision,
            "capture_enabled": self._capture_enabled,
            "modules": [item.mapping() for item in KERNEL_MODULE_IDENTITIES],
            "perspective_registers": _records_mapping(self._perspective_registers),
            "consents": _records_mapping(self._consents),
            "artifacts": _records_mapping(self._artifacts),
            "interactions": _records_mapping(self._interactions),
            "evidence": _records_mapping(self._evidence),
            "learning_records": _records_mapping(self._learning_records),
            "memories": _records_mapping(self._memories),
            "graph_projections": _records_mapping(self._graph_projections),
            "standing_goals": _records_mapping(self._standing_goals),
            "question_candidates": _records_mapping(self._question_candidates),
            "wake_requests": _records_mapping(self._wake_requests),
            "wake_leases": _records_mapping(self._wake_leases),
            "cognition_snapshots": _records_mapping(self._cognition_snapshots),
            "proactive_messages": _records_mapping(self._proactive_messages),
            "receipts": [item.mapping() for item in self._receipts],
        }

    def _persist(self) -> None:
        encoded = (
            json.dumps(
                self._state_mapping(),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        if len(encoded) > MAX_KERNEL_STATE_BYTES:
            raise ValueError("MOIRA kernel state exceeds its byte budget")
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{self.state_path.name}.",
                suffix=".tmp",
                dir=self.state_path.parent,
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.state_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _load(self) -> None:
        value = _read_state(self.state_path)
        if set(value) != _STATE_FIELDS:
            raise ValueError("MOIRA kernel state fields do not match its schema")
        if (
            value["protocol"] != MOIRA_KERNEL_PROTOCOL
            or value["schema_version"] != MOIRA_KERNEL_STATE_VERSION
            or type(value["revision"]) is not int
            or value["revision"] < 0
            or type(value["capture_enabled"]) is not bool
        ):
            raise ValueError("MOIRA kernel state identity is invalid")
        modules = tuple(
            KernelModuleIdentityV1.from_mapping(item)
            for item in _sequence(value["modules"], "kernel modules")
        )
        if modules != KERNEL_MODULE_IDENTITIES:
            raise ValueError("MOIRA kernel module identities changed")

        self._revision = value["revision"]
        self._capture_enabled = value["capture_enabled"]
        self._perspective_registers = _decode_records(
            value["perspective_registers"],
            PerspectiveRegisterV1.from_mapping,
            "register_id",
        )
        self._consents = _decode_records(
            value["consents"], ConsentScopeV1.from_mapping, "consent_id"
        )
        self._artifacts = _decode_records(
            value["artifacts"], ArtifactEnvelopeV1.from_mapping, "artifact_id"
        )
        self._interactions = _decode_records(
            value["interactions"], InteractionEventV1.from_mapping, "event_id"
        )
        self._evidence = _decode_records(
            value["evidence"], EvidenceReceiptV1.from_mapping, "evidence_id"
        )
        self._learning_records = _decode_records(
            value["learning_records"], LearningRecordV1.from_mapping, "learning_id"
        )
        self._memories = _decode_records(
            value["memories"], MemoryRecordV1.from_mapping, "memory_id"
        )
        self._graph_projections = _decode_records(
            value["graph_projections"],
            GraphProjectionV1.from_mapping,
            "projection_id",
        )
        self._standing_goals = _decode_records(
            value["standing_goals"], StandingGoalV1.from_mapping, "goal_id"
        )
        self._question_candidates = _decode_records(
            value["question_candidates"],
            QuestionCandidateV1.from_mapping,
            "question_id",
        )
        self._wake_requests = _decode_records(
            value["wake_requests"], WakeRequestV1.from_mapping, "request_id"
        )
        self._wake_leases = _decode_records(
            value["wake_leases"], WakeLeaseV1.from_mapping, "lease_id"
        )
        self._cognition_snapshots = _decode_records(
            value["cognition_snapshots"],
            CognitionSnapshotV1.from_mapping,
            "snapshot_id",
        )
        self._proactive_messages = _decode_records(
            value["proactive_messages"],
            ProactiveMessageV1.from_mapping,
            "message_id",
        )
        self._receipts = [
            ReceiptEntryV1.from_mapping(item)
            for item in _sequence(value["receipts"], "kernel receipts")
        ]
        self._verify_receipt_chain()
        self._verify_current_records()

    def _verify_receipt_chain(self) -> None:
        previous = _GENESIS
        for index, receipt in enumerate(self._receipts):
            if receipt.domain not in _IMPLEMENTED_RECEIPT_DOMAINS:
                raise ValueError("MOIRA kernel receipt domain is not implemented")
            if receipt.index != index or receipt.previous_digest != previous:
                raise ValueError("MOIRA kernel receipt chain is invalid")
            receipt.verify()
            previous = receipt.receipt_digest

    def _verify_current_records(self) -> None:
        latest = {(item.domain, item.record_id): item for item in self._receipts}
        records = (
            (_DOMAIN_PERSPECTIVE, self._perspective_registers),
            (_DOMAIN_CONSENT, self._consents),
            (_DOMAIN_ARTIFACT, self._artifacts),
            (_DOMAIN_INTERACTION, self._interactions),
            (_DOMAIN_EVIDENCE, self._evidence),
            (_DOMAIN_LEARNING, self._learning_records),
            (_DOMAIN_MEMORY, self._memories),
            (_DOMAIN_PROJECTION, self._graph_projections),
            (_DOMAIN_GOAL, self._standing_goals),
            (_DOMAIN_QUESTION, self._question_candidates),
            (_DOMAIN_WAKE_REQUEST, self._wake_requests),
            (_DOMAIN_WAKE_LEASE, self._wake_leases),
            (_DOMAIN_SNAPSHOT, self._cognition_snapshots),
            (_DOMAIN_MESSAGE, self._proactive_messages),
        )
        for domain, bucket in records:
            for identifier, record in bucket.items():
                receipt = latest.get((domain, identifier))
                expected = domain_digest(f"{domain.value}.record", record.mapping())
                if receipt is None or receipt.record_digest != expected:
                    raise ValueError("MOIRA kernel state is not bound to its receipts")
        capture_receipt = latest.get((_DOMAIN_CAPTURE, "capture-policy"))
        if capture_receipt is None:
            if self._capture_enabled:
                raise ValueError("enabled capture has no policy receipt")
        elif capture_receipt.record_digest != domain_digest(
            f"{_DOMAIN_CAPTURE.value}.record",
            {"capture_enabled": self._capture_enabled},
        ):
            raise ValueError("capture policy is not bound to its receipt")


def _records_mapping(records: Mapping[str, Any]) -> dict[str, object]:
    return {key: records[key].mapping() for key in sorted(records)}


def _decode_records(
    value: object,
    decoder: Callable[[object], T],
    identifier_field: str,
) -> dict[str, T]:
    if not isinstance(value, Mapping):
        raise ValueError("MOIRA kernel record bucket must be a mapping")
    records: dict[str, T] = {}
    for identifier in sorted(value):
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("MOIRA kernel record ID must be non-empty text")
        record = decoder(value[identifier])
        if getattr(record, identifier_field) != identifier:
            raise ValueError("MOIRA kernel record key does not match its value")
        records[identifier] = record
    return records


def _read_state(path: Path) -> dict[str, Any]:
    try:
        status = path.lstat()
    except OSError as error:
        raise ValueError("MOIRA kernel state cannot be read") from error
    if path.is_symlink() or not stat.S_ISREG(status.st_mode):
        raise ValueError("MOIRA kernel state must be a regular non-symlink file")
    if not 0 < status.st_size <= MAX_KERNEL_STATE_BYTES:
        raise ValueError("MOIRA kernel state exceeds its byte budget")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("MOIRA kernel state cannot be decoded") from error
    if not isinstance(value, dict):
        raise ValueError("MOIRA kernel state must be a JSON object")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("MOIRA kernel state contains duplicate JSON keys")
        value[key] = item
    return value


def _validate_state_parent(path: Path) -> None:
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("MOIRA kernel state parent must be a non-symlink directory")
    if os.path.lexists(path):
        try:
            status = path.lstat()
        except OSError as error:
            raise ValueError("MOIRA kernel state is unavailable") from error
        if path.is_symlink() or not stat.S_ISREG(status.st_mode):
            raise ValueError("MOIRA kernel state must be a regular non-symlink file")


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list")
    return tuple(value)


def _learning_use(destination: LearningDestination) -> ConsentUse:
    return {
        LearningDestination.MEMORY_ONLY: ConsentUse.MEMORY,
        LearningDestination.PERSONAL_ADAPTER: ConsentUse.PERSONAL_TRAINING,
        LearningDestination.PROJECT_ADAPTER: ConsentUse.PROJECT_TRAINING,
    }[destination]


def _require_type(value: object, expected: type[object]) -> None:
    if not isinstance(value, expected):
        raise ValueError(f"record must be {expected.__name__}")


def _sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


__all__ = [
    "KERNEL_MODULE_IDENTITIES",
    "KERNEL_MODULE_NAMES",
    "KernelModuleIdentityV1",
    "MAX_KERNEL_STATE_BYTES",
    "MOIRA_KERNEL_PROTOCOL",
    "MOIRA_KERNEL_STATE_VERSION",
    "MoiraKernel",
    "ReceiptEntryV1",
    "ReceiptDomain",
]
