"""Bounded MOIRA-to-PANDORA agent protocol.

This module constructs proposals and authorization envelopes. It deliberately
contains no tool executor: external effects remain PANDORA's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Protocol

from .masks import CognitiveMode


BRIDGE_PROTOCOL = "moira-pandora-agent-v1"
_MAX_EFFECT_PAYLOAD_BYTES = 16 * 1024 * 1024


class AgentRole(str, Enum):
    PLANNER = "planner"
    ANALYST = "analyst"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    EVALUATOR = "evaluator"


class DecisionStatus(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class AgentLease:
    lease_id: str
    task_id: str
    scope_id: str
    mode: CognitiveMode
    role: AgentRole
    max_turns: int
    max_effects: int
    allowed_actions: tuple[str, ...]


@dataclass(frozen=True)
class PandoraEffectRequest:
    lease_id: str
    task_id: str
    scope_id: str
    mode: CognitiveMode
    role: AgentRole
    capability: str
    operation: str
    target_kind: str
    target: str
    payload_digest: str
    request_digest: str

    def authorization_payload(self) -> dict[str, object]:
        return {
            "bridge_protocol": BRIDGE_PROTOCOL,
            "lease_id": self.lease_id,
            "task_id": self.task_id,
            "scope_id": self.scope_id,
            "mode": self.mode.value,
            "role": self.role.value,
            "capability": self.capability,
            "operation": self.operation,
            "target_kind": self.target_kind,
            "target": self.target,
            "payload_digest": self.payload_digest,
        }


@dataclass(frozen=True)
class PandoraDecision:
    request_digest: str
    status: DecisionStatus
    policy_version: int
    reason: str
    permit_id: str | None = None


@dataclass(frozen=True)
class PandoraEffectReceipt:
    request_digest: str
    permit_id: str
    receipt_id: str
    outcome: str


@dataclass(frozen=True)
class AuthorizedEffect:
    """A PANDORA-authorized envelope, not proof that an effect was executed."""

    request: PandoraEffectRequest
    permit_id: str
    policy_version: int


@dataclass(frozen=True)
class EffectOutcome:
    request: PandoraEffectRequest
    decision: PandoraDecision
    authorized: AuthorizedEffect | None


@dataclass(frozen=True)
class AgentTraceEvent:
    index: int
    kind: str
    lease_id: str
    detail_digest: str
    chain_digest: str


class PandoraGateway(Protocol):
    """Adapter implemented by the PANDORA-AGENT integration."""

    def decide(self, request: PandoraEffectRequest) -> PandoraDecision: ...


@dataclass
class _LeaseState:
    lease: AgentLease
    turns_used: int = 0
    effects_used: int = 0
    closed: bool = False


class PandoraAgentLayer:
    """Assign bounded roles and submit exact effect requests to PANDORA."""

    def __init__(self, gateway: PandoraGateway) -> None:
        self._gateway = gateway
        self._leases: dict[str, _LeaseState] = {}
        self._trace: list[AgentTraceEvent] = []
        self._receipted_requests: set[str] = set()

    def open_lease(
        self,
        *,
        lease_id: str,
        task_id: str,
        scope_id: str,
        mode: CognitiveMode,
        role: AgentRole,
        max_turns: int,
        max_effects: int,
        allowed_actions: tuple[str, ...] = (),
    ) -> AgentLease:
        for field, value in (
            ("lease_id", lease_id),
            ("task_id", task_id),
            ("scope_id", scope_id),
        ):
            _validate_text(field, value)
        if not isinstance(mode, CognitiveMode):
            raise ValueError("mode must be a CognitiveMode")
        if not isinstance(role, AgentRole):
            raise ValueError("role must be an AgentRole")
        if (
            not isinstance(max_turns, int)
            or isinstance(max_turns, bool)
            or max_turns <= 0
        ):
            raise ValueError("max_turns must be a positive integer")
        if (
            not isinstance(max_effects, int)
            or isinstance(max_effects, bool)
            or max_effects < 0
        ):
            raise ValueError("max_effects must be a non-negative integer")
        if lease_id in self._leases:
            raise ValueError("lease_id already exists")
        normalized_actions = tuple(sorted(set(allowed_actions)))
        if any(not _valid_action(action) for action in normalized_actions):
            raise ValueError("allowed_actions must use non-empty capability:operation pairs")
        lease = AgentLease(
            lease_id=lease_id,
            task_id=task_id,
            scope_id=scope_id,
            mode=mode,
            role=role,
            max_turns=max_turns,
            max_effects=max_effects,
            allowed_actions=normalized_actions,
        )
        self._leases[lease_id] = _LeaseState(lease)
        self._append_trace("lease_opened", lease_id, _canonical_bytes(_lease_dict(lease)))
        return lease

    def record_turn(self, lease_id: str, proposal: str) -> int:
        state = self._active_lease(lease_id)
        _validate_proposal(proposal)
        if state.turns_used >= state.lease.max_turns:
            raise ValueError("agent turn budget is exhausted")
        state.turns_used += 1
        self._append_trace("turn_recorded", lease_id, proposal.encode("utf-8"))
        return state.turns_used

    def request_effect(
        self,
        lease_id: str,
        *,
        capability: str,
        operation: str,
        target_kind: str,
        target: str,
        payload: bytes = b"",
    ) -> EffectOutcome:
        state = self._active_lease(lease_id)
        if state.turns_used == 0:
            raise ValueError("an agent turn must be recorded before requesting an effect")
        if state.effects_used >= state.lease.max_effects:
            raise ValueError("agent effect budget is exhausted")
        for field, value in (
            ("capability", capability),
            ("operation", operation),
            ("target_kind", target_kind),
            ("target", target),
        ):
            _validate_text(field, value)
        if not isinstance(payload, bytes):
            raise ValueError("payload must be bytes")
        if len(payload) > _MAX_EFFECT_PAYLOAD_BYTES:
            raise ValueError("payload exceeds the effect request size budget")
        action = f"{capability}:{operation}"
        if action not in state.lease.allowed_actions:
            raise ValueError("effect is outside the lease action set")

        request = _effect_request(
            state.lease,
            capability=capability,
            operation=operation,
            target_kind=target_kind,
            target=target,
            payload=payload,
        )
        state.effects_used += 1
        self._append_trace(
            "effect_requested", lease_id, request.request_digest.encode("ascii")
        )
        decision = self._gateway.decide(request)
        _validate_decision(request, decision)
        authorized = None
        if decision.status is DecisionStatus.ALLOW:
            assert decision.permit_id is not None
            authorized = AuthorizedEffect(
                request=request,
                permit_id=decision.permit_id,
                policy_version=decision.policy_version,
            )
        self._append_trace(
            f"effect_{decision.status.value}",
            lease_id,
            _canonical_bytes(_decision_dict(decision)),
        )
        return EffectOutcome(request, decision, authorized)

    def record_receipt(
        self, authorized: AuthorizedEffect, receipt: PandoraEffectReceipt
    ) -> None:
        if not isinstance(authorized, AuthorizedEffect) or not isinstance(
            receipt, PandoraEffectReceipt
        ):
            raise ValueError("authorized effect and PANDORA receipt are required")
        if receipt.request_digest != authorized.request.request_digest:
            raise ValueError("PANDORA receipt does not match the effect request")
        if receipt.permit_id != authorized.permit_id:
            raise ValueError("PANDORA receipt does not match the effect permit")
        _validate_text("PANDORA receipt_id", receipt.receipt_id)
        if receipt.outcome not in ("succeeded", "failed"):
            raise ValueError("PANDORA receipt outcome must be succeeded or failed")
        if receipt.request_digest in self._receipted_requests:
            raise ValueError("PANDORA receipt was already recorded")
        self._receipted_requests.add(receipt.request_digest)
        self._append_trace(
            "effect_completed",
            authorized.request.lease_id,
            _canonical_bytes(
                {
                    "request_digest": receipt.request_digest,
                    "permit_id": receipt.permit_id,
                    "receipt_id": receipt.receipt_id,
                    "outcome": receipt.outcome,
                }
            ),
        )

    def close_lease(self, lease_id: str) -> None:
        state = self._active_lease(lease_id)
        state.closed = True
        self._append_trace("lease_closed", lease_id, b"")

    def trace(self) -> tuple[AgentTraceEvent, ...]:
        return tuple(self._trace)

    def _active_lease(self, lease_id: str) -> _LeaseState:
        state = self._leases.get(lease_id)
        if state is None:
            raise ValueError("unknown agent lease")
        if state.closed:
            raise ValueError("agent lease is closed")
        return state

    def _append_trace(self, kind: str, lease_id: str, detail: bytes) -> None:
        detail_digest = _digest("moira-agent-detail-v1", detail)
        previous = self._trace[-1].chain_digest if self._trace else "genesis"
        material = _canonical_bytes(
            {
                "index": len(self._trace),
                "kind": kind,
                "lease_id": lease_id,
                "detail_digest": detail_digest,
                "previous": previous,
            }
        )
        self._trace.append(
            AgentTraceEvent(
                index=len(self._trace),
                kind=kind,
                lease_id=lease_id,
                detail_digest=detail_digest,
                chain_digest=_digest("moira-agent-trace-v1", material),
            )
        )


def agent_contract() -> dict[str, object]:
    return {
        "model": "MOIRA",
        "integration": "PANDORA-AGENT",
        "bridge_protocol": BRIDGE_PROTOCOL,
        "orientations": [mode.value for mode in CognitiveMode],
        "roles": [role.value for role in AgentRole],
        "any_orientation_can_hold_any_role": True,
        "role_leases_are_bounded": True,
        "request_digest_binding": True,
        "external_authority": [
            "ShadowCouncil",
            "Parliament",
            "ReferenceMonitor",
            "EffectExecutor",
        ],
        "direct_effect_execution": False,
        "model_can_authorize_its_own_effects": False,
        "effect_receipt_required_from_pandora": True,
    }


def _effect_request(
    lease: AgentLease,
    *,
    capability: str,
    operation: str,
    target_kind: str,
    target: str,
    payload: bytes,
) -> PandoraEffectRequest:
    payload_digest = _digest("moira-agent-payload-v1", payload)
    fields: dict[str, object] = {
        "bridge_protocol": BRIDGE_PROTOCOL,
        "lease_id": lease.lease_id,
        "task_id": lease.task_id,
        "scope_id": lease.scope_id,
        "mode": lease.mode.value,
        "role": lease.role.value,
        "capability": capability,
        "operation": operation,
        "target_kind": target_kind,
        "target": target,
        "payload_digest": payload_digest,
    }
    request_digest = _digest("moira-pandora-request-v1", _canonical_bytes(fields))
    return PandoraEffectRequest(
        lease_id=lease.lease_id,
        task_id=lease.task_id,
        scope_id=lease.scope_id,
        mode=lease.mode,
        role=lease.role,
        capability=capability,
        operation=operation,
        target_kind=target_kind,
        target=target,
        payload_digest=payload_digest,
        request_digest=request_digest,
    )


def _validate_decision(
    request: PandoraEffectRequest, decision: PandoraDecision
) -> None:
    if not isinstance(decision, PandoraDecision):
        raise ValueError("PANDORA gateway returned an invalid decision")
    if decision.request_digest != request.request_digest:
        raise ValueError("PANDORA decision does not match the effect request")
    if not isinstance(decision.status, DecisionStatus):
        raise ValueError("PANDORA decision status is invalid")
    if (
        not isinstance(decision.policy_version, int)
        or isinstance(decision.policy_version, bool)
        or decision.policy_version <= 0
    ):
        raise ValueError("PANDORA policy version must be a positive integer")
    _validate_text("PANDORA decision reason", decision.reason)
    if decision.status is DecisionStatus.ALLOW:
        if decision.permit_id is None:
            raise ValueError("allowed PANDORA decision requires a permit_id")
        _validate_text("PANDORA permit_id", decision.permit_id)
    elif decision.permit_id is not None:
        raise ValueError("non-allow PANDORA decision cannot carry a permit_id")


def _validate_text(field: str, value: object, *, maximum: int = 512) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{field} must be bounded non-control text")


def _validate_proposal(value: object) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > 1_000_000
        or "\x00" in value
    ):
        raise ValueError("proposal must be bounded non-empty text")


def _valid_action(action: object) -> bool:
    if not isinstance(action, str) or action.count(":") != 1:
        return False
    capability, operation = action.split(":")
    try:
        _validate_text("capability", capability)
        _validate_text("operation", operation)
    except ValueError:
        return False
    return True


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(domain: str, value: bytes) -> str:
    hasher = hashlib.sha256()
    hasher.update(domain.encode("ascii"))
    hasher.update(b"\x00")
    hasher.update(value)
    return f"sha256:{hasher.hexdigest()}"


def _lease_dict(lease: AgentLease) -> dict[str, object]:
    return {
        "lease_id": lease.lease_id,
        "task_id": lease.task_id,
        "scope_id": lease.scope_id,
        "mode": lease.mode.value,
        "role": lease.role.value,
        "max_turns": lease.max_turns,
        "max_effects": lease.max_effects,
        "allowed_actions": list(lease.allowed_actions),
    }


def _decision_dict(decision: PandoraDecision) -> dict[str, object]:
    return {
        "request_digest": decision.request_digest,
        "status": decision.status.value,
        "policy_version": decision.policy_version,
        "reason": decision.reason,
        "permit_id": decision.permit_id,
    }


__all__ = [
    "BRIDGE_PROTOCOL",
    "AgentLease",
    "AgentRole",
    "AgentTraceEvent",
    "AuthorizedEffect",
    "DecisionStatus",
    "EffectOutcome",
    "PandoraAgentLayer",
    "PandoraDecision",
    "PandoraEffectReceipt",
    "PandoraEffectRequest",
    "PandoraGateway",
    "agent_contract",
]
