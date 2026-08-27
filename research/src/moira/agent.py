"""MOIRA's native bounded agent-control layer.

The layer belongs to the whole model. Task-selected sisters can plan, analyze,
execute, review, or evaluate through the same policy and tool boundary.
Optional host harnesses may restrict this layer but cannot replace MOIRA's
cognition or expand its authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import json
import secrets
from typing import Callable, Protocol

from .context import CONTEXT_PROTOCOL, MAX_CONTEXT_TOKENS
from .masks import CognitiveMode


AGENT_PROTOCOL = "moira-agent-protocol-v1"
MAX_EXTERNAL_CONTEXT_PRIORITY = 60
CONTROL_KERNEL = "moira-agent-kernel-v1"
_MAX_EFFECT_PAYLOAD_BYTES = 16 * 1024 * 1024
_MAX_TOOL_OUTPUT_BYTES = 16 * 1024 * 1024


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
class AgentEffectRequest:
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
    payload_size: int = 0

    def authorization_payload(self) -> dict[str, object]:
        return {
            "agent_protocol": AGENT_PROTOCOL,
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
            "payload_size": self.payload_size,
        }


@dataclass(frozen=True)
class AgentDecision:
    request_digest: str
    status: DecisionStatus
    policy_version: int
    reason: str
    permit_id: str | None = None


@dataclass(frozen=True)
class AgentEffectReceipt:
    request_digest: str
    permit_id: str
    receipt_id: str
    outcome: str


@dataclass(frozen=True)
class AuthorizedEffect:
    """A MOIRA-authorized envelope, not proof that an effect was executed."""

    request: AgentEffectRequest
    permit_id: str
    policy_version: int


@dataclass(frozen=True)
class EffectOutcome:
    request: AgentEffectRequest
    decision: AgentDecision
    authorized: AuthorizedEffect | None


@dataclass(frozen=True)
class AgentTraceEvent:
    index: int
    kind: str
    lease_id: str
    detail_digest: str
    chain_digest: str


class AgentAuthority(Protocol):
    """Decision component used inside the native control kernel."""

    def decide(self, request: AgentEffectRequest) -> AgentDecision: ...

    def approve(
        self, request: AgentEffectRequest, approval_id: str
    ) -> AgentDecision: ...


@dataclass
class _LeaseState:
    lease: AgentLease
    turns_used: int = 0
    effects_used: int = 0
    closed: bool = False


class MoiraAgentControl:
    """Low-level role, request, decision, and trace machinery."""

    def __init__(self, authority: AgentAuthority) -> None:
        self._authority = authority
        self._leases: dict[str, _LeaseState] = {}
        self._trace: list[AgentTraceEvent] = []
        self._requested_requests: set[str] = set()
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

    def lease(self, lease_id: str) -> AgentLease:
        """Return the exact active lease bound to one runtime invocation."""
        return self._active_lease(lease_id).lease

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
        if request.request_digest in self._requested_requests:
            raise ValueError("exact MOIRA effect request was already submitted")
        self._requested_requests.add(request.request_digest)
        state.effects_used += 1
        self._append_trace(
            "effect_requested", lease_id, request.request_digest.encode("ascii")
        )
        decision = self._authority.decide(request)
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

    def approve_effect(
        self, outcome: EffectOutcome, approval_id: str
    ) -> AuthorizedEffect:
        if not isinstance(outcome, EffectOutcome):
            raise ValueError("pending effect outcome is required")
        if outcome.decision.status is not DecisionStatus.REQUIRE_APPROVAL:
            raise ValueError("effect is not awaiting approval")
        _validate_text("approval_id", approval_id)
        decision = self._authority.approve(outcome.request, approval_id)
        _validate_decision(outcome.request, decision)
        if decision.status is not DecisionStatus.ALLOW:
            raise ValueError("approved effect did not receive an allow decision")
        assert decision.permit_id is not None
        authorized = AuthorizedEffect(
            request=outcome.request,
            permit_id=decision.permit_id,
            policy_version=decision.policy_version,
        )
        self._append_trace(
            "effect_approved",
            outcome.request.lease_id,
            _canonical_bytes(_decision_dict(decision)),
        )
        return authorized

    def record_receipt(
        self, authorized: AuthorizedEffect, receipt: AgentEffectReceipt
    ) -> None:
        if not isinstance(authorized, AuthorizedEffect) or not isinstance(
            receipt, AgentEffectReceipt
        ):
            raise ValueError("authorized effect and MOIRA receipt are required")
        if receipt.request_digest != authorized.request.request_digest:
            raise ValueError("MOIRA receipt does not match the effect request")
        if receipt.permit_id != authorized.permit_id:
            raise ValueError("MOIRA receipt does not match the effect permit")
        _validate_text("MOIRA receipt_id", receipt.receipt_id)
        if receipt.outcome not in ("succeeded", "failed"):
            raise ValueError("MOIRA receipt outcome must be succeeded or failed")
        if receipt.request_digest in self._receipted_requests:
            raise ValueError("MOIRA receipt was already recorded")
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


@dataclass(frozen=True)
class AgentPolicy:
    policy_version: int
    allowed_actions: frozenset[str]
    approval_actions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.policy_version, int)
            or isinstance(self.policy_version, bool)
            or self.policy_version <= 0
        ):
            raise ValueError("policy_version must be a positive integer")
        if not isinstance(self.allowed_actions, frozenset) or not isinstance(
            self.approval_actions, frozenset
        ):
            raise ValueError("policy actions must be frozensets")
        if any(not _valid_action(action) for action in self.allowed_actions):
            raise ValueError("allowed policy actions are invalid")
        if not self.approval_actions.issubset(self.allowed_actions):
            raise ValueError("approval actions must also be allowed actions")

    @classmethod
    def deny_all(cls) -> AgentPolicy:
        return cls(policy_version=1, allowed_actions=frozenset())


@dataclass(frozen=True)
class OperatorApproval:
    approval_id: str
    request_digest: str


class HostGuard(Protocol):
    """Optional harness restriction; it cannot expand a local decision."""

    def review(
        self, request: AgentEffectRequest, local_status: DecisionStatus
    ) -> DecisionStatus: ...


@dataclass(frozen=True)
class MoiraTool:
    tool_id: str
    capability: str
    operation: str
    target_kind: str
    handler: Callable[[str, bytes], bytes]
    description: str = ""

    def __post_init__(self) -> None:
        _validate_text("tool_id", self.tool_id)
        _validate_text("capability", self.capability)
        _validate_text("operation", self.operation)
        _validate_text("target_kind", self.target_kind)
        if not isinstance(self.description, str) or len(
            self.description.encode("utf-8")
        ) > 4_096:
            raise ValueError("tool description exceeds its size budget")
        if not callable(self.handler):
            raise ValueError("tool handler must be callable")

    @property
    def action(self) -> str:
        return f"{self.capability}:{self.operation}"


@dataclass(frozen=True)
class ToolExecution:
    receipt: AgentEffectReceipt
    output: bytes | None
    error_code: str | None


@dataclass(frozen=True)
class ToolDescriptor:
    tool_id: str
    capability: str
    operation: str
    target_kind: str
    description: str


class _NativeControlAuthority:
    def __init__(
        self,
        policy: AgentPolicy,
        permit_secret: bytes,
        host_guard: HostGuard | None,
    ) -> None:
        if not isinstance(policy, AgentPolicy):
            raise ValueError("policy must be an AgentPolicy")
        if not isinstance(permit_secret, bytes) or len(permit_secret) < 32:
            raise ValueError("permit_secret must contain at least 32 bytes")
        self.policy = policy
        self._secret = permit_secret
        self._host_guard = host_guard
        self._next_permit = 1
        self._issued: dict[str, str] = {}
        self._pending: dict[str, AgentEffectRequest] = {}

    def decide(self, request: AgentEffectRequest) -> AgentDecision:
        action = f"{request.capability}:{request.operation}"
        if action not in self.policy.allowed_actions:
            status = DecisionStatus.DENY
            reason = "action is not allowed by the active MOIRA policy"
        elif action in self.policy.approval_actions:
            status = DecisionStatus.REQUIRE_APPROVAL
            reason = "action requires operator approval"
        else:
            status = DecisionStatus.ALLOW
            reason = "action is allowed by the active MOIRA policy"

        status = self._apply_host_guard(request, status)
        permit_id = self._issue(request) if status is DecisionStatus.ALLOW else None
        if status is DecisionStatus.REQUIRE_APPROVAL:
            self._pending[request.request_digest] = request
        if status is DecisionStatus.DENY and action in self.policy.allowed_actions:
            reason = "optional host policy restricted the local MOIRA decision"
        elif (
            status is DecisionStatus.REQUIRE_APPROVAL
            and action not in self.policy.approval_actions
        ):
            reason = "optional host policy requires operator approval"
        return AgentDecision(
            request_digest=request.request_digest,
            status=status,
            policy_version=self.policy.policy_version,
            reason=reason,
            permit_id=permit_id,
        )

    def approve(
        self, request: AgentEffectRequest, approval_id: str
    ) -> AgentDecision:
        pending = self._pending.get(request.request_digest)
        if pending != request:
            raise ValueError("effect is not pending exact operator approval")
        _validate_text("approval_id", approval_id)
        del self._pending[request.request_digest]
        return AgentDecision(
            request_digest=request.request_digest,
            status=DecisionStatus.ALLOW,
            policy_version=self.policy.policy_version,
            reason="operator approval consumed",
            permit_id=self._issue(request),
        )

    def consume(self, authorized: AuthorizedEffect) -> None:
        expected = self._issued.pop(authorized.permit_id, None)
        if expected is None:
            raise ValueError("effect permit is unknown or already consumed")
        if expected != authorized.request.request_digest:
            raise ValueError("effect permit is bound to a different request")
        if authorized.policy_version != self.policy.policy_version:
            raise ValueError("effect permit policy version is stale")

    def _apply_host_guard(
        self, request: AgentEffectRequest, local: DecisionStatus
    ) -> DecisionStatus:
        if self._host_guard is None:
            return local
        try:
            reviewed = self._host_guard.review(request, local)
        except Exception:
            return DecisionStatus.DENY
        if not isinstance(reviewed, DecisionStatus):
            return DecisionStatus.DENY
        if local is DecisionStatus.DENY:
            return DecisionStatus.DENY
        if local is DecisionStatus.REQUIRE_APPROVAL and reviewed is DecisionStatus.ALLOW:
            return DecisionStatus.REQUIRE_APPROVAL
        return reviewed

    def _issue(self, request: AgentEffectRequest) -> str:
        nonce = self._next_permit
        self._next_permit += 1
        material = (
            f"{CONTROL_KERNEL}:{self.policy.policy_version}:{nonce}:"
            f"{request.request_digest}"
        ).encode("utf-8")
        signature = hmac.new(self._secret, material, hashlib.sha256).hexdigest()
        permit_id = f"moira-permit-v1:{nonce}:{signature}"
        self._issued[permit_id] = request.request_digest
        return permit_id


class MoiraAgentKernel:
    """Native MOIRA agent authority, tool registry, executor, and receipt ledger."""

    def __init__(
        self,
        policy: AgentPolicy | None = None,
        *,
        permit_secret: bytes | None = None,
        host_guard: HostGuard | None = None,
    ) -> None:
        active_policy = policy if policy is not None else AgentPolicy.deny_all()
        authority = _NativeControlAuthority(
            active_policy,
            permit_secret if permit_secret is not None else secrets.token_bytes(32),
            host_guard,
        )
        self._authority = authority
        self._layer = MoiraAgentControl(authority)
        self._tools: dict[str, MoiraTool] = {}
        self._payloads: dict[str, bytes] = {}
        self._next_receipt = 1

    def register_tool(self, tool: MoiraTool) -> None:
        if not isinstance(tool, MoiraTool):
            raise ValueError("tool must be a MoiraTool")
        if tool.tool_id in self._tools:
            raise ValueError("tool_id is already registered")
        self._tools[tool.tool_id] = tool

    def tool_catalog(self) -> tuple[ToolDescriptor, ...]:
        return tuple(
            ToolDescriptor(
                tool_id=tool.tool_id,
                capability=tool.capability,
                operation=tool.operation,
                target_kind=tool.target_kind,
                description=tool.description,
            )
            for tool in sorted(self._tools.values(), key=lambda item: item.tool_id)
        )

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
        return self._layer.open_lease(
            lease_id=lease_id,
            task_id=task_id,
            scope_id=scope_id,
            mode=mode,
            role=role,
            max_turns=max_turns,
            max_effects=max_effects,
            allowed_actions=allowed_actions,
        )

    def record_turn(self, lease_id: str, proposal: str) -> int:
        return self._layer.record_turn(lease_id, proposal)

    def lease(self, lease_id: str) -> AgentLease:
        return self._layer.lease(lease_id)

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
        outcome = self._layer.request_effect(
            lease_id,
            capability=capability,
            operation=operation,
            target_kind=target_kind,
            target=target,
            payload=payload,
        )
        self._payloads[outcome.request.request_digest] = payload
        return outcome

    def request_tool(
        self,
        lease_id: str,
        *,
        tool_id: str,
        target: str,
        payload: bytes = b"",
    ) -> EffectOutcome:
        tool = self._tools.get(tool_id)
        if tool is None:
            raise ValueError("tool is not registered")
        return self.request_effect(
            lease_id,
            capability=tool.capability,
            operation=tool.operation,
            target_kind=tool.target_kind,
            target=target,
            payload=payload,
        )

    def approve_effect(
        self, outcome: EffectOutcome, approval: OperatorApproval
    ) -> AuthorizedEffect:
        if not isinstance(approval, OperatorApproval):
            raise ValueError("operator approval is required")
        _validate_text("approval_id", approval.approval_id)
        if approval.request_digest != outcome.request.request_digest:
            raise ValueError("operator approval is bound to a different request")
        return self._layer.approve_effect(outcome, approval.approval_id)

    def execute_effect(
        self, authorized: AuthorizedEffect, tool_id: str
    ) -> ToolExecution:
        if not isinstance(authorized, AuthorizedEffect):
            raise ValueError("authorized effect is required")
        tool = self._tools.get(tool_id)
        if tool is None:
            raise ValueError("tool is not registered")
        request = authorized.request
        if (
            tool.action != f"{request.capability}:{request.operation}"
            or tool.target_kind != request.target_kind
        ):
            raise ValueError("tool does not match the authorized effect")
        payload = self._payloads.get(request.request_digest)
        if payload is None or _digest("moira-agent-payload-v1", payload) != request.payload_digest:
            raise ValueError("effect payload is unavailable or does not match")
        self._authority.consume(authorized)
        del self._payloads[request.request_digest]

        receipt_id = f"moira-receipt-v1:{self._next_receipt}"
        self._next_receipt += 1
        try:
            output = tool.handler(request.target, payload)
            if not isinstance(output, bytes):
                raise TypeError("tool output must be bytes")
            if len(output) > _MAX_TOOL_OUTPUT_BYTES:
                raise ValueError("tool output exceeds the result size budget")
            outcome = "succeeded"
            error_code = None
        except Exception as error:
            output = None
            outcome = "failed"
            error_code = type(error).__name__
        receipt = AgentEffectReceipt(
            request_digest=request.request_digest,
            permit_id=authorized.permit_id,
            receipt_id=receipt_id,
            outcome=outcome,
        )
        self._layer.record_receipt(authorized, receipt)
        return ToolExecution(receipt=receipt, output=output, error_code=error_code)

    def close_lease(self, lease_id: str) -> None:
        self._layer.close_lease(lease_id)

    def trace(self) -> tuple[AgentTraceEvent, ...]:
        return self._layer.trace()


def agent_contract() -> dict[str, object]:
    return {
        "model": "MOIRA",
        "control_kernel": CONTROL_KERNEL,
        "whole_model_agentic": True,
        "agent_layer_scope": "shared-across-task-selected-sisters",
        "external_agent_runtime_required": False,
        "external_model_required_for_agent_loop": False,
        "agent_protocol": AGENT_PROTOCOL,
        "context_protocol": CONTEXT_PROTOCOL,
        "context_scope_bound_to_lease": True,
        "context_receipt_binds_labels_and_content": True,
        "context_content_token_budget_maximum": MAX_CONTEXT_TOKENS,
        "context_full_request_preflight_implemented": True,
        "context_external_priority_ceiling": MAX_EXTERNAL_CONTEXT_PRIORITY,
        "context_latest_observation_required": True,
        "context_persistence_implemented": False,
        "context_compression_implemented": False,
        "orientations": [mode.value for mode in CognitiveMode],
        "roles": [role.value for role in AgentRole],
        "any_orientation_can_hold_any_role": True,
        "role_leases_are_bounded": True,
        "request_digest_binding": True,
        "internal_control_components": [
            "RoleScheduler",
            "ContextAssembler",
            "PolicyEngine",
            "ReferenceMonitor",
            "ToolRegistry",
            "ReceiptLedger",
        ],
        "optional_hosts": [
            "codex",
            "claude-code",
            "hermes-agent",
            "local",
        ],
        "host_can_expand_local_authority": False,
        "effects_require_registered_tools": True,
        "model_can_authorize_its_own_effects": False,
        "effect_receipt_required": True,
        "standalone_profiles": {
            "read-only": ["workspace.read", "workspace.search"],
            "developer": [
                "workspace.read",
                "workspace.search",
                "workspace.write",
                "process.run",
            ],
        },
        "developer_effects_require_operator_approval": True,
    }


def _effect_request(
    lease: AgentLease,
    *,
    capability: str,
    operation: str,
    target_kind: str,
    target: str,
    payload: bytes,
) -> AgentEffectRequest:
    payload_digest = _digest("moira-agent-payload-v1", payload)
    fields: dict[str, object] = {
        "agent_protocol": AGENT_PROTOCOL,
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
        "payload_size": len(payload),
    }
    request_digest = _digest("moira-agent-request-v1", _canonical_bytes(fields))
    return AgentEffectRequest(
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
        payload_size=len(payload),
    )


def _validate_decision(
    request: AgentEffectRequest, decision: AgentDecision
) -> None:
    if not isinstance(decision, AgentDecision):
        raise ValueError("MOIRA authority returned an invalid decision")
    if decision.request_digest != request.request_digest:
        raise ValueError("MOIRA decision does not match the effect request")
    if not isinstance(decision.status, DecisionStatus):
        raise ValueError("MOIRA decision status is invalid")
    if (
        not isinstance(decision.policy_version, int)
        or isinstance(decision.policy_version, bool)
        or decision.policy_version <= 0
    ):
        raise ValueError("MOIRA policy version must be a positive integer")
    _validate_text("MOIRA decision reason", decision.reason)
    if decision.status is DecisionStatus.ALLOW:
        if decision.permit_id is None:
            raise ValueError("allowed MOIRA decision requires a permit_id")
        _validate_text("MOIRA permit_id", decision.permit_id)
    elif decision.permit_id is not None:
        raise ValueError("non-allow MOIRA decision cannot carry a permit_id")


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


def _decision_dict(decision: AgentDecision) -> dict[str, object]:
    return {
        "request_digest": decision.request_digest,
        "status": decision.status.value,
        "policy_version": decision.policy_version,
        "reason": decision.reason,
        "permit_id": decision.permit_id,
    }


__all__ = [
    "AGENT_PROTOCOL",
    "MAX_EXTERNAL_CONTEXT_PRIORITY",
    "CONTROL_KERNEL",
    "AgentLease",
    "AgentPolicy",
    "AgentRole",
    "AgentTraceEvent",
    "AuthorizedEffect",
    "DecisionStatus",
    "EffectOutcome",
    "HostGuard",
    "MoiraAgentKernel",
    "MoiraTool",
    "OperatorApproval",
    "MoiraAgentControl",
    "AgentDecision",
    "AgentEffectReceipt",
    "AgentEffectRequest",
    "AgentAuthority",
    "ToolExecution",
    "ToolDescriptor",
    "agent_contract",
]
