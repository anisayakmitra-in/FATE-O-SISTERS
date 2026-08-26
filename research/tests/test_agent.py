from dataclasses import replace

import pytest

from moira.agent import (
    AgentPolicy,
    AgentRole,
    DecisionStatus,
    MoiraAgentKernel,
    MoiraTool,
    OperatorApproval,
    MoiraAgentControl,
    AgentDecision,
    AgentEffectReceipt,
    AgentEffectRequest,
    agent_contract,
)
from moira.masks import CognitiveMode


class RecordingGateway:
    def __init__(self, status: DecisionStatus = DecisionStatus.ALLOW) -> None:
        self.status = status
        self.requests: list[AgentEffectRequest] = []

    def decide(self, request: AgentEffectRequest) -> AgentDecision:
        self.requests.append(request)
        return AgentDecision(
            request_digest=request.request_digest,
            status=self.status,
            policy_version=7,
            reason="test policy decision",
            permit_id="permit-1" if self.status is DecisionStatus.ALLOW else None,
        )

    def approve(
        self, request: AgentEffectRequest, approval_id: str
    ) -> AgentDecision:
        return AgentDecision(
            request_digest=request.request_digest,
            status=DecisionStatus.ALLOW,
            policy_version=7,
            reason=f"approved {approval_id}",
            permit_id="permit-approved",
        )


@pytest.mark.parametrize("mode", list(CognitiveMode))
@pytest.mark.parametrize("role", list(AgentRole))
def test_any_orientation_can_hold_any_agent_role(
    mode: CognitiveMode, role: AgentRole
) -> None:
    layer = MoiraAgentControl(RecordingGateway())

    lease = layer.open_lease(
        lease_id=f"{mode.value}-{role.value}",
        task_id="task-1",
        scope_id="workspace-1",
        mode=mode,
        role=role,
        max_turns=2,
        max_effects=0,
    )

    assert lease.mode is mode
    assert lease.role is role


def test_allowed_effect_requires_exact_agent_permit_and_does_not_execute() -> None:
    gateway = RecordingGateway()
    layer = MoiraAgentControl(gateway)
    layer.open_lease(
        lease_id="cyber-lease",
        task_id="review-advisory",
        scope_id="workspace-1",
        mode=CognitiveMode.CLOTHO,
        role=AgentRole.ANALYST,
        max_turns=2,
        max_effects=1,
        allowed_actions=("filesystem_read:read",),
    )
    layer.record_turn("cyber-lease", "Inspect the bounded advisory fixture")

    outcome = layer.request_effect(
        "cyber-lease",
        capability="filesystem_read",
        operation="read",
        target_kind="path",
        target="workspace/advisory.json",
        payload=b'{"limit":4096}',
    )

    assert len(gateway.requests) == 1
    assert outcome.authorized is not None
    assert outcome.request.payload_size == len(b'{"limit":4096}')
    assert outcome.request.authorization_payload()["payload_size"] == len(
        b'{"limit":4096}'
    )
    assert outcome.authorized.permit_id == "permit-1"
    assert outcome.authorized.request == gateway.requests[0]
    assert not hasattr(layer, "execute")

    receipt = AgentEffectReceipt(
        request_digest=outcome.request.request_digest,
        permit_id=outcome.authorized.permit_id,
        receipt_id="receipt-1",
        outcome="succeeded",
    )
    layer.record_receipt(outcome.authorized, receipt)
    assert layer.trace()[-1].kind == "effect_completed"
    with pytest.raises(ValueError, match="already recorded"):
        layer.record_receipt(outcome.authorized, receipt)


@pytest.mark.parametrize(
    "status", (DecisionStatus.DENY, DecisionStatus.REQUIRE_APPROVAL)
)
def test_non_allow_decisions_never_create_an_authorized_effect(
    status: DecisionStatus,
) -> None:
    layer = MoiraAgentControl(RecordingGateway(status))
    layer.open_lease(
        lease_id="lease-1",
        task_id="task-1",
        scope_id="workspace-1",
        mode=CognitiveMode.ATROPOS,
        role=AgentRole.EXECUTOR,
        max_turns=1,
        max_effects=1,
        allowed_actions=("process_execute:execute",),
    )
    layer.record_turn("lease-1", "Propose the bounded verification command")

    outcome = layer.request_effect(
        "lease-1",
        capability="process_execute",
        operation="execute",
        target_kind="process",
        target="pytest",
    )

    assert outcome.decision.status is status
    assert outcome.authorized is None


def test_lease_action_turn_and_effect_budgets_fail_closed() -> None:
    gateway = RecordingGateway()
    layer = MoiraAgentControl(gateway)
    layer.open_lease(
        lease_id="lease-1",
        task_id="task-1",
        scope_id="workspace-1",
        mode=CognitiveMode.LACHESIS,
        role=AgentRole.PLANNER,
        max_turns=1,
        max_effects=1,
        allowed_actions=("filesystem_read:read",),
    )

    with pytest.raises(ValueError, match="turn must be recorded"):
        layer.request_effect(
            "lease-1",
            capability="filesystem_read",
            operation="read",
            target_kind="path",
            target="workspace/input.txt",
        )
    layer.record_turn("lease-1", "Read only the scoped input")
    with pytest.raises(ValueError, match="turn budget"):
        layer.record_turn("lease-1", "Try another turn")
    with pytest.raises(ValueError, match="outside the lease"):
        layer.request_effect(
            "lease-1",
            capability="network_connect",
            operation="connect",
            target_kind="network",
            target="example.test:443",
        )
    layer.request_effect(
        "lease-1",
        capability="filesystem_read",
        operation="read",
        target_kind="path",
        target="workspace/input.txt",
    )
    with pytest.raises(ValueError, match="effect budget"):
        layer.request_effect(
            "lease-1",
            capability="filesystem_read",
            operation="read",
            target_kind="path",
            target="workspace/other.txt",
        )


class MismatchedGateway(RecordingGateway):
    def decide(self, request: AgentEffectRequest) -> AgentDecision:
        decision = super().decide(request)
        return replace(decision, request_digest="sha256:" + "0" * 64)


def test_mismatched_agent_decision_is_rejected() -> None:
    layer = MoiraAgentControl(MismatchedGateway())
    layer.open_lease(
        lease_id="lease-1",
        task_id="task-1",
        scope_id="workspace-1",
        mode=CognitiveMode.ATROPOS,
        role=AgentRole.REVIEWER,
        max_turns=1,
        max_effects=1,
        allowed_actions=("filesystem_read:read",),
    )
    layer.record_turn("lease-1", "Review one file")

    with pytest.raises(ValueError, match="does not match"):
        layer.request_effect(
            "lease-1",
            capability="filesystem_read",
            operation="read",
            target_kind="path",
            target="workspace/file.txt",
        )


def test_receipt_must_match_the_exact_request_and_permit() -> None:
    layer = MoiraAgentControl(RecordingGateway())
    layer.open_lease(
        lease_id="lease-1",
        task_id="task-1",
        scope_id="workspace-1",
        mode=CognitiveMode.ATROPOS,
        role=AgentRole.EXECUTOR,
        max_turns=1,
        max_effects=1,
        allowed_actions=("process_execute:execute",),
    )
    layer.record_turn("lease-1", "Run the bounded verifier")
    outcome = layer.request_effect(
        "lease-1",
        capability="process_execute",
        operation="execute",
        target_kind="process",
        target="pytest",
    )
    assert outcome.authorized is not None

    with pytest.raises(ValueError, match="effect permit"):
        layer.record_receipt(
            outcome.authorized,
            AgentEffectReceipt(
                request_digest=outcome.request.request_digest,
                permit_id="wrong-permit",
                receipt_id="receipt-1",
                outcome="succeeded",
            ),
        )


def test_trace_is_immutable_to_callers_and_digest_chained() -> None:
    layer = MoiraAgentControl(RecordingGateway(DecisionStatus.DENY))
    layer.open_lease(
        lease_id="lease-1",
        task_id="task-1",
        scope_id="workspace-1",
        mode=CognitiveMode.CLOTHO,
        role=AgentRole.EVALUATOR,
        max_turns=1,
        max_effects=0,
    )
    layer.record_turn("lease-1", "Evaluate without tools")
    first = layer.trace()
    layer.close_lease("lease-1")
    second = layer.trace()

    assert isinstance(first, tuple)
    assert len(first) == 2
    assert len(second) == 3
    assert first == second[:2]
    assert len({event.chain_digest for event in second}) == 3
    assert [event.index for event in second] == [0, 1, 2]


def test_agent_contract_keeps_authority_outside_the_model() -> None:
    contract = agent_contract()

    assert contract["any_orientation_can_hold_any_role"] is True
    assert contract["whole_model_agentic"] is True
    assert contract["agent_layer_scope"] == "shared-across-task-selected-sisters"
    assert contract["external_agent_runtime_required"] is False
    assert contract["external_model_required_for_agent_loop"] is False
    assert contract["context_protocol"] == "moira-context-v1"
    assert contract["context_scope_bound_to_lease"] is True
    assert contract["context_receipt_binds_labels_and_content"] is True
    assert contract["context_full_request_preflight_implemented"] is False
    assert contract["context_external_priority_ceiling"] == 60
    assert contract["context_latest_observation_required"] is True
    assert contract["context_persistence_implemented"] is False
    assert contract["context_compression_implemented"] is False
    assert contract["model_can_authorize_its_own_effects"] is False
    assert contract["developer_effects_require_operator_approval"] is True
    assert contract["standalone_profiles"]["developer"] == [
        "workspace.read",
        "workspace.search",
        "workspace.write",
        "process.run",
    ]
    assert contract["internal_control_components"] == [
        "RoleScheduler",
        "ContextAssembler",
        "PolicyEngine",
        "ReferenceMonitor",
        "ToolRegistry",
        "ReceiptLedger",
    ]
    assert contract["optional_hosts"] == [
        "codex",
        "claude-code",
        "hermes-agent",
        "local",
    ]


def _standalone_kernel(
    *, approval_actions: frozenset[str] = frozenset()
) -> MoiraAgentKernel:
    return MoiraAgentKernel(
        AgentPolicy(
            policy_version=3,
            allowed_actions=frozenset({"filesystem_read:read"}),
            approval_actions=approval_actions,
        ),
        permit_secret=b"k" * 32,
    )


def _open_kernel_lease(kernel: MoiraAgentKernel, lease_id: str = "lease-1") -> None:
    kernel.open_lease(
        lease_id=lease_id,
        task_id="task-1",
        scope_id="workspace-1",
        mode=CognitiveMode.CLOTHO,
        role=AgentRole.ANALYST,
        max_turns=3,
        max_effects=2,
        allowed_actions=("filesystem_read:read",),
    )
    kernel.record_turn(lease_id, "Read the scoped evidence")


def test_standalone_kernel_authorizes_executes_and_receipts_registered_tool() -> None:
    kernel = _standalone_kernel()
    kernel.register_tool(
        MoiraTool(
            tool_id="workspace.read",
            capability="filesystem_read",
            operation="read",
            target_kind="path",
            handler=lambda target, payload: target.encode() + b":" + payload,
        )
    )
    _open_kernel_lease(kernel)

    outcome = kernel.request_tool(
        "lease-1", tool_id="workspace.read", target="README.md", payload=b"bounded"
    )
    assert outcome.authorized is not None
    execution = kernel.execute_effect(outcome.authorized, "workspace.read")

    assert execution.output == b"README.md:bounded"
    assert execution.receipt.outcome == "succeeded"
    assert kernel.trace()[-1].kind == "effect_completed"
    with pytest.raises(ValueError, match="unavailable|already consumed"):
        kernel.execute_effect(outcome.authorized, "workspace.read")


def test_standalone_kernel_requires_exact_operator_approval() -> None:
    kernel = _standalone_kernel(
        approval_actions=frozenset({"filesystem_read:read"})
    )
    kernel.register_tool(
        MoiraTool(
            tool_id="workspace.read",
            capability="filesystem_read",
            operation="read",
            target_kind="path",
            handler=lambda target, payload: b"ok",
        )
    )
    _open_kernel_lease(kernel)
    outcome = kernel.request_tool(
        "lease-1", tool_id="workspace.read", target="README.md"
    )

    assert outcome.decision.status is DecisionStatus.REQUIRE_APPROVAL
    assert outcome.authorized is None
    with pytest.raises(ValueError, match="different request"):
        kernel.approve_effect(
            outcome,
            OperatorApproval("approval-1", "sha256:" + "0" * 64),
        )
    authorized = kernel.approve_effect(
        outcome,
        OperatorApproval("approval-1", outcome.request.request_digest),
    )
    assert kernel.execute_effect(authorized, "workspace.read").output == b"ok"


class ExpandingHost:
    def review(
        self, request: AgentEffectRequest, local_status: DecisionStatus
    ) -> DecisionStatus:
        return DecisionStatus.ALLOW


def test_optional_host_cannot_expand_a_local_denial() -> None:
    kernel = MoiraAgentKernel(
        AgentPolicy.deny_all(), permit_secret=b"k" * 32, host_guard=ExpandingHost()
    )
    kernel.register_tool(
        MoiraTool(
            tool_id="workspace.read",
            capability="filesystem_read",
            operation="read",
            target_kind="path",
            handler=lambda target, payload: b"should not run",
        )
    )
    _open_kernel_lease(kernel)

    outcome = kernel.request_tool(
        "lease-1", tool_id="workspace.read", target="README.md"
    )

    assert outcome.decision.status is DecisionStatus.DENY
    assert outcome.authorized is None
