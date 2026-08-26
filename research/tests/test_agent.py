from dataclasses import replace

import pytest

from moira.agent import (
    AgentRole,
    DecisionStatus,
    PandoraAgentLayer,
    PandoraDecision,
    PandoraEffectReceipt,
    PandoraEffectRequest,
    agent_contract,
)
from moira.masks import CognitiveMode


class RecordingGateway:
    def __init__(self, status: DecisionStatus = DecisionStatus.ALLOW) -> None:
        self.status = status
        self.requests: list[PandoraEffectRequest] = []

    def decide(self, request: PandoraEffectRequest) -> PandoraDecision:
        self.requests.append(request)
        return PandoraDecision(
            request_digest=request.request_digest,
            status=self.status,
            policy_version=7,
            reason="test policy decision",
            permit_id="permit-1" if self.status is DecisionStatus.ALLOW else None,
        )


@pytest.mark.parametrize("mode", list(CognitiveMode))
@pytest.mark.parametrize("role", list(AgentRole))
def test_any_orientation_can_hold_any_agent_role(
    mode: CognitiveMode, role: AgentRole
) -> None:
    layer = PandoraAgentLayer(RecordingGateway())

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


def test_allowed_effect_requires_exact_pandora_permit_and_does_not_execute() -> None:
    gateway = RecordingGateway()
    layer = PandoraAgentLayer(gateway)
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
    assert outcome.authorized.permit_id == "permit-1"
    assert outcome.authorized.request == gateway.requests[0]
    assert not hasattr(layer, "execute")

    receipt = PandoraEffectReceipt(
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
    layer = PandoraAgentLayer(RecordingGateway(status))
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
    layer = PandoraAgentLayer(gateway)
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
    def decide(self, request: PandoraEffectRequest) -> PandoraDecision:
        decision = super().decide(request)
        return replace(decision, request_digest="sha256:" + "0" * 64)


def test_mismatched_pandora_decision_is_rejected() -> None:
    layer = PandoraAgentLayer(MismatchedGateway())
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
    layer = PandoraAgentLayer(RecordingGateway())
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
            PandoraEffectReceipt(
                request_digest=outcome.request.request_digest,
                permit_id="wrong-permit",
                receipt_id="receipt-1",
                outcome="succeeded",
            ),
        )


def test_trace_is_immutable_to_callers_and_digest_chained() -> None:
    layer = PandoraAgentLayer(RecordingGateway(DecisionStatus.DENY))
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
    assert contract["direct_effect_execution"] is False
    assert contract["model_can_authorize_its_own_effects"] is False
    assert contract["external_authority"] == [
        "ShadowCouncil",
        "Parliament",
        "ReferenceMonitor",
        "EffectExecutor",
    ]
