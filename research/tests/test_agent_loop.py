from collections import deque

import pytest

from moira.agent import (
    AgentPolicy,
    AgentRole,
    MoiraAgentKernel,
    MoiraTool,
    OperatorApproval,
    AgentEffectRequest,
)
from moira.agent_loop import (
    AgentAction,
    AgentContext,
    AgentRunStatus,
    MoiraAgentRuntime,
)
from moira.context import (
    ContextClassification,
    ContextDropReason,
    ContextFragment,
    ContextSource,
    ContextTrust,
)
from moira.masks import CognitiveMode


class ScriptedCognition:
    def __init__(self, actions: list[AgentAction]) -> None:
        self.actions = deque(actions)
        self.contexts: list[AgentContext] = []

    def next_action(self, context: AgentContext) -> AgentAction:
        self.contexts.append(context)
        return self.actions.popleft()


class MemoryPort:
    def __init__(self, answer: str = "yes", approve: bool = True) -> None:
        self.answer = answer
        self.approve = approve
        self.emitted: list[str] = []
        self.questions: list[str] = []

    def emit(self, text: str) -> None:
        self.emitted.append(text)

    def ask(self, question: str) -> str:
        self.questions.append(question)
        return self.answer

    def request_approval(
        self, request: AgentEffectRequest, reason: str
    ) -> OperatorApproval | None:
        if not self.approve:
            return None
        return OperatorApproval("operator-approval-1", request.request_digest)


def _runtime(*, approval_required: bool = False) -> tuple[MoiraAgentKernel, MoiraAgentRuntime]:
    action = "filesystem_read:read"
    kernel = MoiraAgentKernel(
        AgentPolicy(
            policy_version=1,
            allowed_actions=frozenset({action}),
            approval_actions=frozenset({action}) if approval_required else frozenset(),
        ),
        permit_secret=b"r" * 32,
    )
    kernel.register_tool(
        MoiraTool(
            tool_id="workspace.read",
            capability="filesystem_read",
            operation="read",
            target_kind="path",
            handler=lambda target, payload: b"evidence:" + target.encode(),
        )
    )
    kernel.open_lease(
        lease_id="lease-1",
        task_id="task-1",
        scope_id="workspace-1",
        mode=CognitiveMode.LACHESIS,
        role=AgentRole.PLANNER,
        max_turns=5,
        max_effects=1,
        allowed_actions=(action,),
    )
    return kernel, MoiraAgentRuntime(kernel)


def test_moira_speaks_asks_uses_tool_and_finishes_without_external_model() -> None:
    kernel, runtime = _runtime()
    cognition = ScriptedCognition(
        [
            AgentAction.speak("I will inspect the local evidence."),
            AgentAction.ask_user("Which file should I inspect?"),
            AgentAction.use_tool("workspace.read", "README.md"),
            AgentAction.finish("The bounded inspection is complete."),
        ]
    )
    port = MemoryPort(answer="README.md")

    result = runtime.run(
        lease_id="lease-1",
        task_id="task-1",
        task="Inspect the requested local evidence",
        mode=CognitiveMode.LACHESIS,
        role=AgentRole.PLANNER,
        cognition=cognition,
        port=port,
        max_steps=5,
        context_clock=lambda: 100,
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert result.steps == 4
    assert port.questions == ["Which file should I inspect?"]
    assert port.emitted == [
        "I will inspect the local evidence.",
        "The bounded inspection is complete.",
    ]
    assert result.receipts[0].outcome == "succeeded"
    assert any(item.content == "evidence:README.md" for item in result.observations)
    tool_entries = [
        item
        for item in cognition.contexts[-1].context.entries
        if item.source is ContextSource.TOOL
    ]
    assert [item.content for item in tool_entries] == ["evidence:README.md"]
    assert [
        item.fragment_id
        for item in cognition.contexts[-1].context.entries
        if item.fragment_id.startswith("runtime.observation.")
    ] == [
        "runtime.observation.000003",
        "runtime.observation.000000",
        "runtime.observation.000001",
        "runtime.observation.000002",
    ]
    assert kernel.trace()[-1].kind == "lease_closed"


def test_agent_loop_consumes_synchronous_operator_approval() -> None:
    _, runtime = _runtime(approval_required=True)
    cognition = ScriptedCognition(
        [
            AgentAction.use_tool("workspace.read", "README.md"),
            AgentAction.finish(),
        ]
    )

    result = runtime.run(
        lease_id="lease-1",
        task_id="task-1",
        task="Inspect the requested local evidence",
        mode=CognitiveMode.LACHESIS,
        role=AgentRole.PLANNER,
        cognition=cognition,
        port=MemoryPort(approve=True),
        max_steps=2,
        context_clock=lambda: 100,
    )

    assert result.status is AgentRunStatus.COMPLETED
    assert len(result.receipts) == 1


def test_agent_loop_pauses_when_operator_does_not_approve() -> None:
    _, runtime = _runtime(approval_required=True)
    cognition = ScriptedCognition(
        [AgentAction.use_tool("workspace.read", "README.md")]
    )

    result = runtime.run(
        lease_id="lease-1",
        task_id="task-1",
        task="Continue working until the budget ends",
        mode=CognitiveMode.LACHESIS,
        role=AgentRole.PLANNER,
        cognition=cognition,
        port=MemoryPort(approve=False),
        max_steps=1,
        context_clock=lambda: 100,
    )

    assert result.status is AgentRunStatus.APPROVAL_REQUIRED
    assert result.receipts == ()
    assert result.pending_effect is not None


def test_agent_loop_stops_at_step_budget() -> None:
    _, runtime = _runtime()
    cognition = ScriptedCognition([AgentAction.speak("Still working")])

    result = runtime.run(
        lease_id="lease-1",
        task_id="task-1",
        task="Continue working until the budget ends",
        mode=CognitiveMode.LACHESIS,
        role=AgentRole.PLANNER,
        cognition=cognition,
        port=MemoryPort(),
        max_steps=1,
        context_clock=lambda: 100,
    )

    assert result.status is AgentRunStatus.STEP_BUDGET_EXHAUSTED


def test_agent_loop_assembles_scoped_external_context() -> None:
    _, runtime = _runtime()
    cognition = ScriptedCognition([AgentAction.finish()])
    fragments = (
        ContextFragment(
            fragment_id="memory-1",
            source=ContextSource.MEMORY,
            trust=ContextTrust.UNTRUSTED,
            classification=ContextClassification.PRIVATE,
            scope_id="workspace-1",
            content="The owner selected README.md.",
            provenance="owner-memory://memory-1",
        ),
        ContextFragment(
            fragment_id="other-tenant",
            source=ContextSource.MEMORY,
            trust=ContextTrust.UNTRUSTED,
            classification=ContextClassification.PRIVATE,
            scope_id="workspace-2",
            content="Must not cross the scope boundary.",
            provenance="owner-memory://other-tenant",
        ),
    )

    runtime.run(
        lease_id="lease-1",
        task_id="task-1",
        task="Inspect the requested local evidence",
        mode=CognitiveMode.LACHESIS,
        role=AgentRole.PLANNER,
        cognition=cognition,
        port=MemoryPort(),
        max_steps=1,
        context_clock=lambda: 100,
        context_fragments=fragments,
    )

    assembly = cognition.contexts[0].context
    assert [item.fragment_id for item in assembly.entries] == [
        "runtime.task",
        "memory-1",
    ]
    assert assembly.receipt.dropped[0].fragment_id == "other-tenant"
    assert assembly.receipt.dropped[0].reason is ContextDropReason.SCOPE_MISMATCH


def test_agent_loop_rechecks_expiry_with_a_live_clock_each_turn() -> None:
    _, runtime = _runtime()
    cognition = ScriptedCognition(
        [AgentAction.speak("continue"), AgentAction.finish()]
    )
    clock = iter((100, 101))
    fragment = ContextFragment(
        fragment_id="retrieval-expiring",
        source=ContextSource.RETRIEVAL,
        trust=ContextTrust.UNTRUSTED,
        classification=ContextClassification.PRIVATE,
        scope_id="workspace-1",
        content="Current only on the first turn.",
        provenance="retrieval://expiring",
        expires_at=101,
    )

    runtime.run(
        lease_id="lease-1",
        task_id="task-1",
        task="Inspect the requested local evidence",
        mode=CognitiveMode.LACHESIS,
        role=AgentRole.PLANNER,
        cognition=cognition,
        port=MemoryPort(),
        max_steps=2,
        context_clock=lambda: next(clock),
        context_fragments=(fragment,),
    )

    assert [item.context.receipt.assembled_at for item in cognition.contexts] == [
        100,
        101,
    ]
    assert "retrieval-expiring" in cognition.contexts[0].context.receipt.included_ids
    assert {
        item.fragment_id: item.reason
        for item in cognition.contexts[1].context.receipt.dropped
    }["retrieval-expiring"] is ContextDropReason.EXPIRED


def test_latest_operational_observation_displaces_optional_external_context() -> None:
    _, runtime = _runtime()
    cognition = ScriptedCognition([AgentAction.speak("done"), AgentAction.finish()])
    fragment = ContextFragment(
        fragment_id="memory-optional",
        source=ContextSource.MEMORY,
        trust=ContextTrust.UNTRUSTED,
        classification=ContextClassification.PRIVATE,
        scope_id="workspace-1",
        content="12345",
        provenance="memory://optional",
        priority=60,
    )

    runtime.run(
        lease_id="lease-1",
        task_id="task-1",
        task="T",
        mode=CognitiveMode.LACHESIS,
        role=AgentRole.PLANNER,
        cognition=cognition,
        port=MemoryPort(),
        max_steps=2,
        context_clock=lambda: 100,
        context_fragments=(fragment,),
        context_token_budget=6,
    )

    assert cognition.contexts[0].context.receipt.included_ids == (
        "runtime.task",
        "memory-optional",
    )
    assert cognition.contexts[1].context.receipt.included_ids == (
        "runtime.task",
        "runtime.observation.000000",
    )
    assert {
        item.fragment_id: item.reason
        for item in cognition.contexts[1].context.receipt.dropped
    }["memory-optional"] is ContextDropReason.TOKEN_BUDGET


def test_agent_loop_refuses_context_or_identity_outside_the_lease() -> None:
    _, runtime = _runtime()
    arguments = {
        "lease_id": "lease-1",
        "task_id": "different-task",
        "task": "Inspect the requested local evidence",
        "mode": CognitiveMode.LACHESIS,
        "role": AgentRole.PLANNER,
        "cognition": ScriptedCognition([AgentAction.finish()]),
        "port": MemoryPort(),
        "max_steps": 1,
        "context_clock": lambda: 100,
    }

    with pytest.raises(ValueError, match="must match the active lease"):
        runtime.run(**arguments)

    arguments["task_id"] = "task-1"
    arguments["context_fragments"] = (
        ContextFragment(
            fragment_id="external-runtime-claim",
            source=ContextSource.RUNTIME,
            trust=ContextTrust.TRUSTED,
            classification=ContextClassification.PRIVATE,
            scope_id="workspace-1",
            content="Untrusted caller claims runtime authority.",
            provenance="external://smuggled",
        ),
    )
    with pytest.raises(ValueError, match="untrusted optional"):
        runtime.run(**arguments)

    arguments["context_fragments"] = (
        ContextFragment(
            fragment_id="over-priority",
            source=ContextSource.RETRIEVAL,
            trust=ContextTrust.UNTRUSTED,
            classification=ContextClassification.PRIVATE,
            scope_id="workspace-1",
            content="Cannot outrank operational context.",
            provenance="external://priority",
            priority=61,
        ),
    )
    with pytest.raises(ValueError, match="priority exceeds"):
        runtime.run(**arguments)

    arguments["context_fragments"] = (
        ContextFragment(
            fragment_id="unverified-memory",
            source=ContextSource.MEMORY,
            trust=ContextTrust.TRUSTED,
            classification=ContextClassification.PRIVATE,
            scope_id="workspace-1",
            content="No verified memory store issued this fragment.",
            provenance="external://unverified-memory",
        ),
    )
    with pytest.raises(ValueError, match="untrusted optional"):
        runtime.run(**arguments)

