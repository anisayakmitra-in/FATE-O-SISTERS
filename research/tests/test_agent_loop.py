from collections import deque

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
    assert cognition.contexts[-1].observations[-1].kind == "tool_result"
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
    )

    assert result.status is AgentRunStatus.STEP_BUDGET_EXHAUSTED

