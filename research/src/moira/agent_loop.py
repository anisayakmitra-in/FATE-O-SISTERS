"""Self-directed MOIRA action loop with optional host I/O."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .agent import (
    AgentRole,
    DecisionStatus,
    EffectOutcome,
    MoiraAgentKernel,
    OperatorApproval,
    AgentEffectRequest,
    AgentEffectReceipt,
    ToolDescriptor,
)
from .masks import CognitiveMode


class AgentActionKind(str, Enum):
    SPEAK = "speak"
    ASK_USER = "ask_user"
    USE_TOOL = "use_tool"
    FINISH = "finish"


@dataclass(frozen=True)
class AgentAction:
    kind: AgentActionKind
    text: str = ""
    tool_id: str | None = None
    target: str | None = None
    payload: bytes = b""

    @classmethod
    def speak(cls, text: str) -> AgentAction:
        return cls(AgentActionKind.SPEAK, text=text)

    @classmethod
    def ask_user(cls, question: str) -> AgentAction:
        return cls(AgentActionKind.ASK_USER, text=question)

    @classmethod
    def use_tool(
        cls, tool_id: str, target: str, payload: bytes = b""
    ) -> AgentAction:
        return cls(
            AgentActionKind.USE_TOOL,
            tool_id=tool_id,
            target=target,
            payload=payload,
        )

    @classmethod
    def finish(cls, text: str = "") -> AgentAction:
        return cls(AgentActionKind.FINISH, text=text)


@dataclass(frozen=True)
class AgentObservation:
    kind: str
    content: str


@dataclass(frozen=True)
class AgentContext:
    task_id: str
    task: str
    mode: CognitiveMode
    role: AgentRole
    step: int
    max_steps: int
    available_tools: tuple[ToolDescriptor, ...]
    observations: tuple[AgentObservation, ...]


class MoiraCognition(Protocol):
    """Implemented by MOIRA inference, never by a required external model."""

    def next_action(self, context: AgentContext) -> AgentAction: ...


class AgentPort(Protocol):
    """Standalone UI or optional harness adapter."""

    def emit(self, text: str) -> None: ...

    def ask(self, question: str) -> str: ...

    def request_approval(
        self, request: AgentEffectRequest, reason: str
    ) -> OperatorApproval | None: ...


class AgentRunStatus(str, Enum):
    COMPLETED = "completed"
    APPROVAL_REQUIRED = "approval_required"
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"


@dataclass(frozen=True)
class AgentRunResult:
    status: AgentRunStatus
    steps: int
    observations: tuple[AgentObservation, ...]
    receipts: tuple[AgentEffectReceipt, ...]
    pending_effect: EffectOutcome | None = None


class MoiraAgentRuntime:
    """Runs actions chosen by MOIRA cognition through its internal kernel."""

    def __init__(self, kernel: MoiraAgentKernel) -> None:
        if not isinstance(kernel, MoiraAgentKernel):
            raise ValueError("kernel must be a MoiraAgentKernel")
        self._kernel = kernel

    def run(
        self,
        *,
        lease_id: str,
        task_id: str,
        task: str,
        mode: CognitiveMode,
        role: AgentRole,
        cognition: MoiraCognition,
        port: AgentPort,
        max_steps: int,
    ) -> AgentRunResult:
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if (
            not isinstance(task, str)
            or not task.strip()
            or "\x00" in task
            or len(task.encode("utf-8")) > 1_000_000
        ):
            raise ValueError("task must be bounded non-empty text")
        observations: list[AgentObservation] = []
        receipts: list[AgentEffectReceipt] = []

        for step in range(1, max_steps + 1):
            context = AgentContext(
                task_id=task_id,
                task=task,
                mode=mode,
                role=role,
                step=step,
                max_steps=max_steps,
                available_tools=self._kernel.tool_catalog(),
                observations=tuple(observations),
            )
            action = cognition.next_action(context)
            _validate_action(action)
            self._kernel.record_turn(lease_id, _action_summary(action))

            if action.kind is AgentActionKind.SPEAK:
                port.emit(action.text)
                observations.append(AgentObservation("moira_speech", action.text))
                continue
            if action.kind is AgentActionKind.ASK_USER:
                response = port.ask(action.text)
                if not isinstance(response, str):
                    raise ValueError("agent port user response must be text")
                observations.extend(
                    (
                        AgentObservation("moira_question", action.text),
                        AgentObservation("user_response", response),
                    )
                )
                continue
            if action.kind is AgentActionKind.FINISH:
                if action.text:
                    port.emit(action.text)
                    observations.append(AgentObservation("moira_speech", action.text))
                self._kernel.close_lease(lease_id)
                return AgentRunResult(
                    AgentRunStatus.COMPLETED,
                    step,
                    tuple(observations),
                    tuple(receipts),
                    None,
                )

            assert action.tool_id is not None and action.target is not None
            outcome = self._kernel.request_tool(
                lease_id,
                tool_id=action.tool_id,
                target=action.target,
                payload=action.payload,
            )
            if outcome.decision.status is DecisionStatus.DENY:
                observations.append(
                    AgentObservation("tool_denied", outcome.decision.reason)
                )
                continue
            authorized = outcome.authorized
            if outcome.decision.status is DecisionStatus.REQUIRE_APPROVAL:
                approval = port.request_approval(
                    outcome.request, outcome.decision.reason
                )
                if approval is None:
                    return AgentRunResult(
                        AgentRunStatus.APPROVAL_REQUIRED,
                        step,
                        tuple(observations),
                        tuple(receipts),
                        outcome,
                    )
                authorized = self._kernel.approve_effect(outcome, approval)
            assert authorized is not None
            execution = self._kernel.execute_effect(authorized, action.tool_id)
            receipts.append(execution.receipt)
            if execution.output is None:
                observations.append(
                    AgentObservation(
                        "tool_failed", execution.error_code or "tool execution failed"
                    )
                )
            else:
                observations.append(
                    AgentObservation(
                        "tool_result", execution.output.decode("utf-8", errors="replace")
                    )
                )

        self._kernel.close_lease(lease_id)
        return AgentRunResult(
            AgentRunStatus.STEP_BUDGET_EXHAUSTED,
            max_steps,
            tuple(observations),
            tuple(receipts),
            None,
        )


def _validate_action(action: object) -> None:
    if not isinstance(action, AgentAction) or not isinstance(
        action.kind, AgentActionKind
    ):
        raise ValueError("MOIRA cognition returned an invalid action")
    if not isinstance(action.text, str) or not isinstance(action.payload, bytes):
        raise ValueError("MOIRA action fields have invalid types")
    if "\x00" in action.text:
        raise ValueError("MOIRA action text cannot contain NUL")
    if len(action.text.encode("utf-8")) > 1_000_000 or len(action.payload) > 16 * 1024 * 1024:
        raise ValueError("MOIRA action exceeds its size budget")
    if action.kind in (AgentActionKind.SPEAK, AgentActionKind.ASK_USER):
        if not action.text.strip() or action.tool_id is not None or action.target is not None:
            raise ValueError("speech and question actions require only non-empty text")
    elif action.kind is AgentActionKind.USE_TOOL:
        if (
            not isinstance(action.tool_id, str)
            or not action.tool_id
            or not isinstance(action.target, str)
            or not action.target
            or action.text
        ):
            raise ValueError("tool actions require a tool ID and target")
    elif action.tool_id is not None or action.target is not None or action.payload:
        raise ValueError("finish actions cannot contain tool fields")


def _action_summary(action: AgentAction) -> str:
    if action.kind is AgentActionKind.USE_TOOL:
        return f"use_tool:{action.tool_id}:{action.target}"
    return f"{action.kind.value}:{action.text}"


__all__ = [
    "AgentAction",
    "AgentActionKind",
    "AgentContext",
    "AgentObservation",
    "AgentPort",
    "AgentRunResult",
    "AgentRunStatus",
    "MoiraAgentRuntime",
    "MoiraCognition",
]

