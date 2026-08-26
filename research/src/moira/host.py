"""Optional harness and standalone I/O adapters for MOIRA's native agent loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import sys
from typing import TextIO
import uuid

from .agent import (
    DecisionStatus,
    OperatorApproval,
    AgentEffectRequest,
)


KNOWN_HOSTS = (
    "standalone",
    "codex",
    "claude-code",
    "hermes-agent",
)


@dataclass(frozen=True)
class HarnessCallbacks:
    emit: Callable[[str], None]
    ask: Callable[[str], str]
    approve: Callable[[AgentEffectRequest, str], bool]
    review: Callable[[AgentEffectRequest, DecisionStatus], DecisionStatus] | None = None


class CallbackHarnessAdapter:
    """Connect an optional host without transferring cognition or authority."""

    def __init__(self, host_name: str, callbacks: HarnessCallbacks) -> None:
        if not isinstance(host_name, str) or not host_name.strip() or "\x00" in host_name:
            raise ValueError("host_name must be non-empty text")
        if len(host_name.encode("utf-8")) > 128:
            raise ValueError("host_name exceeds its size budget")
        if not isinstance(callbacks, HarnessCallbacks):
            raise ValueError("callbacks must be HarnessCallbacks")
        self.host_name = host_name
        self._callbacks = callbacks

    def emit(self, text: str) -> None:
        self._callbacks.emit(text)

    def ask(self, question: str) -> str:
        return self._callbacks.ask(question)

    def request_approval(
        self, request: AgentEffectRequest, reason: str
    ) -> OperatorApproval | None:
        if not self._callbacks.approve(request, reason):
            return None
        return OperatorApproval(
            approval_id=f"{self.host_name}-approval-{uuid.uuid4().hex}",
            request_digest=request.request_digest,
        )

    def review(
        self, request: AgentEffectRequest, local_status: DecisionStatus
    ) -> DecisionStatus:
        if self._callbacks.review is None:
            return local_status
        return self._callbacks.review(request, local_status)


class StandaloneConsolePort:
    """Minimal terminal port for MOIRA without any surrounding harness."""

    def __init__(self, *, output: TextIO | None = None) -> None:
        self._output = output if output is not None else sys.stdout

    def emit(self, text: str) -> None:
        print(text, file=self._output)

    def ask(self, question: str) -> str:
        print(f"{question}\n> ", file=self._output, end="", flush=True)
        return input()

    def request_approval(
        self, request: AgentEffectRequest, reason: str
    ) -> OperatorApproval | None:
        print(
            f"MOIRA requests {request.capability}:{request.operation} "
            f"for {request.target!r}. Payload: {request.payload_size} bytes, "
            f"{request.payload_digest}. Request: {request.request_digest}. "
            f"{reason}. Approve? [y/N]\n> ",
            file=self._output,
            end="",
            flush=True,
        )
        answer = input()
        if answer.strip().casefold() not in {"y", "yes"}:
            return None
        return OperatorApproval(
            approval_id=f"standalone-approval-{uuid.uuid4().hex}",
            request_digest=request.request_digest,
        )


__all__ = [
    "KNOWN_HOSTS",
    "CallbackHarnessAdapter",
    "HarnessCallbacks",
    "StandaloneConsolePort",
]

