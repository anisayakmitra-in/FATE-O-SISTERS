from moira.agent import DecisionStatus, AgentEffectRequest
from moira.host import CallbackHarnessAdapter, HarnessCallbacks, KNOWN_HOSTS
from moira.masks import CognitiveMode
from moira.agent import AgentRole


def _request() -> AgentEffectRequest:
    return AgentEffectRequest(
        lease_id="lease-1",
        task_id="task-1",
        scope_id="workspace-1",
        mode=CognitiveMode.CLOTHO,
        role=AgentRole.ANALYST,
        capability="filesystem_read",
        operation="read",
        target_kind="path",
        target="README.md",
        payload_digest="sha256:" + "1" * 64,
        request_digest="sha256:" + "2" * 64,
    )


def test_known_hosts_include_standalone_and_open_harness_categories() -> None:
    assert KNOWN_HOSTS == (
        "standalone",
        "codex",
        "claude-code",
        "hermes-agent",
    )


def test_callback_harness_adapts_io_approval_and_restriction() -> None:
    emitted: list[str] = []
    questions: list[str] = []
    adapter = CallbackHarnessAdapter(
        "custom-harness",
        HarnessCallbacks(
            emit=emitted.append,
            ask=lambda question: questions.append(question) or "answer",
            approve=lambda request, reason: True,
            review=lambda request, status: DecisionStatus.REQUIRE_APPROVAL,
        ),
    )
    request = _request()

    adapter.emit("hello")
    assert adapter.ask("question?") == "answer"
    approval = adapter.request_approval(request, "policy")

    assert emitted == ["hello"]
    assert questions == ["question?"]
    assert approval is not None
    assert approval.request_digest == request.request_digest
    assert adapter.review(request, DecisionStatus.ALLOW) is DecisionStatus.REQUIRE_APPROVAL

