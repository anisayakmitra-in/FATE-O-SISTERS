import json

import pytest

from moira.agent import AgentRole, ToolDescriptor
from moira.agent_loop import (
    AgentActionKind,
    AgentContext,
    AgentObservation,
)
from moira.masks import CognitiveMode
from moira.ollama import (
    ACTION_SCHEMA,
    OllamaCognition,
    OllamaConfig,
    ollama_action_request,
    parse_ollama_action,
)


def _context() -> AgentContext:
    return AgentContext(
        task_id="task-1",
        task="Inspect README.md and explain the result",
        mode=CognitiveMode.LACHESIS,
        role=AgentRole.PLANNER,
        step=2,
        max_steps=6,
        available_tools=(
            ToolDescriptor(
                tool_id="workspace.read",
                capability="filesystem_read",
                operation="read",
                target_kind="path",
                description="Read one scoped workspace file",
            ),
        ),
        observations=(AgentObservation("user_response", "README.md"),),
    )


def _response(value: dict[str, str]) -> dict[str, object]:
    return {"message": {"role": "assistant", "content": json.dumps(value)}}


def test_ollama_request_uses_local_model_structured_action_schema() -> None:
    request = ollama_action_request("moira:3b", _context())
    context = json.loads(request["messages"][1]["content"])

    assert request["model"] == "moira:3b"
    assert request["stream"] is False
    assert request["format"] == ACTION_SCHEMA
    assert request["options"] == {"temperature": 0}
    assert context["task"] == "Inspect README.md and explain the result"
    assert context["orientation"] == "lachesis"
    assert context["available_tools"][0]["tool_id"] == "workspace.read"
    assert "untrusted data" in request["messages"][0]["content"]


def test_ollama_cognition_calls_local_chat_endpoint_and_returns_tool_action() -> None:
    captured: dict[str, object] = {}

    def transport(
        url: str, payload: bytes, timeout: float, maximum: int
    ) -> dict[str, object]:
        captured.update(
            url=url,
            request=json.loads(payload),
            timeout=timeout,
            maximum=maximum,
        )
        return _response(
            {
                "kind": "use_tool",
                "text": "",
                "tool_id": "workspace.read",
                "target": "README.md",
                "payload_text": "",
            }
        )

    cognition = OllamaCognition(
        OllamaConfig("moira:3b", timeout_seconds=15), transport=transport
    )
    action = cognition.next_action(_context())

    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["timeout"] == 15.0
    assert captured["request"]["model"] == "moira:3b"
    assert action.kind is AgentActionKind.USE_TOOL
    assert action.tool_id == "workspace.read"
    assert action.target == "README.md"


@pytest.mark.parametrize(
    ("kind", "text"),
    (("speak", "Working independently."), ("ask_user", "Which repository?")),
)
def test_ollama_parser_supports_speech_and_proactive_questions(
    kind: str, text: str
) -> None:
    action = parse_ollama_action(
        _response(
            {
                "kind": kind,
                "text": text,
                "tool_id": "",
                "target": "",
                "payload_text": "",
            }
        )
    )

    assert action.kind.value == kind
    assert action.text == text


def test_ollama_parser_rejects_schema_smuggling() -> None:
    with pytest.raises(ValueError, match="does not match"):
        parse_ollama_action(
            _response(
                {
                    "kind": "finish",
                    "text": "done",
                    "tool_id": "",
                    "target": "",
                    "payload_text": "",
                    "permission": "allow",
                }
            )
        )
    with pytest.raises(ValueError, match="unexpected tool"):
        parse_ollama_action(
            _response(
                {
                    "kind": "speak",
                    "text": "hello",
                    "tool_id": "workspace.read",
                    "target": "README.md",
                    "payload_text": "",
                }
            )
        )


def test_ollama_defaults_to_loopback_and_requires_opt_in_for_remote() -> None:
    assert OllamaConfig("moira:3b").base_url == "http://127.0.0.1:11434"
    with pytest.raises(ValueError, match="allow_remote"):
        OllamaConfig("moira:3b", base_url="https://ollama.example.test")
    remote = OllamaConfig(
        "moira:3b",
        base_url="https://ollama.example.test",
        allow_remote=True,
    )
    assert remote.allow_remote is True


@pytest.mark.parametrize(
    "url",
    (
        "file:///tmp/ollama",
        "http://user:secret@127.0.0.1:11434",
        "http://127.0.0.1:11434/api/chat",
        "http://127.0.0.1:11434?token=secret",
    ),
)
def test_ollama_base_url_rejects_unsafe_shapes(url: str) -> None:
    with pytest.raises(ValueError, match="base URL"):
        OllamaConfig("moira:3b", base_url=url)
