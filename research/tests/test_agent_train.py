import json
from pathlib import Path
import uuid

import pytest
import torch

from moira.agent_loop import AgentActionKind
from moira.agent_train import (
    AGENT_ACTION_OBJECTIVE,
    canonical_agent_prompt,
    canonical_agent_target,
    evaluate_agent_jsonl,
    load_agent_trajectories,
    train_agent_jsonl,
    trajectory_examples,
)
from moira.config import ModelConfig
from moira.m1_train import build_causal_batch


FIXTURE = Path("research/tests/fixtures/agent-trajectories.jsonl")


def test_agent_trajectory_schema_covers_all_native_action_kinds() -> None:
    trajectories = load_agent_trajectories(FIXTURE, max_records=4)

    assert [trajectory.action.kind for trajectory in trajectories] == [
        AgentActionKind.USE_TOOL,
        AgentActionKind.FINISH,
        AgentActionKind.ASK_USER,
        AgentActionKind.SPEAK,
    ]
    assert all(len(trajectory.record_sha256) == 64 for trajectory in trajectories)
    prompt = json.loads(canonical_agent_prompt(trajectories[0]))
    target = json.loads(canonical_agent_target(trajectories[0].action))
    assert prompt["agent_protocol"] == "moira-agent-protocol-v1"
    assert "action" not in prompt
    assert prompt["available_tools"][0]["tool_id"] == "workspace.search"
    assert target == {
        "kind": "use_tool",
        "payload_text": "",
        "target": "MOIRA",
        "text": "",
        "tool_id": "workspace.search",
    }


def test_agent_batch_trains_only_on_canonical_action_tokens() -> None:
    trajectories = load_agent_trajectories(FIXTURE, max_records=4)
    examples = trajectory_examples(trajectories)
    config = ModelConfig.agent_smoke()
    batch = build_causal_batch(examples, sequence_length=config.max_seq_len)

    assert batch.input_ids.shape == (4, config.max_seq_len - 1)
    assert batch.loss_mask.any(dim=1).all()
    assert batch.loss_mask.sum(dim=1).tolist() == [
        len(canonical_agent_target(trajectory.action).encode("utf-8")) + 1
        for trajectory in trajectories
    ]
    assert all(
        example.preprocess_version == AGENT_ACTION_OBJECTIVE for example in examples
    )
    assert all(
        example.inputs["text"] not in example.targets["text"] for example in examples
    )


def test_agent_training_exactly_resumes_and_evaluates_without_mutation() -> None:
    checkpoint = Path("research") / f".agent-train-{uuid.uuid4().hex}.pt"
    uninterrupted = Path("research") / f".agent-full-{uuid.uuid4().hex}.pt"
    config = ModelConfig.agent_smoke()
    try:
        torch.manual_seed(20260826)
        first = train_agent_jsonl(
            FIXTURE,
            steps=1,
            checkpoint_path=checkpoint,
            model_config=config,
        )
        resumed = train_agent_jsonl(
            FIXTURE,
            steps=2,
            checkpoint_path=checkpoint,
            resume=True,
            model_config=config,
        )
        torch.manual_seed(20260826)
        train_agent_jsonl(
            FIXTURE,
            steps=2,
            checkpoint_path=uninterrupted,
            model_config=config,
        )
        resumed_payload = torch.load(checkpoint, weights_only=True)
        uninterrupted_payload = torch.load(uninterrupted, weights_only=True)
        before_evaluation = checkpoint.read_bytes()
        evaluation = evaluate_agent_jsonl(
            FIXTURE,
            checkpoint_path=checkpoint,
            max_records=4,
            model_config=config,
        )

        assert first.samples == 1
        assert resumed.samples == 2
        assert len(resumed.losses) == 2
        assert resumed_payload["manifest"]["objective"] == AGENT_ACTION_OBJECTIVE
        assert resumed_payload["losses"] == uninterrupted_payload["losses"]
        for name, tensor in resumed_payload["model_state"].items():
            assert torch.equal(tensor, uninterrupted_payload["model_state"][name])
        assert evaluation.samples == 4
        assert evaluation.tokens > 0
        assert evaluation.loss > 0
        assert 0.0 <= evaluation.token_accuracy <= 1.0
        assert checkpoint.read_bytes() == before_evaluation
    finally:
        checkpoint.unlink(missing_ok=True)
        uninterrupted.unlink(missing_ok=True)


def test_agent_trajectory_rejects_unavailable_tool_action() -> None:
    path = Path("research") / f".agent-invalid-{uuid.uuid4().hex}.jsonl"
    value = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    value["action"]["tool_id"] = "workspace.missing"
    try:
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unavailable tool"):
            load_agent_trajectories(path, max_records=1)
    finally:
        path.unlink(missing_ok=True)


def test_agent_trajectory_rejects_unknown_fields() -> None:
    path = Path("research") / f".agent-invalid-{uuid.uuid4().hex}.jsonl"
    value = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    value["unexpected"] = True
    try:
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="exact schema"):
            load_agent_trajectories(path, max_records=1)
    finally:
        path.unlink(missing_ok=True)
