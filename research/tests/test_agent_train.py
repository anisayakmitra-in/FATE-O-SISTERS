import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import pytest
import torch

from moira.agent_loop import AgentActionKind
from moira.agent_train import (
    AGENT_ACTION_OBJECTIVE,
    canonical_agent_prompt,
    canonical_agent_target,
    evaluate_agent_jsonl,
    generate_agent_action,
    load_agent_trajectories,
    train_agent_jsonl,
    trajectory_examples,
)
from moira.config import ModelConfig
from moira.m1_train import BYTE_OFFSET, EOS_ID, build_causal_batch
from moira.masks import CognitiveMode


FIXTURE = Path("research/tests/fixtures/agent-trajectories.jsonl")
HELD_OUT_FIXTURE = Path(
    "research/tests/fixtures/agent-trajectories-held-out.jsonl"
)


class _ScriptedModel:
    def __init__(self, text: str) -> None:
        self.config = SimpleNamespace(max_seq_len=768)
        self._tokens = [BYTE_OFFSET + value for value in text.encode("utf-8")]
        self._tokens.append(EOS_ID)
        self.modes = []
        self.visibility_modes = []
        self.source_lengths = []

    def __call__(
        self,
        tokens,
        *,
        mode,
        visibility_mode,
        valid_tokens,
        source_lengths,
    ):
        self.modes.append(mode)
        self.visibility_modes.append(visibility_mode)
        self.source_lengths.append(int(source_lengths.item()))
        next_id = self._tokens[len(self.modes) - 1]
        logits = torch.full((1, tokens.shape[1], 512), -1.0)
        logits[0, -1, next_id] = 1.0
        return SimpleNamespace(logits=logits)


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
            steps=4,
            checkpoint_path=checkpoint,
            resume=True,
            model_config=config,
        )
        torch.manual_seed(20260826)
        train_agent_jsonl(
            FIXTURE,
            steps=4,
            checkpoint_path=uninterrupted,
            model_config=config,
        )
        resumed_payload = torch.load(checkpoint, weights_only=True)
        uninterrupted_payload = torch.load(uninterrupted, weights_only=True)
        before_evaluation = checkpoint.read_bytes()
        evaluation = evaluate_agent_jsonl(
            HELD_OUT_FIXTURE,
            training_path=FIXTURE,
            checkpoint_path=checkpoint,
            max_records=4,
            max_new_tokens=4,
            model_config=config,
        )

        assert first.samples == 1
        assert resumed.samples == 4
        assert len(resumed.losses) == 4
        assert resumed_payload["manifest"]["objective"] == AGENT_ACTION_OBJECTIVE
        assert resumed_payload["manifest"]["orientation_conditioning"] == (
            "native-mode-embedding-v1"
        )
        assert resumed_payload["manifest"]["action_visibility"] == (
            "bidirectional-source-causal-target-v1"
        )
        assert resumed_payload["losses"] == uninterrupted_payload["losses"]
        for name, tensor in resumed_payload["model_state"].items():
            assert torch.equal(tensor, uninterrupted_payload["model_state"][name])
        assert evaluation.samples == 4
        assert evaluation.held_out is True
        assert evaluation.training_dataset_sha256 != evaluation.evaluation_dataset_sha256
        assert evaluation.tokens > 0
        assert evaluation.loss > 0
        assert 0.0 <= evaluation.token_accuracy <= 1.0
        assert 0.0 <= evaluation.schema_validity <= 1.0
        assert 0.0 <= evaluation.exact_action_accuracy <= 1.0
        assert evaluation.tool_selection_samples == 1
        assert checkpoint.read_bytes() == before_evaluation
    finally:
        checkpoint.unlink(missing_ok=True)
        uninterrupted.unlink(missing_ok=True)


def test_agent_generation_uses_selected_orientation_and_scores_exact_action() -> None:
    trajectory = load_agent_trajectories(HELD_OUT_FIXTURE, max_records=1)[0]
    model = _ScriptedModel(canonical_agent_target(trajectory.action))

    generation = generate_agent_action(
        model,
        trajectory,
        max_new_tokens=128,
        device="cpu",
    )

    assert generation.terminated is True
    assert generation.schema_valid is True
    assert generation.canonical is True
    assert generation.exact_action is True
    assert generation.action_kind_correct is True
    assert generation.tool_selection_correct is True
    assert set(model.modes) == {trajectory.mode}
    assert set(model.visibility_modes) == {CognitiveMode.LACHESIS}
    assert set(model.source_lengths) == {
        len(canonical_agent_prompt(trajectory).encode("utf-8")) + 2
    }


def test_agent_evaluation_rejects_training_data_as_held_out() -> None:
    with pytest.raises(ValueError, match="held out"):
        evaluate_agent_jsonl(
            FIXTURE,
            training_path=FIXTURE,
            checkpoint_path=Path("research/missing-agent-checkpoint.pt"),
            max_records=1,
            max_new_tokens=4,
        )


def test_agent_evaluation_rejects_rows_copied_from_training() -> None:
    path = Path("research") / f".agent-overlap-{uuid.uuid4().hex}.jsonl"
    value = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    try:
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="overlap"):
            evaluate_agent_jsonl(
                path,
                training_path=FIXTURE,
                checkpoint_path=Path("research/missing-agent-checkpoint.pt"),
                max_records=1,
                max_new_tokens=4,
            )
    finally:
        path.unlink(missing_ok=True)


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
