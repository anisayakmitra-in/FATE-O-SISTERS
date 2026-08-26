import json
from pathlib import Path
import uuid

import pytest
import torch

from moira.config import ModelConfig
from moira.dataset import normalize_row
from moira.m1_train import (
    ByteTokenizer,
    CapabilityLoom,
    build_capability_batch,
    build_causal_batch,
    iter_cataloged_text_examples,
    train_m1_capabilities,
)
from moira.masks import CognitiveMode
from moira.model import MoiraModel


def test_byte_tokenizer_builds_target_only_causal_masks() -> None:
    batch = ByteTokenizer().encode_pair("prompt", "answer", sequence_length=32)

    assert batch.input_ids.shape == (1, 31)
    assert batch.targets.shape == (1, 31)
    assert batch.valid_tokens[0].sum() == 1 + len("prompt".encode()) + 1 + len("answer".encode()) + 1 - 1
    first_target_prediction = len("prompt".encode()) + 1
    assert not batch.loss_mask[0, :first_target_prediction].any()
    assert batch.loss_mask[0, first_target_prediction]


def test_build_causal_batch_rejects_non_text_capability_records() -> None:
    example = normalize_row(
        "chartqa",
        {
            "query": "what?",
            "image": "https://example.invalid/image.png",
            "label": ["1"],
        },
        record_id="chart-1",
        split="train",
        preprocess_version="m1-v1",
    )
    with pytest.raises(ValueError, match="text inputs"):
        build_causal_batch((example,), sequence_length=32)


def test_cataloged_text_iterator_keeps_streaming_contract() -> None:
    path = Path("research") / f".m1-jsonl-{uuid.uuid4().hex}.jsonl"
    try:
        path.write_text(
            json.dumps({"id": "row-1", "prompt": "p", "response": "r"}) + "\n",
            encoding="utf-8",
        )
        examples = tuple(
            iter_cataloged_text_examples(
                "research/configs/m1-sources.toml",
                path,
                source_id="zai-longcite-45k",
                max_records=1,
            )
        )
        assert [example.record_id for example in examples] == ["row-1"]
    finally:
        path.unlink(missing_ok=True)


def test_capability_batch_runs_through_the_model_prefix_boundary() -> None:
    config = ModelConfig(512, 32, 1, 4, 64, 3, 8, 4, 8, 64)
    model = MoiraModel(config)
    example = normalize_row(
        "chartqa",
        {"query": "what?", "image": "x.png", "label": ["1"]},
        record_id="chart-prefix",
        split="train",
        preprocess_version="m1-v1",
    )
    loom = CapabilityLoom(
        modality="vision",
        capsule_width=config.capsule_width,
        d_model=config.d_model,
        patch_size=2,
        max_slots=3,
    )
    prefix = loom(torch.rand(1, 3, 4, 4), (example,))
    batch = build_capability_batch(
        (example,), prefix_hidden=prefix, sequence_length=config.max_seq_len
    )
    output = model.forward_with_prefix(
        batch.input_ids,
        batch.prefix_hidden,
        mode=CognitiveMode.ATROPOS,
        valid_tokens=batch.valid_tokens,
        tokens_include_prefix=True,
    )
    text_only_output = model.forward_with_prefix(
        batch.input_ids[:, 3:],
        batch.prefix_hidden,
        mode=CognitiveMode.ATROPOS,
        valid_tokens=batch.valid_tokens[:, 3:],
    )
    assert batch.prefix_hidden.shape == (1, 3, config.d_model)
    assert output.logits.shape == (1, config.max_seq_len - 1, config.vocab_size)
    assert text_only_output.logits.shape == output.logits.shape
    assert batch.loss_mask[:, :3].sum() == 0
    loss = torch.nn.functional.cross_entropy(
        output.logits.reshape(-1, config.vocab_size),
        batch.targets.reshape(-1),
        reduction="none",
    )
    (loss * batch.loss_mask.reshape(-1)).sum().backward()
    assert loom.projector.projection.weight.grad is not None


def test_capability_training_writes_and_resumes_a_bounded_checkpoint() -> None:
    jsonl_path = Path("research") / f".m1-capability-{uuid.uuid4().hex}.jsonl"
    checkpoint_path = Path("research") / f".m1-checkpoint-{uuid.uuid4().hex}.pt"
    try:
        jsonl_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {"id": "chart-a", "query": "what?", "image": "a.png", "label": ["1"]}
                    ),
                    json.dumps(
                        {"id": "chart-b", "query": "where?", "image": "b.png", "label": ["2"]}
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        config = ModelConfig(512, 32, 1, 4, 64, 3, 8, 4, 8, 64)
        observations = torch.rand(2, 3, 4, 4)
        first = train_m1_capabilities(
            "research/configs/m1-sources.toml",
            jsonl_path,
            observations,
            source_id="chartqa",
            modality="vision",
            steps=1,
            batch_size=1,
            model_config=config,
            checkpoint_path=checkpoint_path,
            patch_size=2,
            max_slots=3,
        )
        assert first.checkpoint_path == checkpoint_path
        assert checkpoint_path.is_file()
        resumed = train_m1_capabilities(
            "research/configs/m1-sources.toml",
            jsonl_path,
            observations,
            source_id="chartqa",
            modality="vision",
            steps=2,
            batch_size=1,
            model_config=config,
            checkpoint_path=checkpoint_path,
            resume=True,
            patch_size=2,
            max_slots=3,
        )
        assert resumed.steps == 2
        assert len(resumed.losses) == 2
        assert all(torch.isfinite(torch.tensor(resumed.losses)))
    finally:
        jsonl_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
