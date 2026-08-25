import json
from pathlib import Path
import uuid

import pytest
import torch

from moira.dataset import normalize_row
from moira.m1_train import ByteTokenizer, build_causal_batch, iter_cataloged_text_examples


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
