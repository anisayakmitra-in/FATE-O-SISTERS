import json
from dataclasses import replace
from pathlib import Path
import uuid

import pytest

from moira.dataset import (
    DatasetManifest,
    SourceCatalog,
    assign_split,
    canonical_sha256,
    iter_jsonl_examples,
    normalize_row,
)


CATALOG = "research/configs/m1-sources.toml"


def test_m1_catalog_contains_datasets_and_teacher_references() -> None:
    catalog = SourceCatalog.from_toml(CATALOG)

    assert len(catalog.sources) == 17
    assert catalog.by_id("zai-longcite-45k").license == "Apache-2.0"
    assert catalog.by_id("chartqa").modality == "vision-text"
    assert catalog.by_id("polyai-minds14").modality == "audio"
    assert catalog.by_id("lerobot-pusht").modality == "robotics"
    assert catalog.by_id("zai-glm-4.1v-9b-thinking").modality == "teacher"
    assert catalog.by_id("xiaomi-mimo-embodied-eval").modality == "evaluation"


@pytest.mark.parametrize(
    ("source_id", "row", "modality", "input_key", "target_key"),
    [
        (
            "zai-longcite-45k",
            {"prompt": "question", "response": "answer", "source": "paper"},
            "text",
            "text",
            "text",
        ),
        (
            "huggingface-smol-smoltalk",
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
                "source": "synthetic",
            },
            "text",
            "text",
            "text",
        ),
        (
            "deepseek-r1-openr1-math-220k",
            {"problem": "2+2", "solution": "4", "uuid": "math-1"},
            "text",
            "text",
            "text",
        ),
        (
            "nvidia-opencode-reasoning",
            {"id": "code-1", "input": "print(2+2)", "solution": "4"},
            "code",
            "text",
            "text",
        ),
        (
            "chartqa",
            {
                "query": "What is the value?",
                "image": {"src": "https://example.invalid/chart.png"},
                "label": ["42"],
            },
            "vision-text",
            "text",
            "labels",
        ),
        (
            "spawning-pd12m",
            {"url": "https://example.invalid/image.jpg", "caption": "a chart"},
            "vision-text",
            "image",
            "text",
        ),
        (
            "polyai-minds14",
            {
                "path": "en-US/response.wav",
                "transcription": "hello",
                "intent_class": 1,
            },
            "audio",
            "audio",
            "text",
        ),
        (
            "lerobot-pusht",
            {"observation.state": [1.0, 2.0], "action": [3.0, 4.0]},
            "robotics",
            "state",
            "action",
        ),
    ],
)
def test_known_sources_normalize_to_bounded_records(
    source_id: str,
    row: dict[str, object],
    modality: str,
    input_key: str,
    target_key: str,
) -> None:
    example = normalize_row(
        source_id,
        row,
        record_id="record-1",
        split="train",
        preprocess_version="m1-v1",
    )

    assert example.modality == modality
    assert input_key in example.inputs
    assert target_key in example.targets
    assert example.record_sha256 == canonical_sha256(row)
    example.validate()


def test_jsonl_stream_is_bounded_and_skips_blank_lines() -> None:
    path = Path("research") / f".dataset-test-{uuid.uuid4().hex}.jsonl"
    try:
        path.write_text(
            "\n".join(
                [
                    json.dumps({"id": "a", "prompt": "p1", "response": "r1"}),
                    "",
                    json.dumps({"id": "b", "prompt": "p2", "response": "r2"}),
                    json.dumps({"id": "c", "prompt": "p3", "response": "r3"}),
                ]
            ),
            encoding="utf-8",
        )
        examples = tuple(
            iter_jsonl_examples(
                path,
                source_id="zai-longcite-45k",
                split="auto",
                preprocess_version="m1-v1",
                split_seed=7,
                split_policy=(("train", 8), ("eval", 2)),
                max_records=2,
            )
        )

        assert [example.record_id for example in examples] == ["a", "b"]
        assert all(example.split in {"train", "eval"} for example in examples)
    finally:
        path.unlink(missing_ok=True)


def test_split_assignment_is_deterministic_and_validates_policy() -> None:
    policy = (("train", 8), ("eval", 2))
    assert assign_split("same", seed=10, policy=policy) == assign_split(
        "same", seed=10, policy=policy
    )
    with pytest.raises(ValueError, match="unique"):
        assign_split("same", seed=10, policy=(("train", 1), ("train", 1)))


def test_manifest_must_match_catalog_revision() -> None:
    catalog = SourceCatalog.from_toml(CATALOG)
    manifest = DatasetManifest(
        source_id="zai-longcite-45k",
        source_revision=catalog.by_id("zai-longcite-45k").revision,
        config="default",
        split="train",
        format="jsonl",
        local_path=None,
        preprocess_version="m1-v1",
        split_seed=1,
        split_policy=(("train", 8), ("eval", 2)),
    )
    manifest.validate(catalog)
    with pytest.raises(ValueError, match="revision"):
        replace(manifest, source_revision="wrong").validate(catalog)
