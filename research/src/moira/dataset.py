"""Manifest-first records for licensed MOIRA M1 data sources.

The repository stores source descriptions and preprocessing contracts. Large
dataset files remain in a caller-owned Hugging Face cache or local workspace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch


DATASET_MANIFEST_VERSION = 1
MAX_CATALOG_BYTES = 1 * 1024 * 1024
MAX_RECORD_BYTES = 8 * 1024 * 1024
_SOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,127}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_KINDS = frozenset({"hf-dataset", "hf-model", "github"})
_MODALITIES = frozenset(
    {
        "text",
        "vision-text",
        "audio",
        "code",
        "structured",
        "robotics",
        "time-series",
        "teacher",
        "evaluation",
    }
)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One external source with its license and intended MOIRA role."""

    source_id: str
    kind: str
    uri: str
    license: str
    license_url: str
    modality: str
    purpose: str
    revision: str
    config: str = ""
    split: str = ""
    row_count: int | None = None
    byte_count: int | None = None
    license_verified: bool = False
    upstream: str = ""
    notes: str = ""

    def validate(self) -> None:
        if not isinstance(self.source_id, str):
            raise ValueError("source_id must be a string")
        if not _SOURCE_ID.fullmatch(self.source_id):
            raise ValueError("source_id must be a lowercase stable identifier")
        if not isinstance(self.kind, str) or self.kind not in _SOURCE_KINDS:
            raise ValueError("unsupported source kind")
        if not isinstance(self.uri, str) or not self.uri.startswith(("https://", "hf://")):
            raise ValueError("source uri must be an https or hf URI")
        for name, value in (
            ("license", self.license),
            ("license_url", self.license_url),
            ("modality", self.modality),
            ("purpose", self.purpose),
            ("revision", self.revision),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.modality not in _MODALITIES:
            raise ValueError("unsupported source modality")
        if type(self.license_verified) is not bool:
            raise ValueError("license_verified must be boolean")
        if self.license_verified and not self.license_url.startswith("https://"):
            raise ValueError("verified licenses require an https license_url")
        if self.row_count is not None and (
            type(self.row_count) is not int or self.row_count <= 0
        ):
            raise ValueError("row_count must be a positive integer when present")
        if self.byte_count is not None and (
            type(self.byte_count) is not int or self.byte_count <= 0
        ):
            raise ValueError("byte_count must be a positive integer when present")


@dataclass(frozen=True, slots=True)
class SourceCatalog:
    schema_version: int
    sources: tuple[SourceSpec, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != DATASET_MANIFEST_VERSION:
            raise ValueError("unsupported source catalog schema_version")
        if not self.sources:
            raise ValueError("source catalog must contain at least one source")
        identifiers = set()
        for source in self.sources:
            if not isinstance(source, SourceSpec):
                raise ValueError("sources must contain SourceSpec values")
            source.validate()
            if source.source_id in identifiers:
                raise ValueError("source_id values must be unique")
            identifiers.add(source.source_id)

    def by_id(self, source_id: str) -> SourceSpec:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(source_id)

    @classmethod
    def from_toml(cls, path: str | Path) -> "SourceCatalog":
        raw = tomllib.loads(_read_bounded_text(Path(path)))
        if set(raw) != {"schema_version", "sources"}:
            raise ValueError("source catalog fields are not exact")
        if type(raw["schema_version"]) is not int:
            raise ValueError("source catalog schema_version must be an integer")
        entries = raw["sources"]
        if not isinstance(entries, list):
            raise ValueError("source catalog sources must be an array of tables")
        fields = {field for field in SourceSpec.__dataclass_fields__}
        required = {
            "source_id",
            "kind",
            "uri",
            "license",
            "license_url",
            "modality",
            "purpose",
            "revision",
        }
        sources: list[SourceSpec] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"source entry {index} must be a table")
            unknown = set(entry) - fields
            missing = required - set(entry)
            if unknown or missing:
                raise ValueError(
                    f"source entry {index} has invalid fields "
                    f"(missing={sorted(missing)}, unknown={sorted(unknown)})"
                )
            sources.append(SourceSpec(**entry))
        return cls(raw["schema_version"], tuple(sources))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sources": [asdict(source) for source in self.sources],
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """A local materialization bound to one catalog source."""

    source_id: str
    source_revision: str
    config: str
    split: str
    format: str
    local_path: str | None
    preprocess_version: str
    split_seed: int
    split_policy: tuple[tuple[str, int], ...]
    content_sha256: str | None = None
    num_rows: int | None = None

    def validate(self, catalog: SourceCatalog | None = None) -> None:
        if not isinstance(self.source_id, str) or not isinstance(
            self.source_revision, str
        ):
            raise ValueError("manifest source identity must be strings")
        if catalog is not None:
            source = catalog.by_id(self.source_id)
            if source.revision != self.source_revision:
                raise ValueError("manifest revision does not match source catalog")
        for name, value in (
            ("source_id", self.source_id),
            ("source_revision", self.source_revision),
            ("config", self.config),
            ("split", self.split),
            ("preprocess_version", self.preprocess_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"manifest {name} must be non-empty")
        if not isinstance(self.format, str) or self.format not in {
            "jsonl",
            "json",
            "parquet",
            "image-folder",
        }:
            raise ValueError("unsupported manifest format")
        if type(self.split_seed) is not int or self.split_seed < 0:
            raise ValueError("split_seed must be a non-negative integer")
        _validate_split_policy(self.split_policy)
        if self.content_sha256 is not None and (
            not isinstance(self.content_sha256, str)
            or not _HASH.fullmatch(self.content_sha256)
        ):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")
        if self.num_rows is not None and (
            type(self.num_rows) is not int or self.num_rows <= 0
        ):
            raise ValueError("num_rows must be positive when present")


@dataclass(frozen=True, slots=True)
class DatasetExample:
    """Normalized example that can enter a text or capability adapter."""

    record_id: str
    source_id: str
    modality: str
    split: str
    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    record_sha256: str
    preprocess_version: str
    metadata: Mapping[str, Any]

    def validate(self) -> None:
        if not isinstance(self.record_id, str) or not isinstance(
            self.source_id, str
        ):
            raise ValueError("example identity must use strings")
        if not isinstance(self.split, str) or not isinstance(
            self.preprocess_version, str
        ):
            raise ValueError("example split and preprocess_version must use strings")
        if not self.record_id.strip() or not self.source_id.strip():
            raise ValueError("example identity must be non-empty")
        if not isinstance(self.modality, str) or self.modality not in _MODALITIES:
            raise ValueError("unsupported example modality")
        if not self.split.strip() or not self.preprocess_version.strip():
            raise ValueError("example split and preprocess_version must be non-empty")
        if not isinstance(self.record_sha256, str) or not _HASH.fullmatch(
            self.record_sha256
        ):
            raise ValueError("record_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.inputs, Mapping) or not isinstance(
            self.targets, Mapping
        ):
            raise ValueError("example inputs and targets must be mappings")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("example metadata must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "record_id": self.record_id,
            "source_id": self.source_id,
            "modality": self.modality,
            "split": self.split,
            "inputs": dict(self.inputs),
            "targets": dict(self.targets),
            "record_sha256": self.record_sha256,
            "preprocess_version": self.preprocess_version,
            "metadata": dict(self.metadata),
        }


def iter_jsonl_examples(
    path: str | Path,
    *,
    source_id: str,
    split: str,
    preprocess_version: str,
    split_seed: int,
    split_policy: tuple[tuple[str, int], ...] | None = None,
    max_records: int | None = None,
) -> Iterator[DatasetExample]:
    """Read bounded JSONL rows without loading the complete file."""
    path = Path(path)
    if max_records is not None and (
        type(max_records) is not int or max_records <= 0
    ):
        raise ValueError("max_records must be positive when present")
    if split == "auto" and split_policy is None:
        raise ValueError("split_policy is required for automatic splitting")
    emitted = 0
    with path.open("rb") as handle:
        for row_index, raw_line in enumerate(handle):
            if not raw_line.strip():
                continue
            if len(raw_line) > MAX_RECORD_BYTES:
                raise ValueError("JSONL record exceeds the bounded record limit")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL record at line {row_index + 1}") from error
            if not isinstance(row, Mapping):
                raise ValueError("JSONL records must be objects")
            record_id = _record_id(row, row_index)
            row_split = (
                assign_split(record_id, seed=split_seed, policy=split_policy)
                if split == "auto"
                else split
            )
            yield normalize_row(
                source_id,
                row,
                record_id=record_id,
                split=row_split,
                preprocess_version=preprocess_version,
            )
            emitted += 1
            if max_records is not None and emitted >= max_records:
                return


def normalize_row(
    source_id: str,
    row: Mapping[str, Any],
    *,
    record_id: str,
    split: str,
    preprocess_version: str,
) -> DatasetExample:
    """Normalize the known M1 sources and preserve unknown fields as metadata."""
    if source_id == "zai-longcite-45k":
        prompt = _string_field(row, "prompt")
        response = _string_field(row, "response")
        inputs, targets, modality = {"text": prompt}, {"text": response}, "text"
    elif source_id == "huggingface-smol-smoltalk":
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("SmolTalk row requires at least two messages")
        assistant_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], Mapping)
                and messages[index].get("role") == "assistant"
            ),
            None,
        )
        if assistant_index is None or assistant_index == 0:
            raise ValueError("SmolTalk row requires an assistant response")
        response = messages[assistant_index].get("content")
        if not isinstance(response, str) or not response.strip():
            raise ValueError("SmolTalk assistant content must be non-empty")
        prompt = _messages_to_text(messages[:assistant_index])
        inputs, targets, modality = {"text": prompt}, {"text": response}, "text"
    elif source_id == "deepseek-r1-openr1-math-220k":
        problem = _string_field(row, "problem")
        solution = row.get("solution", row.get("answer", ""))
        if isinstance(solution, list):
            solution = solution[0] if solution else ""
        if not isinstance(solution, str) or not solution.strip():
            messages = row.get("messages")
            if isinstance(messages, list) and messages:
                solution = messages[-1].get("content", "") if isinstance(messages[-1], Mapping) else ""
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError("OpenR1 row has no usable solution or answer")
        inputs, targets, modality = {"text": problem}, {"text": solution}, "text"
    elif source_id == "nvidia-openmathinstruct-2":
        problem = _string_field(row, "problem")
        solution = _string_field(row, "generated_solution")
        inputs, targets, modality = {"text": problem}, {"text": solution}, "text"
    elif source_id == "openai-gsm8k":
        problem = _string_field(row, "question")
        solution = _string_field(row, "answer")
        inputs, targets, modality = {"text": problem}, {"text": solution}, "text"
    elif source_id == "nvidia-opencode-reasoning":
        problem = _string_field(row, "input")
        solution = row.get("solution", row.get("output", ""))
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError("OpenCodeReasoning row has no usable solution or output")
        inputs, targets, modality = {"text": problem}, {"text": solution}, "code"
    elif source_id == "polyai-minds14":
        audio = row.get("audio", row.get("path"))
        audio_ref = _audio_reference(audio)
        transcription = row.get("transcription", row.get("english_transcription"))
        if not isinstance(transcription, str) or not transcription.strip():
            raise ValueError("Minds14 row requires a transcription")
        inputs, targets, modality = {"audio": audio_ref}, {"text": transcription}, "audio"
    elif source_id in {"facebook-voxpopuli", "google-fleurs"}:
        audio = row.get("audio", row.get("path"))
        audio_ref = _audio_reference(audio)
        transcription = row.get(
            "transcription",
            row.get("raw_text", row.get("normalized_text", row.get("text"))),
        )
        if not isinstance(transcription, str) or not transcription.strip():
            raise ValueError("speech row requires a transcription or text field")
        inputs, targets, modality = {"audio": audio_ref}, {"text": transcription}, "audio"
    elif source_id == "chartqa":
        query = _string_field(row, "query")
        image = row.get("image")
        image_ref = _image_reference(image)
        label = row.get("label")
        labels = label if isinstance(label, list) else [label]
        if not labels or any(not isinstance(value, str) for value in labels):
            raise ValueError("ChartQA label must be a non-empty string list")
        inputs = {"text": query, "image": image_ref}
        targets, modality = {"labels": labels}, "vision-text"
    elif source_id == "spawning-pd12m":
        image = _string_field(row, "url")
        caption = _string_field(row, "caption")
        inputs, targets, modality = {"image": image}, {"text": caption}, "vision-text"
    elif source_id == "lerobot-pusht":
        state = row.get("observation.state")
        action = row.get("action")
        if not _numeric_sequence(state) or not _numeric_sequence(action):
            raise ValueError("PushT row requires numeric observation.state and action")
        inputs, targets, modality = {"state": list(state)}, {"action": list(action)}, "robotics"
    elif source_id == "open-graph-benchmark":
        prompt = row.get("text", row.get("task", "graph prediction"))
        target = row.get("target", row.get("label", row.get("y")))
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("graph row requires a text or task field")
        if target is None:
            raise ValueError("graph row requires a target, label, or y field")
        inputs = {"text": prompt}
        targets = {"text": target} if isinstance(target, str) else {"target": target}
        modality = "structured"
    else:
        text = row.get("text", row.get("prompt", row.get("input")))
        target = row.get("target", row.get("response", row.get("output", "")))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("generic record requires text, prompt, or input")
        if not isinstance(target, str):
            target = json.dumps(target, sort_keys=True, ensure_ascii=False)
        inputs, targets, modality = {"text": text}, {"text": target}, "text"

    example = DatasetExample(
        record_id=record_id,
        source_id=source_id,
        modality=modality,
        split=split,
        inputs=inputs,
        targets=targets,
        record_sha256=canonical_sha256(row),
        preprocess_version=preprocess_version,
        metadata={
            key: value
            for key, value in row.items()
            if key not in inputs and key not in targets
        },
    )
    example.validate()
    return example


def assign_split(
    record_id: str,
    *,
    seed: int,
    policy: tuple[tuple[str, int], ...],
) -> str:
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("record_id must be non-empty")
    if type(seed) is not int or seed < 0:
        raise ValueError("split seed must be a non-negative integer")
    _validate_split_policy(policy)
    digest = sha256(f"{seed}:{record_id}".encode("utf-8")).digest()
    point = int.from_bytes(digest[:8], "big") % sum(weight for _, weight in policy)
    cursor = 0
    for name, weight in policy:
        cursor += weight
        if point < cursor:
            return name
    raise AssertionError("split policy did not select a name")


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_image_tensor(
    path: str | Path,
    *,
    size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Decode one RGB image as float CHW in [0, 1] using optional Pillow."""
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or any(type(value) is not int or value <= 0 for value in size)
    ):
        raise ValueError("size must contain two positive integers")
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required for image decoding") from error
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(size)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("decoded image contains non-finite values")
    return tensor


def _read_bounded_text(path: Path) -> str:
    data = path.read_bytes()
    if len(data) > MAX_CATALOG_BYTES:
        raise ValueError("source catalog exceeds the bounded size limit")
    return data.decode("utf-8")


def _validate_split_policy(policy: Sequence[tuple[str, int]]) -> None:
    if isinstance(policy, (str, bytes)) or not isinstance(policy, Sequence) or not policy:
        raise ValueError("split_policy must not be empty")
    names = set()
    for name, weight in policy:
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("split names must be unique and non-empty")
        if type(weight) is not int or weight <= 0:
            raise ValueError("split weights must be positive integers")
        names.add(name)


def _record_id(row: Mapping[str, Any], row_index: int) -> str:
    for key in ("id", "uuid", "record_id"):
        value = row.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value)
    return f"row-{row_index:012d}"


def _string_field(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row field {name!r} must be a non-empty string")
    return value


def _messages_to_text(messages: Sequence[object]) -> str:
    rendered: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("conversation messages must be objects")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError("conversation message role must be non-empty")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("conversation message content must be non-empty")
        rendered.append(f"{role.strip()}: {content}")
    if not rendered:
        raise ValueError("conversation prompt must be non-empty")
    return "\n\n".join(rendered)


def _audio_reference(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, Mapping):
        for key in ("path", "src", "url"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        samples = value.get("array")
        if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
            return "inline-audio"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if isinstance(item, Mapping):
                return _audio_reference(item)
        return "inline-audio"
    raise ValueError("audio row requires a path, URL, or encoded reference")


def _numeric_sequence(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return False
    return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)


def _image_reference(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, Mapping):
        for key in ("path", "src", "url", "bytes"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    raise ValueError("vision row requires an image path, URL, or encoded reference")
