"""Prompt-only DSPy/GEPA candidate backend for governed MOIRA research.

The backend deliberately receives an injected optimizer runner.  DSPy needs a
task-specific module, metric, and LM configuration; those must be frozen when
the worker starts, rather than supplied by a candidate-work request.  It never
trains weights, invokes MOIRA tools, or promotes an artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol

from .candidate_search import CandidateKind
from .candidate_worker import (
    CandidateWorkRequest,
    CandidateWorkUsage,
    CandidateWorkerOutput,
)


DSPY_GEPA_BACKEND_PROTOCOL = "moira-dspy-gepa-prompt-backend-v1"
MAX_PROMPT_RECORDS = 100_000
MAX_PROGRAM_BYTES = 1_048_576
MAX_OPTIMIZER_STEPS = 1_000_000
MAX_RECORD_TEXT_BYTES = 65_536


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """One immutable text row for a prompt/program optimization lane."""

    record_id: str
    input_text: str
    expected_text: str
    feedback: str = ""

    def __post_init__(self) -> None:
        _bounded_text("prompt record ID", self.record_id, 256)
        _bounded_text("prompt input", self.input_text, MAX_RECORD_TEXT_BYTES)
        _bounded_text("prompt expected output", self.expected_text, MAX_RECORD_TEXT_BYTES)
        if not isinstance(self.feedback, str) or "\x00" in self.feedback:
            raise ValueError("prompt feedback must be text without NUL bytes")
        if len(self.feedback.encode("utf-8")) > MAX_RECORD_TEXT_BYTES:
            raise ValueError("prompt feedback is outside its bounded range")

    @property
    def content_digest(self) -> str:
        return _digest(
            "moira-dspy-gepa-prompt-record-v1",
            {
                "input_text": self.input_text,
                "expected_text": self.expected_text,
                "feedback": self.feedback,
            },
        )

    def mapping(self) -> dict[str, str]:
        return {
            "record_id": self.record_id,
            "input_text": self.input_text,
            "expected_text": self.expected_text,
            "feedback": self.feedback,
        }


@dataclass(frozen=True, slots=True)
class PromptDatasetBundle:
    """Disjoint optimizer and selection rows; promotion holdouts are excluded."""

    optimizer_train: tuple[PromptRecord, ...]
    selection_evaluation: tuple[PromptRecord, ...]
    dataset_id: str
    preprocess_version: str = DSPY_GEPA_BACKEND_PROTOCOL

    def __post_init__(self) -> None:
        _bounded_text("prompt dataset ID", self.dataset_id, 256)
        _bounded_text("prompt preprocess version", self.preprocess_version, 256)
        _record_tuple("optimizer_train", self.optimizer_train)
        _record_tuple("selection_evaluation", self.selection_evaluation)
        if not self.optimizer_train or not self.selection_evaluation:
            raise ValueError("prompt optimization data lanes must be non-empty")
        train_ids = {record.record_id for record in self.optimizer_train}
        selection_ids = {record.record_id for record in self.selection_evaluation}
        if train_ids & selection_ids:
            raise ValueError("optimizer and selection record IDs must be disjoint")
        train_content = {record.content_digest for record in self.optimizer_train}
        selection_content = {
            record.content_digest for record in self.selection_evaluation
        }
        if train_content & selection_content:
            raise ValueError("optimizer and selection record content must be disjoint")

    @property
    def manifest_digest(self) -> str:
        return _digest(DSPY_GEPA_BACKEND_PROTOCOL, self.mapping())

    def mapping(self) -> dict[str, object]:
        return {
            "protocol": DSPY_GEPA_BACKEND_PROTOCOL,
            "dataset_id": self.dataset_id,
            "preprocess_version": self.preprocess_version,
            "optimizer_train": _record_selection(self.optimizer_train),
            "selection_evaluation": _record_selection(self.selection_evaluation),
            "is_promotion_holdout": False,
        }


@dataclass(frozen=True, slots=True)
class DspyGepaProfile:
    """Frozen identity and limits of an injected DSPy ``GEPA`` runner."""

    dspy_version: str
    provider_id: str
    model_id: str
    metric_version: str
    program_schema: str
    seed: int
    max_optimizer_steps: int
    max_program_bytes: int = MAX_PROGRAM_BYTES

    def __post_init__(self) -> None:
        for name, value in (
            ("DSPy version", self.dspy_version),
            ("provider ID", self.provider_id),
            ("model ID", self.model_id),
            ("metric version", self.metric_version),
            ("program schema", self.program_schema),
        ):
            _bounded_text(name, value, 512)
        _bounded_int("DSPy seed", self.seed, 0, (1 << 63) - 1)
        _bounded_int(
            "DSPy optimizer steps", self.max_optimizer_steps, 1, MAX_OPTIMIZER_STEPS
        )
        _bounded_int(
            "DSPy program bytes", self.max_program_bytes, 1, MAX_PROGRAM_BYTES
        )

    def mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DspyGepaRun:
    """Result returned by a preconfigured GEPA runner over optimizer rows only."""

    program: str
    trace_digest: str
    steps: int
    samples: int
    tokens: int

    def __post_init__(self) -> None:
        _bounded_text("optimized prompt program", self.program, MAX_PROGRAM_BYTES)
        _sha256("DSPy GEPA trace digest", self.trace_digest)
        _bounded_int("DSPy GEPA steps", self.steps, 0, MAX_OPTIMIZER_STEPS)
        _bounded_int("DSPy GEPA samples", self.samples, 0, 10_000_000_000)
        _bounded_int("DSPy GEPA tokens", self.tokens, 0, 10_000_000_000_000)


@dataclass(frozen=True, slots=True)
class PromptSelection:
    """A selection-only score from a separate, immutable record lane."""

    score: float
    viable: bool
    mutation_verified: bool
    remaining_failure_ids: tuple[str, ...]
    feedback_digest: str

    def __post_init__(self) -> None:
        if type(self.score) is not float or not _finite(self.score):
            raise ValueError("prompt selection score must be finite")
        if type(self.viable) is not bool or type(self.mutation_verified) is not bool:
            raise ValueError("prompt selection flags must be boolean")
        _unique_text_tuple(
            "remaining prompt failure IDs", self.remaining_failure_ids, 64
        )
        _sha256("prompt selection feedback digest", self.feedback_digest)


class DspyGepaRunner(Protocol):
    """A worker-startup configured callable which may invoke ``dspy.GEPA``."""

    def __call__(
        self,
        parent_program: str,
        optimizer_records: tuple[PromptRecord, ...],
        profile: DspyGepaProfile,
    ) -> DspyGepaRun: ...


class PromptSelectionEvaluator(Protocol):
    """Selection evaluation has no access to optimizer records or holdouts."""

    def __call__(
        self,
        program: str,
        selection_records: tuple[PromptRecord, ...],
        profile: DspyGepaProfile,
    ) -> PromptSelection: ...


class DspyGepaPromptBackend:
    """Emit one staged prompt candidate from a bounded, injected GEPA runner."""

    def __init__(
        self,
        *,
        parent_program: str | Path,
        dataset: PromptDatasetBundle,
        profile: DspyGepaProfile,
        runner: DspyGepaRunner,
        evaluator: PromptSelectionEvaluator,
    ) -> None:
        self.parent_program = _strict_file(parent_program, "DSPy GEPA parent program")
        if not isinstance(dataset, PromptDatasetBundle):
            raise ValueError("DSPy GEPA dataset is invalid")
        if not isinstance(profile, DspyGepaProfile):
            raise ValueError("DSPy GEPA profile is invalid")
        if not callable(runner) or not callable(evaluator):
            raise ValueError("DSPy GEPA runner and evaluator must be callable")
        self.dataset = dataset
        self.profile = profile
        self._runner = runner
        self._evaluator = evaluator

    @property
    def active_checkpoint_digest(self) -> str:
        return _file_sha256(self.parent_program)

    @property
    def dataset_manifest_digest(self) -> str:
        return self.dataset.manifest_digest

    @property
    def trainer_manifest_digest(self) -> str:
        return _digest(
            DSPY_GEPA_BACKEND_PROTOCOL,
            {
                "profile": self.profile.mapping(),
                "optimizer": "dspy.GEPA",
                "accepts_network_or_tool_config_from_request": False,
                "is_promotion_holdout": False,
            },
        )

    def __call__(
        self, request: CandidateWorkRequest, workspace: Path
    ) -> CandidateWorkerOutput:
        if not isinstance(request, CandidateWorkRequest):
            raise ValueError("DSPy GEPA candidate request is invalid")
        if request.kind is not CandidateKind.PROMPT:
            raise ValueError("DSPy GEPA backend accepts prompt candidates only")
        if request.active_checkpoint_digest != self.active_checkpoint_digest:
            raise ValueError("DSPy GEPA request does not bind the parent program")
        if request.dataset_manifest_digest != self.dataset_manifest_digest:
            raise ValueError("DSPy GEPA request dataset manifest does not match")
        if request.trainer_manifest_digest != self.trainer_manifest_digest:
            raise ValueError("DSPy GEPA request trainer manifest does not match")
        if not isinstance(workspace, Path) or not workspace.is_dir():
            raise ValueError("DSPy GEPA staging workspace is invalid")
        if any(workspace.iterdir()):
            raise ValueError("DSPy GEPA staging workspace must be empty")

        parent_program = self.parent_program.read_text(encoding="utf-8")
        _bounded_text("DSPy GEPA parent program", parent_program, MAX_PROGRAM_BYTES)
        run = self._runner(parent_program, self.dataset.optimizer_train, self.profile)
        if not isinstance(run, DspyGepaRun):
            raise ValueError("DSPy GEPA runner returned an invalid result")
        _preflight_usage(request, run, self.profile)
        selection = self._evaluator(
            run.program, self.dataset.selection_evaluation, self.profile
        )
        if not isinstance(selection, PromptSelection):
            raise ValueError("prompt selection evaluator returned an invalid result")
        if not set(selection.remaining_failure_ids).issubset(
            request.target_failure_ids
        ):
            raise ValueError("prompt selection references untargeted failure evidence")

        artifact_file = "program.json"
        creation_file = "creation_manifest.json"
        evaluation_file = "selection_evaluation.json"
        metric_file = "metrics.json"
        program_digest = _sha256_bytes(run.program.encode("utf-8"))
        _write_json(
            workspace / artifact_file,
            {
                "protocol": DSPY_GEPA_BACKEND_PROTOCOL,
                "program_schema": self.profile.program_schema,
                "program": run.program,
                "program_digest": program_digest,
            },
        )
        _write_json(
            workspace / creation_file,
            {
                "protocol": DSPY_GEPA_BACKEND_PROTOCOL,
                "request_digest": request.request_digest,
                "candidate_id": request.candidate_id,
                "lineage_id": request.lineage_id,
                "parent_ids": list(request.parent_ids),
                "parent_record_digests": list(request.parent_record_digests),
                "parent_program_digest": self.active_checkpoint_digest,
                "dataset_manifest_digest": self.dataset_manifest_digest,
                "trainer_manifest_digest": self.trainer_manifest_digest,
                "optimizer_train_manifest": _record_selection(
                    self.dataset.optimizer_train
                ),
                "profile": self.profile.mapping(),
                "runner_trace_digest": run.trace_digest,
                "can_authorize": False,
                "can_promote": False,
            },
        )
        _write_json(
            workspace / evaluation_file,
            {
                "protocol": DSPY_GEPA_BACKEND_PROTOCOL,
                "split": "selection-evaluation",
                "selection_manifest": _record_selection(
                    self.dataset.selection_evaluation
                ),
                "program_digest": program_digest,
                "score": selection.score,
                "viable": selection.viable,
                "mutation_verified": selection.mutation_verified,
                "remaining_failure_ids": list(selection.remaining_failure_ids),
                "feedback_digest": selection.feedback_digest,
                "is_selection_only": True,
                "is_promotion_holdout": False,
                "can_authorize": False,
                "can_promote": False,
            },
        )
        _write_json(
            workspace / metric_file,
            {
                "protocol": DSPY_GEPA_BACKEND_PROTOCOL,
                "optimizer": "dspy.GEPA",
                "steps": run.steps,
                "samples": run.samples,
                "tokens": run.tokens,
                "selection_score": selection.score,
                "selection_feedback_digest": selection.feedback_digest,
            },
        )
        return CandidateWorkerOutput(
            artifact_file=artifact_file,
            creation_manifest_file=creation_file,
            evaluation_manifest_file=evaluation_file,
            metric_file=metric_file,
            selection_score=selection.score,
            viable=selection.viable,
            mutation_verified=selection.mutation_verified,
            remaining_failure_ids=selection.remaining_failure_ids,
            usage=CandidateWorkUsage(
                steps=run.steps,
                samples=run.samples,
                tokens=run.tokens,
            ),
        )


def dspy_gepa_backend_contract() -> dict[str, object]:
    return {
        "protocol": DSPY_GEPA_BACKEND_PROTOCOL,
        "candidate_kind": CandidateKind.PROMPT.value,
        "optimizer": "dspy.GEPA",
        "requires_preconfigured_runner": True,
        "requires_disjoint_optimizer_and_selection_data": True,
        "accepts_network_or_tool_config_from_request": False,
        "trains_model_weights": False,
        "is_promotion_holdout": False,
        "can_authorize": False,
        "can_promote": False,
    }


def _preflight_usage(
    request: CandidateWorkRequest, run: DspyGepaRun, profile: DspyGepaProfile
) -> None:
    if run.steps > profile.max_optimizer_steps:
        raise ValueError("DSPy GEPA run exceeds its frozen optimizer-step profile")
    if (
        run.steps > request.budget.max_steps
        or run.samples > request.budget.max_samples
        or run.tokens > request.budget.max_tokens
    ):
        raise ValueError("DSPy GEPA run exceeds its candidate-work budget")
    if len(run.program.encode("utf-8")) > profile.max_program_bytes:
        raise ValueError("DSPy GEPA program exceeds its frozen profile budget")


def _record_selection(records: tuple[PromptRecord, ...]) -> dict[str, object]:
    return {
        "record_ids": [record.record_id for record in records],
        "record_digests": [record.content_digest for record in records],
        "count": len(records),
    }


def _record_tuple(name: str, records: object) -> None:
    if (
        not isinstance(records, tuple)
        or not 1 <= len(records) <= MAX_PROMPT_RECORDS
        or any(not isinstance(record, PromptRecord) for record in records)
    ):
        raise ValueError(f"{name} must be a bounded PromptRecord tuple")
    ids = [record.record_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} record IDs must be unique")


def _strict_file(path: str | Path, name: str) -> Path:
    candidate = Path(path).absolute()
    try:
        for component in (candidate, *candidate.parents):
            is_junction = getattr(component, "is_junction", lambda: False)
            if component.is_symlink() or is_junction():
                raise ValueError(f"{name} path may not contain links")
        if not candidate.is_file():
            raise ValueError(f"{name} must be a regular file")
        status = candidate.stat()
        if status.st_nlink != 1 or status.st_size <= 0:
            raise ValueError(f"{name} must be a non-empty, non-hard-linked file")
        return candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} cannot be inspected") from error


def _file_sha256(path: Path) -> str:
    _strict_file(path, "DSPy GEPA parent program")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _bounded_text(name: str, value: object, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValueError(f"{name} must be bounded non-empty text")


def _bounded_int(name: str, value: object, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its bounded range")


def _unique_text_tuple(name: str, value: object, maximum: int) -> None:
    if not isinstance(value, tuple) or len(value) > maximum:
        raise ValueError(f"{name} is outside its bounded tuple contract")
    for item in value:
        _bounded_text(name, item, 256)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must be unique")


def _sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(domain: str, value: object) -> str:
    return hashlib.sha256(
        domain.encode("ascii") + b"\x00" + _canonical_bytes(value)
    ).hexdigest()


__all__ = [
    "DSPY_GEPA_BACKEND_PROTOCOL",
    "DspyGepaProfile",
    "DspyGepaPromptBackend",
    "DspyGepaRun",
    "DspyGepaRunner",
    "PromptDatasetBundle",
    "PromptRecord",
    "PromptSelection",
    "PromptSelectionEvaluator",
    "dspy_gepa_backend_contract",
]
