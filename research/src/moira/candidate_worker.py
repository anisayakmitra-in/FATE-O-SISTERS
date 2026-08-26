"""Permit-gated staging boundary for native MOIRA candidate work."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

from .agent import MoiraTool
from .candidate_search import (
    CandidateArtifact,
    CandidateEvaluation,
    CandidateKind,
    CandidateRecord,
    CandidateSearch,
    CandidateSelectionReceipt,
    FailureSelectionReceipt,
)


CANDIDATE_WORK_PROTOCOL = "moira-candidate-work-v1"
CANDIDATE_WORK_TOOL_ID = "moira-candidate-worker"
CANDIDATE_WORK_CAPABILITY = "candidate_training"
CANDIDATE_WORK_OPERATION = "run"
CANDIDATE_WORK_TARGET_KIND = "candidate_work"
MAX_WORK_DURATION_MS = 7 * 24 * 60 * 60 * 1_000
MAX_WORK_STEPS = 10_000_000
MAX_WORK_SAMPLES = 10_000_000_000
MAX_WORK_TOKENS = 10_000_000_000_000
MAX_WORK_OUTPUT_BYTES = 16 * 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CandidateWorkBudget:
    max_steps: int
    max_samples: int
    max_tokens: int
    max_wall_ms: int
    max_output_bytes: int

    def __post_init__(self) -> None:
        _positive_int("max_steps", self.max_steps, MAX_WORK_STEPS)
        _positive_int("max_samples", self.max_samples, MAX_WORK_SAMPLES)
        _positive_int("max_tokens", self.max_tokens, MAX_WORK_TOKENS)
        _positive_int("max_wall_ms", self.max_wall_ms, MAX_WORK_DURATION_MS)
        _positive_int(
            "max_output_bytes", self.max_output_bytes, MAX_WORK_OUTPUT_BYTES
        )

    def mapping(self) -> dict[str, int]:
        return {
            "max_steps": self.max_steps,
            "max_samples": self.max_samples,
            "max_tokens": self.max_tokens,
            "max_wall_ms": self.max_wall_ms,
            "max_output_bytes": self.max_output_bytes,
        }


@dataclass(frozen=True, slots=True)
class CandidateWorkRequest:
    work_id: str
    candidate_id: str
    lineage_id: str
    kind: CandidateKind
    iteration: int
    parent_ids: tuple[str, ...]
    parent_record_digests: tuple[str, ...]
    target_failure_ids: tuple[str, ...]
    parent_selection_digest: str
    failure_selection_digest: str
    active_checkpoint_digest: str
    dataset_manifest_digest: str
    trainer_manifest_digest: str
    issued_at_ms: int
    expires_at_ms: int
    budget: CandidateWorkBudget

    def __post_init__(self) -> None:
        for name, value in (
            ("work_id", self.work_id),
            ("candidate_id", self.candidate_id),
            ("lineage_id", self.lineage_id),
        ):
            _bounded_text(name, value, 256)
        if not isinstance(self.kind, CandidateKind):
            raise ValueError("candidate work kind is invalid")
        if type(self.iteration) is not int or self.iteration <= 0:
            raise ValueError("candidate work iteration must be positive")
        _unique_text_tuple("parent_ids", self.parent_ids, 16)
        if not self.parent_ids:
            raise ValueError("candidate work requires at least one parent")
        if (
            not isinstance(self.parent_record_digests, tuple)
            or len(self.parent_record_digests) != len(self.parent_ids)
        ):
            raise ValueError("parent record digests must align with parent IDs")
        for digest in self.parent_record_digests:
            _sha256("parent record digest", digest)
        _unique_text_tuple("target_failure_ids", self.target_failure_ids, 64)
        if not self.target_failure_ids:
            raise ValueError("candidate work requires target failure evidence")
        for name, value in (
            ("parent_selection_digest", self.parent_selection_digest),
            ("failure_selection_digest", self.failure_selection_digest),
            ("active_checkpoint_digest", self.active_checkpoint_digest),
            ("dataset_manifest_digest", self.dataset_manifest_digest),
            ("trainer_manifest_digest", self.trainer_manifest_digest),
        ):
            _sha256(name, value)
        if (
            type(self.issued_at_ms) is not int
            or type(self.expires_at_ms) is not int
            or self.issued_at_ms < 0
            or self.expires_at_ms <= self.issued_at_ms
            or self.expires_at_ms - self.issued_at_ms > MAX_WORK_DURATION_MS
        ):
            raise ValueError("candidate work time bounds are invalid")
        if not isinstance(self.budget, CandidateWorkBudget):
            raise ValueError("candidate work budget is invalid")
        if self.expires_at_ms - self.issued_at_ms > self.budget.max_wall_ms:
            raise ValueError("candidate work expiry exceeds its wall budget")

    def mapping(self) -> dict[str, object]:
        return {
            "protocol": CANDIDATE_WORK_PROTOCOL,
            "work_id": self.work_id,
            "candidate_id": self.candidate_id,
            "lineage_id": self.lineage_id,
            "kind": self.kind.value,
            "iteration": self.iteration,
            "parent_ids": list(self.parent_ids),
            "parent_record_digests": list(self.parent_record_digests),
            "target_failure_ids": list(self.target_failure_ids),
            "parent_selection_digest": self.parent_selection_digest,
            "failure_selection_digest": self.failure_selection_digest,
            "active_checkpoint_digest": self.active_checkpoint_digest,
            "dataset_manifest_digest": self.dataset_manifest_digest,
            "trainer_manifest_digest": self.trainer_manifest_digest,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "budget": self.budget.mapping(),
            "can_authorize": False,
            "can_promote": False,
        }

    @property
    def payload(self) -> bytes:
        return _canonical_bytes(self.mapping())

    @property
    def request_digest(self) -> str:
        return _digest("moira-candidate-work-request-v1", self.mapping())


@dataclass(frozen=True, slots=True)
class CandidateWorkUsage:
    steps: int
    samples: int
    tokens: int

    def __post_init__(self) -> None:
        _non_negative_int("work steps", self.steps, MAX_WORK_STEPS)
        _non_negative_int("work samples", self.samples, MAX_WORK_SAMPLES)
        _non_negative_int("work tokens", self.tokens, MAX_WORK_TOKENS)


@dataclass(frozen=True, slots=True)
class CandidateWorkerOutput:
    artifact_file: str
    creation_manifest_file: str
    evaluation_manifest_file: str
    metric_file: str
    selection_score: float
    viable: bool
    mutation_verified: bool
    remaining_failure_ids: tuple[str, ...]
    usage: CandidateWorkUsage

    def __post_init__(self) -> None:
        for name, value in (
            ("artifact_file", self.artifact_file),
            ("creation_manifest_file", self.creation_manifest_file),
            ("evaluation_manifest_file", self.evaluation_manifest_file),
            ("metric_file", self.metric_file),
        ):
            _relative_file(name, value)
        output_files = (
            self.artifact_file,
            self.creation_manifest_file,
            self.evaluation_manifest_file,
            self.metric_file,
        )
        if len(set(output_files)) != len(output_files):
            raise ValueError("candidate output files must be distinct")
        if type(self.selection_score) is not float or not _finite(
            self.selection_score
        ):
            raise ValueError("candidate work score must be finite")
        if type(self.viable) is not bool or type(self.mutation_verified) is not bool:
            raise ValueError("candidate work evaluation flags must be boolean")
        _unique_text_tuple(
            "remaining_failure_ids", self.remaining_failure_ids, 64
        )
        if not isinstance(self.usage, CandidateWorkUsage):
            raise ValueError("candidate work usage is invalid")


@dataclass(frozen=True, slots=True)
class CandidateWorkReceipt:
    protocol: str
    work_id: str
    request_digest: str
    result_digest: str
    artifact_file: str
    artifact_digest: str
    creation_manifest_digest: str
    evaluation_manifest_digest: str
    metric_digest: str
    output_bytes: int
    wall_ms: int
    candidate: CandidateRecord
    can_authorize: bool = False
    can_promote: bool = False

    def __post_init__(self) -> None:
        if self.protocol != CANDIDATE_WORK_PROTOCOL:
            raise ValueError("candidate work receipt protocol is invalid")
        _bounded_text("candidate receipt work_id", self.work_id, 256)
        for name, value in (
            ("candidate receipt request_digest", self.request_digest),
            ("candidate receipt result_digest", self.result_digest),
            ("candidate receipt artifact_digest", self.artifact_digest),
            ("candidate receipt creation_manifest_digest", self.creation_manifest_digest),
            ("candidate receipt evaluation_manifest_digest", self.evaluation_manifest_digest),
            ("candidate receipt metric_digest", self.metric_digest),
        ):
            _sha256(name, value)
        _relative_file("candidate receipt artifact_file", self.artifact_file)
        _non_negative_int(
            "candidate receipt output_bytes", self.output_bytes, MAX_WORK_OUTPUT_BYTES
        )
        _non_negative_int(
            "candidate receipt wall_ms", self.wall_ms, MAX_WORK_DURATION_MS
        )
        if not isinstance(self.candidate, CandidateRecord):
            raise ValueError("candidate work receipt record is invalid")
        if self.can_authorize is not False or self.can_promote is not False:
            raise ValueError("candidate work receipts cannot authorize or promote")

    def mapping(self) -> dict[str, object]:
        return {
            "protocol": self.protocol,
            "work_id": self.work_id,
            "request_digest": self.request_digest,
            "result_digest": self.result_digest,
            "artifact_file": self.artifact_file,
            "artifact_digest": self.artifact_digest,
            "creation_manifest_digest": self.creation_manifest_digest,
            "evaluation_manifest_digest": self.evaluation_manifest_digest,
            "metric_digest": self.metric_digest,
            "output_bytes": self.output_bytes,
            "wall_ms": self.wall_ms,
            "candidate_record_digest": self.candidate.record_digest,
            "can_authorize": False,
            "can_promote": False,
        }


CandidateBackend = Callable[[CandidateWorkRequest, Path], CandidateWorkerOutput]


def prepare_candidate_work(
    search: CandidateSearch,
    parent_selection: CandidateSelectionReceipt,
    failure_selection: FailureSelectionReceipt,
    *,
    work_id: str,
    candidate_id: str,
    kind: CandidateKind,
    active_checkpoint_digest: str,
    dataset_manifest_digest: str,
    trainer_manifest_digest: str,
    issued_at_ms: int,
    expires_at_ms: int,
    budget: CandidateWorkBudget,
) -> CandidateWorkRequest:
    if not isinstance(search, CandidateSearch):
        raise ValueError("candidate search archive is required")
    search.verify_parent_selection(parent_selection)
    search.verify_failure_selection(failure_selection)
    parent_ids = tuple(dict.fromkeys(parent_selection.selected_ids))
    if failure_selection.candidate_id not in parent_ids:
        raise ValueError("failure selection is not bound to a selected parent")
    records = {item.artifact.candidate_id: item for item in search.candidates}
    if candidate_id in records:
        raise ValueError("candidate work ID already exists in the archive")
    parents = tuple(records[parent_id] for parent_id in parent_ids)
    lineage_ids = {item.artifact.lineage_id for item in parents}
    if len(lineage_ids) != 1:
        raise ValueError("candidate work parents do not share one lineage")
    return CandidateWorkRequest(
        work_id=work_id,
        candidate_id=candidate_id,
        lineage_id=lineage_ids.pop(),
        kind=kind,
        iteration=parent_selection.iteration,
        parent_ids=parent_ids,
        parent_record_digests=tuple(item.record_digest for item in parents),
        target_failure_ids=failure_selection.selected_failure_ids,
        parent_selection_digest=parent_selection.selection_digest,
        failure_selection_digest=failure_selection.selection_digest,
        active_checkpoint_digest=active_checkpoint_digest,
        dataset_manifest_digest=dataset_manifest_digest,
        trainer_manifest_digest=trainer_manifest_digest,
        issued_at_ms=issued_at_ms,
        expires_at_ms=expires_at_ms,
        budget=budget,
    )


class CandidateWorker:
    """One-shot candidate tool backed by a pre-registered local trainer."""

    def __init__(
        self,
        backend: CandidateBackend,
        *,
        active_checkpoint: str | Path,
        staging_root: str | Path,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if not callable(backend):
            raise ValueError("candidate backend must be callable")
        self._backend = backend
        self._active_checkpoint = Path(active_checkpoint)
        _regular_file(self._active_checkpoint, "active checkpoint")
        self._staging_root = Path(staging_root)
        if self._staging_root.exists() and (
            not self._staging_root.is_dir() or self._staging_root.is_symlink()
        ):
            raise ValueError("candidate staging root must be a real directory")
        self._staging_root.mkdir(parents=True, exist_ok=True)
        self._staging_root = self._staging_root.resolve(strict=True)
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._requests: dict[str, CandidateWorkRequest] = {}
        self._consumed: set[str] = set()
        self._results: dict[str, CandidateWorkReceipt] = {}

    def issue(
        self,
        request: CandidateWorkRequest,
        *,
        search: CandidateSearch,
        parent_selection: CandidateSelectionReceipt,
        failure_selection: FailureSelectionReceipt,
    ) -> None:
        if not isinstance(request, CandidateWorkRequest):
            raise ValueError("candidate work request is invalid")
        expected = prepare_candidate_work(
            search,
            parent_selection,
            failure_selection,
            work_id=request.work_id,
            candidate_id=request.candidate_id,
            kind=request.kind,
            active_checkpoint_digest=request.active_checkpoint_digest,
            dataset_manifest_digest=request.dataset_manifest_digest,
            trainer_manifest_digest=request.trainer_manifest_digest,
            issued_at_ms=request.issued_at_ms,
            expires_at_ms=request.expires_at_ms,
            budget=request.budget,
        )
        if expected != request:
            raise ValueError("candidate work request does not match selection evidence")
        now = self._now()
        if not request.issued_at_ms <= now < request.expires_at_ms:
            raise ValueError("candidate work request is not currently valid")
        if request.active_checkpoint_digest != _file_sha256(self._active_checkpoint):
            raise ValueError("candidate work request does not bind the active checkpoint")
        if request.work_id in self._requests:
            raise ValueError("candidate work ID is already issued")
        if any(
            item.request_digest == request.request_digest
            for item in self._requests.values()
        ):
            raise ValueError("candidate work request is already issued")
        self._requests[request.work_id] = request

    def tool(self) -> MoiraTool:
        return MoiraTool(
            tool_id=CANDIDATE_WORK_TOOL_ID,
            capability=CANDIDATE_WORK_CAPABILITY,
            operation=CANDIDATE_WORK_OPERATION,
            target_kind=CANDIDATE_WORK_TARGET_KIND,
            handler=self._handle,
            description="Run one permit-bound MOIRA candidate job in staging.",
        )

    def result(self, work_id: str) -> CandidateWorkReceipt:
        result = self._results.get(work_id)
        if result is None:
            raise ValueError("candidate work result is unavailable")
        return result

    def _handle(self, target: str, payload: bytes) -> bytes:
        request = self._requests.get(target)
        if request is None:
            raise ValueError("candidate work target is not issued")
        if request.request_digest in self._consumed:
            raise ValueError("candidate work request is already consumed")
        if payload != request.payload:
            raise ValueError("candidate work payload does not match the issued request")
        self._consumed.add(request.request_digest)
        started = self._now()
        if not request.issued_at_ms <= started < request.expires_at_ms:
            raise ValueError("candidate work request expired before execution")
        active_before = _file_sha256(self._active_checkpoint)
        if active_before != request.active_checkpoint_digest:
            raise ValueError("active checkpoint changed before candidate work")
        workspace = self._staging_root / request.request_digest
        workspace.mkdir(parents=False, exist_ok=False)
        try:
            output = self._backend(request, workspace)
        finally:
            active_after = _file_sha256(self._active_checkpoint)
            if active_after != active_before:
                raise RuntimeError("active checkpoint changed during candidate work")
        completed = self._now()
        wall_ms = completed - started
        if completed >= request.expires_at_ms:
            raise ValueError("candidate work expired during execution")
        if wall_ms < 0 or wall_ms > request.budget.max_wall_ms:
            raise ValueError("candidate work exceeded its wall budget")
        receipt = self._finalize(request, output, workspace, wall_ms)
        self._results[request.work_id] = receipt
        return _canonical_bytes(receipt.mapping())

    def _finalize(
        self,
        request: CandidateWorkRequest,
        output: CandidateWorkerOutput,
        workspace: Path,
        wall_ms: int,
    ) -> CandidateWorkReceipt:
        if not isinstance(output, CandidateWorkerOutput):
            raise ValueError("candidate backend returned an invalid result")
        usage = output.usage
        if (
            usage.steps > request.budget.max_steps
            or usage.samples > request.budget.max_samples
            or usage.tokens > request.budget.max_tokens
        ):
            raise ValueError("candidate work exceeded its training budget")
        if not set(output.remaining_failure_ids).issubset(
            request.target_failure_ids
        ):
            raise ValueError("candidate result references untargeted failure evidence")
        paths = tuple(
            _workspace_file(workspace, name)
            for name in (
                output.artifact_file,
                output.creation_manifest_file,
                output.evaluation_manifest_file,
                output.metric_file,
            )
        )
        declared_files = {
            Path(name).as_posix()
            for name in (
                output.artifact_file,
                output.creation_manifest_file,
                output.evaluation_manifest_file,
                output.metric_file,
            )
        }
        if _workspace_files(workspace) != declared_files:
            raise ValueError("candidate workspace contains undeclared output files")
        sizes = tuple(path.stat().st_size for path in paths)
        output_bytes = sum(sizes)
        if output_bytes > request.budget.max_output_bytes:
            raise ValueError("candidate work exceeded its output budget")
        artifact_digest, creation_digest, evaluation_digest, metric_digest = (
            _file_sha256(path) for path in paths
        )
        if artifact_digest == request.active_checkpoint_digest:
            raise ValueError("candidate artifact is identical to the active checkpoint")
        candidate = CandidateRecord(
            artifact=CandidateArtifact(
                candidate_id=request.candidate_id,
                lineage_id=request.lineage_id,
                kind=request.kind,
                artifact_digest=artifact_digest,
                creation_manifest_digest=creation_digest,
                parent_ids=request.parent_ids,
                iteration=request.iteration,
            ),
            evaluation=CandidateEvaluation(
                evaluation_manifest_digest=evaluation_digest,
                metric_digest=metric_digest,
                selection_score=output.selection_score,
                viable=output.viable,
                mutation_verified=output.mutation_verified,
                failure_ids=output.remaining_failure_ids,
            ),
        )
        result_value = {
            "protocol": CANDIDATE_WORK_PROTOCOL,
            "work_id": request.work_id,
            "request_digest": request.request_digest,
            "artifact_file": output.artifact_file,
            "artifact_digest": artifact_digest,
            "creation_manifest_digest": creation_digest,
            "evaluation_manifest_digest": evaluation_digest,
            "metric_digest": metric_digest,
            "output_bytes": output_bytes,
            "wall_ms": wall_ms,
            "usage": {
                "steps": usage.steps,
                "samples": usage.samples,
                "tokens": usage.tokens,
            },
            "candidate_record_digest": candidate.record_digest,
            "can_authorize": False,
            "can_promote": False,
        }
        result_digest = _digest("moira-candidate-work-result-v1", result_value)
        return CandidateWorkReceipt(
            protocol=CANDIDATE_WORK_PROTOCOL,
            work_id=request.work_id,
            request_digest=request.request_digest,
            result_digest=result_digest,
            artifact_file=output.artifact_file,
            artifact_digest=artifact_digest,
            creation_manifest_digest=creation_digest,
            evaluation_manifest_digest=evaluation_digest,
            metric_digest=metric_digest,
            output_bytes=output_bytes,
            wall_ms=wall_ms,
            candidate=candidate,
        )

    def _now(self) -> int:
        value = self._clock_ms()
        if type(value) is not int or value < 0:
            raise ValueError("candidate worker clock must return non-negative milliseconds")
        return value


def candidate_worker_contract() -> dict[str, object]:
    return {
        "protocol": CANDIDATE_WORK_PROTOCOL,
        "agent_action": f"{CANDIDATE_WORK_CAPABILITY}:{CANDIDATE_WORK_OPERATION}",
        "requires_native_one_use_effect_permit": True,
        "accepts_commands_from_request": False,
        "dedicated_staging_workspace": True,
        "rejects_undeclared_output_files": True,
        "hashes_output_artifacts": True,
        "checks_active_checkpoint_before_and_after": True,
        "os_read_only_mount_required_for_prevention": True,
        "can_authorize": False,
        "can_promote": False,
    }


def _workspace_file(workspace: Path, relative: str) -> Path:
    unresolved = workspace / relative
    current = workspace
    for part in Path(relative).parts:
        current = current / part
        is_junction = getattr(current, "is_junction", lambda: False)
        if current.is_symlink() or is_junction():
            raise ValueError("candidate output path may not contain links")
    candidate = unresolved.resolve(strict=True)
    if not candidate.is_relative_to(workspace.resolve(strict=True)):
        raise ValueError("candidate output escapes its staging workspace")
    _regular_file(candidate, "candidate output")
    return candidate


def _workspace_files(workspace: Path) -> set[str]:
    files: set[str] = set()
    try:
        for candidate in workspace.rglob("*"):
            is_junction = getattr(candidate, "is_junction", lambda: False)
            if candidate.is_symlink() or is_junction():
                raise ValueError("candidate workspace may not contain links")
            if candidate.is_dir():
                continue
            _regular_file(candidate, "candidate output")
            files.add(candidate.relative_to(workspace).as_posix())
    except OSError as error:
        raise ValueError("candidate workspace cannot be inspected") from error
    return files


def _regular_file(path: Path, name: str) -> None:
    try:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{name} must be a regular non-symlink file")
        status = path.stat()
        if status.st_nlink != 1:
            raise ValueError(f"{name} may not be a hard-linked file")
        if status.st_size <= 0:
            raise ValueError(f"{name} must not be empty")
    except OSError as error:
        raise ValueError(f"{name} cannot be inspected") from error


def _file_sha256(path: Path) -> str:
    _regular_file(path, "candidate artifact")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError("candidate artifact cannot be read") from error
    return digest.hexdigest()


def _relative_file(name: str, value: object) -> None:
    _bounded_text(name, value, 512)
    assert isinstance(value, str)
    path = Path(value)
    if path.is_absolute() or path.drive or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{name} must be a bounded relative file path")


def _unique_text_tuple(name: str, value: object, maximum: int) -> None:
    if not isinstance(value, tuple) or len(value) > maximum:
        raise ValueError(f"{name} is outside its bounded tuple contract")
    for item in value:
        _bounded_text(name, item, 256)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must be unique")


def _bounded_text(name: str, value: object, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValueError(f"{name} must be bounded non-empty text")


def _positive_int(name: str, value: object, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} is outside its positive bounded range")


def _non_negative_int(name: str, value: object, maximum: int) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} is outside its non-negative bounded range")


def _sha256(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


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
    "CANDIDATE_WORK_PROTOCOL",
    "CANDIDATE_WORK_TOOL_ID",
    "CANDIDATE_WORK_CAPABILITY",
    "CANDIDATE_WORK_OPERATION",
    "CANDIDATE_WORK_TARGET_KIND",
    "CandidateBackend",
    "CandidateWorkBudget",
    "CandidateWorkReceipt",
    "CandidateWorkRequest",
    "CandidateWorkUsage",
    "CandidateWorker",
    "CandidateWorkerOutput",
    "candidate_worker_contract",
    "prepare_candidate_work",
]
