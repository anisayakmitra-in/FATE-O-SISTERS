"""Bounded PEFT LoRA backend for permit-gated MOIRA candidate work."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW

from .candidate_search import CandidateKind
from .candidate_worker import (
    CandidateWorkRequest,
    CandidateWorkUsage,
    CandidateWorkerOutput,
)
from .config import ModelConfig
from .dataset import (
    DatasetExample,
    SourceCatalog,
    canonical_sha256,
    file_sha256,
    iter_jsonl_examples,
)
from .m1_checkpoint import load_m1_checkpoint
from .m1_train import CausalBatch, build_causal_batch
from .masks import CognitiveMode
from .model import MoiraModel


LORA_BACKEND_PROTOCOL = "moira-peft-lora-backend-v1"
LORA_DATASET_PROTOCOL = "moira-lora-dataset-v1"
MAX_LORA_RECORDS = 100_000
MAX_LORA_RANK = 256
MAX_LORA_STEPS = 1_000
MAX_LORA_BATCH_SIZE = 8
_ALLOWED_TARGET_MODULES = frozenset(
    {
        "query_projection",
        "key_projection",
        "value_projection",
        "output_projection",
    }
)


@dataclass(frozen=True, slots=True)
class LoraTrainingConfig:
    """Exact bounded optimizer surface for one adapter candidate."""

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "query_projection",
        "key_projection",
        "value_projection",
        "output_projection",
    )
    steps: int = 10
    batch_size: int = 1
    eval_batch_size: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    min_improvement: float = 0.0
    seed: int = 20260826
    device: str = "cpu"
    precision: str = "fp32"
    mode: CognitiveMode = CognitiveMode.ATROPOS

    def __post_init__(self) -> None:
        _bounded_int("LoRA rank", self.rank, 1, MAX_LORA_RANK)
        _bounded_int("LoRA alpha", self.alpha, 1, MAX_LORA_RANK * 16)
        _bounded_int("LoRA steps", self.steps, 1, MAX_LORA_STEPS)
        _bounded_int("LoRA batch size", self.batch_size, 1, MAX_LORA_BATCH_SIZE)
        _bounded_int(
            "LoRA evaluation batch size",
            self.eval_batch_size,
            1,
            MAX_LORA_BATCH_SIZE,
        )
        if (
            not isinstance(self.dropout, float)
            or not math.isfinite(self.dropout)
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("LoRA dropout must be finite and in [0, 1)")
        for name, value, allow_zero in (
            ("LoRA learning rate", self.learning_rate, False),
            ("LoRA weight decay", self.weight_decay, True),
            ("LoRA gradient norm", self.max_grad_norm, False),
        ):
            if (
                not isinstance(value, float)
                or not math.isfinite(value)
                or value < 0.0
                or (not allow_zero and value == 0.0)
            ):
                raise ValueError(f"{name} is invalid")
        if not isinstance(self.min_improvement, float) or not math.isfinite(
            self.min_improvement
        ):
            raise ValueError("LoRA minimum improvement must be finite")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("LoRA seed must be a non-negative integer")
        if (
            not isinstance(self.target_modules, tuple)
            or not self.target_modules
            or len(set(self.target_modules)) != len(self.target_modules)
            or not set(self.target_modules).issubset(_ALLOWED_TARGET_MODULES)
        ):
            raise ValueError("LoRA target modules are outside the MOIRA adapter surface")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("LoRA device must be cpu or cuda")
        if self.precision != "fp32":
            raise ValueError("the v1 LoRA backend supports exact fp32 training only")
        if self.mode is not CognitiveMode.ATROPOS:
            raise ValueError("the causal v1 LoRA backend requires the Atropos orientation")

    def mapping(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "steps": self.steps,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "min_improvement": self.min_improvement,
            "seed": self.seed,
            "device": self.device,
            "precision": self.precision,
            "mode": self.mode.value,
        }


class LoraDatasetBundle:
    """Content-addressed, license-checked train and selection-evaluation rows."""

    def __init__(
        self,
        catalog_path: str | Path,
        train_jsonl: str | Path,
        evaluation_jsonl: str | Path,
        *,
        source_id: str,
        preprocess_version: str = LORA_DATASET_PROTOCOL,
        max_train_records: int = 10_000,
        max_evaluation_records: int = 1_000,
    ) -> None:
        _bounded_int(
            "maximum LoRA training records",
            max_train_records,
            1,
            MAX_LORA_RECORDS,
        )
        _bounded_int(
            "maximum LoRA evaluation records",
            max_evaluation_records,
            1,
            MAX_LORA_RECORDS,
        )
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("LoRA source ID must be non-empty")
        if not isinstance(preprocess_version, str) or not preprocess_version.strip():
            raise ValueError("LoRA preprocess version must be non-empty")
        self.catalog_path = _strict_file(catalog_path, "source catalog")
        self.train_jsonl = _strict_file(train_jsonl, "LoRA training dataset")
        self.evaluation_jsonl = _strict_file(
            evaluation_jsonl, "LoRA selection-evaluation dataset"
        )
        if self.train_jsonl == self.evaluation_jsonl:
            raise ValueError("LoRA training and evaluation files must be distinct")
        self.source_id = source_id
        self.preprocess_version = preprocess_version
        self.max_train_records = max_train_records
        self.max_evaluation_records = max_evaluation_records
        catalog = SourceCatalog.from_toml(self.catalog_path)
        source = catalog.by_id(source_id)
        if source.modality not in {"text", "code"}:
            raise ValueError("LoRA causal data source must be text or code")
        if not source.license_verified:
            raise ValueError("LoRA data source license must be verified")
        self.source = source
        self.train_examples = tuple(
            _freeze_example(item)
            for item in iter_jsonl_examples(
                self.train_jsonl,
                source_id=source_id,
                split="candidate-train",
                preprocess_version=preprocess_version,
                split_seed=0,
                max_records=max_train_records,
            )
        )
        self.evaluation_examples = tuple(
            _freeze_example(item)
            for item in iter_jsonl_examples(
                self.evaluation_jsonl,
                source_id=source_id,
                split="selection-evaluation",
                preprocess_version=preprocess_version,
                split_seed=0,
                max_records=max_evaluation_records,
            )
        )
        if not self.train_examples or not self.evaluation_examples:
            raise ValueError("LoRA train and evaluation selections must be non-empty")
        train_ids = {item.record_id for item in self.train_examples}
        evaluation_ids = {item.record_id for item in self.evaluation_examples}
        if train_ids & evaluation_ids:
            raise ValueError("LoRA train and evaluation record IDs must be disjoint")
        train_content = {_example_content_digest(item) for item in self.train_examples}
        evaluation_content = {
            _example_content_digest(item) for item in self.evaluation_examples
        }
        if train_content & evaluation_content:
            raise ValueError("LoRA train and evaluation content must be disjoint")
    def mapping(self) -> dict[str, object]:
        mapping = self._current_mapping()
        return json.loads(_canonical_bytes(mapping))

    @property
    def manifest_digest(self) -> str:
        return _domain_digest(LORA_DATASET_PROTOCOL, self._current_mapping())

    def _current_mapping(self) -> dict[str, object]:
        catalog_path = _strict_file(self.catalog_path, "source catalog")
        train_jsonl = _strict_file(self.train_jsonl, "LoRA training dataset")
        evaluation_jsonl = _strict_file(
            self.evaluation_jsonl, "LoRA selection-evaluation dataset"
        )
        return {
            "protocol": LORA_DATASET_PROTOCOL,
            "catalog_sha256": file_sha256(catalog_path),
            "source": {
                "source_id": self.source.source_id,
                "revision": self.source.revision,
                "license": self.source.license,
                "license_url": self.source.license_url,
                "license_verified": self.source.license_verified,
                "modality": self.source.modality,
            },
            "preprocess_version": self.preprocess_version,
            "train": _selection_mapping(
                train_jsonl,
                self.train_examples,
                self.max_train_records,
            ),
            "selection_evaluation": _selection_mapping(
                evaluation_jsonl,
                self.evaluation_examples,
                self.max_evaluation_records,
            ),
            "is_promotion_holdout": False,
        }


class LoraCandidateBackend:
    """Train only LoRA tensors and emit four immutable candidate files."""

    def __init__(
        self,
        *,
        base_checkpoint: str | Path,
        model_config: ModelConfig,
        checkpoint_manifest: Mapping[str, object],
        dataset: LoraDatasetBundle,
        training: LoraTrainingConfig,
    ) -> None:
        self.base_checkpoint = _strict_file(base_checkpoint, "LoRA base checkpoint")
        if not isinstance(model_config, ModelConfig):
            raise ValueError("LoRA model config must be a ModelConfig")
        if model_config.vocab_size < 260:
            raise ValueError("LoRA byte-token training requires vocabulary size at least 260")
        if not isinstance(checkpoint_manifest, Mapping):
            raise ValueError("LoRA checkpoint manifest must be a mapping")
        try:
            normalized_manifest = json.loads(_canonical_bytes(dict(checkpoint_manifest)))
        except (TypeError, ValueError) as error:
            raise ValueError("LoRA checkpoint manifest must be JSON-compatible") from error
        if not isinstance(dataset, LoraDatasetBundle):
            raise ValueError("LoRA dataset bundle is invalid")
        if not isinstance(training, LoraTrainingConfig):
            raise ValueError("LoRA training config is invalid")
        self.model_config = model_config
        self.checkpoint_manifest = normalized_manifest
        self.dataset = dataset
        self.training = training
        self._versions = _dependency_versions()
        self._trainer_mapping = {
            "protocol": LORA_BACKEND_PROTOCOL,
            "model_config": asdict(model_config),
            "checkpoint_manifest": normalized_manifest,
            "training": training.mapping(),
            "dependencies": self._versions,
            "optimizer": "torch.optim.AdamW",
            "artifact_format": "safetensors",
            "selection_metric": "baseline_loss-final_loss",
            "is_promotion_holdout": False,
        }

    @property
    def dataset_manifest_digest(self) -> str:
        return self.dataset.manifest_digest

    @property
    def trainer_manifest_digest(self) -> str:
        return _domain_digest(LORA_BACKEND_PROTOCOL, self._trainer_mapping)

    @property
    def active_checkpoint_digest(self) -> str:
        return file_sha256(_strict_file(self.base_checkpoint, "LoRA base checkpoint"))

    def __call__(
        self, request: CandidateWorkRequest, workspace: Path
    ) -> CandidateWorkerOutput:
        if not isinstance(request, CandidateWorkRequest):
            raise ValueError("LoRA candidate request is invalid")
        if request.kind is not CandidateKind.ADAPTER:
            raise ValueError("LoRA backend accepts adapter candidates only")
        if request.active_checkpoint_digest != self.active_checkpoint_digest:
            raise ValueError("LoRA request does not bind the configured base checkpoint")
        if request.dataset_manifest_digest != self.dataset_manifest_digest:
            raise ValueError("LoRA request dataset manifest does not match")
        if request.trainer_manifest_digest != self.trainer_manifest_digest:
            raise ValueError("LoRA request trainer manifest does not match")
        if not isinstance(workspace, Path) or not workspace.is_dir():
            raise ValueError("LoRA staging workspace is invalid")
        device = torch.device(self.training.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("configured CUDA device is unavailable")
        _preflight_budget(request, self.dataset, self.training, self.model_config)
        return self._train(request, workspace, device)

    def _train(
        self, request: CandidateWorkRequest, workspace: Path, device: torch.device
    ) -> CandidateWorkerOutput:
        try:
            from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
            from safetensors.torch import save_file
        except ImportError as error:  # pragma: no cover - constructor checks metadata
            raise RuntimeError("PEFT LoRA training dependencies are unavailable") from error

        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            base = MoiraModel(self.model_config)
            checkpoint_optimizer = AdamW(base.parameters(), lr=1e-3)
            loaded = load_m1_checkpoint(
                self.base_checkpoint,
                model=base,
                optimizer=checkpoint_optimizer,
                loom=None,
                expected_manifest=self.checkpoint_manifest,
            )
            torch.manual_seed(self.training.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(self.training.seed)
            base.to(device)
            model = get_peft_model(
                base,
                LoraConfig(
                    r=self.training.rank,
                    lora_alpha=self.training.alpha,
                    target_modules=list(self.training.target_modules),
                    lora_dropout=self.training.dropout,
                    bias="none",
                ),
            )
            trainable = tuple(
                parameter for parameter in model.parameters() if parameter.requires_grad
            )
            trainable_names = tuple(
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            )
            if not trainable or any("lora_" not in name for name in trainable_names):
                raise RuntimeError("PEFT exposed parameters outside the LoRA surface")
            initial_state = get_peft_model_state_dict(model)
            initial_digest = _tensor_mapping_digest(initial_state)
            baseline = _evaluate(
                model,
                self.dataset.evaluation_examples,
                batch_size=self.training.eval_batch_size,
                sequence_length=self.model_config.max_seq_len,
                mode=self.training.mode,
                device=device,
            )
            optimizer = AdamW(
                trainable,
                lr=self.training.learning_rate,
                weight_decay=self.training.weight_decay,
            )
            model.train()
            train_samples = 0
            train_tokens = 0
            losses: list[float] = []
            cursor = 0
            for _ in range(self.training.steps):
                examples, cursor = _next_batch(
                    self.dataset.train_examples,
                    cursor,
                    self.training.batch_size,
                )
                batch = build_causal_batch(
                    examples, sequence_length=self.model_config.max_seq_len
                )
                batch = _batch_to(batch, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    batch.input_ids,
                    mode=self.training.mode,
                    valid_tokens=batch.valid_tokens,
                )
                loss = _masked_loss(output.logits, batch.targets, batch.loss_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, self.training.max_grad_norm)
                optimizer.step()
                value = float(loss.detach().cpu())
                if not math.isfinite(value):
                    raise ValueError("LoRA training produced a non-finite loss")
                losses.append(value)
                train_samples += len(examples)
                train_tokens += int(batch.valid_tokens.sum().item())
            final_state = get_peft_model_state_dict(model)
            final_digest = _tensor_mapping_digest(final_state)
            evaluation_state_before = final_digest
            final_evaluation = _evaluate(
                model,
                self.dataset.evaluation_examples,
                batch_size=self.training.eval_batch_size,
                sequence_length=self.model_config.max_seq_len,
                mode=self.training.mode,
                device=device,
            )
            evaluation_state_after = _tensor_mapping_digest(
                get_peft_model_state_dict(model)
            )
            if evaluation_state_before != evaluation_state_after:
                raise RuntimeError("LoRA selection evaluation mutated adapter weights")
            adapter_state = {
                name: value.detach().to(device="cpu").contiguous()
                for name, value in final_state.items()
            }
            artifact_file = "adapter_model.safetensors"
            creation_file = "creation_manifest.json"
            evaluation_file = "selection_evaluation.json"
            metric_file = "metrics.json"
            save_file(adapter_state, workspace / artifact_file)
            improvement = baseline["loss"] - final_evaluation["loss"]
            mutation_verified = initial_digest != final_digest
            viable = (
                math.isfinite(improvement)
                and mutation_verified
                and improvement >= self.training.min_improvement
            )
            remaining = () if viable else request.target_failure_ids
            creation = {
                "protocol": LORA_BACKEND_PROTOCOL,
                "request_digest": request.request_digest,
                "candidate_id": request.candidate_id,
                "lineage_id": request.lineage_id,
                "parent_ids": list(request.parent_ids),
                "parent_record_digests": list(request.parent_record_digests),
                "active_checkpoint_digest": request.active_checkpoint_digest,
                "base_checkpoint_lineage_id": loaded.lineage_id,
                "base_checkpoint_step": loaded.global_step,
                "dataset_manifest_digest": self.dataset_manifest_digest,
                "trainer_manifest_digest": self.trainer_manifest_digest,
                "adapter_state_before": initial_digest,
                "adapter_state_after": final_digest,
                "adapter_tensors": [
                    {
                        "name": name,
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                    for name, value in sorted(adapter_state.items())
                ],
                "trainable_parameter_names": list(trainable_names),
                "can_authorize": False,
                "can_promote": False,
            }
            evaluation = {
                "protocol": LORA_BACKEND_PROTOCOL,
                "split": "selection-evaluation",
                "record_ids": list(final_evaluation["record_ids"]),
                "record_digests": list(final_evaluation["record_digests"]),
                "baseline_loss": baseline["loss"],
                "final_loss": final_evaluation["loss"],
                "improvement": improvement,
                "state_before_evaluation": evaluation_state_before,
                "state_after_evaluation": evaluation_state_after,
                "mutation_verified": mutation_verified,
                "viable": viable,
                "is_selection_only": True,
                "is_promotion_holdout": False,
                "can_authorize": False,
                "can_promote": False,
            }
            metrics = {
                "protocol": LORA_BACKEND_PROTOCOL,
                "training_losses": losses,
                "selection_score": improvement,
                "steps": self.training.steps,
                "training_samples": train_samples,
                "training_tokens": train_tokens,
                "evaluation_samples_per_pass": final_evaluation["samples"],
                "evaluation_tokens_per_pass": final_evaluation["tokens"],
                "adapter_parameters": sum(parameter.numel() for parameter in trainable),
            }
            _write_json(workspace / creation_file, creation)
            _write_json(workspace / evaluation_file, evaluation)
            _write_json(workspace / metric_file, metrics)
            return CandidateWorkerOutput(
                artifact_file=artifact_file,
                creation_manifest_file=creation_file,
                evaluation_manifest_file=evaluation_file,
                metric_file=metric_file,
                selection_score=float(improvement),
                viable=viable,
                mutation_verified=mutation_verified,
                remaining_failure_ids=remaining,
                usage=CandidateWorkUsage(
                    steps=self.training.steps,
                    samples=train_samples + 2 * final_evaluation["samples"],
                    tokens=train_tokens + 2 * final_evaluation["tokens"],
                ),
            )
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)


def lora_backend_contract() -> dict[str, object]:
    return {
        "protocol": LORA_BACKEND_PROTOCOL,
        "candidate_kind": CandidateKind.ADAPTER.value,
        "training_precision": "fp32",
        "orientation": CognitiveMode.ATROPOS.value,
        "artifact_format": "safetensors",
        "trains_lora_parameters_only": True,
        "requires_disjoint_selection_data": True,
        "is_promotion_holdout": False,
        "can_authorize": False,
        "can_promote": False,
    }


def _preflight_budget(
    request: CandidateWorkRequest,
    dataset: LoraDatasetBundle,
    training: LoraTrainingConfig,
    model_config: ModelConfig,
) -> None:
    train_samples = training.steps * min(
        training.batch_size, len(dataset.train_examples)
    )
    evaluation_samples = 2 * len(dataset.evaluation_examples)
    upper_tokens = (train_samples + evaluation_samples) * (model_config.max_seq_len - 1)
    if training.steps > request.budget.max_steps:
        raise ValueError("LoRA plan exceeds the candidate step budget")
    if train_samples + evaluation_samples > request.budget.max_samples:
        raise ValueError("LoRA plan exceeds the candidate sample budget")
    if upper_tokens > request.budget.max_tokens:
        raise ValueError("LoRA plan exceeds the candidate token budget")


def _evaluate(
    model: torch.nn.Module,
    examples: tuple[DatasetExample, ...],
    *,
    batch_size: int,
    sequence_length: int,
    mode: CognitiveMode,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_targets = 0
    samples = 0
    tokens = 0
    record_ids: list[str] = []
    record_digests: list[str] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            selected = examples[start : start + batch_size]
            batch = _batch_to(
                build_causal_batch(selected, sequence_length=sequence_length), device
            )
            output = model(
                batch.input_ids,
                mode=mode,
                valid_tokens=batch.valid_tokens,
            )
            loss_sum, targets = _masked_loss_sum(
                output.logits, batch.targets, batch.loss_mask
            )
            total_loss += float(loss_sum.detach().cpu())
            total_targets += targets
            samples += len(selected)
            tokens += int(batch.valid_tokens.sum().item())
            record_ids.extend(item.record_id for item in selected)
            record_digests.extend(item.record_sha256 for item in selected)
    if total_targets <= 0:
        raise ValueError("LoRA evaluation has no supervised target tokens")
    loss = total_loss / total_targets
    if not math.isfinite(loss):
        raise ValueError("LoRA evaluation produced a non-finite loss")
    return {
        "loss": loss,
        "samples": samples,
        "tokens": tokens,
        "record_ids": tuple(record_ids),
        "record_digests": tuple(record_digests),
    }


def _masked_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    loss_sum, count = _masked_loss_sum(logits, targets, mask)
    return loss_sum / count


def _masked_loss_sum(
    logits: Tensor, targets: Tensor, mask: Tensor
) -> tuple[Tensor, int]:
    if mask.dtype is not torch.bool or mask.shape != targets.shape:
        raise ValueError("LoRA loss mask is invalid")
    count = int(mask.sum().item())
    if count <= 0:
        raise ValueError("LoRA batch has no supervised target tokens")
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    return losses.masked_select(mask).sum(), count


def _batch_to(batch: CausalBatch, device: torch.device) -> CausalBatch:
    return CausalBatch(
        input_ids=batch.input_ids.to(device),
        targets=batch.targets.to(device),
        valid_tokens=batch.valid_tokens.to(device),
        loss_mask=batch.loss_mask.to(device),
        record_ids=batch.record_ids,
    )


def _next_batch(
    examples: tuple[DatasetExample, ...], cursor: int, batch_size: int
) -> tuple[tuple[DatasetExample, ...], int]:
    selected = tuple(
        examples[(cursor + offset) % len(examples)]
        for offset in range(min(batch_size, len(examples)))
    )
    return selected, (cursor + len(selected)) % len(examples)


def _selection_mapping(
    path: Path, examples: Sequence[DatasetExample], maximum: int
) -> dict[str, object]:
    return {
        "file_sha256": file_sha256(path),
        "max_records": maximum,
        "selected_records": [
            {
                "record_id": item.record_id,
                "record_sha256": item.record_sha256,
                "content_sha256": _example_content_digest(item),
            }
            for item in examples
        ],
    }


def _example_content_digest(example: DatasetExample) -> str:
    return canonical_sha256(
        {
            "source_id": example.source_id,
            "modality": example.modality,
            "inputs": dict(example.inputs),
            "targets": dict(example.targets),
            "preprocess_version": example.preprocess_version,
        }
    )


def _freeze_example(example: DatasetExample) -> DatasetExample:
    return replace(
        example,
        inputs=MappingProxyType(dict(example.inputs)),
        targets=MappingProxyType(dict(example.targets)),
        metadata=MappingProxyType(dict(example.metadata)),
    )


def _tensor_mapping_digest(values: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256(b"moira-lora-tensors-v1\x00")
    for name, value in sorted(values.items()):
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\x00")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\x00")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    versions = {"torch": torch.__version__}
    for package in ("peft", "safetensors"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "PEFT LoRA training requires the research training extras"
            ) from error
    return versions


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


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _domain_digest(domain: str, value: object) -> str:
    return hashlib.sha256(domain.encode("ascii") + b"\x00" + _canonical_bytes(value)).hexdigest()


def _bounded_int(name: str, value: object, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its bounded range")


__all__ = [
    "LORA_BACKEND_PROTOCOL",
    "LORA_DATASET_PROTOCOL",
    "LoraCandidateBackend",
    "LoraDatasetBundle",
    "LoraTrainingConfig",
    "lora_backend_contract",
]
