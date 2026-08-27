"""Executable INT2/INT4/INT8 weight QAT for proof-scale MOIRA models.

The optimizer owns floating-point master parameters. Linear and embedding
forwards see a signed symmetric fake-quantized view, so gradients can train the
master through a straight-through estimator. Increasing the logical bit width
therefore reveals a finer view of retained state; it never reconstructs detail
from a low-bit-only artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.optim import AdamW, Optimizer

from .config import TrainingConfig
from .elastic import format_bit_precision, next_bit_precision, parse_bit_precision
from .masks import CognitiveMode
from .model import MoiraModel
from .synthetic import SYNTHETIC_CURSOR_KIND, DataCursor, SyntheticStream


QAT_CHECKPOINT_SCHEMA_VERSION = 3
_MAX_QAT_SMOKE_STEPS_PER_STAGE = 32


def qat_config_digest(config: TrainingConfig) -> str:
    """Bind a continuation checkpoint to one exact validated training config."""
    if not isinstance(config, TrainingConfig):
        raise ValueError("QAT config digest requires a TrainingConfig")
    encoded = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"moira.qat.training-config.v1\x00")
    digest.update(encoded)
    return digest.hexdigest()


def _validate_config_digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("QAT config digest must be a lowercase SHA-256 digest")
    return value


def _cursor_mapping(cursor: DataCursor) -> dict[str, object]:
    if not isinstance(cursor, DataCursor):
        raise ValueError("QAT checkpoint data cursor is invalid")
    return {
        "kind": cursor.kind,
        "next_sample_index": cursor.next_sample_index,
        "samples_seen": cursor.samples_seen,
        "tokens_seen": cursor.tokens_seen,
    }


def _cursor_from_mapping(value: object) -> DataCursor:
    if not isinstance(value, Mapping) or set(value) != {
        "kind",
        "next_sample_index",
        "samples_seen",
        "tokens_seen",
    }:
        raise ValueError("QAT checkpoint data cursor fields are invalid")
    try:
        return DataCursor(
            kind=value["kind"],
            next_sample_index=value["next_sample_index"],
            samples_seen=value["samples_seen"],
            tokens_seen=value["tokens_seen"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError("QAT checkpoint data cursor is invalid") from error


def _validate_cursor_for_config(
    cursor: DataCursor,
    *,
    global_step: int,
    config: TrainingConfig,
) -> None:
    expected_samples = global_step * config.data.batch_size
    expected_tokens = expected_samples * config.model.max_seq_len
    if (
        cursor.next_sample_index != expected_samples
        or cursor.samples_seen != expected_samples
        or cursor.tokens_seen != expected_tokens
    ):
        raise ValueError("QAT checkpoint cursor does not match its step and config")


def _validate_qat_bits(value: int | str) -> int:
    return parse_bit_precision(value)


def fake_quantize_weight_ste(
    weight: Tensor,
    bits: int | str,
    *,
    per_row: bool = False,
) -> Tensor:
    """Return a signed fake-quantized weight with identity backward gradient."""
    level = _validate_qat_bits(bits)
    if (
        not isinstance(weight, Tensor)
        or not weight.is_floating_point()
        or weight.numel() <= 0
    ):
        raise ValueError("QAT weight must be a non-empty floating tensor")
    working_dtype = torch.float32
    working = weight.to(dtype=working_dtype)
    qmax = (1 << (level - 1)) - 1
    if not isinstance(per_row, bool):
        raise ValueError("QAT per_row must be boolean")
    maximum = (
        working.detach().abs().amax(dim=-1, keepdim=True)
        if per_row and working.ndim >= 2
        else working.detach().abs().amax()
    )
    scale = torch.clamp(
        maximum / float(qmax),
        min=torch.finfo(working_dtype).tiny,
    )
    quantized = torch.clamp(torch.round(working / scale), -qmax, qmax)
    dequantized = (quantized * scale).to(dtype=weight.dtype)
    return weight + (dequantized - weight).detach()


class IntXWeightParametrization(nn.Module):
    """Parametrization retaining an FP32 master behind an INT2/4/8 view."""

    def __init__(self, bits: int | str) -> None:
        super().__init__()
        self.register_buffer(
            "logical_bits",
            torch.tensor(_validate_qat_bits(bits), dtype=torch.int16),
        )

    @property
    def bits(self) -> int:
        return int(self.logical_bits.item())

    def set_bits(self, bits: int | str) -> None:
        self.logical_bits.fill_(_validate_qat_bits(bits))

    def forward(self, weight: Tensor) -> Tensor:
        return fake_quantize_weight_ste(weight, self.bits, per_row=True)


@dataclass(frozen=True, slots=True)
class QATPreparation:
    logical_bits: int
    quantized_modules: int
    unique_master_tensors: int
    master_parameters: int
    master_dtypes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "logical_bits": format_bit_precision(self.logical_bits),
            "quantized_modules": self.quantized_modules,
            "unique_master_tensors": self.unique_master_tensors,
            "master_parameters": self.master_parameters,
            "master_dtypes": list(self.master_dtypes),
            "master_weights_retained": True,
            "forward_is_fake_quantized": True,
            "scale_granularity": "per-weight-row",
            "packed_training_kernel": False,
        }


def _qat_parametrizations(model: nn.Module) -> tuple[IntXWeightParametrization, ...]:
    found: list[IntXWeightParametrization] = []
    for module in model.modules():
        if not parametrize.is_parametrized(module, "weight"):
            continue
        parametrizations = module.parametrizations.weight
        if len(parametrizations) != 1 or not isinstance(
            parametrizations[0], IntXWeightParametrization
        ):
            raise ValueError("model contains an incompatible weight parametrization")
        found.append(parametrizations[0])
    if not found:
        raise ValueError("model has not been prepared for MOIRA QAT")
    return tuple(found)


def prepare_model_for_qat(
    model: nn.Module,
    bits: int | str = 2,
) -> QATPreparation:
    """Attach a standard integer view to every Linear and Embedding weight."""
    if not isinstance(model, nn.Module):
        raise ValueError("QAT model must be a torch module")
    level = _validate_qat_bits(bits)
    targets = tuple(
        module
        for module in model.modules()
        if isinstance(module, (nn.Linear, nn.Embedding))
    )
    if not targets:
        raise ValueError("QAT model has no Linear or Embedding weights")
    for module in targets:
        if parametrize.is_parametrized(module, "weight"):
            raise ValueError("QAT model is already parametrized")
        if module.weight.dtype is not torch.float32:
            raise ValueError("QAT master weights must use FP32")
    for module in targets:
        parametrize.register_parametrization(
            module,
            "weight",
            IntXWeightParametrization(level),
        )

    masters: dict[int, Tensor] = {}
    for module in targets:
        master = module.parametrizations.weight.original
        masters.setdefault(id(master), master)
    return QATPreparation(
        logical_bits=level,
        quantized_modules=len(targets),
        unique_master_tensors=len(masters),
        master_parameters=sum(tensor.numel() for tensor in masters.values()),
        master_dtypes=tuple(sorted({str(tensor.dtype) for tensor in masters.values()})),
    )


def qat_model_state(model: nn.Module) -> dict[str, object]:
    """Describe a prepared model and prove that floating masters remain present."""
    parametrizations = _qat_parametrizations(model)
    levels = {item.bits for item in parametrizations}
    if len(levels) != 1:
        raise ValueError("QAT model has mixed runtime precision levels")
    master_tensors: dict[int, Tensor] = {}
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            master = module.parametrizations.weight.original
            master_tensors.setdefault(id(master), master)
    return {
        "logical_bits": format_bit_precision(levels.pop()),
        "quantized_modules": len(parametrizations),
        "unique_master_tensors": len(master_tensors),
        "master_parameters": sum(value.numel() for value in master_tensors.values()),
        "master_dtypes": sorted({str(value.dtype) for value in master_tensors.values()}),
        "master_weights_retained": True,
        "scale_granularity": "per-weight-row",
        "promotion_recovers_discarded_low_bit_detail": False,
    }


def promote_model_qat(model: nn.Module, target_bits: int | str) -> dict[str, object]:
    """Promote every prepared weight view while retaining the same masters."""
    target = _validate_qat_bits(target_bits)
    parametrizations = _qat_parametrizations(model)
    current_levels = {item.bits for item in parametrizations}
    if len(current_levels) != 1:
        raise ValueError("QAT model has mixed runtime precision levels")
    current = current_levels.pop()
    if target <= current:
        raise ValueError("QAT promotion target must exceed the current precision")
    if target != next_bit_precision(current):
        raise ValueError("QAT promotion must advance by exactly one INT2/INT4/INT8 level")
    for item in parametrizations:
        item.set_bits(target)
    return {
        "source_bits": format_bit_precision(current),
        "target_bits": format_bit_precision(target),
        "quantized_modules": len(parametrizations),
        "master_weights_retained": True,
        "requires_continued_training_and_evaluation": True,
        "reconstructed_from_low_bit_artifact": False,
    }


def save_qat_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    global_step: int,
    precision_history: Sequence[int | str],
    config_digest: str,
    data_cursor: DataCursor,
) -> Path:
    """Atomically save an experimental trusted-local QAT continuation checkpoint."""
    if type(global_step) is not int or global_step <= 0:
        raise ValueError("QAT checkpoint step must be a positive integer")
    if not isinstance(optimizer, Optimizer):
        raise ValueError("QAT checkpoint optimizer is invalid")
    validated_config_digest = _validate_config_digest(config_digest)
    cursor_mapping = _cursor_mapping(data_cursor)
    history = _validate_precision_history(precision_history)
    state = qat_model_state(model)
    if state["logical_bits"] != format_bit_precision(history[-1]):
        raise ValueError("QAT checkpoint history does not end at the model precision")

    checkpoint_path = Path(path)
    if not checkpoint_path.parent.is_dir() or checkpoint_path.parent.is_symlink():
        raise ValueError("QAT checkpoint parent must be a non-symlink directory")
    if os.path.lexists(checkpoint_path):
        status = checkpoint_path.lstat()
        if checkpoint_path.is_symlink() or not stat.S_ISREG(status.st_mode):
            raise ValueError("QAT checkpoint must be a regular non-symlink file")
    payload = {
        "schema_version": QAT_CHECKPOINT_SCHEMA_VERSION,
        "global_step": global_step,
        "precision_history": list(history),
        "config_digest": validated_config_digest,
        "data_cursor": cursor_mapping,
        "qat_state": state,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "trusted_local_experimental_checkpoint": True,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            dir=checkpoint_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, checkpoint_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    verify_qat_checkpoint(checkpoint_path)
    return checkpoint_path.resolve()


def verify_qat_checkpoint(path: str | Path) -> dict[str, object]:
    """Verify safe payload shape and retained-master evidence without executing code."""
    checkpoint_path = Path(path)
    try:
        status = checkpoint_path.lstat()
    except OSError as error:
        raise ValueError("QAT checkpoint cannot be read") from error
    if checkpoint_path.is_symlink() or not stat.S_ISREG(status.st_mode):
        raise ValueError("QAT checkpoint must be a regular non-symlink file")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, ValueError) as error:
        raise ValueError("QAT checkpoint cannot be read safely") from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "global_step",
        "precision_history",
        "config_digest",
        "data_cursor",
        "qat_state",
        "model_state",
        "optimizer_state",
        "trusted_local_experimental_checkpoint",
    }:
        raise ValueError("QAT checkpoint fields do not match the schema")
    if payload["schema_version"] != QAT_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("QAT checkpoint schema version is unsupported")
    if type(payload["global_step"]) is not int or payload["global_step"] <= 0:
        raise ValueError("QAT checkpoint step is invalid")
    config_digest = _validate_config_digest(payload["config_digest"])
    data_cursor = _cursor_from_mapping(payload["data_cursor"])
    history = payload["precision_history"]
    if not isinstance(history, list) or not history:
        raise ValueError("QAT checkpoint precision history is invalid")
    levels = _validate_precision_history(history)
    qat_state = payload["qat_state"]
    model_state = payload["model_state"]
    if not isinstance(qat_state, dict) or not isinstance(model_state, Mapping):
        raise ValueError("QAT checkpoint states are invalid")
    master_keys = tuple(
        key
        for key in model_state
        if isinstance(key, str) and key.endswith("parametrizations.weight.original")
    )
    if not master_keys or any(
        not isinstance(model_state[key], Tensor)
        or model_state[key].dtype is not torch.float32
        for key in master_keys
    ):
        raise ValueError("QAT checkpoint does not retain FP32 master weights")
    if qat_state.get("logical_bits") != format_bit_precision(levels[-1]):
        raise ValueError("QAT checkpoint precision state and history disagree")
    if qat_state.get("master_weights_retained") is not True:
        raise ValueError("QAT checkpoint master-weight declaration is missing")
    if payload["trusted_local_experimental_checkpoint"] is not True:
        raise ValueError("QAT checkpoint trust-scope declaration is missing")
    if not isinstance(payload["optimizer_state"], Mapping):
        raise ValueError("QAT checkpoint optimizer state is invalid")
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "global_step": payload["global_step"],
        "precision_history": [format_bit_precision(value) for value in levels],
        "config_digest": config_digest,
        "data_cursor": _cursor_mapping(data_cursor),
        "logical_bits": qat_state["logical_bits"],
        "master_tensor_entries": len(master_keys),
        "master_weights_retained": True,
        "weights_only_loader": True,
    }


def load_qat_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    expected_config_digest: str,
) -> dict[str, object]:
    """Restore a verified trusted-local continuation checkpoint."""
    verified = verify_qat_checkpoint(path)
    expected_digest = _validate_config_digest(expected_config_digest)
    if verified["config_digest"] != expected_digest:
        raise ValueError("QAT checkpoint config digest does not match")
    if not isinstance(model, nn.Module):
        raise ValueError("QAT checkpoint model must be a torch module")
    if optimizer is not None and not isinstance(optimizer, Optimizer):
        raise ValueError("QAT checkpoint optimizer is invalid")
    checkpoint_path = Path(path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, ValueError) as error:
        raise ValueError("QAT checkpoint cannot be read safely") from error
    saved_state = payload["model_state"]
    live_state = model.state_dict()
    if set(saved_state) != set(live_state):
        raise ValueError("QAT checkpoint model state keys do not match")
    for name, live_tensor in live_state.items():
        saved_tensor = saved_state[name]
        if (
            not isinstance(saved_tensor, Tensor)
            or saved_tensor.shape != live_tensor.shape
            or saved_tensor.dtype != live_tensor.dtype
        ):
            raise ValueError("QAT checkpoint model tensor metadata does not match")
    model.load_state_dict(saved_state, strict=True)
    if optimizer is not None:
        try:
            optimizer.load_state_dict(payload["optimizer_state"])
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError("QAT checkpoint optimizer state does not match") from error
    restored_state = qat_model_state(model)
    if restored_state["logical_bits"] != verified["logical_bits"]:
        raise ValueError("QAT checkpoint restored the wrong precision state")
    return {
        **verified,
        "model_restored": True,
        "optimizer_restored": optimizer is not None,
    }


def _validate_precision_history(
    values: Sequence[int | str],
) -> tuple[int, ...]:
    history = tuple(_validate_qat_bits(value) for value in values)
    if not history:
        raise ValueError("QAT checkpoint precision history must not be empty")
    for source, target in zip(history, history[1:]):
        if target != next_bit_precision(source):
            raise ValueError(
                "QAT checkpoint precision history must contain consecutive lattice levels"
            )
    return history


def run_moira_qat_smoke(
    config: TrainingConfig,
    *,
    initial_bits: int | str = 2,
    promote_to: int | str = 4,
    steps_per_stage: int = 1,
    device: str = "cpu",
    checkpoint: str | Path | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Train a bounded MOIRA model at INT8 or across one precision promotion."""
    if not isinstance(config, TrainingConfig):
        raise ValueError("QAT smoke config must be a TrainingConfig")
    if (
        type(steps_per_stage) is not int
        or not 0 < steps_per_stage <= _MAX_QAT_SMOKE_STEPS_PER_STAGE
    ):
        raise ValueError("QAT steps per stage must be between 1 and 32")
    if device not in {"cpu", "cuda"}:
        raise ValueError("QAT device must be cpu or cuda")
    if type(resume) is not bool:
        raise ValueError("QAT resume flag must be boolean")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available for the QAT smoke run")
    initial = _validate_qat_bits(initial_bits)
    target = _validate_qat_bits(promote_to)
    single_stage = initial == target
    if single_stage and initial != 8:
        raise ValueError("equal QAT precisions are supported only for single-stage int8")
    if not single_stage and target != next_bit_precision(initial):
        raise ValueError("QAT promotion must advance exactly one precision level")
    if resume and (not single_stage or initial != 8):
        raise ValueError("QAT resume currently supports only single-stage int8")
    if resume and config.model.dropout != 0.0:
        raise ValueError("QAT resume requires dropout 0 until RNG state is checkpointed")
    if resume and checkpoint is None:
        raise ValueError("QAT resume requires an existing checkpoint")
    if resume and checkpoint is not None and not os.path.lexists(checkpoint):
        raise ValueError("QAT resume checkpoint does not exist")
    if not resume and checkpoint is not None and os.path.lexists(checkpoint):
        raise ValueError("QAT checkpoint already exists; use resume to continue it")
    precision_history = (initial,) if single_stage else (initial, target)
    config_digest = qat_config_digest(config)

    torch.manual_seed(config.reproducibility.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.reproducibility.seed)
        torch.cuda.reset_peak_memory_stats()
    model = MoiraModel(config.model).to(device=device, dtype=torch.float32)
    preparation = prepare_model_for_qat(model, initial)
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
        foreach=False,
        fused=False,
    )
    stream = SyntheticStream(
        config.data.seed,
        config.model.vocab_size,
        config.model.max_seq_len,
    )
    cursor = DataCursor(SYNTHETIC_CURSOR_KIND, 0, 0, 0)
    starting_global_step = 0
    if resume:
        restored = load_qat_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_config_digest=config_digest,
        )
        if restored["precision_history"] != ["int8"]:
            raise ValueError("QAT resume checkpoint is not a single-stage int8 lineage")
        starting_global_step = int(restored["global_step"])
        cursor = _cursor_from_mapping(restored["data_cursor"])
        _validate_cursor_for_config(
            cursor,
            global_step=starting_global_step,
            config=config,
        )
    losses: list[float] = []
    bf16_available = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast_bf16 = False

    def train_stage() -> None:
        nonlocal cursor
        model.train()
        for _ in range(steps_per_stage):
            optimizer.zero_grad(set_to_none=True)
            batch, cursor = stream.batch(
                split=config.data.train_split,
                cursor=cursor,
                batch_size=config.data.batch_size,
            )
            input_ids = batch.input_ids.to(device=device)
            targets = batch.targets.to(device=device)
            loss_mask = batch.loss_mask.to(device=device)
            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=autocast_bf16,
            ):
                output = model(input_ids, mode=CognitiveMode.ATROPOS)
                per_token = F.cross_entropy(
                    output.logits.reshape(-1, output.logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                ).view_as(targets)
                weights = loss_mask.to(dtype=per_token.dtype)
                loss = (per_token * weights).sum() / weights.sum()
            if not bool(torch.isfinite(loss)):
                raise ValueError("QAT smoke loss became non-finite")
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                config.optimization.clip_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            losses.append(float(loss.detach().item()))

    train_stage()
    promotion = None
    if not single_stage:
        promotion = promote_model_qat(model, target)
        train_stage()
    if device == "cuda":
        torch.cuda.synchronize()
    global_step = starting_global_step + steps_per_stage * len(precision_history)
    _validate_cursor_for_config(cursor, global_step=global_step, config=config)
    checkpoint_path = None
    if checkpoint is not None:
        checkpoint_path = save_qat_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            precision_history=precision_history,
            config_digest=config_digest,
            data_cursor=cursor,
        )
    return {
        "architecture": "MOIRA-master-retained-standard-INT-QAT-v2",
        "model": "MOIRA",
        "device": device,
        "compute_dtype": "fp32",
        "bf16_compute_available": bf16_available,
        "bf16_compute_enabled": False,
        "bf16_disabled_reason": "Pulse state currently requires FP32 dtype equality",
        "preparation": preparation.to_dict(),
        "promotion": promotion,
        "final_qat_state": qat_model_state(model),
        "precision_history": [
            format_bit_precision(level) for level in precision_history
        ],
        "steps_per_stage": steps_per_stage,
        "resumed": resume,
        "starting_global_step": starting_global_step,
        "global_step": global_step,
        "config_digest": config_digest,
        "data_cursor": _cursor_mapping(cursor),
        "resume_scope": "weights-optimizer-and-synthetic-cursor",
        "exact_resume_guaranteed": False,
        "losses": losses,
        "all_losses_finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "trainable_parameters": model.parameter_ledger().total_trainable,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated() if device == "cuda" else 0
        ),
        "peak_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved() if device == "cuda" else 0
        ),
        "scope": "proof-scale-weight-only-QAT-no-parameter-growth",
    }


__all__ = [
    "IntXWeightParametrization",
    "QAT_CHECKPOINT_SCHEMA_VERSION",
    "QATPreparation",
    "fake_quantize_weight_ste",
    "load_qat_checkpoint",
    "prepare_model_for_qat",
    "promote_model_qat",
    "qat_config_digest",
    "qat_model_state",
    "run_moira_qat_smoke",
    "save_qat_checkpoint",
    "verify_qat_checkpoint",
]
