"""Deterministic CPU trainer for the MOIRA M0 research checkpoint."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import random
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from .checkpoint import (
    _phase0_schedule_lambda,
    checkpoint_run_lock,
    load_checkpoint,
    save_checkpoint,
)
from .config import TrainingConfig
from .masks import CognitiveMode
from .model import MoiraModel
from .synthetic import SYNTHETIC_CURSOR_KIND, DataCursor, SyntheticStream


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: MoiraModel
    optimizer: Optimizer
    scheduler: LRScheduler
    scaler: object | None
    lineage_id: str
    global_step: int
    data_cursor: DataCursor
    losses: tuple[float, ...]
    checkpoint_path: Path
    counters: dict[str, int]

    @property
    def samples_seen(self) -> int:
        return self.data_cursor.samples_seen

    @property
    def tokens_seen(self) -> int:
        return self.data_cursor.tokens_seen


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    loss: float
    batches: int
    samples: int
    tokens: int


def train_until(
    config: TrainingConfig,
    run_dir: str | os.PathLike[str],
    *,
    stop_at_step: int,
    resume: str | os.PathLike[str] | None = None,
    device: str = "cpu",
) -> TrainingResult:
    _validate_phase_zero(config, stop_at_step=stop_at_step, device=device)
    with checkpoint_run_lock(run_dir):
        return _train_until_locked(
            config,
            run_dir,
            stop_at_step=stop_at_step,
            resume=resume,
            device=device,
        )


def _train_until_locked(
    config: TrainingConfig,
    run_dir: str | os.PathLike[str],
    *,
    stop_at_step: int,
    resume: str | os.PathLike[str] | None,
    device: str,
) -> TrainingResult:
    """Train to one operational stop without changing the semantic schedule."""
    _seed_strict(config.reproducibility.seed)
    model, optimizer, scheduler = _build_components(config)
    scaler = None
    manifests = _semantic_manifests(config)
    provenance = {"git": _git_provenance()}
    ledger = model.parameter_ledger()
    stream = SyntheticStream(
        config.data.seed,
        config.model.vocab_size,
        config.model.max_seq_len,
    )
    cursor = DataCursor(SYNTHETIC_CURSOR_KIND, 0, 0, 0)
    global_step = 0
    lineage_id = uuid.uuid4().hex
    counters = {"optimizer_steps": 0, "batches_seen": 0}
    checkpoint_path: Path | None = None

    if resume is not None:
        auto = resume == "auto"
        source = Path(run_dir) if auto else Path(resume)
        loaded = load_checkpoint(
            source,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_manifests=manifests,
            auto=auto,
        )
        global_step = loaded.global_step
        cursor = loaded.data_cursor
        lineage_id = loaded.lineage_id
        counters = loaded.counters
        checkpoint_path = loaded.checkpoint_path
    if stop_at_step < global_step:
        raise ValueError("stop_at_step is behind the loaded global step")

    model.train()
    losses: list[float] = []
    while global_step < stop_at_step:
        optimizer.zero_grad(set_to_none=True)
        batch, next_cursor = stream.batch(
            split=config.data.train_split,
            cursor=cursor,
            batch_size=config.data.batch_size,
        )
        output = model(batch.input_ids, mode=CognitiveMode.ATROPOS)
        loss = _masked_next_token_loss(
            output.logits, batch.targets, batch.loss_mask
        )
        if not bool(torch.isfinite(loss)):
            raise ValueError("training loss became non-finite")
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.parameters(),
            config.optimization.clip_grad_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        scheduler.step()
        global_step += 1
        cursor = next_cursor
        counters = {
            "optimizer_steps": global_step,
            "batches_seen": global_step,
        }
        loss_value = float(loss.detach().item())
        losses.append(loss_value)

        if (
            global_step % config.run.checkpoint_every_steps == 0
            or global_step == stop_at_step
        ):
            record = save_checkpoint(
                run_dir,
                lineage_id=lineage_id,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                data_cursor=cursor,
                semantic_manifests=manifests,
                provenance=provenance,
                parameter_ledger=ledger,
                counters=counters,
                last_metrics={
                    "loss": loss_value,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                },
                keep_last_checkpoints=config.run.keep_last_checkpoints,
            )
            checkpoint_path = record.checkpoint_path

    if checkpoint_path is None:
        raise ValueError("training did not produce or load a checkpoint")
    return TrainingResult(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        lineage_id=lineage_id,
        global_step=global_step,
        data_cursor=cursor,
        losses=tuple(losses),
        checkpoint_path=checkpoint_path,
        counters=counters,
    )


def evaluate_smoke(
    config: TrainingConfig,
    model: MoiraModel,
    *,
    batches: int = 2,
) -> EvaluationResult:
    """Evaluate fixed indexed samples without touching training cursor or RNG."""
    if not isinstance(config, TrainingConfig):
        raise ValueError("config must be a TrainingConfig")
    if not isinstance(model, MoiraModel):
        raise ValueError("model must be a MoiraModel")
    _require_positive_integer("batches", batches)
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("Phase 0 evaluation requires a CPU model")

    stream = SyntheticStream(
        config.data.seed,
        config.model.vocab_size,
        config.model.max_seq_len,
    )
    cursor = DataCursor(SYNTHETIC_CURSOR_KIND, 0, 0, 0)
    weighted_loss = 0.0
    token_count = 0
    module_modes = tuple((module, module.training) for module in model.modules())
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(batches):
                batch, cursor = stream.batch(
                    split=config.data.eval_split,
                    cursor=cursor,
                    batch_size=config.data.batch_size,
                )
                output = model(batch.input_ids, mode=CognitiveMode.ATROPOS)
                loss = _masked_next_token_loss(
                    output.logits, batch.targets, batch.loss_mask
                )
                loss_value = float(loss.item())
                if not math.isfinite(loss_value):
                    raise ValueError("evaluation loss became non-finite")
                current_tokens = int(batch.loss_mask.sum().item())
                weighted_loss += loss_value * current_tokens
                token_count += current_tokens
    finally:
        for module, training in module_modes:
            module.training = training
    return EvaluationResult(
        loss=weighted_loss / token_count,
        batches=batches,
        samples=cursor.samples_seen,
        tokens=cursor.tokens_seen,
    )


def _build_components(
    config: TrainingConfig,
) -> tuple[MoiraModel, Optimizer, LRScheduler]:
    model = MoiraModel(config.model).to(device="cpu", dtype=torch.float32)
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
        foreach=False,
        fused=False,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_phase0_schedule_lambda(
            config.optimization.warmup_steps,
            config.optimization.schedule_total_steps,
        ),
    )
    return model, optimizer, scheduler


def _masked_next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    if (
        logits.ndim != 3
        or targets.dtype is not torch.int64
        or loss_mask.dtype is not torch.bool
        or logits.shape[:2] != targets.shape
        or targets.shape != loss_mask.shape
    ):
        raise ValueError("masked next-token tensors have incompatible shapes")
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    weights = loss_mask.to(dtype=per_token.dtype)
    denominator = weights.sum()
    if not bool(denominator > 0):
        raise ValueError("loss mask must select at least one token")
    return (per_token * weights).sum() / denominator


def _semantic_manifests(config: TrainingConfig) -> dict[str, dict[str, Any]]:
    data = asdict(config.data)
    data.update(
        {
            "stream_version": 1,
            "sequence_length": config.model.max_seq_len,
            "vocab_size": config.model.vocab_size,
        }
    )
    return {
        "model": asdict(config.model),
        "data": data,
        "optimization": asdict(config.optimization),
        "reproducibility": asdict(config.reproducibility),
        "source": _source_manifest(),
        "runtime": _runtime_manifest(),
    }


def _source_manifest() -> dict[str, Any]:
    package_dir = Path(__file__).resolve().parent
    return {
        f"research/src/moira/{path.name}": _file_sha256(path)
        for path in sorted(package_dir.glob("*.py"), key=lambda item: item.name)
    }


def _git_provenance() -> dict[str, Any]:
    repository = Path(__file__).resolve().parent.parents[2]
    try:
        revision = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [
                    "git",
                    "-c",
                    "safe.directory=*",
                    "-C",
                    str(repository),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--",
                    "research/src/moira",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        revision = None
        dirty = None
    return {
        "available": revision is not None,
        "revision": revision,
        "dirty": dirty,
    }


def _runtime_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": "cpu",
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_threads": torch.get_num_threads(),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_strict(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)


def _validate_phase_zero(
    config: object, *, stop_at_step: object, device: object
) -> None:
    if not isinstance(config, TrainingConfig):
        raise ValueError("config must be a TrainingConfig")
    _require_positive_integer("stop_at_step", stop_at_step)
    if stop_at_step > config.optimization.schedule_total_steps:
        raise ValueError("stop_at_step exceeds the semantic schedule")
    if device != "cpu":
        raise ValueError("Phase 0 device must be cpu")
    if config.optimization.gradient_accumulation_steps != 1:
        raise ValueError("Phase 0 does not support gradient accumulation")
    if (
        config.optimization.optimizer != "adamw"
        or config.optimization.schedule != "cosine-with-warmup"
        or config.optimization.precision != "fp32"
        or config.reproducibility.deterministic != "strict"
    ):
        raise ValueError("config is outside the deterministic Phase 0 scope")


def _require_positive_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
