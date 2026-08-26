"""Bounded local-JSONL M1 training paths.

Text/code records use a small in-memory causal smoke path. Capability records
can additionally cross the typed loom boundary and use the separate bounded
M1 checkpoint contract; neither path downloads third-party weights.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import uuid

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW

from .config import ModelConfig
from .dataset import DatasetExample, SourceCatalog, iter_jsonl_examples
from .loom import CapsuleProjector, FeatureCapsuleEncoder, VisionPatchEncoder
from .m1_checkpoint import (
    M1LoadedCheckpoint,
    load_m1_checkpoint,
    save_m1_checkpoint,
)
from .masks import CognitiveMode
from .model import MoiraModel


M1_MAX_STEPS = 1_000
M1_MAX_BATCH_SIZE = 8
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
SEP_ID = 3
BYTE_OFFSET = 4
BYTE_VOCAB_SIZE = BYTE_OFFSET + 256


@dataclass(frozen=True, slots=True)
class CausalBatch:
    input_ids: Tensor
    targets: Tensor
    valid_tokens: Tensor
    loss_mask: Tensor
    record_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CapabilityBatch:
    """Causal targets plus a projected capability prefix.

    Prefix positions use PAD token IDs only as placeholders; the model reads
    their values from ``prefix_hidden`` through ``forward_with_prefix``.
    """

    input_ids: Tensor
    targets: Tensor
    valid_tokens: Tensor
    loss_mask: Tensor
    prefix_hidden: Tensor
    record_ids: tuple[str, ...]
    modalities: tuple[str, ...]


class CapabilityLoom(nn.Module):
    """Trainable, bounded adapter from one capability tensor to model prefix slots."""

    def __init__(
        self,
        *,
        modality: str,
        capsule_width: int,
        d_model: int,
        feature_dim: int | None = None,
        patch_size: int = 16,
        max_slots: int = 16,
    ) -> None:
        super().__init__()
        if not isinstance(modality, str) or not modality.strip():
            raise ValueError("modality must be non-empty")
        if type(capsule_width) is not int or capsule_width <= 0:
            raise ValueError("capsule_width must be positive")
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be positive")
        self.modality = modality.strip()
        self.capsule_width = capsule_width
        self.d_model = d_model
        self.max_slots = max_slots
        if self.modality == "vision":
            self.encoder: nn.Module = VisionPatchEncoder(
                capsule_width=capsule_width,
                patch_size=patch_size,
                max_slots=max_slots,
            )
        else:
            if type(feature_dim) is not int or feature_dim <= 0:
                raise ValueError("feature_dim must be positive for feature modalities")
            self.encoder = FeatureCapsuleEncoder(
                feature_dim=feature_dim,
                capsule_width=capsule_width,
                source_modality=self.modality,
                max_slots=max_slots,
            )
        self.projector = CapsuleProjector(
            capsule_width=capsule_width,
            d_model=d_model,
        )

    def forward(
        self,
        observations: Tensor,
        examples: Sequence[DatasetExample],
        *,
        task_id: str = "moira-m1-capability",
        scope_id: str = "local",
    ) -> Tensor:
        if isinstance(examples, (str, bytes)) or not examples:
            raise ValueError("examples must be a non-empty sequence")
        if (
            not isinstance(observations, Tensor)
            or observations.ndim not in {3, 4}
            or observations.shape[0] != len(examples)
        ):
            raise ValueError("observations and examples must have the same batch")
        if observations.device != next(self.parameters()).device:
            raise ValueError("observations must use the loom device")
        accepted_modalities = {
            "vision-text" if self.modality == "vision" else self.modality
        }
        if any(
            not isinstance(example, DatasetExample)
            or example.modality not in accepted_modalities
            for example in examples
        ):
            raise ValueError("examples do not match the capability modality")
        source_hashes = tuple(f"sha256:{example.record_sha256}" for example in examples)
        capsules = self.encoder(
            observations,
            task_id=task_id,
            scope_id=scope_id,
            source_hashes=source_hashes,
        )
        projected = tuple(
            self.projector(
                capsule,
                task_id=task_id,
                allowed_scope_ids=(scope_id,),
            )
            for capsule in capsules
        )
        slots = {value.shape[0] for value in projected}
        if len(slots) != 1:
            raise ValueError("capability loom must emit one prefix length per batch")
        return torch.stack(projected, dim=0)


@dataclass(frozen=True, slots=True)
class M1TrainingResult:
    steps: int
    samples: int
    tokens: int
    final_loss: float
    trainable_parameters: int
    losses: tuple[float, ...] = ()
    checkpoint_path: Path | None = None
    capability_parameters: int = 0


class ByteTokenizer:
    """Small deterministic UTF-8 tokenizer that fits inside the M1 vocabulary."""

    vocab_size = BYTE_VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str) or not text:
            raise ValueError("tokenizer input must be a non-empty string")
        return [BYTE_OFFSET + value for value in text.encode("utf-8")]

    def encode_pair(self, prompt: str, target: str, *, sequence_length: int) -> CausalBatch:
        if type(sequence_length) is not int or sequence_length < 4:
            raise ValueError("sequence_length must be at least 4")
        prompt_ids = self.encode(prompt)
        target_ids = self.encode(target)
        sequence = [BOS_ID, *prompt_ids, SEP_ID, *target_ids, EOS_ID]
        if len(sequence) < 2:
            raise ValueError("encoded pair must contain at least two tokens")
        sequence = sequence[:sequence_length]
        input_ids = torch.full((sequence_length - 1,), PAD_ID, dtype=torch.int64)
        targets = torch.full((sequence_length - 1,), PAD_ID, dtype=torch.int64)
        valid_tokens = torch.zeros((sequence_length - 1,), dtype=torch.bool)
        loss_mask = torch.zeros((sequence_length - 1,), dtype=torch.bool)
        input_ids[: len(sequence) - 1] = torch.tensor(sequence[:-1], dtype=torch.int64)
        targets[: len(sequence) - 1] = torch.tensor(sequence[1:], dtype=torch.int64)
        valid_tokens[: len(sequence) - 1] = True
        target_start = len(prompt_ids) + 2
        if target_start >= len(sequence):
            raise ValueError("sequence_length leaves no target token")
        loss_mask[target_start - 1 : len(sequence) - 1] = True
        return CausalBatch(
            input_ids=input_ids.unsqueeze(0),
            targets=targets.unsqueeze(0),
            valid_tokens=valid_tokens.unsqueeze(0),
            loss_mask=loss_mask.unsqueeze(0),
            record_ids=(),
        )


def build_causal_batch(
    examples: tuple[DatasetExample, ...],
    *,
    sequence_length: int,
    tokenizer: ByteTokenizer | None = None,
) -> CausalBatch:
    """Convert text/code examples into a padded batch with target-only loss."""
    if not examples:
        raise ValueError("examples must not be empty")
    tokenizer = tokenizer or ByteTokenizer()
    rows = []
    for example in examples:
        if not isinstance(example, DatasetExample):
            raise ValueError("examples must contain DatasetExample values")
        example.validate()
        prompt = example.inputs.get("text")
        target = example.targets.get("text")
        if not isinstance(prompt, str) or not isinstance(target, str):
            raise ValueError("M1 causal batches require text inputs and targets")
        rows.append(
            tokenizer.encode_pair(
                prompt,
                target,
                sequence_length=sequence_length,
            )
        )
    return CausalBatch(
        input_ids=torch.cat([row.input_ids for row in rows], dim=0),
        targets=torch.cat([row.targets for row in rows], dim=0),
        valid_tokens=torch.cat([row.valid_tokens for row in rows], dim=0),
        loss_mask=torch.cat([row.loss_mask for row in rows], dim=0),
        record_ids=tuple(example.record_id for example in examples),
    )


def build_capability_batch(
    examples: tuple[DatasetExample, ...],
    *,
    prefix_hidden: Tensor,
    sequence_length: int,
    tokenizer: ByteTokenizer | None = None,
) -> CapabilityBatch:
    """Build target-only causal supervision around projected capability slots."""
    if not examples:
        raise ValueError("examples must not be empty")
    if (
        not isinstance(prefix_hidden, Tensor)
        or not prefix_hidden.is_floating_point()
        or prefix_hidden.ndim != 3
        or prefix_hidden.shape[0] != len(examples)
        or prefix_hidden.shape[1] <= 0
        or not bool(torch.isfinite(prefix_hidden).all())
    ):
        raise ValueError("prefix_hidden must be finite floating [batch, prefix, width]")
    if type(sequence_length) is not int or sequence_length < 4:
        raise ValueError("sequence_length must be at least 4")
    prefix_length = int(prefix_hidden.shape[1])
    text_length = sequence_length - prefix_length
    if text_length < 4:
        raise ValueError("sequence_length leaves no room for a causal text target")
    tokenizer = tokenizer or ByteTokenizer()
    rows = []
    for example in examples:
        if not isinstance(example, DatasetExample):
            raise ValueError("examples must contain DatasetExample values")
        example.validate()
        prompt, target = _capability_text_pair(example)
        rows.append(
            tokenizer.encode_pair(
                prompt,
                target,
                sequence_length=text_length,
            )
        )
    text_inputs = torch.cat([row.input_ids for row in rows], dim=0)
    text_targets = torch.cat([row.targets for row in rows], dim=0)
    text_valid = torch.cat([row.valid_tokens for row in rows], dim=0)
    text_loss = torch.cat([row.loss_mask for row in rows], dim=0)
    batch = len(examples)
    prefix_ids = torch.full(
        (batch, prefix_length), PAD_ID, dtype=torch.int64, device=text_inputs.device
    )
    prefix_targets = torch.full_like(prefix_ids, PAD_ID)
    prefix_valid = torch.ones(
        (batch, prefix_length), dtype=torch.bool, device=text_inputs.device
    )
    prefix_loss = torch.zeros_like(prefix_valid)
    return CapabilityBatch(
        input_ids=torch.cat((prefix_ids, text_inputs), dim=1),
        targets=torch.cat((prefix_targets, text_targets), dim=1),
        valid_tokens=torch.cat((prefix_valid, text_valid), dim=1),
        loss_mask=torch.cat((prefix_loss, text_loss), dim=1),
        prefix_hidden=prefix_hidden,
        record_ids=tuple(example.record_id for example in examples),
        modalities=tuple(example.modality for example in examples),
    )


def _capability_text_pair(example: DatasetExample) -> tuple[str, str]:
    text = example.inputs.get("text")
    if isinstance(text, str) and text.strip():
        prompt = text
    else:
        prompt = json.dumps(
            {"modality": example.modality, "inputs": dict(example.inputs)},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    target_value = example.targets.get("text")
    if not isinstance(target_value, str) or not target_value.strip():
        target_value = next(
            (
                value
                for key, value in example.targets.items()
                if key in {"labels", "action", "target", "output"}
            ),
            None,
        )
    if isinstance(target_value, str) and target_value.strip():
        target = target_value
    elif target_value is not None:
        target = json.dumps(target_value, sort_keys=True, ensure_ascii=False, default=str)
    else:
        raise ValueError("capability example requires a non-empty target")
    return prompt, target


def iter_cataloged_text_examples(
    catalog_path: str | Path,
    jsonl_path: str | Path,
    *,
    source_id: str,
    max_records: int,
) -> Iterator[DatasetExample]:
    catalog = SourceCatalog.from_toml(catalog_path)
    source = catalog.by_id(source_id)
    if source.modality not in {"text", "code"}:
        raise ValueError("M1 causal training accepts text or code catalog sources")
    return iter_jsonl_examples(
        jsonl_path,
        source_id=source_id,
        split="train",
        preprocess_version="m1-v1",
        split_seed=0,
        max_records=max_records,
    )


def iter_cataloged_capability_examples(
    catalog_path: str | Path,
    jsonl_path: str | Path,
    *,
    source_id: str,
    max_records: int,
) -> Iterator[DatasetExample]:
    """Stream vision/audio/structured/robotics examples from a cataloged JSONL."""
    catalog = SourceCatalog.from_toml(catalog_path)
    source = catalog.by_id(source_id)
    if source.modality not in {
        "vision-text",
        "audio",
        "structured",
        "robotics",
        "time-series",
    }:
        raise ValueError("M1 capability training requires a capability catalog source")
    return iter_jsonl_examples(
        jsonl_path,
        source_id=source_id,
        split="train",
        preprocess_version="m1-v1",
        split_seed=0,
        max_records=max_records,
    )


def train_m1_jsonl(
    catalog_path: str | Path,
    jsonl_path: str | Path,
    *,
    source_id: str,
    steps: int = 1,
    batch_size: int = 1,
    device: str = "cpu",
) -> M1TrainingResult:
    """Run bounded in-memory M1 causal training on a caller-owned JSONL file."""
    if type(steps) is not int or not 0 < steps <= M1_MAX_STEPS:
        raise ValueError("steps must be between 1 and the M1 step limit")
    if type(batch_size) is not int or not 0 < batch_size <= M1_MAX_BATCH_SIZE:
        raise ValueError("batch_size exceeds the M1 in-memory limit")
    if device not in {"cpu", "cuda"}:
        raise ValueError("M1 device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    config = ModelConfig.m1_cpu()
    model = MoiraModel(config).to(device=device, dtype=torch.float32)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, foreach=False)
    examples = iter_cataloged_text_examples(
        catalog_path,
        jsonl_path,
        source_id=source_id,
        max_records=steps * batch_size,
    )
    tokenizer = ByteTokenizer()
    model.train()
    final_loss = 0.0
    samples = 0
    tokens = 0
    for _ in range(steps):
        batch_examples_list: list[DatasetExample] = []
        for _ in range(batch_size):
            try:
                batch_examples_list.append(next(examples))
            except StopIteration as error:
                raise ValueError("JSONL source ended before the requested M1 steps") from error
        batch_examples = tuple(batch_examples_list)
        batch = build_causal_batch(
            batch_examples,
            sequence_length=config.max_seq_len,
            tokenizer=tokenizer,
        )
        input_ids = batch.input_ids.to(device)
        targets = batch.targets.to(device)
        valid_tokens = batch.valid_tokens.to(device)
        loss_mask = batch.loss_mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids,
            mode=CognitiveMode.ATROPOS,
            valid_tokens=valid_tokens,
        )
        loss = _masked_loss(output.logits, targets, loss_mask)
        if not bool(torch.isfinite(loss)):
            raise ValueError("M1 training loss became non-finite")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
        samples += batch_size
        tokens += int(loss_mask.sum().item())
    return M1TrainingResult(
        steps=steps,
        samples=samples,
        tokens=tokens,
        final_loss=final_loss,
        trainable_parameters=model.parameter_ledger().total_trainable,
    )


def train_m1_capabilities(
    catalog_path: str | Path,
    jsonl_path: str | Path,
    observations: Tensor,
    *,
    source_id: str,
    modality: str,
    steps: int = 1,
    batch_size: int = 1,
    device: str = "cpu",
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    model_config: ModelConfig | None = None,
    max_slots: int = 16,
    patch_size: int = 16,
    scope_id: str = "local",
) -> M1TrainingResult:
    """Train the M1 model on text plus one bounded capability observation tensor.

    ``observations`` is intentionally caller-owned: vision uses RGB ``[N,3,H,W]``
    tensors, while audio/time-series/graph/robotics/structured inputs use
    ``[N,T,F]`` tensors.  The JSONL rows provide the text target and provenance;
    the loom supplies a fixed-capacity prefix without importing third-party
    model weights.
    """
    _validate_m1_limits(steps=steps, batch_size=batch_size, device=device)
    if not isinstance(modality, str) or not modality.strip():
        raise ValueError("modality must be non-empty")
    if not isinstance(scope_id, str) or not scope_id.strip():
        raise ValueError("scope_id must be non-empty")
    if (
        not isinstance(observations, Tensor)
        or not observations.is_floating_point()
        or observations.ndim not in {3, 4}
        or observations.shape[0] <= 0
        or not bool(torch.isfinite(observations).all())
    ):
        raise ValueError("observations must be finite floating [N,T,F] or [N,3,H,W]")
    if modality == "vision":
        if observations.ndim != 4 or observations.shape[1] != 3:
            raise ValueError("vision observations must be RGB [N,3,H,W]")
    elif observations.ndim != 3 or observations.shape[1] <= 0 or observations.shape[2] <= 0:
        raise ValueError("feature observations must be [N,T,F]")
    config = model_config or ModelConfig.m1_cpu()
    model = MoiraModel(config).to(device=device, dtype=torch.float32)
    observations = observations.to(device=device, dtype=torch.float32)
    loom = CapabilityLoom(
        modality=modality,
        capsule_width=config.capsule_width,
        d_model=config.d_model,
        feature_dim=None if modality == "vision" else int(observations.shape[-1]),
        patch_size=patch_size,
        max_slots=max_slots,
    ).to(device=device, dtype=torch.float32)
    optimizer = AdamW(
        list(model.parameters()) + list(loom.parameters()),
        lr=3e-4,
        weight_decay=0.01,
        foreach=False,
    )
    manifest = {
        "model_config": asdict(config),
        "source_id": source_id,
        "modality": modality,
        "sequence_length": config.max_seq_len,
        "observation_shape": list(observations.shape[1:]),
        "max_slots": max_slots,
        "patch_size": patch_size,
        "scope_id": scope_id,
    }
    examples = tuple(
        iter_cataloged_capability_examples(
            catalog_path,
            jsonl_path,
            source_id=source_id,
            max_records=steps * batch_size,
        )
    )
    if len(examples) < steps * batch_size or observations.shape[0] < steps * batch_size:
        raise ValueError("capability data ended before the requested M1 steps")
    lineage_id = uuid.uuid4().hex
    global_step = 0
    sample_cursor = 0
    losses: list[float] = []
    tokens_seen = 0
    checkpoint: M1LoadedCheckpoint | None = None
    if resume:
        if checkpoint_path is None:
            raise ValueError("resume requires checkpoint_path")
        checkpoint = load_m1_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            loom=loom,
            expected_manifest=manifest,
        )
        lineage_id = checkpoint.lineage_id
        global_step = checkpoint.global_step
        sample_cursor = checkpoint.sample_cursor
        losses = list(checkpoint.losses)
    if global_step > steps:
        raise ValueError("requested steps are behind the checkpoint")
    if sample_cursor + (steps - global_step) * batch_size > len(examples):
        raise ValueError("checkpoint cursor exceeds the bounded capability data")
    model.train()
    loom.train()
    while global_step < steps:
        start = sample_cursor
        end = start + batch_size
        batch_examples = tuple(examples[start:end])
        prefix_hidden = loom(
            observations[start:end],
            batch_examples,
            scope_id=scope_id,
        )
        batch = build_capability_batch(
            batch_examples,
            prefix_hidden=prefix_hidden,
            sequence_length=config.max_seq_len,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_with_prefix(
            batch.input_ids,
            batch.prefix_hidden,
            mode=CognitiveMode.ATROPOS,
            valid_tokens=batch.valid_tokens,
            tokens_include_prefix=True,
        )
        loss = _masked_loss(output.logits, batch.targets, batch.loss_mask)
        if not bool(torch.isfinite(loss)):
            raise ValueError("M1 capability training loss became non-finite")
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(loom.parameters()),
            1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()
        global_step += 1
        sample_cursor = end
        loss_value = float(loss.detach().cpu().item())
        tokens_seen += int(batch.loss_mask.sum().item())
        losses.append(loss_value)
        if checkpoint_path is not None:
            save_m1_checkpoint(
                checkpoint_path,
                lineage_id=lineage_id,
                global_step=global_step,
                sample_cursor=sample_cursor,
                model=model,
                optimizer=optimizer,
                loom=loom,
                losses=tuple(losses),
                manifest=manifest,
            )
    return M1TrainingResult(
        steps=steps,
        samples=sample_cursor,
        tokens=tokens_seen,
        final_loss=losses[-1] if losses else 0.0,
        trainable_parameters=model.parameter_ledger().total_trainable,
        losses=tuple(losses),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path is not None else None,
        capability_parameters=sum(
            parameter.numel()
            for parameter in {
                id(parameter): parameter
                for parameter in loom.parameters()
                if parameter.requires_grad
            }.values()
        ),
    )


def _validate_m1_limits(*, steps: int, batch_size: int, device: str) -> None:
    if type(steps) is not int or not 0 < steps <= M1_MAX_STEPS:
        raise ValueError("steps must be between 1 and the M1 step limit")
    if type(batch_size) is not int or not 0 < batch_size <= M1_MAX_BATCH_SIZE:
        raise ValueError("batch_size exceeds the M1 in-memory limit")
    if device not in {"cpu", "cuda"}:
        raise ValueError("M1 device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")


def _masked_loss(logits: Tensor, targets: Tensor, loss_mask: Tensor) -> Tensor:
    if logits.shape[:2] != targets.shape or targets.shape != loss_mask.shape:
        raise ValueError("M1 causal batch tensors have incompatible shapes")
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    weights = loss_mask.to(dtype=per_token.dtype)
    denominator = weights.sum()
    if not bool(denominator > 0):
        raise ValueError("M1 loss mask must select at least one token")
    return (per_token * weights).sum() / denominator
