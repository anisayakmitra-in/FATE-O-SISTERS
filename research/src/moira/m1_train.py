"""Bounded local-JSONL M1 training path.

This path is intentionally in-memory and checkpoint-free.  It proves that a
cataloged text/code source can produce a deterministic causal-language batch
for the M1 model without adding a tokenizer or dataset dependency.  The
existing checkpointed trainer remains the M0 reference until M1 checkpoint
limits and resume contracts are reviewed.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW

from .config import ModelConfig
from .dataset import DatasetExample, SourceCatalog, iter_jsonl_examples
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
class M1TrainingResult:
    steps: int
    samples: int
    tokens: int
    final_loss: float
    trainable_parameters: int


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
