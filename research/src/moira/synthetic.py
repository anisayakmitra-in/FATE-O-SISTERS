"""Deterministic indexed synthetic data for CPU MOIRA research runs."""

from dataclasses import dataclass

import torch
from torch import Tensor


SYNTHETIC_CURSOR_KIND = "synthetic-indexed-v1"
_UINT64_MASK = (1 << 64) - 1
_TORCH_SEED_MASK = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class SyntheticSample:
    input_ids: Tensor
    targets: Tensor
    loss_mask: Tensor
    split: int
    index: int


@dataclass(frozen=True, slots=True)
class SyntheticBatch:
    input_ids: Tensor
    targets: Tensor
    loss_mask: Tensor
    split: int
    start_index: int


@dataclass(frozen=True, slots=True)
class DataCursor:
    kind: str
    next_sample_index: int
    samples_seen: int
    tokens_seen: int

    def __post_init__(self) -> None:
        if self.kind != SYNTHETIC_CURSOR_KIND:
            raise ValueError("cursor kind must be synthetic-indexed-v1")
        for name, value in (
            ("next_sample_index", self.next_sample_index),
            ("samples_seen", self.samples_seen),
            ("tokens_seen", self.tokens_seen),
        ):
            _require_non_negative_integer(f"cursor {name}", value)
        if self.next_sample_index != self.samples_seen:
            raise ValueError("cursor sample counters must agree")


class SyntheticStream:
    """Generate each sample from its address without touching global RNG state."""

    def __init__(
        self,
        seed: int,
        vocab_size: int,
        sequence_length: int,
        version: int = 1,
    ) -> None:
        _require_non_negative_integer("seed", seed)
        _require_positive_integer("version", version)
        _require_positive_integer("vocab_size", vocab_size)
        _require_positive_integer("sequence_length", sequence_length)
        if vocab_size < 4:
            raise ValueError("vocab_size must be at least 4")
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.seed = seed
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.version = version

    def sample(self, *, split: int, index: int) -> SyntheticSample:
        _require_non_negative_integer("split", split)
        _require_non_negative_integer("index", index)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._sample_seed(split, index))
        pattern_length = min(8, max(2, self.sequence_length // 2))
        pattern = torch.randint(
            0,
            self.vocab_size,
            (pattern_length,),
            dtype=torch.int64,
            generator=generator,
        )
        phase = int(
            torch.randint(
                0,
                pattern_length,
                (),
                dtype=torch.int64,
                generator=generator,
            ).item()
        )
        positions = (
            torch.arange(self.sequence_length + 1, dtype=torch.int64) + phase
        ) % pattern_length
        sequence = pattern[positions]
        return SyntheticSample(
            input_ids=sequence[:-1].clone(),
            targets=sequence[1:].clone(),
            loss_mask=torch.ones(self.sequence_length, dtype=torch.bool),
            split=split,
            index=index,
        )

    def batch(
        self,
        *,
        split: int,
        cursor: DataCursor,
        batch_size: int,
    ) -> tuple[SyntheticBatch, DataCursor]:
        _require_non_negative_integer("split", split)
        _require_positive_integer("batch_size", batch_size)
        if not isinstance(cursor, DataCursor):
            raise ValueError("cursor must be a DataCursor")
        if cursor.tokens_seen != cursor.samples_seen * self.sequence_length:
            raise ValueError("cursor token counter does not match the stream")

        samples = tuple(
            self.sample(split=split, index=cursor.next_sample_index + offset)
            for offset in range(batch_size)
        )
        batch = SyntheticBatch(
            input_ids=torch.stack(tuple(sample.input_ids for sample in samples)),
            targets=torch.stack(tuple(sample.targets for sample in samples)),
            loss_mask=torch.stack(tuple(sample.loss_mask for sample in samples)),
            split=split,
            start_index=cursor.next_sample_index,
        )
        next_samples = cursor.samples_seen + batch_size
        return batch, DataCursor(
            kind=SYNTHETIC_CURSOR_KIND,
            next_sample_index=cursor.next_sample_index + batch_size,
            samples_seen=next_samples,
            tokens_seen=cursor.tokens_seen + batch_size * self.sequence_length,
        )

    def _sample_seed(self, split: int, index: int) -> int:
        mixed = _splitmix64(self.seed)
        mixed = _splitmix64(mixed ^ self.version)
        mixed = _splitmix64(mixed ^ split)
        mixed = _splitmix64(mixed ^ index)
        return mixed & _TORCH_SEED_MASK


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _require_non_negative_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
