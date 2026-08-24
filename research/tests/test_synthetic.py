import random
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch

from moira.synthetic import (
    SYNTHETIC_CURSOR_KIND,
    DataCursor,
    SyntheticStream,
    _splitmix64,
)


def _rng_snapshot() -> tuple[object, tuple[object, ...], torch.Tensor]:
    numpy_state = np.random.get_state()
    return (
        random.getstate(),
        (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        torch.get_rng_state().clone(),
    )


def _assert_rng_equal(
    left: tuple[object, tuple[object, ...], torch.Tensor],
    right: tuple[object, tuple[object, ...], torch.Tensor],
) -> None:
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    assert np.array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


def _assert_sample_equal(left: object, right: object) -> None:
    assert type(left) is type(right)
    assert left.split == right.split
    assert left.index == right.index
    assert torch.equal(left.input_ids, right.input_ids)
    assert torch.equal(left.targets, right.targets)
    assert torch.equal(left.loss_mask, right.loss_mask)


def _assert_batch_equal(left: object, right: object) -> None:
    assert type(left) is type(right)
    assert left.split == right.split
    assert left.start_index == right.start_index
    assert torch.equal(left.input_ids, right.input_ids)
    assert torch.equal(left.targets, right.targets)
    assert torch.equal(left.loss_mask, right.loss_mask)


def test_sample_is_an_exact_function_of_version_seed_split_and_index() -> None:
    first = SyntheticStream(seed=41, vocab_size=19, sequence_length=9, version=1)
    rebuilt = SyntheticStream(seed=41, vocab_size=19, sequence_length=9, version=1)

    sample = first.sample(split=3, index=7)
    _assert_sample_equal(sample, rebuilt.sample(split=3, index=7))

    variants = (
        SyntheticStream(seed=42, vocab_size=19, sequence_length=9, version=1).sample(
            split=3, index=7
        ),
        first.sample(split=4, index=7),
        first.sample(split=3, index=8),
        SyntheticStream(seed=41, vocab_size=19, sequence_length=9, version=2).sample(
            split=3, index=7
        ),
    )
    assert all(
        not torch.equal(sample.input_ids, variant.input_ids) for variant in variants
    )

    assert sample.input_ids.dtype is torch.int64
    assert sample.targets.dtype is torch.int64
    assert sample.loss_mask.dtype is torch.bool
    assert sample.input_ids.shape == sample.targets.shape == sample.loss_mask.shape == (9,)
    assert bool(torch.all(sample.input_ids >= 0))
    assert bool(torch.all(sample.input_ids < 19))
    assert bool(torch.all(sample.targets >= 0))
    assert bool(torch.all(sample.targets < 19))
    assert torch.equal(sample.input_ids[1:], sample.targets[:-1])
    assert bool(sample.loss_mask.all())

    with pytest.raises(FrozenInstanceError):
        sample.index = 9


def test_sample_and_batch_do_not_change_global_rng_states() -> None:
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    before = _rng_snapshot()
    stream = SyntheticStream(seed=9, vocab_size=23, sequence_length=7)
    cursor = DataCursor(SYNTHETIC_CURSOR_KIND, 0, 0, 0)

    stream.sample(split=0, index=4)
    stream.batch(split=0, cursor=cursor, batch_size=3)

    _assert_rng_equal(before, _rng_snapshot())


def test_saved_cursor_rebuilds_the_exact_next_batch_without_mutation() -> None:
    stream = SyntheticStream(seed=5, vocab_size=31, sequence_length=6)
    initial = DataCursor(SYNTHETIC_CURSOR_KIND, 0, 0, 0)

    first, saved = stream.batch(split=0, cursor=initial, batch_size=2)
    expected, expected_cursor = stream.batch(split=0, cursor=saved, batch_size=3)
    rebuilt, rebuilt_cursor = stream.batch(split=0, cursor=saved, batch_size=3)

    assert initial == DataCursor(SYNTHETIC_CURSOR_KIND, 0, 0, 0)
    assert first.start_index == 0
    assert saved == DataCursor(SYNTHETIC_CURSOR_KIND, 2, 2, 12)
    _assert_batch_equal(expected, rebuilt)
    assert expected_cursor == rebuilt_cursor == DataCursor(
        SYNTHETIC_CURSOR_KIND, 5, 5, 30
    )
    with pytest.raises(FrozenInstanceError):
        saved.samples_seen = 99


def test_evaluation_split_cannot_advance_the_training_cursor() -> None:
    stream = SyntheticStream(seed=7, vocab_size=29, sequence_length=5)
    training_cursor = DataCursor(SYNTHETIC_CURSOR_KIND, 0, 0, 0)
    expected_train, expected_cursor = stream.batch(
        split=0, cursor=training_cursor, batch_size=2
    )

    evaluation, evaluation_cursor = stream.batch(
        split=1,
        cursor=DataCursor(SYNTHETIC_CURSOR_KIND, 0, 0, 0),
        batch_size=2,
    )
    actual_train, actual_cursor = stream.batch(
        split=0, cursor=training_cursor, batch_size=2
    )

    assert training_cursor == DataCursor(SYNTHETIC_CURSOR_KIND, 0, 0, 0)
    assert evaluation_cursor == DataCursor(SYNTHETIC_CURSOR_KIND, 2, 2, 10)
    assert not torch.equal(evaluation.input_ids, actual_train.input_ids)
    _assert_batch_equal(expected_train, actual_train)
    assert expected_cursor == actual_cursor


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1, "vocab_size": 8, "sequence_length": 4},
        {"seed": True, "vocab_size": 8, "sequence_length": 4},
        {"seed": 1, "vocab_size": 3, "sequence_length": 4},
        {"seed": 1, "vocab_size": 8, "sequence_length": 1},
        {"seed": 1, "vocab_size": 8, "sequence_length": 4, "version": 0},
    ],
)
def test_stream_rejects_invalid_constructor_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SyntheticStream(**kwargs)


@pytest.mark.parametrize("split", [-1, True, 1.5])
def test_sample_rejects_invalid_splits(split: object) -> None:
    stream = SyntheticStream(seed=1, vocab_size=8, sequence_length=4)
    with pytest.raises(ValueError, match="split"):
        stream.sample(split=split, index=0)


@pytest.mark.parametrize("index", [-1, True, 1.5])
def test_sample_rejects_invalid_indices(index: object) -> None:
    stream = SyntheticStream(seed=1, vocab_size=8, sequence_length=4)
    with pytest.raises(ValueError, match="index"):
        stream.sample(split=0, index=index)


@pytest.mark.parametrize(
    "values",
    [
        ("wrong-kind", 0, 0, 0),
        (SYNTHETIC_CURSOR_KIND, -1, 0, 0),
        (SYNTHETIC_CURSOR_KIND, 1, 0, 0),
        (SYNTHETIC_CURSOR_KIND, 0, True, 0),
    ],
)
def test_cursor_rejects_invalid_kind_or_counters(values: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="cursor"):
        DataCursor(*values)


def test_batch_rejects_a_cursor_with_the_wrong_token_counter() -> None:
    stream = SyntheticStream(seed=1, vocab_size=8, sequence_length=4)
    cursor = DataCursor(SYNTHETIC_CURSOR_KIND, 1, 1, 3)
    with pytest.raises(ValueError, match="cursor"):
        stream.batch(split=0, cursor=cursor, batch_size=1)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_batch_rejects_invalid_batch_sizes(batch_size: object) -> None:
    stream = SyntheticStream(seed=1, vocab_size=8, sequence_length=4)
    cursor = DataCursor(SYNTHETIC_CURSOR_KIND, 0, 0, 0)
    with pytest.raises(ValueError, match="batch_size"):
        stream.batch(split=0, cursor=cursor, batch_size=batch_size)


def test_splitmix64_address_boundaries_are_explicit() -> None:
    maximum = (1 << 64) - 1
    stream = SyntheticStream(
        seed=maximum,
        vocab_size=8,
        sequence_length=4,
        version=maximum,
    )

    sample = stream.sample(split=maximum, index=maximum)

    assert sample.split == maximum
    assert sample.index == maximum
    with pytest.raises(ValueError, match="seed"):
        SyntheticStream(seed=1 << 64, vocab_size=8, sequence_length=4)
    with pytest.raises(ValueError, match="version"):
        SyntheticStream(
            seed=1, vocab_size=8, sequence_length=4, version=1 << 64
        )
    with pytest.raises(ValueError, match="split"):
        stream.sample(split=1 << 64, index=0)
    with pytest.raises(ValueError, match="index"):
        stream.sample(split=0, index=1 << 64)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (0, 0xE220A8397B1DCDAF),
        (1, 0x910A2DEC89025CC1),
        ((1 << 64) - 1, 0xE4D971771B652C20),
    ),
)
def test_splitmix64_known_answer_vectors(value: int, expected: int) -> None:
    assert _splitmix64(value) == expected


def test_consuming_the_maximum_address_returns_only_the_exhausted_sentinel() -> None:
    maximum = (1 << 64) - 1
    stream = SyntheticStream(seed=1, vocab_size=8, sequence_length=4)
    cursor = DataCursor(
        SYNTHETIC_CURSOR_KIND,
        maximum,
        maximum,
        maximum * stream.sequence_length,
    )

    batch, exhausted = stream.batch(split=0, cursor=cursor, batch_size=1)

    assert batch.start_index == maximum
    assert exhausted.next_sample_index == 1 << 64
    with pytest.raises(ValueError, match="address"):
        stream.batch(split=0, cursor=exhausted, batch_size=1)
    with pytest.raises(ValueError, match="cursor"):
        DataCursor(SYNTHETIC_CURSOR_KIND, (1 << 64) + 1, (1 << 64) + 1, 0)


def test_batch_address_overflow_fails_before_sample_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maximum = (1 << 64) - 1
    stream = SyntheticStream(seed=1, vocab_size=8, sequence_length=4)
    cursor = DataCursor(
        SYNTHETIC_CURSOR_KIND,
        maximum,
        maximum,
        maximum * stream.sequence_length,
    )
    calls = 0
    original_sample = stream.sample

    def observed_sample(*, split: int, index: int):
        nonlocal calls
        calls += 1
        return original_sample(split=split, index=index)

    monkeypatch.setattr(stream, "sample", observed_sample)
    with pytest.raises(ValueError, match="address"):
        stream.batch(split=0, cursor=cursor, batch_size=2)

    assert calls == 0
    assert cursor.next_sample_index == maximum
