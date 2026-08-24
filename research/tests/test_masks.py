import pytest
import torch

from moira.masks import CognitiveMode, build_visibility_mask


@pytest.mark.parametrize(
    ("mode", "source_lengths", "expected"),
    [
        (
            CognitiveMode.CLOTHO,
            None,
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        ),
        (
            CognitiveMode.LACHESIS,
            torch.tensor([2]),
            [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
        ),
        (
            CognitiveMode.ATROPOS,
            None,
            [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
        ),
    ],
)
def test_orientation_truth_tables(mode, source_lengths, expected) -> None:
    """Catches an incorrect directional rule for any cognitive orientation."""
    valid_tokens = torch.ones((1, 4), dtype=torch.bool)

    actual = build_visibility_mask(mode, valid_tokens, source_lengths)

    assert torch.equal(actual[0], torch.tensor(expected, dtype=torch.bool))


@pytest.mark.parametrize(
    ("mode", "source_lengths", "expected"),
    [
        (
            CognitiveMode.CLOTHO,
            None,
            [
                [[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
        ),
        (
            CognitiveMode.LACHESIS,
            torch.tensor([2, 1]),
            [
                [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
        ),
        (
            CognitiveMode.ATROPOS,
            None,
            [
                [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            ],
        ),
    ],
)
def test_padding_removes_each_padded_query_row_and_key_column(
    mode, source_lengths, expected
) -> None:
    """Catches padding leaking as either a visible query or visible key."""
    valid_tokens = torch.tensor(
        [[True, True, True, False], [True, False, False, False]]
    )

    actual = build_visibility_mask(mode, valid_tokens, source_lengths)

    assert torch.equal(actual, torch.tensor(expected, dtype=torch.bool))


@pytest.mark.parametrize(
    "mode",
    [CognitiveMode.CLOTHO, CognitiveMode.LACHESIS, CognitiveMode.ATROPOS],
)
def test_mask_is_bool_and_uses_the_input_device(mode) -> None:
    """Catches output allocation on a different device or with a numeric dtype."""
    valid_tokens = torch.ones((2, 3), dtype=torch.bool)
    source_lengths = torch.tensor([1, 2]) if mode is CognitiveMode.LACHESIS else None

    actual = build_visibility_mask(mode, valid_tokens, source_lengths)

    assert actual.dtype is torch.bool
    assert actual.device == valid_tokens.device


@pytest.mark.parametrize(
    "mode",
    ["clotho", object(), None],
)
def test_mask_rejects_modes_outside_cognitive_mode(mode) -> None:
    """Catches accepting a mode that bypasses the defined visibility regimes."""
    with pytest.raises(ValueError, match="mode"):
        build_visibility_mask(mode, torch.ones((1, 2), dtype=torch.bool))


@pytest.mark.parametrize(
    "valid_tokens",
    [
        torch.ones((1, 2), dtype=torch.int64),
        torch.ones((2,), dtype=torch.bool),
        torch.ones((1, 2, 1), dtype=torch.bool),
    ],
)
def test_mask_rejects_non_boolean_or_non_matrix_valid_tokens(valid_tokens) -> None:
    """Catches malformed token-validity input that can produce unsafe padding masks."""
    with pytest.raises(ValueError, match="valid_tokens"):
        build_visibility_mask(CognitiveMode.CLOTHO, valid_tokens)


@pytest.mark.parametrize(
    "source_lengths",
    [
        None,
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([[2]], dtype=torch.int64),
        torch.tensor([2, 1], dtype=torch.int64),
        torch.tensor([-1], dtype=torch.int64),
        torch.tensor([4], dtype=torch.int64),
    ],
)
def test_lachesis_rejects_missing_malformed_or_out_of_range_source_lengths(
    source_lengths,
) -> None:
    """Catches an unsealed, malformed, or padded source segment boundary."""
    valid_tokens = torch.tensor([[True, True, True, False]])

    with pytest.raises(ValueError, match="source_lengths"):
        build_visibility_mask(CognitiveMode.LACHESIS, valid_tokens, source_lengths)


def test_lachesis_rejects_source_lengths_from_a_different_device() -> None:
    """Catches cross-device source boundaries that cannot safely form a mask."""
    valid_tokens = torch.ones((1, 2), dtype=torch.bool)
    source_lengths = torch.tensor([1], dtype=torch.int64, device="meta")

    with pytest.raises(ValueError, match="source_lengths"):
        build_visibility_mask(CognitiveMode.LACHESIS, valid_tokens, source_lengths)
