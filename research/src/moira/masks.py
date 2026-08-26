"""Cognitive-orientation visibility masks for MOIRA attention."""

from enum import Enum

import torch


class CognitiveMode(str, Enum):
    CLOTHO = "clotho"
    LACHESIS = "lachesis"
    ATROPOS = "atropos"


def build_visibility_mask(
    mode: CognitiveMode,
    valid_tokens: torch.Tensor,
    source_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the full visibility matrix for callers that need arbitrary keys."""
    batch, length = validate_visibility_inputs(mode, valid_tokens, source_lengths)
    device = valid_tokens.device
    query = torch.arange(length, device=device).view(1, length, 1)
    key = torch.arange(length, device=device).view(1, 1, length)

    if mode is CognitiveMode.CLOTHO:
        allowed = torch.ones((batch, length, length), dtype=torch.bool, device=device)
    elif mode is CognitiveMode.ATROPOS:
        allowed = (key <= query).expand(batch, -1, -1)
    else:
        assert source_lengths is not None
        source = source_lengths.view(batch, 1, 1)
        query_is_source = query < source
        key_is_source = key < source
        allowed = (query_is_source & key_is_source) | (
            ~query_is_source & (key_is_source | (key <= query))
        )

    return allowed & valid_tokens[:, :, None] & valid_tokens[:, None, :]


def build_local_visibility_mask(
    mode: CognitiveMode,
    valid_tokens: torch.Tensor,
    radius: int,
    source_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build compact visibility for a symmetric ``2 * radius + 1`` key window.

    Out-of-range key slots remain present so every query has the same window
    width, but are marked invisible. This keeps Thread attention linear in the
    local window rather than quadratic in sequence length.
    """
    batch, length = validate_visibility_inputs(mode, valid_tokens, source_lengths)
    if type(radius) is not int or radius < 0:
        raise ValueError("radius must be a non-negative integer")

    device = valid_tokens.device
    query = torch.arange(length, device=device).view(length, 1)
    offsets = torch.arange(-radius, radius + 1, device=device).view(1, -1)
    key = query + offsets
    in_bounds = (key >= 0) & (key < length)
    clamped_key = key.clamp(0, max(length - 1, 0))
    key_valid = valid_tokens[:, clamped_key]

    if mode is CognitiveMode.CLOTHO:
        allowed = torch.ones(
            (batch, length, offsets.shape[1]), dtype=torch.bool, device=device
        )
    elif mode is CognitiveMode.ATROPOS:
        allowed = (key <= query).unsqueeze(0).expand(batch, -1, -1)
    else:
        assert source_lengths is not None
        source = source_lengths.view(batch, 1, 1)
        query_is_source = query.view(1, length, 1) < source
        key_is_source = key.view(1, length, -1) < source
        allowed = (query_is_source & key_is_source) | (
            ~query_is_source
            & (key_is_source | (key.view(1, length, -1) <= query.view(1, length, 1)))
        )

    return (
        allowed
        & in_bounds.unsqueeze(0)
        & valid_tokens[:, :, None]
        & key_valid
    )


def validate_visibility_inputs(
    mode: CognitiveMode,
    valid_tokens: torch.Tensor,
    source_lengths: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Validate the shared orientation inputs without materializing a mask."""
    if not isinstance(mode, CognitiveMode):
        raise ValueError("mode must be a CognitiveMode")
    if not isinstance(valid_tokens, torch.Tensor) or (
        valid_tokens.dtype is not torch.bool or valid_tokens.ndim != 2
    ):
        raise ValueError("valid_tokens must be bool [batch, sequence]")

    batch, length = valid_tokens.shape
    if mode is CognitiveMode.LACHESIS:
        _validate_source_lengths(source_lengths, valid_tokens)
    return batch, length


def _validate_source_lengths(
    source_lengths: torch.Tensor | None, valid_tokens: torch.Tensor
) -> None:
    batch, _ = valid_tokens.shape
    if (
        not isinstance(source_lengths, torch.Tensor)
        or source_lengths.dtype is not torch.int64
        or source_lengths.shape != (batch,)
        or source_lengths.device != valid_tokens.device
    ):
        raise ValueError("source_lengths must be int64 [batch] on valid_tokens.device")
    if torch.any(~valid_tokens[:, :-1] & valid_tokens[:, 1:]):
        raise ValueError("valid_tokens must be right-padded for lachesis source lengths")
    valid_lengths = valid_tokens.sum(dim=1)
    if torch.any(source_lengths < 0) or torch.any(source_lengths > valid_lengths):
        raise ValueError("source_lengths must be within each row's valid token length")
