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
    if not isinstance(mode, CognitiveMode):
        raise ValueError("mode must be a CognitiveMode")
    if not isinstance(valid_tokens, torch.Tensor) or (
        valid_tokens.dtype is not torch.bool or valid_tokens.ndim != 2
    ):
        raise ValueError("valid_tokens must be bool [batch, sequence]")

    batch, length = valid_tokens.shape
    device = valid_tokens.device
    query = torch.arange(length, device=device).view(1, length, 1)
    key = torch.arange(length, device=device).view(1, 1, length)

    if mode is CognitiveMode.CLOTHO:
        allowed = torch.ones((batch, length, length), dtype=torch.bool, device=device)
    elif mode is CognitiveMode.ATROPOS:
        allowed = (key <= query).expand(batch, -1, -1)
    else:
        _validate_source_lengths(source_lengths, valid_tokens)
        source = source_lengths.view(batch, 1, 1)
        query_is_source = query < source
        key_is_source = key < source
        allowed = (query_is_source & key_is_source) | (
            ~query_is_source & (key_is_source | (key <= query))
        )

    return allowed & valid_tokens[:, :, None] & valid_tokens[:, None, :]


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
    valid_lengths = valid_tokens.sum(dim=1)
    if torch.any(source_lengths < 0) or torch.any(source_lengths > valid_lengths):
        raise ValueError("source_lengths must be within each row's valid token length")
