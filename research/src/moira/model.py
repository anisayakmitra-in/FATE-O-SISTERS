"""CPU reference implementation of MOIRA's Thread, Spool, and Knot macrocell."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .ledger import (
    ParameterLedger,
    ParameterSubsystem,
    build_parameter_ledger,
)
from .masks import CognitiveMode, build_visibility_mask
from .pulse import Pulse, PulseState


@dataclass(frozen=True, slots=True)
class SpoolState:
    memory: Tensor


@dataclass(frozen=True, slots=True)
class MoiraState:
    spools: tuple[SpoolState, ...]
    pulse: PulseState


@dataclass(frozen=True, slots=True)
class MoiraOutput:
    logits: Tensor
    appraisal: Tensor
    state: MoiraState


class ThreadLocalAttention(nn.Module):
    """Bias-free multi-head attention restricted to an exact local window."""

    def __init__(self, d_model: int, n_heads: int, radius: int) -> None:
        super().__init__()
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be positive")
        if not isinstance(n_heads, int) or n_heads <= 0 or d_model % n_heads:
            raise ValueError("n_heads must be positive and divide d_model")
        if (d_model // n_heads) % 2:
            raise ValueError("Thread head dimension must be even for RoPE")
        if not isinstance(radius, int) or radius <= 0:
            raise ValueError("radius must be positive")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.radius = radius
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden: Tensor, visibility: Tensor) -> Tensor:
        batch, length = _validate_hidden("hidden", hidden, self.d_model, self)
        if (
            not isinstance(visibility, Tensor)
            or visibility.dtype is not torch.bool
            or visibility.shape != (batch, length, length)
            or visibility.device != hidden.device
        ):
            raise ValueError(
                "visibility must be bool [batch, sequence, sequence] on hidden.device"
            )

        queries = self._split_heads(self.query_projection(hidden))
        keys = self._split_heads(self.key_projection(hidden))
        values = self._split_heads(self.value_projection(hidden))
        queries = _apply_rope(queries)
        keys = _apply_rope(keys)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        positions = torch.arange(length, device=hidden.device)
        local = (positions[:, None] - positions[None, :]).abs() <= self.radius
        allowed = visibility[:, None] & local[None, None]
        weights = _safe_masked_softmax(scores, allowed)
        attended = torch.matmul(weights, values)
        combined = attended.transpose(1, 2).reshape(batch, length, self.d_model)
        return self.output_projection(combined)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)


class SpoolRecurrent(nn.Module):
    """Explicit gated delta-memory recurrence with caller-owned state."""

    def __init__(self, d_model: int, rank: int) -> None:
        super().__init__()
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be positive")
        if not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be positive")
        self.d_model = d_model
        self.rank = rank
        self.query_projection = nn.Linear(d_model, rank)
        self.key_projection = nn.Linear(d_model, rank)
        self.value_projection = nn.Linear(d_model, d_model)
        self.decay_projection = nn.Linear(d_model, 1)
        self.write_projection = nn.Linear(d_model, 1)

    def forward(
        self,
        hidden: Tensor,
        state: SpoolState | None = None,
        *,
        valid_tokens: Tensor | None = None,
    ) -> tuple[Tensor, SpoolState]:
        batch, length = _validate_hidden("hidden", hidden, self.d_model, self)
        if state is None:
            memory = hidden.new_zeros((batch, self.rank, self.d_model))
        else:
            memory = self._validate_state(state, batch, hidden)
        if valid_tokens is None:
            valid_tokens = torch.ones(
                (batch, length), dtype=torch.bool, device=hidden.device
            )
        elif (
            not isinstance(valid_tokens, Tensor)
            or valid_tokens.dtype is not torch.bool
            or valid_tokens.shape != (batch, length)
            or valid_tokens.device != hidden.device
        ):
            raise ValueError(
                "valid_tokens must be bool [batch, sequence] on hidden.device"
            )

        queries = F.normalize(self.query_projection(hidden), dim=-1)
        keys = F.normalize(self.key_projection(hidden), dim=-1)
        values = self.value_projection(hidden)
        decays = torch.sigmoid(self.decay_projection(hidden))
        writes = torch.sigmoid(self.write_projection(hidden))
        outputs = []
        for index in range(length):
            key = keys[:, index]
            prediction = torch.einsum("br,brh->bh", key, memory)
            error = values[:, index] - prediction
            candidate = (
                decays[:, index, :, None] * memory
                + writes[:, index, :, None]
                * key[:, :, None]
                * error[:, None, :]
            )
            valid = valid_tokens[:, index]
            memory = torch.where(valid[:, None, None], candidate, memory)
            read = torch.einsum("br,brh->bh", queries[:, index], memory)
            outputs.append(torch.where(valid[:, None], read, 0.0))
        if outputs:
            output = torch.stack(outputs, dim=1)
        else:
            output = hidden.new_empty((batch, 0, self.d_model))
        return output, SpoolState(memory)

    def _validate_state(
        self, state: object, batch: int, hidden: Tensor
    ) -> Tensor:
        if (
            not isinstance(state, SpoolState)
            or not isinstance(state.memory, Tensor)
            or not state.memory.is_floating_point()
            or state.memory.shape != (batch, self.rank, self.d_model)
            or state.memory.device != hidden.device
            or state.memory.dtype != hidden.dtype
            or state.memory.device.type == "meta"
            or not bool(torch.isfinite(state.memory).all())
        ):
            raise ValueError(
                "state must contain finite memory [batch, rank, hidden] "
                "matching hidden device and dtype"
            )
        return state.memory


class KnotLandmarkAttention(nn.Module):
    """Bias-free attention over a bounded bank of explicitly typed landmarks."""

    def __init__(
        self,
        d_model: int,
        max_landmarks: int,
        *,
        num_landmark_types: int = 8,
    ) -> None:
        super().__init__()
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be positive")
        if not isinstance(max_landmarks, int) or max_landmarks <= 0:
            raise ValueError("max_landmarks must be positive")
        if not isinstance(num_landmark_types, int) or num_landmark_types <= 0:
            raise ValueError("num_landmark_types must be positive")
        self.d_model = d_model
        self.max_landmarks = max_landmarks
        self.num_landmark_types = num_landmark_types
        self.type_embedding = nn.Embedding(num_landmark_types, d_model)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        hidden: Tensor,
        landmarks: Tensor,
        landmark_type_ids: Tensor,
        visibility: Tensor,
    ) -> Tensor:
        batch, length = _validate_hidden("hidden", hidden, self.d_model, self)
        if (
            not isinstance(landmarks, Tensor)
            or not landmarks.is_floating_point()
            or landmarks.ndim != 3
            or landmarks.shape[0] != batch
            or landmarks.shape[2] != self.d_model
            or not 0 < landmarks.shape[1] <= self.max_landmarks
            or landmarks.device != hidden.device
            or landmarks.dtype != hidden.dtype
            or landmarks.device.type == "meta"
            or not bool(torch.isfinite(landmarks).all())
        ):
            raise ValueError(
                "landmarks must be finite floating [batch, landmarks, hidden] "
                "within fixed capacity and matching hidden"
            )
        landmark_count = landmarks.shape[1]
        if (
            not isinstance(landmark_type_ids, Tensor)
            or landmark_type_ids.dtype is not torch.int64
            or landmark_type_ids.shape != (batch, landmark_count)
            or landmark_type_ids.device != hidden.device
            or bool(torch.any(landmark_type_ids < 0))
            or bool(torch.any(landmark_type_ids >= self.num_landmark_types))
        ):
            raise ValueError(
                "landmark_type_ids must be in-range int64 [batch, landmarks] "
                "on hidden.device"
            )
        if (
            not isinstance(visibility, Tensor)
            or visibility.dtype is not torch.bool
            or visibility.shape != (batch, length, landmark_count)
            or visibility.device != hidden.device
        ):
            raise ValueError(
                "visibility must be bool [batch, sequence, landmarks] on hidden.device"
            )

        typed_landmarks = landmarks + self.type_embedding(landmark_type_ids)
        queries = self.query_projection(hidden)
        keys = self.key_projection(typed_landmarks)
        values = self.value_projection(typed_landmarks)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(
            self.d_model
        )
        weights = _safe_masked_softmax(scores, visibility)
        return self.output_projection(torch.matmul(weights, values))


class TSKBlock(nn.Module):
    """One native Thread-Spool-Knot macrocell without a Transformer FFN."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        thread_radius: int,
        spool_rank: int,
        knot_slots: int,
        num_landmark_types: int = 8,
    ) -> None:
        super().__init__()
        self.knot_slots = knot_slots
        self.thread_norm = nn.LayerNorm(d_model)
        self.thread = ThreadLocalAttention(d_model, n_heads, thread_radius)
        self.spool_norm = nn.LayerNorm(d_model)
        self.spool = SpoolRecurrent(d_model, spool_rank)
        self.knot_norm = nn.LayerNorm(d_model)
        self.knot = KnotLandmarkAttention(
            d_model,
            knot_slots,
            num_landmark_types=num_landmark_types,
        )

    def forward(
        self,
        hidden: Tensor,
        visibility: Tensor,
        valid_tokens: Tensor,
        *,
        mode: CognitiveMode,
        source_lengths: Tensor | None = None,
        state: SpoolState | None = None,
    ) -> tuple[Tensor, SpoolState]:
        thread_output = self.thread(self.thread_norm(hidden), visibility)
        hidden = hidden + thread_output
        spool_output, next_state = self.spool(
            self.spool_norm(hidden), state, valid_tokens=valid_tokens
        )
        hidden = hidden + spool_output
        normalized = self.knot_norm(hidden)
        landmarks, type_ids, _, landmark_visibility = self.sequence_landmarks(
            normalized,
            valid_tokens,
            mode=mode,
            source_lengths=source_lengths,
        )
        hidden = hidden + self.knot(
            normalized, landmarks, type_ids, landmark_visibility
        )
        return hidden, next_state

    def sequence_landmarks(
        self,
        hidden: Tensor,
        valid_tokens: Tensor,
        *,
        mode: CognitiveMode,
        source_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Build deterministic contiguous means and their query visibility."""
        batch, length = _validate_hidden(
            "hidden", hidden, self.knot.d_model, self
        )
        if (
            not isinstance(valid_tokens, Tensor)
            or valid_tokens.dtype is not torch.bool
            or valid_tokens.shape != (batch, length)
            or valid_tokens.device != hidden.device
        ):
            raise ValueError(
                "valid_tokens must be bool [batch, sequence] on hidden.device"
            )
        if bool(torch.any(~valid_tokens[:, :-1] & valid_tokens[:, 1:])):
            raise ValueError("valid_tokens must be right-padded")
        build_visibility_mask(mode, valid_tokens, source_lengths)

        capacity = min(self.knot_slots, length)
        landmarks = hidden.new_zeros((batch, capacity, hidden.shape[-1]))
        end_positions = torch.full(
            (batch, capacity), -1, dtype=torch.int64, device=hidden.device
        )
        valid_lengths = valid_tokens.sum(dim=1)
        for batch_index in range(batch):
            valid_length = int(valid_lengths[batch_index].item())
            landmark_count = min(capacity, valid_length)
            for landmark_index in range(landmark_count):
                start = landmark_index * valid_length // landmark_count
                end = (landmark_index + 1) * valid_length // landmark_count
                landmarks[batch_index, landmark_index] = hidden[
                    batch_index, start:end
                ].mean(dim=0)
                end_positions[batch_index, landmark_index] = end - 1

        type_ids = torch.zeros(
            (batch, capacity), dtype=torch.int64, device=hidden.device
        )
        landmark_valid = end_positions >= 0
        query_positions = torch.arange(length, device=hidden.device).view(1, length, 1)
        query_valid = valid_tokens[:, :, None]
        if mode is CognitiveMode.CLOTHO:
            landmark_visibility = query_valid & landmark_valid[:, None, :]
        elif mode is CognitiveMode.ATROPOS:
            landmark_visibility = (
                query_valid
                & landmark_valid[:, None, :]
                & (end_positions[:, None, :] <= query_positions)
            )
        else:
            assert source_lengths is not None
            source = source_lengths[:, None, None]
            query_is_source = query_positions < source
            landmark_is_source = end_positions[:, None, :] < source
            completed = end_positions[:, None, :] <= query_positions
            landmark_visibility = (
                query_valid
                & landmark_valid[:, None, :]
                & (
                    (query_is_source & landmark_is_source)
                    | (~query_is_source & (landmark_is_source | completed))
                )
            )
        return landmarks, type_ids, end_positions, landmark_visibility


class MoiraModel(nn.Module):
    """Small CPU-first MOIRA reference with explicit recurrent and affect state."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise ValueError("config must be a ModelConfig")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.mode_embedding = nn.Embedding(len(CognitiveMode), config.d_model)
        self.blocks = nn.ModuleList(
            [
                TSKBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    thread_radius=config.thread_radius,
                    spool_rank=config.spool_rank,
                    knot_slots=config.knot_slots,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.pulse = Pulse(config.d_model)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )
        if config.tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight

    def forward(
        self,
        tokens: Tensor,
        *,
        mode: CognitiveMode,
        valid_tokens: Tensor | None = None,
        source_lengths: Tensor | None = None,
        state: MoiraState | None = None,
    ) -> MoiraOutput:
        batch, length = self._validate_tokens(tokens)
        if not isinstance(mode, CognitiveMode):
            raise ValueError("mode must be a CognitiveMode")
        if valid_tokens is None:
            valid_tokens = torch.ones(
                (batch, length), dtype=torch.bool, device=tokens.device
            )
        elif (
            not isinstance(valid_tokens, Tensor)
            or valid_tokens.dtype is not torch.bool
            or valid_tokens.shape != (batch, length)
            or valid_tokens.device != tokens.device
        ):
            raise ValueError(
                "valid_tokens must be bool [batch, sequence] on tokens.device"
            )
        if bool(torch.any(~valid_tokens[:, :-1] & valid_tokens[:, 1:])):
            raise ValueError("valid_tokens must be right-padded")
        if mode is not CognitiveMode.LACHESIS and source_lengths is not None:
            raise ValueError("source_lengths is only valid for Lachesis")
        visibility = build_visibility_mask(mode, valid_tokens, source_lengths)

        if state is None:
            spool_states: tuple[SpoolState | None, ...] = (None,) * len(self.blocks)
            pulse_state = self.pulse.initial_state(
                batch,
                device=self.token_embedding.weight.device,
                dtype=self.token_embedding.weight.dtype,
            )
        else:
            if (
                not isinstance(state, MoiraState)
                or not isinstance(state.spools, tuple)
                or len(state.spools) != len(self.blocks)
                or not isinstance(state.pulse, PulseState)
            ):
                raise ValueError(
                    "state must contain one SpoolState per block and one PulseState"
                )
            spool_states = state.spools
            pulse_state = state.pulse

        mode_index = {
            CognitiveMode.CLOTHO: 0,
            CognitiveMode.LACHESIS: 1,
            CognitiveMode.ATROPOS: 2,
        }[mode]
        hidden = self.token_embedding(tokens) + self.mode_embedding.weight[mode_index]
        next_spools = []
        for block, spool_state in zip(self.blocks, spool_states):
            hidden, next_spool = block(
                hidden,
                visibility,
                valid_tokens,
                mode=mode,
                source_lengths=source_lengths,
                state=spool_state,
            )
            next_spools.append(next_spool)

        valid_weight = valid_tokens.to(dtype=hidden.dtype).unsqueeze(-1)
        summary = (hidden * valid_weight).sum(dim=1) / valid_weight.sum(
            dim=1
        ).clamp_min(1.0)
        appraisal = self.pulse.appraise(summary)
        next_pulse = self.pulse.step(
            pulse_state,
            appraisal,
            update_mask=valid_tokens.any(dim=1),
        )
        hidden = self.pulse.modulate(hidden, pulse_state)
        logits = self.output_projection(self.final_norm(hidden))
        logits = torch.where(valid_tokens[:, :, None], logits, 0.0)
        return MoiraOutput(
            logits=logits,
            appraisal=appraisal,
            state=MoiraState(tuple(next_spools), next_pulse),
        )

    def parameter_ledger(self) -> ParameterLedger:
        rules: list[tuple[str, ParameterSubsystem]] = [
            ("token_embedding", ParameterSubsystem.TOKEN_IO),
            ("mode_embedding", ParameterSubsystem.TOKEN_IO),
            ("pulse", ParameterSubsystem.PULSE),
            ("final_norm", ParameterSubsystem.TOKEN_IO),
            ("output_projection", ParameterSubsystem.TOKEN_IO),
        ]
        for layer_index in range(len(self.blocks)):
            prefix = f"blocks.{layer_index}"
            rules.extend(
                [
                    (f"{prefix}.thread_norm", ParameterSubsystem.THREAD),
                    (f"{prefix}.thread", ParameterSubsystem.THREAD),
                    (f"{prefix}.spool_norm", ParameterSubsystem.SPOOL),
                    (f"{prefix}.spool", ParameterSubsystem.SPOOL),
                    (f"{prefix}.knot_norm", ParameterSubsystem.KNOT),
                    (f"{prefix}.knot", ParameterSubsystem.KNOT),
                ]
            )
        return build_parameter_ledger(
            self,
            prefix_rules=tuple(rules),
            expected_trainable_parameters=self.config.expected_trainable_parameters,
        )

    def _validate_tokens(self, tokens: object) -> tuple[int, int]:
        parameter = self.token_embedding.weight
        if (
            not isinstance(tokens, Tensor)
            or tokens.dtype is not torch.int64
            or tokens.ndim != 2
            or tokens.device != parameter.device
            or tokens.device.type == "meta"
        ):
            raise ValueError(
                "tokens must be int64 [batch, sequence] on the model device"
            )
        batch, length = tokens.shape
        if batch <= 0 or length <= 0 or length > self.config.max_seq_len:
            raise ValueError(
                "tokens must have positive batch and sequence within max_seq_len"
            )
        if bool(torch.any(tokens < 0)) or bool(
            torch.any(tokens >= self.config.vocab_size)
        ):
            raise ValueError("tokens must contain in-range vocabulary IDs")
        return batch, length


def _apply_rope(tensor: Tensor) -> Tensor:
    _, _, length, head_dim = tensor.shape
    positions = torch.arange(length, device=tensor.device, dtype=tensor.dtype)
    frequencies = torch.arange(
        0, head_dim, 2, device=tensor.device, dtype=tensor.dtype
    )
    frequencies = 1.0 / (10000.0 ** (frequencies / head_dim))
    angles = positions[:, None] * frequencies[None, :]
    cosine = angles.cos()[None, None]
    sine = angles.sin()[None, None]
    even = tensor[..., 0::2]
    odd = tensor[..., 1::2]
    return torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)


def _safe_masked_softmax(scores: Tensor, allowed: Tensor) -> Tensor:
    masked = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=-1)
    return torch.where(allowed.any(dim=-1, keepdim=True), weights, 0.0)


def _validate_hidden(
    name: str, tensor: object, d_model: int, module: nn.Module
) -> tuple[int, int]:
    parameter = next(module.parameters())
    if (
        not isinstance(tensor, Tensor)
        or not tensor.is_floating_point()
        or tensor.ndim != 3
        or tensor.shape[2] != d_model
        or tensor.device != parameter.device
        or tensor.dtype != parameter.dtype
        or tensor.device.type == "meta"
        or not bool(torch.isfinite(tensor).all())
    ):
        raise ValueError(
            f"{name} must be finite floating [batch, sequence, {d_model}] "
            "matching module device and dtype"
        )
    return tensor.shape[0], tensor.shape[1]
