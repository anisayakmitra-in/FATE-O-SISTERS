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
from .masks import (
    CognitiveMode,
    build_local_visibility_mask,
    validate_visibility_inputs,
)
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

        key_indices, in_bounds = _local_window_indices(
            length, self.radius, hidden.device
        )
        local_visibility = visibility.gather(
            dim=2,
            index=key_indices.unsqueeze(0).expand(batch, -1, -1),
        ) & in_bounds.unsqueeze(0)
        return self._forward_local(hidden, local_visibility)

    def forward_local(self, hidden: Tensor, visibility: Tensor) -> Tensor:
        """Run Thread from pre-windowed visibility without a full ``L × L`` mask."""
        batch, length = _validate_hidden("hidden", hidden, self.d_model, self)
        window_width = 2 * self.radius + 1
        if (
            not isinstance(visibility, Tensor)
            or visibility.dtype is not torch.bool
            or visibility.shape != (batch, length, window_width)
            or visibility.device != hidden.device
        ):
            raise ValueError(
                "visibility must be bool [batch, sequence, local_window] "
                "on hidden.device"
            )
        _, in_bounds = _local_window_indices(length, self.radius, hidden.device)
        return self._forward_local(hidden, visibility & in_bounds.unsqueeze(0))

    def _forward_local(self, hidden: Tensor, visibility: Tensor) -> Tensor:
        batch, length, _ = hidden.shape

        queries = self._split_heads(self.query_projection(hidden))
        keys = self._split_heads(self.key_projection(hidden))
        values = self._split_heads(self.value_projection(hidden))
        queries = _apply_rope(queries)
        keys = _apply_rope(keys)
        key_indices, _ = _local_window_indices(length, self.radius, hidden.device)
        local_keys = keys[:, :, key_indices]
        local_values = values[:, :, key_indices]
        scores = torch.einsum("bhld,bhlwd->bhlw", queries, local_keys) / math.sqrt(
            self.head_dim
        )
        weights = _safe_masked_softmax(scores, visibility[:, None])
        attended = torch.einsum("bhlw,bhlwd->bhld", weights, local_values)
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
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.knot_slots = knot_slots
        self.residual_dropout = nn.Dropout(dropout)
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
        thread_visibility: Tensor,
        valid_tokens: Tensor,
        *,
        mode: CognitiveMode,
        source_lengths: Tensor | None = None,
        state: SpoolState | None = None,
        ablate: frozenset[str] = frozenset(),
    ) -> tuple[Tensor, SpoolState]:
        if "thread" not in ablate:
            thread_output = self.thread.forward_local(
                self.thread_norm(hidden), thread_visibility
            )
            hidden = hidden + self.residual_dropout(thread_output)
        if "spool" not in ablate:
            spool_output, next_state = self.spool(
                self.spool_norm(hidden), state, valid_tokens=valid_tokens
            )
            hidden = hidden + self.residual_dropout(spool_output)
        else:
            next_state = self._preserve_spool_state(hidden, state)
        if "knot" not in ablate:
            normalized = self.knot_norm(hidden)
            landmarks, type_ids, _, landmark_visibility = self.sequence_landmarks(
                normalized,
                valid_tokens,
                mode=mode,
                source_lengths=source_lengths,
            )
            hidden = hidden + self.residual_dropout(
                self.knot(normalized, landmarks, type_ids, landmark_visibility)
            )
        return hidden, next_state

    def _preserve_spool_state(
        self, hidden: Tensor, state: SpoolState | None
    ) -> SpoolState:
        if state is None:
            return SpoolState(
                hidden.new_zeros((hidden.shape[0], self.spool.rank, self.spool.d_model))
            )
        return SpoolState(self.spool._validate_state(state, hidden.shape[0], hidden))

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
        validate_visibility_inputs(mode, valid_tokens, source_lengths)

        capacity = min(self.knot_slots, length)
        landmarks = hidden.new_zeros((batch, capacity, hidden.shape[-1]))
        end_positions = torch.full(
            (batch, capacity), -1, dtype=torch.int64, device=hidden.device
        )
        valid_lengths = valid_tokens.sum(dim=1)
        for batch_index in range(batch):
            valid_length = int(valid_lengths[batch_index].item())
            landmark_count = min(capacity, valid_length)
            source_length = None
            if mode is CognitiveMode.LACHESIS:
                assert source_lengths is not None
                source_length = int(source_lengths[batch_index].item())
            spans = _landmark_spans(
                valid_length,
                landmark_count,
                source_length=source_length,
            )
            for landmark_index, (start, end) in enumerate(spans):
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
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.pulse = Pulse(config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
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
        visibility_mode: CognitiveMode | None = None,
        valid_tokens: Tensor | None = None,
        source_lengths: Tensor | None = None,
        state: MoiraState | None = None,
        ablate: frozenset[str] | None = None,
    ) -> MoiraOutput:
        batch, length = self._validate_tokens(tokens)
        if not isinstance(mode, CognitiveMode):
            raise ValueError("mode must be a CognitiveMode")
        mode_index = {
            CognitiveMode.CLOTHO: 0,
            CognitiveMode.LACHESIS: 1,
            CognitiveMode.ATROPOS: 2,
        }.get(mode)
        hidden = self.embedding_dropout(
            self.token_embedding(tokens) + self.mode_embedding.weight[mode_index]
        )
        return self._run_hidden(
            hidden,
            mode=mode,
            visibility_mode=visibility_mode,
            valid_tokens=valid_tokens,
            source_lengths=source_lengths,
            state=state,
            ablate=ablate,
        )

    def forward_with_prefix(
        self,
        tokens: Tensor,
        prefix_hidden: Tensor,
        *,
        mode: CognitiveMode,
        visibility_mode: CognitiveMode | None = None,
        valid_tokens: Tensor | None = None,
        source_lengths: Tensor | None = None,
        state: MoiraState | None = None,
        ablate: frozenset[str] | None = None,
        tokens_include_prefix: bool = False,
    ) -> MoiraOutput:
        """Run causal decoding with a fixed-capacity capability prefix.

        ``prefix_hidden`` is supplied by the external capability loom.  The
        core model keeps its original parameter surface: prefix slots are
        projected into ``d_model`` before they cross this boundary and share
        the same cognitive visibility mask as token positions. By default
        ``tokens`` contains only text positions. A packed batch that already
        contains dummy prefix slots can set ``tokens_include_prefix=True``;
        those leading token IDs are ignored and only their validity mask is
        checked.
        """
        batch, input_length = self._validate_tokens(tokens)
        prefix_length = self._validate_prefix_hidden(
            prefix_hidden, batch=batch, device=tokens.device
        )
        if type(tokens_include_prefix) is not bool:
            raise ValueError("tokens_include_prefix must be a boolean")
        if tokens_include_prefix:
            if input_length <= prefix_length:
                raise ValueError("prefix must leave at least one token position")
            token_length = input_length - prefix_length
            token_inputs = tokens[:, prefix_length:]
            combined_length = input_length
        else:
            token_length = input_length
            token_inputs = tokens
            combined_length = prefix_length + token_length
            if combined_length > self.config.max_seq_len:
                raise ValueError("prefix and token sequence exceed max_seq_len")
        if valid_tokens is None:
            valid_tokens = torch.ones(
                (batch, combined_length if tokens_include_prefix else token_length),
                dtype=torch.bool,
                device=tokens.device,
            )
        else:
            valid_shapes = {(batch, token_length)}
            if tokens_include_prefix:
                valid_shapes.add((batch, combined_length))
            if (
                not isinstance(valid_tokens, Tensor)
                or valid_tokens.dtype is not torch.bool
                or valid_tokens.shape not in valid_shapes
                or valid_tokens.device != tokens.device
            ):
                raise ValueError(
                    "valid_tokens must be bool [batch, token or combined sequence] "
                    "on tokens.device"
                )
        if tokens_include_prefix and valid_tokens.shape[1] == combined_length:
            if not bool(valid_tokens[:, :prefix_length].all()):
                raise ValueError("combined valid_tokens must keep prefix slots valid")
            valid_tokens = valid_tokens[:, prefix_length:]
        if bool(torch.any(~valid_tokens[:, :-1] & valid_tokens[:, 1:])):
            raise ValueError("valid_tokens must be right-padded")
        if source_lengths is not None:
            if (
                not isinstance(source_lengths, Tensor)
                or source_lengths.dtype is not torch.int64
                or source_lengths.shape != (batch,)
                or source_lengths.device != tokens.device
            ):
                raise ValueError(
                    "source_lengths must be int64 [batch] on tokens.device"
                )
            if bool(torch.any(source_lengths < 0)) or bool(
                torch.any(source_lengths > valid_tokens.sum(dim=1))
            ):
                raise ValueError("source_lengths must fit the token sequence")
            source_lengths = source_lengths + prefix_length
        if not isinstance(mode, CognitiveMode):
            raise ValueError("mode must be a CognitiveMode")
        mode_index = {
            CognitiveMode.CLOTHO: 0,
            CognitiveMode.LACHESIS: 1,
            CognitiveMode.ATROPOS: 2,
        }.get(mode)
        token_hidden = self.embedding_dropout(
            self.token_embedding(token_inputs) + self.mode_embedding.weight[mode_index]
        )
        prefix_hidden = self.embedding_dropout(
            prefix_hidden + self.mode_embedding.weight[mode_index]
        )
        combined_hidden = torch.cat((prefix_hidden, token_hidden), dim=1)
        combined_valid = torch.cat(
            (
                torch.ones(
                    (batch, prefix_length), dtype=torch.bool, device=tokens.device
                ),
                valid_tokens,
            ),
            dim=1,
        )
        return self._run_hidden(
            combined_hidden,
            mode=mode,
            visibility_mode=visibility_mode,
            valid_tokens=combined_valid,
            source_lengths=source_lengths,
            state=state,
            ablate=ablate,
        )

    def _run_hidden(
        self,
        hidden: Tensor,
        *,
        mode: CognitiveMode,
        visibility_mode: CognitiveMode | None,
        valid_tokens: Tensor | None,
        source_lengths: Tensor | None,
        state: MoiraState | None,
        ablate: frozenset[str] | None,
    ) -> MoiraOutput:
        batch, length = _validate_hidden("hidden", hidden, self.config.d_model, self)
        if length > self.config.max_seq_len:
            raise ValueError("hidden sequence exceeds max_seq_len")
        if not isinstance(mode, CognitiveMode):
            raise ValueError("mode must be a CognitiveMode")
        if visibility_mode is None:
            visibility_mode = mode
        if not isinstance(visibility_mode, CognitiveMode):
            raise ValueError("visibility_mode must be a CognitiveMode")
        if ablate is None:
            ablate = frozenset()
        if not isinstance(ablate, frozenset) or not ablate.issubset(
            {"thread", "spool", "knot"}
        ):
            raise ValueError("ablate must be a subset of thread, spool, and knot")
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
        if bool(torch.any(~valid_tokens[:, :-1] & valid_tokens[:, 1:])):
            raise ValueError("valid_tokens must be right-padded")
        if (
            visibility_mode is not CognitiveMode.LACHESIS
            and source_lengths is not None
        ):
            raise ValueError("source_lengths is only valid for Lachesis")
        thread_visibility = build_local_visibility_mask(
            visibility_mode,
            valid_tokens,
            self.config.thread_radius,
            source_lengths,
        )

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

        next_spools = []
        for block, spool_state in zip(self.blocks, spool_states):
            hidden, next_spool = block(
                hidden,
                thread_visibility,
                valid_tokens,
                mode=visibility_mode,
                source_lengths=source_lengths,
                state=spool_state,
                ablate=ablate,
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

    def _validate_prefix_hidden(
        self, prefix_hidden: object, *, batch: int, device: torch.device
    ) -> int:
        if (
            not isinstance(prefix_hidden, Tensor)
            or not prefix_hidden.is_floating_point()
            or prefix_hidden.ndim != 3
            or prefix_hidden.shape[0] != batch
            or prefix_hidden.shape[1] <= 0
            or prefix_hidden.shape[2] != self.config.d_model
            or prefix_hidden.device != device
            or prefix_hidden.dtype != self.token_embedding.weight.dtype
            or prefix_hidden.device.type == "meta"
            or not bool(torch.isfinite(prefix_hidden).all())
        ):
            raise ValueError(
                "prefix_hidden must be finite floating [batch, prefix, d_model] "
                "matching the model device and dtype"
            )
        return int(prefix_hidden.shape[1])


def _landmark_spans(
    valid_length: int,
    landmark_count: int,
    *,
    source_length: int | None,
) -> tuple[tuple[int, int], ...]:
    """Partition valid tokens without crossing a Lachesis source boundary."""
    if landmark_count == 0:
        return ()

    def uniform(start: int, end: int, count: int) -> tuple[tuple[int, int], ...]:
        width = end - start
        return tuple(
            (
                start + index * width // count,
                start + (index + 1) * width // count,
            )
            for index in range(count)
        )

    if source_length is None or source_length in (0, valid_length):
        return uniform(0, valid_length, landmark_count)
    if landmark_count == 1:
        return ((0, source_length),)

    target_length = valid_length - source_length
    proportional = (
        landmark_count * source_length + valid_length // 2
    ) // valid_length
    minimum_source = max(1, landmark_count - target_length)
    maximum_source = min(source_length, landmark_count - 1)
    source_count = min(max(proportional, minimum_source), maximum_source)
    target_count = landmark_count - source_count
    return uniform(0, source_length, source_count) + uniform(
        source_length, valid_length, target_count
    )


def _local_window_indices(
    length: int, radius: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Return clamped local-key indices and a mask for the in-range slots."""
    positions = torch.arange(length, device=device).view(length, 1)
    offsets = torch.arange(-radius, radius + 1, device=device).view(1, -1)
    keys = positions + offsets
    return (
        keys.clamp(0, max(length - 1, 0)),
        (keys >= 0) & (keys < length),
    )


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
    finite_allowed = allowed & torch.isfinite(scores)
    has_finite_allowed = finite_allowed.any(dim=-1, keepdim=True)
    masked = scores.masked_fill(~finite_allowed, -torch.inf)
    masked = torch.where(has_finite_allowed, masked, 0.0)
    weights = torch.softmax(masked, dim=-1)
    weights = torch.where(finite_allowed & torch.isfinite(weights), weights, 0.0)
    total = weights.sum(dim=-1, keepdim=True)
    return torch.where(
        total > 0,
        weights / total.clamp_min(torch.finfo(scores.dtype).tiny),
        0.0,
    )


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
