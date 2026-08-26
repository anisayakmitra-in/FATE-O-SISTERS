import copy
import math
from dataclasses import replace

import pytest
import torch

from moira.config import ModelConfig
from moira.ledger import ParameterSubsystem
from moira.masks import CognitiveMode
from moira.pulse import PulseState
from moira.model import (
    KnotLandmarkAttention,
    MoiraModel,
    MoiraState,
    SpoolRecurrent,
    SpoolState,
    TSKBlock,
    ThreadLocalAttention,
    _apply_rope,
    _safe_masked_softmax,
)


def _tiny_config(
    *, tie_embeddings: bool = True, dropout: float = 0.0
) -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        d_model=8,
        n_layers=2,
        n_heads=2,
        ffn_dim=16,
        thread_radius=1,
        spool_rank=3,
        knot_slots=3,
        capsule_width=4,
        max_seq_len=8,
        dropout=dropout,
        tie_embeddings=tie_embeddings,
    )


def _full_thread_reference(
    layer: ThreadLocalAttention, hidden: torch.Tensor, visibility: torch.Tensor
) -> torch.Tensor:
    """The original full-score formulation, retained only as a kernel oracle."""
    batch, length, _ = hidden.shape
    queries = _apply_rope(layer._split_heads(layer.query_projection(hidden)))
    keys = _apply_rope(layer._split_heads(layer.key_projection(hidden)))
    values = layer._split_heads(layer.value_projection(hidden))
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(layer.head_dim)
    positions = torch.arange(length, device=hidden.device)
    local = (positions[:, None] - positions[None, :]).abs() <= layer.radius
    weights = _safe_masked_softmax(scores, visibility[:, None] & local[None, None])
    attended = torch.matmul(weights, values)
    combined = attended.transpose(1, 2).reshape(batch, length, layer.d_model)
    return layer.output_projection(combined)


def _local_visibility_from_full(visibility: torch.Tensor, radius: int) -> torch.Tensor:
    """Extract the fixed-width local slice used by Thread's compact entry point."""
    batch, length, _ = visibility.shape
    query = torch.arange(length, device=visibility.device).view(length, 1)
    key = query + torch.arange(-radius, radius + 1, device=visibility.device).view(
        1, -1
    )
    in_bounds = (key >= 0) & (key < length)
    return visibility.gather(
        2, key.clamp(0, length - 1).unsqueeze(0).expand(batch, -1, -1)
    ) & in_bounds.unsqueeze(0)


def test_selected_orientation_can_use_causal_source_visibility() -> None:
    model = MoiraModel(_tiny_config())
    tokens = torch.tensor([[1, 4, 5, 3, 6, 7]], dtype=torch.int64)

    output = model(
        tokens,
        mode=CognitiveMode.CLOTHO,
        visibility_mode=CognitiveMode.LACHESIS,
        valid_tokens=torch.ones_like(tokens, dtype=torch.bool),
        source_lengths=torch.tensor([4], dtype=torch.int64),
    )

    assert output.logits.shape == (1, 6, 16)
    assert torch.isfinite(output.logits).all()


def test_thread_intersects_visibility_with_locality_and_zeros_blocked_rows() -> None:
    """Catches non-local keys, ignored visible neighbors, and unsafe empty softmax rows."""
    layer = ThreadLocalAttention(d_model=4, n_heads=2, radius=1)
    with torch.no_grad():
        for projection in (
            layer.query_projection,
            layer.key_projection,
            layer.value_projection,
            layer.output_projection,
        ):
            projection.weight.copy_(torch.eye(4))

    x = torch.tensor(
        [[[1.0, 0.0, 0.0, 1.0],
          [0.0, 1.0, 1.0, 0.0],
          [1.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 1.0]]]
    )
    visible = torch.ones((1, 4, 4), dtype=torch.bool)
    baseline = layer(x, visible)

    outside = x.clone()
    outside[:, 3] += 100.0
    assert torch.allclose(layer(outside, visible)[:, 1], baseline[:, 1])

    neighbor = x.clone()
    neighbor[:, 2] += 2.0
    assert not torch.allclose(layer(neighbor, visible)[:, 1], baseline[:, 1])

    blocked = visible.clone()
    blocked[:, 1] = False
    output = layer(x, blocked)
    assert torch.isfinite(output).all()
    assert torch.equal(output[:, 1], torch.zeros_like(output[:, 1]))


def test_thread_overflow_cannot_select_a_blocked_future_value() -> None:
    """Catches masked sentinels outranking an allowed score that overflowed."""
    layer = ThreadLocalAttention(d_model=2, n_heads=1, radius=1)
    with torch.no_grad():
        for projection in (
            layer.query_projection,
            layer.key_projection,
            layer.value_projection,
            layer.output_projection,
        ):
            projection.weight.zero_()
        layer.query_projection.weight[0, 0] = 2e20
        layer.key_projection.weight[0, 0] = -2e20
        layer.value_projection.weight.copy_(torch.eye(2))
        layer.output_projection.weight.copy_(torch.eye(2))

    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    visibility = torch.zeros((1, 2, 2), dtype=torch.bool)
    visibility[:, 0, 0] = True

    output = layer(hidden, visibility)

    assert torch.isfinite(output).all()
    assert torch.equal(output[:, 0], torch.zeros_like(output[:, 0]))


def test_windowed_thread_kernel_matches_full_reference_outputs_and_gradients() -> None:
    """Catches a compact kernel changing Thread's old full-score semantics."""
    torch.manual_seed(41)
    compact = ThreadLocalAttention(d_model=8, n_heads=2, radius=2)
    reference = copy.deepcopy(compact)
    full_visibility = torch.rand((2, 7, 7)) > 0.35
    full_visibility[:, 4] = False
    compact_hidden = torch.randn(2, 7, 8, requires_grad=True)
    reference_hidden = compact_hidden.detach().clone().requires_grad_(True)
    local_visibility = _local_visibility_from_full(full_visibility, compact.radius)

    actual = compact.forward_local(compact_hidden, local_visibility)
    expected = _full_thread_reference(reference, reference_hidden, full_visibility)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(
        compact_hidden.grad, reference_hidden.grad, atol=1e-6, rtol=1e-5
    )
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in compact.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            reference_parameters[name].grad,
            atol=1e-6,
            rtol=1e-5,
        )

    with torch.no_grad():
        public_output = compact(compact_hidden.detach(), full_visibility)
    torch.testing.assert_close(public_output, expected.detach(), atol=1e-6, rtol=1e-5)


def test_model_builds_one_compact_thread_mask_per_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches reintroducing a quadratic model-level Thread visibility allocation."""
    import moira.model as model_module

    recorded_shapes: list[tuple[int, ...]] = []
    original = model_module.build_local_visibility_mask

    def record_local_mask(*args: object, **kwargs: object) -> torch.Tensor:
        result = original(*args, **kwargs)
        recorded_shapes.append(tuple(result.shape))
        return result

    monkeypatch.setattr(model_module, "build_local_visibility_mask", record_local_mask)
    model = MoiraModel(_tiny_config()).eval()

    model(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), mode=CognitiveMode.ATROPOS)

    assert recorded_shapes == [(1, 8, 3)]


def test_spool_full_sequence_matches_chunked_execution() -> None:
    """Catches recurrence that depends on arbitrary forward-call chunk boundaries."""
    torch.manual_seed(7)
    layer = SpoolRecurrent(d_model=8, rank=3)
    x = torch.randn(2, 9, 8)

    full, full_state = layer(x)
    first, state = layer(x[:, :4])
    second, state = layer(x[:, 4:7], state)
    third, state = layer(x[:, 7:], state)

    assert torch.allclose(
        full, torch.cat((first, second, third), dim=1), atol=1e-6
    )
    assert torch.allclose(full_state.memory, state.memory, atol=1e-6)


def test_spool_invalid_tokens_emit_zero_and_do_not_change_state() -> None:
    """Catches padding writes to memory or nonzero padding outputs."""
    torch.manual_seed(17)
    layer = SpoolRecurrent(d_model=6, rank=2)
    initial = SpoolState(torch.randn(2, 2, 6))
    initial_snapshot = initial.memory.clone()
    x = torch.randn(2, 3, 6)
    valid = torch.tensor([[False, False, False], [True, False, True]])

    output, state = layer(x, initial, valid_tokens=valid)
    expected_second, expected_state = layer(
        x[1:2, [0, 2]],
        SpoolState(initial.memory[1:2]),
    )

    assert torch.equal(output[0], torch.zeros_like(output[0]))
    assert torch.equal(state.memory[0], initial.memory[0])
    assert torch.equal(output[1, 1], torch.zeros_like(output[1, 1]))
    assert torch.allclose(output[1, [0, 2]], expected_second[0], atol=1e-6)
    assert torch.allclose(state.memory[1], expected_state.memory[0], atol=1e-6)
    assert torch.equal(initial.memory, initial_snapshot)


def test_spool_rejects_malformed_state() -> None:
    """Catches accepting memory that cannot represent [batch, rank, hidden]."""
    layer = SpoolRecurrent(d_model=6, rank=2)

    with pytest.raises(ValueError, match="state"):
        layer(torch.randn(2, 3, 6), SpoolState(torch.zeros(2, 3, 6)))


def test_knot_respects_typed_causal_visibility_and_zeros_blocked_rows() -> None:
    """Catches future-landmark leakage, ignored types, and unsafe empty rows."""
    layer = KnotLandmarkAttention(
        d_model=4, max_landmarks=3, num_landmark_types=3
    )
    with torch.no_grad():
        for projection in (
            layer.query_projection,
            layer.key_projection,
            layer.value_projection,
            layer.output_projection,
        ):
            projection.weight.copy_(torch.eye(4))
        layer.type_embedding.weight.zero_()
        layer.type_embedding.weight[1] = torch.tensor([2.0, 0.0, 0.0, 0.0])

    hidden = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0]]]
    )
    landmarks = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0]]]
    )
    source_end_positions = torch.tensor([[0, 1, 2]])
    query_positions = torch.arange(3).view(1, 3, 1)
    visible = source_end_positions[:, None, :] <= query_positions
    visible[:, 2] = False
    type_ids = torch.zeros((1, 3), dtype=torch.int64)

    baseline = layer(hidden, landmarks, type_ids, visible)
    changed_future = landmarks.clone()
    changed_future[:, 1] += 100.0
    assert torch.allclose(
        layer(hidden, changed_future, type_ids, visible)[:, 0], baseline[:, 0]
    )

    changed_types = type_ids.clone()
    changed_types[:, 0] = 1
    assert not torch.allclose(
        layer(hidden, landmarks, changed_types, visible)[:, 0], baseline[:, 0]
    )
    assert torch.isfinite(baseline).all()
    assert torch.equal(baseline[:, 2], torch.zeros_like(baseline[:, 2]))


def test_knot_overflow_cannot_select_a_blocked_landmark() -> None:
    """Catches non-finite allowed scores exposing a forbidden landmark."""
    layer = KnotLandmarkAttention(d_model=2, max_landmarks=2)
    with torch.no_grad():
        for projection in (
            layer.query_projection,
            layer.key_projection,
            layer.value_projection,
            layer.output_projection,
        ):
            projection.weight.zero_()
        layer.type_embedding.weight.zero_()
        layer.query_projection.weight[0, 0] = 2e20
        layer.key_projection.weight[0, 0] = -2e20
        layer.value_projection.weight.copy_(torch.eye(2))
        layer.output_projection.weight.copy_(torch.eye(2))

    hidden = torch.tensor([[[1.0, 0.0]]])
    landmarks = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    visibility = torch.tensor([[[True, False]]])

    output = layer(
        hidden,
        landmarks,
        torch.zeros((1, 2), dtype=torch.int64),
        visibility,
    )

    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.zeros_like(output))


def test_sequence_landmarks_are_contiguous_means_with_mode_correct_visibility() -> None:
    """Catches nondeterministic pooling, padded values, or a wrong landmark boundary."""
    block = TSKBlock(
        d_model=4,
        n_heads=2,
        thread_radius=1,
        spool_rank=2,
        knot_slots=2,
    )
    hidden = torch.tensor(
        [
            [[1.0] * 4, [2.0] * 4, [3.0] * 4, [4.0] * 4, [5.0] * 4],
            [[10.0] * 4, [20.0] * 4, [999.0] * 4, [999.0] * 4, [999.0] * 4],
        ]
    )
    valid = torch.tensor(
        [[True, True, True, True, True], [True, True, False, False, False]]
    )

    landmarks, type_ids, end_positions, atropos = block.sequence_landmarks(
        hidden, valid, mode=CognitiveMode.ATROPOS
    )
    _, _, _, clotho = block.sequence_landmarks(
        hidden, valid, mode=CognitiveMode.CLOTHO
    )
    _, _, _, lachesis = block.sequence_landmarks(
        hidden,
        valid,
        mode=CognitiveMode.LACHESIS,
        source_lengths=torch.tensor([3, 1]),
    )

    assert torch.equal(
        landmarks,
        torch.tensor(
            [
                [[1.5] * 4, [4.0] * 4],
                [[10.0] * 4, [20.0] * 4],
            ]
        ),
    )
    assert torch.equal(type_ids, torch.zeros((2, 2), dtype=torch.int64))
    assert torch.equal(end_positions, torch.tensor([[1, 4], [0, 1]]))
    assert torch.equal(
        atropos[0],
        torch.tensor(
            [[0, 0], [1, 0], [1, 0], [1, 0], [1, 1]], dtype=torch.bool
        ),
    )
    assert torch.equal(
        clotho[1],
        torch.tensor(
            [[1, 1], [1, 1], [0, 0], [0, 0], [0, 0]], dtype=torch.bool
        ),
    )
    assert torch.equal(
        lachesis[0],
        torch.tensor(
            [[1, 0], [1, 0], [1, 0], [1, 0], [1, 1]], dtype=torch.bool
        ),
    )


def test_lachesis_landmarks_never_pool_across_the_source_boundary() -> None:
    """Catches a landmark that mixes bidirectional source and causal target tokens."""
    block = TSKBlock(
        d_model=4,
        n_heads=2,
        thread_radius=1,
        spool_rank=2,
        knot_slots=3,
    )
    hidden = torch.tensor([[[float(value)] * 4 for value in range(1, 6)]])
    valid = torch.ones((1, 5), dtype=torch.bool)

    landmarks, _, end_positions, visibility = block.sequence_landmarks(
        hidden,
        valid,
        mode=CognitiveMode.LACHESIS,
        source_lengths=torch.tensor([2]),
    )

    assert torch.equal(
        landmarks,
        torch.tensor([[[1.5] * 4, [3.0] * 4, [4.5] * 4]]),
    )
    assert torch.equal(end_positions, torch.tensor([[1, 2, 4]]))
    assert torch.equal(
        visibility[0],
        torch.tensor(
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.bool,
        ),
    )


def test_lachesis_single_landmark_preserves_source_without_crossing_boundary() -> None:
    """Catches forced source/target mixing when Knot capacity is one."""
    block = TSKBlock(
        d_model=4,
        n_heads=2,
        thread_radius=1,
        spool_rank=2,
        knot_slots=1,
    )
    hidden = torch.tensor([[[float(value)] * 4 for value in range(1, 5)]])

    landmarks, _, end_positions, _ = block.sequence_landmarks(
        hidden,
        torch.ones((1, 4), dtype=torch.bool),
        mode=CognitiveMode.LACHESIS,
        source_lengths=torch.tensor([2]),
    )

    assert torch.equal(landmarks, torch.tensor([[[1.5] * 4]]))
    assert torch.equal(end_positions, torch.tensor([[1]]))


@pytest.mark.parametrize(
    ("source_length", "expected_visibility"),
    [
        (
            0,
            torch.tensor(
                [[0, 0], [1, 0], [1, 0], [1, 1]], dtype=torch.bool
            ),
        ),
        (4, torch.ones((4, 2), dtype=torch.bool)),
    ],
)
def test_lachesis_empty_or_full_source_keeps_the_unsplit_layout(
    source_length: int,
    expected_visibility: torch.Tensor,
) -> None:
    """Catches edge sources producing empty spans or the wrong visibility policy."""
    block = TSKBlock(
        d_model=4,
        n_heads=2,
        thread_radius=1,
        spool_rank=2,
        knot_slots=2,
    )
    hidden = torch.tensor([[[float(value)] * 4 for value in range(1, 5)]])

    landmarks, _, end_positions, visibility = block.sequence_landmarks(
        hidden,
        torch.ones((1, 4), dtype=torch.bool),
        mode=CognitiveMode.LACHESIS,
        source_lengths=torch.tensor([source_length]),
    )

    assert torch.equal(
        landmarks, torch.tensor([[[1.5] * 4, [3.5] * 4]])
    )
    assert torch.equal(end_positions, torch.tensor([[1, 3]]))
    assert torch.equal(visibility[0], expected_visibility)


def test_atropos_future_token_cannot_change_prior_logits() -> None:
    """Catches future public-token leakage through any Thread-Spool-Knot path."""
    torch.manual_seed(11)
    model = MoiraModel(_tiny_config()).eval()
    left = torch.tensor([[1, 2, 3, 4, 5]])
    right = torch.tensor([[1, 2, 3, 4, 9]])

    a = model(left, mode=CognitiveMode.ATROPOS).logits
    b = model(right, mode=CognitiveMode.ATROPOS).logits

    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-6)


def test_lachesis_future_target_cannot_change_source_or_prior_target_logits() -> None:
    """Catches target-side future leakage through any Thread-Spool-Knot path."""
    torch.manual_seed(13)
    model = MoiraModel(_tiny_config()).eval()
    left = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    right = torch.tensor([[1, 2, 3, 4, 5, 9], [7, 8, 9, 10, 15, 14]])
    valid = torch.tensor(
        [[True, True, True, True, True, True], [True, True, True, True, True, False]]
    )
    source_lengths = torch.tensor([3, 2])

    a = model(
        left,
        mode=CognitiveMode.LACHESIS,
        valid_tokens=valid,
        source_lengths=source_lengths,
    ).logits
    b = model(
        right,
        mode=CognitiveMode.LACHESIS,
        valid_tokens=valid,
        source_lengths=source_lengths,
    ).logits

    assert torch.allclose(a[0, :5], b[0, :5], atol=1e-6)
    assert torch.allclose(a[1, :4], b[1, :4], atol=1e-6)
    assert torch.equal(a[1, 5], torch.zeros_like(a[1, 5]))
    assert torch.equal(b[1, 5], torch.zeros_like(b[1, 5]))


def test_configured_dropout_is_used_and_seed_replay_is_deterministic() -> None:
    """Catches an accepted dropout setting that is ignored by the model."""
    torch.manual_seed(19)
    model = MoiraModel(_tiny_config(dropout=0.5)).train()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])

    torch.manual_seed(101)
    first = model(tokens, mode=CognitiveMode.ATROPOS).logits
    torch.manual_seed(101)
    replay = model(tokens, mode=CognitiveMode.ATROPOS).logits
    torch.manual_seed(102)
    different_seed = model(tokens, mode=CognitiveMode.ATROPOS).logits

    assert torch.equal(first, replay)
    assert not torch.equal(first, different_seed)

    model.eval()
    torch.manual_seed(201)
    first_eval = model(tokens, mode=CognitiveMode.ATROPOS).logits
    torch.manual_seed(202)
    second_eval = model(tokens, mode=CognitiveMode.ATROPOS).logits
    assert torch.equal(first_eval, second_eval)

    zero_dropout = MoiraModel(_tiny_config(dropout=0.0)).train()
    first_zero = zero_dropout(tokens, mode=CognitiveMode.ATROPOS).logits
    second_zero = zero_dropout(tokens, mode=CognitiveMode.ATROPOS).logits
    assert torch.equal(first_zero, second_zero)


def test_every_trainable_parameter_receives_a_finite_gradient() -> None:
    """Catches any counted parameter disconnected from the representative objective."""
    torch.manual_seed(23)
    model = MoiraModel(_tiny_config())
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        warm = model(tokens, mode=CognitiveMode.ATROPOS).state
    incoming = MoiraState(
        spools=tuple(
            SpoolState(spool.memory.detach().clone()) for spool in warm.spools
        ),
        pulse=PulseState(
            fast=warm.pulse.fast.detach().clone(),
            slow=warm.pulse.slow.detach().clone(),
        ),
    )
    model.zero_grad(set_to_none=True)
    output = model(
        tokens,
        mode=CognitiveMode.ATROPOS,
        state=incoming,
    )

    (
        output.logits.square().mean()
        + output.appraisal.square().mean()
        + output.state.pulse.slow.mean()
    ).backward()

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
    ]
    all_zero = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool(torch.any(parameter.grad != 0))
    ]
    assert missing == []
    assert nonfinite == []
    assert all_zero == []


def test_model_ledger_counts_unique_tied_weights_and_classifies_every_layer() -> None:
    """Catches duplicate tied storage, missing dynamic rules, or Pulse misclassification."""
    model = MoiraModel(_tiny_config())

    ledger = model.parameter_ledger()
    unique_count = sum(
        parameter.numel()
        for parameter in {id(item): item for item in model.parameters()}.values()
        if parameter.requires_grad
    )

    assert ledger.total_trainable == unique_count
    assert any(
        entry.aliases == ("output_projection.weight", "token_embedding.weight")
        for entry in ledger.entries
    )
    assert all(
        entry.subsystem is ParameterSubsystem.PULSE
        for entry in ledger.entries
        if entry.aliases[0].startswith("pulse.")
    )
    for layer_index in range(model.config.n_layers):
        assert any(
            alias.startswith(f"blocks.{layer_index}.")
            for entry in ledger.entries
            for alias in entry.aliases
        )
    ledger.assert_matches()


@pytest.mark.parametrize("tie_embeddings", [True, False])
def test_model_ledger_matches_tied_and_untied_output_heads(
    tie_embeddings: bool,
) -> None:
    """Catches double-counting tied storage or omitting an untied output head."""
    model = MoiraModel(_tiny_config(tie_embeddings=tie_embeddings))

    ledger = model.parameter_ledger()
    unique_count = sum(
        parameter.numel()
        for parameter in {id(item): item for item in model.parameters()}.values()
        if parameter.requires_grad
    )

    assert ledger.total_trainable == unique_count
    assert (
        model.output_projection.weight is model.token_embedding.weight
    ) is tie_embeddings


def test_model_ledger_enforces_the_configured_expected_count() -> None:
    """Catches model-level expected-count validation being bypassed."""
    config = replace(
        _tiny_config(),
        expected_trainable_parameters=1,
    )

    with pytest.raises(ValueError, match="expected"):
        MoiraModel(config).parameter_ledger()


def test_model_state_is_explicit_single_pulse_and_forward_is_deterministic() -> None:
    """Catches hidden cross-call state, per-layer Pulse state, or stochastic CPU output."""
    torch.manual_seed(29)
    model = MoiraModel(_tiny_config()).eval()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])

    first = model(tokens, mode=CognitiveMode.CLOTHO)
    second = model(tokens, mode=CognitiveMode.CLOTHO)

    assert torch.equal(first.logits, second.logits)
    assert len(first.state.spools) == model.config.n_layers
    assert isinstance(first.state.pulse, PulseState)
    assert not hasattr(first.state, "pulses")


def test_model_forward_does_not_mutate_any_caller_owned_state_tensor() -> None:
    """Catches in-place Spool or Pulse updates crossing an execution boundary."""
    torch.manual_seed(30)
    model = MoiraModel(_tiny_config()).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    incoming = model(tokens, mode=CognitiveMode.ATROPOS).state
    spool_snapshots = tuple(spool.memory.clone() for spool in incoming.spools)
    fast_snapshot = incoming.pulse.fast.clone()
    slow_snapshot = incoming.pulse.slow.clone()

    model(tokens, mode=CognitiveMode.ATROPOS, state=incoming)

    for spool, snapshot in zip(incoming.spools, spool_snapshots):
        assert torch.equal(spool.memory, snapshot)
    assert torch.equal(incoming.pulse.fast, fast_snapshot)
    assert torch.equal(incoming.pulse.slow, slow_snapshot)


def test_padding_is_zero_and_cannot_change_valid_outputs_or_state() -> None:
    """Catches padded token values leaking into logits, summary, Spool, or Pulse."""
    torch.manual_seed(31)
    model = MoiraModel(_tiny_config()).eval()
    valid = torch.tensor([[True, True, False, False]])
    left = torch.tensor([[1, 2, 3, 4]])
    right = torch.tensor([[1, 2, 14, 15]])

    a = model(left, mode=CognitiveMode.ATROPOS, valid_tokens=valid)
    b = model(right, mode=CognitiveMode.ATROPOS, valid_tokens=valid)

    assert torch.equal(a.logits[:, 2:], torch.zeros_like(a.logits[:, 2:]))
    assert torch.allclose(a.logits[:, :2], b.logits[:, :2], atol=1e-6)
    assert torch.allclose(a.appraisal, b.appraisal, atol=1e-6)
    assert torch.allclose(a.state.pulse.fast, b.state.pulse.fast, atol=1e-6)
    assert torch.allclose(a.state.pulse.slow, b.state.pulse.slow, atol=1e-6)
    for a_spool, b_spool in zip(a.state.spools, b.state.spools):
        assert torch.allclose(a_spool.memory, b_spool.memory, atol=1e-6)


def test_model_all_masked_rows_are_finite_zero_and_preserve_zero_state() -> None:
    """Catches NaNs or state writes when a batch row contains no valid tokens."""
    model = MoiraModel(_tiny_config()).eval()
    valid = torch.zeros((1, 3), dtype=torch.bool)

    output = model(
        torch.tensor([[1, 2, 3]]),
        mode=CognitiveMode.ATROPOS,
        valid_tokens=valid,
    )

    assert torch.isfinite(output.logits).all()
    assert torch.equal(output.logits, torch.zeros_like(output.logits))
    assert torch.equal(output.state.pulse.fast, torch.zeros_like(output.state.pulse.fast))
    assert torch.equal(output.state.pulse.slow, torch.zeros_like(output.state.pulse.slow))


@pytest.mark.parametrize(
    "tokens",
    [
        torch.ones((1, 2), dtype=torch.float32),
        torch.ones((1, 2, 1), dtype=torch.int64),
        torch.empty((1, 0), dtype=torch.int64),
        torch.ones((1, 9), dtype=torch.int64),
        torch.tensor([[-1, 1]], dtype=torch.int64),
        torch.tensor([[1, 16]], dtype=torch.int64),
        torch.ones((1, 2), dtype=torch.int64, device="meta"),
    ],
)
def test_model_rejects_malformed_token_inputs(tokens: torch.Tensor) -> None:
    """Catches malformed shape, dtype, length, device, or vocabulary range."""
    with pytest.raises(ValueError, match="tokens"):
        MoiraModel(_tiny_config())(tokens, mode=CognitiveMode.ATROPOS)


def test_model_rejects_non_right_padding_wrong_mode_and_state_cardinality() -> None:
    """Catches ambiguous padding, visibility bypass, or incomplete recurrent state."""
    model = MoiraModel(_tiny_config())
    tokens = torch.tensor([[1, 2, 3]])
    pulse = model.pulse.initial_state(
        1, device=torch.device("cpu"), dtype=torch.float32
    )

    with pytest.raises(ValueError, match="right-padded"):
        model(
            tokens,
            mode=CognitiveMode.ATROPOS,
            valid_tokens=torch.tensor([[True, False, True]]),
        )
    with pytest.raises(ValueError, match="mode"):
        model(tokens, mode="atropos")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="state"):
        model(
            tokens,
            mode=CognitiveMode.ATROPOS,
            state=MoiraState(spools=(), pulse=pulse),
        )
