import pytest
import torch

from moira.pulse import FAST_AXES, SLOW_AXES, Pulse, PulseState


def test_pulse_exposes_the_fixed_affect_axes_and_zero_initial_state() -> None:
    """Catches changing the fixed 12/6 affect contract or initial bounds."""
    pulse = Pulse(d_model=4)

    state = pulse.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)

    assert FAST_AXES == (
        "valence", "arousal", "control", "certainty", "novelty", "urgency",
        "conflict", "risk", "care_relevance", "curiosity", "frustration", "completion",
    )
    assert SLOW_AXES == (
        "coherence_pressure", "completion_pressure", "exploration_pressure",
        "safety_pressure", "social_care_pressure", "resource_reserve",
    )
    assert state.fast.shape == (2, 12)
    assert state.slow.shape == (2, 6)
    assert state.fast.dtype is torch.float32
    assert state.fast.device.type == "cpu"
    assert torch.equal(state.fast, torch.zeros((2, 12)))
    assert torch.equal(state.slow, torch.zeros((2, 6)))


@pytest.mark.parametrize(
    ("device", "dtype"),
    [(torch.device("cpu"), torch.float64), (torch.device("meta"), torch.float32)],
)
def test_initial_state_rejects_device_or_dtype_outside_pulse_parameters(
    device: torch.device, dtype: torch.dtype
) -> None:
    """Catches state allocation that later cannot enter Pulse's learned layers."""
    with pytest.raises(ValueError, match="Pulse device and dtype"):
        Pulse(d_model=4).initial_state(2, device=device, dtype=dtype)


def test_appraise_returns_bounded_fast_values_on_the_summary_device_and_dtype() -> None:
    """Catches an appraisal path that leaks outside fast-state bounds."""
    pulse = Pulse(d_model=4)
    summary = torch.full((3, 4), 1000.0)

    appraisal = pulse.appraise(summary)

    assert appraisal.shape == (3, 12)
    assert appraisal.dtype == summary.dtype
    assert appraisal.device == summary.device
    assert torch.all((appraisal >= -1) & (appraisal <= 1))


@pytest.mark.parametrize(
    "summary",
    [
        torch.ones((2, 4), dtype=torch.int64),
        torch.ones((2, 3)),
        torch.ones((2, 4, 1)),
        torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]),
    ],
)
def test_appraise_rejects_malformed_summaries(summary: torch.Tensor) -> None:
    """Catches invalid summary shape, dtype, or finite-value input."""
    with pytest.raises(ValueError, match="summary"):
        Pulse(d_model=4).appraise(summary)


def test_step_remains_bounded_and_masked_rows_stay_bit_identical() -> None:
    """Catches unbounded recurrent state or an update leaking into a masked row."""
    pulse = Pulse(d_model=4)
    state = pulse.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    appraisal = torch.full((2, 12), 1000.0)
    mask = torch.tensor([True, False])

    for _ in range(100):
        next_state = pulse.step(state, appraisal, update_mask=mask)
        assert torch.equal(next_state.fast[1], state.fast[1])
        assert torch.equal(next_state.slow[1], state.slow[1])
        state = next_state

    assert torch.all((state.fast >= -1) & (state.fast <= 1))
    assert torch.all((state.slow >= 0) & (state.slow <= 1))


@pytest.mark.parametrize(
    "state",
    [
        PulseState(torch.zeros((2, 11)), torch.zeros((2, 6))),
        PulseState(torch.zeros((2, 12)), torch.zeros((2, 5))),
        PulseState(torch.full((2, 12), float("nan")), torch.zeros((2, 6))),
        PulseState(torch.full((2, 12), 1.1), torch.zeros((2, 6))),
        PulseState(torch.zeros((2, 12)), torch.full((2, 6), -0.1)),
    ],
)
def test_step_rejects_malformed_or_out_of_bounds_state(state: PulseState) -> None:
    """Catches state that cannot be a bounded unified Pulse state."""
    with pytest.raises(ValueError, match="state"):
        Pulse(d_model=4).step(state, torch.zeros((2, 12)), update_mask=torch.ones(2, dtype=torch.bool))


@pytest.mark.parametrize(
    "appraisal",
    [
        torch.zeros((2, 11)),
        torch.zeros((2, 12), dtype=torch.int64),
        torch.full((2, 12), float("inf")),
    ],
)
def test_step_rejects_malformed_appraisal(appraisal: torch.Tensor) -> None:
    """Catches an invalid field appraisal entering the recurrent update."""
    pulse = Pulse(d_model=4)
    state = pulse.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)

    with pytest.raises(ValueError, match="appraisal"):
        pulse.step(state, appraisal, update_mask=torch.ones(2, dtype=torch.bool))


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(2, dtype=torch.int64),
        torch.ones((2, 1), dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
    ],
)
def test_step_rejects_malformed_update_mask(mask: torch.Tensor) -> None:
    """Catches a mask that cannot select whole batch rows exactly."""
    pulse = Pulse(d_model=4)
    state = pulse.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)

    with pytest.raises(ValueError, match="update_mask"):
        pulse.step(state, torch.zeros((2, 12)), update_mask=mask)


def test_modulate_preserves_hidden_shape_device_and_dtype() -> None:
    """Catches modulation that changes the model sequence contract."""
    pulse = Pulse(d_model=4)
    state = pulse.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    hidden = torch.randn((2, 3, 4))

    actual = pulse.modulate(hidden, state)

    assert actual.shape == hidden.shape
    assert actual.dtype == hidden.dtype
    assert actual.device == hidden.device


@pytest.mark.parametrize(
    "hidden",
    [
        torch.ones((2, 3, 4), dtype=torch.int64),
        torch.ones((2, 4)),
        torch.ones((2, 3, 5)),
        torch.full((2, 3, 4), float("nan")),
    ],
)
def test_modulate_rejects_malformed_hidden(hidden: torch.Tensor) -> None:
    """Catches a modulation input outside the [batch, sequence, hidden] contract."""
    pulse = Pulse(d_model=4)
    state = pulse.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)

    with pytest.raises(ValueError, match="hidden"):
        pulse.modulate(hidden, state)


def test_pulse_rejects_invalid_decay_configuration() -> None:
    """Catches a decay that would no longer be a stable interpolation."""
    with pytest.raises(ValueError, match="decay"):
        Pulse(d_model=4, fast_decay=1.1)
    with pytest.raises(ValueError, match="decay"):
        Pulse(d_model=4, slow_decay=-0.1)


def test_pulse_path_backpropagates_through_every_parameter_without_mutating_inputs() -> None:
    """Catches a disconnected Pulse parameter or in-place mutation of caller state."""
    torch.manual_seed(19)
    pulse = Pulse(d_model=4)
    state = pulse.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)
    summary = torch.randn((2, 4), requires_grad=True)
    hidden = torch.randn((2, 3, 4), requires_grad=True)
    mask = torch.tensor([True, True])
    state_before = (state.fast.clone(), state.slow.clone())
    input_before = (summary.detach().clone(), hidden.detach().clone())
    versions = (state.fast._version, state.slow._version, summary._version, hidden._version)

    appraisal = pulse.appraise(summary)
    next_state = pulse.step(state, appraisal, update_mask=mask)
    output = pulse.modulate(hidden, next_state)
    output.square().mean().backward()

    for name, parameter in pulse.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    assert torch.equal(state.fast, state_before[0])
    assert torch.equal(state.slow, state_before[1])
    assert torch.equal(summary.detach(), input_before[0])
    assert torch.equal(hidden.detach(), input_before[1])
    assert (state.fast._version, state.slow._version, summary._version, hidden._version) == versions
