import pytest
import torch
from torch import nn

from moira.ledger import ParameterSubsystem, build_parameter_ledger


class TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Embedding(8, 4)
        self.output = nn.Linear(4, 8, bias=False)
        self.output.weight = self.token.weight


def test_ledger_counts_tied_storage_once() -> None:
    ledger = build_parameter_ledger(
        TiedModel(),
        prefix_rules=(
            ("token", ParameterSubsystem.TOKEN_IO),
            ("output", ParameterSubsystem.TOKEN_IO),
        ),
    )
    assert ledger.total_trainable == 32
    assert ledger.entries[0].aliases == ("output.weight", "token.weight")


def test_ledger_uses_the_longest_matching_prefix() -> None:
    model = nn.Module()
    model.token = nn.Linear(2, 2, bias=False)
    ledger = build_parameter_ledger(
        model,
        prefix_rules=(
            ("token", ParameterSubsystem.THREAD),
            ("token.weight", ParameterSubsystem.TOKEN_IO),
        ),
    )
    assert ledger.entries[0].subsystem is ParameterSubsystem.TOKEN_IO


def test_ledger_rejects_ambiguous_equal_length_prefix_rules() -> None:
    model = nn.Module()
    model.token = nn.Linear(2, 2, bias=False)
    with pytest.raises(ValueError, match="ambiguous"):
        build_parameter_ledger(
            model,
            prefix_rules=(
                ("token", ParameterSubsystem.TOKEN_IO),
                ("token", ParameterSubsystem.THREAD),
            ),
        )


def test_ledger_allows_duplicate_equal_length_prefix_rules_with_the_same_subsystem() -> None:
    model = nn.Module()
    model.token = nn.Linear(2, 2, bias=False)
    ledger = build_parameter_ledger(
        model,
        prefix_rules=(
            ("token", ParameterSubsystem.TOKEN_IO),
            ("token", ParameterSubsystem.TOKEN_IO),
        ),
    )
    assert ledger.entries[0].subsystem is ParameterSubsystem.TOKEN_IO


def test_ledger_rejects_unclassified_parameters() -> None:
    with pytest.raises(ValueError, match="unclassified"):
        build_parameter_ledger(nn.Linear(2, 2), prefix_rules=())


def test_ledger_rejects_tied_aliases_with_different_subsystems() -> None:
    with pytest.raises(ValueError, match="disagree"):
        build_parameter_ledger(
            TiedModel(),
            prefix_rules=(
                ("token", ParameterSubsystem.TOKEN_IO),
                ("output", ParameterSubsystem.THREAD),
            ),
        )


def test_ledger_rejects_an_unexpected_parameter_total() -> None:
    with pytest.raises(ValueError, match="expected"):
        build_parameter_ledger(
            TiedModel(),
            prefix_rules=(("token", ParameterSubsystem.TOKEN_IO), ("output", ParameterSubsystem.TOKEN_IO)),
            expected_trainable_parameters=31,
        )


def test_ledger_detects_model_parameter_drift() -> None:
    model = TiedModel()
    ledger = build_parameter_ledger(
        model,
        prefix_rules=(
            ("token", ParameterSubsystem.TOKEN_IO),
            ("output", ParameterSubsystem.TOKEN_IO),
        ),
    )
    model.token.weight = nn.Parameter(torch.zeros(8, 4))
    with pytest.raises(ValueError, match="drift"):
        ledger.assert_matches()
