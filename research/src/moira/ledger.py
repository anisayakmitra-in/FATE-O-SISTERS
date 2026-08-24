"""Trainable-parameter accounting that preserves tied-weight aliases."""

from dataclasses import dataclass
from enum import Enum

from torch import nn


class ParameterSubsystem(str, Enum):
    TOKEN_IO = "token_io"
    THREAD = "thread"
    SPOOL = "spool"
    KNOT = "knot"
    CAPSULE = "capsule"


@dataclass(frozen=True, slots=True)
class ParameterLedgerEntry:
    aliases: tuple[str, ...]
    subsystem: ParameterSubsystem
    trainable_parameters: int


@dataclass(frozen=True, slots=True)
class ParameterLedger:
    entries: tuple[ParameterLedgerEntry, ...]
    total_trainable: int
    _model: nn.Module

    def assert_matches(self) -> None:
        observed = _unique_trainable_parameter_count(self._model)
        if observed != self.total_trainable:
            raise ValueError(
                "parameter ledger drift: "
                f"expected {self.total_trainable}, observed {observed}"
            )


def build_parameter_ledger(
    model: nn.Module,
    *,
    prefix_rules: tuple[tuple[str, ParameterSubsystem], ...],
    expected_trainable_parameters: int | None = None,
) -> ParameterLedger:
    grouped: dict[int, list[tuple[str, nn.Parameter]]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if parameter.requires_grad:
            grouped.setdefault(id(parameter), []).append((name, parameter))

    entries = []
    for aliases_and_parameter in grouped.values():
        aliases = tuple(sorted(name for name, _ in aliases_and_parameter))
        subsystems = {_classify(name, prefix_rules) for name in aliases}
        if None in subsystems:
            raise ValueError(f"unclassified parameter aliases: {aliases}")
        if len(subsystems) != 1:
            raise ValueError(f"tied parameter alias classifications disagree: {aliases}")
        parameter = aliases_and_parameter[0][1]
        entries.append(
            ParameterLedgerEntry(
                aliases=aliases,
                subsystem=subsystems.pop(),
                trainable_parameters=parameter.numel(),
            )
        )

    ordered_entries = tuple(sorted(entries, key=lambda entry: entry.aliases))
    observed = sum(entry.trainable_parameters for entry in ordered_entries)
    if (
        expected_trainable_parameters is not None
        and observed != expected_trainable_parameters
    ):
        raise ValueError(
            "unexpected trainable parameter count: "
            f"expected {expected_trainable_parameters}, observed {observed}"
        )
    return ParameterLedger(ordered_entries, observed, model)


def _classify(
    name: str,
    prefix_rules: tuple[tuple[str, ParameterSubsystem], ...],
) -> ParameterSubsystem | None:
    matches = [
        (prefix, subsystem)
        for prefix, subsystem in prefix_rules
        if name == prefix or name.startswith(f"{prefix}.")
    ]
    if not matches:
        return None
    longest_prefix = max(len(prefix) for prefix, _ in matches)
    subsystems = {
        subsystem
        for prefix, subsystem in matches
        if len(prefix) == longest_prefix
    }
    if len(subsystems) != 1:
        raise ValueError(f"ambiguous parameter classification for {name}")
    return subsystems.pop()


def _unique_trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in {
            id(parameter): parameter
            for _, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        }.values()
    )
