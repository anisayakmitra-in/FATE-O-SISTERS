"""Equal-parameter component ablations for the M1 model surface."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .masks import CognitiveMode
from .model import MoiraModel


ABLATION_COMPONENTS = ("thread", "spool", "knot")


@dataclass(frozen=True, slots=True)
class AblationResult:
    """One forward result with the full parameter surface retained."""

    ablated: tuple[str, ...]
    trainable_parameters: int
    logits_finite: bool
    appraisal_finite: bool


def equal_parameter_ablation_report(
    model: MoiraModel,
    tokens: Tensor,
    *,
    mode: CognitiveMode,
) -> tuple[AblationResult, ...]:
    """Run each leave-one-component-out variant without deleting parameters."""
    if not isinstance(model, MoiraModel):
        raise ValueError("model must be a MoiraModel")
    if not isinstance(mode, CognitiveMode):
        raise ValueError("mode must be a CognitiveMode")
    parameter_count = model.parameter_ledger().total_trainable
    results: list[AblationResult] = []
    model.eval()
    with torch.no_grad():
        for component in ABLATION_COMPONENTS:
            output = model(
                tokens,
                mode=mode,
                ablate=frozenset({component}),
            )
            results.append(
                AblationResult(
                    ablated=(component,),
                    trainable_parameters=parameter_count,
                    logits_finite=bool(torch.isfinite(output.logits).all()),
                    appraisal_finite=bool(torch.isfinite(output.appraisal).all()),
                )
            )
    return tuple(results)


def assert_equal_parameter_surface(model: MoiraModel) -> int:
    """Return the ledger count after checking all ablation variants are equal."""
    count = model.parameter_ledger().total_trainable
    report = tuple(
        _parameter_count_for_ablation(model, component)
        for component in ABLATION_COMPONENTS
    )
    if report != (count, count, count):
        raise ValueError("component ablations changed the trainable parameter surface")
    return count


def _parameter_count_for_ablation(model: MoiraModel, component: str) -> int:
    if component not in ABLATION_COMPONENTS:
        raise ValueError("unknown ablation component")
    return model.parameter_ledger().total_trainable
