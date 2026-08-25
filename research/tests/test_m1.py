import torch

from moira.ablation import equal_parameter_ablation_report
from moira.config import ModelConfig
from moira.masks import CognitiveMode
from moira.model import MoiraModel


def test_m1_reference_is_inside_the_30m_to_50m_budget() -> None:
    config = ModelConfig.m1_cpu()
    with torch.device("meta"):
        model = MoiraModel(config)

    assert model.parameter_ledger().total_trainable == 33_032_562
    assert 30_000_000 <= model.parameter_ledger().total_trainable <= 50_000_000


def test_component_ablations_keep_the_full_parameter_surface() -> None:
    config = ModelConfig.m0_cpu()
    model = MoiraModel(config).eval()
    report = equal_parameter_ablation_report(
        model,
        torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
        mode=CognitiveMode.ATROPOS,
    )

    assert [item.ablated for item in report] == [("thread",), ("spool",), ("knot",)]
    assert len({item.trainable_parameters for item in report}) == 1
    assert all(item.logits_finite and item.appraisal_finite for item in report)
