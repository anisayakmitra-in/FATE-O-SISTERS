import pytest
import torch

from moira.scale_probe import scale_probe_config
from moira.model import MoiraModel


@pytest.mark.parametrize(
    ("milestone", "parameters"),
    (("m1", 33_032_562), ("m2", 153_349_122), ("m3", 485_735_322)),
)
def test_scale_probe_milestones_have_fixed_parameter_surfaces(
    milestone: str, parameters: int
) -> None:
    with torch.device("meta"):
        model = MoiraModel(scale_probe_config(milestone))
    assert model.parameter_ledger().total_trainable == parameters


def test_scale_probe_rejects_unknown_milestone() -> None:
    with pytest.raises(ValueError, match="milestone"):
        scale_probe_config("m4")
