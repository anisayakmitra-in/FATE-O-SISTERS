from collections.abc import Iterator
from pathlib import Path
import shutil
import uuid

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from moira.config import ModelConfig
from moira.masks import CognitiveMode
from moira.model import MoiraModel
from moira.qat import (
    fake_quantize_weight_ste,
    load_qat_checkpoint,
    prepare_model_for_qat,
    promote_model_qat,
    qat_model_state,
    save_qat_checkpoint,
    verify_qat_checkpoint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def qat_root() -> Iterator[Path]:
    root = REPOSITORY_ROOT / "research" / "runs" / f"test-qat-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _tiny_model() -> MoiraModel:
    return MoiraModel(
        ModelConfig(
            vocab_size=16,
            d_model=8,
            n_layers=1,
            n_heads=2,
            ffn_dim=16,
            thread_radius=1,
            spool_rank=2,
            knot_slots=2,
            capsule_width=4,
            max_seq_len=6,
        )
    )


def test_int3_forward_has_seven_symmetric_levels_and_ste_gradient() -> None:
    master = torch.linspace(-1.0, 1.0, 101, requires_grad=True)

    forward = fake_quantize_weight_ste(master, "int3")
    forward.sum().backward()

    assert torch.unique(forward).numel() == 7
    assert torch.equal(master.grad, torch.ones_like(master))
    assert not torch.equal(forward, master.detach())


def test_whole_moira_weight_surface_promotes_while_retaining_tied_master() -> None:
    model = _tiny_model()

    prepared = prepare_model_for_qat(model, "int3")
    token_master = model.token_embedding.parametrizations.weight.original
    output_master = model.output_projection.parametrizations.weight.original
    master_identity = id(token_master)
    output = model(
        torch.randint(0, 16, (2, 6)),
        mode=CognitiveMode.ATROPOS,
    )
    output.logits.square().mean().backward()
    promoted = promote_model_qat(model, "int6")

    assert prepared.logical_bits == 3
    assert prepared.quantized_modules > 2
    assert token_master is output_master
    assert token_master.grad is not None
    assert torch.isfinite(token_master.grad).all()
    assert id(model.token_embedding.parametrizations.weight.original) == master_identity
    assert promoted["source_bits"] == "int3"
    assert promoted["target_bits"] == "int6"
    assert qat_model_state(model)["logical_bits"] == "int6"


def test_qat_rejects_logical_levels_beyond_real_fp64_resolution() -> None:
    with pytest.raises(ValueError, match="52 logical bits"):
        fake_quantize_weight_ste(torch.ones(2), "int81")


def test_qat_promotion_cannot_skip_an_intermediate_precision_level() -> None:
    model = nn.Linear(2, 2)
    prepare_model_for_qat(model, "int3")

    with pytest.raises(ValueError, match="exactly one"):
        promote_model_qat(model, "int9")


def test_qat_checkpoint_retains_master_and_precision_history(qat_root: Path) -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.GELU(), nn.Linear(3, 2))
    prepare_model_for_qat(model, "int3")
    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    promote_model_qat(model, "int6")
    checkpoint = qat_root / "qat.pt"

    saved = save_qat_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        global_step=1,
        precision_history=("int3", "int6"),
    )
    verified = verify_qat_checkpoint(saved)
    payload = torch.load(saved, map_location="cpu", weights_only=True)
    restored_model = nn.Sequential(nn.Linear(4, 3), nn.GELU(), nn.Linear(3, 2))
    prepare_model_for_qat(restored_model, "int3")
    restored_optimizer = AdamW(restored_model.parameters(), lr=1e-3)
    restored = load_qat_checkpoint(
        saved,
        model=restored_model,
        optimizer=restored_optimizer,
    )

    assert verified["logical_bits"] == "int6"
    assert verified["precision_history"] == ["int3", "int6"]
    assert verified["master_weights_retained"] is True
    assert restored["model_restored"] is True
    assert restored["optimizer_restored"] is True
    assert qat_model_state(restored_model)["logical_bits"] == "int6"
    assert restored_optimizer.state
    assert any(
        name.endswith("parametrizations.weight.original")
        for name in payload["model_state"]
    )
