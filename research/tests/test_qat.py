from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
import shutil
import uuid

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from moira.config import ModelConfig, TrainingConfig
from moira.masks import CognitiveMode
from moira.model import MoiraModel
from moira.qat import (
    fake_quantize_weight_ste,
    load_qat_checkpoint,
    prepare_model_for_qat,
    promote_model_qat,
    qat_config_digest,
    qat_model_state,
    run_moira_qat_smoke,
    save_qat_checkpoint,
    verify_qat_checkpoint,
)
from moira.synthetic import SYNTHETIC_CURSOR_KIND, DataCursor


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


def test_int2_forward_has_three_symmetric_levels_and_ste_gradient() -> None:
    master = torch.linspace(-1.0, 1.0, 101, requires_grad=True)

    forward = fake_quantize_weight_ste(master, "int2")
    forward.sum().backward()

    assert torch.unique(forward).numel() == 3
    assert torch.equal(master.grad, torch.ones_like(master))
    assert not torch.equal(forward, master.detach())


def test_whole_moira_weight_surface_promotes_while_retaining_tied_master() -> None:
    model = _tiny_model()

    prepared = prepare_model_for_qat(model, "int2")
    token_master = model.token_embedding.parametrizations.weight.original
    output_master = model.output_projection.parametrizations.weight.original
    master_identity = id(token_master)
    output = model(
        torch.randint(0, 16, (2, 6)),
        mode=CognitiveMode.ATROPOS,
    )
    output.logits.square().mean().backward()
    promoted = promote_model_qat(model, "int4")

    assert prepared.logical_bits == 2
    assert prepared.quantized_modules > 2
    assert token_master is output_master
    assert token_master.grad is not None
    assert torch.isfinite(token_master.grad).all()
    assert id(model.token_embedding.parametrizations.weight.original) == master_identity
    assert promoted["source_bits"] == "int2"
    assert promoted["target_bits"] == "int4"
    assert qat_model_state(model)["logical_bits"] == "int4"


def test_qat_rejects_nonstandard_integer_precision() -> None:
    with pytest.raises(ValueError, match="int2, int4, or int8"):
        fake_quantize_weight_ste(torch.ones(2), "int3")


def test_qat_promotion_cannot_skip_an_intermediate_precision_level() -> None:
    model = nn.Linear(2, 2)
    prepare_model_for_qat(model, "int2")

    with pytest.raises(ValueError, match="exactly one"):
        promote_model_qat(model, "int8")


def test_qat_checkpoint_retains_master_and_precision_history(qat_root: Path) -> None:
    model = nn.Sequential(nn.Linear(4, 3), nn.GELU(), nn.Linear(3, 2))
    prepare_model_for_qat(model, "int2")
    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    promote_model_qat(model, "int4")
    checkpoint = qat_root / "qat.pt"
    config_digest = qat_config_digest(TrainingConfig.m0_cpu())
    data_cursor = DataCursor(SYNTHETIC_CURSOR_KIND, 1, 1, 32)

    saved = save_qat_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        global_step=1,
        precision_history=("int2", "int4"),
        config_digest=config_digest,
        data_cursor=data_cursor,
    )
    verified = verify_qat_checkpoint(saved)
    payload = torch.load(saved, map_location="cpu", weights_only=True)
    restored_model = nn.Sequential(nn.Linear(4, 3), nn.GELU(), nn.Linear(3, 2))
    prepare_model_for_qat(restored_model, "int2")
    restored_optimizer = AdamW(restored_model.parameters(), lr=1e-3)
    restored = load_qat_checkpoint(
        saved,
        model=restored_model,
        optimizer=restored_optimizer,
        expected_config_digest=config_digest,
    )

    assert verified["logical_bits"] == "int4"
    assert verified["precision_history"] == ["int2", "int4"]
    assert verified["master_weights_retained"] is True
    assert verified["config_digest"] == config_digest
    assert verified["data_cursor"]["samples_seen"] == 1
    assert restored["model_restored"] is True
    assert restored["optimizer_restored"] is True
    assert qat_model_state(restored_model)["logical_bits"] == "int4"
    assert restored_optimizer.state
    assert any(
        name.endswith("parametrizations.weight.original")
        for name in payload["model_state"]
    )


def test_int8_qat_smoke_can_train_as_one_stage(qat_root: Path) -> None:
    checkpoint = qat_root / "int8.pt"

    result = run_moira_qat_smoke(
        TrainingConfig.m0_cpu(),
        initial_bits="int8",
        promote_to="int8",
        steps_per_stage=2,
        checkpoint=checkpoint,
    )
    verified = verify_qat_checkpoint(checkpoint)

    assert result["precision_history"] == ["int8"]
    assert result["promotion"] is None
    assert result["global_step"] == 2
    assert len(result["losses"]) == 2
    assert result["trainable_parameters"] == 24_478
    assert result["peak_cuda_memory_bytes"] == 0
    assert result["peak_cuda_reserved_bytes"] == 0
    assert result["final_qat_state"]["logical_bits"] == "int8"
    assert result["all_losses_finite"] is True
    assert verified["precision_history"] == ["int8"]
    assert verified["global_step"] == 2


def test_int8_qat_smoke_resumes_weights_optimizer_and_cursor(qat_root: Path) -> None:
    checkpoint = qat_root / "resume-int8.pt"
    base = TrainingConfig.m0_cpu()
    config = replace(base, model=replace(base.model, dropout=0.0))

    first = run_moira_qat_smoke(
        config,
        initial_bits="int8",
        promote_to="int8",
        steps_per_stage=1,
        checkpoint=checkpoint,
    )
    before = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="use resume"):
        run_moira_qat_smoke(
            config,
            initial_bits="int8",
            promote_to="int8",
            checkpoint=checkpoint,
        )

    resumed = run_moira_qat_smoke(
        config,
        initial_bits="int8",
        promote_to="int8",
        steps_per_stage=2,
        checkpoint=checkpoint,
        resume=True,
    )
    verified = verify_qat_checkpoint(checkpoint)

    assert first["global_step"] == 1
    assert resumed["resumed"] is True
    assert resumed["starting_global_step"] == 1
    assert resumed["global_step"] == 3
    assert resumed["exact_resume_guaranteed"] is False
    assert verified["global_step"] == 3
    assert checkpoint.read_bytes() != before


def test_int8_qat_resume_rejects_uncheckpointed_dropout(qat_root: Path) -> None:
    checkpoint = qat_root / "dropout-int8.pt"
    base = TrainingConfig.m0_cpu()
    config = replace(base, model=replace(base.model, dropout=0.1))
    run_moira_qat_smoke(
        config,
        initial_bits="int8",
        promote_to="int8",
        checkpoint=checkpoint,
    )

    with pytest.raises(ValueError, match="dropout 0"):
        run_moira_qat_smoke(
            config,
            initial_bits="int8",
            promote_to="int8",
            checkpoint=checkpoint,
            resume=True,
        )


def test_int8_qat_resume_rejects_shape_compatible_data_change(
    qat_root: Path,
) -> None:
    checkpoint = qat_root / "data-change-int8.pt"
    base = TrainingConfig.m0_cpu()
    config = replace(base, model=replace(base.model, dropout=0.0))
    run_moira_qat_smoke(
        config,
        initial_bits="int8",
        promote_to="int8",
        checkpoint=checkpoint,
    )
    changed = replace(config, data=replace(config.data, seed=config.data.seed + 1))

    with pytest.raises(ValueError, match="config digest does not match"):
        run_moira_qat_smoke(
            changed,
            initial_bits="int8",
            promote_to="int8",
            checkpoint=checkpoint,
            resume=True,
        )


@pytest.mark.parametrize("bits", ("int2", "int4"))
def test_equal_low_bit_qat_smoke_is_not_a_promotion(bits: str) -> None:
    with pytest.raises(ValueError, match="single-stage int8"):
        run_moira_qat_smoke(
            TrainingConfig.m0_cpu(),
            initial_bits=bits,
            promote_to=bits,
        )
