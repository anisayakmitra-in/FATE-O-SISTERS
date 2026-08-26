from dataclasses import replace

import pytest
import moira
import torch

from moira.config import (
    DataConfig,
    ModelConfig,
    OptimizationConfig,
    ReproducibilityConfig,
    RunConfig,
    TrainingConfig,
)
from moira.model import MoiraModel


def test_m0_config_is_valid_and_head_aligned() -> None:
    config = ModelConfig.m0_cpu()
    assert config.d_model % config.n_heads == 0
    assert config.head_dim == config.d_model // config.n_heads


def test_m2_config_parameter_contract() -> None:
    config = ModelConfig.m2_gpu()
    with torch.device("meta"):
        model = MoiraModel(config)

    assert model.parameter_ledger().total_trainable == 153_349_122
    assert config.expected_trainable_parameters == 153_349_122


def test_package_exposes_the_task_one_configuration_interfaces() -> None:
    assert moira.ModelConfig is ModelConfig
    assert moira.TrainingConfig is TrainingConfig
    assert moira.ParameterSubsystem.__name__ == "ParameterSubsystem"
    assert moira.ParameterLedger.__name__ == "ParameterLedger"
    assert callable(moira.build_parameter_ledger)


def test_config_rejects_misaligned_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(
            vocab_size=32,
            d_model=15,
            n_layers=1,
            n_heads=2,
            ffn_dim=32,
            thread_radius=2,
            spool_rank=4,
            knot_slots=2,
            capsule_width=8,
            max_seq_len=16,
        )


def test_config_rejects_an_unknown_causal_knot_policy() -> None:
    with pytest.raises(ValueError, match="causal_knot_policy"):
        replace(
            ModelConfig.m0_cpu(),
            causal_knot_policy="dynamic-final-length-v1",
        )


def test_training_m0_has_the_approved_cpu_defaults() -> None:
    config = TrainingConfig.m0_cpu()
    assert config.schema_version == 1
    assert config.data.kind == "synthetic-indexed-v1"
    assert config.data.seed == 13031997
    assert config.data.train_split == 0
    assert config.data.eval_split == 1
    assert config.data.batch_size == 8
    assert config.optimization.optimizer == "adamw"
    assert config.optimization.learning_rate == 3e-4
    assert config.optimization.weight_decay == 0.01
    assert config.optimization.gradient_accumulation_steps == 1
    assert config.optimization.schedule == "cosine-with-warmup"
    assert config.optimization.warmup_steps == 2
    assert config.optimization.schedule_total_steps == 20
    assert config.optimization.precision == "fp32"
    assert config.optimization.clip_grad_norm == 1.0
    assert config.reproducibility.seed == 20260824
    assert config.reproducibility.deterministic == "strict"
    assert config.run.log_every_steps == 1
    assert config.run.checkpoint_every_steps == 2
    assert config.run.keep_last_checkpoints == 2


def test_training_config_rejects_invalid_cross_field_values() -> None:
    with pytest.raises(ValueError, match="different"):
        TrainingConfig(
            schema_version=1,
            model=ModelConfig.m0_cpu(),
            data=DataConfig(train_split=0, eval_split=0),
            optimization=OptimizationConfig(),
            reproducibility=ReproducibilityConfig(),
            run=RunConfig(),
        )


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: DataConfig(kind="unknown"), "kind"),
        (lambda: DataConfig(batch_size=0), "positive"),
        (lambda: OptimizationConfig(optimizer="sgd"), "optimizer"),
        (lambda: OptimizationConfig(learning_rate=0.0), "positive"),
        (lambda: OptimizationConfig(warmup_steps=20), "less"),
        (lambda: ReproducibilityConfig(deterministic="best-effort"), "deterministic"),
        (lambda: RunConfig(keep_last_checkpoints=0), "positive"),
    ],
)
def test_training_sections_reject_unsupported_or_invalid_values(factory, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_training_config_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        TrainingConfig(
            schema_version=2,
            model=ModelConfig.m0_cpu(),
            data=DataConfig(),
            optimization=OptimizationConfig(),
            reproducibility=ReproducibilityConfig(),
            run=RunConfig(),
        )
