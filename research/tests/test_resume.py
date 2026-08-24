import random
import shutil
import tomllib
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import moira.train as train_module
from moira.config import (
    DataConfig,
    ModelConfig,
    OptimizationConfig,
    ReproducibilityConfig,
    RunConfig,
    TrainingConfig,
)
from moira.model import MoiraModel
from moira.synthetic import SyntheticStream
from moira.train import evaluate_smoke, train_until


@pytest.fixture
def run_root() -> Iterator[Path]:
    path = Path("research/runs") / f"task5-resume-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _config() -> TrainingConfig:
    return TrainingConfig(
        schema_version=1,
        model=ModelConfig(
            vocab_size=16,
            d_model=8,
            n_layers=1,
            n_heads=2,
            ffn_dim=16,
            thread_radius=1,
            spool_rank=2,
            knot_slots=2,
            capsule_width=4,
            max_seq_len=5,
            dropout=0.25,
        ),
        data=DataConfig(seed=17, train_split=0, eval_split=1, batch_size=2),
        optimization=OptimizationConfig(
            learning_rate=2e-3,
            weight_decay=0.01,
            gradient_accumulation_steps=1,
            warmup_steps=2,
            schedule_total_steps=8,
            clip_grad_norm=1.0,
        ),
        reproducibility=ReproducibilityConfig(seed=29, deterministic="strict"),
        run=RunConfig(
            log_every_steps=1,
            checkpoint_every_steps=4,
            keep_last_checkpoints=3,
        ),
    )


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        assert isinstance(right, Sequence)
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def _rng_snapshot() -> tuple[object, tuple[object, ...], torch.Tensor]:
    numpy_state = np.random.get_state()
    return (
        random.getstate(),
        (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        torch.get_rng_state().clone(),
    )


def _assert_rng_equal(
    left: tuple[object, tuple[object, ...], torch.Tensor],
    right: tuple[object, tuple[object, ...], torch.Tensor],
) -> None:
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    assert np.array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


def test_interrupted_resume_is_exact_with_dropout_optimizer_scheduler_and_rng(
    run_root: Path,
) -> None:
    config = _config()
    uninterrupted = train_until(
        config, run_root / "uninterrupted", stop_at_step=8
    )
    uninterrupted_draw = torch.rand(7)
    uninterrupted_python_draw = random.random()
    uninterrupted_numpy_draw = np.random.random(7)

    first_half = train_until(config, run_root / "resumed", stop_at_step=4)
    checkpoint_path = first_half.checkpoint_path
    del first_half
    resumed = train_until(
        config,
        run_root / "resumed",
        stop_at_step=8,
        resume=checkpoint_path,
    )
    resumed_draw = torch.rand(7)
    resumed_python_draw = random.random()
    resumed_numpy_draw = np.random.random(7)

    assert uninterrupted.global_step == resumed.global_step == 8
    assert uninterrupted.data_cursor == resumed.data_cursor
    assert uninterrupted.samples_seen == resumed.samples_seen == 16
    assert uninterrupted.tokens_seen == resumed.tokens_seen == 80
    assert uninterrupted.losses[4:] == resumed.losses
    assert resumed.scaler is None
    assert uninterrupted.scaler is None
    for name, tensor in uninterrupted.model.state_dict().items():
        assert torch.equal(tensor, resumed.model.state_dict()[name])
    _assert_nested_equal(
        uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict()
    )
    _assert_nested_equal(
        uninterrupted.scheduler.state_dict(), resumed.scheduler.state_dict()
    )
    assert torch.equal(uninterrupted_draw, resumed_draw)
    assert uninterrupted_python_draw == resumed_python_draw
    assert np.array_equal(uninterrupted_numpy_draw, resumed_numpy_draw)

    stream = SyntheticStream(
        config.data.seed,
        config.model.vocab_size,
        config.model.max_seq_len,
    )
    uninterrupted_next, uninterrupted_cursor = stream.batch(
        split=config.data.train_split,
        cursor=uninterrupted.data_cursor,
        batch_size=config.data.batch_size,
    )
    resumed_next, resumed_cursor = stream.batch(
        split=config.data.train_split,
        cursor=resumed.data_cursor,
        batch_size=config.data.batch_size,
    )
    assert torch.equal(uninterrupted_next.input_ids, resumed_next.input_ids)
    assert torch.equal(uninterrupted_next.targets, resumed_next.targets)
    assert torch.equal(uninterrupted_next.loss_mask, resumed_next.loss_mask)
    assert uninterrupted_cursor == resumed_cursor


def test_evaluation_uses_fixed_eval_indices_without_training_rng_or_mode_mutation(
    run_root: Path,
) -> None:
    config = _config()
    trained = train_until(config, run_root / "evaluation", stop_at_step=2)
    cursor_before = trained.data_cursor
    rng_before = _rng_snapshot()
    assert trained.model.training

    first = evaluate_smoke(config, trained.model, batches=2)
    second = evaluate_smoke(config, trained.model, batches=2)

    assert first == second
    assert first.batches == 2
    assert first.samples == 4
    assert first.tokens == 20
    assert first.loss > 0.0
    assert trained.data_cursor == cursor_before
    assert trained.model.training
    _assert_rng_equal(rng_before, _rng_snapshot())


def test_evaluation_restores_every_descendant_training_flag() -> None:
    config = _config()
    model = MoiraModel(config.model)
    model.train()
    model.blocks[0].residual_dropout.eval()
    modes_before = tuple(module.training for module in model.modules())

    evaluate_smoke(config, model, batches=1)

    assert tuple(module.training for module in model.modules()) == modes_before


def test_evaluation_rejects_non_finite_loss_and_restores_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    model = MoiraModel(config.model)
    model.train()
    model.blocks[0].residual_dropout.eval()
    modes_before = tuple(module.training for module in model.modules())

    monkeypatch.setattr(
        train_module,
        "_masked_next_token_loss",
        lambda *args, **kwargs: torch.tensor(float("nan")),
    )
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_smoke(config, model, batches=1)

    assert tuple(module.training for module in model.modules()) == modes_before


def test_resume_rejects_a_target_behind_the_loaded_step(run_root: Path) -> None:
    config = _config()
    trained = train_until(config, run_root / "behind", stop_at_step=4)

    with pytest.raises(ValueError, match="behind"):
        train_until(
            config,
            run_root / "behind",
            stop_at_step=3,
            resume=trained.checkpoint_path,
        )


def test_operational_stop_always_commits_without_changing_the_semantic_schedule(
    run_root: Path,
) -> None:
    config = _config()
    trained = train_until(config, run_root / "stop", stop_at_step=3)
    payload = torch.load(
        trained.checkpoint_path, map_location="cpu", weights_only=True
    )

    assert trained.global_step == 3
    assert trained.checkpoint_path.name == "step-000000003.pt"
    assert payload["semantic_manifests"]["optimization"][
        "schedule_total_steps"
    ] == 8
    assert "stop_at_step" not in payload["semantic_manifests"]["optimization"]
    package_files = sorted(Path("research/src/moira").glob("*.py"))
    assert set(payload["semantic_manifests"]["source"]) == {
        path.as_posix() for path in package_files
    }
    assert payload["semantic_manifests"]["runtime"]["device"] == "cpu"
    assert payload["semantic_manifests"]["runtime"][
        "deterministic_algorithms"
    ] is True


def test_source_hashes_survive_missing_git_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_git(*args: object, **kwargs: object) -> None:
        raise OSError("git unavailable")

    monkeypatch.setattr(train_module.subprocess, "run", missing_git)
    source = train_module._source_manifest()
    provenance = train_module._git_provenance()

    assert provenance == {"available": False, "revision": None, "dirty": None}
    assert source
    assert all(len(digest) == 64 for digest in source.values())


def test_git_context_changes_do_not_change_strict_source_identity(
    run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = {"revision": "revision-a", "dirty": False}

    def git_result(args: list[str], **kwargs: object) -> object:
        stdout = (
            f"{context['revision']}\n"
            if "rev-parse" in args
            else (" M unrelated\n" if context["dirty"] else "")
        )
        return type("GitResult", (), {"stdout": stdout})()

    monkeypatch.setattr(train_module.subprocess, "run", git_result)
    config = _config()
    first = train_until(config, run_root / "git-context", stop_at_step=1)

    context.update(revision="revision-b", dirty=True)
    resumed = train_until(
        config,
        run_root / "git-context",
        stop_at_step=2,
        resume=first.checkpoint_path,
    )
    payload = torch.load(
        first.checkpoint_path, map_location="cpu", weights_only=True
    )

    assert resumed.global_step == 2
    assert payload["semantic_manifests"]["source"]
    assert all(
        len(digest) == 64
        for digest in payload["semantic_manifests"]["source"].values()
    )
    assert payload["provenance"]["git"] == {
        "available": True,
        "revision": "revision-a",
        "dirty": False,
    }


def test_phase_zero_rejects_gradient_accumulation_and_non_cpu_targets(
    run_root: Path,
) -> None:
    config = _config()
    accumulation = replace(
        config,
        optimization=replace(config.optimization, gradient_accumulation_steps=2),
    )
    with pytest.raises(ValueError, match="accumulation"):
        train_until(accumulation, run_root / "accumulation", stop_at_step=1)
    with pytest.raises(ValueError, match="device"):
        train_until(config, run_root / "device", stop_at_step=1, device="cuda")


def test_m0_smoke_toml_matches_the_task_one_schema_and_parameter_ledger() -> None:
    with Path("research/configs/m0-smoke.toml").open("rb") as handle:
        raw = tomllib.load(handle)
    config = TrainingConfig(
        schema_version=raw["schema_version"],
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        optimization=OptimizationConfig(**raw["optimization"]),
        reproducibility=ReproducibilityConfig(**raw["reproducibility"]),
        run=RunConfig(**raw["run"]),
    )

    assert config.model.dropout > 0.0
    assert config.run.checkpoint_every_steps == 2
    assert config.optimization.schedule_total_steps == 20
    assert MoiraModel(config.model).parameter_ledger().total_trainable == 24478
