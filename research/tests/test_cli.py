from collections.abc import Iterator
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "research" / "configs" / "m0-smoke.toml"


@pytest.fixture
def cli_root() -> Iterator[Path]:
    root = (
        REPOSITORY_ROOT
        / "research"
        / "runs"
        / f"test-cli-{uuid.uuid4().hex}"
    )
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _run_cli(*arguments: object) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_path = str(REPOSITORY_ROOT / "research" / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_path, environment.get("PYTHONPATH")))
    )
    return subprocess.run(
        [sys.executable, "-m", "moira", *(str(value) for value in arguments)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _output_json(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    return json.loads(result.stdout)


@pytest.mark.parametrize("profile", ("consumer", "enterprise"))
def test_deployment_descriptions_are_read_only_and_keep_one_identity(
    profile: str,
) -> None:
    result = _run_cli("deployment", "describe", profile)

    output = _output_json(result)
    assert output["model"] == "MOIRA"
    assert output["profile"] == profile
    assert output["implemented_in_phase_0"] is False
    assert output["profile_changes_model_identity"] is False
    assert output["switch_command_available"] is False


def test_doctor_reports_the_exact_m0_cpu_surface(cli_root: Path) -> None:
    run_dir = cli_root / "doctor-run"

    result = _run_cli(
        "doctor",
        "--config",
        CONFIG_PATH,
        "--run-dir",
        run_dir,
        "--device",
        "cpu",
    )

    output = _output_json(result)
    assert output["ok"] is True
    assert output["phase"] == "M0"
    assert output["device"] == "cpu"
    assert output["trainable_parameters"] == 24_478
    assert output["expected_trainable_parameters"] == 24_478
    assert output["synthetic_data"] is True
    assert output["run_dir_writable"] is True
    assert run_dir.is_dir()


def test_train_verify_evaluate_and_auto_resume_cli(cli_root: Path) -> None:
    run_dir = cli_root / "training-run"
    trained = _run_cli(
        "train",
        "--config",
        CONFIG_PATH,
        "--run-dir",
        run_dir,
        "--device",
        "cpu",
        "--stop-at-step",
        2,
        "--resume",
        "none",
    )

    train_output = _output_json(trained)
    checkpoint_path = Path(str(train_output["checkpoint"]))
    assert train_output["global_step"] == 2
    assert train_output["steps_run"] == 2
    assert checkpoint_path.is_file()

    explicit_verify = _output_json(
        _run_cli(
            "checkpoint",
            "verify",
            "--config",
            CONFIG_PATH,
            "--checkpoint",
            checkpoint_path,
        )
    )
    assert explicit_verify["global_step"] == 2
    assert explicit_verify["checkpoint"] == str(checkpoint_path.resolve())
    assert explicit_verify["samples_seen"] == 16
    assert explicit_verify["tokens_seen"] == 512

    run_verify = _output_json(
        _run_cli(
            "checkpoint",
            "verify",
            "--config",
            CONFIG_PATH,
            "--run-dir",
            run_dir,
        )
    )
    assert run_verify["global_step"] == 2
    assert run_verify["checkpoint"] == str(checkpoint_path.resolve())

    evaluated = _output_json(
        _run_cli(
            "evaluate",
            "--config",
            CONFIG_PATH,
            "--run-dir",
            run_dir,
            "--device",
            "cpu",
            "--batches",
            1,
        )
    )
    assert evaluated["batches"] == 1
    assert evaluated["checkpoint_step"] == 2
    assert evaluated["lineage_id"] == train_output["lineage_id"]
    assert evaluated["loss"] > 0.0
    assert evaluated["samples"] == 8
    assert evaluated["tokens"] == 256

    explicit_evaluated = _output_json(
        _run_cli(
            "evaluate",
            "--config",
            CONFIG_PATH,
            "--checkpoint",
            checkpoint_path,
            "--device",
            "cpu",
            "--batches",
            1,
        )
    )
    assert explicit_evaluated == evaluated

    invalid_evaluation = _run_cli(
        "evaluate",
        "--config",
        CONFIG_PATH,
        "--checkpoint",
        checkpoint_path,
        "--device",
        "cpu",
        "--batches",
        0,
    )
    assert invalid_evaluation.returncode == 2
    assert invalid_evaluation.stdout == ""
    assert "batches must be a positive integer" in invalid_evaluation.stderr

    excessive_evaluation = _run_cli(
        "evaluate",
        "--config",
        CONFIG_PATH,
        "--checkpoint",
        checkpoint_path,
        "--device",
        "cpu",
        "--batches",
        10_001,
    )
    assert excessive_evaluation.returncode == 2
    assert excessive_evaluation.stdout == ""
    assert "batches exceeds the Phase 0 evaluation budget" in (
        excessive_evaluation.stderr
    )

    work_limited_evaluation = _run_cli(
        "evaluate",
        "--config",
        CONFIG_PATH,
        "--checkpoint",
        checkpoint_path,
        "--device",
        "cpu",
        "--batches",
        2_000,
    )
    assert work_limited_evaluation.returncode == 2
    assert work_limited_evaluation.stdout == ""
    assert "evaluation request exceeds the Phase 0 M0 work budget" in (
        work_limited_evaluation.stderr
    )
    assert "Traceback" not in work_limited_evaluation.stderr

    explicit_resume = _output_json(
        _run_cli(
            "train",
            "--config",
            CONFIG_PATH,
            "--run-dir",
            run_dir,
            "--device",
            "cpu",
            "--stop-at-step",
            2,
            "--resume",
            checkpoint_path,
        )
    )
    assert explicit_resume["global_step"] == 2
    assert explicit_resume["steps_run"] == 0
    assert explicit_resume["lineage_id"] == train_output["lineage_id"]
    assert explicit_resume["checkpoint"] == str(checkpoint_path.resolve())

    resumed = _output_json(
        _run_cli(
            "train",
            "--config",
            CONFIG_PATH,
            "--run-dir",
            run_dir,
            "--device",
            "cpu",
            "--stop-at-step",
            3,
            "--resume",
            "auto",
        )
    )
    assert resumed["global_step"] == 3
    assert resumed["steps_run"] == 1
    assert resumed["lineage_id"] == train_output["lineage_id"]


def test_train_requires_an_explicit_resume_choice(cli_root: Path) -> None:
    result = _run_cli(
        "train",
        "--config",
        CONFIG_PATH,
        "--run-dir",
        cli_root / "missing-resume-run",
        "--device",
        "cpu",
        "--stop-at-step",
        1,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "--resume" in result.stderr


@pytest.mark.parametrize(
    "source_arguments",
    (
        (),
        ("--run-dir", "one", "--checkpoint", "two.pt"),
    ),
)
def test_checkpoint_verify_requires_one_source(
    source_arguments: tuple[str, ...],
) -> None:
    result = _run_cli(
        "checkpoint",
        "verify",
        "--config",
        CONFIG_PATH,
        *source_arguments,
    )

    assert result.returncode == 2
    assert result.stdout == ""


def test_phase_zero_cli_rejects_cuda(cli_root: Path) -> None:
    result = _run_cli(
        "doctor",
        "--config",
        CONFIG_PATH,
        "--run-dir",
        cli_root / "cuda-run",
        "--device",
        "cuda",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Phase 0 device must be cpu" in result.stderr


def test_train_rejects_a_step_past_the_schedule(cli_root: Path) -> None:
    run_dir = cli_root / "past-schedule-run"
    result = _run_cli(
        "train",
        "--config",
        CONFIG_PATH,
        "--run-dir",
        run_dir,
        "--device",
        "cpu",
        "--stop-at-step",
        21,
        "--resume",
        "none",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "stop_at_step exceeds the semantic schedule" in result.stderr
    assert not run_dir.exists()


def test_train_rejects_total_work_before_creating_a_run(cli_root: Path) -> None:
    config_path = cli_root / "excessive-work.toml"
    source = CONFIG_PATH.read_text(encoding="utf-8")
    source = source.replace(
        "schedule_total_steps = 20",
        "schedule_total_steps = 100000",
        1,
    )
    config_path.write_text(source, encoding="utf-8")
    run_dir = cli_root / "excessive-work-run"

    result = _run_cli(
        "train",
        "--config",
        config_path,
        "--run-dir",
        run_dir,
        "--device",
        "cpu",
        "--stop-at-step",
        100_000,
        "--resume",
        "none",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "training request exceeds the Phase 0 M0 work budget" in result.stderr
    assert "Traceback" not in result.stderr
    assert not run_dir.exists()


def test_cli_rejects_unknown_config_sections(cli_root: Path) -> None:
    config_path = cli_root / "invalid.toml"
    config_path.write_text(
        CONFIG_PATH.read_text(encoding="utf-8") + "\n[unknown]\nvalue = 1\n",
        encoding="utf-8",
    )

    result = _run_cli(
        "doctor",
        "--config",
        config_path,
        "--run-dir",
        cli_root / "invalid-run",
        "--device",
        "cpu",
    )

    assert result.returncode == 2
    assert "config sections" in result.stderr


@pytest.mark.parametrize(
    ("old", "new", "expected_error"),
    (
        (
            "[model]\n",
            "[model]\nunknown_model_field = 1\n",
            "config section 'model' has invalid fields",
        ),
        (
            "vocab_size = 64",
            'vocab_size = "64"',
            "config field model.vocab_size has the wrong TOML type",
        ),
        (
            "vocab_size = 64",
            "vocab_size = 3",
            "vocab_size must be at least 4",
        ),
        (
            "max_seq_len = 32",
            "max_seq_len = 1",
            "max_seq_len must be at least 2",
        ),
        (
            "learning_rate = 0.0003",
            "learning_rate = inf",
            "config field optimization.learning_rate must be finite",
        ),
        (
            "clip_grad_norm = 1.0",
            "clip_grad_norm = nan",
            "config field optimization.clip_grad_norm must be finite",
        ),
        (
            "batch_size = 8",
            "batch_size = 999999999",
            "batch_size exceeds the Phase 0 M0 resource budget",
        ),
        (
            "schedule_total_steps = 20",
            "schedule_total_steps = 999999999",
            "training schedule exceeds the Phase 0 M0 resource budget",
        ),
        (
            "vocab_size = 64",
            "vocab_size = 1000000",
            "logit tensor exceeds the Phase 0 M0 resource budget",
        ),
        (
            "n_layers = 2",
            "n_layers = 999999999",
            "n_layers exceeds the Phase 0 M0 resource budget",
        ),
        (
            "max_seq_len = 32",
            "max_seq_len = 2048",
            "attention tensor exceeds the Phase 0 M0 resource budget",
        ),
        (
            "keep_last_checkpoints = 2",
            "keep_last_checkpoints = 999999999",
            "checkpoint retention exceeds the Phase 0 M0 resource budget",
        ),
    ),
)
def test_cli_rejects_invalid_nested_config_fields(
    cli_root: Path,
    old: str,
    new: str,
    expected_error: str,
) -> None:
    config_path = cli_root / "invalid-nested.toml"
    source = CONFIG_PATH.read_text(encoding="utf-8")
    assert old in source
    config_path.write_text(source.replace(old, new, 1), encoding="utf-8")
    run_dir = cli_root / "invalid-nested-run"

    result = _run_cli(
        "doctor",
        "--config",
        config_path,
        "--run-dir",
        run_dir,
        "--device",
        "cpu",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert expected_error in result.stderr
    assert "Traceback" not in result.stderr
    assert not run_dir.exists()


def test_cli_rejects_a_non_regular_config(cli_root: Path) -> None:
    run_dir = cli_root / "non-regular-run"
    result = _run_cli(
        "doctor",
        "--config",
        cli_root,
        "--run-dir",
        run_dir,
        "--device",
        "cpu",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "config must be a regular file" in result.stderr
    assert "Traceback" not in result.stderr
    assert not run_dir.exists()


def test_cli_rejects_an_oversized_config(cli_root: Path) -> None:
    config_path = cli_root / "oversized.toml"
    config_path.write_bytes(b"#" * (1024 * 1024 + 1))
    run_dir = cli_root / "oversized-run"

    result = _run_cli(
        "doctor",
        "--config",
        config_path,
        "--run-dir",
        run_dir,
        "--device",
        "cpu",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "config exceeds the 1 MiB Phase 0 limit" in result.stderr
    assert "Traceback" not in result.stderr
    assert not run_dir.exists()


@pytest.mark.parametrize(
    ("replacements", "expected_error"),
    (
        (
            {
                "vocab_size = 64": "vocab_size = 4",
                "d_model = 32": "d_model = 768",
                "n_layers = 2": "n_layers = 1",
                "ffn_dim = 64": "ffn_dim = 768",
                "max_seq_len = 32": "max_seq_len = 8",
                "batch_size = 8": "batch_size = 4096",
                "expected_trainable_parameters = 24478\n": "",
            },
            "hidden activations exceed the Phase 0 M0 resource budget",
        ),
        (
            {
                "vocab_size = 64": "vocab_size = 4",
                "d_model = 32": "d_model = 2",
                "n_layers = 2": "n_layers = 1",
                "n_heads = 4": "n_heads = 1",
                "ffn_dim = 64": "ffn_dim = 2",
                "spool_rank = 8": "spool_rank = 1000000",
                "max_seq_len = 32": "max_seq_len = 2",
                "batch_size = 8": "batch_size = 4096",
                "expected_trainable_parameters = 24478\n": "",
            },
            "Spool recurrence exceeds the Phase 0 M0 resource budget",
        ),
    ),
)
def test_cli_rejects_aggregate_activation_budgets_before_allocation(
    cli_root: Path,
    replacements: dict[str, str],
    expected_error: str,
) -> None:
    config_path = cli_root / "activation-budget.toml"
    source = CONFIG_PATH.read_text(encoding="utf-8")
    for old, new in replacements.items():
        assert old in source
        source = source.replace(old, new, 1)
    config_path.write_text(source, encoding="utf-8")
    run_dir = cli_root / "activation-budget-run"

    result = _run_cli(
        "doctor",
        "--config",
        config_path,
        "--run-dir",
        run_dir,
        "--device",
        "cpu",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert expected_error in result.stderr
    assert "Traceback" not in result.stderr
    assert not run_dir.exists()
