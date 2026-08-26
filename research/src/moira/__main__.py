"""Command-line interface for the MOIRA Phase 0 research package."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, asdict, fields
import json
import math
import os
from pathlib import Path
import pickle
import platform
import stat
import sys
import tempfile
import tomllib
from types import NoneType
from typing import TypeVar, get_args
import uuid

import numpy as np
import torch

from .checkpoint import (
    LoadedCheckpoint,
    _validate_phase0_model_budget,
    load_checkpoint,
    verify_checkpoint,
)
from .ablation import assert_equal_parameter_surface
from .agent import AgentPolicy, AgentRole, MoiraAgentKernel, agent_contract
from .agent_loop import MoiraAgentRuntime
from .agent_train import (
    AGENT_ACTION_OBJECTIVE,
    evaluate_agent_jsonl,
    train_agent_jsonl,
)
from .config import (
    DataConfig,
    ModelConfig,
    OptimizationConfig,
    ReproducibilityConfig,
    RunConfig,
    TrainingConfig,
)
from .dataset import SourceCatalog, iter_jsonl_examples
from .deployment import switch_deployment_profile
from .elastic import (
    BIT_PRECISION_LADDER,
    CANONICAL_PARAMETER_LADDER,
    PrecisionPolicy,
    build_elastic_plan,
)
from .host import StandaloneConsolePort
from .local_tools import DeveloperWorkspace, ReadOnlyWorkspace
from .masks import CognitiveMode
from .model import MoiraModel
from .m1_train import (
    evaluate_m1_capabilities,
    train_m1_capabilities,
    train_m1_jsonl,
)
from .scale_probe import run_scale_probe
from .qat import run_moira_qat_smoke
from .ollama import DEFAULT_OLLAMA_URL, OllamaConfig, OllamaCognition
from .process_tools import AllowlistedProcessRunner
from .synthetic import SyntheticStream
from .train import (
    _build_components,
    _seed_strict,
    _semantic_manifests,
    _validate_phase_zero,
    evaluate_smoke,
    train_until,
)


_CONFIG_SECTIONS = {
    "schema_version",
    "model",
    "data",
    "optimization",
    "reproducibility",
    "run",
}
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_M0_TRAINABLE_PARAMETERS = 10_000_000
_MAX_M0_LAYERS = 32
_MAX_M0_BATCH_SIZE = 4_096
_MAX_M0_SEQUENCE_LENGTH = 8_192
_MAX_M0_BATCH_TOKENS = 262_144
_MAX_M0_LOGIT_ELEMENTS = 16_777_216
_MAX_M0_ATTENTION_ELEMENTS = 16_777_216
_MAX_M0_HIDDEN_ELEMENTS = 16_777_216
_MAX_M0_SPOOL_ELEMENTS = 16_777_216
_MAX_M0_SCHEDULE_STEPS = 100_000
_MAX_M0_RETAINED_CHECKPOINTS = 8
_MAX_EVALUATION_BATCHES = 10_000
_MAX_M0_TRAINING_WORK = 100_000_000_000
_MAX_M0_EVALUATION_WORK = 10_000_000_000
_ConfigSection = TypeVar(
    "_ConfigSection",
    ModelConfig,
    DataConfig,
    OptimizationConfig,
    ReproducibilityConfig,
    RunConfig,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m moira",
        description="MOIRA M0 deterministic CPU and M1 inspection commands.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    agent = commands.add_parser(
        "agent", help="Inspect MOIRA's native bounded agent-layer contract."
    )
    agent_commands = agent.add_subparsers(dest="agent_command", required=True)
    agent_commands.add_parser(
        "inspect", help="Show roles, orientations, budgets, and authority boundaries."
    )
    run_ollama = agent_commands.add_parser(
        "run-ollama",
        help="Run a read-only native MOIRA agent using local Ollama inference.",
    )
    run_ollama.add_argument("--model", required=True)
    run_ollama.add_argument("--task", required=True)
    run_ollama.add_argument("--workspace", type=Path, required=True)
    run_ollama.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in CognitiveMode),
        default=CognitiveMode.LACHESIS.value,
    )
    run_ollama.add_argument(
        "--role",
        choices=tuple(role.value for role in AgentRole),
        default=AgentRole.PLANNER.value,
    )
    run_ollama.add_argument("--max-steps", type=int, default=8)
    run_ollama.add_argument("--max-effects", type=int)
    run_ollama.add_argument("--base-url", default=DEFAULT_OLLAMA_URL)
    run_ollama.add_argument("--timeout", type=float, default=60.0)
    run_ollama.add_argument(
        "--allow-write",
        action="store_true",
        help="Register atomic workspace.write; every request requires approval.",
    )
    run_ollama.add_argument(
        "--allow-command",
        action="append",
        default=[],
        metavar="COMMAND",
        help=(
            "Register process.run for this command; repeat to allow more commands. "
            "Every invocation requires approval."
        ),
    )
    run_ollama.add_argument("--command-timeout", type=float, default=30.0)
    train_actions = agent_commands.add_parser(
        "train-actions",
        help="Train the proof-scale shared decoder on canonical agent trajectories.",
    )
    train_actions.add_argument("--path", type=Path, required=True)
    train_actions.add_argument("--checkpoint", type=Path, required=True)
    train_actions.add_argument("--steps", type=int, default=1)
    train_actions.add_argument("--batch-size", type=int, default=1)
    train_actions.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train_actions.add_argument("--resume", action="store_true")
    evaluate_actions = agent_commands.add_parser(
        "evaluate-actions",
        help="Evaluate a checkpoint on distinct held-out action trajectories.",
    )
    evaluate_actions.add_argument("--path", type=Path, required=True)
    evaluate_actions.add_argument("--training-path", type=Path, required=True)
    evaluate_actions.add_argument("--checkpoint", type=Path, required=True)
    evaluate_actions.add_argument("--batch-size", type=int, default=1)
    evaluate_actions.add_argument("--max-records", type=int, default=128)
    evaluate_actions.add_argument("--max-new-tokens", type=int, default=128)
    evaluate_actions.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu"
    )

    deployment = commands.add_parser(
        "deployment", help="Inspect future deployment profile contracts."
    )
    deployment_commands = deployment.add_subparsers(
        dest="deployment_command", required=True
    )
    describe = deployment_commands.add_parser(
        "describe", help="Describe a deployment profile without changing state."
    )
    describe.add_argument("profile", choices=("consumer", "enterprise"))
    switch = deployment_commands.add_parser(
        "switch", help="Persist a local Consumer or Enterprise policy selection."
    )
    switch.add_argument("profile", choices=("consumer", "enterprise"))
    switch.add_argument(
        "--state",
        type=Path,
        required=True,
        help="JSON state path; this command does not convert or allocate weights.",
    )

    elastic = commands.add_parser(
        "elastic",
        help="Plan unbounded parameter growth and custom bitplane precision.",
    )
    elastic_commands = elastic.add_subparsers(
        dest="elastic_command", required=True
    )
    elastic_policy = elastic_commands.add_parser(
        "policy", help="Show the Consumer or Enterprise precision policy."
    )
    elastic_policy.add_argument(
        "--profile", required=True, choices=("consumer", "enterprise")
    )
    elastic_plan = elastic_commands.add_parser(
        "plan", help="Plan a growth tier without allocating the target weights."
    )
    elastic_plan.add_argument("--current", default="3B")
    elastic_plan.add_argument("--target", required=True)
    elastic_plan.add_argument(
        "--profile", required=True, choices=("consumer", "enterprise")
    )
    elastic_plan.add_argument(
        "--runtime-bits",
        help="Override the profile core level, for example int3, int9, or int81.",
    )
    qat_smoke = elastic_commands.add_parser(
        "qat-smoke",
        help="Train a proof-scale MOIRA model at one bit level, then promote it.",
    )
    qat_smoke.add_argument("--config", required=True, type=Path)
    qat_smoke.add_argument("--initial-bits", default="int3")
    qat_smoke.add_argument("--promote-to", default="int6")
    qat_smoke.add_argument("--steps-per-stage", default=1, type=int)
    qat_smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    qat_smoke.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional trusted-local continuation checkpoint path.",
    )

    m1 = commands.add_parser(
        "m1", help="Inspect the fixed M1 model surface without downloading weights."
    )
    m1_commands = m1.add_subparsers(dest="m1_command", required=True)
    m1_inspect = m1_commands.add_parser(
        "inspect", help="Count the M1 reference parameters on meta or CPU."
    )
    m1_inspect.add_argument("--device", choices=("meta", "cpu"), default="meta")
    m1_train = m1_commands.add_parser(
        "train-jsonl", help="Run bounded in-memory M1 training on local text/code JSONL."
    )
    m1_train.add_argument("--catalog", required=True, type=Path)
    m1_train.add_argument("--source", required=True)
    m1_train.add_argument("--path", required=True, type=Path)
    m1_train.add_argument("--steps", default=1, type=int)
    m1_train.add_argument("--batch-size", default=1, type=int)
    m1_train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    m1_train.add_argument("--checkpoint", type=Path)
    m1_train.add_argument("--resume", action="store_true")
    m1_probe = m1_commands.add_parser(
        "scale-probe", help="Run one bounded CUDA capacity probe without checkpointing."
    )
    m1_probe.add_argument("--milestone", choices=("m1", "m2", "m3"), required=True)
    m1_probe.add_argument(
        "--optimizer", choices=("adamw", "adamw8bit"), default="adamw"
    )
    m1_probe.add_argument("--sequence-length", default=64, type=int)
    m1_capabilities = m1_commands.add_parser(
        "train-capabilities",
        help="Train bounded vision/audio/structured capability prefixes on local tensors.",
    )
    m1_capabilities.add_argument("--catalog", required=True, type=Path)
    m1_capabilities.add_argument("--source", required=True)
    m1_capabilities.add_argument("--path", required=True, type=Path)
    m1_capabilities.add_argument("--observations", required=True, type=Path)
    m1_capabilities.add_argument(
        "--modality",
        required=True,
        choices=("vision", "audio", "time-series", "structured", "robotics", "cyber"),
    )
    m1_capabilities.add_argument("--steps", default=1, type=int)
    m1_capabilities.add_argument("--batch-size", default=1, type=int)
    m1_capabilities.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    m1_capabilities.add_argument("--checkpoint", type=Path)
    m1_capabilities.add_argument("--resume", action="store_true")
    m1_capabilities.add_argument("--max-slots", default=16, type=int)
    m1_capabilities.add_argument("--patch-size", default=16, type=int)
    m1_capabilities.add_argument("--scope-id", default="local")
    m1_evaluate = m1_commands.add_parser(
        "evaluate-capabilities",
        help="Evaluate a held-out capability checkpoint without updating weights.",
    )
    m1_evaluate.add_argument("--catalog", required=True, type=Path)
    m1_evaluate.add_argument("--source", required=True)
    m1_evaluate.add_argument("--path", required=True, type=Path)
    m1_evaluate.add_argument("--observations", required=True, type=Path)
    m1_evaluate.add_argument(
        "--modality",
        required=True,
        choices=("vision", "audio", "time-series", "structured", "robotics", "cyber"),
    )
    m1_evaluate.add_argument("--checkpoint", required=True, type=Path)
    m1_evaluate.add_argument("--split", default="eval")
    m1_evaluate.add_argument("--max-records", type=int)
    m1_evaluate.add_argument("--batch-size", default=1, type=int)
    m1_evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    m1_evaluate.add_argument("--max-slots", default=16, type=int)
    m1_evaluate.add_argument("--patch-size", default=16, type=int)
    m1_evaluate.add_argument("--scope-id", default="local")

    m2 = commands.add_parser(
        "m2", help="Train or inspect the fixed 153M MOIRA M2 reference surface."
    )
    m2_commands = m2.add_subparsers(dest="m2_command", required=True)
    m2_commands.add_parser(
        "inspect", help="Count the M2 reference parameters without allocating weights."
    )
    m2_train = m2_commands.add_parser(
        "train-jsonl", help="Run bounded resumable M2 training on local text/code JSONL."
    )
    m2_train.add_argument("--catalog", required=True, type=Path)
    m2_train.add_argument("--source", required=True)
    m2_train.add_argument("--path", required=True, type=Path)
    m2_train.add_argument("--steps", default=1, type=int)
    m2_train.add_argument("--batch-size", default=1, type=int)
    m2_train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    m2_train.add_argument("--checkpoint", type=Path)
    m2_train.add_argument("--resume", action="store_true")

    dataset = commands.add_parser(
        "dataset", help="Validate manifest-first local dataset metadata and samples."
    )
    dataset_commands = dataset.add_subparsers(
        dest="dataset_command", required=True
    )
    dataset_validate = dataset_commands.add_parser(
        "validate", help="Validate a source catalog without network access."
    )
    dataset_validate.add_argument("--catalog", required=True, type=Path)
    dataset_sample = dataset_commands.add_parser(
        "sample", help="Normalize a bounded local JSONL sample."
    )
    dataset_sample.add_argument("--catalog", required=True, type=Path)
    dataset_sample.add_argument("--source", required=True)
    dataset_sample.add_argument("--path", required=True, type=Path)
    dataset_sample.add_argument("--split", default="train")
    dataset_sample.add_argument("--preprocess-version", default="m1-v1")
    dataset_sample.add_argument("--split-seed", default=0, type=int)
    dataset_sample.add_argument("--max-records", default=5, type=int)

    doctor = commands.add_parser(
        "doctor", help="Validate an M0 config and local CPU run directory."
    )
    _add_config_argument(doctor)
    _add_run_dir_argument(doctor)
    _add_device_argument(doctor)

    train = commands.add_parser(
        "train", help="Run deterministic synthetic M0 training."
    )
    _add_config_argument(train)
    _add_run_dir_argument(train)
    _add_device_argument(train)
    train.add_argument("--stop-at-step", required=True, type=int)
    train.add_argument(
        "--resume",
        required=True,
        help="Use none, auto, or an explicit checkpoint path.",
    )

    checkpoint = commands.add_parser(
        "checkpoint", help="Operate on deterministic M0 checkpoints."
    )
    checkpoint_commands = checkpoint.add_subparsers(
        dest="checkpoint_command", required=True
    )
    verify = checkpoint_commands.add_parser(
        "verify", help="Fully verify and load an M0 checkpoint."
    )
    _add_config_argument(verify)
    _add_checkpoint_source_arguments(verify)

    evaluate = commands.add_parser(
        "evaluate", help="Evaluate a verified M0 checkpoint on fixed samples."
    )
    _add_config_argument(evaluate)
    _add_checkpoint_source_arguments(evaluate)
    _add_device_argument(evaluate)
    evaluate.add_argument("--batches", default=2, type=int)
    return parser


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)


def _add_run_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True, type=Path)


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", required=True)


def _add_checkpoint_source_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--checkpoint", type=Path)


def _load_config(path: Path) -> TrainingConfig:
    raw = tomllib.loads(_read_config_bytes(path).decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config must be a TOML table")
    actual_sections = set(raw)
    if actual_sections != _CONFIG_SECTIONS:
        missing = sorted(_CONFIG_SECTIONS - actual_sections)
        unknown = sorted(actual_sections - _CONFIG_SECTIONS)
        raise ValueError(
            "config sections must match the Phase 0 schema "
            f"(missing={missing}, unknown={unknown})"
        )
    if type(raw["schema_version"]) is not int:
        raise ValueError("schema_version must be an integer")
    config = TrainingConfig(
        schema_version=raw["schema_version"],
        model=_build_config_section("model", ModelConfig, raw["model"]),
        data=_build_config_section("data", DataConfig, raw["data"]),
        optimization=_build_config_section(
            "optimization", OptimizationConfig, raw["optimization"]
        ),
        reproducibility=_build_config_section(
            "reproducibility", ReproducibilityConfig, raw["reproducibility"]
        ),
        run=_build_config_section("run", RunConfig, raw["run"]),
    )
    _validate_config_resource_budget(config)
    return config


def _read_config_bytes(path: Path) -> bytes:
    initial = path.lstat()
    if not stat.S_ISREG(initial.st_mode):
        raise ValueError("config must be a regular file")
    if initial.st_size > _MAX_CONFIG_BYTES:
        raise ValueError("config exceeds the 1 MiB Phase 0 limit")

    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= int(getattr(os, name, 0))
    with os.fdopen(os.open(path, flags), "rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(
            initial, opened
        ):
            raise ValueError("config file identity changed before reading")
        content = handle.read(_MAX_CONFIG_BYTES + 1)
        handle.seek(0)
        if handle.read(_MAX_CONFIG_BYTES + 1) != content:
            raise ValueError("config file changed while reading")
        finished = os.fstat(handle.fileno())

    final = path.lstat()
    if (
        _file_status_signature(initial) != _file_status_signature(opened)
        or _file_status_signature(opened) != _file_status_signature(finished)
        or _file_status_signature(finished) != _file_status_signature(final)
    ):
        raise ValueError("config file changed while reading")
    if len(content) > _MAX_CONFIG_BYTES:
        raise ValueError("config exceeds the 1 MiB Phase 0 limit")
    return content


def _file_status_signature(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
    )


def _validate_config_resource_budget(config: TrainingConfig) -> None:
    if any(
        value > (1 << 64) - 1
        for value in (
            config.data.seed,
            config.data.train_split,
            config.data.eval_split,
        )
    ):
        raise ValueError("data seeds and splits exceed the Phase 0 limit")
    if config.reproducibility.seed > (1 << 32) - 1:
        raise ValueError("reproducibility seed exceeds the Phase 0 limit")
    if config.model.n_layers > _MAX_M0_LAYERS:
        raise ValueError("n_layers exceeds the Phase 0 M0 resource budget")
    if config.model.vocab_size < 4:
        raise ValueError("vocab_size must be at least 4")
    if config.model.max_seq_len < 2:
        raise ValueError("max_seq_len must be at least 2")
    if config.data.batch_size > _MAX_M0_BATCH_SIZE:
        raise ValueError("batch_size exceeds the Phase 0 M0 resource budget")
    if config.model.max_seq_len > _MAX_M0_SEQUENCE_LENGTH:
        raise ValueError("max_seq_len exceeds the Phase 0 M0 resource budget")
    (
        batch_tokens,
        logit_elements,
        attention_elements,
        hidden_elements,
        spool_elements,
    ) = _resource_products(config)
    if batch_tokens > _MAX_M0_BATCH_TOKENS:
        raise ValueError("batch token count exceeds the Phase 0 M0 resource budget")
    if logit_elements > _MAX_M0_LOGIT_ELEMENTS:
        raise ValueError("logit tensor exceeds the Phase 0 M0 resource budget")
    if attention_elements > _MAX_M0_ATTENTION_ELEMENTS:
        raise ValueError("attention tensor exceeds the Phase 0 M0 resource budget")
    if hidden_elements > _MAX_M0_HIDDEN_ELEMENTS:
        raise ValueError("hidden activations exceed the Phase 0 M0 resource budget")
    if spool_elements > _MAX_M0_SPOOL_ELEMENTS:
        raise ValueError("Spool recurrence exceeds the Phase 0 M0 resource budget")
    if config.optimization.schedule_total_steps > _MAX_M0_SCHEDULE_STEPS:
        raise ValueError("training schedule exceeds the Phase 0 M0 resource budget")
    if config.run.keep_last_checkpoints > _MAX_M0_RETAINED_CHECKPOINTS:
        raise ValueError("checkpoint retention exceeds the Phase 0 M0 resource budget")
    if any(
        value > _MAX_M0_SCHEDULE_STEPS
        for value in (
            config.run.log_every_steps,
            config.run.checkpoint_every_steps,
        )
    ):
        raise ValueError("run counts exceed the Phase 0 M0 resource budget")

    total_trainable = _meta_trainable_parameters(config.model)
    if total_trainable > _MAX_M0_TRAINABLE_PARAMETERS:
        raise ValueError("model exceeds the 10M-parameter Phase 0 M0 budget")


def _resource_products(
    config: TrainingConfig,
) -> tuple[int, int, int, int, int]:
    batch = config.data.batch_size
    layers = config.model.n_layers
    length = config.model.max_seq_len
    batch_tokens = batch * length
    hidden_elements = batch * layers * length * config.model.d_model
    return (
        batch_tokens,
        batch_tokens * config.model.vocab_size,
        batch * layers * config.model.n_heads * length * length,
        hidden_elements,
        hidden_elements * config.model.spool_rank,
    )


def _meta_trainable_parameters(config: ModelConfig) -> int:
    _validate_phase0_model_budget(config)
    with torch.device("meta"):
        probe = MoiraModel(config)
    return probe.parameter_ledger().total_trainable


def _m1_inspect(device: str) -> dict[str, object]:
    if device not in {"meta", "cpu"}:
        raise ValueError("M1 inspection device must be meta or cpu")
    config = ModelConfig.m1_cpu()
    if device == "meta":
        with torch.device("meta"):
            model = MoiraModel(config)
    else:
        model = MoiraModel(config)
    count = model.parameter_ledger().total_trainable
    if count != config.expected_trainable_parameters:
        raise ValueError("M1 reference parameter count does not match its contract")
    assert_equal_parameter_surface(model)
    return {
        "ablation_parameter_surface": "equal",
        "device": device,
        "expected_trainable_parameters": config.expected_trainable_parameters,
        "max_trainable_parameters": 50_000_000,
        "min_trainable_parameters": 30_000_000,
        "model": "MOIRA",
        "phase": "M1",
        "trainable_parameters": count,
        "weights_downloaded": False,
    }


def _m1_train(
    catalog_path: Path,
    source_id: str,
    jsonl_path: Path,
    steps: int,
    batch_size: int,
    device: str,
    checkpoint_path: Path | None,
    resume: bool,
    *,
    model_config: ModelConfig | None = None,
    phase: str = "M1",
) -> dict[str, object]:
    result = train_m1_jsonl(
        catalog_path,
        jsonl_path,
        source_id=source_id,
        steps=steps,
        batch_size=batch_size,
        device=device,
        checkpoint_path=checkpoint_path,
        resume=resume,
        model_config=model_config,
    )
    return {
        "checkpoint": (
            str(result.checkpoint_path.resolve())
            if result.checkpoint_path is not None
            else None
        ),
        "device": device,
        "final_loss": result.final_loss,
        "phase": phase,
        "samples": result.samples,
        "steps": result.steps,
        "tokens": result.tokens,
        "trainable_parameters": result.trainable_parameters,
        "checkpoint_written": result.checkpoint_path is not None,
        "losses": list(result.losses),
    }


def _m2_inspect() -> dict[str, object]:
    config = ModelConfig.m2_gpu()
    with torch.device("meta"):
        model = MoiraModel(config)
    count = model.parameter_ledger().total_trainable
    if count != config.expected_trainable_parameters:
        raise ValueError("M2 reference parameter count does not match its contract")
    assert_equal_parameter_surface(model)
    return {
        "ablation_parameter_surface": "equal",
        "device": "meta",
        "expected_trainable_parameters": config.expected_trainable_parameters,
        "max_trainable_parameters": 200_000_000,
        "min_trainable_parameters": 100_000_000,
        "model": "MOIRA",
        "phase": "M2",
        "trainable_parameters": count,
        "weights_downloaded": False,
    }


def _m1_train_capabilities(
    catalog_path: Path,
    source_id: str,
    jsonl_path: Path,
    observations_path: Path,
    modality: str,
    steps: int,
    batch_size: int,
    device: str,
    checkpoint_path: Path | None,
    resume: bool,
    max_slots: int,
    patch_size: int,
    scope_id: str,
) -> dict[str, object]:
    observations = _load_m1_observations(observations_path)
    result = train_m1_capabilities(
        catalog_path,
        jsonl_path,
        observations,
        source_id=source_id,
        modality=modality,
        steps=steps,
        batch_size=batch_size,
        device=device,
        checkpoint_path=checkpoint_path,
        resume=resume,
        max_slots=max_slots,
        patch_size=patch_size,
        scope_id=scope_id,
    )
    return {
        "capability_parameters": result.capability_parameters,
        "checkpoint": (
            str(result.checkpoint_path.resolve())
            if result.checkpoint_path is not None
            else None
        ),
        "checkpoint_written": result.checkpoint_path is not None,
        "device": device,
        "final_loss": result.final_loss,
        "losses": list(result.losses),
        "modality": modality,
        "phase": "M1",
        "samples": result.samples,
        "steps": result.steps,
        "tokens": result.tokens,
        "trainable_parameters": result.trainable_parameters,
    }


def _m1_evaluate_capabilities(
    catalog_path: Path,
    source_id: str,
    jsonl_path: Path,
    observations_path: Path,
    modality: str,
    checkpoint_path: Path,
    split: str,
    max_records: int | None,
    batch_size: int,
    device: str,
    max_slots: int,
    patch_size: int,
    scope_id: str,
) -> dict[str, object]:
    observations = _load_m1_observations(observations_path)
    result = evaluate_m1_capabilities(
        catalog_path,
        jsonl_path,
        observations,
        source_id=source_id,
        modality=modality,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        device=device,
        max_slots=max_slots,
        patch_size=patch_size,
        scope_id=scope_id,
        split=split,
        max_records=max_records,
    )
    return {
        "batch_losses": list(result.losses),
        "batches": result.batches,
        "capability_parameters": result.capability_parameters,
        "checkpoint": str(result.checkpoint_path.resolve()),
        "checkpoint_step": result.checkpoint_step,
        "device": device,
        "lineage_id": result.lineage_id,
        "loss": result.loss,
        "modality": modality,
        "phase": "M1",
        "samples": result.samples,
        "split": split,
        "tokens": result.tokens,
        "trainable_parameters": result.trainable_parameters,
        "weights_updated": False,
    }


def _load_m1_observations(path: Path) -> torch.Tensor:
    try:
        status = path.stat()
        if not path.is_file() or path.is_symlink():
            raise ValueError("M1 observations must be a regular file")
        if status.st_size <= 0 or status.st_size > 512 * 1024 * 1024:
            raise ValueError("M1 observations exceed the 512 MiB input limit")
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("M1 observations cannot be loaded safely") from error
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.ndim not in {3, 4}
        or value.shape[0] <= 0
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("M1 observations must be finite floating [N,T,F] or [N,3,H,W]")
    return value


def _dataset_validate(catalog_path: Path) -> dict[str, object]:
    catalog = SourceCatalog.from_toml(catalog_path)
    by_kind: dict[str, int] = {}
    by_modality: dict[str, int] = {}
    verified = 0
    for source in catalog.sources:
        by_kind[source.kind] = by_kind.get(source.kind, 0) + 1
        by_modality[source.modality] = by_modality.get(source.modality, 0) + 1
        verified += int(source.license_verified)
    return {
        "catalog": str(catalog_path.resolve()),
        "license_verified_sources": verified,
        "modalities": dict(sorted(by_modality.items())),
        "schema_version": catalog.schema_version,
        "source_count": len(catalog.sources),
        "source_kinds": dict(sorted(by_kind.items())),
        "weights_or_datasets_downloaded": False,
    }


def _dataset_sample(
    catalog_path: Path,
    source_id: str,
    path: Path,
    split: str,
    preprocess_version: str,
    split_seed: int,
    max_records: int,
) -> dict[str, object]:
    catalog = SourceCatalog.from_toml(catalog_path)
    source = catalog.by_id(source_id)
    if source.kind != "hf-dataset":
        raise ValueError("dataset sample requires an hf-dataset catalog source")
    policy = (("train", 8), ("eval", 2)) if split == "auto" else None
    examples = tuple(
        iter_jsonl_examples(
            path,
            source_id=source_id,
            split=split,
            preprocess_version=preprocess_version,
            split_seed=split_seed,
            split_policy=policy,
            max_records=max_records,
        )
    )
    return {
        "modalities": sorted({example.modality for example in examples}),
        "record_ids": [example.record_id for example in examples],
        "records": len(examples),
        "source_id": source_id,
        "weights_or_datasets_downloaded": False,
    }


def _validate_operation_work(
    config: TrainingConfig,
    *,
    units: int,
    limit: int,
    operation: str,
) -> None:
    batch_tokens, logit, attention, hidden, spool = _resource_products(config)
    per_unit = (
        _meta_trainable_parameters(config.model) * batch_tokens
        + logit
        + attention
        + hidden
        + spool
    )
    if units * per_unit > limit:
        raise ValueError(f"{operation} request exceeds the Phase 0 M0 work budget")


def _build_config_section(
    name: str,
    section_type: type[_ConfigSection],
    raw: object,
) -> _ConfigSection:
    if not isinstance(raw, dict):
        raise ValueError(f"config section {name!r} must be a table")
    section_fields = {field.name: field for field in fields(section_type)}
    unknown = sorted(set(raw) - set(section_fields))
    missing = sorted(
        field.name
        for field in section_fields.values()
        if field.default is MISSING
        and field.default_factory is MISSING
        and field.name not in raw
    )
    if unknown or missing:
        raise ValueError(
            f"config section {name!r} has invalid fields "
            f"(missing={missing}, unknown={unknown})"
        )
    for field_name, value in raw.items():
        annotation = section_fields[field_name].type
        if not _matches_annotation(value, annotation):
            raise ValueError(
                f"config field {name}.{field_name} has the wrong TOML type"
            )
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"config field {name}.{field_name} must be finite")
    return section_type(**raw)


def _matches_annotation(value: object, annotation: object) -> bool:
    alternatives = get_args(annotation)
    if alternatives:
        return any(_matches_annotation(value, item) for item in alternatives)
    if annotation is None or annotation is NoneType:
        return value is None
    return isinstance(annotation, type) and type(value) is annotation


def _elastic_policy(profile: str) -> dict[str, object]:
    policy = PrecisionPolicy.for_profile(profile)
    return {
        "architecture": "MOIRA-precision-lattice-v1",
        "profile": profile,
        "precision_policy": policy.to_dict(),
        "parameter_growth_ladder": [
            f"{value // 1_000_000_000}B" for value in CANONICAL_PARAMETER_LADDER
        ],
        "bit_precision_ladder": [f"int{value}" for value in BIT_PRECISION_LADDER],
        "unbounded_parameter_growth": True,
        "weights_allocated": False,
    }


def _elastic_plan(
    current: str,
    target: str,
    profile: str,
    runtime_bits: str | None,
) -> dict[str, object]:
    return build_elastic_plan(
        current,
        target,
        profile=profile,
        runtime_bits=runtime_bits,
    ).to_dict()


def _describe_deployment(profile: str) -> dict[str, object]:
    descriptions = {
        "consumer": (
            "One resident field, sequential support fields, local encrypted "
            "memory, lazy organs, and zero paid compute by default."
        ),
        "enterprise": (
            "A future profile with distributed placement, parallel fields, "
            "tenant isolation, audit sinks, training queues, and explicit cost budgets."
        ),
    }
    policy = _elastic_policy(profile)
    return {
        "implemented_in_phase_0": False,
        "model": "MOIRA",
        "profile": profile,
        "profile_changes_model_identity": False,
        "status": "local profile state available; runtime orchestration pending",
        "description": descriptions[profile],
        "switch_command_available": True,
        "switch_changes_weights": False,
        "profile_state_implemented": True,
        "runtime_profile_application_implemented": False,
        "precision_policy": policy["precision_policy"],
        "parameter_growth_ladder": policy["parameter_growth_ladder"],
        "bit_precision_ladder": policy["bit_precision_ladder"],
        "unbounded_parameter_growth": True,
    }


def _doctor(config_path: Path, run_dir: Path, device: str) -> dict[str, object]:
    config = _load_config(config_path)
    _validate_phase_zero(config, stop_at_step=1, device=device)
    _seed_strict(config.reproducibility.seed)
    model, _, _ = _build_components(config)
    ledger = model.parameter_ledger()
    ledger.assert_matches()
    SyntheticStream(
        config.data.seed,
        config.model.vocab_size,
        config.model.max_seq_len,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=run_dir, prefix=".moira-write-check-", delete=True
    ) as handle:
        handle.write(b"ok")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "expected_trainable_parameters": config.model.expected_trainable_parameters,
        "numpy": np.__version__,
        "ok": True,
        "phase": "M0",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "run_dir": str(run_dir.resolve()),
        "run_dir_writable": True,
        "synthetic_data": config.data.kind == "synthetic-indexed-v1",
        "trainable_parameters": ledger.total_trainable,
    }


def _train(
    config_path: Path,
    run_dir: Path,
    device: str,
    stop_at_step: int,
    resume: str,
) -> dict[str, object]:
    config = _load_config(config_path)
    _validate_phase_zero(config, stop_at_step=stop_at_step, device=device)
    _validate_operation_work(
        config,
        units=stop_at_step,
        limit=_MAX_M0_TRAINING_WORK,
        operation="training",
    )
    resume_source: str | Path | None
    if resume == "none":
        resume_source = None
    elif resume == "auto":
        resume_source = "auto"
    else:
        resume_source = Path(resume)
    result = train_until(
        config,
        run_dir,
        stop_at_step=stop_at_step,
        resume=resume_source,
        device=device,
    )
    return {
        "checkpoint": str(result.checkpoint_path.resolve()),
        "global_step": result.global_step,
        "lineage_id": result.lineage_id,
        "samples_seen": result.samples_seen,
        "steps_run": len(result.losses),
        "tokens_seen": result.tokens_seen,
    }


def _load_disposable_checkpoint(
    config: TrainingConfig,
    *,
    run_dir: Path | None,
    checkpoint: Path | None,
    device: str,
) -> tuple[MoiraModel, LoadedCheckpoint]:
    _validate_phase_zero(config, stop_at_step=1, device=device)
    _seed_strict(config.reproducibility.seed)
    model, optimizer, scheduler = _build_components(config)
    manifests = _semantic_manifests(config)
    source = run_dir if run_dir is not None else checkpoint
    if source is None:
        raise ValueError("a run directory or checkpoint is required")
    loaded = load_checkpoint(
        source,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=manifests,
        auto=run_dir is not None,
    )
    return model, loaded


def _verify(
    config_path: Path,
    run_dir: Path | None,
    checkpoint: Path | None,
) -> dict[str, object]:
    config = _load_config(config_path)
    if checkpoint is not None:
        _validate_phase_zero(config, stop_at_step=1, device="cpu")
        _seed_strict(config.reproducibility.seed)
        payload = verify_checkpoint(
            checkpoint,
            expected_manifests=_semantic_manifests(config),
        )
        cursor = payload["data_cursor"]
        return {
            "checkpoint": str(checkpoint.resolve()),
            "global_step": payload["global_step"],
            "lineage_id": payload["lineage_id"],
            "samples_seen": cursor["samples_seen"],
            "tokens_seen": cursor["tokens_seen"],
            "verified": True,
        }
    _, loaded = _load_disposable_checkpoint(
        config,
        run_dir=run_dir,
        checkpoint=None,
        device="cpu",
    )
    return {
        "checkpoint": str(loaded.checkpoint_path.resolve()),
        "global_step": loaded.global_step,
        "lineage_id": loaded.lineage_id,
        "samples_seen": loaded.data_cursor.samples_seen,
        "tokens_seen": loaded.data_cursor.tokens_seen,
        "verified": True,
    }


def _evaluate(
    config_path: Path,
    run_dir: Path | None,
    checkpoint: Path | None,
    device: str,
    batches: int,
) -> dict[str, object]:
    if batches <= 0:
        raise ValueError("batches must be a positive integer")
    if batches > _MAX_EVALUATION_BATCHES:
        raise ValueError("batches exceeds the Phase 0 evaluation budget")
    config = _load_config(config_path)
    _validate_operation_work(
        config,
        units=batches,
        limit=_MAX_M0_EVALUATION_WORK,
        operation="evaluation",
    )
    model, loaded = _load_disposable_checkpoint(
        config,
        run_dir=run_dir,
        checkpoint=checkpoint,
        device=device,
    )
    result = evaluate_smoke(config, model, batches=batches)
    return {
        "batches": result.batches,
        "checkpoint": str(loaded.checkpoint_path.resolve()),
        "checkpoint_step": loaded.global_step,
        "lineage_id": loaded.lineage_id,
        "loss": result.loss,
        "samples": result.samples,
        "tokens": result.tokens,
    }


def _run_ollama_agent(
    *,
    model: str,
    task: str,
    workspace: Path,
    mode: str,
    role: str,
    max_steps: int,
    max_effects: int | None,
    base_url: str,
    timeout: float,
    allow_write: bool,
    allowed_commands: tuple[str, ...],
    command_timeout: float,
) -> dict[str, object]:
    if (
        not isinstance(max_steps, int)
        or isinstance(max_steps, bool)
        or max_steps <= 0
        or max_steps > 1_000
    ):
        raise ValueError("max_steps must be between 1 and 1000")
    effective_max_effects = min(4, max_steps) if max_effects is None else max_effects
    if (
        not isinstance(effective_max_effects, int)
        or isinstance(effective_max_effects, bool)
        or effective_max_effects < 0
        or effective_max_effects > max_steps
    ):
        raise ValueError("max_effects must be between 0 and max_steps")

    selected_mode = CognitiveMode(mode)
    selected_role = AgentRole(role)
    local_workspace: ReadOnlyWorkspace
    if allow_write:
        local_workspace = DeveloperWorkspace(workspace)
    else:
        local_workspace = ReadOnlyWorkspace(workspace)
    tools = list(local_workspace.tools())
    if allowed_commands:
        tools.append(
            AllowlistedProcessRunner(
                local_workspace.root,
                allowed_commands,
                timeout_seconds=command_timeout,
            ).tool()
        )
    actions = frozenset(tool.action for tool in tools)
    approval_actions = frozenset(
        action
        for action in actions
        if action in {"filesystem_write:write", "process:run"}
    )
    kernel = MoiraAgentKernel(
        AgentPolicy(
            policy_version=1,
            allowed_actions=actions,
            approval_actions=approval_actions,
        ),
    )
    for tool in tools:
        kernel.register_tool(tool)

    task_id = f"ollama-task-{uuid.uuid4().hex}"
    lease_id = f"ollama-lease-{uuid.uuid4().hex}"
    kernel.open_lease(
        lease_id=lease_id,
        task_id=task_id,
        scope_id=str(local_workspace.root),
        mode=selected_mode,
        role=selected_role,
        max_turns=max_steps,
        max_effects=effective_max_effects,
        allowed_actions=tuple(sorted(actions)),
    )
    result = MoiraAgentRuntime(kernel).run(
        lease_id=lease_id,
        task_id=task_id,
        task=task,
        mode=selected_mode,
        role=selected_role,
        cognition=OllamaCognition(
            OllamaConfig(
                model=model,
                base_url=base_url,
                timeout_seconds=timeout,
            )
        ),
        port=StandaloneConsolePort(output=sys.stderr),
        max_steps=max_steps,
    )
    return {
        "agent_protocol": "moira-agent-protocol-v1",
        "effects": len(result.receipts),
        "enabled_tools": [tool.tool_id for tool in tools],
        "model": model,
        "mode": selected_mode.value,
        "profile": "developer" if approval_actions else "read-only",
        "read_only": not approval_actions,
        "receipts": [receipt.receipt_id for receipt in result.receipts],
        "role": selected_role.value,
        "status": result.status.value,
        "steps": result.steps,
        "workspace": str(local_workspace.root),
    }


def _train_agent_actions(
    path: Path,
    checkpoint: Path,
    *,
    steps: int,
    batch_size: int,
    device: str,
    resume: bool,
) -> dict[str, object]:
    result = train_agent_jsonl(
        path,
        steps=steps,
        batch_size=batch_size,
        device=device,
        checkpoint_path=checkpoint,
        resume=resume,
    )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "final_loss": result.final_loss,
        "model_surface": "agent-smoke",
        "objective": AGENT_ACTION_OBJECTIVE,
        "samples": result.samples,
        "steps": result.steps,
        "tokens": result.tokens,
        "trainable_parameters": result.trainable_parameters,
        "weights_updated": True,
    }


def _evaluate_agent_actions(
    path: Path,
    training_path: Path,
    checkpoint: Path,
    *,
    batch_size: int,
    max_records: int,
    max_new_tokens: int,
    device: str,
) -> dict[str, object]:
    result = evaluate_agent_jsonl(
        path,
        training_path=training_path,
        checkpoint_path=checkpoint,
        batch_size=batch_size,
        max_records=max_records,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    return {
        "action_kind_accuracy": result.action_kind_accuracy,
        "canonical_validity": result.canonical_validity,
        "checkpoint": str(result.checkpoint_path.resolve()),
        "checkpoint_step": result.checkpoint_step,
        "evaluation_dataset_sha256": result.evaluation_dataset_sha256,
        "exact_action_accuracy": result.exact_action_accuracy,
        "held_out": result.held_out,
        "loss": result.loss,
        "model_surface": "agent-smoke",
        "objective": AGENT_ACTION_OBJECTIVE,
        "samples": result.samples,
        "schema_validity": result.schema_validity,
        "termination_rate": result.termination_rate,
        "token_accuracy": result.token_accuracy,
        "tokens": result.tokens,
        "tool_selection_accuracy": result.tool_selection_accuracy,
        "tool_selection_samples": result.tool_selection_samples,
        "training_dataset_sha256": result.training_dataset_sha256,
        "trainable_parameters": result.trainable_parameters,
        "weights_updated": False,
    }


def _dispatch(arguments: argparse.Namespace) -> Mapping[str, object]:
    if arguments.command == "agent":
        if arguments.agent_command == "inspect":
            return agent_contract()
        if arguments.agent_command == "run-ollama":
            return _run_ollama_agent(
                model=arguments.model,
                task=arguments.task,
                workspace=arguments.workspace,
                mode=arguments.mode,
                role=arguments.role,
                max_steps=arguments.max_steps,
                max_effects=arguments.max_effects,
                base_url=arguments.base_url,
                timeout=arguments.timeout,
                allow_write=arguments.allow_write,
                allowed_commands=tuple(arguments.allow_command),
                command_timeout=arguments.command_timeout,
            )
        if arguments.agent_command == "train-actions":
            return _train_agent_actions(
                arguments.path,
                arguments.checkpoint,
                steps=arguments.steps,
                batch_size=arguments.batch_size,
                device=arguments.device,
                resume=arguments.resume,
            )
        if arguments.agent_command == "evaluate-actions":
            return _evaluate_agent_actions(
                arguments.path,
                arguments.training_path,
                arguments.checkpoint,
                batch_size=arguments.batch_size,
                max_records=arguments.max_records,
                max_new_tokens=arguments.max_new_tokens,
                device=arguments.device,
            )
        raise ValueError("unknown agent command")
    if arguments.command == "deployment":
        if arguments.deployment_command == "describe":
            return _describe_deployment(arguments.profile)
        if arguments.deployment_command == "switch":
            return switch_deployment_profile(arguments.profile, arguments.state)
        raise ValueError("unknown deployment command")
    if arguments.command == "elastic":
        if arguments.elastic_command == "policy":
            return _elastic_policy(arguments.profile)
        if arguments.elastic_command == "plan":
            return _elastic_plan(
                arguments.current,
                arguments.target,
                arguments.profile,
                arguments.runtime_bits,
            )
        if arguments.elastic_command == "qat-smoke":
            return run_moira_qat_smoke(
                _load_config(arguments.config),
                initial_bits=arguments.initial_bits,
                promote_to=arguments.promote_to,
                steps_per_stage=arguments.steps_per_stage,
                device=arguments.device,
                checkpoint=arguments.checkpoint,
            )
        raise ValueError("unknown elastic command")
    if arguments.command == "m1":
        if arguments.m1_command == "inspect":
            return _m1_inspect(arguments.device)
        if arguments.m1_command == "train-jsonl":
            return _m1_train(
                arguments.catalog,
                arguments.source,
                arguments.path,
                arguments.steps,
                arguments.batch_size,
                arguments.device,
                arguments.checkpoint,
                arguments.resume,
            )
        if arguments.m1_command == "scale-probe":
            return asdict(
                run_scale_probe(
                    arguments.milestone,
                    optimizer_name=arguments.optimizer,
                    sequence_length=arguments.sequence_length,
                )
            )
        if arguments.m1_command == "train-capabilities":
            return _m1_train_capabilities(
                arguments.catalog,
                arguments.source,
                arguments.path,
                arguments.observations,
                arguments.modality,
                arguments.steps,
                arguments.batch_size,
                arguments.device,
                arguments.checkpoint,
                arguments.resume,
                arguments.max_slots,
                arguments.patch_size,
                arguments.scope_id,
            )
        if arguments.m1_command == "evaluate-capabilities":
            return _m1_evaluate_capabilities(
                arguments.catalog,
                arguments.source,
                arguments.path,
                arguments.observations,
                arguments.modality,
                arguments.checkpoint,
                arguments.split,
                arguments.max_records,
                arguments.batch_size,
                arguments.device,
                arguments.max_slots,
                arguments.patch_size,
                arguments.scope_id,
            )
        raise ValueError("unknown M1 command")
    if arguments.command == "m2":
        if arguments.m2_command == "inspect":
            return _m2_inspect()
        if arguments.m2_command == "train-jsonl":
            return _m1_train(
                arguments.catalog,
                arguments.source,
                arguments.path,
                arguments.steps,
                arguments.batch_size,
                arguments.device,
                arguments.checkpoint,
                arguments.resume,
                model_config=ModelConfig.m2_gpu(),
                phase="M2",
            )
        raise ValueError("unknown M2 command")
    if arguments.command == "dataset":
        if arguments.dataset_command == "validate":
            return _dataset_validate(arguments.catalog)
        if arguments.dataset_command == "sample":
            return _dataset_sample(
                arguments.catalog,
                arguments.source,
                arguments.path,
                arguments.split,
                arguments.preprocess_version,
                arguments.split_seed,
                arguments.max_records,
            )
        raise ValueError("unknown dataset command")
    if arguments.command == "doctor":
        return _doctor(arguments.config, arguments.run_dir, arguments.device)
    if arguments.command == "train":
        return _train(
            arguments.config,
            arguments.run_dir,
            arguments.device,
            arguments.stop_at_step,
            arguments.resume,
        )
    if arguments.command == "checkpoint":
        return _verify(arguments.config, arguments.run_dir, arguments.checkpoint)
    if arguments.command == "evaluate":
        return _evaluate(
            arguments.config,
            arguments.run_dir,
            arguments.checkpoint,
            arguments.device,
            arguments.batches,
        )
    raise ValueError("unknown command")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        output = _dispatch(parser.parse_args(argv))
    except (KeyError, OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            output,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
