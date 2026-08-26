from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import uuid

import pytest
import torch
from torch.optim import AdamW

from moira.agent import AgentPolicy, AgentRole, MoiraAgentKernel
from moira.candidate_search import (
    CandidateArtifact,
    CandidateEvaluation,
    CandidateKind,
    CandidateRecord,
    CandidateSearch,
    FailureEvidence,
)
from moira.candidate_worker import (
    CANDIDATE_WORK_TOOL_ID,
    CandidateWorkBudget,
    CandidateWorker,
    prepare_candidate_work,
)
from moira.config import ModelConfig
from moira.dataset import file_sha256
from moira.lora_backend import (
    LORA_BACKEND_PROTOCOL,
    LoraCandidateBackend,
    LoraDatasetBundle,
    LoraTrainingConfig,
    lora_backend_contract,
)
from moira.m1_checkpoint import save_m1_checkpoint
from moira.masks import CognitiveMode
from moira.model import MoiraModel


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CATALOG = REPOSITORY_ROOT / "research" / "configs" / "m1-sources.toml"


@pytest.fixture
def lora_root() -> Iterator[Path]:
    root = REPOSITORY_ROOT / "research" / "runs" / f"lora-backend-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _sha(value: str | bytes) -> str:
    encoded = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _model_config() -> ModelConfig:
    return ModelConfig(260, 16, 1, 4, 32, 2, 4, 2, 8, 24)


def _checkpoint(path: Path, config: ModelConfig, manifest: dict[str, object]) -> None:
    torch.manual_seed(17)
    model = MoiraModel(config)
    save_m1_checkpoint(
        path,
        lineage_id="lora-lineage",
        global_step=1,
        sample_cursor=0,
        model=model,
        optimizer=AdamW(model.parameters(), lr=1e-3),
        loom=None,
        losses=(1.0,),
        manifest=manifest,
    )


def _bundle(root: Path) -> LoraDatasetBundle:
    train = root / "train.jsonl"
    evaluation = root / "evaluation.jsonl"
    _write_rows(
        train,
        [
            {"id": "train-1", "prompt": "one", "response": "two"},
            {"id": "train-2", "prompt": "red", "response": "blue"},
        ],
    )
    _write_rows(
        evaluation,
        [{"id": "eval-1", "prompt": "left", "response": "right"}],
    )
    return LoraDatasetBundle(
        CATALOG,
        train,
        evaluation,
        source_id="zai-longcite-45k",
        max_train_records=2,
        max_evaluation_records=1,
    )


def _search(active_digest: str) -> CandidateSearch:
    failure = FailureEvidence(
        failure_id="reasoning-selection-1",
        category="reasoning",
        dataset_id="selection-lora-v1",
        evidence_digest=_sha("failure-evidence"),
    )
    search = CandidateSearch((failure,))
    search.add_candidate(
        CandidateRecord(
            artifact=CandidateArtifact(
                candidate_id="base",
                lineage_id="lora-lineage",
                kind=CandidateKind.CORE_CHECKPOINT,
                artifact_digest=active_digest,
                creation_manifest_digest=_sha("base-creation"),
                parent_ids=(),
                iteration=0,
            ),
            evaluation=CandidateEvaluation(
                evaluation_manifest_digest=_sha("base-evaluation"),
                metric_digest=_sha("base-metric"),
                selection_score=0.0,
                viable=True,
                mutation_verified=True,
                failure_ids=("reasoning-selection-1",),
            ),
        )
    )
    return search


def _request(backend: LoraCandidateBackend, search: CandidateSearch):
    parents = search.select_parents(count=1, seed=7, iteration=1)
    failures = search.select_failures("base", count=1, seed=11)
    request = prepare_candidate_work(
        search,
        parents,
        failures,
        work_id="lora-work-1",
        candidate_id="lora-candidate-1",
        kind=CandidateKind.ADAPTER,
        active_checkpoint_digest=backend.active_checkpoint_digest,
        dataset_manifest_digest=backend.dataset_manifest_digest,
        trainer_manifest_digest=backend.trainer_manifest_digest,
        issued_at_ms=100,
        expires_at_ms=5_000,
        budget=CandidateWorkBudget(
            max_steps=2,
            max_samples=8,
            max_tokens=512,
            max_wall_ms=4_900,
            max_output_bytes=2 * 1024 * 1024,
        ),
    )
    return parents, failures, request


def _kernel(worker: CandidateWorker) -> MoiraAgentKernel:
    kernel = MoiraAgentKernel(
        AgentPolicy(
            policy_version=1,
            allowed_actions=frozenset({"candidate_training:run"}),
        ),
        permit_secret=b"moira-lora-backend-test-secret" * 2,
    )
    kernel.register_tool(worker.tool())
    kernel.open_lease(
        lease_id="lora-lease",
        task_id="lora-task",
        scope_id="lora-lineage",
        mode=CognitiveMode.ATROPOS,
        role=AgentRole.EXECUTOR,
        max_turns=1,
        max_effects=1,
        allowed_actions=("candidate_training:run",),
    )
    kernel.record_turn("lora-lease", "Train the selected bounded LoRA candidate")
    return kernel


def test_lora_training_config_and_contract_are_narrow() -> None:
    with pytest.raises(ValueError, match="Atropos"):
        LoraTrainingConfig(mode=CognitiveMode.CLOTHO)
    with pytest.raises(ValueError, match="target modules"):
        LoraTrainingConfig(target_modules=("token_embedding",))
    with pytest.raises(ValueError, match="fp32"):
        LoraTrainingConfig(precision="bf16")

    contract = lora_backend_contract()
    assert contract["protocol"] == LORA_BACKEND_PROTOCOL
    assert contract["trains_lora_parameters_only"] is True
    assert contract["requires_disjoint_selection_data"] is True
    assert contract["is_promotion_holdout"] is False
    assert contract["can_authorize"] is False
    assert contract["can_promote"] is False


def test_lora_dataset_rejects_train_evaluation_content_overlap(
    lora_root: Path,
) -> None:
    train = lora_root / "train.jsonl"
    evaluation = lora_root / "evaluation.jsonl"
    _write_rows(train, [{"id": "train", "prompt": "same", "response": "same"}])
    _write_rows(evaluation, [{"id": "eval", "prompt": "same", "response": "same"}])

    with pytest.raises(ValueError, match="content must be disjoint"):
        LoraDatasetBundle(
            CATALOG,
            train,
            evaluation,
            source_id="zai-longcite-45k",
            max_train_records=1,
            max_evaluation_records=1,
        )


def test_lora_dataset_is_immutable_and_digest_tracks_file_mutation(
    lora_root: Path,
) -> None:
    bundle = _bundle(lora_root)
    original = bundle.manifest_digest
    inputs = bundle.train_examples[0].inputs
    with pytest.raises(TypeError):
        inputs["text"] = "mutated after request preparation"  # type: ignore[index]
    bundle.train_jsonl.write_text(
        '{"id":"changed","prompt":"changed","response":"changed"}\n',
        encoding="utf-8",
    )

    assert bundle.manifest_digest != original


def test_permit_gated_lora_backend_emits_adapter_only_candidate(
    lora_root: Path,
) -> None:
    pytest.importorskip("peft")
    safetensors = pytest.importorskip("safetensors.torch")
    device = os.environ.get("MOIRA_LORA_TEST_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA LoRA gate requested but CUDA is unavailable")
    config = _model_config()
    manifest = {"schema": "test-base-v1", "purpose": "LoRA candidate test"}
    checkpoint = lora_root / "base.pt"
    _checkpoint(checkpoint, config, manifest)
    checkpoint_before = checkpoint.read_bytes()
    backend = LoraCandidateBackend(
        base_checkpoint=checkpoint,
        model_config=config,
        checkpoint_manifest=manifest,
        dataset=_bundle(lora_root),
        training=LoraTrainingConfig(
            rank=2,
            alpha=4,
            steps=2,
            batch_size=1,
            eval_batch_size=1,
            learning_rate=1e-2,
            min_improvement=-1_000.0,
            seed=23,
            device=device,
        ),
    )
    search = _search(backend.active_checkpoint_digest)
    parents, failures, request = _request(backend, search)
    worker = CandidateWorker(
        backend,
        active_checkpoint=checkpoint,
        staging_root=lora_root / "staging",
        clock_ms=lambda: 100,
    )
    worker.issue(
        request,
        search=search,
        parent_selection=parents,
        failure_selection=failures,
    )

    kernel = _kernel(worker)
    outcome = kernel.request_tool(
        "lora-lease",
        tool_id=CANDIDATE_WORK_TOOL_ID,
        target=request.work_id,
        payload=request.payload,
    )
    assert outcome.authorized is not None
    execution = kernel.execute_effect(outcome.authorized, CANDIDATE_WORK_TOOL_ID)
    receipt = worker.result(request.work_id)

    assert execution.receipt.outcome == "succeeded"
    assert checkpoint.read_bytes() == checkpoint_before
    assert receipt.candidate.artifact.kind is CandidateKind.ADAPTER
    assert receipt.candidate.evaluation.mutation_verified is True
    assert math.isfinite(receipt.candidate.evaluation.selection_score)
    workspace = lora_root / "staging" / request.request_digest
    assert {path.name for path in workspace.iterdir()} == {
        "adapter_model.safetensors",
        "creation_manifest.json",
        "selection_evaluation.json",
        "metrics.json",
    }
    tensors = safetensors.load_file(workspace / "adapter_model.safetensors")
    assert tensors
    assert all("lora_" in name for name in tensors)
    creation = json.loads((workspace / "creation_manifest.json").read_text("utf-8"))
    evaluation = json.loads(
        (workspace / "selection_evaluation.json").read_text("utf-8")
    )
    assert creation["active_checkpoint_digest"] == file_sha256(checkpoint)
    assert creation["dataset_manifest_digest"] == backend.dataset_manifest_digest
    assert evaluation["state_before_evaluation"] == evaluation["state_after_evaluation"]
    assert evaluation["is_selection_only"] is True
    assert evaluation["is_promotion_holdout"] is False

    search.add_candidate(receipt.candidate)
    assert {item.artifact.candidate_id for item in search.candidates} == {
        "base",
        "lora-candidate-1",
    }


def test_lora_backend_rejects_manifest_and_budget_mismatch(lora_root: Path) -> None:
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")
    config = _model_config()
    manifest = {"schema": "test-base-v1"}
    checkpoint = lora_root / "base.pt"
    _checkpoint(checkpoint, config, manifest)
    backend = LoraCandidateBackend(
        base_checkpoint=checkpoint,
        model_config=config,
        checkpoint_manifest=manifest,
        dataset=_bundle(lora_root),
        training=LoraTrainingConfig(
            rank=2,
            alpha=4,
            steps=2,
            batch_size=1,
            eval_batch_size=1,
            min_improvement=-1_000.0,
        ),
    )
    search = _search(backend.active_checkpoint_digest)
    _, _, request = _request(backend, search)
    workspace = lora_root / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="dataset manifest"):
        backend(
            replace(request, dataset_manifest_digest=_sha("different-dataset")),
            workspace,
        )
    with pytest.raises(ValueError, match="trainer manifest"):
        backend(
            replace(request, trainer_manifest_digest=_sha("different-trainer")),
            workspace,
        )
    with pytest.raises(ValueError, match="token budget"):
        backend(
            replace(
                request,
                budget=replace(request.budget, max_tokens=1),
            ),
            workspace,
        )
    backend.dataset.train_jsonl.write_text(
        '{"id":"changed","prompt":"changed","response":"changed"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dataset manifest"):
        backend(request, workspace)
