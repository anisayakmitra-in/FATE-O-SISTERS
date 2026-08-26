import hashlib
import json
import math
import multiprocessing
import os
import platform
import pickle
import pickletools
import random
import shutil
import struct
import subprocess
import uuid
import zipfile
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
import torch
import torch.optim.optimizer as torch_optimizer_module

import moira.checkpoint as checkpoint_module
from moira.checkpoint import load_checkpoint, save_checkpoint
from moira.config import ModelConfig
from moira.ledger import ParameterLedger
from moira.masks import CognitiveMode
from moira.model import MoiraModel
from moira.synthetic import SYNTHETIC_CURSOR_KIND, DataCursor, SyntheticStream
from moira.train import _masked_next_token_loss


@pytest.fixture
def run_dir() -> Iterator[Path]:
    path = Path("research/runs") / f"task5-checkpoint-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _runtime_manifest() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": "cpu",
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_threads": torch.get_num_threads(),
    }


def _set_runtime_field(field: str, value: object) -> None:
    if field == "deterministic_algorithms":
        torch.use_deterministic_algorithms(value)
    elif field == "cudnn_benchmark":
        torch.backends.cudnn.benchmark = value
    elif field == "cudnn_deterministic":
        torch.backends.cudnn.deterministic = value
    elif field == "matmul_allow_tf32":
        torch.backends.cuda.matmul.allow_tf32 = value
    elif field == "cudnn_allow_tf32":
        torch.backends.cudnn.allow_tf32 = value
    elif field == "float32_matmul_precision":
        torch.set_float32_matmul_precision(value)
    elif field == "torch_threads":
        torch.set_num_threads(value)
    else:
        raise AssertionError(f"unsupported runtime field {field}")


_STRICT_PHASE0_RUNTIME = {
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "float32_matmul_precision": "highest",
    "torch_threads": 1,
}


def _enable_strict_phase0_runtime() -> None:
    for field, value in _STRICT_PHASE0_RUNTIME.items():
        _set_runtime_field(field, value)


@pytest.fixture(scope="module", autouse=True)
def strict_phase0_runtime() -> Iterator[None]:
    before = _runtime_manifest()
    _enable_strict_phase0_runtime()
    try:
        yield
    finally:
        for field in _STRICT_PHASE0_RUNTIME:
            _set_runtime_field(field, before[field])


def _model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        n_heads=2,
        ffn_dim=16,
        thread_radius=1,
        spool_rank=2,
        knot_slots=2,
        capsule_width=4,
        max_seq_len=4,
        dropout=0.2,
    )


def _components(
    model_config: ModelConfig | None = None,
    lr_lambda: Callable[[int], float] | None = None,
) -> tuple[
    MoiraModel,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    model = MoiraModel(_model_config() if model_config is None else model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.01, foreach=False, fused=False
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        (
            checkpoint_module._phase0_schedule_lambda(2, 10)
            if lr_lambda is None
            else lr_lambda
        ),
    )
    return model, optimizer, scheduler


def _components_for_model(
    model: MoiraModel,
) -> tuple[
    MoiraModel,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=0.01, foreach=False, fused=False
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, checkpoint_module._phase0_schedule_lambda(2, 10)
    )
    return model, optimizer, scheduler


def _test_learning_rate_factor(step: int) -> float:
    if step < 2:
        return float(step + 1) / 2
    progress = min(max((step - 2) / 8, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _wrong_next_learning_rate_factor(step: int) -> float:
    factor = _test_learning_rate_factor(step)
    return factor if step <= 4 else factor + 0.125


def _wrong_later_learning_rate_factor(step: int) -> float:
    factor = _test_learning_rate_factor(step)
    return factor if step <= 5 else factor + 0.125


def _advance_components(
    model: MoiraModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    steps: int,
) -> DataCursor:
    stream = SyntheticStream(seed=11, vocab_size=16, sequence_length=4)
    cursor = DataCursor(SYNTHETIC_CURSOR_KIND, 0, 0, 0)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        batch, cursor = stream.batch(split=0, cursor=cursor, batch_size=2)
        output = model(batch.input_ids, mode=CognitiveMode.ATROPOS)
        loss = _masked_next_token_loss(output.logits, batch.targets, batch.loss_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
    return cursor


def _blocked_step_writer(
    run_dir: str,
    ready: object,
    release: object,
    result_path: str,
) -> None:
    _enable_strict_phase0_runtime()
    original_write = checkpoint_module._atomic_write_json

    def blocked_pointer(path: Path, value: Mapping[str, object]) -> None:
        if path.name == "latest.json" and value.get("step") == 2:
            ready.set()
            if not release.wait(20):
                raise RuntimeError("writer test release timed out")
        original_write(path, value)

    checkpoint_module._atomic_write_json = blocked_pointer
    try:
        _save(Path(run_dir), step=2, keep=4)
    except Exception as error:
        result = f"{type(error).__name__}:{error}"
    else:
        result = "ok"
    Path(result_path).write_text(result, encoding="utf-8")


def _crash_while_holding_run_lock(run_dir: str, ready: object) -> None:
    with checkpoint_module.checkpoint_run_lock(Path(run_dir)):
        ready.set()
        os._exit(0)


def _crash_during_checkpoint_publication(run_dir: str, boundary: str) -> None:
    _enable_strict_phase0_runtime()
    original_claim = checkpoint_module._claim_file
    original_copy = checkpoint_module._copy_checkpoint_snapshot

    def crash_after_claim(staged: Path, canonical: Path) -> bool:
        claimed = original_claim(staged, canonical)
        if claimed and boundary == "canonical" and canonical.suffix == ".pt":
            os._exit(0)
        if (
            claimed
            and boundary == "sidecar"
            and canonical.suffix == ".json"
            and canonical.name.startswith("step-")
        ):
            os._exit(0)
        return claimed

    def crash_after_hash(
        source: object, snapshot: object, expected_size: int
    ) -> str:
        digest = original_copy(source, snapshot, expected_size)
        if boundary == "hash":
            os._exit(0)
        return digest

    checkpoint_module._claim_file = crash_after_claim
    checkpoint_module._copy_checkpoint_snapshot = crash_after_hash
    _save(Path(run_dir), step=1)


def _manifests() -> dict[str, dict[str, object]]:
    config = _model_config()
    return {
        "model": {
            "vocab_size": config.vocab_size,
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "ffn_dim": config.ffn_dim,
            "thread_radius": config.thread_radius,
            "spool_rank": config.spool_rank,
            "knot_slots": config.knot_slots,
            "capsule_width": config.capsule_width,
            "max_seq_len": config.max_seq_len,
            "causal_knot_policy": config.causal_knot_policy,
            "dropout": config.dropout,
            "tie_embeddings": config.tie_embeddings,
            "expected_trainable_parameters": config.expected_trainable_parameters,
        },
        "data": {
            "kind": SYNTHETIC_CURSOR_KIND,
            "stream_version": 1,
            "seed": 11,
            "train_split": 0,
            "eval_split": 1,
            "batch_size": 2,
            "sequence_length": 4,
            "vocab_size": 16,
        },
        "optimization": {
            "optimizer": "adamw",
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 1,
            "schedule": "cosine-with-warmup",
            "warmup_steps": 2,
            "schedule_total_steps": 10,
            "precision": "fp32",
            "clip_grad_norm": 1.0,
        },
        "reproducibility": {"seed": 23, "deterministic": "strict"},
        "source": {
            "model.py": "1" * 64,
            "train.py": "2" * 64,
            "synthetic.py": "3" * 64,
            "checkpoint.py": "4" * 64,
        },
        "runtime": _runtime_manifest(),
    }


def verify_checkpoint(
    checkpoint_path: Path,
    *,
    expected_manifests: Mapping[str, Mapping[str, object]] | None = None,
    expected_lineage_id: str | None = None,
) -> dict[str, object]:
    return checkpoint_module.verify_checkpoint(
        checkpoint_path,
        expected_manifests=(
            _manifests() if expected_manifests is None else expected_manifests
        ),
        expected_lineage_id=expected_lineage_id,
    )


def _save(
    run_dir: Path,
    *,
    step: int = 4,
    keep: int = 2,
    lineage_id: str = "lineage-test",
    manifests: dict[str, dict[str, object]] | None = None,
    provenance: dict[str, object] | None = None,
):
    random.seed(31)
    np.random.seed(32)
    torch.manual_seed(33)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=step)
    selected_manifests = _manifests() if manifests is None else manifests
    return save_checkpoint(
        run_dir,
        lineage_id=lineage_id,
        global_step=step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        data_cursor=cursor,
        semantic_manifests=selected_manifests,
        provenance=(
            {
                "git": {
                    "available": True,
                    "revision": "abc123",
                    "dirty": False,
                }
            }
            if provenance is None
            else provenance
        ),
        parameter_ledger=model.parameter_ledger(),
        counters={"optimizer_steps": step, "batches_seen": step},
        last_metrics={
            "loss": 1.25,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        },
        keep_last_checkpoints=keep,
    )


def _save_supplied_components(
    run_dir: Path,
    model: MoiraModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cursor: DataCursor,
    parameter_ledger: ParameterLedger | None = None,
    semantic_manifests: Mapping[str, Mapping[str, object]] | None = None,
) -> object:
    return save_checkpoint(
        run_dir,
        lineage_id="lineage-test",
        global_step=4,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        data_cursor=cursor,
        semantic_manifests=(
            _manifests() if semantic_manifests is None else semantic_manifests
        ),
        provenance={
            "git": {"available": True, "revision": "abc123", "dirty": False}
        },
        parameter_ledger=(
            model.parameter_ledger() if parameter_ledger is None else parameter_ledger
        ),
        counters={"optimizer_steps": 4, "batches_seen": 4},
        last_metrics={
            "loss": 1.0,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        },
        keep_last_checkpoints=2,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rewrite_sidecar(record, **updates: object) -> None:
    sidecar = json.loads(record.sidecar_path.read_text(encoding="utf-8"))
    sidecar.update(updates)
    record.sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True), encoding="utf-8"
    )


def _rewrite_payload(record, mutate) -> None:
    payload = torch.load(record.checkpoint_path, map_location="cpu", weights_only=True)
    mutate(payload)
    torch.save(payload, record.checkpoint_path)
    _rewrite_sidecar(
        record,
        sha256=_sha256(record.checkpoint_path),
        byte_size=record.checkpoint_path.stat().st_size,
    )


def _rewrite_semantic_evidence(record, mutate) -> None:
    def update(payload: dict[str, object]) -> None:
        mutate(payload)
        payload["manifest_hashes"] = checkpoint_module._manifest_hashes(
            payload["semantic_manifests"]
        )

    _rewrite_payload(record, update)
    payload = torch.load(
        record.checkpoint_path, map_location="cpu", weights_only=True
    )
    _rewrite_sidecar(record, manifest_hashes=payload["manifest_hashes"])


def _copy_candidate(record, checkpoint_dir: Path, name: str) -> tuple[Path, Path]:
    target = checkpoint_dir / name
    sidecar_target = target.with_suffix(".json")
    shutil.copyfile(record.checkpoint_path, target)
    sidecar = json.loads(record.sidecar_path.read_text(encoding="utf-8"))
    sidecar["checkpoint"] = target.name
    sidecar_target.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")
    return target, sidecar_target


def _archive_entries(path: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(path, "r") as archive:
        return [(info.filename, archive.read(info)) for info in archive.infolist()]


def _replace_archive(path: Path, entries: Sequence[tuple[str, bytes]]) -> None:
    replacement = path.with_name(f".{path.name}.rewritten")
    with zipfile.ZipFile(
        replacement, "w", compression=zipfile.ZIP_STORED, allowZip64=False
    ) as archive:
        for name, value in entries:
            archive.writestr(name, value)
    os.replace(replacement, path)


def _pickle_value(value: object) -> bytes:
    encoded = pickle.dumps(value, protocol=2)
    assert encoded[:2] == b"\x80\x02" and encoded[-1:] == b"."
    return encoded[2:-1]


def _pickle_unicode(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return b"X" + struct.pack("<I", len(encoded)) + encoded


def _replace_pickle_op(
    value: bytes,
    operations: Sequence[tuple[object, object, int]],
    index: int,
    replacement: bytes,
) -> bytes:
    return _replace_pickle_ops(value, operations, index, index + 1, replacement)


def _replace_pickle_ops(
    value: bytes,
    operations: Sequence[tuple[object, object, int]],
    start_index: int,
    stop_index: int,
    replacement: bytes,
) -> bytes:
    start = operations[start_index][2]
    end = operations[stop_index][2]
    return value[:start] + replacement + value[end:]


def _pickle_memo_get(index: int) -> bytes:
    return b"h" + bytes((index,)) if index < 256 else b"j" + struct.pack("<I", index)


def _storage_pid_ops(value: bytes) -> tuple[
    list[tuple[object, object, int]], list[tuple[int, int, int]]
]:
    operations = list(pickletools.genops(value))
    descriptors: list[tuple[int, int, int]] = []
    for pid_index, (opcode, _, _) in enumerate(operations):
        if opcode.name != "BINPERSID":
            continue
        mark_index = max(
            index
            for index in range(pid_index)
            if operations[index][0].name == "MARK"
        )
        key_indices = [
            index
            for index in range(mark_index, pid_index)
            if operations[index][0].name == "BINUNICODE"
            and isinstance(operations[index][1], str)
            and operations[index][1].isdecimal()
        ]
        numel_indices = [
            index
            for index in range(mark_index, pid_index)
            if operations[index][0].name in {"BININT", "BININT1", "BININT2", "LONG1"}
        ]
        if key_indices and numel_indices:
            descriptors.append((pid_index, max(key_indices), max(numel_indices)))
    return operations, descriptors


def _shared_empty_tensor_pickle(count: int) -> bytes:
    value = bytearray(b"\x80\x02}q\x00")
    value += _pickle_unicode("junk")
    value += b"]q\x01("
    value += b"ctorch._utils\n_rebuild_tensor_v2\nq\x02("
    value += b"(" + _pickle_unicode("storage")
    value += b"ctorch\nFloatStorage\nq\x03"
    value += _pickle_unicode("0") + _pickle_unicode("cpu")
    value += b"K\x00tQq\x04K\x00K\x00\x85K\x01\x85\x89"
    value += b"ccollections\nOrderedDict\nq\x05)Rq\x06tR"
    for _ in range(count - 1):
        value += b"h\x02(h\x04K\x00K\x00\x85K\x01\x85\x89h\x06tR"
    value += b"es."
    return bytes(value)


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.shape == right.shape
        assert left.dtype == right.dtype
        assert left.device == right.device
        assert left.layout == right.layout
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


def _assert_failed_load_is_nonmutating(
    record,
    model: MoiraModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    match: str,
    expected_manifests: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    before = (
        deepcopy(model.state_dict()),
        deepcopy(optimizer.state_dict()),
        deepcopy(scheduler.state_dict()),
    )
    rng_before = _rng_snapshot()
    with pytest.raises(ValueError, match=match):
        load_checkpoint(
            record.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=(
                _manifests()
                if expected_manifests is None
                else expected_manifests
            ),
        )
    _assert_nested_equal(before[0], model.state_dict())
    _assert_nested_equal(before[1], optimizer.state_dict())
    _assert_nested_equal(before[2], scheduler.state_dict())
    _assert_rng_equal(rng_before, _rng_snapshot())


def test_checkpoint_is_weights_only_loadable_complete_and_pointer_committed(
    run_dir: Path,
) -> None:
    record = _save(run_dir)

    payload = torch.load(
        record.checkpoint_path, map_location="cpu", weights_only=True
    )
    required = {
        "schema_version",
        "lineage_id",
        "global_step",
        "micro_step",
        "samples_seen",
        "tokens_seen",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "data_cursor",
        "semantic_manifests",
        "manifest_hashes",
        "provenance",
        "parameter_ledger",
        "counters",
        "last_metrics",
    }
    assert set(payload) == required
    assert payload["schema_version"] == 2
    assert payload["lineage_id"] == "lineage-test"
    assert payload["global_step"] == 4
    assert payload["micro_step"] == 0
    assert payload["samples_seen"] == 8
    assert payload["tokens_seen"] == 32
    assert set(payload["rng_state"]) == {"python", "numpy", "torch_cpu"}
    assert isinstance(payload["rng_state"]["python"]["state"], torch.Tensor)
    assert isinstance(payload["rng_state"]["numpy"]["keys"], torch.Tensor)

    sidecar = json.loads(record.sidecar_path.read_text(encoding="utf-8"))
    assert set(sidecar) == checkpoint_module._SIDECAR_KEYS
    assert sidecar["sha256"] == _sha256(record.checkpoint_path)
    assert sidecar["byte_size"] == record.checkpoint_path.stat().st_size
    assert sidecar["step"] == 4
    assert sidecar["lineage_id"] == "lineage-test"
    assert sidecar["manifest_hashes"] == payload["manifest_hashes"]

    pointer = json.loads(record.latest_path.read_text(encoding="utf-8"))
    assert set(pointer) == checkpoint_module._POINTER_KEYS
    assert pointer["checkpoint"] == record.checkpoint_path.name
    assert pointer["sidecar"] == record.sidecar_path.name
    assert pointer["sha256"] == sidecar["sha256"]
    assert pointer["byte_size"] == sidecar["byte_size"]
    verified = verify_checkpoint(
        record.checkpoint_path,
        expected_manifests=_manifests(),
        expected_lineage_id="lineage-test",
    )
    assert verified["global_step"] == 4


def test_public_verify_requires_caller_trusted_manifests(run_dir: Path) -> None:
    record = _save(run_dir, step=1)

    with pytest.raises(TypeError, match="expected_manifests"):
        checkpoint_module.verify_checkpoint(record.checkpoint_path)


def test_load_restores_component_states_and_returns_the_cursor(run_dir: Path) -> None:
    record = _save(run_dir)
    payload = torch.load(record.checkpoint_path, map_location="cpu", weights_only=True)
    model, optimizer, scheduler = _components()

    loaded = load_checkpoint(
        record.checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
    )

    assert loaded.global_step == 4
    assert loaded.lineage_id == "lineage-test"
    assert loaded.data_cursor == DataCursor(SYNTHETIC_CURSOR_KIND, 8, 8, 32)
    assert loaded.last_metrics == {
        "loss": 1.25,
        "learning_rate": 3e-4 * _test_learning_rate_factor(4),
    }
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, payload["model_state"][name])
    _assert_nested_equal(optimizer.state_dict(), payload["optimizer_state"])
    _assert_nested_equal(scheduler.state_dict(), payload["scheduler_state"])


def test_save_rejects_same_shape_model_config_manifest_mismatch(
    run_dir: Path,
) -> None:
    changed_config = replace(_model_config(), dropout=0.7)
    model, optimizer, scheduler = _components(changed_config)
    cursor = _advance_components(model, optimizer, scheduler, steps=4)

    with pytest.raises(ValueError, match="model|config|manifest"):
        save_checkpoint(
            run_dir,
            lineage_id="lineage-test",
            global_step=4,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            data_cursor=cursor,
            semantic_manifests=_manifests(),
            provenance={
                "git": {"available": True, "revision": "abc123", "dirty": False}
            },
            parameter_ledger=model.parameter_ledger(),
            counters={"optimizer_steps": 4, "batches_seen": 4},
            last_metrics={
                "loss": 1.0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            },
            keep_last_checkpoints=2,
        )


def test_load_rejects_same_shape_live_model_config_without_mutation(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components(
        replace(_model_config(), dropout=0.7)
    )

    _assert_failed_load_is_nonmutating(
        record, model, optimizer, scheduler, match="model|config|manifest"
    )


class _MoiraSubclass(MoiraModel):
    pass


class _DropoutSubclass(torch.nn.Dropout):
    pass


class _ForeignParameter(torch.nn.Parameter):
    pass


class _StorageCallbackParameter(torch.nn.Parameter):
    storage_calls = 0

    def untyped_storage(self):
        type(self).storage_calls += 1
        return super().untyped_storage()


class _GhostHooks(OrderedDict):
    def __init__(self, hook=None) -> None:
        super().__init__()
        self._ghost_hook = hook
        self._hidden_checks = 100

    def __bool__(self) -> bool:
        if self._hidden_checks:
            self._hidden_checks -= 1
            return False
        return True

    def items(self):
        return ((9_999, self._ghost_hook),)


class _GhostGlobalHooks(_GhostHooks):
    pass


class _GhostOptimizerHooks(_GhostHooks):
    pass


class _GhostGlobalOptimizerHooks(_GhostHooks):
    pass


class _CallbackParameterGroup(dict):
    callback_calls = 0

    def items(self):
        type(self).callback_calls += 1
        return super().items()

    def get(self, key, default=None):
        type(self).callback_calls += 1
        return super().get(key, default)


class _TraversalCallbackEmbedding(torch.nn.Embedding):
    callback_calls = 0

    def named_modules(self, *args: object, **kwargs: object):
        type(self).callback_calls += 1
        return super().named_modules(*args, **kwargs)


class _TraversalCallbackModules(dict):
    callback_calls = 0

    def items(self):
        type(self).callback_calls += 1
        return super().items()


class _CallbackCursor(DataCursor):
    callback_calls = 0

    def __getattribute__(self, name: str):
        if name in {"samples_seen", "tokens_seen", "next_sample_index"}:
            type(self).callback_calls += 1
        return super().__getattribute__(name)


class _CallbackLedger(ParameterLedger):
    callback_calls = 0

    def assert_matches(self) -> None:
        type(self).callback_calls += 1


def _components_with_parameter_storage_alias(
    attack: str, timing: str
) -> tuple[
    MoiraModel,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    torch.nn.Parameter,
    torch.nn.Parameter,
]:
    model = MoiraModel(_model_config())

    def apply_alias() -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
        parameters = dict(model.named_parameters(remove_duplicate=False))
        source = parameters["blocks.0.thread.query_projection.weight"]
        target = parameters["blocks.0.thread.key_projection.weight"]
        if attack == "partial-offset-zero":
            target = parameters["blocks.0.thread_norm.weight"]
            target.data = source.data.reshape(-1)[: target.numel()].view_as(target)
        elif attack == "distinct-wrapper-overlap":
            backing = np.zeros((8, 8), dtype=np.float32)
            source.data = torch.from_numpy(backing)
            target.data = torch.from_numpy(backing)
        elif attack == "distinct-wrapper-partial-overlap":
            backing = np.zeros(80, dtype=np.float32)
            source.data = torch.from_numpy(backing[:64].reshape(8, 8))
            target.data = torch.from_numpy(backing[16:].reshape(8, 8))
        else:
            target.data = source.data
        return source, target

    if timing == "before-optimizer":
        source, target = apply_alias()
    model, optimizer, scheduler = _components_for_model(model)
    if timing == "after-optimizer":
        source, target = apply_alias()
    assert source is not target
    source_storage = source.untyped_storage()
    target_storage = target.untyped_storage()
    if attack.startswith("distinct-wrapper"):
        assert source_storage is not target_storage
        source_start = source_storage.data_ptr()
        target_start = target_storage.data_ptr()
        assert max(source_start, target_start) < min(
            source_start + source_storage.nbytes(),
            target_start + target_storage.nbytes(),
        )
    else:
        assert source_storage is target_storage
    return model, optimizer, scheduler, source, target


def test_load_rejects_a_moira_model_subclass_without_mutation(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components_for_model(
        _MoiraSubclass(_model_config())
    )

    _assert_failed_load_is_nonmutating(
        record, model, optimizer, scheduler, match="model|graph|runtime|Phase 0"
    )


def test_load_rejects_foreign_parameter_runtime_semantics_without_mutation(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    model = MoiraModel(_model_config())
    foreign = _ForeignParameter(
        model.token_embedding.weight.detach().clone(), requires_grad=True
    )
    foreign.runtime_marker = "foreign"
    model.token_embedding.weight = foreign
    model.output_projection.weight = foreign
    model, optimizer, scheduler = _components_for_model(model)

    _assert_failed_load_is_nonmutating(
        record, model, optimizer, scheduler, match="parameter|model|runtime|semantic"
    )


@pytest.mark.parametrize("operation", ("save", "load"))
def test_rejects_foreign_parameter_before_storage_callback(
    run_dir: Path, operation: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    parameter_ledger = model.parameter_ledger()
    projection = model.blocks[0].thread.query_projection
    projection.weight = _StorageCallbackParameter(
        projection.weight.detach().clone(), requires_grad=True
    )
    _StorageCallbackParameter.storage_calls = 0

    with pytest.raises(ValueError, match="parameter|model|runtime|semantic"):
        if operation == "save":
            _save_supplied_components(
                run_dir / "foreign-storage-callback-save",
                model,
                optimizer,
                scheduler,
                cursor,
                parameter_ledger,
            )
        else:
            load_checkpoint(
                record.checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                expected_manifests=_manifests(),
            )

    assert _StorageCallbackParameter.storage_calls == 0


@pytest.mark.parametrize("operation", ("save", "load"))
def test_rejects_hidden_parameter_hook_registry_without_execution(
    run_dir: Path, operation: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    parameter = model.blocks[0].thread.query_projection.weight
    hook_calls = 0

    def backward_hook(gradient: torch.Tensor) -> torch.Tensor:
        nonlocal hook_calls
        hook_calls += 1
        return gradient

    handle = parameter.register_hook(backward_hook)
    original = parameter._backward_hooks
    parameter._backward_hooks = _GhostHooks(backward_hook)
    try:
        with pytest.raises(ValueError, match="parameter|hook|mapping|runtime"):
            if operation == "save":
                _save_supplied_components(
                    run_dir / "hidden-parameter-hook-save",
                    model,
                    optimizer,
                    scheduler,
                    cursor,
                )
            else:
                load_checkpoint(
                    record.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    expected_manifests=_manifests(),
                )
    finally:
        parameter._backward_hooks = original
        handle.remove()

    assert hook_calls == 0


@pytest.mark.parametrize(
    "attack", ("child-subclass", "root-shadow", "module-registry-subclass")
)
@pytest.mark.parametrize("operation", ("save", "load"))
def test_rejects_live_model_traversal_callbacks_before_execution(
    run_dir: Path, attack: str, operation: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    parameter_ledger = model.parameter_ledger()
    callback_calls = 0
    _TraversalCallbackEmbedding.callback_calls = 0
    _TraversalCallbackModules.callback_calls = 0
    if attack == "child-subclass":
        original = model.token_embedding
        replacement = _TraversalCallbackEmbedding(
            original.num_embeddings, original.embedding_dim
        )
        replacement.weight = original.weight
        model.token_embedding = replacement
    elif attack == "root-shadow":
        original_named_modules = model.named_modules

        def named_modules_callback(*args: object, **kwargs: object):
            nonlocal callback_calls
            callback_calls += 1
            return original_named_modules(*args, **kwargs)

        model.named_modules = named_modules_callback
    else:
        model._modules = _TraversalCallbackModules(model._modules)

    with pytest.raises(ValueError, match="model|graph|runtime|registry|module"):
        if operation == "save":
            _save_supplied_components(
                run_dir / "traversal-callback-save",
                model,
                optimizer,
                scheduler,
                cursor,
                parameter_ledger,
            )
        else:
            load_checkpoint(
                record.checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                expected_manifests=_manifests(),
            )

    assert callback_calls == 0
    assert _TraversalCallbackEmbedding.callback_calls == 0
    assert _TraversalCallbackModules.callback_calls == 0


@pytest.mark.parametrize("attack", ("cursor", "ledger"))
def test_save_rejects_phase0_input_subclasses_before_callbacks(
    run_dir: Path, attack: str
) -> None:
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    ledger = model.parameter_ledger()
    supplied_cursor: DataCursor = cursor
    supplied_ledger: ParameterLedger = ledger
    if attack == "cursor":
        supplied_cursor = _CallbackCursor(
            cursor.kind,
            cursor.next_sample_index,
            cursor.samples_seen,
            cursor.tokens_seen,
        )
        _CallbackCursor.callback_calls = 0
    else:
        supplied_ledger = _CallbackLedger(
            ledger.entries, ledger.total_trainable, model
        )
        _CallbackLedger.callback_calls = 0

    with pytest.raises(ValueError, match="DataCursor|ledger|Phase 0|exact"):
        save_checkpoint(
            run_dir,
            lineage_id="lineage-test",
            global_step=4,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            data_cursor=supplied_cursor,
            semantic_manifests=_manifests(),
            provenance={
                "git": {"available": True, "revision": "abc123", "dirty": False}
            },
            parameter_ledger=supplied_ledger,
            counters={"optimizer_steps": 4, "batches_seen": 4},
            last_metrics={
                "loss": 1.0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            },
            keep_last_checkpoints=2,
        )

    assert _CallbackCursor.callback_calls == 0
    assert _CallbackLedger.callback_calls == 0


@pytest.mark.parametrize("operation", ("save", "load"))
def test_checkpoint_sanitizes_hidden_post_accumulate_hooks(
    run_dir: Path, operation: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    parameter = model.blocks[0].thread.query_projection.weight
    hook_calls = 0

    def post_accumulate_hook(_parameter: torch.Tensor) -> None:
        nonlocal hook_calls
        hook_calls += 1

    parameter.register_post_accumulate_grad_hook(post_accumulate_hook)
    parameter._post_accumulate_grad_hooks = None
    if operation == "save":
        _save_supplied_components(run_dir / "hidden-post-hook-save", model, optimizer, scheduler, cursor)
    else:
        load_checkpoint(
            record.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
        )

    optimizer.zero_grad(set_to_none=True)
    (parameter * parameter).sum().backward()
    assert hook_calls == 0


@pytest.mark.parametrize("hook_api", ("register_hook", "register_prehook"))
@pytest.mark.parametrize("operation", ("save", "load"))
def test_checkpoint_rejects_retained_accumulate_grad_hooks_before_callbacks(
    run_dir: Path, operation: str, hook_api: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    parameter = model.blocks[0].thread.query_projection.weight
    accumulator = torch.autograd.graph.get_gradient_edge(parameter).node
    hook_calls = 0

    def hook(*_args: object) -> None:
        nonlocal hook_calls
        hook_calls += 1

    handle = getattr(accumulator, hook_api)(hook)
    try:
        if operation == "save":
            with pytest.raises(ValueError, match="hook|semantic|exact"):
                _save_supplied_components(
                    run_dir / "retained-accumulate-grad-save",
                    model,
                    optimizer,
                    scheduler,
                    cursor,
                )
        else:
            _assert_failed_load_is_nonmutating(
                record,
                model,
                optimizer,
                scheduler,
                match="hook|semantic|exact",
            )
        assert hook_calls == 0
    finally:
        handle.remove()


@pytest.mark.parametrize(
    "attack",
    (
        "full",
        "partial-offset-zero",
        "distinct-wrapper-overlap",
        "distinct-wrapper-partial-overlap",
    ),
)
@pytest.mark.parametrize("timing", ("before-optimizer", "after-optimizer"))
def test_load_rejects_unintended_parameter_storage_alias_without_mutation(
    run_dir: Path, attack: str, timing: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler, source, target = (
        _components_with_parameter_storage_alias(attack, timing)
    )

    _assert_failed_load_is_nonmutating(
        record, model, optimizer, scheduler, match="parameter|storage|alias|model"
    )
    source_storage = source.untyped_storage()
    target_storage = target.untyped_storage()
    if attack.startswith("distinct-wrapper"):
        assert source_storage is not target_storage
    else:
        assert source_storage is target_storage


@pytest.mark.parametrize(
    "attack",
    (
        "full",
        "partial-offset-zero",
        "distinct-wrapper-overlap",
        "distinct-wrapper-partial-overlap",
    ),
)
@pytest.mark.parametrize("timing", ("before-optimizer", "after-optimizer"))
def test_save_rejects_unintended_parameter_storage_alias(
    run_dir: Path, attack: str, timing: str
) -> None:
    model, optimizer, scheduler, _, _ = _components_with_parameter_storage_alias(
        attack, timing
    )
    cursor = _advance_components(model, optimizer, scheduler, steps=4)

    with pytest.raises(ValueError, match="parameter|storage|alias|model"):
        _save_supplied_components(run_dir, model, optimizer, scheduler, cursor)


@pytest.mark.parametrize("attack", ("shared-storage", "oversized-storage-span"))
def test_named_buffer_graph_rejects_storage_topology_drift(attack: str) -> None:
    assert not list(MoiraModel(_model_config()).named_buffers())
    canonical = [("first", torch.zeros(4)), ("second", torch.zeros(4))]
    if attack == "shared-storage":
        first = torch.zeros(4)
        live = [("first", first), ("second", first.view_as(first))]
    else:
        live = [("first", torch.zeros(8)[:4]), ("second", torch.zeros(4))]

    with pytest.raises(ValueError, match="buffer|storage|alias|topology"):
        checkpoint_module._validate_named_tensor_graph(live, canonical, "buffer")


def _active_parameter_and_state(
    model: MoiraModel, optimizer: torch.optim.Optimizer, alias: str
) -> tuple[torch.nn.Parameter, dict[str, torch.Tensor]]:
    parameter = dict(model.named_parameters())[alias]
    state = optimizer.state[parameter]
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    return parameter, state


@pytest.mark.parametrize("operation", ("save", "verify", "load"))
def test_rejects_model_optimizer_storage_overlap(
    run_dir: Path, operation: str
) -> None:
    alias = "blocks.0.thread.query_projection.weight"
    if operation == "save":
        model, optimizer, scheduler = _components()
        cursor = _advance_components(model, optimizer, scheduler, steps=4)
        parameter, state = _active_parameter_and_state(model, optimizer, alias)
        state["exp_avg"] = parameter.data

        with pytest.raises(ValueError, match="optimizer|storage|overlap|model"):
            _save_supplied_components(run_dir, model, optimizer, scheduler, cursor)
        return

    record = _save(run_dir)

    def overlap(payload: dict[str, object]) -> None:
        canonical = MoiraModel(_model_config())
        parameter_names = [name for name, _ in canonical.named_parameters()]
        saved_ids = payload["optimizer_state"]["param_groups"][0]["params"]
        saved_id = saved_ids[parameter_names.index(alias)]
        payload["optimizer_state"]["state"][saved_id]["exp_avg"] = payload[
            "model_state"
        ][alias]

    _rewrite_payload(record, overlap)
    if operation == "verify":
        with pytest.raises(ValueError, match="optimizer|storage|overlap|model"):
            verify_checkpoint(record.checkpoint_path)
    else:
        model, optimizer, scheduler = _components()
        _assert_failed_load_is_nonmutating(
            record,
            model,
            optimizer,
            scheduler,
            match="optimizer|storage|overlap|model",
        )


def test_save_rejects_partially_overlapping_optimizer_moments(
    run_dir: Path,
) -> None:
    aliases = (
        "blocks.0.thread.query_projection.weight",
        "blocks.0.thread.key_projection.weight",
    )
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    backing = np.zeros(96, dtype=np.float32)
    for alias, view in zip(aliases, (backing[:64], backing[32:96])):
        _, state = _active_parameter_and_state(model, optimizer, alias)
        state["exp_avg"] = torch.from_numpy(view.reshape(8, 8))

    with pytest.raises(ValueError, match="optimizer|storage|overlap"):
        _save_supplied_components(run_dir, model, optimizer, scheduler, cursor)


def test_load_rejects_live_model_optimizer_storage_overlap_without_mutation(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    _advance_components(model, optimizer, scheduler, steps=4)
    parameter, state = _active_parameter_and_state(
        model, optimizer, "blocks.0.thread.query_projection.weight"
    )
    state["exp_avg"] = parameter.data

    _assert_failed_load_is_nonmutating(
        record,
        model,
        optimizer,
        scheduler,
        match="optimizer|storage|overlap|model",
    )
    assert (
        parameter.untyped_storage().data_ptr()
        == state["exp_avg"].untyped_storage().data_ptr()
    )


@pytest.mark.parametrize("operation", ("save", "verify", "load"))
def test_rejects_negative_adamw_second_moment(
    run_dir: Path, operation: str
) -> None:
    if operation == "save":
        model, optimizer, scheduler = _components()
        cursor = _advance_components(model, optimizer, scheduler, steps=4)
        state = next(iter(optimizer.state.values()))
        state["exp_avg_sq"].fill_(-1.0)
        with pytest.raises(ValueError, match="AdamW|optimizer|second moment"):
            _save_supplied_components(run_dir, model, optimizer, scheduler, cursor)
        return

    record = _save(run_dir)

    def make_negative(payload: dict[str, object]) -> None:
        state = next(iter(payload["optimizer_state"]["state"].values()))
        state["exp_avg_sq"].fill_(-1.0)

    _rewrite_payload(record, make_negative)
    if operation == "verify":
        with pytest.raises(ValueError, match="AdamW|optimizer|second moment"):
            verify_checkpoint(record.checkpoint_path)
    else:
        model, optimizer, scheduler = _components()
        _assert_failed_load_is_nonmutating(
            record,
            model,
            optimizer,
            scheduler,
            match="AdamW|optimizer|second moment",
        )


@pytest.mark.parametrize(
    "attack",
    (
        "unwrapped-step",
        "parameter-group-subclass",
        "scheduler-shadow",
        "zero-grad-profile",
        "warning-flag",
        "false-opt-called",
    ),
)
@pytest.mark.parametrize("operation", ("save", "load"))
def test_rejects_live_optimizer_scheduler_runtime_callbacks_before_capture(
    run_dir: Path, attack: str, operation: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    before = (
        deepcopy(model.state_dict()),
        deepcopy(optimizer.state_dict()),
        deepcopy(scheduler.state_dict()),
    )
    rng_before = _rng_snapshot()
    callback_calls = 0
    _CallbackParameterGroup.callback_calls = 0
    if attack == "unwrapped-step":
        optimizer.step = lambda: None
    elif attack == "parameter-group-subclass":
        optimizer.param_groups[0] = _CallbackParameterGroup(
            optimizer.param_groups[0]
        )
    elif attack == "scheduler-shadow":
        scheduler_state = deepcopy(before[2])

        def hidden_state_dict() -> dict[str, object]:
            nonlocal callback_calls
            callback_calls += 1
            return deepcopy(scheduler_state)

        scheduler.state_dict = hidden_state_dict
    elif attack == "zero-grad-profile":
        optimizer._zero_grad_profile_name = None
    elif attack == "warning-flag":
        optimizer._warned_capturable_if_run_uncaptured = False
    else:
        assert optimizer._opt_called is True
        optimizer._opt_called = False

    with pytest.raises(ValueError, match="optimizer|scheduler|runtime|Phase 0"):
        if operation == "save":
            _save_supplied_components(
                run_dir / "component-runtime-save",
                model,
                optimizer,
                scheduler,
                cursor,
            )
        else:
            load_checkpoint(
                record.checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                expected_manifests=_manifests(),
            )

    assert callback_calls == 0
    assert _CallbackParameterGroup.callback_calls == 0
    _assert_nested_equal(before[0], model.state_dict())
    if attack == "parameter-group-subclass":
        optimizer.param_groups[0] = dict.copy(optimizer.param_groups[0])
    _assert_nested_equal(before[1], optimizer.state_dict())
    if attack != "scheduler-shadow":
        _assert_nested_equal(before[2], scheduler.state_dict())
    else:
        del scheduler.state_dict
        _assert_nested_equal(before[2], scheduler.state_dict())
    _assert_rng_equal(rng_before, _rng_snapshot())


@pytest.mark.parametrize(
    "hook_kind",
    (
        "step_pre",
        "step_post",
        "state_dict_pre",
        "state_dict_post",
        "load_state_dict_pre",
        "load_state_dict_post",
    ),
)
@pytest.mark.parametrize("operation", ("save", "load"))
def test_rejects_live_optimizer_hooks_before_execution_or_state_capture(
    run_dir: Path, hook_kind: str, operation: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    before = (
        deepcopy(model.state_dict()),
        deepcopy(optimizer.state_dict()),
        deepcopy(scheduler.state_dict()),
    )
    rng_before = _rng_snapshot()
    hook_calls = 0

    def optimizer_hook(*_args: object, **_kwargs: object) -> None:
        nonlocal hook_calls
        hook_calls += 1

    handle = getattr(optimizer, f"register_{hook_kind}_hook")(optimizer_hook)
    try:
        with pytest.raises(ValueError, match="optimizer|hook|runtime|Phase 0"):
            if operation == "save":
                _save_supplied_components(
                    run_dir / "optimizer-hook-save",
                    model,
                    optimizer,
                    scheduler,
                    cursor,
                )
            else:
                load_checkpoint(
                    record.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    expected_manifests=_manifests(),
                )
    finally:
        handle.remove()

    assert hook_calls == 0
    _assert_nested_equal(before[0], model.state_dict())
    _assert_nested_equal(before[1], optimizer.state_dict())
    _assert_nested_equal(before[2], scheduler.state_dict())
    _assert_rng_equal(rng_before, _rng_snapshot())


@pytest.mark.parametrize("operation", ("save", "load"))
def test_rejects_hidden_optimizer_hook_registry_subclass_without_execution(
    run_dir: Path, operation: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    hook_calls = 0

    def optimizer_hook(*_args: object, **_kwargs: object) -> None:
        nonlocal hook_calls
        hook_calls += 1

    original = optimizer._optimizer_step_pre_hooks
    optimizer._optimizer_step_pre_hooks = _GhostOptimizerHooks(optimizer_hook)
    try:
        with pytest.raises(ValueError, match="optimizer|hook|runtime|Phase 0"):
            if operation == "save":
                _save_supplied_components(
                    run_dir / "hidden-optimizer-hook-save",
                    model,
                    optimizer,
                    scheduler,
                    cursor,
                )
            else:
                load_checkpoint(
                    record.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    expected_manifests=_manifests(),
                )
    finally:
        optimizer._optimizer_step_pre_hooks = original

    assert hook_calls == 0


@pytest.mark.parametrize("hook_kind", ("pre", "post"))
@pytest.mark.parametrize("operation", ("save", "load"))
def test_rejects_process_global_optimizer_hooks_without_execution(
    run_dir: Path, hook_kind: str, operation: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    hook_calls = 0

    def optimizer_hook(*_args: object, **_kwargs: object) -> None:
        nonlocal hook_calls
        hook_calls += 1

    optimizer_module = torch_optimizer_module
    register = getattr(optimizer_module, f"register_optimizer_step_{hook_kind}_hook")
    handle = register(optimizer_hook)
    try:
        with pytest.raises(ValueError, match="global|optimizer|hook|runtime"):
            if operation == "save":
                _save_supplied_components(
                    run_dir / "global-optimizer-hook-save",
                    model,
                    optimizer,
                    scheduler,
                    cursor,
                )
            else:
                load_checkpoint(
                    record.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    expected_manifests=_manifests(),
                )
    finally:
        handle.remove()

    assert hook_calls == 0


@pytest.mark.parametrize("operation", ("save", "load"))
def test_rejects_hidden_global_optimizer_hook_registry_subclass(
    run_dir: Path, operation: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    hook_calls = 0

    def optimizer_hook(*_args: object, **_kwargs: object) -> None:
        nonlocal hook_calls
        hook_calls += 1

    optimizer_module = torch_optimizer_module
    original = optimizer_module._global_optimizer_pre_hooks
    optimizer_module._global_optimizer_pre_hooks = _GhostGlobalOptimizerHooks(
        optimizer_hook
    )
    try:
        with pytest.raises(ValueError, match="global|optimizer|hook|runtime"):
            if operation == "save":
                _save_supplied_components(
                    run_dir / "hidden-global-optimizer-hook-save",
                    model,
                    optimizer,
                    scheduler,
                    cursor,
                )
            else:
                load_checkpoint(
                    record.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    expected_manifests=_manifests(),
                )
    finally:
        optimizer_module._global_optimizer_pre_hooks = original

    assert hook_calls == 0


def test_load_rejects_a_hidden_module_hook_mapping_subclass_without_execution(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    hook_calls = 0

    def ghost_hook(_module: torch.nn.Module, inputs: tuple[object, ...]):
        nonlocal hook_calls
        hook_calls += 1
        return inputs

    model.embedding_dropout._forward_pre_hooks = _GhostHooks(ghost_hook)

    _assert_failed_load_is_nonmutating(
        record, model, optimizer, scheduler, match="hook|mapping|model|runtime"
    )
    assert hook_calls == 0


@pytest.mark.parametrize("hook_kind", ("forward", "registration"))
def test_load_rejects_process_global_module_hooks_before_canonical_construction(
    run_dir: Path, hook_kind: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    registration_calls = 0

    def registration_hook(
        _module: torch.nn.Module, _name: str, child: torch.nn.Module
    ) -> torch.nn.Module:
        nonlocal registration_calls
        registration_calls += 1
        return child

    if hook_kind == "forward":
        handle = torch.nn.modules.module.register_module_forward_hook(
            lambda _module, _inputs, output: output
        )
    else:
        handle = torch.nn.modules.module.register_module_module_registration_hook(
            registration_hook
        )
    try:
        _assert_failed_load_is_nonmutating(
            record, model, optimizer, scheduler, match="global|hook|model|runtime"
        )
    finally:
        handle.remove()

    assert registration_calls == 0


@pytest.mark.parametrize("operation", ("save", "load"))
def test_rejects_a_hidden_process_global_hook_mapping_subclass_without_execution(
    run_dir: Path, operation: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    hook_calls = 0

    def ghost_hook(
        _module: torch.nn.Module, _inputs: tuple[object, ...], output: object
    ) -> object:
        nonlocal hook_calls
        hook_calls += 1
        return output

    module_globals = torch.nn.modules.module
    original = module_globals._global_forward_hooks
    module_globals._global_forward_hooks = _GhostGlobalHooks(ghost_hook)
    try:
        with pytest.raises(ValueError, match="global|hook|mapping|model|runtime"):
            if operation == "load":
                load_checkpoint(
                    record.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    expected_manifests=_manifests(),
                )
            else:
                save_checkpoint(
                    run_dir / "global-hook-save",
                    lineage_id="lineage-test",
                    global_step=4,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    data_cursor=cursor,
                    semantic_manifests=_manifests(),
                    provenance={
                        "git": {
                            "available": True,
                            "revision": "abc123",
                            "dirty": False,
                        }
                    },
                    parameter_ledger=model.parameter_ledger(),
                    counters={"optimizer_steps": 4, "batches_seen": 4},
                    last_metrics={
                        "loss": 1.0,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    },
                    keep_last_checkpoints=2,
                )
    finally:
        module_globals._global_forward_hooks = original

    assert hook_calls == 0


@pytest.mark.parametrize(
    "mutate",
    (
        lambda model: setattr(model.embedding_dropout, "p", 0.7),
        lambda model: setattr(model.blocks[0].residual_dropout, "inplace", True),
        lambda model: setattr(model.pulse, "fast_decay", 0.25),
        lambda model: setattr(model.pulse, "slow_decay", 0.5),
        lambda model: setattr(model.pulse, "d_model", 9),
        lambda model: setattr(model.blocks[0].thread, "radius", 2),
        lambda model: setattr(model.blocks[0].thread, "n_heads", 1),
        lambda model: setattr(model.blocks[0].spool, "rank", 3),
        lambda model: setattr(model.blocks[0].knot, "max_landmarks", 3),
        lambda model: setattr(model.blocks[0].knot, "num_landmark_types", 7),
        lambda model: setattr(model.blocks[0], "knot_slots", 3),
        lambda model: setattr(model.final_norm, "eps", 1e-3),
        lambda model: setattr(model.token_embedding, "padding_idx", 0),
        lambda model: setattr(
            model,
            "embedding_dropout",
            _DropoutSubclass(model.config.dropout),
        ),
        lambda model: setattr(
            model.blocks[0], "residual_dropout", model.embedding_dropout
        ),
        lambda model: model.add_module("unexpected_runtime_module", torch.nn.Identity()),
        lambda model: model.blocks[0].register_forward_hook(
            lambda _module, _inputs, _output: None
        ),
        lambda model: next(model.parameters()).register_hook(lambda gradient: gradient),
    ),
    ids=(
        "embedding-dropout-probability",
        "residual-dropout-inplace",
        "pulse-fast-decay",
        "pulse-slow-decay",
        "pulse-width",
        "thread-radius",
        "thread-heads",
        "spool-rank",
        "knot-capacity",
        "knot-type-count",
        "block-knot-slots",
        "layer-norm-epsilon",
        "embedding-padding-index",
        "module-subclass",
        "module-alias-collapse",
        "extra-module",
        "forward-hook",
        "parameter-hook",
    ),
)
def test_load_rejects_live_module_graph_semantic_drift_without_mutation(
    run_dir: Path, mutate
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    mutate(model)

    _assert_failed_load_is_nonmutating(
        record, model, optimizer, scheduler, match="model|graph|runtime|semantic"
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "dropout",
        "state-dict-hook",
        "parameter-hook",
        "hidden-hook-mapping",
        "model-subclass",
    ),
)
def test_save_rejects_live_runtime_drift_before_state_dict_hooks_run(
    run_dir: Path, mutation: str
) -> None:
    model: MoiraModel = (
        _MoiraSubclass(_model_config())
        if mutation == "model-subclass"
        else MoiraModel(_model_config())
    )
    model, optimizer, scheduler = _components_for_model(model)
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    hook_ran = False
    if mutation == "dropout":
        model.embedding_dropout.p = 0.7
    elif mutation == "state-dict-hook":
        def forbidden_state_dict_hook(*args: object, **kwargs: object) -> None:
            nonlocal hook_ran
            hook_ran = True
            raise AssertionError("state_dict hook ran before semantic validation")

        model.register_state_dict_pre_hook(forbidden_state_dict_hook)
    elif mutation == "parameter-hook":
        next(model.parameters()).register_hook(lambda gradient: gradient)
    elif mutation == "hidden-hook-mapping":
        def ghost_hook(_module: torch.nn.Module, inputs: tuple[object, ...]):
            nonlocal hook_ran
            hook_ran = True
            return inputs

        model.embedding_dropout._forward_pre_hooks = _GhostHooks(ghost_hook)

    with pytest.raises(ValueError, match="model|graph|runtime|hook|Phase 0"):
        save_checkpoint(
            run_dir,
            lineage_id="lineage-test",
            global_step=4,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            data_cursor=cursor,
            semantic_manifests=_manifests(),
            provenance={
                "git": {"available": True, "revision": "abc123", "dirty": False}
            },
            parameter_ledger=model.parameter_ledger(),
            counters={"optimizer_steps": 4, "batches_seen": 4},
            last_metrics={
                "loss": 1.0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            },
            keep_last_checkpoints=2,
        )

    assert not hook_ran


def test_save_revalidates_live_components_after_provenance_normalization(
    run_dir: Path,
) -> None:
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    ledger = model.parameter_ledger()
    hook_ran = False
    hook_installed = False

    def forbidden_state_dict_hook(*_args: object, **_kwargs: object) -> None:
        nonlocal hook_ran
        hook_ran = True
        raise AssertionError("state_dict hook ran after provenance callback")

    class MutatingProvenance(dict):
        def items(self):
            nonlocal hook_installed
            if not hook_installed:
                model.register_state_dict_pre_hook(forbidden_state_dict_hook)
                hook_installed = True
            return super().items()

    provenance = MutatingProvenance(
        git={"available": True, "revision": "abc123", "dirty": False}
    )
    with pytest.raises(ValueError, match="model|runtime|hook|exact"):
        save_checkpoint(
            run_dir,
            lineage_id="lineage-test",
            global_step=4,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            data_cursor=cursor,
            semantic_manifests=_manifests(),
            provenance=provenance,
            parameter_ledger=ledger,
            counters={"optimizer_steps": 4, "batches_seen": 4},
            last_metrics={
                "loss": 1.0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            },
            keep_last_checkpoints=2,
        )

    assert hook_installed
    assert not hook_ran


@pytest.mark.parametrize("operation", ("save", "load"))
def test_checkpoint_rejects_config_subclass_without_attribute_callbacks(
    run_dir: Path, operation: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    ledger = model.parameter_ledger()
    callback_names: list[str] = []

    class EvilConfig(ModelConfig):
        def __getattribute__(self, name: str) -> object:
            callback_names.append(name)
            return super().__getattribute__(name)

    config = _model_config()
    evil = EvilConfig(
        **{
            name: object.__getattribute__(config, name)
            for name in config.__dataclass_fields__
        }
    )
    model.config = evil
    callback_names.clear()

    if operation == "save":
        with pytest.raises(ValueError, match="config|model|exact"):
            _save_supplied_components(
                run_dir / "evil-config-save",
                model,
                optimizer,
                scheduler,
                cursor,
                parameter_ledger=ledger,
            )
    else:
        _assert_failed_load_is_nonmutating(
            record, model, optimizer, scheduler, match="config|model|exact"
        )
    assert callback_names == []


@pytest.mark.parametrize("operation", ("save", "load"))
def test_checkpoint_rejects_foreign_config_field_without_callbacks(
    run_dir: Path, operation: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    ledger = model.parameter_ledger()
    callback_calls = 0

    class ForeignDropout:
        def __deepcopy__(self, memo: dict[int, object]) -> "ForeignDropout":
            nonlocal callback_calls
            callback_calls += 1
            return self

        def __eq__(self, other: object) -> bool:
            nonlocal callback_calls
            callback_calls += 1
            return False

        def __bool__(self) -> bool:
            nonlocal callback_calls
            callback_calls += 1
            return True

    object.__setattr__(model.config, "dropout", ForeignDropout())

    if operation == "save":
        with pytest.raises(ValueError, match="config|model|exact"):
            _save_supplied_components(
                run_dir / "foreign-config-field-save",
                model,
                optimizer,
                scheduler,
                cursor,
                parameter_ledger=ledger,
            )
    else:
        _assert_failed_load_is_nonmutating(
            record, model, optimizer, scheduler, match="config|model|exact"
        )
    assert callback_calls == 0


@pytest.mark.parametrize("operation", ("save", "load"))
@pytest.mark.parametrize("training_value", (False, "foreign"))
def test_checkpoint_rejects_non_training_module_state_without_callbacks(
    run_dir: Path, operation: str, training_value: object
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    callback_calls = 0

    class ForeignTraining:
        def __bool__(self) -> bool:
            nonlocal callback_calls
            callback_calls += 1
            return True

    model.embedding_dropout.training = (
        ForeignTraining() if training_value == "foreign" else training_value
    )
    if operation == "save":
        with pytest.raises(ValueError, match="training|model|runtime|exact"):
            _save_supplied_components(
                run_dir / "non-training-module-save",
                model,
                optimizer,
                scheduler,
                cursor,
            )
    else:
        _assert_failed_load_is_nonmutating(
            record, model, optimizer, scheduler, match="training|model|runtime|exact"
        )
    assert callback_calls == 0


@pytest.mark.parametrize("operation", ("save", "load"))
def test_checkpoint_rejects_foreign_module_dict_key_without_callbacks(
    run_dir: Path, operation: str
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    ledger = model.parameter_ledger()
    callback_calls = 0

    class ForeignTrainingKey:
        def __hash__(self) -> int:
            return hash("training")

        def __eq__(self, other: object) -> bool:
            nonlocal callback_calls
            callback_calls += 1
            return other == "training"

    module_vars = object.__getattribute__(model.embedding_dropout, "__dict__")
    training = dict.pop(module_vars, "training")
    module_vars[ForeignTrainingKey()] = training
    callback_calls = 0

    if operation == "save":
        with pytest.raises(ValueError, match="model|graph|key|exact"):
            _save_supplied_components(
                run_dir / "foreign-module-key-save",
                model,
                optimizer,
                scheduler,
                cursor,
                parameter_ledger=ledger,
            )
    else:
        _assert_failed_load_is_nonmutating(
            record, model, optimizer, scheduler, match="model|graph|key|exact"
        )
    assert callback_calls == 0


def test_load_preserves_the_all_modules_training_boundary(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    model.train()
    modes_before = tuple(module.training for module in model.modules())
    assert all(modes_before)

    load_checkpoint(
        record.checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
    )

    assert tuple(module.training for module in model.modules()) == modes_before


@pytest.mark.parametrize("operation", ("save", "load"))
def test_checkpoint_rejects_foreign_optimizer_state_key_without_callbacks(
    run_dir: Path, operation: str
) -> None:
    record = _save(run_dir) if operation == "load" else None
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    model_before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    hash_calls = 0

    class ForeignStateKey:
        def __hash__(self) -> int:
            nonlocal hash_calls
            hash_calls += 1
            return 17

    foreign_key = ForeignStateKey()
    optimizer.state[foreign_key] = deepcopy(next(iter(optimizer.state.values())))
    hash_calls = 0

    if operation == "save":
        with pytest.raises(ValueError, match="optimizer|AdamW|state|exact"):
            _save_supplied_components(
                run_dir / "foreign-optimizer-key-save",
                model,
                optimizer,
                scheduler,
                cursor,
            )
    else:
        assert record is not None
        with pytest.raises(ValueError, match="optimizer|AdamW|state|exact"):
            load_checkpoint(
                record.checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                expected_manifests=_manifests(),
            )
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, model_before[name])
    assert hash_calls == 0


@pytest.mark.parametrize("operation", ("save", "load"))
def test_checkpoint_rejects_foreign_optimizer_inner_key_without_callbacks(
    run_dir: Path, operation: str
) -> None:
    record = _save(run_dir) if operation == "load" else None
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    model_before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    callback_calls = 0

    class ForeignStepKey:
        def __hash__(self) -> int:
            return hash("step")

        def __eq__(self, other: object) -> bool:
            nonlocal callback_calls
            callback_calls += 1
            return other == "step"

    state = next(iter(optimizer.state.values()))
    step = state.pop("step")
    state[ForeignStepKey()] = step
    callback_calls = 0

    if operation == "save":
        with pytest.raises(ValueError, match="optimizer|AdamW|state|key|exact"):
            _save_supplied_components(
                run_dir / "foreign-optimizer-inner-key-save",
                model,
                optimizer,
                scheduler,
                cursor,
            )
    else:
        assert record is not None
        with pytest.raises(ValueError, match="optimizer|AdamW|state|key|exact"):
            load_checkpoint(
                record.checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                expected_manifests=_manifests(),
            )
        for name, tensor in model.state_dict().items():
            assert torch.equal(tensor, model_before[name])
    assert callback_calls == 0


@pytest.mark.parametrize(
    ("location", "key"),
    (
        ("module-registry", "thread_norm"),
        ("optimizer-vars", "state"),
        ("scheduler-vars", "optimizer"),
        ("optimizer-defaults", "lr"),
        ("parameter-group", "lr"),
        ("step-wrapper", "hooked"),
    ),
)
def test_live_runtime_rejects_foreign_dict_keys_without_callbacks(
    location: str, key: str
) -> None:
    model, optimizer, scheduler = _components()
    _advance_components(model, optimizer, scheduler, steps=4)
    callback_calls = 0

    class ForeignKey:
        def __hash__(self) -> int:
            return hash(key)

        def __eq__(self, other: object) -> bool:
            nonlocal callback_calls
            callback_calls += 1
            return other == key

    targets = {
        "module-registry": object.__getattribute__(
            model.blocks[0], "__dict__"
        )["_modules"],
        "optimizer-vars": vars(optimizer),
        "scheduler-vars": vars(scheduler),
        "optimizer-defaults": optimizer.defaults,
        "parameter-group": optimizer.param_groups[0],
        "step-wrapper": vars(optimizer.step),
    }
    target = targets[location]
    value = target.pop(key)
    target[ForeignKey()] = value
    callback_calls = 0

    with pytest.raises(ValueError, match="model|optimizer|scheduler|key|exact"):
        checkpoint_module._validate_live_component_runtime(
            model, optimizer, scheduler, _manifests()
        )

    assert callback_calls == 0


@pytest.mark.parametrize("register_name", ("register_hook", "register_prehook"))
def test_accumulate_grad_rejects_foreign_hook_keys_before_callbacks(
    register_name: str,
) -> None:
    model, _, _ = _components()
    parameter = next(model.parameters())
    accumulator = torch.autograd.graph.get_gradient_edge(parameter).node
    handle = getattr(accumulator, register_name)(lambda *_args: None)
    hooks = handle.hooks_dict_ref()
    handle.remove()
    target_id = torch.utils.hooks.RemovableHandle.next_id + (
        1 if register_name == "register_prehook" else 0
    )
    callback_calls = 0

    class ForeignHookKey:
        def __hash__(self) -> int:
            return hash(target_id)

        def __eq__(self, other: object) -> bool:
            nonlocal callback_calls
            callback_calls += 1
            return False

    hooks[ForeignHookKey()] = lambda *_args: None
    callback_calls = 0

    with pytest.raises(ValueError, match="AccumulateGrad|hook|key"):
        checkpoint_module._validate_accumulate_grad_hooks(parameter)

    assert callback_calls == 0


def test_save_rejects_scheduler_callable_that_diverges_at_the_next_step(
    run_dir: Path,
) -> None:
    model, optimizer, scheduler = _components(
        lr_lambda=_wrong_next_learning_rate_factor
    )
    cursor = _advance_components(model, optimizer, scheduler, steps=4)

    with pytest.raises(ValueError, match="scheduler|schedule|callable|LambdaLR"):
        save_checkpoint(
            run_dir,
            lineage_id="lineage-test",
            global_step=4,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            data_cursor=cursor,
            semantic_manifests=_manifests(),
            provenance={
                "git": {"available": True, "revision": "abc123", "dirty": False}
            },
            parameter_ledger=model.parameter_ledger(),
            counters={"optimizer_steps": 4, "batches_seen": 4},
            last_metrics={
                "loss": 1.0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            },
            keep_last_checkpoints=2,
        )


def test_load_rejects_wrong_scheduler_callable_without_mutation(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components(
        lr_lambda=_wrong_next_learning_rate_factor
    )

    _assert_failed_load_is_nonmutating(
        record, model, optimizer, scheduler, match="scheduler|schedule|callable|LambdaLR"
    )


def test_failed_scheduler_probe_does_not_mutate_callable_closure(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    calls: list[int] = []

    def stateful_wrong_factor(step: int) -> float:
        calls.append(step)
        return _wrong_next_learning_rate_factor(step)

    model, optimizer, scheduler = _components(lr_lambda=stateful_wrong_factor)
    calls_before = tuple(calls)

    _assert_failed_load_is_nonmutating(
        record, model, optimizer, scheduler, match="scheduler|schedule|callable|LambdaLR"
    )

    assert tuple(calls) == calls_before


def test_load_rejects_callable_that_diverges_after_the_next_step(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components(
        lr_lambda=_wrong_later_learning_rate_factor
    )

    _assert_failed_load_is_nonmutating(
        record, model, optimizer, scheduler, match="scheduler|schedule|callable|LambdaLR"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.__setitem__("schema_version", 3),
        lambda payload: payload.__setitem__("micro_step", 1),
        lambda payload: payload["data_cursor"].__setitem__("tokens_seen", 31),
        lambda payload: payload.__setitem__("lineage_id", "other-lineage"),
        lambda payload: payload["model_state"].pop(next(iter(payload["model_state"]))),
    ],
    ids=("schema", "micro-step", "cursor", "lineage", "model-state"),
)
def test_deep_payload_mismatches_are_rejected(run_dir: Path, mutation) -> None:
    record = _save(run_dir)
    _rewrite_payload(record, mutation)

    with pytest.raises(ValueError):
        verify_checkpoint(record.checkpoint_path)


@pytest.mark.parametrize(
    ("field", "value"), (("schema_version", 2.0), ("micro_step", False))
)
def test_payload_control_integers_require_exact_types(
    run_dir: Path, field: str, value: object
) -> None:
    record = _save(run_dir)
    _rewrite_payload(record, lambda payload: payload.__setitem__(field, value))

    with pytest.raises(ValueError, match="schema|micro step|integer"):
        verify_checkpoint(record.checkpoint_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (("schema_version", 2.0), ("step", 4.0), ("step", True)),
)
def test_sidecar_control_integers_require_exact_types(
    run_dir: Path, field: str, value: object
) -> None:
    record = _save(run_dir)
    _rewrite_sidecar(record, **{field: value})

    with pytest.raises(ValueError, match="sidecar|schema|step|integer"):
        verify_checkpoint(record.checkpoint_path)


@pytest.mark.parametrize("document", ("run", "latest"))
def test_run_control_schema_versions_require_exact_integers(
    run_dir: Path, document: str
) -> None:
    record = _save(run_dir)
    path = (
        record.checkpoint_path.parent / "run.json"
        if document == "run"
        else record.latest_path
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["schema_version"] = 2.0
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    if document == "latest":
        run_metadata = checkpoint_module._read_run_metadata(
            record.checkpoint_path.parent
        )
        with pytest.raises(ValueError, match="pointer|schema|integer"):
            checkpoint_module._validate_pointer_document(value, run_metadata)
        return
    model, optimizer, scheduler = _components()

    with pytest.raises(ValueError, match="metadata|pointer|schema|integer"):
        load_checkpoint(
            run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
            auto=True,
        )


@pytest.mark.parametrize("field", ["sha256", "byte_size"])
def test_wrong_sidecar_integrity_is_rejected(run_dir: Path, field: str) -> None:
    record = _save(run_dir)
    sidecar = json.loads(record.sidecar_path.read_text(encoding="utf-8"))
    replacement = "0" * 64 if field == "sha256" else sidecar[field] + 1
    _rewrite_sidecar(record, **{field: replacement})

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        verify_checkpoint(record.checkpoint_path)


def test_replacement_after_hash_never_reaches_live_model(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)
    payload = torch.load(record.checkpoint_path, map_location="cpu", weights_only=True)
    payload["model_state"]["final_norm.bias"] = (
        payload["model_state"]["final_norm.bias"] + 7.0
    )
    replacement = record.checkpoint_path.parent / ".replacement.pt"
    torch.save(payload, replacement)
    original_copy = checkpoint_module._copy_checkpoint_snapshot
    replaced = False

    def replace_after_hash(
        source: object, snapshot: object, expected_size: int
    ) -> str:
        nonlocal replaced
        digest = original_copy(source, snapshot, expected_size)
        if not replaced:
            replaced = True
            with record.checkpoint_path.open("r+b") as target:
                target.write(replacement.read_bytes())
                target.truncate()
                target.flush()
                os.fsync(target.fileno())
        return digest

    monkeypatch.setattr(
        checkpoint_module, "_copy_checkpoint_snapshot", replace_after_hash
    )
    model, optimizer, scheduler = _components()
    _assert_failed_load_is_nonmutating(
        record, model, optimizer, scheduler, match="identity|checksum|sha256|changed"
    )
    assert replaced


def test_save_never_points_to_a_valid_substitute_after_the_claim_check(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _save(run_dir / "substitute-source", step=2)
    substitute = torch.load(
        source.checkpoint_path, map_location="cpu", weights_only=True
    )
    substitute["model_state"]["final_norm.bias"] = (
        substitute["model_state"]["final_norm.bias"] + 7.0
    )
    substitute_path = run_dir / "substitute.pt"
    torch.save(substitute, substitute_path)
    first = _save(run_dir / "target", step=1, keep=3)
    original_sha256 = checkpoint_module._sha256
    injected = False

    def substitute_after_claim_hash(path: Path) -> str:
        nonlocal injected
        digest = original_sha256(path)
        if path.name == "step-000000002.pt" and not injected:
            injected = True
            with path.open("r+b") as target:
                target.write(substitute_path.read_bytes())
                target.truncate()
                target.flush()
                os.fsync(target.fileno())
        return digest

    monkeypatch.setattr(checkpoint_module, "_sha256", substitute_after_claim_hash)

    with pytest.raises(ValueError, match="published|intended|state"):
        _save(run_dir / "target", step=2, keep=3)

    assert injected
    assert json.loads(first.latest_path.read_text(encoding="utf-8"))["step"] == 1
    assert verify_checkpoint(first.checkpoint_path)["global_step"] == 1


def test_truncated_or_byte_modified_explicit_checkpoint_never_falls_back(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    original = record.checkpoint_path.read_bytes()
    record.checkpoint_path.write_bytes(original[: len(original) // 2])

    model, optimizer, scheduler = _components()
    with pytest.raises(ValueError):
        load_checkpoint(
            record.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
        )


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        (
            "optimization",
            lambda manifest: manifest.__setitem__("schedule_total_steps", 11),
        ),
        (
            "source",
            lambda manifest: manifest.__setitem__("train.py", "f" * 64),
        ),
        ("runtime", lambda manifest: manifest.__setitem__("pytorch", "changed")),
    ],
)
def test_changed_strict_manifest_is_rejected(
    run_dir: Path, section: str, mutate
) -> None:
    record = _save(run_dir)
    changed = deepcopy(_manifests())
    mutate(changed[section])

    with pytest.raises(ValueError, match="manifest"):
        verify_checkpoint(record.checkpoint_path, expected_manifests=changed)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload["semantic_manifests"]["data"].__setitem__(
            "unexpected", 1
        ),
        lambda payload: payload["semantic_manifests"]["data"].pop("batch_size"),
        lambda payload: payload["semantic_manifests"]["data"].__setitem__(
            "seed", -1
        ),
        lambda payload: payload["semantic_manifests"]["data"].__setitem__(
            "seed", True
        ),
        lambda payload: payload["semantic_manifests"]["data"].__setitem__(
            "eval_split", True
        ),
        lambda payload: payload["semantic_manifests"]["data"].__setitem__(
            "eval_split", 1 << 64
        ),
        lambda payload: payload["semantic_manifests"]["reproducibility"].__setitem__(
            "unexpected", 1
        ),
        lambda payload: payload["semantic_manifests"]["reproducibility"].pop(
            "seed"
        ),
        lambda payload: payload["semantic_manifests"]["reproducibility"].__setitem__(
            "deterministic", "best-effort"
        ),
        lambda payload: payload["semantic_manifests"]["reproducibility"].__setitem__(
            "seed", True
        ),
        lambda payload: payload["semantic_manifests"]["reproducibility"].__setitem__(
            "seed", 1 << 32
        ),
        lambda payload: payload["semantic_manifests"]["runtime"].__setitem__(
            "unexpected", 1
        ),
        lambda payload: payload["semantic_manifests"]["runtime"].pop(
            "torch_threads"
        ),
        lambda payload: payload["semantic_manifests"]["runtime"].__setitem__(
            "python", ""
        ),
    ),
    ids=(
        "data-extra",
        "data-missing",
        "data-invalid",
        "data-boolean-seed",
        "data-boolean-split",
        "data-out-of-range-split",
        "reproducibility-extra",
        "reproducibility-missing",
        "reproducibility-invalid",
        "reproducibility-boolean-seed",
        "reproducibility-out-of-range-seed",
        "runtime-extra",
        "runtime-missing",
        "runtime-context-invalid",
    ),
)
def test_standalone_verify_reconstructs_exact_data_reproducibility_and_runtime(
    run_dir: Path, mutate
) -> None:
    record = _save(run_dir)
    _rewrite_semantic_evidence(record, mutate)
    payload = torch.load(
        record.checkpoint_path, map_location="cpu", weights_only=True
    )

    with pytest.raises(
        ValueError,
        match="data|reproducibility|runtime|manifest|pickle|integer|budget",
    ):
        verify_checkpoint(
            record.checkpoint_path,
            expected_manifests=payload["semantic_manifests"],
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload["semantic_manifests"]["data"].__setitem__(
            "kind", "synthetic-indexed-v2"
        ),
        lambda payload: payload["semantic_manifests"]["data"].__setitem__(
            "stream_version", 2
        ),
        lambda payload: payload["semantic_manifests"]["data"].__setitem__(
            "stream_version", True
        ),
        lambda payload: payload["data_cursor"].__setitem__(
            "kind", "foreign-stream"
        ),
        lambda payload: payload["semantic_manifests"]["data"].__setitem__(
            "sequence_length", 3
        ),
        lambda payload: payload["semantic_manifests"]["data"].__setitem__(
            "sequence_length", 4.0
        ),
        lambda payload: payload["semantic_manifests"]["data"].__setitem__(
            "vocab_size", 15
        ),
        lambda payload: payload["semantic_manifests"]["data"].__setitem__(
            "vocab_size", 16.0
        ),
        lambda payload: payload["semantic_manifests"]["data"].__setitem__(
            "eval_split", 0
        ),
    ),
    ids=(
        "stream-kind",
        "stream-version",
        "stream-version-boolean",
        "cursor-kind",
        "sequence-model-binding",
        "sequence-type",
        "vocabulary-model-binding",
        "vocabulary-type",
        "distinct-splits",
    ),
)
def test_standalone_verify_binds_synthetic_stream_and_model_cross_fields(
    run_dir: Path, mutate
) -> None:
    record = _save(run_dir)
    _rewrite_semantic_evidence(record, mutate)
    payload = torch.load(
        record.checkpoint_path, map_location="cpu", weights_only=True
    )

    with pytest.raises(ValueError, match="data|stream|cursor|model|split|manifest"):
        verify_checkpoint(
            record.checkpoint_path,
            expected_manifests=payload["semantic_manifests"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("device", "cuda"),
        ("deterministic_algorithms", False),
        ("cudnn_benchmark", True),
        ("cudnn_deterministic", False),
        ("matmul_allow_tf32", True),
        ("cudnn_allow_tf32", True),
        ("float32_matmul_precision", "high"),
        ("torch_threads", 2),
    ),
)
def test_standalone_verify_enforces_phase_zero_cpu_runtime_semantics(
    run_dir: Path, field: str, value: object
) -> None:
    record = _save(run_dir)
    _rewrite_semantic_evidence(
        record,
        lambda payload: payload["semantic_manifests"]["runtime"].__setitem__(
            field, value
        ),
    )
    payload = torch.load(
        record.checkpoint_path, map_location="cpu", weights_only=True
    )

    with pytest.raises(ValueError, match="runtime|CPU|deterministic|manifest"):
        verify_checkpoint(
            record.checkpoint_path,
            expected_manifests=payload["semantic_manifests"],
        )


@pytest.mark.parametrize(
    ("field", "drifted"),
    (
        ("deterministic_algorithms", False),
        ("cudnn_benchmark", True),
        ("cudnn_deterministic", False),
        ("matmul_allow_tf32", True),
        ("cudnn_allow_tf32", True),
        ("float32_matmul_precision", "high"),
        ("torch_threads", 2),
    ),
)
@pytest.mark.parametrize("operation", ("save", "verify", "load"))
def test_checkpoint_rejects_current_runtime_drift_before_state_mutation(
    run_dir: Path, operation: str, field: str, drifted: object
) -> None:
    manifests = _manifests()
    record = _save(run_dir) if operation != "save" else None
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=4)
    original = _runtime_manifest()[field]
    try:
        _set_runtime_field(field, drifted)
        if operation == "save":
            with pytest.raises(ValueError, match="runtime|deterministic|CPU"):
                _save_supplied_components(
                    run_dir / "runtime-drift-save",
                    model,
                    optimizer,
                    scheduler,
                    cursor,
                    semantic_manifests=manifests,
                )
        elif operation == "verify":
            assert record is not None
            with pytest.raises(ValueError, match="runtime|deterministic|CPU"):
                verify_checkpoint(
                    record.checkpoint_path, expected_manifests=manifests
                )
        else:
            assert record is not None
            _assert_failed_load_is_nonmutating(
                record,
                model,
                optimizer,
                scheduler,
                match="runtime|deterministic|CPU",
                expected_manifests=manifests,
            )
    finally:
        _set_runtime_field(field, original)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("model", "n_layers", True),
        ("model", "causal_knot_policy", True),
        ("optimization", "clip_grad_norm", True),
    ),
)
def test_standalone_verify_requires_exact_model_and_optimization_scalar_types(
    run_dir: Path, section: str, field: str, value: object
) -> None:
    record = _save(run_dir)

    def mutate(payload: dict[str, object]) -> None:
        payload["semantic_manifests"][section][field] = value
        if section == "model":
            payload["parameter_ledger"]["model_config"][field] = value

    _rewrite_semantic_evidence(record, mutate)
    payload = torch.load(
        record.checkpoint_path, map_location="cpu", weights_only=True
    )

    with pytest.raises(ValueError, match="model|optimization|config|manifest|scalar"):
        verify_checkpoint(
            record.checkpoint_path,
            expected_manifests=payload["semantic_manifests"],
        )


def test_standalone_verify_requires_the_causal_knot_policy_manifest_field(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    _rewrite_semantic_evidence(
        record,
        lambda payload: payload["semantic_manifests"]["model"].pop(
            "causal_knot_policy"
        ),
    )
    payload = torch.load(
        record.checkpoint_path, map_location="cpu", weights_only=True
    )

    with pytest.raises(ValueError, match="model|manifest|scalar"):
        verify_checkpoint(
            record.checkpoint_path,
            expected_manifests=payload["semantic_manifests"],
        )


def test_checkpoint_provenance_has_an_explicit_non_strict_schema(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    _rewrite_payload(
        record,
        lambda payload: payload["provenance"].__setitem__("unexpected", True),
    )

    with pytest.raises(ValueError, match="provenance"):
        verify_checkpoint(record.checkpoint_path)


def test_failed_temporary_save_keeps_the_previous_pointer_and_checkpoint(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _save(run_dir, step=1)
    pointer_before = first.latest_path.read_bytes()
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=2)

    def fail_save(*args: object, **kwargs: object) -> None:
        raise OSError("injected save failure")

    monkeypatch.setattr(checkpoint_module.torch, "save", fail_save)
    with pytest.raises(OSError, match="injected"):
        save_checkpoint(
            run_dir,
            lineage_id="lineage-test",
            global_step=2,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            data_cursor=cursor,
            semantic_manifests=_manifests(),
            provenance={
                "git": {
                    "available": True,
                    "revision": "abc123",
                    "dirty": False,
                }
            },
            parameter_ledger=model.parameter_ledger(),
            counters={"optimizer_steps": 2, "batches_seen": 2},
            last_metrics={
                "loss": 1.0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            },
            keep_last_checkpoints=2,
        )

    assert first.latest_path.read_bytes() == pointer_before
    assert verify_checkpoint(first.checkpoint_path)["global_step"] == 1
    assert not (run_dir / "checkpoints" / "step-000000002.pt").exists()


def test_fresh_save_uses_three_bounded_artifact_passes(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_copy = checkpoint_module._copy_checkpoint_snapshot
    passes = 0

    def counted_copy(source: object, snapshot: object, expected_size: int) -> str:
        nonlocal passes
        passes += 1
        return original_copy(source, snapshot, expected_size)

    monkeypatch.setattr(
        checkpoint_module, "_copy_checkpoint_snapshot", counted_copy
    )

    _save(run_dir, step=1)

    assert passes == 3


def test_auto_resume_scans_only_when_the_pointer_is_damaged(run_dir: Path) -> None:
    first = _save(run_dir, step=1)
    second = _save(run_dir, step=2)
    damaged = b"not-json\n"
    second.latest_path.write_bytes(damaged)
    model, optimizer, scheduler = _components()

    loaded = load_checkpoint(
        run_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
        auto=True,
    )

    assert loaded.checkpoint_path.name == "step-000000002.pt"
    assert first.checkpoint_path.exists()
    assert second.latest_path.read_bytes() == damaged


def test_auto_resume_keeps_a_missing_pointer_missing(run_dir: Path) -> None:
    record = _save(run_dir, step=1)
    record.latest_path.unlink()
    model, optimizer, scheduler = _components()

    loaded = load_checkpoint(
        run_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
        auto=True,
    )

    assert loaded.global_step == 1
    assert not record.latest_path.exists()


def test_read_only_auto_recovery_allows_the_next_save_to_publish(
    run_dir: Path,
) -> None:
    record = _save(run_dir, step=1)
    damaged = b"not-json\n"
    record.latest_path.write_bytes(damaged)
    model, optimizer, scheduler = _components()

    loaded = load_checkpoint(
        run_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
        auto=True,
    )
    assert loaded.global_step == 1
    assert record.latest_path.read_bytes() == damaged

    advanced = _save(run_dir, step=2)

    assert json.loads(advanced.latest_path.read_text(encoding="utf-8"))["step"] == 2


@pytest.mark.parametrize(
    "failure",
    (
        "scaler",
        "live-config",
        "runtime-graph",
        "scheduler",
        "component-restore",
        "rng-restore",
    ),
)
def test_failed_auto_load_never_repairs_a_damaged_pointer(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    record = _save(run_dir, step=1)
    pointer_bytes = b"damaged-pointer\n"
    record.latest_path.write_bytes(pointer_bytes)
    if failure == "live-config":
        model, optimizer, scheduler = _components(
            replace(_model_config(), dropout=0.7)
        )
    elif failure == "scheduler":
        model, optimizer, scheduler = _components(
            lr_lambda=_wrong_next_learning_rate_factor
        )
    else:
        model, optimizer, scheduler = _components()
    if failure == "runtime-graph":
        model.pulse.fast_decay = 0.25
    if failure == "component-restore":
        def fail_component_restore(state: object) -> None:
            raise RuntimeError("injected component restore failure")

        monkeypatch.setattr(scheduler, "load_state_dict", fail_component_restore)
    injected_rng_failure = False
    original_restore_rng = checkpoint_module._restore_rng_state
    payload = torch.load(
        record.checkpoint_path, map_location="cpu", weights_only=True
    )

    def fail_live_rng_restore(state: object) -> None:
        nonlocal injected_rng_failure
        if (
            failure == "rng-restore"
            and not injected_rng_failure
            and checkpoint_module._values_equal(
                dict(model.state_dict()), payload["model_state"]
            )
        ):
            injected_rng_failure = True
            raise RuntimeError("injected RNG restore failure")
        original_restore_rng(state)

    if failure == "rng-restore":
        monkeypatch.setattr(
            checkpoint_module, "_restore_rng_state", fail_live_rng_restore
        )
    before = (
        deepcopy(model.state_dict()),
        deepcopy(optimizer.state_dict()),
        deepcopy(scheduler.state_dict()),
    )
    rng_before = _rng_snapshot()

    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint(
            run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=object() if failure == "scaler" else None,
            expected_manifests=_manifests(),
            auto=True,
        )

    assert record.latest_path.read_bytes() == pointer_bytes
    _assert_nested_equal(before[0], model.state_dict())
    _assert_nested_equal(before[1], optimizer.state_dict())
    _assert_nested_equal(before[2], scheduler.state_dict())
    _assert_rng_equal(rng_before, _rng_snapshot())
    if failure == "rng-restore":
        assert injected_rng_failure


def test_auto_resume_does_not_hide_corruption_behind_a_valid_pointer(
    run_dir: Path,
) -> None:
    first = _save(run_dir, step=1)
    second = _save(run_dir, step=2)
    second.checkpoint_path.write_bytes(b"corrupt")
    model, optimizer, scheduler = _components()

    with pytest.raises(ValueError):
        load_checkpoint(
            run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
            auto=True,
        )
    assert first.checkpoint_path.exists()


def test_pruning_removes_only_older_validated_checkpoints(run_dir: Path) -> None:
    first = _save(run_dir, step=1, keep=2)
    second = _save(run_dir, step=2, keep=2)
    third = _save(run_dir, step=3, keep=2)

    assert not first.checkpoint_path.exists()
    assert not first.sidecar_path.exists()
    assert verify_checkpoint(second.checkpoint_path)["global_step"] == 2
    assert verify_checkpoint(third.checkpoint_path)["global_step"] == 3


@pytest.mark.parametrize(
    "boundary",
    ("canonical", "reopen", "hash", "sidecar", "pointer"),
)
def test_same_step_retry_survives_each_publication_boundary(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    first = _save(run_dir, step=1, keep=3)
    pointer_before = first.latest_path.read_bytes()
    original_link = checkpoint_module.os.link
    original_load = checkpoint_module._load_payload_with_fingerprint
    original_sha256 = checkpoint_module._sha256
    original_write_json = checkpoint_module._atomic_write_json
    injected = False

    def link_then_fail(source: object, target: object, *args: object, **kwargs: object):
        nonlocal injected
        result = original_link(source, target, *args, **kwargs)
        target_path = Path(target)
        is_boundary = (
            boundary == "canonical" and target_path.suffix == ".pt"
        ) or (
            boundary == "sidecar"
            and target_path.name == "step-000000002.json"
        )
        if is_boundary and not injected:
            injected = True
            raise OSError(f"injected {boundary} failure")
        return result

    def load_then_fail(path: Path):
        nonlocal injected
        if boundary == "reopen" and not injected:
            injected = True
            raise OSError("injected reopen failure")
        return original_load(path)

    def hash_then_fail(path: Path) -> str:
        nonlocal injected
        if boundary == "hash" and not injected:
            injected = True
            raise OSError("injected hash failure")
        return original_sha256(path)

    def pointer_then_fail(path: Path, value: object) -> None:
        nonlocal injected
        if boundary == "pointer" and path.name == "latest.json" and not injected:
            injected = True
            raise OSError("injected pointer failure")
        original_write_json(path, value)

    with monkeypatch.context() as context:
        context.setattr(checkpoint_module.os, "link", link_then_fail)
        context.setattr(
            checkpoint_module, "_load_payload_with_fingerprint", load_then_fail
        )
        context.setattr(checkpoint_module, "_sha256", hash_then_fail)
        context.setattr(checkpoint_module, "_atomic_write_json", pointer_then_fail)
        with pytest.raises(OSError, match=f"injected {boundary}"):
            _save(run_dir, step=2, keep=3)

    assert injected
    assert first.latest_path.read_bytes() == pointer_before
    assert verify_checkpoint(first.checkpoint_path)["global_step"] == 1

    retried = _save(run_dir, step=2, keep=3)

    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 2
    assert json.loads(retried.latest_path.read_text(encoding="utf-8"))["step"] == 2


def test_first_checkpoint_claim_orphan_is_validated_and_adopted_without_overwrite(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    named_run = run_dir / "first-claim-orphan"
    checkpoint = named_run / "checkpoints" / "step-000000001.pt"
    original_claim = checkpoint_module._claim_file

    def fail_after_canonical_claim(staged: Path, canonical: Path) -> bool:
        claimed = original_claim(staged, canonical)
        if claimed and canonical == checkpoint:
            raise OSError("injected first canonical claim failure")
        return claimed

    with monkeypatch.context() as context:
        context.setattr(
            checkpoint_module, "_claim_file", fail_after_canonical_claim
        )
        with pytest.raises(OSError, match="first canonical claim"):
            _save(named_run, step=1)

    assert checkpoint.exists()
    assert not checkpoint.with_suffix(".json").exists()
    assert not (checkpoint.parent / "latest.json").exists()
    claimed_status = checkpoint.stat()
    claimed_bytes = checkpoint.read_bytes()

    retried = _save(named_run, step=1)

    assert os.path.samestat(claimed_status, checkpoint.stat())
    assert checkpoint.read_bytes() == claimed_bytes
    assert retried.checkpoint_path == checkpoint
    assert verify_checkpoint(checkpoint)["global_step"] == 1
    assert json.loads(retried.latest_path.read_text(encoding="utf-8"))["step"] == 1


def test_atomic_claim_never_overwrites_a_concurrent_winner(run_dir: Path) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir()
    staged_a = checkpoint_dir / ".a.tmp"
    staged_b = checkpoint_dir / ".b.tmp"
    canonical = checkpoint_dir / "step-000000001.pt"
    staged_a.write_bytes(b"writer-a")
    staged_b.write_bytes(b"writer-b")
    barrier = Barrier(2)

    def claim(staged: Path) -> bool:
        barrier.wait()
        return checkpoint_module._claim_file(staged, canonical)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(claim, (staged_a, staged_b)))

    assert sorted(results) == [False, True]
    assert canonical.read_bytes() in {b"writer-a", b"writer-b"}


def test_retry_never_adopts_a_different_same_step_payload(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save(run_dir, step=1, keep=3)
    original_write_json = checkpoint_module._atomic_write_json

    def fail_before_pointer(path: Path, value: object) -> None:
        if path.name == "latest.json":
            raise OSError("injected pointer failure")
        original_write_json(path, value)

    with monkeypatch.context() as context:
        context.setattr(checkpoint_module, "_atomic_write_json", fail_before_pointer)
        with pytest.raises(OSError, match="pointer"):
            _save(run_dir, step=2, keep=3)

    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=2)
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="different|committed"):
        save_checkpoint(
            run_dir,
            lineage_id="lineage-test",
            global_step=2,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            data_cursor=cursor,
            semantic_manifests=_manifests(),
            provenance={
                "git": {
                    "available": True,
                    "revision": "abc123",
                    "dirty": False,
                }
            },
            parameter_ledger=model.parameter_ledger(),
            counters={"optimizer_steps": 2, "batches_seen": 2},
            last_metrics={
                "loss": 1.0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            },
            keep_last_checkpoints=3,
        )


def test_retry_reclaims_an_invalid_unreferenced_same_step_artifact(
    run_dir: Path,
) -> None:
    first = _save(run_dir, step=1, keep=3)
    checkpoint = first.checkpoint_path.parent / "step-000000002.pt"
    sidecar = checkpoint.with_suffix(".json")
    checkpoint.write_bytes(b"invalid")
    sidecar.write_text("{}", encoding="utf-8")

    retried = _save(run_dir, step=2, keep=3)

    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 2


@pytest.mark.parametrize("boundary", ("canonical-unlink", "canonical-sync"))
def test_first_checkpoint_retry_recovers_a_tombstone_only_crash(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    named_run = run_dir / f"first-invalid-{boundary}"
    checkpoint = named_run / "checkpoints" / "step-000000001.pt"
    tombstone = checkpoint.parent / f".{checkpoint.name}.invalid"
    original_claim = checkpoint_module._claim_file

    def orphan_first_claim(staged: Path, canonical: Path) -> bool:
        claimed = original_claim(staged, canonical)
        if claimed and canonical == checkpoint:
            raise OSError("injected first canonical orphan")
        return claimed

    with monkeypatch.context() as context:
        context.setattr(checkpoint_module, "_claim_file", orphan_first_claim)
        with pytest.raises(OSError, match="first canonical orphan"):
            _save(named_run, step=1)

    checkpoint.write_bytes(b"invalid first generation")
    original_unlink = Path.unlink
    original_fsync = checkpoint_module._fsync_directory
    injected = False

    def unlink_then_fail(path: Path, *args: object, **kwargs: object):
        nonlocal injected
        result = original_unlink(path, *args, **kwargs)
        if path == checkpoint and boundary == "canonical-unlink" and not injected:
            injected = True
            raise OSError("injected canonical-unlink failure")
        return result

    def fsync_then_fail(path: Path) -> None:
        nonlocal injected
        original_fsync(path)
        if (
            boundary == "canonical-sync"
            and tombstone.exists()
            and not checkpoint.exists()
            and not injected
        ):
            injected = True
            raise OSError("injected canonical-sync failure")

    with monkeypatch.context() as context:
        context.setattr(Path, "unlink", unlink_then_fail)
        context.setattr(checkpoint_module, "_fsync_directory", fsync_then_fail)
        with pytest.raises(OSError, match=boundary):
            _save(named_run, step=1)

    assert injected
    assert tombstone.exists()
    assert not checkpoint.exists()
    assert not (checkpoint.parent / "latest.json").exists()

    retried = _save(named_run, step=1)

    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 1
    assert not tombstone.exists()


@pytest.mark.parametrize("pointer_state", ("missing", "damaged"))
@pytest.mark.parametrize("material", ("invalid-canonical", "tombstone-only"))
def test_retry_reconciles_invalid_target_with_unusable_pointer(
    run_dir: Path, pointer_state: str, material: str
) -> None:
    first = _save(run_dir, step=1, keep=3)
    if pointer_state == "missing":
        first.latest_path.unlink()
    else:
        first.latest_path.write_bytes(b"{damaged pointer")
    checkpoint = first.checkpoint_path.parent / "step-000000002.pt"
    tombstone = checkpoint.parent / f".{checkpoint.name}.invalid"
    checkpoint.write_bytes(b"invalid next generation")
    if material == "tombstone-only":
        os.link(checkpoint, tombstone)
        checkpoint.unlink()

    retried = _save(run_dir, step=2, keep=3)

    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 2
    assert json.loads(retried.latest_path.read_text(encoding="utf-8"))["step"] == 2
    assert not tombstone.exists()


def test_retry_repairs_only_an_invalid_unreferenced_sidecar(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save(run_dir, step=1, keep=3)
    original_write_json = checkpoint_module._atomic_write_json

    def fail_before_pointer(path: Path, value: object) -> None:
        if path.name == "latest.json":
            raise OSError("injected pointer failure")
        original_write_json(path, value)

    with monkeypatch.context() as context:
        context.setattr(checkpoint_module, "_atomic_write_json", fail_before_pointer)
        with pytest.raises(OSError, match="pointer"):
            _save(run_dir, step=2, keep=3)
    sidecar = run_dir / "checkpoints" / "step-000000002.json"
    sidecar.write_text("{}", encoding="utf-8")

    retried = _save(run_dir, step=2, keep=3)

    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 2


def test_same_step_retry_ignores_changed_non_strict_git_context(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save(run_dir, step=1, keep=3)
    original_write_json = checkpoint_module._atomic_write_json

    def fail_before_pointer(path: Path, value: object) -> None:
        if path.name == "latest.json":
            raise OSError("injected pointer failure")
        original_write_json(path, value)

    with monkeypatch.context() as context:
        context.setattr(checkpoint_module, "_atomic_write_json", fail_before_pointer)
        with pytest.raises(OSError, match="pointer"):
            _save(run_dir, step=2, keep=3)

    retried = _save(
        run_dir,
        step=2,
        keep=3,
        provenance={
            "git": {"available": True, "revision": "changed", "dirty": True}
        },
    )

    assert verify_checkpoint(retried.checkpoint_path)["provenance"]["git"] == {
        "available": True,
        "revision": "abc123",
        "dirty": False,
    }


def test_retry_never_reclaims_a_pointer_referenced_invalid_artifact(
    run_dir: Path,
) -> None:
    record = _save(run_dir, step=1)
    record.checkpoint_path.write_bytes(b"invalid referenced bytes")
    artifact_before = record.checkpoint_path.read_bytes()
    pointer_before = record.latest_path.read_bytes()

    with pytest.raises(ValueError):
        _save(run_dir, step=1)

    assert record.checkpoint_path.read_bytes() == artifact_before
    assert record.latest_path.read_bytes() == pointer_before


def _malformed_optimizer(payload: dict[str, object]) -> None:
    payload["optimizer_state"]["param_groups"][0]["params"].append(999)


def _malformed_scheduler(payload: dict[str, object]) -> None:
    payload["scheduler_state"]["base_lrs"].append(9.0)


def _malformed_scheduler_scalar(payload: dict[str, object]) -> None:
    payload["scheduler_state"]["last_epoch"] = "not-an-integer"


def _malformed_optimizer_group(payload: dict[str, object]) -> None:
    payload["optimizer_state"]["param_groups"][0]["betas"] = "invalid"


def _malformed_python_rng(payload: dict[str, object]) -> None:
    state = payload["rng_state"]["python"]["state"]
    payload["rng_state"]["python"]["state"] = state[:-1]


def _nonfinite_optimizer(payload: dict[str, object]) -> None:
    optimizer_state = payload["optimizer_state"]
    parameter = optimizer_state["param_groups"][0]["params"][0]
    optimizer_state["state"][parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.tensor([float("nan")]),
        "exp_avg_sq": torch.tensor([0.0]),
    }


def _meta_optimizer(payload: dict[str, object]) -> None:
    optimizer_state = payload["optimizer_state"]
    parameter = optimizer_state["param_groups"][0]["params"][0]
    optimizer_state["state"][parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.empty(1, device="meta"),
        "exp_avg_sq": torch.tensor([0.0]),
    }


def _unsupported_optimizer_dtype(payload: dict[str, object]) -> None:
    optimizer_state = payload["optimizer_state"]
    parameter = optimizer_state["param_groups"][0]["params"][0]
    first_parameter = next(iter(payload["model_state"].values()))
    optimizer_state["state"][parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros(first_parameter.shape, dtype=torch.uint16),
        "exp_avg_sq": torch.zeros(first_parameter.shape, dtype=torch.uint16),
    }


@pytest.mark.parametrize(
    "mutation",
    (
        _malformed_optimizer,
        _malformed_scheduler,
        _malformed_scheduler_scalar,
        _malformed_optimizer_group,
        _malformed_python_rng,
        _nonfinite_optimizer,
        _meta_optimizer,
        _unsupported_optimizer_dtype,
    ),
    ids=(
        "optimizer",
        "scheduler-shape",
        "scheduler-scalar",
        "optimizer-group",
        "rng",
        "nonfinite",
        "meta",
        "unsupported-dtype",
    ),
)
def test_failed_load_never_mutates_live_components_or_global_rng(
    run_dir: Path,
    mutation,
) -> None:
    record = _save(run_dir)
    _rewrite_payload(record, mutation)
    model, optimizer, scheduler = _components()
    with torch.no_grad():
        next(model.parameters()).add_(0.75)
    before = (
        deepcopy(model.state_dict()),
        deepcopy(optimizer.state_dict()),
        deepcopy(scheduler.state_dict()),
    )
    rng_before = _rng_snapshot()

    with pytest.raises(ValueError):
        load_checkpoint(
            record.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
        )

    _assert_nested_equal(before[0], model.state_dict())
    _assert_nested_equal(before[1], optimizer.state_dict())
    _assert_nested_equal(before[2], scheduler.state_dict())
    _assert_rng_equal(rng_before, _rng_snapshot())


def test_injected_component_load_failure_is_preflighted_off_live_objects(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    with torch.no_grad():
        next(model.parameters()).add_(0.75)

    live_load_called = False

    def fail_load(
        target: torch.optim.lr_scheduler.LambdaLR, state: object
    ) -> None:
        nonlocal live_load_called
        if target is scheduler:
            live_load_called = True
        raise RuntimeError("injected component failure")

    monkeypatch.setattr(type(scheduler), "load_state_dict", fail_load)
    before = (
        deepcopy(model.state_dict()),
        deepcopy(optimizer.state_dict()),
        deepcopy(scheduler.state_dict()),
    )
    rng_before = _rng_snapshot()
    with pytest.raises(RuntimeError, match="injected component"):
        load_checkpoint(
            record.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
        )

    _assert_nested_equal(before[0], model.state_dict())
    _assert_nested_equal(before[1], optimizer.state_dict())
    _assert_nested_equal(before[2], scheduler.state_dict())
    _assert_rng_equal(rng_before, _rng_snapshot())
    assert not live_load_called


def test_phase_zero_checkpoint_rejects_scaler_and_omits_cuda_rng(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    payload = torch.load(record.checkpoint_path, map_location="cpu", weights_only=True)

    assert payload["scaler_state"] is None
    assert set(payload["rng_state"]) == {"python", "numpy", "torch_cpu"}

    class FakeScaler:
        def state_dict(self) -> dict[str, float]:
            return {"scale": 2.0}

    model, optimizer, scheduler = _components()
    with pytest.raises(ValueError, match="Phase 0|scaler"):
        save_checkpoint(
            run_dir / "scaler",
            lineage_id="lineage-test",
            global_step=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=FakeScaler(),
            data_cursor=DataCursor(SYNTHETIC_CURSOR_KIND, 2, 2, 8),
            semantic_manifests=_manifests(),
            provenance={
                "git": {
                    "available": True,
                    "revision": "abc123",
                    "dirty": False,
                }
            },
            parameter_ledger=model.parameter_ledger(),
            counters={"optimizer_steps": 1, "batches_seen": 1},
            last_metrics={"loss": 1.0},
            keep_last_checkpoints=2,
        )


def test_run_metadata_durably_binds_lineage_and_strict_manifests(
    run_dir: Path,
) -> None:
    record = _save(run_dir, step=1)
    metadata_path = record.checkpoint_path.parent / "run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload = verify_checkpoint(record.checkpoint_path)

    assert metadata == {
        "schema_version": 2,
        "lineage_id": "lineage-test",
        "manifest_hashes": payload["manifest_hashes"],
    }
    with pytest.raises(ValueError, match="lineage|run metadata"):
        _save(run_dir, step=2, lineage_id="foreign-lineage")
    changed = deepcopy(_manifests())
    changed["source"]["train.py"] = "f" * 64
    with pytest.raises(ValueError, match="run metadata"):
        _save(run_dir, step=2, manifests=changed)
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata
    assert not (record.checkpoint_path.parent / "step-000000002.pt").exists()


def test_pointer_field_mismatch_recovers_by_scanning_the_bound_lineage(
    run_dir: Path,
) -> None:
    first = _save(run_dir, step=1, keep=3)
    second = _save(run_dir, step=2, keep=3)
    pointer = json.loads(second.latest_path.read_text(encoding="utf-8"))
    pointer["checkpoint"] = first.checkpoint_path.name
    pointer["sidecar"] = first.sidecar_path.name
    second.latest_path.write_text(json.dumps(pointer), encoding="utf-8")
    model, optimizer, scheduler = _components()

    loaded = load_checkpoint(
        run_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
        expected_lineage_id="lineage-test",
        auto=True,
    )

    assert loaded.checkpoint_path.name == "step-000000002.pt"


@pytest.mark.parametrize("metadata_state", ("missing", "corrupt"))
def test_auto_recovery_fails_closed_without_trusted_run_metadata(
    run_dir: Path,
    metadata_state: str,
) -> None:
    record = _save(run_dir, step=1)
    metadata_path = record.checkpoint_path.parent / "run.json"
    record.latest_path.write_text("not-json", encoding="utf-8")
    if metadata_state == "missing":
        metadata_path.unlink(missing_ok=True)
    else:
        metadata_path.write_text("not-json", encoding="utf-8")
    model, optimizer, scheduler = _components()

    with pytest.raises(ValueError, match="run metadata"):
        load_checkpoint(
            run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
            auto=True,
        )


def test_recovery_filters_higher_foreign_and_incompatible_candidates(
    run_dir: Path,
) -> None:
    first = _save(run_dir, step=1, keep=4)
    second = _save(run_dir, step=2, keep=4)
    checkpoint_dir = second.checkpoint_path.parent
    foreign = _save(
        run_dir / "foreign", step=9, lineage_id="foreign-lineage", keep=4
    )
    changed = deepcopy(_manifests())
    changed["source"]["train.py"] = "f" * 64
    incompatible = _save(
        run_dir / "incompatible", step=8, manifests=changed, keep=4
    )
    _copy_candidate(foreign, checkpoint_dir, "step-000000009.pt")
    _copy_candidate(incompatible, checkpoint_dir, "step-000000008.pt")
    second.latest_path.write_text("not-json", encoding="utf-8")
    model, optimizer, scheduler = _components()

    loaded = load_checkpoint(
        run_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
        expected_lineage_id="lineage-test",
        auto=True,
    )

    assert loaded.checkpoint_path == second.checkpoint_path
    assert first.checkpoint_path.exists()


def test_recovery_rejects_filename_payload_step_mismatch(run_dir: Path) -> None:
    first = _save(run_dir, step=1, keep=3)
    second = _save(run_dir, step=2, keep=3)
    _copy_candidate(first, second.checkpoint_path.parent, "step-000000009.pt")
    second.latest_path.write_text("not-json", encoding="utf-8")
    model, optimizer, scheduler = _components()

    loaded = load_checkpoint(
        run_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
        auto=True,
    )

    assert loaded.checkpoint_path == second.checkpoint_path


def test_retention_counts_only_active_compatible_pairs(run_dir: Path) -> None:
    first = _save(run_dir, step=1, keep=4)
    second = _save(run_dir, step=2, keep=4)
    checkpoint_dir = second.checkpoint_path.parent
    foreign = _save(
        run_dir / "foreign-retention",
        step=9,
        lineage_id="foreign-lineage",
        keep=4,
    )
    changed = deepcopy(_manifests())
    changed["source"]["train.py"] = "f" * 64
    incompatible = _save(
        run_dir / "incompatible-retention",
        step=8,
        manifests=changed,
        keep=4,
    )
    foreign_copy, foreign_sidecar = _copy_candidate(
        foreign, checkpoint_dir, "step-000000009.pt"
    )
    incompatible_copy, incompatible_sidecar = _copy_candidate(
        incompatible, checkpoint_dir, "step-000000008.pt"
    )
    invalid = checkpoint_dir / "step-000000007.pt"
    invalid.write_bytes(b"invalid")
    invalid_sidecar = invalid.with_suffix(".json")
    invalid_sidecar.write_text("{}", encoding="utf-8")

    third = _save(run_dir, step=3, keep=2)

    assert not first.checkpoint_path.exists()
    assert not first.sidecar_path.exists()
    assert verify_checkpoint(second.checkpoint_path)["global_step"] == 2
    assert verify_checkpoint(third.checkpoint_path)["global_step"] == 3
    for path in (
        foreign_copy,
        foreign_sidecar,
        incompatible_copy,
        incompatible_sidecar,
        invalid,
        invalid_sidecar,
    ):
        assert path.exists()


def test_retention_never_deletes_the_current_pointer_target(run_dir: Path) -> None:
    first = _save(run_dir, step=1, keep=3)
    _save(run_dir, step=2, keep=3)
    _save(run_dir, step=3, keep=3)
    sidecar = json.loads(first.sidecar_path.read_text(encoding="utf-8"))
    pointer = dict(sidecar)
    pointer["sidecar"] = first.sidecar_path.name
    first.latest_path.write_text(json.dumps(pointer), encoding="utf-8")

    retried = _save(run_dir, step=1, keep=1)

    assert retried.checkpoint_path == first.checkpoint_path
    assert verify_checkpoint(first.checkpoint_path)["global_step"] == 1


def test_retention_tombstone_recovers_after_pair_deletion_failure(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _save(run_dir, step=1, keep=3)
    _save(run_dir, step=2, keep=3)
    original_link = checkpoint_module.os.link
    injected = False

    def fail_between_pair_operations(
        source: object, target: object, *args: object, **kwargs: object
    ) -> object:
        nonlocal injected
        if Path(target).name == ".step-000000001.pt.pruning" and not injected:
            injected = True
            raise OSError("injected pair deletion failure")
        return original_link(source, target, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(checkpoint_module.os, "link", fail_between_pair_operations)
        with pytest.raises(OSError, match="pair deletion"):
            _save(run_dir, step=3, keep=2)

    assert injected
    assert not first.sidecar_path.exists()
    assert first.checkpoint_path.exists()
    assert (first.sidecar_path.parent / ".step-000000001.json.pruning").exists()

    retried = _save(run_dir, step=3, keep=2)

    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 3
    assert not first.checkpoint_path.exists()
    assert not (first.sidecar_path.parent / ".step-000000001.json.pruning").exists()


def test_retention_marker_finishes_cleanup_after_tombstone_unlink_failure(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _save(run_dir, step=1, keep=3)
    _save(run_dir, step=2, keep=3)
    original_unlink = Path.unlink
    injected = False

    def fail_last_tombstone_unlink(path: Path, missing_ok: bool = False) -> None:
        nonlocal injected
        if path.name == ".step-000000001.pt.pruning" and not injected:
            injected = True
            raise OSError("injected tombstone unlink failure")
        original_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as context:
        context.setattr(Path, "unlink", fail_last_tombstone_unlink)
        with pytest.raises(OSError, match="tombstone unlink"):
            _save(run_dir, step=3, keep=2)

    assert injected
    assert not first.checkpoint_path.exists()
    assert not first.sidecar_path.exists()
    assert (first.checkpoint_path.parent / ".step-000000001.pt.pruning").exists()

    retried = _save(run_dir, step=3, keep=2)

    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 3
    assert not (first.checkpoint_path.parent / ".step-000000001.pt.pruning").exists()


def test_retention_syncs_tombstone_cleanup_before_removing_the_marker(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _save(run_dir, step=1, keep=3)
    _save(run_dir, step=2, keep=3)
    checkpoint_dir = first.checkpoint_path.parent
    marker = checkpoint_dir / ".step-000000001.pruning.json"
    checkpoint_tombstone = checkpoint_dir / ".step-000000001.pt.pruning"
    sidecar_tombstone = checkpoint_dir / ".step-000000001.json.pruning"
    original_fsync = checkpoint_module._fsync_directory
    injected = False

    def fail_cleanup_sync(path: Path) -> None:
        nonlocal injected
        original_fsync(path)
        if (
            marker.exists()
            and not first.checkpoint_path.exists()
            and not first.sidecar_path.exists()
            and not checkpoint_tombstone.exists()
            and not sidecar_tombstone.exists()
            and not injected
        ):
            injected = True
            raise OSError("injected pruning cleanup sync failure")

    with monkeypatch.context() as context:
        context.setattr(checkpoint_module, "_fsync_directory", fail_cleanup_sync)
        with pytest.raises(OSError, match="cleanup sync"):
            _save(run_dir, step=3, keep=2)

    assert injected
    assert marker.exists()
    retried = _save(run_dir, step=3, keep=2)
    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 3
    assert not marker.exists()


@pytest.mark.parametrize(
    "limit_name",
    (
        "PHASE0_MAX_CHECKPOINT_BYTES",
        "PHASE0_MAX_ARCHIVE_MEMBERS",
        "PHASE0_MAX_ARCHIVE_UNCOMPRESSED_BYTES",
    ),
)
def test_artifact_budgets_reject_before_torch_load(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
) -> None:
    record = _save(run_dir)
    monkeypatch.setattr(checkpoint_module, limit_name, 1, raising=False)
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="Phase 0|archive|budget|size"):
        checkpoint_module._load_payload(record.checkpoint_path)

    assert not called


def test_compressed_archive_is_rejected_before_torch_load(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = run_dir / "step-000000001.pt"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("archive/data.pkl", b"not a torch payload")
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="archive|compression"):
        checkpoint_module._load_payload(path)

    assert not called


def test_control_json_budget_is_enforced_before_decode(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)
    monkeypatch.setattr(
        checkpoint_module, "PHASE0_MAX_CONTROL_JSON_BYTES", 1, raising=False
    )

    with pytest.raises(ValueError, match="JSON|size|budget"):
        checkpoint_module._read_json(
            record.checkpoint_path.parent / "run.json", "run metadata"
        )


@pytest.mark.parametrize("target_kind", ("artifact", "directory", "control"))
def test_symlinked_checkpoint_paths_fail_closed(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    record = _save(run_dir)
    checkpoint_dir = record.checkpoint_path.parent
    target = {
        "artifact": record.checkpoint_path,
        "directory": checkpoint_dir,
        "control": checkpoint_dir / "run.json",
    }[target_kind]
    original_is_symlink = Path.is_symlink

    def reported_symlink(path: Path) -> bool:
        return path == target or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", reported_symlink)
    model, optimizer, scheduler = _components()
    if target_kind == "artifact":
        operation = lambda: verify_checkpoint(record.checkpoint_path)
    elif target_kind == "control":
        operation = lambda: load_checkpoint(
            run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
            auto=True,
        )
    else:
        operation = lambda: _save(run_dir, step=2)

    with pytest.raises(ValueError, match="symlink"):
        operation()


def test_explicit_load_is_bound_to_the_trusted_run_lineage_without_mutation(
    run_dir: Path,
) -> None:
    trusted = _save(run_dir / "trusted", step=1, lineage_id="trusted-lineage")
    foreign = _save(run_dir / "foreign", step=2, lineage_id="foreign-lineage")
    copied_checkpoint = trusted.checkpoint_path.parent / foreign.checkpoint_path.name
    copied_sidecar = copied_checkpoint.with_suffix(".json")
    shutil.copyfile(foreign.checkpoint_path, copied_checkpoint)
    shutil.copyfile(foreign.sidecar_path, copied_sidecar)
    model, optimizer, scheduler = _components()
    before = (
        deepcopy(model.state_dict()),
        deepcopy(optimizer.state_dict()),
        deepcopy(scheduler.state_dict()),
    )
    rng_before = _rng_snapshot()

    with pytest.raises(ValueError, match="lineage"):
        load_checkpoint(
            copied_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
        )

    _assert_nested_equal(before[0], model.state_dict())
    _assert_nested_equal(before[1], optimizer.state_dict())
    _assert_nested_equal(before[2], scheduler.state_dict())
    _assert_rng_equal(rng_before, _rng_snapshot())


@pytest.mark.parametrize("dtype", (torch.int64, torch.float32))
def test_logically_huge_expanded_tensor_is_rejected_before_elementwise_work(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    record = _save(run_dir)

    def mutate(payload: dict[str, object]) -> None:
        optimizer_state = payload["optimizer_state"]
        parameter = optimizer_state["param_groups"][0]["params"][0]
        optimizer_state["state"][parameter]["exp_avg"] = torch.zeros(
            1, dtype=dtype
        ).expand(1_000_000, 1_000_000)

    _rewrite_payload(record, mutate)
    original_isfinite = checkpoint_module.torch.isfinite

    def guarded_isfinite(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 1_000_000_000_000:
            raise AssertionError("elementwise validation reached the huge tensor")
        return original_isfinite(tensor)

    monkeypatch.setattr(checkpoint_module.torch, "isfinite", guarded_isfinite)
    with pytest.raises(ValueError, match="logical|tensor|stride|bounds"):
        verify_checkpoint(record.checkpoint_path)


def test_zero_stride_float_view_is_rejected_before_finiteness_probe(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)

    def mutate(payload: dict[str, object]) -> None:
        optimizer_state = payload["optimizer_state"]
        parameter = optimizer_state["param_groups"][0]["params"][0]
        optimizer_state["state"][parameter]["exp_avg"] = torch.zeros(1).expand(
            32, 32
        )

    _rewrite_payload(record, mutate)
    original_isfinite = checkpoint_module.torch.isfinite

    def guarded_isfinite(tensor: torch.Tensor) -> torch.Tensor:
        expanded = any(
            size > 1 and stride == 0
            for size, stride in zip(tensor.shape, tensor.stride())
        )
        if expanded:
            raise AssertionError("finiteness probe reached an expanded tensor")
        return original_isfinite(tensor)

    monkeypatch.setattr(checkpoint_module.torch, "isfinite", guarded_isfinite)
    with pytest.raises(ValueError, match="stride|tensor|bounds"):
        verify_checkpoint(record.checkpoint_path)


@pytest.mark.parametrize(
    "dtype", (torch.int16, torch.float16, torch.float64, torch.complex64), ids=str
)
def test_phase_zero_rejects_tensor_dtypes_outside_the_runtime_surface(
    run_dir: Path,
    dtype: torch.dtype,
) -> None:
    record = _save(run_dir)

    def mutate(payload: dict[str, object]) -> None:
        optimizer_state = payload["optimizer_state"]
        parameter = optimizer_state["param_groups"][0]["params"][0]
        original = optimizer_state["state"][parameter]["exp_avg"]
        optimizer_state["state"][parameter]["exp_avg"] = torch.zeros(
            original.shape, dtype=dtype
        )

    _rewrite_payload(record, mutate)
    with pytest.raises(ValueError, match="tensor|dtype|Phase 0"):
        verify_checkpoint(record.checkpoint_path)


def test_sidecar_integrity_is_checked_before_torch_load(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)
    _rewrite_sidecar(record, sha256="0" * 64)
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load ran before sidecar integrity")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="sha256"):
        verify_checkpoint(record.checkpoint_path)
    assert not called


def test_oversized_sidecar_declaration_is_rejected_before_hashing(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)
    with record.checkpoint_path.open("r+b") as handle:
        handle.truncate(checkpoint_module.PHASE0_MAX_CHECKPOINT_BYTES + 1)
    _rewrite_sidecar(
        record,
        sha256="0" * 64,
        byte_size=record.checkpoint_path.stat().st_size,
    )

    def forbidden_hash(path: Path) -> str:
        raise AssertionError("oversized artifact was hashed")

    monkeypatch.setattr(checkpoint_module, "_sha256", forbidden_hash)
    with pytest.raises(ValueError, match="size|budget|Phase 0"):
        verify_checkpoint(record.checkpoint_path)


def test_zip_entry_budget_is_checked_before_zipfile_construction(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)
    with zipfile.ZipFile(
        record.checkpoint_path, "w", compression=zipfile.ZIP_STORED
    ) as archive:
        for index in range(20_000):
            archive.writestr(f"entry-{index:05d}", b"")
    _rewrite_sidecar(
        record,
        sha256=_sha256(record.checkpoint_path),
        byte_size=record.checkpoint_path.stat().st_size,
    )

    def forbidden_zipfile(*args: object, **kwargs: object) -> object:
        raise AssertionError("ZipFile was constructed before the EOCD budget check")

    monkeypatch.setattr(checkpoint_module.zipfile, "ZipFile", forbidden_zipfile)
    with pytest.raises(ValueError, match="central|entry|member|budget"):
        verify_checkpoint(record.checkpoint_path)


def test_forged_small_eocd_count_is_rejected_before_zipfile_construction(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)
    with zipfile.ZipFile(
        record.checkpoint_path, "w", compression=zipfile.ZIP_STORED
    ) as archive:
        for index in range(20_000):
            archive.writestr(f"entry-{index:05d}", b"")
    encoded = bytearray(record.checkpoint_path.read_bytes())
    eocd = encoded.rfind(b"PK\x05\x06")
    assert eocd >= 0
    struct.pack_into("<H", encoded, eocd + 8, 1)
    struct.pack_into("<H", encoded, eocd + 10, 1)
    record.checkpoint_path.write_bytes(encoded)
    _rewrite_sidecar(
        record,
        sha256=_sha256(record.checkpoint_path),
        byte_size=record.checkpoint_path.stat().st_size,
    )

    def forbidden_zipfile(*args: object, **kwargs: object) -> object:
        raise AssertionError("ZipFile materialized a forged central directory")

    monkeypatch.setattr(checkpoint_module.zipfile, "ZipFile", forbidden_zipfile)
    with pytest.raises(ValueError, match="central|entry|member|budget"):
        verify_checkpoint(record.checkpoint_path)


def test_pickle_container_bomb_is_rejected_before_torch_load(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)
    torch.save({"junk": [{} for _ in range(300_000)]}, record.checkpoint_path)
    _rewrite_sidecar(
        record,
        sha256=_sha256(record.checkpoint_path),
        byte_size=record.checkpoint_path.stat().st_size,
    )
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load ran before pickle graph preflight")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="pickle|opcode|container|graph|budget"):
        verify_checkpoint(record.checkpoint_path)
    assert not called


def test_pickle_oversized_tuple_is_rejected_before_torch_load(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)
    torch.save(
        {"junk": tuple(None for _ in range(
            checkpoint_module.PHASE0_MAX_CONTAINER_ENTRIES + 1
        ))},
        record.checkpoint_path,
    )
    _rewrite_sidecar(
        record,
        sha256=_sha256(record.checkpoint_path),
        byte_size=record.checkpoint_path.stat().st_size,
    )
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load reached an oversized tuple")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="checkpoint|pickle|tuple|container|graph|budget"):
        verify_checkpoint(record.checkpoint_path)
    assert not called


def test_pickle_deep_append_graph_is_rejected_before_torch_load(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)
    nested: object = None
    for _ in range(checkpoint_module.PHASE0_MAX_STATE_DEPTH + 2):
        nested = [nested]
    torch.save({"junk": nested}, record.checkpoint_path)
    _rewrite_sidecar(
        record,
        sha256=_sha256(record.checkpoint_path),
        byte_size=record.checkpoint_path.stat().st_size,
    )
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load reached a deep APPEND graph")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="checkpoint|pickle|depth|nesting|graph|budget"):
        verify_checkpoint(record.checkpoint_path)
    assert not called


def test_pickle_tensor_rebuild_count_is_bounded_before_torch_load(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)
    entries = _archive_entries(record.checkpoint_path)
    data_index = next(
        index for index, (name, _) in enumerate(entries) if name.endswith("/data.pkl")
    )
    data_name, _ = entries[data_index]
    archive_root = data_name.removesuffix("/data.pkl")
    entries[data_index] = (
        data_name,
        _shared_empty_tensor_pickle(checkpoint_module.PHASE0_MAX_TENSORS + 1),
    )
    entries = [
        entry for entry in entries if not entry[0].startswith(f"{archive_root}/data/")
    ]
    entries.append((f"{archive_root}/data/0", b""))
    _replace_archive(record.checkpoint_path, entries)
    _rewrite_sidecar(
        record,
        sha256=_sha256(record.checkpoint_path),
        byte_size=record.checkpoint_path.stat().st_size,
    )
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load reached excessive tensor rebuilds")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="checkpoint|pickle|tensor|rebuild|budget"):
        verify_checkpoint(record.checkpoint_path)
    assert not called


def test_pickle_archive_rejects_an_ignored_same_root_member_before_torch_load(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir)
    entries = _archive_entries(record.checkpoint_path)
    data_name = next(name for name, _ in entries if name.endswith("/data.pkl"))
    archive_root = data_name.removesuffix("/data.pkl")
    entries.append((f"{archive_root}/ignored", b"ignored"))
    _replace_archive(record.checkpoint_path, entries)
    _rewrite_sidecar(
        record,
        sha256=_sha256(record.checkpoint_path),
        byte_size=record.checkpoint_path.stat().st_size,
    )
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load reached an unapproved archive member")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="archive|member|pickle|checkpoint"):
        verify_checkpoint(record.checkpoint_path)
    assert not called


@pytest.mark.parametrize(
    ("member_suffix", "canonical", "replacement"),
    (
        ("/byteorder", b"little", b"big"),
        ("/version", b"3\n", b"2\n"),
        ("/.format_version", b"1", b"2"),
        ("/.storage_alignment", b"64", b"32"),
        ("/.data/serialization_id", None, b"x" * 40),
    ),
)
def test_pickle_archive_control_records_are_exact_before_torch_load(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    member_suffix: str,
    canonical: bytes | None,
    replacement: bytes,
) -> None:
    record = _save(run_dir)
    entries = _archive_entries(record.checkpoint_path)
    index = next(
        index for index, (name, _) in enumerate(entries) if name.endswith(member_suffix)
    )
    name, value = entries[index]
    if canonical is None:
        assert len(value) == 40 and value.isascii() and value.isdigit()
    else:
        assert value == canonical
    entries[index] = (name, replacement)
    _replace_archive(record.checkpoint_path, entries)
    _rewrite_sidecar(
        record,
        sha256=_sha256(record.checkpoint_path),
        byte_size=record.checkpoint_path.stat().st_size,
    )
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load reached an invalid archive control record")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="checkpoint|archive|control|record|member"):
        verify_checkpoint(record.checkpoint_path)
    assert not called


@pytest.mark.parametrize(
    "attack",
    (
        "four-tib-numel",
        "negative-numel",
        "invalid-tag",
        "invalid-key",
        "non-cpu-location",
        "duplicate-key",
        "missing-member",
        "duplicate-member",
        "mismatched-member-size",
        "huge-storage-offset",
        "huge-tensor-dimension",
        "huge-tensor-stride",
    ),
)
def test_pickle_storage_and_tensor_references_are_bound_before_torch_load(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    record = _save(run_dir, step=1)
    entries = _archive_entries(record.checkpoint_path)
    data_index = next(
        index for index, (name, _) in enumerate(entries) if name.endswith("/data.pkl")
    )
    data_name, pickle_value = entries[data_index]
    operations, descriptors = _storage_pid_ops(pickle_value)
    assert len(descriptors) >= 2
    pid_index, key_index, numel_index = descriptors[0]
    key = operations[key_index][1]
    assert isinstance(key, str)
    archive_root = data_name.removesuffix("/data.pkl")
    member_name = f"{archive_root}/data/{key}"
    member_index = next(
        index for index, (name, _) in enumerate(entries) if name == member_name
    )
    if attack == "four-tib-numel":
        pickle_value = _replace_pickle_op(
            pickle_value, operations, numel_index, _pickle_value(1 << 40)
        )
    elif attack == "negative-numel":
        pickle_value = _replace_pickle_op(
            pickle_value, operations, numel_index, _pickle_value(-1)
        )
    elif attack == "invalid-tag":
        tag_index = next(
            index
            for index in range(pid_index)
            if operations[index][0].name == "BINUNICODE"
            and operations[index][1] == "storage"
        )
        pickle_value = _replace_pickle_op(
            pickle_value, operations, tag_index, _pickle_unicode("storagx")
        )
    elif attack == "invalid-key":
        pickle_value = _replace_pickle_op(
            pickle_value, operations, key_index, _pickle_unicode("x")
        )
    elif attack == "non-cpu-location":
        location_index = next(
            index
            for index in range(pid_index)
            if operations[index][0].name == "BINUNICODE"
            and operations[index][1] == "cpu"
        )
        pickle_value = _replace_pickle_op(
            pickle_value, operations, location_index, _pickle_unicode("gpu")
        )
    elif attack == "duplicate-key":
        second_key_index = descriptors[1][1]
        pickle_value = _replace_pickle_op(
            pickle_value, operations, second_key_index, _pickle_unicode(key)
        )
    elif attack == "missing-member":
        entries.pop(member_index)
    elif attack == "duplicate-member":
        entries.append(entries[member_index])
    elif attack == "mismatched-member-size":
        entries[member_index] = (member_name, entries[member_index][1] + b"\0")
    else:
        integer_indices = [
            index
            for index in range(pid_index + 1, len(operations))
            if operations[index][0].name
            in {"BININT", "BININT1", "BININT2", "LONG1"}
        ]
        target = {
            "huge-storage-offset": integer_indices[0],
            "huge-tensor-dimension": integer_indices[1],
            "huge-tensor-stride": integer_indices[3],
        }[attack]
        pickle_value = _replace_pickle_op(
            pickle_value, operations, target, _pickle_value(1 << 40)
        )
    entries[data_index] = (data_name, pickle_value)
    if attack == "duplicate-member":
        with pytest.warns(UserWarning, match="Duplicate name"):
            _replace_archive(record.checkpoint_path, entries)
    else:
        _replace_archive(record.checkpoint_path, entries)
    _rewrite_sidecar(
        record,
        sha256=_sha256(record.checkpoint_path),
        byte_size=record.checkpoint_path.stat().st_size,
    )
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load reached a forged storage reference")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(
        ValueError, match="checkpoint|storage|pickle|archive|member|tensor|budget"
    ):
        verify_checkpoint(record.checkpoint_path)
    assert not called


def test_pickle_storage_aggregate_budget_is_enforced_before_torch_load(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _save(run_dir, step=1)
    entries = _archive_entries(record.checkpoint_path)
    total_storage_bytes = sum(
        len(value) for name, value in entries if "/data/" in name
    )
    assert total_storage_bytes > 1
    monkeypatch.setattr(
        checkpoint_module,
        "PHASE0_MAX_TOTAL_TENSOR_LOGICAL_BYTES",
        total_storage_bytes - 1,
    )
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load reached aggregate storage overflow")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="storage|tensor|pickle|budget|checkpoint"):
        verify_checkpoint(record.checkpoint_path)
    assert not called


@pytest.mark.parametrize(
    "attack",
    (
        "requires-grad",
        "hooks-list",
        "hooks-dict",
        "hooks-tensor",
        "hooks-nonempty",
    ),
)
def test_pickle_tensor_rebuild_metadata_is_exact_before_torch_load(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    record = _save(run_dir, step=1)
    entries = _archive_entries(record.checkpoint_path)
    data_index = next(
        index for index, (name, _) in enumerate(entries) if name.endswith("/data.pkl")
    )
    data_name, pickle_value = entries[data_index]
    operations = list(pickletools.genops(pickle_value))
    reductions = [
        index
        for index, (opcode, _, _) in enumerate(operations)
        if opcode.name == "REDUCE"
    ]
    assert len(reductions) >= 4
    first_tensor_reduce = reductions[1]
    second_hooks_reduce = reductions[2]
    hooks_start = second_hooks_reduce - 2
    assert [
        operations[index][0].name
        for index in range(hooks_start, second_hooks_reduce + 1)
    ] == ["BINGET", "EMPTY_TUPLE", "REDUCE"]
    if attack == "requires-grad":
        requires_grad = next(
            index
            for index in range(second_hooks_reduce - 1, first_tensor_reduce, -1)
            if operations[index][0].name == "NEWFALSE"
        )
        pickle_value = _replace_pickle_op(
            pickle_value, operations, requires_grad, _pickle_value(True)
        )
    else:
        replacement = {
            "hooks-list": b"]",
            "hooks-dict": b"}",
            "hooks-nonempty": b"}K\x01K\x01s",
        }.get(attack)
        if attack == "hooks-tensor":
            memo_opcode, memo_index, _ = operations[first_tensor_reduce + 1]
            assert memo_opcode.name in {"BINPUT", "LONG_BINPUT"}
            assert isinstance(memo_index, int)
            replacement = _pickle_memo_get(memo_index)
        assert replacement is not None
        pickle_value = _replace_pickle_ops(
            pickle_value,
            operations,
            hooks_start,
            second_hooks_reduce + 1,
            replacement,
        )
    entries[data_index] = (data_name, pickle_value)
    _replace_archive(record.checkpoint_path, entries)
    _rewrite_sidecar(
        record,
        sha256=_sha256(record.checkpoint_path),
        byte_size=record.checkpoint_path.stat().st_size,
    )
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load reached malformed tensor rebuild metadata")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="pickle|tensor|rebuild|hook|checkpoint"):
        verify_checkpoint(record.checkpoint_path)
    assert not called


def test_pickle_repeated_storage_key_binds_the_exact_storage_type(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _save(run_dir, step=1)
    torch.save(
        {
            "byte": torch.zeros(1, dtype=torch.uint8),
            "boolean": torch.zeros(1, dtype=torch.bool),
        },
        record.checkpoint_path,
    )
    entries = _archive_entries(record.checkpoint_path)
    data_index = next(
        index for index, (name, _) in enumerate(entries) if name.endswith("/data.pkl")
    )
    data_name, pickle_value = entries[data_index]
    operations, descriptors = _storage_pid_ops(pickle_value)
    assert len(descriptors) == 2
    first_key = operations[descriptors[0][1]][1]
    second_key = operations[descriptors[1][1]][1]
    assert isinstance(first_key, str) and isinstance(second_key, str)
    pickle_value = _replace_pickle_op(
        pickle_value,
        operations,
        descriptors[1][1],
        _pickle_unicode(first_key),
    )
    archive_root = data_name.removesuffix("/data.pkl")
    entries[data_index] = (data_name, pickle_value)
    entries = [
        entry
        for entry in entries
        if entry[0] != f"{archive_root}/data/{second_key}"
    ]
    _replace_archive(record.checkpoint_path, entries)
    _rewrite_sidecar(
        record,
        sha256=_sha256(record.checkpoint_path),
        byte_size=record.checkpoint_path.stat().st_size,
    )
    called = False

    def forbidden_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("torch.load reached a cross-type storage-key collision")

    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)
    with pytest.raises(ValueError, match="pickle|storage|type|checkpoint"):
        verify_checkpoint(record.checkpoint_path)
    assert not called


@pytest.mark.parametrize(
    "mutation",
    ("multi-disk", "classic-sentinel", "central-budget", "zip64-mismatch"),
)
def test_zip_trailer_mismatches_are_rejected_before_zipfile_construction(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    record = _save(run_dir)
    encoded = bytearray(record.checkpoint_path.read_bytes())
    eocd = encoded.rfind(b"PK\x05\x06")
    zip64 = encoded.rfind(b"PK\x06\x06", 0, eocd)
    assert eocd >= 0 and zip64 >= 0
    if mutation == "multi-disk":
        struct.pack_into("<H", encoded, eocd + 4, 1)
    elif mutation == "classic-sentinel":
        struct.pack_into("<H", encoded, eocd + 10, 0xFFFF)
    elif mutation == "central-budget":
        struct.pack_into(
            "<L",
            encoded,
            eocd + 12,
            checkpoint_module.PHASE0_MAX_CENTRAL_DIRECTORY_BYTES + 1,
        )
    else:
        current_entries = struct.unpack_from("<Q", encoded, zip64 + 32)[0]
        struct.pack_into("<Q", encoded, zip64 + 32, current_entries + 1)
    record.checkpoint_path.write_bytes(encoded)
    _rewrite_sidecar(
        record,
        sha256=_sha256(record.checkpoint_path),
        byte_size=record.checkpoint_path.stat().st_size,
    )

    def forbidden_zipfile(*args: object, **kwargs: object) -> object:
        raise AssertionError("ZipFile was constructed before EOCD rejection")

    monkeypatch.setattr(checkpoint_module.zipfile, "ZipFile", forbidden_zipfile)
    with pytest.raises(ValueError, match="ZIP|central|entry|budget|disk|trailer"):
        verify_checkpoint(record.checkpoint_path)


@pytest.mark.parametrize("operation", ("auto", "retention"))
def test_candidate_count_budget_includes_invalid_high_steps(
    run_dir: Path,
    operation: str,
) -> None:
    record = _save(run_dir, step=1, keep=100)
    for step in range(100, 165):
        (record.checkpoint_path.parent / f"step-{step:09d}.pt").write_bytes(b"x")

    if operation == "auto":
        record.latest_path.write_text("not-json", encoding="utf-8")
        model, optimizer, scheduler = _components()
        with pytest.raises(ValueError, match="candidate|scan|budget"):
            load_checkpoint(
                run_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                expected_manifests=_manifests(),
                auto=True,
            )
    else:
        with pytest.raises(ValueError, match="candidate|scan|budget"):
            _save(run_dir, step=2, keep=100)


@pytest.mark.parametrize("candidate_part", ("artifact", "sidecar"))
def test_candidate_byte_budget_counts_incompatible_files(
    run_dir: Path,
    candidate_part: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir, step=1, keep=100)
    base_size = record.checkpoint_path.stat().st_size + record.sidecar_path.stat().st_size
    monkeypatch.setattr(checkpoint_module, "PHASE0_MAX_SCAN_BYTES", base_size + 30)
    candidate_size = 11
    for step in range(100, 103):
        path = record.checkpoint_path.parent / f"step-{step:09d}.pt"
        sized_path = path if candidate_part == "artifact" else path.with_suffix(".json")
        path.write_bytes(b"x")
        with sized_path.open("wb") as handle:
            handle.seek(candidate_size - 1)
            handle.write(b"\0")
    record.latest_path.write_text("not-json", encoding="utf-8")
    model, optimizer, scheduler = _components()

    with pytest.raises(ValueError, match="candidate|scan|byte|budget"):
        load_checkpoint(
            run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
            auto=True,
        )


def test_scan_budget_can_admit_one_maximum_artifact_and_sidecar() -> None:
    assert checkpoint_module.PHASE0_MAX_SCAN_BYTES >= (
        3
        * (
            checkpoint_module.PHASE0_MAX_CHECKPOINT_BYTES
            + checkpoint_module.PHASE0_MAX_CONTROL_JSON_BYTES
        )
    )


def _empty_optimizer_state(payload: dict[str, object]) -> None:
    payload["optimizer_state"]["state"].clear()


def _vector_optimizer_step(payload: dict[str, object]) -> None:
    state = payload["optimizer_state"]["state"]
    state[next(iter(state))]["step"] = torch.tensor([float(payload["global_step"])])


def _stale_scheduler_progress(payload: dict[str, object]) -> None:
    payload["scheduler_state"]["last_epoch"] = payload["global_step"] - 1


def _wrong_adamw_betas(payload: dict[str, object]) -> None:
    payload["optimizer_state"]["param_groups"][0]["betas"] = (0.8, 0.9)


def _wrong_adamw_epsilon(payload: dict[str, object]) -> None:
    payload["optimizer_state"]["param_groups"][0]["eps"] = 1e-7


def _missing_active_optimizer_state(payload: dict[str, object]) -> None:
    state = payload["optimizer_state"]["state"]
    state.pop(next(iter(state)))


def _wrong_scalar_optimizer_step(payload: dict[str, object]) -> None:
    state = payload["optimizer_state"]["state"]
    state[next(iter(state))]["step"] = torch.tensor(
        float(payload["global_step"] - 1)
    )


def _stale_scheduler_step_count(payload: dict[str, object]) -> None:
    payload["scheduler_state"]["_step_count"] = payload["global_step"]


def _wrong_moment_shape(payload: dict[str, object]) -> None:
    state = payload["optimizer_state"]["state"]
    parameter_state = state[next(iter(state))]
    size = parameter_state["exp_avg"].numel() + 1
    parameter_state["exp_avg"] = torch.zeros(size, dtype=torch.float32)
    parameter_state["exp_avg_sq"] = torch.zeros(size, dtype=torch.float32)


def _tied_alias_mismatch(payload: dict[str, object]) -> None:
    state = payload["model_state"]
    state["output_projection.weight"] = (
        state["output_projection.weight"].clone() + 1.0
    )


def _shared_optimizer_storage(payload: dict[str, object]) -> None:
    state = payload["optimizer_state"]["state"]
    first, second = tuple(state)[:2]
    state[second]["step"] = state[first]["step"]


def _within_state_moment_alias(payload: dict[str, object]) -> None:
    state = payload["optimizer_state"]["state"]
    parameter_state = state[next(iter(state))]
    parameter_state["exp_avg_sq"] = parameter_state["exp_avg"]


def _cross_state_moment_alias(payload: dict[str, object]) -> None:
    state = payload["optimizer_state"]["state"]
    by_shape: dict[tuple[int, ...], list[int]] = {}
    for parameter, parameter_state in state.items():
        by_shape.setdefault(tuple(parameter_state["exp_avg"].shape), []).append(
            parameter
        )
    first, second = next(ids[:2] for ids in by_shape.values() if len(ids) >= 2)
    state[second]["exp_avg"] = state[first]["exp_avg"]


def _extra_cursor_key(payload: dict[str, object]) -> None:
    payload["data_cursor"]["unexpected"] = 1


def _extra_counter_key(payload: dict[str, object]) -> None:
    payload["counters"]["unexpected"] = 1


def _extra_metric_key(payload: dict[str, object]) -> None:
    payload["last_metrics"]["unexpected"] = 1.0


@pytest.mark.parametrize(
    "mutation",
    (
        _empty_optimizer_state,
        _vector_optimizer_step,
        _wrong_scalar_optimizer_step,
        _missing_active_optimizer_state,
        _wrong_adamw_betas,
        _wrong_adamw_epsilon,
        _stale_scheduler_progress,
        _stale_scheduler_step_count,
    ),
    ids=(
        "empty-optimizer",
        "vector-step",
        "wrong-scalar-step",
        "missing-active-state",
        "wrong-betas",
        "wrong-epsilon",
        "stale-scheduler",
        "stale-scheduler-step-count",
    ),
)
def test_phase_zero_progress_evidence_is_bound_to_global_step_without_mutation(
    run_dir: Path,
    mutation,
) -> None:
    record = _save(run_dir)
    _rewrite_payload(record, mutation)
    model, optimizer, scheduler = _components()
    before = (
        deepcopy(model.state_dict()),
        deepcopy(optimizer.state_dict()),
        deepcopy(scheduler.state_dict()),
    )
    rng_before = _rng_snapshot()

    with pytest.raises(
        ValueError, match="optimizer|scheduler|step|progress|AdamW|parameter|coverage"
    ):
        load_checkpoint(
            record.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
        )

    _assert_nested_equal(before[0], model.state_dict())
    _assert_nested_equal(before[1], optimizer.state_dict())
    _assert_nested_equal(before[2], scheduler.state_dict())
    _assert_rng_equal(rng_before, _rng_snapshot())


def _ledger_extra_key(payload: dict[str, object]) -> None:
    payload["parameter_ledger"]["unexpected"] = "value"


def _ledger_missing_subsystem(payload: dict[str, object]) -> None:
    payload["parameter_ledger"]["entries"][0].pop("subsystem")


def _ledger_wrong_alias(payload: dict[str, object]) -> None:
    payload["parameter_ledger"]["entries"][0]["aliases"] = ["unrelated.weight"]


def _ledger_wrong_total(payload: dict[str, object]) -> None:
    ledger = payload["parameter_ledger"]
    ledger["entries"][0]["trainable_parameters"] += 1
    ledger["total_trainable"] += 1


def _ledger_extra_model_schema_key(payload: dict[str, object]) -> None:
    schema = payload["parameter_ledger"]["model_state_schema"]
    schema[next(iter(schema))]["unexpected"] = 1


@pytest.mark.parametrize(
    "mutation",
    (
        _ledger_extra_key,
        _ledger_missing_subsystem,
        _ledger_wrong_alias,
        _ledger_wrong_total,
        _ledger_extra_model_schema_key,
    ),
    ids=(
        "extra-key",
        "missing-subsystem",
        "wrong-alias",
        "wrong-total",
        "extra-model-schema-key",
    ),
)
def test_parameter_ledger_is_exactly_bound_to_the_constructed_model_without_mutation(
    run_dir: Path,
    mutation,
) -> None:
    record = _save(run_dir)
    _rewrite_payload(record, mutation)
    model, optimizer, scheduler = _components()
    before = (
        deepcopy(model.state_dict()),
        deepcopy(optimizer.state_dict()),
        deepcopy(scheduler.state_dict()),
    )
    rng_before = _rng_snapshot()

    with pytest.raises(ValueError, match="ledger|model-state|parameter"):
        load_checkpoint(
            record.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
        )

    _assert_nested_equal(before[0], model.state_dict())
    _assert_nested_equal(before[1], optimizer.state_dict())
    _assert_nested_equal(before[2], scheduler.state_dict())
    _assert_rng_equal(rng_before, _rng_snapshot())


@pytest.mark.parametrize(
    "mutation",
    (
        _missing_active_optimizer_state,
        _wrong_moment_shape,
        _ledger_wrong_alias,
        _tied_alias_mismatch,
        _shared_optimizer_storage,
        _within_state_moment_alias,
        _cross_state_moment_alias,
        _extra_cursor_key,
        _extra_counter_key,
        _extra_metric_key,
    ),
    ids=(
        "missing-active-state",
        "wrong-moment-shape",
        "foreign-ledger-alias",
        "tied-alias-mismatch",
        "shared-optimizer-storage",
        "within-state-moment-alias",
        "cross-state-moment-alias",
        "extra-cursor-key",
        "extra-counter-key",
        "extra-metric-key",
    ),
)
def test_verify_and_load_share_the_exact_resumability_contract(
    run_dir: Path,
    mutation,
) -> None:
    record = _save(run_dir)
    _rewrite_payload(record, mutation)

    with pytest.raises(ValueError):
        verify_checkpoint(record.checkpoint_path)
    model, optimizer, scheduler = _components()
    _assert_failed_load_is_nonmutating(
        record, model, optimizer, scheduler, match=".*"
    )


def test_verify_rejects_model_manifest_drift_bound_to_saved_model_evidence(
    run_dir: Path,
) -> None:
    record = _save(run_dir)

    def mutate(payload: dict[str, object]) -> None:
        payload["semantic_manifests"]["model"]["dropout"] = 0.7
        payload["manifest_hashes"] = checkpoint_module._manifest_hashes(
            payload["semantic_manifests"]
        )

    _rewrite_payload(record, mutate)
    payload = torch.load(record.checkpoint_path, map_location="cpu", weights_only=True)
    _rewrite_sidecar(record, manifest_hashes=payload["manifest_hashes"])

    with pytest.raises(ValueError, match="model|config|manifest"):
        verify_checkpoint(
            record.checkpoint_path,
            expected_manifests=payload["semantic_manifests"],
        )


def test_verify_bounds_manifest_model_allocation_before_cpu_construction(
    run_dir: Path,
) -> None:
    record = _save(run_dir)

    def mutate(payload: dict[str, object]) -> None:
        model_manifest = payload["semantic_manifests"]["model"]
        model_manifest.update(
            {
                "vocab_size": 1 << 32,
                "d_model": 1 << 20,
                "n_heads": 1,
                "ffn_dim": 1 << 20,
            }
        )
        payload["parameter_ledger"]["model_config"] = deepcopy(model_manifest)
        payload["manifest_hashes"] = checkpoint_module._manifest_hashes(
            payload["semantic_manifests"]
        )

    _rewrite_payload(record, mutate)
    payload = torch.load(record.checkpoint_path, map_location="cpu", weights_only=True)
    _rewrite_sidecar(record, manifest_hashes=payload["manifest_hashes"])

    with pytest.raises(ValueError, match="model|dimension|budget|Phase 0"):
        verify_checkpoint(
            record.checkpoint_path,
            expected_manifests=payload["semantic_manifests"],
        )


def test_valid_loaded_optimizer_step_cannot_share_counter_updates(
    run_dir: Path,
) -> None:
    record = _save(run_dir)
    model, optimizer, scheduler = _components()
    loaded = load_checkpoint(
        record.checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
    )
    for parameter in optimizer.state:
        parameter.grad = torch.zeros_like(parameter)

    optimizer.step()

    assert {
        float(state["step"].item()) for state in optimizer.state.values()
    } == {float(loaded.global_step + 1)}


def test_save_rejects_a_ledger_owned_by_another_model(run_dir: Path) -> None:
    model, optimizer, scheduler = _components()
    cursor = _advance_components(model, optimizer, scheduler, steps=1)
    other_model = MoiraModel(_model_config())

    with pytest.raises(ValueError, match="ledger|model"):
        save_checkpoint(
            run_dir,
            lineage_id="lineage-test",
            global_step=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            data_cursor=cursor,
            semantic_manifests=_manifests(),
            provenance={
                "git": {"available": True, "revision": "abc123", "dirty": False}
            },
            parameter_ledger=other_model.parameter_ledger(),
            counters={"optimizer_steps": 1, "batches_seen": 1},
            last_metrics={
                "loss": 1.0,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            },
            keep_last_checkpoints=2,
        )


def test_tensor_equality_requires_metadata_as_well_as_values() -> None:
    assert not checkpoint_module._values_equal(
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([1], dtype=torch.float32),
    )


@pytest.mark.parametrize("entrypoint", ("save", "auto", "retention"))
def test_pointer_target_is_recovered_from_pruning_before_verification(
    run_dir: Path,
    entrypoint: str,
) -> None:
    record = _save(run_dir, step=1, keep=3)
    checkpoint_dir = record.checkpoint_path.parent
    run_metadata = json.loads((checkpoint_dir / "run.json").read_text(encoding="utf-8"))
    marker = checkpoint_dir / f".{record.checkpoint_path.stem}.pruning.json"
    checkpoint_tombstone = checkpoint_dir / f".{record.checkpoint_path.name}.pruning"
    sidecar_tombstone = checkpoint_dir / f".{record.sidecar_path.name}.pruning"
    marker.write_text(
        json.dumps(
            {
                **run_metadata,
                "checkpoint": record.checkpoint_path.name,
                "sidecar": record.sidecar_path.name,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.link(record.checkpoint_path, checkpoint_tombstone)
    os.link(record.sidecar_path, sidecar_tombstone)
    record.checkpoint_path.unlink()
    record.sidecar_path.unlink()

    if entrypoint == "save":
        result_path = _save(run_dir, step=2, keep=3).checkpoint_path
    elif entrypoint == "auto":
        model, optimizer, scheduler = _components()
        result_path = load_checkpoint(
            run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
            auto=True,
        ).checkpoint_path
    else:
        checkpoint_module._prune_validated(checkpoint_dir, 3)
        result_path = record.checkpoint_path

    expected_step = 2 if entrypoint == "save" else 1
    assert verify_checkpoint(result_path)["global_step"] == expected_step
    assert verify_checkpoint(record.checkpoint_path)["global_step"] == 1
    assert not marker.exists()
    assert not checkpoint_tombstone.exists()
    assert not sidecar_tombstone.exists()


def test_retry_reconciles_an_old_invalid_tombstone_and_new_canonical(
    run_dir: Path,
) -> None:
    first = _save(run_dir, step=1, keep=3)
    checkpoint = first.checkpoint_path.parent / "step-000000002.pt"
    tombstone = checkpoint.parent / f".{checkpoint.name}.invalid"
    checkpoint.write_bytes(b"old invalid generation")
    os.link(checkpoint, tombstone)
    checkpoint.unlink()
    checkpoint.write_bytes(b"new invalid generation")

    retried = _save(run_dir, step=2, keep=3)

    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 2
    assert not tombstone.exists()


def test_pointer_referenced_invalid_tombstone_restores_a_missing_canonical(
    run_dir: Path,
) -> None:
    first = _save(run_dir, step=1, keep=3)
    tombstone = first.checkpoint_path.parent / f".{first.checkpoint_path.name}.invalid"
    os.link(first.checkpoint_path, tombstone)
    first.checkpoint_path.unlink()

    second = _save(run_dir, step=2, keep=3)

    assert verify_checkpoint(first.checkpoint_path)["global_step"] == 1
    assert verify_checkpoint(second.checkpoint_path)["global_step"] == 2
    assert not tombstone.exists()


def test_pointer_referenced_invalid_tombstone_conflict_preserves_both_files(
    run_dir: Path,
) -> None:
    record = _save(run_dir, step=1, keep=3)
    tombstone = record.checkpoint_path.parent / f".{record.checkpoint_path.name}.invalid"
    os.link(record.checkpoint_path, tombstone)
    record.checkpoint_path.unlink()
    record.checkpoint_path.write_bytes(b"different replacement")
    canonical_before = record.checkpoint_path.read_bytes()
    tombstone_before = tombstone.read_bytes()

    with pytest.raises(ValueError, match="conflict|pointer|invalid"):
        _save(run_dir, step=2, keep=3)

    assert record.checkpoint_path.read_bytes() == canonical_before
    assert tombstone.read_bytes() == tombstone_before


@pytest.mark.parametrize(
    "boundary",
    (
        "link",
        "link-sync",
        "canonical-unlink",
        "canonical-sync",
        "tombstone-unlink",
        "tombstone-sync",
    ),
)
def test_invalid_reclamation_retry_converges_after_every_filesystem_boundary(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    first = _save(run_dir, step=1, keep=3)
    checkpoint = first.checkpoint_path.parent / "step-000000002.pt"
    tombstone = checkpoint.parent / f".{checkpoint.name}.invalid"
    checkpoint.write_bytes(b"invalid generation")
    original_link = checkpoint_module.os.link
    original_unlink = Path.unlink
    original_fsync = checkpoint_module._fsync_directory
    injected = False

    def link_then_fail(source: object, target: object, *args: object, **kwargs: object):
        nonlocal injected
        result = original_link(source, target, *args, **kwargs)
        if Path(target) == tombstone and boundary == "link" and not injected:
            injected = True
            raise OSError("injected invalid link failure")
        return result

    def unlink_then_fail(path: Path, *args: object, **kwargs: object):
        nonlocal injected
        result = original_unlink(path, *args, **kwargs)
        target_boundary = (
            "canonical-unlink" if path == checkpoint else "tombstone-unlink"
        )
        if path in (checkpoint, tombstone) and boundary == target_boundary and not injected:
            injected = True
            raise OSError(f"injected {boundary} failure")
        return result

    def fsync_then_fail(path: Path) -> None:
        nonlocal injected
        original_fsync(path)
        same_inode = False
        if checkpoint.exists() and tombstone.exists():
            same_inode = os.path.samefile(checkpoint, tombstone)
        observed = None
        if same_inode:
            observed = "link-sync"
        elif tombstone.exists() and not checkpoint.exists():
            observed = "canonical-sync"
        elif not tombstone.exists() and not checkpoint.exists():
            observed = "tombstone-sync"
        if boundary == observed and not injected:
            injected = True
            raise OSError(f"injected {boundary} failure")

    with monkeypatch.context() as context:
        context.setattr(checkpoint_module.os, "link", link_then_fail)
        context.setattr(Path, "unlink", unlink_then_fail)
        context.setattr(checkpoint_module, "_fsync_directory", fsync_then_fail)
        with pytest.raises(OSError, match="injected"):
            _save(run_dir, step=2, keep=3)

    assert injected
    retried = _save(run_dir, step=2, keep=3)
    assert verify_checkpoint(retried.checkpoint_path)["global_step"] == 2
    assert not tombstone.exists()


def test_different_step_process_writers_cannot_regress_the_pointer(
    run_dir: Path,
) -> None:
    _save(run_dir, step=1, keep=4)
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    result_path = run_dir / "writer-result.txt"
    process = context.Process(
        target=_blocked_step_writer,
        args=(str(run_dir), ready, release, str(result_path)),
    )
    process.start()
    try:
        assert ready.wait(20), "step-2 writer did not reach the pointer boundary"
        with pytest.raises(ValueError, match="writer lock"):
            _save(run_dir, step=3, keep=4)
    finally:
        release.set()
        process.join(20)
        if process.is_alive():
            process.terminate()
            process.join(5)

    assert process.exitcode == 0
    assert result_path.read_text(encoding="utf-8") == "ok"
    latest = _save(run_dir, step=3, keep=4)
    pointer = json.loads(latest.latest_path.read_text(encoding="utf-8"))
    assert pointer["step"] == 3
    assert verify_checkpoint(latest.checkpoint_path)["global_step"] == 3


@pytest.mark.parametrize("run_name", ("experiment.pt", "checkpoints"))
def test_run_directory_names_do_not_change_lock_or_artifact_placement(
    run_dir: Path,
    run_name: str,
) -> None:
    named_run = run_dir / run_name
    record = _save(named_run, step=1)

    assert record.checkpoint_path.parent == named_run / "checkpoints"
    assert not (named_run.parent / "run.json").exists()
    assert verify_checkpoint(record.checkpoint_path)["global_step"] == 1
    model, optimizer, scheduler = _components()
    loaded = load_checkpoint(
        named_run,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        expected_manifests=_manifests(),
        auto=True,
    )
    assert loaded.checkpoint_path == record.checkpoint_path


@pytest.mark.parametrize(
    "material_name",
    (".step-000000001.pt.invalid", ".step-000000001.pt.pruning"),
)
def test_missing_run_metadata_cannot_rebind_a_tombstone_only_directory(
    run_dir: Path,
    material_name: str,
) -> None:
    named_run = run_dir / "tombstone-only"
    checkpoint_dir = named_run / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    material = checkpoint_dir / material_name
    material.write_bytes(b"pre-existing generation")

    with pytest.raises(ValueError, match="run metadata|non-empty"):
        _save(named_run, step=1)

    assert material.read_bytes() == b"pre-existing generation"
    assert not (checkpoint_dir / "run.json").exists()


def test_auto_recovery_does_not_treat_a_nonregular_pointer_as_content_damage(
    run_dir: Path,
) -> None:
    record = _save(run_dir, step=1)
    record.latest_path.unlink()
    record.latest_path.mkdir()
    model, optimizer, scheduler = _components()

    with pytest.raises(ValueError, match="regular|symlink|pointer|path"):
        load_checkpoint(
            run_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            expected_manifests=_manifests(),
            auto=True,
        )


def test_save_checks_a_dangling_pointer_entry_before_the_absence_branch(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _save(run_dir, step=1)
    pointer_before = record.latest_path.read_bytes()
    original_exists = Path.exists
    original_is_symlink = Path.is_symlink

    monkeypatch.setattr(
        Path,
        "exists",
        lambda path: False if path == record.latest_path else original_exists(path),
    )
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path == record.latest_path or original_is_symlink(path),
    )

    with pytest.raises(ValueError, match="symlink|regular|pointer|path"):
        _save(run_dir, step=2)

    assert record.latest_path.read_bytes() == pointer_before


def test_writer_lock_is_released_by_the_os_after_process_death(
    run_dir: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(
        target=_crash_while_holding_run_lock,
        args=(str(run_dir), ready),
    )
    process.start()
    try:
        assert ready.wait(20), "crashing writer did not acquire the run lock"
        process.join(20)
        assert not process.is_alive(), "crashing writer did not exit"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)

    assert process.exitcode == 0
    record = _save(run_dir, step=1)
    assert verify_checkpoint(record.checkpoint_path)["global_step"] == 1
    assert (record.checkpoint_path.parent / ".run.lock").is_file()


def test_nonregular_writer_lock_fails_closed_before_publication(run_dir: Path) -> None:
    named_run = run_dir / "nonregular-lock"
    lock_path = named_run / "checkpoints" / ".run.lock"
    lock_path.mkdir(parents=True)

    with pytest.raises(ValueError, match="lock|regular|safely"):
        _save(named_run, step=1)

    assert not (lock_path.parent / "run.json").exists()


@pytest.mark.parametrize("boundary", ("canonical", "hash", "sidecar"))
def test_process_crash_temps_are_cleaned_before_retry(
    run_dir: Path,
    boundary: str,
) -> None:
    named_run = run_dir / f"crash-{boundary}"
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_crash_during_checkpoint_publication,
        args=(str(named_run), boundary),
    )
    process.start()
    process.join(20)
    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail(f"{boundary} crash worker did not exit")
    assert process.exitcode == 0

    record = _save(named_run, step=1)

    assert verify_checkpoint(record.checkpoint_path)["global_step"] == 1
    assert not [
        path
        for path in record.checkpoint_path.parent.iterdir()
        if path.name.endswith(".tmp")
    ]


def test_parent_symlink_is_rejected_before_checkpoint_directory_creation() -> None:
    root = Path("research/runs")
    target = root / f"task5-symlink-target-{uuid.uuid4().hex}"
    link = root / f"task5-symlink-link-{uuid.uuid4().hex}"
    target.mkdir(parents=True)
    try:
        try:
            os.symlink(target.resolve(), link, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"directory symlink creation is unavailable: {error}")
        with pytest.raises(ValueError, match="symlink|junction|reparse|indirection"):
            _save(link / "external-run", step=1)
        assert not (target / "external-run").exists()
    finally:
        if link.is_symlink():
            link.unlink()
        shutil.rmtree(target, ignore_errors=True)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction integration")
def test_windows_junction_is_rejected_before_checkpoint_directory_creation() -> None:
    root = Path("research/runs")
    target = root / f"task5-junction-target-{uuid.uuid4().hex}"
    junction = root / f"task5-junction-link-{uuid.uuid4().hex}"
    target.mkdir(parents=True)
    created = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(target.resolve())],
        check=False,
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        pytest.skip(f"junction creation is unavailable: {created.stderr.strip()}")
    try:
        with pytest.raises(ValueError, match="junction|reparse|indirection"):
            _save(junction / "external-run", step=1)
        assert not (target / "external-run").exists()
    finally:
        if junction.exists():
            junction.rmdir()
        shutil.rmtree(target, ignore_errors=True)
