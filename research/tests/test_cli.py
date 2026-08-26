from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import uuid

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "research" / "configs" / "m0-smoke.toml"
AGENT_TRAIN_MANIFEST = (
    REPOSITORY_ROOT
    / "research"
    / "tests"
    / "fixtures"
    / "agent-trajectories.manifest.json"
)
AGENT_HELD_OUT_MANIFEST = (
    REPOSITORY_ROOT
    / "research"
    / "tests"
    / "fixtures"
    / "agent-trajectories-held-out.manifest.json"
)


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


def _run_cli(
    *arguments: object, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
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
        input=input_text,
        timeout=60,
        check=False,
    )


def _output_json(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    return json.loads(result.stdout)


def _tool_context_entry(request: dict[str, object]) -> dict[str, object]:
    context = json.loads(request["messages"][1]["content"])
    entries = [item for item in context["context"] if item["source"] == "tool"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["trust"] == "untrusted"
    assert entry["provenance"] == "moira-agent-observation-v1:tool_result"
    receipt = context["context_receipt"]
    assert entry["fragment_id"] in receipt["included_ids"]
    assert entry["fragment_digest"] in receipt["included_digests"]
    return entry


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
    assert output["switch_command_available"] is True
    assert output["switch_changes_weights"] is False


def test_agent_inspect_reports_native_authority_boundary() -> None:
    output = _output_json(_run_cli("agent", "inspect"))

    assert output["model"] == "MOIRA"
    assert output["control_kernel"] == "moira-agent-kernel-v1"
    assert output["orientations"] == ["clotho", "lachesis", "atropos"]
    assert output["roles"] == [
        "planner",
        "analyst",
        "executor",
        "reviewer",
        "evaluator",
    ]
    assert output["any_orientation_can_hold_any_role"] is True
    assert output["whole_model_agentic"] is True
    assert output["agent_layer_scope"] == "shared-across-task-selected-sisters"
    assert output["external_agent_runtime_required"] is False
    assert output["context_protocol"] == "moira-context-v1"
    assert output["context_scope_bound_to_lease"] is True
    assert output["context_receipt_binds_labels_and_content"] is True
    assert output["context_persistence_implemented"] is False
    assert output["model_can_authorize_its_own_effects"] is False
    assert output["optional_hosts"] == [
        "codex",
        "claude-code",
        "hermes-agent",
        "local",
    ]


def test_agent_run_ollama_uses_local_structured_api_and_finishes(
    cli_root: Path,
) -> None:
    requests: list[dict[str, object]] = []
    (cli_root / "evidence.txt").write_text("needle\n", encoding="utf-8")

    class OllamaHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length)))
            if len(requests) == 1:
                action_value = {
                    "kind": "use_tool",
                    "text": "",
                    "tool_id": "workspace.search",
                    "target": "needle",
                    "payload_text": "",
                }
            else:
                action_value = {
                    "kind": "finish",
                    "text": "The read-only task is complete.",
                    "tool_id": "",
                    "target": "",
                    "payload_text": "",
                }
            action = json.dumps(action_value)
            response = json.dumps({"message": {"content": action}}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), OllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = _run_cli(
            "agent",
            "run-ollama",
            "--model",
            "moira-test:latest",
            "--task",
            "Inspect the workspace without changing it.",
            "--workspace",
            cli_root,
            "--base-url",
            f"http://127.0.0.1:{server.server_port}",
            "--max-steps",
            2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert result.stderr == "The read-only task is complete.\n"
    assert output["status"] == "completed"
    assert output["read_only"] is True
    assert output["profile"] == "read-only"
    assert output["enabled_tools"] == ["workspace.read", "workspace.search"]
    assert output["model"] == "moira-test:latest"
    assert output["steps"] == 2
    assert output["effects"] == 1
    assert len(output["receipts"]) == 1
    assert len(requests) == 2
    assert requests[0]["model"] == "moira-test:latest"
    assert requests[0]["stream"] is False
    assert requests[0]["format"]["additionalProperties"] is False
    tool_context = _tool_context_entry(requests[1])
    assert tool_context["content"] == "evidence.txt:1:needle"


def test_agent_run_ollama_requires_exact_approval_for_workspace_write(
    cli_root: Path,
) -> None:
    requests: list[dict[str, object]] = []

    class OllamaHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length)))
            if len(requests) == 1:
                action_value = {
                    "kind": "use_tool",
                    "text": "",
                    "tool_id": "workspace.write",
                    "target": "answer.txt",
                    "payload_text": "approved content",
                }
            else:
                action_value = {
                    "kind": "finish",
                    "text": "Write complete.",
                    "tool_id": "",
                    "target": "",
                    "payload_text": "",
                }
            action = json.dumps(action_value)
            response = json.dumps({"message": {"content": action}}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), OllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = _run_cli(
            "agent",
            "run-ollama",
            "--model",
            "moira-test:latest",
            "--task",
            "Write the approved answer.",
            "--workspace",
            cli_root,
            "--base-url",
            f"http://127.0.0.1:{server.server_port}",
            "--max-steps",
            2,
            "--allow-write",
            input_text="yes\n",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["profile"] == "developer"
    assert output["read_only"] is False
    assert output["effects"] == 1
    assert "workspace.write" in output["enabled_tools"]
    assert (cli_root / "answer.txt").read_text(encoding="utf-8") == (
        "approved content"
    )
    assert "filesystem_write:write" in result.stderr
    assert "Payload: 16 bytes, sha256:" in result.stderr
    assert "Approve? [y/N]" in result.stderr
    tool_context = _tool_context_entry(requests[1])
    assert tool_context["content"] == "wrote 16 bytes to answer.txt"


def test_agent_run_ollama_requires_approval_for_allowlisted_process(
    cli_root: Path,
) -> None:
    requests: list[dict[str, object]] = []
    command_target = json.dumps(
        {
            "command": sys.executable,
            "args": ["-c", "print('agent-process-ok')"],
        }
    )

    class OllamaHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length)))
            if len(requests) == 1:
                action_value = {
                    "kind": "use_tool",
                    "text": "",
                    "tool_id": "process.run",
                    "target": command_target,
                    "payload_text": "",
                }
            else:
                action_value = {
                    "kind": "finish",
                    "text": "Process complete.",
                    "tool_id": "",
                    "target": "",
                    "payload_text": "",
                }
            action = json.dumps(action_value)
            response = json.dumps({"message": {"content": action}}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), OllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = _run_cli(
            "agent",
            "run-ollama",
            "--model",
            "moira-test:latest",
            "--task",
            "Run the approved process.",
            "--workspace",
            cli_root,
            "--base-url",
            f"http://127.0.0.1:{server.server_port}",
            "--max-steps",
            2,
            "--allow-command",
            sys.executable,
            input_text="yes\n",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["profile"] == "developer"
    assert output["effects"] == 1
    assert "process.run" in output["enabled_tools"]
    assert "process:run" in result.stderr
    assert "Payload: 0 bytes, sha256:" in result.stderr
    assert "Request: sha256:" in result.stderr
    tool_context = _tool_context_entry(requests[1])
    process_result = json.loads(tool_context["content"])
    assert process_result["command"] == sys.executable
    assert process_result["exit_code"] == 0
    assert process_result["output"].strip() == "agent-process-ok"


def test_agent_run_ollama_rejects_effect_budget_above_step_budget(
    cli_root: Path,
) -> None:
    result = _run_cli(
        "agent",
        "run-ollama",
        "--model",
        "moira-test:latest",
        "--task",
        "Inspect the workspace.",
        "--workspace",
        cli_root,
        "--max-steps",
        1,
        "--max-effects",
        2,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "max_effects must be between 0 and max_steps" in result.stderr


def test_agent_run_ollama_rejects_unavailable_allowlisted_command(
    cli_root: Path,
) -> None:
    result = _run_cli(
        "agent",
        "run-ollama",
        "--model",
        "moira-test:latest",
        "--task",
        "Run a command.",
        "--workspace",
        cli_root,
        "--allow-command",
        "definitely-not-a-real-moira-command",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "allowlisted command is unavailable" in result.stderr


def test_agent_action_training_and_evaluation_cli_are_checkpointed(
    cli_root: Path,
) -> None:
    checkpoint = cli_root / "agent-actions.pt"
    training = _output_json(
        _run_cli(
            "agent",
            "train-actions",
            "--manifest",
            AGENT_TRAIN_MANIFEST,
            "--checkpoint",
            checkpoint,
            "--steps",
            1,
        )
    )
    before_evaluation = checkpoint.read_bytes()
    evaluation = _output_json(
        _run_cli(
            "agent",
            "evaluate-actions",
            "--manifest",
            AGENT_HELD_OUT_MANIFEST,
            "--training-manifest",
            AGENT_TRAIN_MANIFEST,
            "--checkpoint",
            checkpoint,
            "--max-records",
            4,
            "--max-new-tokens",
            4,
        )
    )

    assert training["objective"] == "moira-agent-action-json-v3"
    assert training["model_surface"] == "agent-smoke"
    assert training["weights_updated"] is True
    assert training["steps"] == 1
    assert training["samples"] == 1
    assert evaluation["objective"] == "moira-agent-action-json-v3"
    assert evaluation["weights_updated"] is False
    assert evaluation["held_out"] is True
    assert evaluation["training_dataset_id"] == "moira-agent-fixture-train-v1"
    assert evaluation["evaluation_dataset_id"] == (
        "moira-agent-fixture-evaluation-v1"
    )
    assert evaluation["training_manifest_sha256"] != (
        evaluation["evaluation_manifest_sha256"]
    )
    assert evaluation["checkpoint_step"] == 1
    assert evaluation["samples"] == 4
    assert 0.0 <= evaluation["token_accuracy"] <= 1.0
    assert 0.0 <= evaluation["schema_validity"] <= 1.0
    assert 0.0 <= evaluation["exact_action_accuracy"] <= 1.0
    assert evaluation["tool_selection_samples"] == 1
    assert checkpoint.read_bytes() == before_evaluation


def test_agent_data_manifest_cli_preflights_without_training() -> None:
    result = _output_json(
        _run_cli(
            "agent",
            "inspect-data",
            "--manifest",
            AGENT_TRAIN_MANIFEST,
            "--split",
            "train",
        )
    )

    assert result["dataset_id"] == "moira-agent-fixture-train-v1"
    assert result["split"] == "train"
    assert result["provenance_class"] == "synthetic"
    assert result["record_count"] == 4
    assert result["contains_personal_data"] is False
    assert result["license_verified"] is False
    assert result["consent_verified"] is False
    assert len(result["manifest_sha256"]) == 64


def test_deployment_switch_persists_policy_without_changing_weights(
    cli_root: Path,
) -> None:
    state_path = cli_root / "deployment.json"

    consumer = _output_json(
        _run_cli("deployment", "switch", "consumer", "--state", state_path)
    )
    enterprise = _output_json(
        _run_cli("deployment", "switch", "enterprise", "--state", state_path)
    )
    persisted = json.loads(state_path.read_text(encoding="utf-8"))

    assert consumer["previous_profile"] is None
    assert enterprise["previous_profile"] == "consumer"
    assert enterprise["model"] == "MOIRA"
    assert enterprise["profile"] == "enterprise"
    assert enterprise["switch_applied"] is True
    assert enterprise["weights_converted"] is False
    assert persisted["profile"] == "enterprise"
    assert persisted["precision_policy"]["verifier_compute_dtype"] == "fp64"
    assert persisted["state_only_no_weight_conversion"] is True


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


def test_m1_inspect_and_catalog_validation_are_read_only() -> None:
    inspected = _output_json(_run_cli("m1", "inspect"))
    assert inspected["phase"] == "M1"
    assert inspected["trainable_parameters"] == 33_032_562
    assert inspected["weights_downloaded"] is False

    catalog = _output_json(
        _run_cli(
            "dataset",
            "validate",
            "--catalog",
            REPOSITORY_ROOT / "research" / "configs" / "m1-sources.toml",
        )
    )
    assert catalog["source_count"] == 38
    assert catalog["weights_or_datasets_downloaded"] is False


def test_m1_capability_evaluation_command_is_registered() -> None:
    result = _run_cli("m1", "evaluate-capabilities", "--help")

    assert result.returncode == 0
    assert result.stderr == ""
    assert "evaluate-capabilities" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--split" in result.stdout


def test_elastic_policy_and_unbounded_growth_plan_cli() -> None:
    policy = _output_json(_run_cli("elastic", "policy", "--profile", "enterprise"))
    assert policy["bit_precision_ladder"] == [
        "int3",
        "int6",
        "int9",
        "int27",
        "int81",
    ]
    assert policy["parameter_growth_ladder"] == [
        "3B",
        "9B",
        "27B",
        "81B",
        "243B",
        "729B",
    ]
    assert policy["unbounded_parameter_growth"] is True
    assert policy["precision_policy"]["verifier_bits"] == "int81"

    plan = _output_json(
        _run_cli(
            "elastic",
            "plan",
            "--current",
            "3B",
            "--target",
            "243B",
            "--profile",
            "consumer",
            "--runtime-bits",
            "int6",
        )
    )
    assert plan["runtime_bits"] == "int6"
    assert plan["growth"]["target"] == "243B"
    assert [step["target"] for step in plan["growth"]["steps"]] == [
        "9B",
        "27B",
        "81B",
        "243B",
    ]
    assert plan["growth"]["unbounded_growth"] is True
    assert plan["status"] == "plan-only-no-target-allocation"


def test_qat_smoke_cli_trains_int3_then_promotes_to_int6(cli_root: Path) -> None:
    checkpoint = cli_root / "int3-to-int6.pt"
    output = _output_json(
        _run_cli(
            "elastic",
            "qat-smoke",
            "--config",
            CONFIG_PATH,
            "--initial-bits",
            "int3",
            "--promote-to",
            "int6",
            "--steps-per-stage",
            1,
            "--device",
            "cpu",
            "--checkpoint",
            checkpoint,
        )
    )

    assert output["architecture"] == "MOIRA-master-retained-INTx-QAT-v1"
    assert output["precision_history"] == ["int3", "int6"]
    assert output["preparation"]["master_weights_retained"] is True
    assert output["promotion"]["reconstructed_from_low_bit_artifact"] is False
    assert output["final_qat_state"]["logical_bits"] == "int6"
    assert output["global_step"] == 2
    assert output["all_losses_finite"] is True
    assert output["scope"] == "proof-scale-weight-only-QAT-not-a-trained-3B-checkpoint"
    assert Path(str(output["checkpoint"])).is_file()


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


def test_scale_probe_cli_rejects_sequence_past_milestone_limit() -> None:
    result = _run_cli(
        "m1",
        "scale-probe",
        "--milestone",
        "m3",
        "--sequence-length",
        "129",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "sequence_length must fit" in result.stderr


def test_m2_inspect_reports_the_fixed_reference_surface() -> None:
    output = _output_json(_run_cli("m2", "inspect"))

    assert output["phase"] == "M2"
    assert output["device"] == "meta"
    assert output["trainable_parameters"] == 153_349_122
    assert output["expected_trainable_parameters"] == 153_349_122
    assert output["weights_downloaded"] is False


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
