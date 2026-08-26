import hashlib
import json
from pathlib import Path
import uuid

import pytest

from moira.agent_data import load_agent_dataset_manifest


FIXTURES = Path("research/tests/fixtures")
TRAIN_MANIFEST = FIXTURES / "agent-trajectories.manifest.json"
EVALUATION_MANIFEST = FIXTURES / "agent-trajectories-held-out.manifest.json"


def test_agent_dataset_manifests_bind_split_content_and_provenance() -> None:
    training = load_agent_dataset_manifest(TRAIN_MANIFEST, expected_split="train")
    evaluation = load_agent_dataset_manifest(
        EVALUATION_MANIFEST,
        expected_split="evaluation",
    )

    assert training.dataset_id == "moira-agent-fixture-train-v1"
    assert training.record_count == 4
    assert training.provenance_class == "synthetic"
    assert training.trajectory_path.name == "agent-trajectories.jsonl"
    assert len(training.manifest_sha256) == 64
    assert training.manifest_sha256 != evaluation.manifest_sha256
    assert training.trajectory_sha256 != evaluation.trajectory_sha256
    assert training.checkpoint_fields()["dataset_manifest_sha256"] == (
        training.manifest_sha256
    )


def test_agent_dataset_manifest_rejects_wrong_split() -> None:
    with pytest.raises(ValueError, match="wrong split"):
        load_agent_dataset_manifest(TRAIN_MANIFEST, expected_split="evaluation")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", True, "schema version"),
        ("trajectory_sha256", "0" * 64, "digest"),
        ("record_count", 5, "record count"),
        ("trajectory_file", "../agent-trajectories.jsonl", "manifest directory"),
        ("deletion_policy", "N/A", "deletion policy"),
    ],
)
def test_agent_dataset_manifest_rejects_false_materialization_claims(
    field: str,
    value: object,
    message: str,
) -> None:
    identifier = uuid.uuid4().hex
    trajectory = Path("research") / f".agent-data-{identifier}.jsonl"
    path = Path("research") / f".agent-data-{identifier}.json"
    manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    try:
        trajectory.write_bytes(
            (FIXTURES / "agent-trajectories.jsonl").read_bytes()
        )
        manifest["trajectory_file"] = trajectory.name
        manifest[field] = value
        path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match=message):
            load_agent_dataset_manifest(path, expected_split="train")
    finally:
        trajectory.unlink(missing_ok=True)
        path.unlink(missing_ok=True)


def test_agent_dataset_manifest_rejects_duplicate_json_keys() -> None:
    identifier = uuid.uuid4().hex
    path = Path("research") / f".agent-duplicate-{identifier}.json"
    raw = TRAIN_MANIFEST.read_text(encoding="utf-8").replace(
        '  "split": "train",',
        '  "split": "train",\n  "split": "evaluation",',
        1,
    )
    try:
        path.write_text(raw, encoding="utf-8")
        with pytest.raises(ValueError, match="valid JSON"):
            load_agent_dataset_manifest(path)
    finally:
        path.unlink(missing_ok=True)


def test_owner_consent_cannot_be_claimed_without_a_consent_reference() -> None:
    identifier = uuid.uuid4().hex
    trajectory = Path("research") / f".agent-consent-{identifier}.jsonl"
    path = Path("research") / f".agent-consent-{identifier}.json"
    data = (FIXTURES / "agent-trajectories.jsonl").read_bytes()
    manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    try:
        trajectory.write_bytes(data)
        manifest.update(
            {
                "trajectory_file": trajectory.name,
                "provenance_class": "owner-consented",
                "source_uri": "owner://local/private-trajectories",
                "consent_reference": "N/A",
                "tenant_scope": "owner-local",
                "redistribution_allowed": False,
                "trajectory_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="consented"):
            load_agent_dataset_manifest(path, expected_split="train")
    finally:
        trajectory.unlink(missing_ok=True)
        path.unlink(missing_ok=True)


def test_owner_consented_private_data_requires_scope_and_deletion_metadata() -> None:
    identifier = uuid.uuid4().hex
    trajectory = Path("research") / f".agent-owner-{identifier}.jsonl"
    path = Path("research") / f".agent-owner-{identifier}.json"
    data = (FIXTURES / "agent-trajectories.jsonl").read_bytes()
    manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    try:
        trajectory.write_bytes(data)
        manifest.update(
            {
                "dataset_id": "owner-consented-agent-slice",
                "trajectory_file": trajectory.name,
                "provenance_class": "owner-consented",
                "source_uri": "owner://local/private-trajectories",
                "consent_reference": "consent://owner/grant-2026-08-26",
                "consent_verified": True,
                "tenant_scope": "owner-local",
                "contains_personal_data": True,
                "redistribution_allowed": False,
                "deletion_policy": "delete-on-owner-request",
                "trajectory_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        path.write_text(json.dumps(manifest), encoding="utf-8")

        loaded = load_agent_dataset_manifest(path, expected_split="train")

        assert loaded.provenance_class == "owner-consented"
        assert loaded.contains_personal_data is True
        assert loaded.redistribution_allowed is False
        assert loaded.tenant_scope == "owner-local"
    finally:
        trajectory.unlink(missing_ok=True)
        path.unlink(missing_ok=True)


def test_public_license_requires_an_https_license_reference() -> None:
    identifier = uuid.uuid4().hex
    trajectory = Path("research") / f".agent-license-{identifier}.jsonl"
    path = Path("research") / f".agent-license-{identifier}.json"
    data = (FIXTURES / "agent-trajectories.jsonl").read_bytes()
    manifest = json.loads(TRAIN_MANIFEST.read_text(encoding="utf-8"))
    try:
        trajectory.write_bytes(data)
        manifest.update(
            {
                "trajectory_file": trajectory.name,
                "provenance_class": "licensed-public",
                "source_uri": "https://example.invalid/agent-data",
                "license": "Apache-2.0",
                "license_url": "N/A",
                "license_verified": True,
                "trajectory_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="licensed"):
            load_agent_dataset_manifest(path, expected_split="train")
    finally:
        trajectory.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
