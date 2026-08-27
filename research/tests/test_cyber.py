from dataclasses import replace

import pytest

from moira.lifelong import EvidenceOutcome

from moira.cyber import (
    CyberCredentialPolicy,
    CyberLabScopeV1,
    CyberLearningEvidenceV1,
    CyberNetworkPolicy,
    CyberResourceLimitsV1,
)


_DIGEST = "a" * 64


def _scope() -> CyberLabScopeV1:
    return CyberLabScopeV1(
        scope_id="lab:owner:1",
        owner_authorization_digest=_DIGEST,
        target_manifest_digest=_DIGEST,
        permitted_techniques=("secure-code-review", "defensive-detection-test"),
        isolated_worker_identity="worker:low-privilege:1",
        network_policy=CyberNetworkPolicy.OFF,
        credential_policy=CyberCredentialPolicy.NONE,
        writable_mounts=("staging",),
        resource_limits=CyberResourceLimitsV1(
            cpu_seconds=60,
            wall_seconds=120,
            peak_memory_bytes=1_000_000_000,
            writable_bytes=100_000_000,
            process_count=4,
        ),
        expires_at_ms=2_000,
        audit_destination="ledger:owner:1",
    )


def test_default_cyber_lab_is_expiring_isolated_and_network_off() -> None:
    scope = _scope()
    scope.validate(now_ms=1_000)

    assert scope.network_policy is CyberNetworkPolicy.OFF
    assert scope.credential_policy is CyberCredentialPolicy.NONE
    assert scope.production_targets_allowed is False
    assert scope.administrator_identity_allowed is False
    assert scope.host_credentials_allowed is False
    assert scope.one_use_effect_permits_required is True


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"production_targets_allowed": True}, "production"),
        ({"administrator_identity_allowed": True}, "administrator"),
        ({"host_credentials_allowed": True}, "credentials"),
        ({"one_use_effect_permits_required": False}, "one-use"),
        ({"writable_mounts": ("../host",)}, "relative"),
        ({"writable_mounts": (".",)}, "normalized"),
        ({"writable_mounts": ("./staging",)}, "normalized"),
        ({"writable_mounts": ("staging//child",)}, "normalized"),
        ({"writable_mounts": ("C:/host",)}, "normalized"),
        ({"writable_mounts": ("staging:stream",)}, "normalized"),
        ({"writable_mounts": ("staging/\x00child",)}, "normalized"),
        ({"permitted_techniques": ({"unhashable": True},)}, "techniques"),
    ],
)
def test_cyber_lab_rejects_authority_or_isolation_expansion(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_scope(), **change).validate(now_ms=1_000)


def test_expired_cyber_scope_fails_closed() -> None:
    with pytest.raises(ValueError, match="expired"):
        _scope().validate(now_ms=2_000)


def test_positive_cyber_learning_requires_independent_authorized_outcome() -> None:
    evidence = CyberLearningEvidenceV1(
        lab_scope_id="lab:owner:1",
        effect_receipt_digest=_DIGEST,
        evaluator_digest=_DIGEST,
        outcome=EvidenceOutcome.PASS,
        independent=True,
        authorized_target=True,
    )
    evidence.validate_positive()

    with pytest.raises(ValueError, match="unauthorized"):
        replace(evidence, authorized_target=False).validate_positive()
    with pytest.raises(ValueError, match="independent"):
        replace(evidence, independent=False).validate_positive()
    with pytest.raises(ValueError, match="typed evidence"):
        replace(evidence, outcome="pass").validate_positive()  # type: ignore[arg-type]
