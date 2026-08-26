import pytest

from moira.context import (
    CONTEXT_PROTOCOL,
    ContextAssembler,
    ContextClassification,
    ContextDropReason,
    ContextFragment,
    ContextSource,
    ContextTrust,
    context_entry_mapping,
    context_receipt_mapping,
)


def _fragment(
    fragment_id: str,
    content: str,
    *,
    source: ContextSource = ContextSource.RETRIEVAL,
    trust: ContextTrust = ContextTrust.UNTRUSTED,
    classification: ContextClassification = ContextClassification.PRIVATE,
    scope_id: str = "scope-1",
    priority: int = 50,
    expires_at: int | None = None,
    required: bool = False,
    provenance: str | None = None,
) -> ContextFragment:
    return ContextFragment(
        fragment_id=fragment_id,
        source=source,
        trust=trust,
        classification=classification,
        scope_id=scope_id,
        content=content,
        provenance=provenance or f"test://{fragment_id}",
        priority=priority,
        expires_at=expires_at,
        required=required,
    )


def test_context_assembly_is_deterministic_and_provenance_bearing() -> None:
    fragments = (
        _fragment("retrieval-b", "second", priority=60),
        _fragment(
            "task",
            "required task",
            source=ContextSource.TASK,
            trust=ContextTrust.USER_SUPPLIED,
            priority=100,
            required=True,
        ),
        _fragment("retrieval-a", "first", priority=60),
    )
    assembler = ContextAssembler()

    first = assembler.assemble(
        task_id="task-1",
        scope_id="scope-1",
        fragments=fragments,
        token_budget=1_024,
        classification_boundary=ContextClassification.PRIVATE,
        now=100,
    )
    second = assembler.assemble(
        task_id="task-1",
        scope_id="scope-1",
        fragments=reversed(fragments),
        token_budget=1_024,
        classification_boundary=ContextClassification.PRIVATE,
        now=100,
    )

    assert [item.fragment_id for item in first.entries] == [
        "task",
        "retrieval-a",
        "retrieval-b",
    ]
    assert first == second
    assert first.receipt.protocol == CONTEXT_PROTOCOL
    assert first.receipt.assembled_at == 100
    assert first.receipt.tokens_used == sum(
        len(item.content.encode("utf-8")) for item in first.entries
    )
    assert len(first.receipt.assembly_digest) == 64
    assert all(len(item.content_digest) == 64 for item in first.entries)

    later = assembler.assemble(
        task_id="task-1",
        scope_id="scope-1",
        fragments=fragments,
        token_budget=1_024,
        classification_boundary=ContextClassification.PRIVATE,
        now=101,
    )
    assert later.receipt.assembled_at == 101
    assert later.receipt.assembly_digest != first.receipt.assembly_digest


def test_context_assembly_reports_every_non_admitted_reason() -> None:
    fragments = (
        _fragment("task", "task", required=True, priority=100),
        _fragment("keep", "keep", priority=90),
        _fragment(
            "duplicate",
            "keep",
            priority=80,
            provenance="test://keep",
        ),
        _fragment("expired", "old", expires_at=100, priority=70),
        _fragment("other-scope", "private", scope_id="scope-2", priority=60),
        _fragment(
            "restricted",
            "secret",
            classification=ContextClassification.RESTRICTED,
            priority=50,
        ),
        _fragment("budget", "does-not-fit", priority=40),
    )

    assembly = ContextAssembler().assemble(
        task_id="task-1",
        scope_id="scope-1",
        fragments=fragments,
        token_budget=8,
        classification_boundary=ContextClassification.PRIVATE,
        now=100,
    )

    assert [item.fragment_id for item in assembly.entries] == ["task", "keep"]
    assert {item.fragment_id: item.reason for item in assembly.receipt.dropped} == {
        "duplicate": ContextDropReason.DUPLICATE_CONTENT,
        "expired": ContextDropReason.EXPIRED,
        "other-scope": ContextDropReason.SCOPE_MISMATCH,
        "restricted": ContextDropReason.CLASSIFICATION_BOUNDARY,
        "budget": ContextDropReason.TOKEN_BUDGET,
    }
    receipt = context_receipt_mapping(assembly.receipt)
    assert receipt["scope_id"] == "scope-1"
    assert receipt["assembled_at"] == 100
    assert receipt["tokens_used"] == 8
    assert "content" not in receipt
    assert context_entry_mapping(assembly.entries[1])["trust"] == "untrusted"


def test_required_context_fails_closed_instead_of_disappearing() -> None:
    assembler = ContextAssembler()
    required = _fragment(
        "task",
        "required",
        required=True,
        classification=ContextClassification.RESTRICTED,
    )

    with pytest.raises(ValueError, match="required context fragment"):
        assembler.assemble(
            task_id="task-1",
            scope_id="scope-1",
            fragments=(required,),
            token_budget=100,
            classification_boundary=ContextClassification.PRIVATE,
            now=100,
        )
    with pytest.raises(ValueError, match="required context exceeds"):
        assembler.assemble(
            task_id="task-1",
            scope_id="scope-1",
            fragments=(_fragment("task", "required", required=True),),
            token_budget=1,
            classification_boundary=ContextClassification.PRIVATE,
            now=100,
        )


def test_deduplication_never_erases_different_trust_or_provenance() -> None:
    assembly = ContextAssembler().assemble(
        task_id="task-1",
        scope_id="scope-1",
        fragments=(
            _fragment(
                "trusted-memory",
                "same statement",
                source=ContextSource.MEMORY,
                trust=ContextTrust.TRUSTED,
                provenance="owner-memory://record-1",
            ),
            _fragment(
                "untrusted-retrieval",
                "same statement",
                source=ContextSource.RETRIEVAL,
                trust=ContextTrust.UNTRUSTED,
                provenance="retrieval://document-1",
            ),
        ),
        token_budget=100,
        classification_boundary=ContextClassification.PRIVATE,
        now=100,
    )

    assert len(assembly.entries) == 2
    assert assembly.entries[0].content_digest == assembly.entries[1].content_digest
    assert assembly.entries[0].fragment_digest != assembly.entries[1].fragment_digest
    assert len(set(assembly.receipt.included_digests)) == 2


def test_context_rejects_ambiguous_or_invalid_contracts() -> None:
    assembler = ContextAssembler()
    fragment = _fragment("same", "content")

    with pytest.raises(ValueError, match="IDs must be unique"):
        assembler.assemble(
            task_id="task-1",
            scope_id="scope-1",
            fragments=(fragment, fragment),
            token_budget=100,
            classification_boundary=ContextClassification.PRIVATE,
            now=100,
        )
    with pytest.raises(ValueError, match="priority"):
        _fragment("invalid", "content", priority=True)
    with pytest.raises(ValueError, match="bounded non-empty"):
        _fragment("invalid", "\x00")
