"""Deterministic scoped context assembly for MOIRA's native agent runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Iterable


CONTEXT_PROTOCOL = "moira-context-v1"
MAX_CONTEXT_FRAGMENTS = 1_024
MAX_CONTEXT_FRAGMENT_BYTES = 16 * 1024 * 1024
MAX_CONTEXT_TOKENS = 1_000_000


class ContextSource(str, Enum):
    TASK = "task"
    USER = "user"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    MODEL = "model"
    RUNTIME = "runtime"


class ContextTrust(str, Enum):
    TRUSTED = "trusted"
    USER_SUPPLIED = "user_supplied"
    UNTRUSTED = "untrusted"


class ContextClassification(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    RESTRICTED = "restricted"


class ContextDropReason(str, Enum):
    EXPIRED = "expired"
    SCOPE_MISMATCH = "scope_mismatch"
    CLASSIFICATION_BOUNDARY = "classification_boundary"
    DUPLICATE_CONTENT = "duplicate_content"
    TOKEN_BUDGET = "token_budget"


_CLASSIFICATION_RANK = {
    ContextClassification.PUBLIC: 0,
    ContextClassification.PRIVATE: 1,
    ContextClassification.RESTRICTED: 2,
}


@dataclass(frozen=True, slots=True)
class ContextFragment:
    fragment_id: str
    source: ContextSource
    trust: ContextTrust
    classification: ContextClassification
    scope_id: str
    content: str
    provenance: str
    priority: int = 50
    expires_at: int | None = None
    required: bool = False

    def __post_init__(self) -> None:
        _bounded_text("fragment_id", self.fragment_id, 256)
        _bounded_text("scope_id", self.scope_id, 256)
        _bounded_text("content", self.content, MAX_CONTEXT_FRAGMENT_BYTES)
        _bounded_text("provenance", self.provenance, 2_048)
        if not isinstance(self.source, ContextSource):
            raise ValueError("context source is invalid")
        if not isinstance(self.trust, ContextTrust):
            raise ValueError("context trust is invalid")
        if not isinstance(self.classification, ContextClassification):
            raise ValueError("context classification is invalid")
        if (
            type(self.priority) is not int
            or not 0 <= self.priority <= 100
        ):
            raise ValueError("context priority must be between 0 and 100")
        if self.expires_at is not None and (
            type(self.expires_at) is not int or self.expires_at < 0
        ):
            raise ValueError("context expiry must be a non-negative integer")
        if type(self.required) is not bool:
            raise ValueError("context required flag must be boolean")

    @property
    def content_digest(self) -> str:
        return _digest("moira-context-content-v1", self.content.encode("utf-8"))

    @property
    def fragment_digest(self) -> str:
        return _digest(
            "moira-context-fragment-v1",
            _canonical_json(
                {
                    "fragment_id": self.fragment_id,
                    "source": self.source.value,
                    "trust": self.trust.value,
                    "classification": self.classification.value,
                    "scope_id": self.scope_id,
                    "content_digest": self.content_digest,
                    "provenance": self.provenance,
                    "priority": self.priority,
                    "expires_at": self.expires_at,
                    "required": self.required,
                }
            ),
        )

    @property
    def deduplication_digest(self) -> str:
        return _digest(
            "moira-context-deduplication-v1",
            _canonical_json(
                {
                    "source": self.source.value,
                    "trust": self.trust.value,
                    "classification": self.classification.value,
                    "scope_id": self.scope_id,
                    "content_digest": self.content_digest,
                    "provenance": self.provenance,
                }
            ),
        )

    @property
    def token_cost(self) -> int:
        """Conservative exact cost for MOIRA's UTF-8 byte tokenizer."""
        return len(self.content.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class ContextEntry:
    fragment_id: str
    source: ContextSource
    trust: ContextTrust
    classification: ContextClassification
    scope_id: str
    content: str
    provenance: str
    priority: int
    content_digest: str
    fragment_digest: str
    token_cost: int


@dataclass(frozen=True, slots=True)
class ContextDrop:
    fragment_id: str
    reason: ContextDropReason
    content_digest: str
    fragment_digest: str


@dataclass(frozen=True, slots=True)
class ContextReceipt:
    protocol: str
    task_id: str
    scope_id: str
    classification_boundary: ContextClassification
    assembled_at: int
    token_budget: int
    tokens_used: int
    included_ids: tuple[str, ...]
    included_digests: tuple[str, ...]
    dropped: tuple[ContextDrop, ...]
    assembly_digest: str


@dataclass(frozen=True, slots=True)
class ContextAssembly:
    entries: tuple[ContextEntry, ...]
    receipt: ContextReceipt


class ContextAssembler:
    """Admit, rank, deduplicate, and budget context without model calls."""

    def assemble(
        self,
        *,
        task_id: str,
        scope_id: str,
        fragments: Iterable[ContextFragment],
        token_budget: int,
        classification_boundary: ContextClassification,
        now: int,
    ) -> ContextAssembly:
        _bounded_text("task_id", task_id, 256)
        _bounded_text("scope_id", scope_id, 256)
        if (
            type(token_budget) is not int
            or not 1 <= token_budget <= MAX_CONTEXT_TOKENS
        ):
            raise ValueError("context token budget is outside its bounded range")
        if not isinstance(classification_boundary, ContextClassification):
            raise ValueError("context classification boundary is invalid")
        if type(now) is not int or now < 0:
            raise ValueError("context assembly time must be a non-negative integer")
        try:
            candidates = tuple(fragments)
        except TypeError as error:
            raise ValueError("context fragments must be iterable") from error
        if not candidates or len(candidates) > MAX_CONTEXT_FRAGMENTS:
            raise ValueError("context fragments are outside their bounded count")
        if any(not isinstance(item, ContextFragment) for item in candidates):
            raise ValueError("context fragments must be ContextFragment values")
        identifiers = [item.fragment_id for item in candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("context fragment IDs must be unique")

        ordered = sorted(
            candidates,
            key=lambda item: (
                not item.required,
                -item.priority,
                item.fragment_id,
            ),
        )
        entries: list[ContextEntry] = []
        dropped: list[ContextDrop] = []
        seen_fragments: set[str] = set()
        tokens_used = 0
        for fragment in ordered:
            reason = self._inadmissible_reason(
                fragment,
                scope_id=scope_id,
                classification_boundary=classification_boundary,
                now=now,
                seen_fragments=seen_fragments,
            )
            if reason is not None:
                if fragment.required:
                    raise ValueError(
                        f"required context fragment {fragment.fragment_id!r} "
                        f"was rejected: {reason.value}"
                    )
                dropped.append(_drop(fragment, reason))
                continue
            if tokens_used + fragment.token_cost > token_budget:
                if fragment.required:
                    raise ValueError("required context exceeds the token budget")
                dropped.append(_drop(fragment, ContextDropReason.TOKEN_BUDGET))
                seen_fragments.add(fragment.deduplication_digest)
                continue
            seen_fragments.add(fragment.deduplication_digest)
            tokens_used += fragment.token_cost
            entries.append(
                ContextEntry(
                    fragment_id=fragment.fragment_id,
                    source=fragment.source,
                    trust=fragment.trust,
                    classification=fragment.classification,
                    scope_id=fragment.scope_id,
                    content=fragment.content,
                    provenance=fragment.provenance,
                    priority=fragment.priority,
                    content_digest=fragment.content_digest,
                    fragment_digest=fragment.fragment_digest,
                    token_cost=fragment.token_cost,
                )
            )
        if not entries:
            raise ValueError("context assembly admitted no fragments")

        receipt_value = {
            "protocol": CONTEXT_PROTOCOL,
            "task_id": task_id,
            "scope_id": scope_id,
            "classification_boundary": classification_boundary.value,
            "assembled_at": now,
            "token_budget": token_budget,
            "tokens_used": tokens_used,
            "included": [
                {
                    "fragment_id": entry.fragment_id,
                    "source": entry.source.value,
                    "trust": entry.trust.value,
                    "classification": entry.classification.value,
                    "scope_id": entry.scope_id,
                    "content_digest": entry.content_digest,
                    "fragment_digest": entry.fragment_digest,
                    "provenance": entry.provenance,
                    "priority": entry.priority,
                    "token_cost": entry.token_cost,
                }
                for entry in entries
            ],
            "dropped": [
                {
                    "fragment_id": item.fragment_id,
                    "reason": item.reason.value,
                    "content_digest": item.content_digest,
                    "fragment_digest": item.fragment_digest,
                }
                for item in dropped
            ],
        }
        assembly_digest = _digest(
            "moira-context-assembly-v1",
            _canonical_json(receipt_value),
        )
        return ContextAssembly(
            entries=tuple(entries),
            receipt=ContextReceipt(
                protocol=CONTEXT_PROTOCOL,
                task_id=task_id,
                scope_id=scope_id,
                classification_boundary=classification_boundary,
                assembled_at=now,
                token_budget=token_budget,
                tokens_used=tokens_used,
                included_ids=tuple(item.fragment_id for item in entries),
                included_digests=tuple(item.fragment_digest for item in entries),
                dropped=tuple(dropped),
                assembly_digest=assembly_digest,
            ),
        )

    @staticmethod
    def _inadmissible_reason(
        fragment: ContextFragment,
        *,
        scope_id: str,
        classification_boundary: ContextClassification,
        now: int,
        seen_fragments: set[str],
    ) -> ContextDropReason | None:
        if fragment.expires_at is not None and fragment.expires_at <= now:
            return ContextDropReason.EXPIRED
        if fragment.scope_id not in {"public", scope_id}:
            return ContextDropReason.SCOPE_MISMATCH
        if (
            _CLASSIFICATION_RANK[fragment.classification]
            > _CLASSIFICATION_RANK[classification_boundary]
        ):
            return ContextDropReason.CLASSIFICATION_BOUNDARY
        if fragment.deduplication_digest in seen_fragments:
            return ContextDropReason.DUPLICATE_CONTENT
        return None


def context_entry_mapping(entry: ContextEntry) -> dict[str, object]:
    value = asdict(entry)
    value["source"] = entry.source.value
    value["trust"] = entry.trust.value
    value["classification"] = entry.classification.value
    return value


def context_receipt_mapping(receipt: ContextReceipt) -> dict[str, object]:
    return {
        "protocol": receipt.protocol,
        "task_id": receipt.task_id,
        "scope_id": receipt.scope_id,
        "classification_boundary": receipt.classification_boundary.value,
        "assembled_at": receipt.assembled_at,
        "token_budget": receipt.token_budget,
        "tokens_used": receipt.tokens_used,
        "included_ids": list(receipt.included_ids),
        "included_digests": list(receipt.included_digests),
        "dropped": [
            {
                "fragment_id": item.fragment_id,
                "reason": item.reason.value,
                "content_digest": item.content_digest,
                "fragment_digest": item.fragment_digest,
            }
            for item in receipt.dropped
        ],
        "assembly_digest": receipt.assembly_digest,
    }


def _drop(fragment: ContextFragment, reason: ContextDropReason) -> ContextDrop:
    return ContextDrop(
        fragment.fragment_id,
        reason,
        fragment.content_digest,
        fragment.fragment_digest,
    )


def _bounded_text(name: str, value: object, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValueError(f"context {name} must be bounded non-empty text")


def _digest(domain: str, content: bytes) -> str:
    return hashlib.sha256(domain.encode("ascii") + b"\x00" + content).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = [
    "CONTEXT_PROTOCOL",
    "MAX_CONTEXT_FRAGMENTS",
    "MAX_CONTEXT_TOKENS",
    "ContextAssembler",
    "ContextAssembly",
    "ContextClassification",
    "ContextDrop",
    "ContextDropReason",
    "ContextEntry",
    "ContextFragment",
    "ContextReceipt",
    "ContextSource",
    "ContextTrust",
    "context_entry_mapping",
    "context_receipt_mapping",
]
