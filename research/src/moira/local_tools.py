"""Workspace-confined tools for standalone MOIRA runs."""

from __future__ import annotations

import os
from pathlib import Path
import stat

from .agent import MoiraTool


_DEFAULT_READ_LIMIT = 1024 * 1024
_DEFAULT_SEARCH_BYTE_LIMIT = 16 * 1024 * 1024
_DEFAULT_SEARCH_FILE_LIMIT = 2_000
_DEFAULT_SEARCH_MATCH_LIMIT = 200


class ReadOnlyWorkspace:
    """Bounded read and text search rooted at one operator-selected directory."""

    def __init__(
        self,
        root: Path,
        *,
        read_limit: int = _DEFAULT_READ_LIMIT,
        search_byte_limit: int = _DEFAULT_SEARCH_BYTE_LIMIT,
        search_file_limit: int = _DEFAULT_SEARCH_FILE_LIMIT,
        search_match_limit: int = _DEFAULT_SEARCH_MATCH_LIMIT,
    ) -> None:
        if not isinstance(root, Path):
            raise ValueError("workspace root must be a Path")
        self.root = _validate_root(root)
        self.read_limit = _positive_limit("read_limit", read_limit)
        self.search_byte_limit = _positive_limit(
            "search_byte_limit", search_byte_limit
        )
        self.search_file_limit = _positive_limit(
            "search_file_limit", search_file_limit
        )
        self.search_match_limit = _positive_limit(
            "search_match_limit", search_match_limit
        )

    def tools(self) -> tuple[MoiraTool, MoiraTool]:
        return (
            MoiraTool(
                tool_id="workspace.read",
                capability="filesystem_read",
                operation="read",
                target_kind="workspace-relative-path",
                handler=self.read,
                description=(
                    "Read one regular file under the selected workspace. "
                    "Target must be a relative path."
                ),
            ),
            MoiraTool(
                tool_id="workspace.search",
                capability="filesystem_read",
                operation="search",
                target_kind="literal-text-query",
                handler=self.search,
                description=(
                    "Search regular UTF-8 text files under the selected workspace "
                    "for a literal query."
                ),
            ),
        )

    def read(self, target: str, payload: bytes) -> bytes:
        _require_empty_payload(payload)
        path = self._regular_file(target)
        size = path.stat().st_size
        if size > self.read_limit:
            raise ValueError("workspace file exceeds the read limit")
        with path.open("rb") as handle:
            content = handle.read(self.read_limit + 1)
        if len(content) > self.read_limit:
            raise ValueError("workspace file exceeds the read limit")
        return content

    def search(self, target: str, payload: bytes) -> bytes:
        _require_empty_payload(payload)
        if (
            not isinstance(target, str)
            or not target
            or "\x00" in target
            or len(target.encode("utf-8")) > 1_024
        ):
            raise ValueError("search query must be bounded non-empty text")

        files_seen = 0
        bytes_seen = 0
        matches: list[str] = []
        for directory, names, filenames in os.walk(self.root, followlinks=False):
            names[:] = sorted(
                name
                for name in names
                if not _is_reparse_point(Path(directory) / name)
            )
            for filename in sorted(filenames):
                path = Path(directory) / filename
                if _is_reparse_point(path):
                    continue
                try:
                    status = path.stat()
                except OSError:
                    continue
                if not stat.S_ISREG(status.st_mode):
                    continue
                files_seen += 1
                if files_seen > self.search_file_limit:
                    raise ValueError("workspace search exceeds the file limit")
                bytes_seen += status.st_size
                if bytes_seen > self.search_byte_limit:
                    raise ValueError("workspace search exceeds the byte limit")
                try:
                    content = path.read_bytes()
                    text = content.decode("utf-8")
                except (OSError, UnicodeError):
                    continue
                relative = path.relative_to(self.root).as_posix()
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if target in line:
                        excerpt = line[:2_000]
                        matches.append(f"{relative}:{line_number}:{excerpt}")
                        if len(matches) >= self.search_match_limit:
                            return "\n".join(matches).encode("utf-8")
        return "\n".join(matches).encode("utf-8")

    def _regular_file(self, target: str) -> Path:
        if (
            not isinstance(target, str)
            or not target
            or "\x00" in target
            or len(target.encode("utf-8")) > 4_096
        ):
            raise ValueError("workspace target must be a bounded relative path")
        relative = Path(target)
        if relative.is_absolute() or relative.anchor or ".." in relative.parts:
            raise ValueError("workspace target must stay below the selected root")
        candidate = self.root.joinpath(relative)
        current = self.root
        for part in relative.parts:
            current = current / part
            if _is_reparse_point(current):
                raise ValueError("workspace target cannot traverse a link")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.root)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError("workspace target is unavailable or outside the root") from error
        if not resolved.is_file():
            raise ValueError("workspace target must be a regular file")
        return resolved


def _validate_root(root: Path) -> Path:
    try:
        if _is_reparse_point(root):
            raise ValueError("workspace root cannot be a link")
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("workspace root is unavailable") from error
    if not resolved.is_dir():
        raise ValueError("workspace root must be a directory")
    return resolved


def _is_reparse_point(path: Path) -> bool:
    try:
        status = path.lstat()
    except OSError:
        return False
    attributes = getattr(status, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_flag)


def _positive_limit(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_empty_payload(payload: bytes) -> None:
    if not isinstance(payload, bytes) or payload:
        raise ValueError("read-only workspace tools do not accept a payload")


__all__ = ["ReadOnlyWorkspace"]
