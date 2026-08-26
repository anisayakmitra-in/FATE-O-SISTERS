from collections.abc import Iterator
from pathlib import Path
import shutil
import uuid

import pytest

from moira.local_tools import ReadOnlyWorkspace


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def workspace_root() -> Iterator[Path]:
    root = REPOSITORY_ROOT / "research" / "runs" / f"local-tools-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def test_workspace_read_and_literal_search_are_bounded_to_root(
    workspace_root: Path,
) -> None:
    (workspace_root / "src").mkdir()
    model_path = workspace_root / "src" / "model.py"
    model_path.write_text(
        "class Moira:\n    pass\n", encoding="utf-8"
    )
    workspace = ReadOnlyWorkspace(workspace_root)

    assert workspace.read("src/model.py", b"") == model_path.read_bytes()
    assert workspace.search("Moira", b"") == b"src/model.py:1:class Moira:"
    assert [tool.tool_id for tool in workspace.tools()] == [
        "workspace.read",
        "workspace.search",
    ]


@pytest.mark.parametrize("target", ("../secret.txt", "", "C:\\secret.txt"))
def test_workspace_read_rejects_escape_or_invalid_targets(
    workspace_root: Path, target: str
) -> None:
    workspace = ReadOnlyWorkspace(workspace_root)

    with pytest.raises(ValueError):
        workspace.read(target, b"")


def test_workspace_read_rejects_oversized_file_and_payload(
    workspace_root: Path,
) -> None:
    (workspace_root / "large.txt").write_bytes(b"12345")
    workspace = ReadOnlyWorkspace(workspace_root, read_limit=4)

    with pytest.raises(ValueError, match="read limit"):
        workspace.read("large.txt", b"")
    with pytest.raises(ValueError, match="do not accept a payload"):
        workspace.search("query", b"unexpected")


def test_workspace_search_enforces_file_and_byte_budgets(
    workspace_root: Path,
) -> None:
    (workspace_root / "a.txt").write_text("one", encoding="utf-8")
    (workspace_root / "b.txt").write_text("two", encoding="utf-8")

    with pytest.raises(ValueError, match="file limit"):
        ReadOnlyWorkspace(workspace_root, search_file_limit=1).search("missing", b"")
    with pytest.raises(ValueError, match="byte limit"):
        ReadOnlyWorkspace(workspace_root, search_byte_limit=5).search("missing", b"")


def test_workspace_search_caps_returned_matches(workspace_root: Path) -> None:
    (workspace_root / "matches.txt").write_text("x\nx\nx\n", encoding="utf-8")
    workspace = ReadOnlyWorkspace(workspace_root, search_match_limit=2)

    assert workspace.search("x", b"").splitlines() == [
        b"matches.txt:1:x",
        b"matches.txt:2:x",
    ]
