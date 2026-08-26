from collections.abc import Iterator
import json
from pathlib import Path
import shutil
import sys
import uuid

import pytest

from moira.process_tools import AllowlistedProcessRunner


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def process_root() -> Iterator[Path]:
    root = REPOSITORY_ROOT / "research" / "runs" / f"process-tools-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _target(command: str, arguments: list[str]) -> str:
    return json.dumps({"command": command, "args": arguments})


def test_process_runner_executes_allowlisted_argv_without_a_shell(
    process_root: Path,
) -> None:
    runner = AllowlistedProcessRunner(process_root, (sys.executable,))
    target = _target(
        sys.executable,
        ["-c", "import sys; print(sys.argv[1])", "& echo not-a-shell"],
    )

    result = json.loads(runner.run(target, b""))

    assert result == {
        "command": sys.executable,
        "exit_code": 0,
        "output": "& echo not-a-shell\r\n" if sys.platform == "win32" else "& echo not-a-shell\n",
    }
    assert runner.tool().tool_id == "process.run"


def test_process_runner_rejects_unlisted_or_malformed_commands(
    process_root: Path,
) -> None:
    runner = AllowlistedProcessRunner(process_root, (sys.executable,))

    with pytest.raises(ValueError, match="not allowlisted"):
        runner.run(_target("not-allowed", []), b"")
    with pytest.raises(ValueError, match="exact command and args"):
        runner.run(json.dumps({"command": sys.executable}), b"")
    with pytest.raises(ValueError, match="does not accept a payload"):
        runner.run(_target(sys.executable, []), b"payload")


def test_process_runner_enforces_timeout(process_root: Path) -> None:
    runner = AllowlistedProcessRunner(
        process_root, (sys.executable,), timeout_seconds=0.05
    )

    with pytest.raises(TimeoutError, match="time budget"):
        runner.run(
            _target(sys.executable, ["-c", "import time; time.sleep(2)"]), b""
        )


def test_process_runner_enforces_output_limit(process_root: Path) -> None:
    runner = AllowlistedProcessRunner(
        process_root, (sys.executable,), output_limit=64
    )

    with pytest.raises(ValueError, match="output limit"):
        runner.run(_target(sys.executable, ["-c", "print('x' * 10000)"]), b"")
