"""Approval-gated, allowlisted, no-shell process execution for MOIRA."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

from .agent import MoiraTool


_DEFAULT_OUTPUT_LIMIT = 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_COMMAND_TARGET_BYTES = 4_096
_MAX_ARGUMENTS = 128
_MAX_ARGUMENT_BYTES = 4_096


class AllowlistedProcessRunner:
    """Run exact argv commands selected from an operator-provided allowlist."""

    def __init__(
        self,
        workspace: Path,
        allowed_commands: tuple[str, ...],
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        output_limit: int = _DEFAULT_OUTPUT_LIMIT,
    ) -> None:
        if not isinstance(workspace, Path):
            raise ValueError("process workspace must be a Path")
        try:
            self.workspace = workspace.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError("process workspace is unavailable") from error
        if not self.workspace.is_dir():
            raise ValueError("process workspace must be a directory")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not 0 < float(timeout_seconds) <= 600
        ):
            raise ValueError("process timeout must be between 0 and 600 seconds")
        if (
            not isinstance(output_limit, int)
            or isinstance(output_limit, bool)
            or output_limit <= 0
        ):
            raise ValueError("process output limit must be a positive integer")
        if not isinstance(allowed_commands, tuple) or not allowed_commands:
            raise ValueError("at least one process command must be allowlisted")
        self.timeout_seconds = float(timeout_seconds)
        self.output_limit = output_limit
        self._commands: dict[str, str] = {}
        display_commands: list[str] = []
        for command in allowed_commands:
            if (
                not isinstance(command, str)
                or not command
                or "\x00" in command
                or len(command.encode("utf-8")) > 4_096
            ):
                raise ValueError("allowlisted command is invalid")
            resolved = shutil.which(command)
            if resolved is None:
                raise ValueError(f"allowlisted command is unavailable: {command}")
            key = os.path.normcase(command)
            if key in self._commands:
                raise ValueError("allowlisted commands must be unique")
            self._commands[key] = str(Path(resolved).resolve(strict=True))
            display_commands.append(command)
        self._display_commands = tuple(display_commands)

    def tool(self) -> MoiraTool:
        names = ", ".join(sorted(self._display_commands))
        return MoiraTool(
            tool_id="process.run",
            capability="process",
            operation="run",
            target_kind="argv-json",
            handler=self.run,
            description=(
                "Run an allowlisted command without a shell. Target must be JSON "
                f"with exact keys command and args. Allowed commands: {names}. "
                "Requires exact operator approval."
            ),
        )

    def run(self, target: str, payload: bytes) -> bytes:
        if not isinstance(payload, bytes) or payload:
            raise ValueError("process tool does not accept a payload")
        command, arguments = self._parse_target(target)
        executable = self._commands.get(os.path.normcase(command))
        if executable is None:
            raise ValueError("process command is not allowlisted")
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [executable, *arguments],
            cwd=self.workspace,
            env=_minimal_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            creationflags=creation_flags,
        )
        assert process.stdout is not None
        chunks: list[bytes] = []
        total = 0
        overflow = threading.Event()

        def read_output() -> None:
            nonlocal total
            while True:
                chunk = process.stdout.read(64 * 1024)
                if not chunk:
                    return
                total += len(chunk)
                if total > self.output_limit:
                    overflow.set()
                    try:
                        process.kill()
                    except OSError:
                        pass
                    continue
                chunks.append(chunk)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            return_code = process.wait(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait()
            reader.join(timeout=5)
            process.stdout.close()
            raise TimeoutError("process exceeded its time budget") from error
        reader.join(timeout=5)
        process.stdout.close()
        if reader.is_alive():
            raise RuntimeError("process output reader did not terminate")
        if overflow.is_set():
            raise ValueError("process output exceeds the output limit")
        output = b"".join(chunks).decode("utf-8", errors="replace")
        result = {
            "command": command,
            "exit_code": return_code,
            "output": output,
        }
        return json.dumps(
            result, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def _parse_target(self, target: str) -> tuple[str, list[str]]:
        if (
            not isinstance(target, str)
            or not target
            or "\x00" in target
            or len(target.encode("utf-8")) > _MAX_COMMAND_TARGET_BYTES
        ):
            raise ValueError("process target must be bounded JSON")
        try:
            value = json.loads(target)
        except (json.JSONDecodeError, RecursionError) as error:
            raise ValueError("process target is not valid JSON") from error
        if not isinstance(value, dict) or set(value) != {"command", "args"}:
            raise ValueError("process target requires exact command and args fields")
        command = value["command"]
        arguments = value["args"]
        if not isinstance(command, str) or not isinstance(arguments, list):
            raise ValueError("process command and args have invalid types")
        if len(arguments) > _MAX_ARGUMENTS:
            raise ValueError("process target has too many arguments")
        if any(
            not isinstance(argument, str)
            or "\x00" in argument
            or len(argument.encode("utf-8")) > _MAX_ARGUMENT_BYTES
            for argument in arguments
        ):
            raise ValueError("process argument is invalid")
        return command, arguments


def _minimal_environment() -> dict[str, str]:
    environment = {
        "NO_COLOR": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    for name in ("SystemRoot", "WINDIR", "TEMP", "TMP", "PATH", "PATHEXT"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


__all__ = ["AllowlistedProcessRunner"]
