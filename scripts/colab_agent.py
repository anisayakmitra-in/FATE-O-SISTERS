#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 ANISAYAK MITRA <https://github.com/anisayakmitra-in>
#
# FATE-O-SISTERS (MOIRA) — Google Colab Remote Execution & Telemetry Daemon
# Runs inside Google Colab and allows Antigravity (`scripts/colab_ctl.py`) to
# remotely execute commands, stream live logs, monitor GPU VRAM/utilization,
# and manage training jobs via Hugging Face Hub (`colab_control/`).

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "anisayakmitra-in/FATE-O-SISTERS"
CMD_PATH_IN_REPO = "colab_control/cmd.json"
STATUS_PATH_IN_REPO = "colab_control/status.json"
LOG_PATH_IN_REPO = "colab_control/output.log"
BOOTSTRAP_PATH_IN_REPO = "colab_control/bootstrap.json"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _try_bootstrap_token(repo_id: str = DEFAULT_REPO_ID) -> str | None:
    """Fetch obfuscated bootstrap credential from repo so Colab connects with zero prompts."""
    import base64

    try:
        from huggingface_hub import hf_hub_download

        local_file = hf_hub_download(
            repo_id=repo_id,
            filename=BOOTSTRAP_PATH_IN_REPO,
            repo_type="model",
            force_download=True,
        )
        data = json.loads(Path(local_file).read_text(encoding="utf-8"))
        raw = base64.b64decode(data.get("payload", ""))
        key = (repo_id + "::moira-369-bridge").encode("utf-8")
        decoded = bytes(b ^ key[i % len(key)] for i, b in enumerate(raw)).decode("utf-8")
        if decoded.startswith("hf_"):
            os.environ["HF_TOKEN"] = decoded
            print("[colab-agent] Authenticated automatically via Antigravity bridge!", flush=True)
            return decoded
    except Exception:
        pass
    return None


def _resolve_hf_token(repo_id: str = DEFAULT_REPO_ID) -> str:
    """Resolve HF_TOKEN from environment, Antigravity bootstrap, Colab Secrets, cache, or login."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token.startswith("hf_"):
        return token

    # Check if user accidentally set the token value as an environment variable key
    for k, v in os.environ.items():
        if k.startswith("hf_") and len(k) > 20:
            os.environ["HF_TOKEN"] = k
            return k
        if isinstance(v, str) and v.strip().startswith("hf_") and len(v.strip()) > 20:
            os.environ["HF_TOKEN"] = v.strip()
            return v.strip()

    # Try zero-prompt Antigravity bootstrap file on Hugging Face Hub
    boot_tok = _try_bootstrap_token(repo_id)
    if boot_tok:
        return boot_tok

    # Try Google Colab userdata secrets
    try:
        from google.colab import userdata  # type: ignore

        for secret_name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN", "hf_token"):
            try:
                val = userdata.get(secret_name)
                if val and val.strip():
                    token = val.strip()
                    os.environ["HF_TOKEN"] = token
                    print(f"[colab-agent] Loaded token from Colab Secret '{secret_name}'.", flush=True)
                    return token
            except Exception:
                continue
    except ImportError:
        pass

    # Try huggingface_hub cached token
    try:
        from huggingface_hub import get_token

        cached = get_token()
        if cached and cached.strip():
            token = cached.strip()
            os.environ["HF_TOKEN"] = token
            print("[colab-agent] Loaded token from Hugging Face cache.", flush=True)
            return token
    except Exception:
        pass

    # Interactive fallback in Colab notebook
    try:
        from huggingface_hub import get_token, login

        print("[colab-agent] Please enter your Hugging Face Write token once to pair with Antigravity:", flush=True)
        login(add_to_git_credential=False)
        cached = get_token()
        if cached and cached.strip():
            token = cached.strip()
            os.environ["HF_TOKEN"] = token
            return token
    except Exception as exc:
        print(f"[colab-agent] Warning: interactive login failed: {exc}", flush=True)

    raise RuntimeError(
        "Could not resolve HF_TOKEN. Set HF_TOKEN in Colab Secrets (key icon on left sidebar) "
        "or run `from huggingface_hub import login; login()` first."
    )


def _gpu_telemetry() -> dict[str, Any]:
    """Query nvidia-smi for GPU name, utilization, and VRAM usage."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        if out:
            first = out.splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 5:
                return {
                    "gpu": parts[0],
                    "vram_used_mb": int(float(parts[1])),
                    "vram_total_mb": int(float(parts[2])),
                    "gpu_util_pct": int(float(parts[3])),
                    "gpu_temp_c": int(float(parts[4])),
                }
    except Exception:
        pass
    return {
        "gpu": "CPU / none detected",
        "vram_used_mb": 0,
        "vram_total_mb": 0,
        "gpu_util_pct": 0,
        "gpu_temp_c": 0,
    }


def _git_head(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return "unknown"


class ColabAgent:
    def __init__(self, repo_id: str, token: str, workdir: Path, poll_interval: float = 3.0) -> None:
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.token = token
        self.workdir = workdir.resolve()
        self.poll_interval = poll_interval
        self.api = HfApi(token=token)
        self.agent_id = f"colab-{socket.gethostname()}-{os.getpid()}"

        self.state = "idle"
        self.active_cmd_id: str | None = None
        self.last_cmd_id: str | None = None
        self.current_command: str = ""
        self.returncode: int | None = None
        self.started_at: str | None = None
        self.started_mono: float | None = None

        self._proc: subprocess.Popen[str] | None = None
        self._log_lock = threading.Lock()
        self._log_buffer: list[str] = []
        self._log_dirty = False
        self._last_flush_mono = 0.0

    def _append_log(self, line: str) -> None:
        with self._log_lock:
            self._log_buffer.append(line)
            if len(self._log_buffer) > 4000:
                self._log_buffer = self._log_buffer[-4000:]
            self._log_dirty = True

    def _get_log_snapshot(self) -> tuple[str, list[str]]:
        with self._log_lock:
            tail = [ln.rstrip("\r\n") for ln in self._log_buffer[-60:]]
            full_text = "".join(self._log_buffer)
            if len(full_text) > 256_000:
                full_text = full_text[-256_000:]
            self._log_dirty = False
            return full_text, tail

    def _fetch_remote_json(self, path_in_repo: str) -> dict[str, Any] | None:
        from huggingface_hub import hf_hub_download

        try:
            local_file = hf_hub_download(
                repo_id=self.repo_id,
                filename=path_in_repo,
                repo_type="model",
                token=self.token,
                force_download=True,
            )
            return json.loads(Path(local_file).read_text(encoding="utf-8"))
        except Exception:
            return None

    def flush_telemetry(self, commit_msg: str, include_log: bool = True) -> None:
        from huggingface_hub import CommitOperationAdd

        gpu_info = _gpu_telemetry()
        full_log, tail_lines = self._get_log_snapshot()
        elapsed = (
            round(time.monotonic() - self.started_mono, 2)
            if self.started_mono is not None and self.state == "running"
            else None
        )
        status_payload = {
            "agent_id": self.agent_id,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "workdir": str(self.workdir),
            "git_head": _git_head(self.workdir),
            **gpu_info,
            "state": self.state,
            "active_cmd_id": self.active_cmd_id,
            "last_cmd_id": self.last_cmd_id,
            "command": self.current_command,
            "returncode": self.returncode,
            "pid": self._proc.pid if self._proc is not None else None,
            "started_at": self.started_at,
            "updated_at": _utc_now(),
            "elapsed_sec": elapsed,
            "tail_lines": tail_lines,
        }
        operations = [
            CommitOperationAdd(
                path_in_repo=STATUS_PATH_IN_REPO,
                path_or_fileobj=io.BytesIO(json.dumps(status_payload, indent=2).encode("utf-8")),
            )
        ]
        if include_log and full_log:
            operations.append(
                CommitOperationAdd(
                    path_in_repo=LOG_PATH_IN_REPO,
                    path_or_fileobj=io.BytesIO(full_log.encode("utf-8", errors="replace")),
                )
            )
        try:
            self.api.create_commit(
                repo_id=self.repo_id,
                repo_type="model",
                operations=operations,
                commit_message=commit_msg,
            )
            self._last_flush_mono = time.monotonic()
        except Exception as exc:
            print(f"[colab-agent] Telemetry push warning: {exc}", flush=True)

    def kill_active_process(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        print(f"[colab-agent] Terminating running process PID {proc.pid}...", flush=True)
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
        except Exception as exc:
            print(f"[colab-agent] Error killing process: {exc}", flush=True)

    def _reader_thread(self, stream: Any) -> None:
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                sys.stdout.write(line)
                sys.stdout.flush()
                self._append_log(line)
        except Exception:
            pass

    def start_command(self, cmd_obj: dict[str, Any]) -> None:
        cmd_id = str(cmd_obj.get("cmd_id", ""))
        action = str(cmd_obj.get("action", "exec"))
        command = str(cmd_obj.get("command", "")).strip()

        if action == "kill":
            self.kill_active_process()
            self.state = "killed"
            self.last_cmd_id = cmd_id
            self.active_cmd_id = None
            self._append_log(f"\n[colab-agent] Process killed by command {cmd_id} at {_utc_now()}\n")
            self.flush_telemetry(f"colab-agent: killed ({cmd_id})", include_log=True)
            return

        if action == "ping":
            self.last_cmd_id = cmd_id
            self.flush_telemetry(f"colab-agent: pong ({cmd_id})", include_log=False)
            return

        if action == "pull":
            command = "git fetch origin main && git reset --hard origin/main"

        if not command:
            return

        # If a process is currently running, terminate it before launching the new command
        if self._proc is not None and self._proc.poll() is None:
            self.kill_active_process()

        with self._log_lock:
            self._log_buffer.clear()

        header = (
            f"================================================================\n"
            f"[colab-agent] Executing cmd_id={cmd_id} at {_utc_now()}\n"
            f"[colab-agent] Command: {command}\n"
            f"[colab-agent] Workdir: {self.workdir}\n"
            f"================================================================\n"
        )
        sys.stdout.write(header)
        sys.stdout.flush()
        self._append_log(header)

        env = os.environ.copy()
        env["HF_TOKEN"] = self.token
        env["PYTHONUNBUFFERED"] = "1"
        extra_env = cmd_obj.get("env")
        if isinstance(extra_env, dict):
            for k, v in extra_env.items():
                env[str(k)] = str(v)

        cwd = Path(cmd_obj.get("cwd") or self.workdir)
        if not cwd.exists():
            cwd = self.workdir

        popen_kwargs: dict[str, Any] = {
            "shell": True,
            "cwd": str(cwd),
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        if hasattr(os, "setsid"):
            popen_kwargs["preexec_fn"] = os.setsid

        self._proc = subprocess.Popen(command, **popen_kwargs)
        self.state = "running"
        self.active_cmd_id = cmd_id
        self.last_cmd_id = cmd_id
        self.current_command = command
        self.returncode = None
        self.started_at = _utc_now()
        self.started_mono = time.monotonic()

        t = threading.Thread(target=self._reader_thread, args=(self._proc.stdout,), daemon=True)
        t.start()

        self.flush_telemetry(f"colab-agent: started {cmd_id}", include_log=True)

    def serve_forever(self) -> None:
        print(f"[colab-agent] Starting daemon ({self.agent_id}) for repo {self.repo_id}", flush=True)
        prev_status = self._fetch_remote_json(STATUS_PATH_IN_REPO)
        prev_completed_id = None
        if isinstance(prev_status, dict) and prev_status.get("state") in ("completed", "failed", "killed"):
            prev_completed_id = prev_status.get("last_cmd_id")

        initial_cmd = self._fetch_remote_json(CMD_PATH_IN_REPO)
        if isinstance(initial_cmd, dict):
            init_id = str(initial_cmd.get("cmd_id", ""))
            run_on_boot = bool(initial_cmd.get("run_on_boot", False))
            if run_on_boot and init_id and init_id != prev_completed_id:
                print(f"[colab-agent] Found pending boot command: {init_id}", flush=True)
                self.last_cmd_id = None
            else:
                self.last_cmd_id = init_id

        self.flush_telemetry(f"colab-agent: online ({self.agent_id})", include_log=False)
        print("[colab-agent] Online and connected to Antigravity! Waiting for commands...", flush=True)

        while True:
            try:
                # 1. Check if active process completed
                if self._proc is not None and self.state == "running":
                    rc = self._proc.poll()
                    if rc is not None:
                        time.sleep(0.5)  # allow stdout reader thread to drain
                        self.returncode = rc
                        self.state = "completed" if rc == 0 else "failed"
                        footer = (
                            f"\n================================================================\n"
                            f"[colab-agent] Command {self.active_cmd_id} finished with exit code {rc} at {_utc_now()}\n"
                            f"================================================================\n"
                        )
                        sys.stdout.write(footer)
                        sys.stdout.flush()
                        self._append_log(footer)
                        done_id = self.active_cmd_id
                        self.active_cmd_id = None
                        self.flush_telemetry(
                            f"colab-agent: {self.state} {done_id} (rc={rc})",
                            include_log=True,
                        )
                    elif time.monotonic() - self._last_flush_mono >= 10.0 and self._log_dirty:
                        self.flush_telemetry(
                            f"colab-agent: progress {self.active_cmd_id}",
                            include_log=True,
                        )

                # 2. Poll remote command file
                cmd_obj = self._fetch_remote_json(CMD_PATH_IN_REPO)
                if isinstance(cmd_obj, dict):
                    cmd_id = str(cmd_obj.get("cmd_id", ""))
                    if cmd_id and cmd_id != self.last_cmd_id:
                        self.start_command(cmd_obj)

            except KeyboardInterrupt:
                print("\n[colab-agent] Stopping daemon on KeyboardInterrupt...", flush=True)
                self.kill_active_process()
                break
            except Exception as exc:
                print(f"[colab-agent] Loop warning: {exc}", flush=True)

            time.sleep(self.poll_interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="FATE-O-SISTERS Google Colab Remote Control Agent")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face Hub repo ID")
    parser.add_argument("--workdir", default=".", help="Working directory for commands")
    parser.add_argument("--poll-interval", type=float, default=3.0, help="Seconds between command polls")
    args = parser.parse_args()

    token = _resolve_hf_token(args.repo_id)
    agent = ColabAgent(
        repo_id=args.repo_id,
        token=token,
        workdir=Path(args.workdir),
        poll_interval=args.poll_interval,
    )
    agent.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
