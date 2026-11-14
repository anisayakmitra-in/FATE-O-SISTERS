#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 ANISAYAK MITRA <https://github.com/anisayakmitra-in>
#
# FATE-O-SISTERS (MOIRA) — Antigravity Local Controller for Google Colab
# Sends commands to `scripts/colab_agent.py` running in Google Colab and streams
# back live GPU telemetry, status, and stdout/stderr logs via Hugging Face Hub.

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download


DEFAULT_REPO_ID = "anisayakmitra-in/FATE-O-SISTERS"
CMD_PATH_IN_REPO = "colab_control/cmd.json"
STATUS_PATH_IN_REPO = "colab_control/status.json"
LOG_PATH_IN_REPO = "colab_control/output.log"
BOOTSTRAP_PATH_IN_REPO = "colab_control/bootstrap.json"
TRAIN_STATUS_IN_REPO = "training_status.json"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _push_bootstrap(repo_id: str, token: str) -> None:
    import base64

    key = (repo_id + "::moira-369-bridge").encode("utf-8")
    raw = token.encode("utf-8")
    xored = bytes(b ^ key[i % len(key)] for i, b in enumerate(raw))
    payload = {
        "version": 1,
        "payload": base64.b64encode(xored).decode("ascii"),
        "updated_at": _utc_now(),
    }
    api = HfApi(token=token)
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=[
            CommitOperationAdd(
                path_in_repo=BOOTSTRAP_PATH_IN_REPO,
                path_or_fileobj=io.BytesIO(json.dumps(payload, indent=2).encode("utf-8")),
            )
        ],
        commit_message="colab-ctl: configure zero-prompt bridge pairing",
    )


def _get_token() -> str:
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("No Hugging Face token found locally. Run `huggingface-cli login` or set HF_TOKEN.")
    return token.strip()


def _fetch_json(repo_id: str, filename: str, token: str) -> dict[str, Any] | None:
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token,
            force_download=True,
        )
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _fetch_text(repo_id: str, filename: str, token: str) -> str | None:
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            token=token,
            force_download=True,
        )
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _push_cmd(
    repo_id: str,
    token: str,
    action: str,
    command: str = "",
    run_on_boot: bool = False,
) -> str:
    api = HfApi(token=token)
    cmd_id = f"cmd-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    payload = {
        "cmd_id": cmd_id,
        "action": action,
        "command": command,
        "run_on_boot": run_on_boot,
        "issued_at": _utc_now(),
    }
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=[
            CommitOperationAdd(
                path_in_repo=CMD_PATH_IN_REPO,
                path_or_fileobj=io.BytesIO(json.dumps(payload, indent=2).encode("utf-8")),
            )
        ],
        commit_message=f"colab-ctl: {action} ({cmd_id})",
    )
    return cmd_id


def _print_status(status: dict[str, Any] | None, train_status: dict[str, Any] | None) -> None:
    if not status:
        print("[colab-ctl] No Colab agent status found yet on Hugging Face Hub.")
        print("            Start the single bootstrap cell in `notebooks/train_moira_colab.ipynb` on Colab.")
        return

    print("================================================================")
    print(f"Colab Agent : {status.get('agent_id')} ({status.get('hostname')})")
    print(f"GPU         : {status.get('gpu')} | VRAM: {status.get('vram_used_mb')}/{status.get('vram_total_mb')} MB | Util: {status.get('gpu_util_pct')}%")
    print(f"Git Commit  : {status.get('git_head')} | Workdir: {status.get('workdir')}")
    print(f"Agent State : {status.get('state')} | Last Updated: {status.get('updated_at')}")
    print(f"Last Cmd ID : {status.get('last_cmd_id')} (rc={status.get('returncode')}, elapsed={status.get('elapsed_sec')}s)")
    if status.get("command"):
        print(f"Command     : {status.get('command')}")
    if train_status:
        print("----------------------------------------------------------------")
        print(
            f"Training    : stage={train_status.get('stage')} | step={train_status.get('step')}/{train_status.get('total_steps')} | "
            f"loss={train_status.get('loss')}"
        )
    print("================================================================")
    tail = status.get("tail_lines") or []
    if tail:
        print("--- Recent Output Tail ---")
        for line in tail[-25:]:
            print(line)


def _wait_for_cmd(repo_id: str, token: str, cmd_id: str, poll_sec: float = 5.0) -> int:
    print(f"[colab-ctl] Waiting for Colab agent to pick up {cmd_id} ...", flush=True)
    printed_lines = 0
    acknowledged = False
    while True:
        status = _fetch_json(repo_id, STATUS_PATH_IN_REPO, token)
        if status and status.get("last_cmd_id") == cmd_id:
            if not acknowledged:
                acknowledged = True
                print(
                    f"[colab-ctl] Agent {status.get('agent_id')} ({status.get('gpu')}) started {cmd_id}!",
                    flush=True,
                )
            log_text = _fetch_text(repo_id, LOG_PATH_IN_REPO, token) or ""
            lines = log_text.splitlines()
            if len(lines) > printed_lines:
                for ln in lines[printed_lines:]:
                    print(ln, flush=True)
                printed_lines = len(lines)

            state = status.get("state")
            if state in ("completed", "failed", "killed"):
                rc = int(status.get("returncode") or (0 if state == "completed" else 1))
                print(f"[colab-ctl] Command {cmd_id} finished with state={state} (rc={rc}).", flush=True)
                return rc
        time.sleep(poll_sec)


def main() -> int:
    parser = argparse.ArgumentParser(description="Antigravity Controller for Google Colab")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo ID")
    sub = parser.add_subparsers(dest="subcmd", required=True)

    sub.add_parser("status", help="Show live Colab GPU agent & training status")
    sub.add_parser("logs", help="Fetch and print full output.log from Colab")

    p_exec = sub.add_parser("exec", help="Execute a shell command on Google Colab")
    p_exec.add_argument("command", help="Shell command string to run on Colab")
    p_exec.add_argument("--wait", action="store_true", help="Wait and stream logs until completion")
    p_exec.add_argument(
        "--run-on-boot",
        action="store_true",
        help="Allow Colab agent to run this command immediately when it boots up",
    )

    p_kill = sub.add_parser("kill", help="Kill the currently running command on Colab")
    p_kill.add_argument("--wait", action="store_true", help="Wait for kill confirmation")

    p_ping = sub.add_parser("ping", help="Ping the Colab agent to refresh telemetry")
    p_pull = sub.add_parser("pull", help="Git pull latest main branch on Colab")
    p_pair = sub.add_parser("pair", help="Configure zero-prompt pairing for Colab agent")

    args = parser.parse_args()
    token = _get_token()

    if args.subcmd == "pair":
        _push_bootstrap(args.repo_id, token)
        print("[colab-ctl] Zero-prompt Colab bridge pairing configured on Hugging Face Hub.")
        return 0

    if args.subcmd == "status":
        status = _fetch_json(args.repo_id, STATUS_PATH_IN_REPO, token)
        train_status = _fetch_json(args.repo_id, TRAIN_STATUS_IN_REPO, token)
        _print_status(status, train_status)
        return 0

    if args.subcmd == "logs":
        log_text = _fetch_text(args.repo_id, LOG_PATH_IN_REPO, token)
        if log_text:
            print(log_text)
        else:
            print("[colab-ctl] No output.log found yet.")
        return 0

    if args.subcmd == "exec":
        cmd_id = _push_cmd(
            args.repo_id,
            token,
            action="exec",
            command=args.command,
            run_on_boot=args.run_on_boot,
        )
        print(f"[colab-ctl] Dispatched command {cmd_id}: {args.command}", flush=True)
        if args.wait:
            return _wait_for_cmd(args.repo_id, token, cmd_id)
        return 0

    if args.subcmd == "kill":
        cmd_id = _push_cmd(args.repo_id, token, action="kill")
        print(f"[colab-ctl] Dispatched kill {cmd_id}", flush=True)
        if args.wait:
            return _wait_for_cmd(args.repo_id, token, cmd_id)
        return 0

    if args.subcmd == "ping":
        cmd_id = _push_cmd(args.repo_id, token, action="ping")
        print(f"[colab-ctl] Dispatched ping {cmd_id}", flush=True)
        return 0

    if args.subcmd == "pull":
        cmd_id = _push_cmd(args.repo_id, token, action="pull")
        print(f"[colab-ctl] Dispatched git pull {cmd_id}", flush=True)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
