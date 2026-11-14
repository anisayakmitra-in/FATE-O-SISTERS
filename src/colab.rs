// Copyright (C) 2026 ANISAYAK MITRA
// SPDX-License-Identifier: GPL-3.0-or-later
//
// FATE-O-SISTERS (MOIRA) — 100% Rust Google Colab Remote Control Bridge & HF Hub Client
// Provides:
//   - `colab-agent`: Runs inside Google Colab GPU VMs, auto-pairs via `colab_control/bootstrap.json`,
//     executes commands, queries `nvidia-smi`, and streams status/logs to Hugging Face Hub.
//   - `colab-ctl`: Runs locally in Antigravity (`status`, `logs`, `exec`, `kill`, `ping`, `pull`, `pair`).
//   - `push-hf`: Uploads compiled `.pnn`, `.safetensors`, and metrics to Hugging Face Hub via NDJSON API.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const DEFAULT_REPO_ID: &str = "anisayakmitra-in/FATE-O-SISTERS";
pub const CMD_PATH_IN_REPO: &str = "colab_control/cmd.json";
pub const STATUS_PATH_IN_REPO: &str = "colab_control/status.json";
pub const LOG_PATH_IN_REPO: &str = "colab_control/output.log";
pub const BOOTSTRAP_PATH_IN_REPO: &str = "colab_control/bootstrap.json";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColabCommand {
    pub cmd_id: String,
    pub action: String,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub run_on_boot: bool,
    pub issued_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColabStatus {
    pub agent_id: String,
    pub hostname: String,
    pub gpu: String,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub gpu_util_pct: u64,
    pub git_head: String,
    pub workdir: String,
    pub state: String,
    pub active_cmd_id: Option<String>,
    pub last_cmd_id: Option<String>,
    pub command: String,
    pub returncode: Option<i32>,
    pub pid: Option<u32>,
    pub updated_at: String,
    pub elapsed_sec: Option<f64>,
    pub tail_lines: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BootstrapPayload {
    version: u32,
    payload: String,
    updated_at: String,
}

pub fn unix_now_str() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{secs}Z")
}

pub fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// Downloads a file from Hugging Face Hub `resolve/main/{path_in_repo}` via `curl`.
pub fn hf_download_bytes(repo_id: &str, path_in_repo: &str, token: Option<&str>) -> Option<Vec<u8>> {
    let url = format!(
        "https://huggingface.co/{repo_id}/resolve/main/{path_in_repo}?nocache={}",
        unix_millis()
    );
    let mut cmd = Command::new("curl");
    cmd.args(["-sS", "-f", "-L", "--max-time", "15"]);
    if let Some(tok) = token {
        if !tok.is_empty() {
            cmd.args(["-H", &format!("Authorization: Bearer {tok}")]);
        }
    }
    cmd.arg(&url);
    let out = cmd.output().ok()?;
    if out.status.success() {
        Some(out.stdout)
    } else {
        None
    }
}

/// Commits one or more files atomically to Hugging Face Hub `main` using the NDJSON Commit API.
pub fn hf_commit_files(
    repo_id: &str,
    token: &str,
    commit_message: &str,
    files: &[(&str, &[u8])],
) -> Result<(), String> {
    let mut ndjson = String::new();
    let header_line = serde_json::json!({
        "key": "header",
        "value": {
            "summary": commit_message,
            "description": ""
        }
    });
    ndjson.push_str(&header_line.to_string());
    ndjson.push('\n');

    for &(path_in_repo, content) in files {
        let b64_content = B64.encode(content);
        let file_line = serde_json::json!({
            "key": "file",
            "value": {
                "content": b64_content,
                "path": path_in_repo,
                "encoding": "base64"
            }
        });
        ndjson.push_str(&file_line.to_string());
        ndjson.push('\n');
    }

    let tmp_ndjson = std::env::temp_dir().join(format!("moira_hf_commit_{}.ndjson", std::process::id()));
    std::fs::write(&tmp_ndjson, ndjson.as_bytes()).map_err(|e| e.to_string())?;

    let url = format!("https://huggingface.co/api/models/{repo_id}/commit/main");
    let out = Command::new("curl")
        .args([
            "-sS",
            "-f",
            "-X",
            "POST",
            "--max-time",
            "30",
            "-H",
            &format!("Authorization: Bearer {token}"),
            "-H",
            "Content-Type: application/x-ndjson",
            "--data-binary",
            &format!("@{}", tmp_ndjson.display()),
            &url,
        ])
        .output()
        .map_err(|e| e.to_string());

    let _ = std::fs::remove_file(&tmp_ndjson);
    let out = out?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("HF commit failed: {stderr}"));
    }
    Ok(())
}

/// Resolves `HF_TOKEN` from environment, `~/.cache/huggingface/token`, or zero-prompt `bootstrap.json`.
pub fn resolve_hf_token(repo_id: &str) -> Result<String, String> {
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        let t = tok.trim().to_string();
        if t.starts_with("hf_") {
            return Ok(t);
        }
    }

    // Check if user accidentally set the token value as an environment variable key
    for (k, v) in std::env::vars() {
        if k.starts_with("hf_") && k.len() > 20 {
            return Ok(k);
        }
        if v.trim().starts_with("hf_") && v.trim().len() > 20 {
            return Ok(v.trim().to_string());
        }
    }

    // Check ~/.cache/huggingface/token
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    if !home.is_empty() {
        let cache_tok = Path::new(&home).join(".cache").join("huggingface").join("token");
        if let Ok(contents) = std::fs::read_to_string(cache_tok) {
            let t = contents.trim().to_string();
            if t.starts_with("hf_") {
                return Ok(t);
            }
        }
    }

    // Try zero-prompt bootstrap.json on Hugging Face Hub
    if let Some(bytes) = hf_download_bytes(repo_id, BOOTSTRAP_PATH_IN_REPO, None) {
        if let Ok(boot) = serde_json::from_slice::<BootstrapPayload>(&bytes) {
            if let Ok(raw) = B64.decode(boot.payload.as_bytes()) {
                let key = format!("{repo_id}::moira-369-bridge").into_bytes();
                let decoded: Vec<u8> = raw
                    .iter()
                    .enumerate()
                    .map(|(i, &b)| b ^ key[i % key.len()])
                    .collect();
                if let Ok(s) = String::from_utf8(decoded) {
                    if s.starts_with("hf_") {
                        println!("[moira-colab-agent] Authenticated automatically via Antigravity Rust bridge!");
                        return Ok(s);
                    }
                }
            }
        }
    }

    Err("Could not resolve HF_TOKEN from environment, cache, or bootstrap.json.".to_string())
}

pub fn push_bootstrap_pairing(repo_id: &str, token: &str) -> Result<(), String> {
    let key = format!("{repo_id}::moira-369-bridge").into_bytes();
    let xored: Vec<u8> = token
        .as_bytes()
        .iter()
        .enumerate()
        .map(|(i, &b)| b ^ key[i % key.len()])
        .collect();
    let payload = BootstrapPayload {
        version: 1,
        payload: B64.encode(&xored),
        updated_at: unix_now_str(),
    };
    let json_bytes = serde_json::to_vec_pretty(&payload).map_err(|e| e.to_string())?;
    hf_commit_files(
        repo_id,
        token,
        "moira-rust-ctl: configure zero-prompt bridge pairing",
        &[(BOOTSTRAP_PATH_IN_REPO, &json_bytes)],
    )
}

fn query_gpu_telemetry() -> (String, u64, u64, u64) {
    if let Ok(out) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            if let Some(first_line) = text.lines().next() {
                let parts: Vec<&str> = first_line.split(',').map(str::trim).collect();
                if parts.len() >= 4 {
                    let used = parts[1].parse::<f64>().unwrap_or(0.0) as u64;
                    let total = parts[2].parse::<f64>().unwrap_or(0.0) as u64;
                    let util = parts[3].parse::<f64>().unwrap_or(0.0) as u64;
                    return (parts[0].to_string(), used, total, util);
                }
            }
        }
    }
    ("CPU / none detected".to_string(), 0, 0, 0)
}

fn query_git_head(workdir: &Path) -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(workdir)
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn query_hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "colab-vm".to_string())
}

/// Runs the 100% Rust Google Colab Remote Execution & Telemetry Daemon (`moira colab-agent`).
pub fn run_colab_agent(repo_id: &str, workdir: &Path, poll_secs: u64) -> Result<(), String> {
    let token = resolve_hf_token(repo_id)?;
    let hostname = query_hostname();
    let agent_id = format!("rust-colab-{hostname}-{}", std::process::id());
    println!("[moira-colab-agent] Starting pure-Rust daemon ({agent_id}) for {repo_id}");

    let prev_completed_id = hf_download_bytes(repo_id, STATUS_PATH_IN_REPO, Some(&token))
        .and_then(|b| serde_json::from_slice::<ColabStatus>(&b).ok())
        .and_then(|st| {
            if matches!(st.state.as_str(), "completed" | "failed" | "killed") {
                st.last_cmd_id
            } else {
                None
            }
        });

    let mut last_cmd_id: Option<String> = None;
    if let Some(cmd_bytes) = hf_download_bytes(repo_id, CMD_PATH_IN_REPO, Some(&token)) {
        if let Ok(init_cmd) = serde_json::from_slice::<ColabCommand>(&cmd_bytes) {
            if init_cmd.run_on_boot && Some(init_cmd.cmd_id.clone()) != prev_completed_id {
                println!(
                    "[moira-colab-agent] Found pending boot command: {}",
                    init_cmd.cmd_id
                );
            } else {
                last_cmd_id = Some(init_cmd.cmd_id);
            }
        }
    }

    let mut state = "idle".to_string();
    let mut active_cmd_id: Option<String> = None;
    let mut current_command = String::new();
    let mut returncode: Option<i32> = None;
    let mut started_instant: Option<Instant> = None;
    let mut active_child: Option<Child> = None;
    let log_lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let mut last_flush = Instant::now();

    let flush = |state: &str,
                 active_cmd_id: &Option<String>,
                 last_cmd_id: &Option<String>,
                 current_command: &str,
                 returncode: Option<i32>,
                 pid: Option<u32>,
                 started: Option<Instant>,
                 log_lines: &Arc<Mutex<Vec<String>>>,
                 msg: &str,
                 include_log: bool| {
        let (gpu, vram_used_mb, vram_total_mb, gpu_util_pct) = query_gpu_telemetry();
        let (full_log, tail_lines) = {
            let guard = log_lines.lock().unwrap();
            let len = guard.len();
            let start = len.saturating_sub(60);
            let tail: Vec<String> = guard[start..].to_vec();
            let joined = guard.join("\n");
            (joined, tail)
        };
        let elapsed_sec = started.map(|t| (t.elapsed().as_secs_f64() * 100.0).round() / 100.0);
        let status_obj = ColabStatus {
            agent_id: agent_id.clone(),
            hostname: hostname.clone(),
            gpu,
            vram_used_mb,
            vram_total_mb,
            gpu_util_pct,
            git_head: query_git_head(workdir),
            workdir: workdir.display().to_string(),
            state: state.to_string(),
            active_cmd_id: active_cmd_id.clone(),
            last_cmd_id: last_cmd_id.clone(),
            command: current_command.to_string(),
            returncode,
            pid,
            updated_at: unix_now_str(),
            elapsed_sec,
            tail_lines,
        };
        if let Ok(status_bytes) = serde_json::to_vec_pretty(&status_obj) {
            let mut ops: Vec<(&str, &[u8])> = vec![(STATUS_PATH_IN_REPO, &status_bytes)];
            let log_bytes = full_log.into_bytes();
            if include_log && !log_bytes.is_empty() {
                ops.push((LOG_PATH_IN_REPO, &log_bytes));
            }
            if let Err(e) = hf_commit_files(repo_id, &token, msg, &ops) {
                eprintln!("[moira-colab-agent] Telemetry push warning: {e}");
            }
        }
    };

    flush(
        &state,
        &active_cmd_id,
        &last_cmd_id,
        &current_command,
        returncode,
        None,
        None,
        &log_lines,
        &format!("moira-rust-agent: online ({agent_id})"),
        false,
    );
    println!("[moira-colab-agent] Online and paired with Antigravity! Polling for commands...");

    loop {
        if let Some(ref mut child) = active_child {
            if let Ok(Some(exit_status)) = child.try_wait() {
                std::thread::sleep(Duration::from_millis(400));
                let rc = exit_status.code().unwrap_or(1);
                returncode = Some(rc);
                state = if rc == 0 {
                    "completed".to_string()
                } else {
                    "failed".to_string()
                };
                let done_id = active_cmd_id.take().unwrap_or_default();
                {
                    let mut guard = log_lines.lock().unwrap();
                    guard.push(format!(
                        "[moira-colab-agent] Command {done_id} finished with exit code {rc}"
                    ));
                }
                active_child = None;
                flush(
                    &state,
                    &active_cmd_id,
                    &last_cmd_id,
                    &current_command,
                    returncode,
                    None,
                    started_instant,
                    &log_lines,
                    &format!("moira-rust-agent: {state} {done_id} (rc={rc})"),
                    true,
                );
                last_flush = Instant::now();
            } else if last_flush.elapsed() >= Duration::from_secs(10) {
                let pid = Some(child.id());
                flush(
                    &state,
                    &active_cmd_id,
                    &last_cmd_id,
                    &current_command,
                    returncode,
                    pid,
                    started_instant,
                    &log_lines,
                    "moira-rust-agent: progress",
                    true,
                );
                last_flush = Instant::now();
            }
        }

        if let Some(cmd_bytes) = hf_download_bytes(repo_id, CMD_PATH_IN_REPO, Some(&token)) {
            if let Ok(cmd_obj) = serde_json::from_slice::<ColabCommand>(&cmd_bytes) {
                if !cmd_obj.cmd_id.is_empty() && Some(cmd_obj.cmd_id.clone()) != last_cmd_id {
                    match cmd_obj.action.as_str() {
                        "kill" => {
                            if let Some(mut ch) = active_child.take() {
                                let _ = ch.kill();
                                let _ = ch.wait();
                            }
                            state = "killed".to_string();
                            last_cmd_id = Some(cmd_obj.cmd_id.clone());
                            active_cmd_id = None;
                            flush(
                                &state,
                                &active_cmd_id,
                                &last_cmd_id,
                                &current_command,
                                Some(-9),
                                None,
                                started_instant,
                                &log_lines,
                                &format!("moira-rust-agent: killed ({})", cmd_obj.cmd_id),
                                true,
                            );
                        }
                        "ping" => {
                            last_cmd_id = Some(cmd_obj.cmd_id.clone());
                            flush(
                                &state,
                                &active_cmd_id,
                                &last_cmd_id,
                                &current_command,
                                returncode,
                                active_child.as_ref().map(Child::id),
                                started_instant,
                                &log_lines,
                                &format!("moira-rust-agent: pong ({})", cmd_obj.cmd_id),
                                false,
                            );
                        }
                        _ => {
                            let shell_cmd = if cmd_obj.action == "pull" {
                                "git fetch origin main && git reset --hard origin/main".to_string()
                            } else {
                                cmd_obj.command.clone()
                            };

                            if let Some(mut ch) = active_child.take() {
                                let _ = ch.kill();
                                let _ = ch.wait();
                            }
                            {
                                let mut guard = log_lines.lock().unwrap();
                                guard.clear();
                                guard.push(format!(
                                    "[moira-colab-agent] Executing {}: {shell_cmd}",
                                    cmd_obj.cmd_id
                                ));
                            }
                            println!(
                                "\n================================================================\n[moira-colab-agent] Executing {}: {shell_cmd}\n================================================================",
                                cmd_obj.cmd_id
                            );

                            let mut spawn_cmd = if cfg!(windows) {
                                let mut c = Command::new("powershell");
                                c.args(["-NoProfile", "-Command", &shell_cmd]);
                                c
                            } else {
                                let mut c = Command::new("sh");
                                c.args(["-c", &shell_cmd]);
                                c
                            };
                            spawn_cmd
                                .current_dir(workdir)
                                .env("HF_TOKEN", &token)
                                .env("PYTHONUNBUFFERED", "1")
                                .stdout(Stdio::piped())
                                .stderr(Stdio::piped());

                            match spawn_cmd.spawn() {
                                Ok(mut child) => {
                                    let pid = child.id();
                                    if let Some(stdout) = child.stdout.take() {
                                        let lines_ref = Arc::clone(&log_lines);
                                        std::thread::spawn(move || {
                                            let reader = BufReader::new(stdout);
                                            for line in reader.lines().map_while(Result::ok) {
                                                println!("{line}");
                                                let mut g = lines_ref.lock().unwrap();
                                                g.push(line);
                                                if g.len() > 3000 {
                                                    let drain = g.len() - 3000;
                                                    g.drain(..drain);
                                                }
                                            }
                                        });
                                    }
                                    if let Some(stderr) = child.stderr.take() {
                                        let lines_ref = Arc::clone(&log_lines);
                                        std::thread::spawn(move || {
                                            let reader = BufReader::new(stderr);
                                            for line in reader.lines().map_while(Result::ok) {
                                                eprintln!("{line}");
                                                let mut g = lines_ref.lock().unwrap();
                                                g.push(line);
                                                if g.len() > 3000 {
                                                    let drain = g.len() - 3000;
                                                    g.drain(..drain);
                                                }
                                            }
                                        });
                                    }

                                    state = "running".to_string();
                                    active_cmd_id = Some(cmd_obj.cmd_id.clone());
                                    last_cmd_id = Some(cmd_obj.cmd_id.clone());
                                    current_command = shell_cmd;
                                    returncode = None;
                                    started_instant = Some(Instant::now());
                                    active_child = Some(child);
                                    flush(
                                        &state,
                                        &active_cmd_id,
                                        &last_cmd_id,
                                        &current_command,
                                        returncode,
                                        Some(pid),
                                        started_instant,
                                        &log_lines,
                                        &format!("moira-rust-agent: started {}", cmd_obj.cmd_id),
                                        true,
                                    );
                                    last_flush = Instant::now();
                                }
                                Err(e) => {
                                    state = "failed".to_string();
                                    last_cmd_id = Some(cmd_obj.cmd_id.clone());
                                    returncode = Some(127);
                                    {
                                        let mut g = log_lines.lock().unwrap();
                                        g.push(format!("Spawn error: {e}"));
                                    }
                                    flush(
                                        &state,
                                        &None,
                                        &last_cmd_id,
                                        &shell_cmd,
                                        returncode,
                                        None,
                                        None,
                                        &log_lines,
                                        "moira-rust-agent: spawn error",
                                        true,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        std::thread::sleep(Duration::from_secs(poll_secs.max(2)));
    }
}

/// Runs the local Antigravity Rust Controller (`moira colab-ctl <subcmd>`).
pub fn run_colab_ctl(repo_id: &str, args: &[String]) -> Result<(), String> {
    let token = resolve_hf_token(repo_id)?;
    let subcmd = args.first().map(String::as_str).unwrap_or("status");

    match subcmd {
        "status" => {
            if let Some(bytes) = hf_download_bytes(repo_id, STATUS_PATH_IN_REPO, Some(&token)) {
                if let Ok(st) = serde_json::from_slice::<ColabStatus>(&bytes) {
                    println!("================================================================");
                    println!("Colab Agent : {} ({})", st.agent_id, st.hostname);
                    println!(
                        "GPU         : {} | VRAM: {}/{} MB | Util: {}%",
                        st.gpu, st.vram_used_mb, st.vram_total_mb, st.gpu_util_pct
                    );
                    println!("Git Commit  : {} | Workdir: {}", st.git_head, st.workdir);
                    println!("Agent State : {} | Updated: {}", st.state, st.updated_at);
                    println!(
                        "Last Cmd ID : {:?} (rc={:?}, elapsed={:?}s)",
                        st.last_cmd_id, st.returncode, st.elapsed_sec
                    );
                    if !st.command.is_empty() {
                        println!("Command     : {}", st.command);
                    }
                    println!("================================================================");
                    for ln in st.tail_lines.iter().rev().take(25).rev() {
                        println!("{ln}");
                    }
                    return Ok(());
                }
            }
            println!("[moira-colab-ctl] No Colab agent status online yet on {repo_id}.");
            Ok(())
        }
        "logs" => {
            if let Some(bytes) = hf_download_bytes(repo_id, LOG_PATH_IN_REPO, Some(&token)) {
                println!("{}", String::from_utf8_lossy(&bytes));
            } else {
                println!("[moira-colab-ctl] No output.log found yet.");
            }
            Ok(())
        }
        "pair" => {
            push_bootstrap_pairing(repo_id, &token)?;
            println!("[moira-colab-ctl] Zero-prompt Rust Colab pairing configured on {repo_id}.");
            Ok(())
        }
        "exec" | "kill" | "ping" | "pull" => {
            let mut command_str = String::new();
            let mut wait = false;
            let mut run_on_boot = false;

            for arg in &args[1..] {
                match arg.as_str() {
                    "--wait" => wait = true,
                    "--run-on-boot" => run_on_boot = true,
                    other => {
                        if command_str.is_empty() {
                            command_str = other.to_string();
                        } else {
                            command_str.push(' ');
                            command_str.push_str(other);
                        }
                    }
                }
            }

            let cmd_id = format!("cmd-{}", unix_millis());
            let payload = ColabCommand {
                cmd_id: cmd_id.clone(),
                action: subcmd.to_string(),
                command: command_str.clone(),
                run_on_boot,
                issued_at: unix_now_str(),
            };
            let json_bytes = serde_json::to_vec_pretty(&payload).map_err(|e| e.to_string())?;
            hf_commit_files(
                repo_id,
                &token,
                &format!("moira-rust-ctl: {subcmd} ({cmd_id})"),
                &[(CMD_PATH_IN_REPO, &json_bytes)],
            )?;
            println!("[moira-colab-ctl] Dispatched {subcmd} ({cmd_id}): {command_str}");

            if wait {
                println!("[moira-colab-ctl] Waiting for Colab agent to complete {cmd_id}...");
                let mut printed = 0usize;
                loop {
                    if let Some(bytes) = hf_download_bytes(repo_id, STATUS_PATH_IN_REPO, Some(&token))
                    {
                        if let Ok(st) = serde_json::from_slice::<ColabStatus>(&bytes) {
                            if st.last_cmd_id.as_deref() == Some(cmd_id.as_str()) {
                                if let Some(lbytes) =
                                    hf_download_bytes(repo_id, LOG_PATH_IN_REPO, Some(&token))
                                {
                                    let text = String::from_utf8_lossy(&lbytes);
                                    let lines: Vec<&str> = text.lines().collect();
                                    if lines.len() > printed {
                                        for ln in &lines[printed..] {
                                            println!("{ln}");
                                        }
                                        printed = lines.len();
                                    }
                                }
                                if matches!(st.state.as_str(), "completed" | "failed" | "killed") {
                                    println!(
                                        "[moira-colab-ctl] Command {cmd_id} finished: state={} (rc={:?})",
                                        st.state, st.returncode
                                    );
                                    break;
                                }
                            }
                        }
                    }
                    std::thread::sleep(Duration::from_secs(4));
                }
            }
            Ok(())
        }
        other => Err(format!("Unknown colab-ctl subcommand: {other}")),
    }
}

/// Uploads the compiled `moira_hf/` bundle (`model.safetensors`, `model_decision.pnn`,
/// `model_decision.bin.pnn`, `training_metrics.json`) to Hugging Face Hub in pure Rust!
pub fn push_hf_bundle(repo_id: &str) -> Result<(), String> {
    let token = resolve_hf_token(repo_id)?;
    let hf_dir = PathBuf::from("moira_hf");
    let candidate_files = [
        "model.safetensors",
        "model_decision.pnn",
        "model_decision.bin.pnn",
        "training_metrics.json",
    ];
    let mut loaded: Vec<(String, Vec<u8>)> = Vec::new();
    for name in candidate_files {
        let p = hf_dir.join(name);
        if let Ok(bytes) = std::fs::read(&p) {
            loaded.push((name.to_string(), bytes));
        }
    }
    if loaded.is_empty() {
        return Err("No artifacts found in ./moira_hf/ to upload.".to_string());
    }
    let refs: Vec<(&str, &[u8])> = loaded
        .iter()
        .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
        .collect();
    hf_commit_files(
        repo_id,
        &token,
        "feat(rust): upload pure-Rust MOIRA 3-6-9 model.safetensors and Tropical model_decision.pnn",
        &refs,
    )?;
    println!("[OK] Uploaded {} artifacts to https://huggingface.co/{repo_id} via pure-Rust HF client!", refs.len());
    Ok(())
}
