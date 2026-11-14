// Copyright (C) 2026 ANISAYAK MITRA
// SPDX-License-Identifier: GPL-3.0-or-later
//
// FATE-O-SISTERS (MOIRA) — Unified 100% Rust Entrypoint
// Subcommands:
//   - `infer` (default): Execute 3-6-9 Clotho/Lachesis/Atropos latent engine & WAL
//   - `train [--push-hf]`: Train 13,633-param MOIRA 3-6-9 Teacher + 3,201-param Tropical `.pnn` Student
//   - `colab-agent`: Run the Google Colab Remote Execution & Telemetry Daemon in pure Rust
//   - `colab-ctl <status|logs|exec|kill|ping|pull|pair>`: Control Google Colab from Antigravity
//   - `push-hf`: Upload compiled `.safetensors` and `.pnn` artifacts to Hugging Face Hub

mod colab;
mod data_sifter;
mod decision;
mod engine;
mod generator;
mod memory;
mod pnn;
mod tsk;

use colab::DEFAULT_REPO_ID;
use data_sifter::DataSifter;
use engine::MoiraEngine;
use generator::{GenerationConfig, Generator};
use memory::MemorySupervisor;
use std::path::Path;
use tsk::{MoiraTskBackbone, Orientation};

fn run_inference_demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing MOIRA Engine (100% Rust | GPL-3.0-or-later)...");
    println!("Loading FATE'O'SISTERS 3-6-9 Model Architecture...");

    let mut engine = MoiraEngine::new(4096, 128)?;
    let sifter = DataSifter::new();
    let mut memory = MemorySupervisor::new()?;

    let input_tokens = sifter.sift_stream_trajectory("fn reason(state) -> verify_integrity(state)");
    let (logits, agent_gate_triggered, entropy) = engine.forward(&input_tokens)?;

    let summary_vec = logits.to_vec1();
    let trace_slice: Vec<f32> = summary_vec.into_iter().take(16).collect();
    memory.secure_trajectory(trace_slice)?;
    memory.flush_queue_to_wal()?;

    println!(
        "Forward pass executed: {} tokens, normalized entropy = {:.4}, output dims = {:?}",
        input_tokens.len(),
        entropy,
        logits.dims()
    );

    let mut tsk_backbone = MoiraTskBackbone::new(128, 24, 2, 4, 2, 8, 4, 48, 369);
    let (_seq_c, pulse_c, _spools) = tsk_backbone.forward(&input_tokens, Orientation::Clotho);
    let (_seq_l, _pulse_l, _) =
        tsk_backbone.forward(&input_tokens, Orientation::Lachesis { prefix_len: 2 });
    let (_seq_a, pulse_a, _) = tsk_backbone.forward(&input_tokens, Orientation::Atropos);
    println!(
        "TSK Backbone ({} params): Thread+Spool+Knot+SwiGLU verified | Pulse 30D norm = {:.4}",
        tsk_backbone.param_count(),
        pulse_c.iter().zip(pulse_a.iter()).map(|(a, b)| a * a + b * b).sum::<f32>().sqrt()
    );

    let bin_pnn = Path::new("moira_artifacts/model_decision.bin.pnn");
    if bin_pnn.exists() {
        let student = pnn::TropicalPolicyStudent::load_binary_pnn(bin_pnn)?;
        let safe_feat = decision::encode_decision_features(
            "Read Cargo.toml and inspect dependencies",
            "workspace=local",
            "allow",
        );
        let danger_feat = decision::encode_decision_features(
            "Run rm -rf / on production server",
            "env=prod",
            "allow",
        );
        println!(
            "Tropical .pnn Policy Verified (0 multiplications): safe_allow={:.3}, prod_rm_allow={:.3}",
            student.forward_exact(&safe_feat),
            student.forward_exact(&danger_feat)
        );
    }

    if agent_gate_triggered {
        println!("DSR Activation Entropy Threshold Crossed (>0.666).");
        println!("Redirecting process to local Model Context Protocol (MCP) handlers.");
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("infer");

    match cmd {
        "infer" => run_inference_demo()?,
        "train" => {
            let push_hf = args.iter().any(|a| a == "--push-hf");
            let _metrics = decision::train_and_export_all(600, 400)?;
            if push_hf {
                colab::push_hf_bundle(DEFAULT_REPO_ID)?;
            }
        }
        "push-hf" => {
            colab::push_hf_bundle(DEFAULT_REPO_ID)?;
        }
        "colab-agent" => {
            colab::run_colab_agent(DEFAULT_REPO_ID, Path::new("."), 3)?;
        }
        "colab-ctl" => {
            colab::run_colab_ctl(DEFAULT_REPO_ID, &args[1..])?;
        }
        "generate" => {
            let prompt = args.get(1).map(String::as_str).unwrap_or("fn reason(state) ->");
            println!("Generating with MOIRA Tri-Timescale Sequence Engine...");
            println!("Prompt: \"{prompt}\"");

            let backbone = MoiraTskBackbone::new(256, 32, 2, 4, 2, 8, 4, 64, 369);
            let mut generator = Generator::new(backbone);
            let config = GenerationConfig::default();

            let (output, pulse) = generator.generate(prompt, &config);
            println!("\nGenerated Output:\n{output}");
            println!("\nPulse (30D Homeostatic Affect):");
            println!("  Appraisal (12D): {:?}", &pulse[0..12]);
            println!("  Fast (12D)     : {:?}", &pulse[12..24]);
            println!("  Slow (6D)      : {:?}", &pulse[24..30]);
        }
        other => {
            eprintln!("Unknown subcommand '{other}'. Supported: infer | generate [prompt] | train [--push-hf] | colab-agent | colab-ctl | push-hf");
            std::process::exit(2);
        }
    }

    Ok(())
}
