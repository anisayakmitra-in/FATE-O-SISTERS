// Copyright (C) 2026 ANISAYAK MITRA
// SPDX-License-Identifier: GPL-3.0-or-later

mod data_sifter;
mod engine;
mod memory;

use data_sifter::DataSifter;
use engine::MoiraEngine;
use memory::MemorySupervisor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing MOIRA Engine (GPL-3.0-or-later)...");
    println!("Loading FATE'O'SISTERS 3-6-9 Model Architecture...");

    let mut engine = MoiraEngine::new(4096, 128)?;
    let sifter = DataSifter::new();
    let mut memory = MemorySupervisor::new()?;

    let input_tokens = sifter.sift_stream_trajectory("fn reason(state) -> verify_integrity(state)");
    let (logits, agent_gate_triggered, entropy) = engine.forward(&input_tokens)?;

    let summary_vec = logits.squeeze(0)?.to_vec1::<f32>()?;
    let trace_slice: Vec<f32> = summary_vec.into_iter().take(16).collect();
    memory.secure_trajectory(trace_slice)?;
    memory.flush_queue_to_wal()?;

    println!(
        "Forward pass executed: {} tokens, normalized entropy = {:.4}",
        input_tokens.len(),
        entropy
    );

    if agent_gate_triggered {
        println!("DSR Activation Entropy Threshold Crossed (>0.666).");
        println!("Redirecting process to local Model Context Protocol (MCP) handlers.");
    }

    Ok(())
}
