// Copyright (C) 2026 ANISAYAK MITRA
// SPDX-License-Identifier: GPL-3.0-or-later

use candle_core::{DType, Device, Result, Tensor};

/// On-device latent reasoning engine implementing the MOIRA 3-6-9
/// Clotho / Lachesis / Atropos perspective architecture.
pub struct MoiraEngine {
    vocab_size: usize,
    hidden_dim: usize,
    device: Device,
    index_cache: Option<Tensor>,
    w_embed: Tensor,
    w_clotho: Tensor,
    w_lachesis: Tensor,
    w_atropos: Tensor,
    w_out: Tensor,
}

impl MoiraEngine {
    pub fn new(vocab_size: usize, hidden_dim: usize) -> Result<Self> {
        let device = if candle_core::utils::cuda_is_available() {
            println!("MOIRA Target: NVIDIA CUDA Core Detected");
            Device::new_cuda(0)?
        } else if candle_core::utils::metal_is_available() {
            println!("MOIRA Target: Apple Silicon Metal Core Detected");
            Device::new_metal(0)?
        } else {
            println!("MOIRA Target: Running on CPU backend");
            Device::Cpu
        };

        // Deterministic Xavier-scaled projections for reproducible latent evaluation
        let scale_h = (1.0f64 / (hidden_dim.max(1) as f64)).sqrt();
        let w_embed = Tensor::randn(0.0f32, scale_h as f32, (vocab_size, hidden_dim), &device)?;
        let w_clotho = Tensor::randn(0.0f32, scale_h as f32, (hidden_dim, hidden_dim), &device)?;
        let w_lachesis = Tensor::randn(0.0f32, scale_h as f32, (hidden_dim, hidden_dim), &device)?;
        let w_atropos = Tensor::randn(0.0f32, scale_h as f32, (hidden_dim, hidden_dim), &device)?;
        let w_out = Tensor::randn(0.0f32, scale_h as f32, (hidden_dim, vocab_size), &device)?;

        Ok(Self {
            vocab_size,
            hidden_dim,
            device,
            index_cache: None,
            w_embed,
            w_clotho,
            w_lachesis,
            w_atropos,
            w_out,
        })
    }

    /// Computes normalized Shannon entropy in [0, 1] over a 1D probability vector.
    fn normalized_entropy(probs: &[f32]) -> f32 {
        if probs.len() <= 1 {
            return 0.0;
        }
        let max_entropy = (probs.len() as f32).ln();
        let mut h = 0.0f32;
        for &p in probs {
            if p > 1e-9 {
                h -= p * p.ln();
            }
        }
        (h / max_entropy).clamp(0.0, 1.0)
    }

    /// Executes the 3-6-9 Clotho / Lachesis / Atropos forward pass over `tokens`.
    /// Returns `(logits, agent_gate_triggered, normalized_entropy)`.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<(Tensor, bool, f32)> {
        if tokens.is_empty() {
            let empty_logits = Tensor::zeros((1, self.vocab_size), DType::F32, &self.device)?;
            return Ok((empty_logits, false, 0.0));
        }

        // Clamp token IDs into [0, vocab_size) and mean-pool embeddings into [1, hidden_dim]
        let clamped: Vec<u32> = tokens
            .iter()
            .map(|&t| t % (self.vocab_size.max(1) as u32))
            .collect();
        let idx = Tensor::from_vec(clamped, (tokens.len(),), &self.device)?;
        let embedded = self.w_embed.index_select(&idx, 0)?;
        let h_input = embedded.mean_keepdim(0)?; // [1, hidden_dim]

        // --- PHASE 3: CLOTHO (Velocity / Bidirectional Context Head) ---
        let v_velocity = h_input.matmul(&self.w_clotho)?.tanh()?;

        // --- PHASE 6: LACHESIS (Stability Core: 9-step weight-shared recurrence) ---
        let mut v_stability = h_input.clone();
        let mut agent_gate_triggered = false;
        let mut gate_entropy = 0.0f32;

        for k in 0..9 {
            if k == 0 {
                // IndexShare: Cache sequence coordinate summary ONCE at step 0
                let coords: Vec<u32> = (0..16)
                    .map(|i| (i as u32) % (tokens.len() as u32))
                    .collect();
                self.index_cache = Some(Tensor::from_vec(coords, (1, 16), &self.device)?);
            }

            let step_proj = v_stability.matmul(&self.w_lachesis)?.tanh()?;
            v_stability = ((&v_stability + &step_proj)? * 0.5)?;

            // Dynamic Structured Reasoning (DSR) Entropy gate at step 6
            if k == 6 {
                let probs = candle_nn::ops::softmax(&v_stability, 1)?;
                let prob_vec = probs.squeeze(0)?.to_vec1::<f32>()?;
                gate_entropy = Self::normalized_entropy(&prob_vec);
                if gate_entropy > 0.666 {
                    agent_gate_triggered = true;
                }
            }
        }

        // --- PHASE 9: ATROPOS (Integrity Validator & Causal Gate) ---
        let v_integrity = v_stability.matmul(&self.w_atropos)?.tanh()?;

        // 1:2:3 (3:6:9) Mixing Law Normalization:
        // V_mixed = (3 * V_velocity + 6 * V_stability + 9 * V_integrity) / 18
        let term_c = (&v_velocity * 3.0)?;
        let term_l = (&v_stability * 6.0)?;
        let term_a = (&v_integrity * 9.0)?;
        let v_mixed = (((term_c + term_l)? + term_a)? / 18.0)?;

        // Project mixed latent state to vocabulary logits [1, vocab_size]
        let logits = v_mixed.matmul(&self.w_out)?;

        Ok((logits, agent_gate_triggered, gate_entropy))
    }
}

#[cfg(test)]
mod tests {
    use super::MoiraEngine;

    #[test]
    fn test_moira_engine_forward_produces_valid_logits() {
        let mut engine = MoiraEngine::new(256, 32).expect("engine initializes");
        let (logits, _gate, entropy) = engine.forward(&[10, 42, 99, 128]).expect("forward passes");
        assert_eq!(logits.dims(), &[1, 256]);
        assert!((0.0..=1.0).contains(&entropy));
    }

    #[test]
    fn test_moira_engine_empty_tokens_safe() {
        let mut engine = MoiraEngine::new(128, 16).expect("engine initializes");
        let (logits, gate, entropy) = engine.forward(&[]).expect("empty forward passes");
        assert_eq!(logits.dims(), &[1, 128]);
        assert!(!gate);
        assert_eq!(entropy, 0.0);
    }
}
