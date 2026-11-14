// Copyright (C) 2026 ANISAYAK MITRA
// SPDX-License-Identifier: GPL-3.0-or-later
//
// FATE-O-SISTERS (MOIRA) — 100% Rust Latent Reasoning Engine
// Implements the 3-6-9 Clotho / Lachesis / Atropos forward pass, IndexShare
// coordinate cache, and step-6 Dynamic Structured Reasoning (DSR) entropy gate.

use crate::pnn::SplitMix64;

/// Compact 2D tensor (`[rows, cols]`) backed by contiguous `Vec<f32>`.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Tensor2D {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

#[allow(dead_code)]
impl Tensor2D {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0f32; rows * cols],
        }
    }

    pub fn dims(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    pub fn to_vec1(&self) -> Vec<f32> {
        self.data.clone()
    }
}

/// On-device latent reasoning engine implementing the MOIRA 3-6-9
/// Clotho / Lachesis / Atropos perspective architecture in pure Rust.
pub struct MoiraEngine {
    vocab_size: usize,
    hidden_dim: usize,
    index_cache: Option<Vec<u32>>,
    w_embed: Vec<f32>,
    w_clotho: Vec<f32>,
    w_lachesis: Vec<f32>,
    w_atropos: Vec<f32>,
    w_out: Vec<f32>,
}

impl MoiraEngine {
    pub fn new(vocab_size: usize, hidden_dim: usize) -> Result<Self, String> {
        let v = vocab_size.max(1);
        let h = hidden_dim.max(1);
        let scale = (1.0f32 / (h as f32)).sqrt();
        let mut rng = SplitMix64::new(369_2026);

        let mut init_vec = |n: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(n);
            for _ in 0..n {
                out.push(rng.next_signed(scale));
            }
            out
        };

        Ok(Self {
            vocab_size: v,
            hidden_dim: h,
            index_cache: None,
            w_embed: init_vec(v * h),
            w_clotho: init_vec(h * h),
            w_lachesis: init_vec(h * h),
            w_atropos: init_vec(h * h),
            w_out: init_vec(h * v),
        })
    }

    fn matvec_tanh(vec_in: &[f32], mat: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; out_dim];
        for j in 0..out_dim {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += vec_in[i] * mat[i * out_dim + j];
            }
            out[j] = acc.tanh();
        }
        out
    }

    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_v = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max_v).exp()).collect();
        let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
        for v in &mut exps {
            *v /= sum;
        }
        exps
    }

    /// Computes normalized Shannon entropy in `[0, 1]` over a 1D probability vector.
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
    pub fn forward(&mut self, tokens: &[u32]) -> Result<(Tensor2D, bool, f32), String> {
        if tokens.is_empty() {
            return Ok((Tensor2D::zeros(1, self.vocab_size), false, 0.0));
        }

        let h = self.hidden_dim;
        let mut h_input = vec![0.0f32; h];
        let inv_len = 1.0 / (tokens.len() as f32);
        for &tok in tokens {
            let idx = (tok as usize) % self.vocab_size;
            let row = idx * h;
            for d in 0..h {
                h_input[d] += self.w_embed[row + d] * inv_len;
            }
        }

        // --- PHASE 3: CLOTHO (Velocity / Bidirectional Context Head) ---
        let v_velocity = Self::matvec_tanh(&h_input, &self.w_clotho, h, h);

        // --- PHASE 6: LACHESIS (Stability Core: 9-step weight-shared recurrence) ---
        let mut v_stability = h_input;
        let mut agent_gate_triggered = false;
        let mut gate_entropy = 0.0f32;

        for k in 0..9 {
            if k == 0 {
                let coords: Vec<u32> = (0..16)
                    .map(|i| (i as u32) % (tokens.len() as u32))
                    .collect();
                self.index_cache = Some(coords);
            }

            let step_proj = Self::matvec_tanh(&v_stability, &self.w_lachesis, h, h);
            for d in 0..h {
                v_stability[d] = 0.5 * (v_stability[d] + step_proj[d]);
            }

            if k == 6 {
                let probs = Self::softmax(&v_stability);
                gate_entropy = Self::normalized_entropy(&probs);
                if gate_entropy > 0.666 {
                    agent_gate_triggered = true;
                }
            }
        }

        // --- PHASE 9: ATROPOS (Integrity Validator & Causal Gate) ---
        let v_integrity = Self::matvec_tanh(&v_stability, &self.w_atropos, h, h);

        // 1:2:3 (3:6:9) Mixing Law Normalization:
        // V_mixed = (3 * V_velocity + 6 * V_stability + 9 * V_integrity) / 18
        let mut v_mixed = vec![0.0f32; h];
        for d in 0..h {
            v_mixed[d] = (3.0 * v_velocity[d] + 6.0 * v_stability[d] + 9.0 * v_integrity[d]) / 18.0;
        }

        let mut logits = vec![0.0f32; self.vocab_size];
        for j in 0..self.vocab_size {
            let mut sum = 0.0f32;
            for i in 0..h {
                sum += v_mixed[i] * self.w_out[i * self.vocab_size + j];
            }
            logits[j] = sum;
        }

        Ok((
            Tensor2D {
                rows: 1,
                cols: self.vocab_size,
                data: logits,
            },
            agent_gate_triggered,
            gate_entropy,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::MoiraEngine;

    #[test]
    fn test_moira_engine_forward_produces_valid_logits() {
        let mut engine = MoiraEngine::new(256, 32).expect("engine initializes");
        let (logits, _gate, entropy) = engine.forward(&[10, 42, 99, 128]).expect("forward passes");
        assert_eq!(logits.dims(), [1, 256]);
        assert!((0.0..=1.0).contains(&entropy));
    }

    #[test]
    fn test_moira_engine_empty_tokens_safe() {
        let mut engine = MoiraEngine::new(128, 16).expect("engine initializes");
        let (logits, gate, entropy) = engine.forward(&[]).expect("empty forward passes");
        assert_eq!(logits.dims(), [1, 128]);
        assert!(!gate);
        assert_eq!(entropy, 0.0);
    }
}
