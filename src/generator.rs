// Copyright (C) 2026 ANISAYAK MITRA
// SPDX-License-Identifier: GPL-3.0-or-later
//
// FATE-O-SISTERS (MOIRA) — Autoregressive Generation & Byte-level Tokenizer
// Provides:
//   - ByteTokenizer: Pure-Rust byte-level tokenizer mapping UTF-8 bytes <-> token IDs
//   - GenerationEngine: Autoregressive text generator with temperature and top-p sampling

use crate::pnn::SplitMix64;
use crate::tsk::{MoiraTskBackbone, Orientation};

pub struct ByteTokenizer {
    pub vocab_size: usize,
}

impl ByteTokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.as_bytes()
            .iter()
            .map(|&b| (b as u32) % (self.vocab_size as u32))
            .collect()
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens.iter().map(|&t| (t % 256) as u8).collect();
        String::from_utf8_lossy(&bytes).to_string()
    }
}

pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 48,
            temperature: 0.7,
            top_p: 0.9,
            seed: 369,
        }
    }
}

pub struct Generator {
    backbone: MoiraTskBackbone,
    tokenizer: ByteTokenizer,
}

impl Generator {
    pub fn new(backbone: MoiraTskBackbone) -> Self {
        let vocab_size = backbone.vocab_size;
        Self {
            backbone,
            tokenizer: ByteTokenizer::new(vocab_size),
        }
    }

    fn sample_token(logits: &[f32], temperature: f32, top_p: f32, rng: &mut SplitMix64) -> u32 {
        let temp = temperature.max(1e-4);
        let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        
        let mut indexed_probs: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(idx, &val)| (idx, ((val - max_val) / temp).exp()))
            .collect();
            
        let sum_exp: f32 = indexed_probs.iter().map(|(_, p)| *p).sum::<f32>().max(1e-12);
        for (_, p) in &mut indexed_probs {
            *p /= sum_exp;
        }

        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumulative = 0.0f32;
        let mut cutoff_idx = indexed_probs.len();
        for (i, (_, p)) in indexed_probs.iter().enumerate() {
            cumulative += *p;
            if cumulative >= top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        let filtered = &indexed_probs[..cutoff_idx];
        let filtered_sum: f32 = filtered.iter().map(|(_, p)| *p).sum::<f32>().max(1e-12);

        let rand_val = (rng.next_u64() as f64 / u64::MAX as f64) as f32 * filtered_sum;
        let mut acc = 0.0f32;
        for &(idx, p) in filtered {
            acc += p;
            if rand_val <= acc {
                return idx as u32;
            }
        }
        filtered[0].0 as u32
    }

    pub fn generate(&mut self, prompt: &str, config: &GenerationConfig) -> (String, [f32; 30]) {
        let mut tokens = self.tokenizer.encode(prompt);
        let mut rng = SplitMix64::new(config.seed);
        let mut final_pulse = [0.0f32; 30];

        let d_model = self.backbone.d_model;
        let vocab_size = self.backbone.vocab_size;

        for _ in 0..config.max_new_tokens {
            let (hidden_seq, pulse, _) = self.backbone.forward(&tokens, Orientation::Atropos);
            final_pulse = pulse;

            if hidden_seq.is_empty() {
                break;
            }
            let last_hidden = &hidden_seq[hidden_seq.len() - 1];

            // Project to vocabulary logits via transpose of embedding table
            let mut logits = vec![0.0f32; vocab_size];
            for v in 0..vocab_size {
                let off = v * d_model;
                let mut dot = 0.0f32;
                for d in 0..d_model {
                    dot += last_hidden[d] * self.backbone.embed[off + d];
                }
                logits[v] = dot;
            }

            let next_token = Self::sample_token(&logits, config.temperature, config.top_p, &mut rng);
            tokens.push(next_token);

            // Stop at newline or null terminator in byte mode
            if next_token == b'\n' as u32 || next_token == 0 {
                break;
            }
        }

        (self.tokenizer.decode(&tokens), final_pulse)
    }
}
