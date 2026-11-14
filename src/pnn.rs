// Copyright (C) 2026 ANISAYAK MITRA
// SPDX-License-Identifier: GPL-3.0-or-later
//
// FATE-O-SISTERS (MOIRA) — 100% Rust Tropical Max-Plus Semiring (.pnn) Engine
// Implements the 3-layer (16 -> 64 -> 32 -> 1, 3,201-parameter) TropicalPolicyStudent
// over the Max-Plus semiring (R U {-inf}, max, +), temperature-annealed log-sum-exp
// distillation (tau -> 0), exact SHA-256 digest computation (AGENT-PANDORA compatible),
// and both JSON `.pnn` and binary `PNN\x01` serialization.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

pub const PNN_MAGIC: &[u8; 4] = b"PNN\x01";
pub const PNN_VERSION: u32 = 1;

/// Deterministic SplitMix64 PRNG for reproducible initialization.
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_signed(&mut self, scale: f32) -> f32 {
        let u = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        ((u * 2.0 - 1.0) as f32) * scale
    }
}

/// Pure-Rust FIPS 180-4 SHA-256 implementation for AGENT-PANDORA `.pnn` digest verification.
pub struct Sha256 {
    state: [u32; 8],
    buffer: Vec<u8>,
    bit_len: u64,
}

impl Sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    pub fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ],
            buffer: Vec::with_capacity(64),
            bit_len: 0,
        }
    }

    fn compress_block(state: &mut [u32; 8], block: &[u8]) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).take(16).enumerate() {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(Self::K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    pub fn update(&mut self, data: &[u8]) {
        self.bit_len = self.bit_len.wrapping_add((data.len() as u64) * 8);
        self.buffer.extend_from_slice(data);
        while self.buffer.len() >= 64 {
            let block: [u8; 64] = self.buffer[..64].try_into().unwrap();
            Self::compress_block(&mut self.state, &block);
            self.buffer.drain(..64);
        }
    }

    pub fn finalize_hex(mut self) -> String {
        self.buffer.push(0x80);
        while (self.buffer.len() % 64) != 56 {
            self.buffer.push(0x00);
        }
        self.buffer.extend_from_slice(&self.bit_len.to_be_bytes());
        for chunk in self.buffer.chunks_exact(64) {
            Self::compress_block(&mut self.state, chunk);
        }
        let mut out = String::with_capacity(64);
        for word in self.state {
            out.push_str(&format!("{word:08x}"));
        }
        out
    }
}

/// Tropical Dense Layer over `(R U {-inf}, max, +)`:
/// Exact (`tau = 0`): `y_j = b_j + max_i (x_i + W[j, i])`
/// Smooth (`tau > 0`): `y_j = b_j + tau * log(sum_i exp((x_i + W[j, i]) / tau))`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TropicalLayerSpec {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PnnArtifactJson {
    pub name: String,
    pub domain: String,
    pub layers: Vec<TropicalLayerSpec>,
    pub sha256_digest: String,
}

#[derive(Clone, Debug)]
pub struct TropicalDense {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub grad_w: Vec<f32>,
    pub grad_b: Vec<f32>,
    pub m_w: Vec<f32>,
    pub v_w: Vec<f32>,
    pub m_b: Vec<f32>,
    pub v_b: Vec<f32>,
}

impl TropicalDense {
    pub fn new(in_features: usize, out_features: usize, rng: &mut SplitMix64, scale: f32) -> Self {
        let n_w = out_features * in_features;
        let mut weights = Vec::with_capacity(n_w);
        for _ in 0..n_w {
            weights.push(rng.next_signed(scale));
        }
        let bias = vec![0.0f32; out_features];
        Self {
            in_features,
            out_features,
            weights,
            bias,
            grad_w: vec![0.0; n_w],
            grad_b: vec![0.0; out_features],
            m_w: vec![0.0; n_w],
            v_w: vec![0.0; n_w],
            m_b: vec![0.0; out_features],
            v_b: vec![0.0; out_features],
        }
    }

    pub fn from_weights_and_bias(
        in_features: usize,
        out_features: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Self {
        let n_w = out_features * in_features;
        Self {
            in_features,
            out_features,
            weights,
            bias,
            grad_w: vec![0.0; n_w],
            grad_b: vec![0.0; out_features],
            m_w: vec![0.0; n_w],
            v_w: vec![0.0; n_w],
            m_b: vec![0.0; out_features],
            v_b: vec![0.0; out_features],
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad_w.fill(0.0);
        self.grad_b.fill(0.0);
    }

    pub fn adamw_step(&mut self, step: usize, lr: f32, weight_decay: f32) {
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let bc1 = 1.0 - beta1.powi(step as i32);
        let bc2 = 1.0 - beta2.powi(step as i32);

        for idx in 0..self.weights.len() {
            let g = self.grad_w[idx].clamp(-2.0, 2.0);
            self.m_w[idx] = beta1 * self.m_w[idx] + (1.0 - beta1) * g;
            self.v_w[idx] = beta2 * self.v_w[idx] + (1.0 - beta2) * g * g;
            let m_hat = self.m_w[idx] / bc1;
            let v_hat = self.v_w[idx] / bc2;
            self.weights[idx] -=
                lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * self.weights[idx]);
        }
        for idx in 0..self.bias.len() {
            let g = self.grad_b[idx].clamp(-2.0, 2.0);
            self.m_b[idx] = beta1 * self.m_b[idx] + (1.0 - beta1) * g;
            self.v_b[idx] = beta2 * self.v_b[idx] + (1.0 - beta2) * g * g;
            let m_hat = self.m_b[idx] / bc1;
            let v_hat = self.v_b[idx] / bc2;
            self.bias[idx] -= lr * (m_hat / (v_hat.sqrt() + eps));
        }
    }

    pub fn param_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    /// Exact zero-multiplication Max-Plus forward pass (`tau == 0`).
    pub fn forward_exact(&self, x: &[f32]) -> (Vec<f32>, Vec<usize>) {
        assert_eq!(x.len(), self.in_features);
        let mut out = vec![0.0f32; self.out_features];
        let mut winners = vec![0usize; self.out_features];
        for j in 0..self.out_features {
            let row = j * self.in_features;
            let mut max_v = f32::NEG_INFINITY;
            let mut best_i = 0usize;
            for i in 0..self.in_features {
                let v = x[i] + self.weights[row + i];
                if v > max_v {
                    max_v = v;
                    best_i = i;
                }
            }
            out[j] = max_v + self.bias[j];
            winners[j] = best_i;
        }
        (out, winners)
    }

    /// Smooth temperature-annealed `logsumexp` forward pass returning `(output, softmax_probs)`.
    pub fn forward_smooth(&self, x: &[f32], tau: f32) -> (Vec<f32>, Vec<f32>) {
        if tau < 1e-4 {
            let (out, winners) = self.forward_exact(x);
            let mut probs = vec![0.0f32; self.out_features * self.in_features];
            for (j, &w_i) in winners.iter().enumerate() {
                probs[j * self.in_features + w_i] = 1.0;
            }
            return (out, probs);
        }

        let mut out = vec![0.0f32; self.out_features];
        let mut probs = vec![0.0f32; self.out_features * self.in_features];
        for j in 0..self.out_features {
            let row = j * self.in_features;
            let mut max_z = f32::NEG_INFINITY;
            for i in 0..self.in_features {
                let z = (x[i] + self.weights[row + i]) / tau;
                if z > max_z {
                    max_z = z;
                }
            }
            let mut sum_exp = 0.0f32;
            for i in 0..self.in_features {
                let e = (((x[i] + self.weights[row + i]) / tau) - max_z).exp();
                probs[row + i] = e;
                sum_exp += e;
            }
            for i in 0..self.in_features {
                probs[row + i] /= sum_exp.max(1e-12);
            }
            out[j] = self.bias[j] + tau * (max_z + sum_exp.max(1e-12).ln());
        }
        (out, probs)
    }

    /// Accumulates gradients into `(grad_w, grad_b)` and returns `grad_x` (`[in_features]`).
    pub fn backward_accum(&mut self, grad_out: &[f32], probs: &[f32]) -> Vec<f32> {
        let mut grad_x = vec![0.0f32; self.in_features];
        for j in 0..self.out_features {
            let go = grad_out[j];
            self.grad_b[j] += go;
            let row = j * self.in_features;
            for i in 0..self.in_features {
                let g = go * probs[row + i];
                self.grad_w[row + i] += g;
                grad_x[i] += g;
            }
        }
        grad_x
    }

    pub fn to_spec(&self) -> TropicalLayerSpec {
        TropicalLayerSpec {
            in_features: self.in_features,
            out_features: self.out_features,
            weights: self.weights.clone(),
            bias: self.bias.clone(),
        }
    }
}

/// 3,201-Parameter Zero-Multiplication Tropical Student (`16 -> 64 -> 32 -> 1`).
#[derive(Clone, Debug)]
pub struct TropicalPolicyStudent {
    pub l1: TropicalDense,
    pub l2: TropicalDense,
    pub l3: TropicalDense,
}

impl TropicalPolicyStudent {
    pub fn new(seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        Self {
            l1: TropicalDense::new(16, 64, &mut rng, 0.35),
            l2: TropicalDense::new(64, 32, &mut rng, 0.35),
            l3: TropicalDense::new(32, 1, &mut rng, 0.35),
        }
    }

    pub fn param_count(&self) -> usize {
        self.l1.param_count() + self.l2.param_count() + self.l3.param_count()
    }

    pub fn zero_grad(&mut self) {
        self.l1.zero_grad();
        self.l2.zero_grad();
        self.l3.zero_grad();
    }

    pub fn adamw_step(&mut self, step: usize, lr: f32, weight_decay: f32) {
        self.l1.adamw_step(step, lr, weight_decay);
        self.l2.adamw_step(step, lr, weight_decay);
        self.l3.adamw_step(step, lr, weight_decay);
    }

    /// Exact zero-multiplication forward pass (`tau = 0.0`).
    pub fn forward_exact(&self, x16: &[f32]) -> f32 {
        let (h1, _) = self.l1.forward_exact(x16);
        let (h2, _) = self.l2.forward_exact(&h1);
        let (out, _) = self.l3.forward_exact(&h2);
        out[0]
    }

    #[allow(dead_code)]
    /// Accumulates gradient for a single sample `(x16, target_y)` at temperature `tau`.
    pub fn forward_backward(&mut self, x16: &[f32], target_y: f32, tau: f32, batch_scale: f32) -> f32 {
        let (h1, p1) = self.l1.forward_smooth(x16, tau);
        let (h2, p2) = self.l2.forward_smooth(&h1, tau);
        let (out, p3) = self.l3.forward_smooth(&h2, tau);

        let diff = out[0] - target_y;
        let loss = diff * diff;
        let grad_out = [2.0 * diff * batch_scale];

        let grad_h2 = self.l3.backward_accum(&grad_out, &p3);
        let grad_h1 = self.l2.backward_accum(&grad_h2, &p2);
        let _grad_x = self.l1.backward_accum(&grad_h1, &p1);
        loss
    }

    /// Accumulates MSE + pairwise margin ranking gradient on `(x_chosen, x_rejected)`.
    pub fn forward_backward_pair(
        &mut self,
        x_c: &[f32],
        t_c: f32,
        x_r: &[f32],
        t_r: f32,
        tau: f32,
        batch_scale: f32,
    ) -> f32 {
        let (h1_c, p1_c) = self.l1.forward_smooth(x_c, tau);
        let (h2_c, p2_c) = self.l2.forward_smooth(&h1_c, tau);
        let (out_c, p3_c) = self.l3.forward_smooth(&h2_c, tau);

        let (h1_r, p1_r) = self.l1.forward_smooth(x_r, tau);
        let (h2_r, p2_r) = self.l2.forward_smooth(&h1_r, tau);
        let (out_r, p3_r) = self.l3.forward_smooth(&h2_r, tau);

        let diff_c = out_c[0] - t_c;
        let diff_r = out_r[0] - t_r;
        let mut g_c = 2.0 * diff_c * batch_scale;
        let mut g_r = 2.0 * diff_r * batch_scale;

        let margin_viol = 4.0 - (out_c[0] - out_r[0]);
        if margin_viol > 0.0 {
            g_c -= 2.5 * batch_scale;
            g_r += 2.5 * batch_scale;
        }

        let gh2_c = self.l3.backward_accum(&[g_c], &p3_c);
        let gh1_c = self.l2.backward_accum(&gh2_c, &p2_c);
        let _ = self.l1.backward_accum(&gh1_c, &p1_c);

        let gh2_r = self.l3.backward_accum(&[g_r], &p3_r);
        let gh1_r = self.l2.backward_accum(&gh2_r, &p2_r);
        let _ = self.l1.backward_accum(&gh1_r, &p1_r);

        0.5 * (diff_c * diff_c + diff_r * diff_r)
    }

    /// Computes the AGENT-PANDORA compatible SHA-256 digest over `(name, domain, layers)`.
    pub fn compute_sha256(name: &str, domain: &str, layers: &[TropicalLayerSpec]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(name.as_bytes());
        hasher.update(domain.as_bytes());
        for layer in layers {
            hasher.update(&(layer.in_features as u64).to_le_bytes());
            hasher.update(&(layer.out_features as u64).to_le_bytes());
            for &w in &layer.weights {
                hasher.update(&w.to_le_bytes());
            }
            for &b in &layer.bias {
                hasher.update(&b.to_le_bytes());
            }
        }
        hasher.finalize_hex()
    }

    /// Exports the AGENT-PANDORA compatible JSON `.pnn` file and returns its SHA-256 hex digest.
    pub fn save_json_pnn(&self, path: &Path) -> std::io::Result<String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let layers = vec![self.l1.to_spec(), self.l2.to_spec(), self.l3.to_spec()];
        let name = "moira".to_string();
        let domain = "decision_policy_scoring".to_string();
        let digest = Self::compute_sha256(&name, &domain, &layers);
        let payload = PnnArtifactJson {
            name,
            domain,
            layers,
            sha256_digest: digest.clone(),
        };
        let json_str = serde_json::to_string_pretty(&payload)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json_str)?;
        Ok(digest)
    }

    /// Exports the compact binary `PNN\x01` representation.
    pub fn save_binary_pnn(&self, path: &Path) -> std::io::Result<usize> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = File::create(path)?;
        f.write_all(PNN_MAGIC)?;
        f.write_all(&PNN_VERSION.to_le_bytes())?;
        let num_layers: u32 = 3;
        f.write_all(&num_layers.to_le_bytes())?;
        for layer in [&self.l1, &self.l2, &self.l3] {
            f.write_all(&(layer.in_features as u32).to_le_bytes())?;
            f.write_all(&(layer.out_features as u32).to_le_bytes())?;
            for &w in &layer.weights {
                f.write_all(&w.to_le_bytes())?;
            }
            for &b in &layer.bias {
                f.write_all(&b.to_le_bytes())?;
            }
        }
        f.flush()?;
        Ok(std::fs::metadata(path)?.len() as usize)
    }

    /// Loads a `TropicalPolicyStudent` from a binary `PNN\x01` file.
    pub fn load_binary_pnn(path: &Path) -> std::io::Result<Self> {
        let mut f = File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != PNN_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid .pnn magic header",
            ));
        }
        let mut b4 = [0u8; 4];
        f.read_exact(&mut b4)?;
        let _ver = u32::from_le_bytes(b4);
        f.read_exact(&mut b4)?;
        let num_layers = u32::from_le_bytes(b4);
        if num_layers != 3 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Expected 3 layers, found {num_layers}"),
            ));
        }

        let read_layer = |file: &mut File| -> std::io::Result<TropicalDense> {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            let in_features = u32::from_le_bytes(buf) as usize;
            file.read_exact(&mut buf)?;
            let out_features = u32::from_le_bytes(buf) as usize;
            let mut weights = vec![0.0f32; out_features * in_features];
            for w in &mut weights {
                file.read_exact(&mut buf)?;
                *w = f32::from_le_bytes(buf);
            }
            let mut bias = vec![0.0f32; out_features];
            for b in &mut bias {
                file.read_exact(&mut buf)?;
                *b = f32::from_le_bytes(buf);
            }
            Ok(TropicalDense::from_weights_and_bias(
                in_features,
                out_features,
                weights,
                bias,
            ))
        };

        Ok(Self {
            l1: read_layer(&mut f)?,
            l2: read_layer(&mut f)?,
            l3: read_layer(&mut f)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::TropicalPolicyStudent;

    #[test]
    fn test_tropical_student_param_count_and_binary_roundtrip() {
        let student = TropicalPolicyStudent::new(369);
        assert_eq!(student.param_count(), 3201);

        let x = [0.25f32; 16];
        let y1 = student.forward_exact(&x);

        let tmp = std::env::temp_dir().join("moira_test_bin.pnn");
        let sz = student.save_binary_pnn(&tmp).expect("save succeeds");
        assert_eq!(sz, 12 + 3 * 8 + 3201 * 4);

        let loaded = TropicalPolicyStudent::load_binary_pnn(&tmp).expect("load succeeds");
        let y2 = loaded.forward_exact(&x);
        assert!((y1 - y2).abs() < 1e-6);

        let _ = std::fs::remove_file(&tmp);
    }
}
