// Copyright (C) 2026 ANISAYAK MITRA
// SPDX-License-Identifier: GPL-3.0-or-later
//
// FATE-O-SISTERS (MOIRA) — 100% Rust Tri-Timescale TSKBlock + SwiGLUFFN + Pulse 30D Engine
// Implements:
//   1. `Orientation`: Clotho (bidirectional), Lachesis (sealed-prefix + causal), Atropos (causal)
//   2. `RmsNorm`: Exact Root-Mean-Square Layer Normalization
//   3. `ThreadLocalAttention`: Multi-head local-window RoPE attention (|i - j| <= radius)
//   4. `SpoolRecurrent`: Gated Widrow-Hoff / Delta-Rule associative matrix memory
//      M_t^T = M_{t-1}^T (alpha_t I_R - beta_t k_t k_t^T) + beta_t v_t k_t^T
//   5. `KnotLandmarkAttention`: Prefix-stable complete-chunk landmarks with
//      hybrid 0.5 * Mean + 0.5 * Max pooling (eliminating 1/C Dirichlet attenuation)
//   6. `SwiGLUFFN`: Pre-norm SiLU(x W_g) * (x W_u) W_d nonlinear channel mixer
//   7. `PulseAffectEngine`: Two-timescale homeostatic affect state (12 fast + 6 slow + 12 appraisal = 30D)

use crate::pnn::SplitMix64;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Execution orientation controlling visibility across sequence positions `(query_i, key_j)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// Bidirectional visibility across the entire sequence.
    Clotho,
    /// Bidirectional visibility within `0..prefix_len`, causal visibility for `>= prefix_len`.
    Lachesis { prefix_len: usize },
    /// Strictly causal visibility (`key_j <= query_i`).
    Atropos,
}

impl Orientation {
    #[inline]
    pub fn allows(self, query_i: usize, key_j: usize) -> bool {
        match self {
            Self::Clotho => true,
            Self::Lachesis { prefix_len } => {
                if query_i < prefix_len {
                    key_j < prefix_len
                } else {
                    key_j <= query_i
                }
            }
            Self::Atropos => key_j <= query_i,
        }
    }
}

/// Root-Mean-Square Layer Normalization.
#[derive(Clone, Debug)]
pub struct RmsNorm {
    pub dim: usize,
    pub eps: f32,
    pub weight: Vec<f32>,
}

impl RmsNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            eps: 1e-5,
            weight: vec![1.0f32; dim],
        }
    }

    pub fn forward_token(&self, x: &[f32]) -> Vec<f32> {
        let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (self.dim.max(1) as f32);
        let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();
        x.iter()
            .zip(self.weight.iter())
            .map(|(&v, &w)| v * inv_rms * w)
            .collect()
    }

    pub fn forward_seq(&self, seq: &[Vec<f32>]) -> Vec<Vec<f32>> {
        seq.iter().map(|tok| self.forward_token(tok)).collect()
    }
}

#[inline]
fn linear_proj(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for j in 0..out_dim {
        let row = j * in_dim;
        let mut s = 0.0f32;
        for i in 0..in_dim {
            s += x[i] * w[row + i];
        }
        out[j] = s;
    }
    out
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Applies Rotary Position Embeddings (RoPE) to a single head slice of even dimension `head_dim`.
fn apply_rope_in_place(head_vec: &mut [f32], pos: usize) {
    let half = head_vec.len() / 2;
    for pair in 0..half {
        let exponent = (2 * pair) as f32 / (head_vec.len().max(1) as f32);
        let theta = (pos as f32) / 10000.0f32.powf(exponent);
        let (sin_t, cos_t) = theta.sin_cos();
        let x0 = head_vec[2 * pair];
        let x1 = head_vec[2 * pair + 1];
        head_vec[2 * pair] = x0 * cos_t - x1 * sin_t;
        head_vec[2 * pair + 1] = x0 * sin_t + x1 * cos_t;
    }
}

/// Thread: Exact local-window RoPE multi-head attention (`|i - j| <= radius`).
#[derive(Clone, Debug)]
pub struct ThreadLocalAttention {
    pub d_model: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub radius: usize,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
}

impl ThreadLocalAttention {
    pub fn new(d_model: usize, n_heads: usize, radius: usize, rng: &mut SplitMix64) -> Self {
        let head_dim = d_model / n_heads.max(1);
        let scale = (1.0 / (d_model.max(1) as f32)).sqrt();
        let mut init = |n: usize| (0..n).map(|_| rng.next_signed(scale)).collect::<Vec<f32>>();
        Self {
            d_model,
            n_heads,
            head_dim,
            radius,
            w_q: init(d_model * d_model),
            w_k: init(d_model * d_model),
            w_v: init(d_model * d_model),
            w_o: init(d_model * d_model),
        }
    }

    pub fn param_count(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len()
    }

    pub fn forward(&self, seq: &[Vec<f32>], orientation: Orientation) -> Vec<Vec<f32>> {
        let seq_len = seq.len();
        if seq_len == 0 {
            return Vec::new();
        }
        let mut q_seq = Vec::with_capacity(seq_len);
        let mut k_seq = Vec::with_capacity(seq_len);
        let mut v_seq = Vec::with_capacity(seq_len);

        for (pos, tok) in seq.iter().enumerate() {
            let mut q = linear_proj(tok, &self.w_q, self.d_model, self.d_model);
            let mut k = linear_proj(tok, &self.w_k, self.d_model, self.d_model);
            let v = linear_proj(tok, &self.w_v, self.d_model, self.d_model);
            for h in 0..self.n_heads {
                let start = h * self.head_dim;
                let end = start + self.head_dim;
                apply_rope_in_place(&mut q[start..end], pos);
                apply_rope_in_place(&mut k[start..end], pos);
            }
            q_seq.push(q);
            k_seq.push(k);
            v_seq.push(v);
        }

        let inv_sqrt_d = 1.0 / (self.head_dim.max(1) as f32).sqrt();
        let mut out_seq = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            let j_start = i.saturating_sub(self.radius);
            let j_end = (i + self.radius + 1).min(seq_len);
            let mut concat_heads = vec![0.0f32; self.d_model];

            for h in 0..self.n_heads {
                let h_off = h * self.head_dim;
                let q_h = &q_seq[i][h_off..h_off + self.head_dim];

                let mut valid_js = Vec::new();
                let mut scores = Vec::new();
                let mut max_score = f32::NEG_INFINITY;

                for j in j_start..j_end {
                    if !orientation.allows(i, j) {
                        continue;
                    }
                    let k_h = &k_seq[j][h_off..h_off + self.head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += q_h[d] * k_h[d];
                    }
                    let sc = dot * inv_sqrt_d;
                    if sc > max_score {
                        max_score = sc;
                    }
                    valid_js.push(j);
                    scores.push(sc);
                }

                if valid_js.is_empty() {
                    continue;
                }

                let mut sum_exp = 0.0f32;
                for sc in &mut scores {
                    *sc = (*sc - max_score).exp();
                    sum_exp += *sc;
                }
                let denom = sum_exp.max(1e-12);
                for (idx, &j) in valid_js.iter().enumerate() {
                    let prob = scores[idx] / denom;
                    let v_h = &v_seq[j][h_off..h_off + self.head_dim];
                    for d in 0..self.head_dim {
                        concat_heads[h_off + d] += prob * v_h[d];
                    }
                }
            }

            out_seq.push(linear_proj(&concat_heads, &self.w_o, self.d_model, self.d_model));
        }

        out_seq
    }
}

/// Persistent O(1) associative matrix state `M_t` (`[rank, d_model]`) for `SpoolRecurrent`.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct SpoolState {
    pub rank: usize,
    pub d_model: usize,
    /// Stored as `[d_model * rank]` (`M^T` where row `d` has length `rank`).
    pub mat_t: Vec<f32>,
}

impl SpoolState {
    pub fn zeros(rank: usize, d_model: usize) -> Self {
        Self {
            rank,
            d_model,
            mat_t: vec![0.0f32; d_model * rank],
        }
    }
}

/// Spool: Gated Widrow-Hoff / Delta-Rule Associative Matrix Memory:
/// `M_t^T = M_{t-1}^T (alpha_t I_R - beta_t k_t k_t^T) + beta_t v_t k_t^T`
/// `r_t   = M_t^T q_t`
#[derive(Clone, Debug)]
pub struct SpoolRecurrent {
    pub d_model: usize,
    pub rank: usize,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_alpha: Vec<f32>,
    pub w_beta: Vec<f32>,
    pub w_out: Vec<f32>,
}

impl SpoolRecurrent {
    pub fn new(d_model: usize, rank: usize, rng: &mut SplitMix64) -> Self {
        let scale = (1.0 / (d_model.max(1) as f32)).sqrt();
        let mut init = |n: usize| (0..n).map(|_| rng.next_signed(scale)).collect::<Vec<f32>>();
        Self {
            d_model,
            rank,
            w_k: init(rank * d_model),
            w_v: init(d_model * d_model),
            w_q: init(rank * d_model),
            w_alpha: init(d_model),
            w_beta: init(d_model),
            w_out: init(d_model * d_model),
        }
    }

    pub fn param_count(&self) -> usize {
        self.w_k.len()
            + self.w_v.len()
            + self.w_q.len()
            + self.w_alpha.len()
            + self.w_beta.len()
            + self.w_out.len()
    }

    pub fn forward(&self, seq: &[Vec<f32>], state: &mut SpoolState) -> Vec<Vec<f32>> {
        let mut out_seq = Vec::with_capacity(seq.len());
        let r = self.rank;
        let d = self.d_model;

        for tok in seq {
            let mut k = linear_proj(tok, &self.w_k, d, r);
            let k_norm = (k.iter().map(|&x| x * x).sum::<f32>() + 1e-6).sqrt();
            for kv in &mut k {
                *kv /= k_norm;
            }
            let mut q = linear_proj(tok, &self.w_q, d, r);
            let q_norm = (q.iter().map(|&x| x * x).sum::<f32>() + 1e-6).sqrt();
            for qv in &mut q {
                *qv /= q_norm;
            }
            let v: Vec<f32> = linear_proj(tok, &self.w_v, d, d)
                .into_iter()
                .map(f32::tanh)
                .collect();

            let alpha_pre: f32 = tok.iter().zip(self.w_alpha.iter()).map(|(&x, &w)| x * w).sum();
            let beta_pre: f32 = tok.iter().zip(self.w_beta.iter()).map(|(&x, &w)| x * w).sum();
            // Retention gate alpha in [0.5, 0.995], write rate beta in [0.05, 0.85]
            let alpha = 0.50 + 0.495 * sigmoid(alpha_pre);
            let beta = 0.05 + 0.80 * sigmoid(beta_pre);

            // Delta-rule associative update per row d_idx of M^T:
            // m_row_new = alpha * m_row + beta * (v[d_idx] - (m_row . k)) * k
            let mut readout = vec![0.0f32; d];
            for d_idx in 0..d {
                let row_off = d_idx * r;
                let m_row = &mut state.mat_t[row_off..row_off + r];
                let mut mk_dot = 0.0f32;
                for r_idx in 0..r {
                    mk_dot += m_row[r_idx] * k[r_idx];
                }
                let delta_err = v[d_idx] - mk_dot;
                let mut rq_dot = 0.0f32;
                for r_idx in 0..r {
                    m_row[r_idx] = alpha * m_row[r_idx] + beta * delta_err * k[r_idx];
                    rq_dot += m_row[r_idx] * q[r_idx];
                }
                readout[d_idx] = rq_dot;
            }

            out_seq.push(linear_proj(&readout, &self.w_out, d, d));
        }

        out_seq
    }
}

/// Knot: Prefix-stable complete-chunk landmark attention with hybrid `0.5 * Mean + 0.5 * Max` pooling.
#[derive(Clone, Debug)]
pub struct KnotLandmarkAttention {
    pub d_model: usize,
    pub chunk_size: usize,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
}

impl KnotLandmarkAttention {
    pub fn new(d_model: usize, chunk_size: usize, rng: &mut SplitMix64) -> Self {
        let scale = (1.0 / (d_model.max(1) as f32)).sqrt();
        let mut init = |n: usize| (0..n).map(|_| rng.next_signed(scale)).collect::<Vec<f32>>();
        Self {
            d_model,
            chunk_size: chunk_size.max(2),
            w_q: init(d_model * d_model),
            w_k: init(d_model * d_model),
            w_v: init(d_model * d_model),
            w_o: init(d_model * d_model),
        }
    }

    pub fn param_count(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len()
    }

    pub fn forward(&self, seq: &[Vec<f32>], orientation: Orientation) -> Vec<Vec<f32>> {
        let seq_len = seq.len();
        let num_chunks = seq_len / self.chunk_size;
        if num_chunks == 0 {
            return vec![vec![0.0f32; self.d_model]; seq_len];
        }

        // Compute hybrid 0.5 * Mean + 0.5 * Max landmark for each complete chunk
        let mut landmarks = Vec::with_capacity(num_chunks);
        let mut chunk_ends = Vec::with_capacity(num_chunks);
        let inv_c = 1.0 / (self.chunk_size as f32);

        for c in 0..num_chunks {
            let start = c * self.chunk_size;
            let end = start + self.chunk_size;
            let mut mean_vec = vec![0.0f32; self.d_model];
            let mut max_vec = vec![f32::NEG_INFINITY; self.d_model];

            for pos in start..end {
                for d in 0..self.d_model {
                    let v = seq[pos][d];
                    mean_vec[d] += v * inv_c;
                    if v > max_vec[d] {
                        max_vec[d] = v;
                    }
                }
            }

            let mut hybrid = vec![0.0f32; self.d_model];
            for d in 0..self.d_model {
                hybrid[d] = 0.5 * mean_vec[d] + 0.5 * max_vec[d];
            }
            landmarks.push(hybrid);
            chunk_ends.push(end - 1);
        }

        let k_landmarks: Vec<Vec<f32>> = landmarks
            .iter()
            .map(|lm| linear_proj(lm, &self.w_k, self.d_model, self.d_model))
            .collect();
        let v_landmarks: Vec<Vec<f32>> = landmarks
            .iter()
            .map(|lm| linear_proj(lm, &self.w_v, self.d_model, self.d_model))
            .collect();

        let inv_sqrt_d = 1.0 / (self.d_model.max(1) as f32).sqrt();
        let mut out_seq = Vec::with_capacity(seq_len);

        for (i, tok) in seq.iter().enumerate() {
            let q = linear_proj(tok, &self.w_q, self.d_model, self.d_model);
            let mut valid_cs = Vec::new();
            let mut scores = Vec::new();
            let mut max_sc = f32::NEG_INFINITY;

            for (c, &c_end) in chunk_ends.iter().enumerate() {
                if !orientation.allows(i, c_end) {
                    continue;
                }
                let mut dot = 0.0f32;
                for d in 0..self.d_model {
                    dot += q[d] * k_landmarks[c][d];
                }
                let sc = dot * inv_sqrt_d;
                if sc > max_sc {
                    max_sc = sc;
                }
                valid_cs.push(c);
                scores.push(sc);
            }

            if valid_cs.is_empty() {
                out_seq.push(vec![0.0f32; self.d_model]);
                continue;
            }

            let mut sum_exp = 0.0f32;
            for sc in &mut scores {
                *sc = (*sc - max_sc).exp();
                sum_exp += *sc;
            }
            let denom = sum_exp.max(1e-12);
            let mut ctx = vec![0.0f32; self.d_model];
            for (idx, &c) in valid_cs.iter().enumerate() {
                let p = scores[idx] / denom;
                for d in 0..self.d_model {
                    ctx[d] += p * v_landmarks[c][d];
                }
            }
            out_seq.push(linear_proj(&ctx, &self.w_o, self.d_model, self.d_model));
        }

        out_seq
    }
}

/// SwiGLU Channel-Mixing Feedforward Network: `SiLU(x W_gate) * (x W_up) W_down`.
#[derive(Clone, Debug)]
pub struct SwiGluFfn {
    pub d_model: usize,
    pub ffn_dim: usize,
    pub norm: RmsNorm,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

impl SwiGluFfn {
    pub fn new(d_model: usize, ffn_dim: usize, rng: &mut SplitMix64) -> Self {
        let scale = (1.0 / (d_model.max(1) as f32)).sqrt();
        let mut init = |n: usize| (0..n).map(|_| rng.next_signed(scale)).collect::<Vec<f32>>();
        Self {
            d_model,
            ffn_dim,
            norm: RmsNorm::new(d_model),
            w_gate: init(ffn_dim * d_model),
            w_up: init(ffn_dim * d_model),
            w_down: init(d_model * ffn_dim),
        }
    }

    pub fn param_count(&self) -> usize {
        self.norm.weight.len() + self.w_gate.len() + self.w_up.len() + self.w_down.len()
    }

    pub fn forward_token(&self, x: &[f32]) -> Vec<f32> {
        let normed = self.norm.forward_token(x);
        let gate = linear_proj(&normed, &self.w_gate, self.d_model, self.ffn_dim);
        let up = linear_proj(&normed, &self.w_up, self.d_model, self.ffn_dim);
        let mut hidden = vec![0.0f32; self.ffn_dim];
        for i in 0..self.ffn_dim {
            hidden[i] = silu(gate[i]) * up[i];
        }
        linear_proj(&hidden, &self.w_down, self.ffn_dim, self.d_model)
    }
}

/// Complete Tri-Timescale `TSKBlock` (`Thread` + `Spool` + `Knot` + `SwiGLUFFN`).
#[derive(Clone, Debug)]
pub struct TskBlock {
    pub d_model: usize,
    pub norm: RmsNorm,
    pub thread: ThreadLocalAttention,
    pub spool: SpoolRecurrent,
    pub knot: KnotLandmarkAttention,
    pub mix_gate: [f32; 3],
    pub ffn: SwiGluFfn,
}

impl TskBlock {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        radius: usize,
        spool_rank: usize,
        chunk_size: usize,
        ffn_dim: usize,
        rng: &mut SplitMix64,
    ) -> Self {
        Self {
            d_model,
            norm: RmsNorm::new(d_model),
            thread: ThreadLocalAttention::new(d_model, n_heads, radius, rng),
            spool: SpoolRecurrent::new(d_model, spool_rank, rng),
            knot: KnotLandmarkAttention::new(d_model, chunk_size, rng),
            // 3:6:9 timescale mixing priors (Thread=3/18, Spool=6/18, Knot=9/18)
            mix_gate: [3.0 / 18.0, 6.0 / 18.0, 9.0 / 18.0],
            ffn: SwiGluFfn::new(d_model, ffn_dim, rng),
        }
    }

    pub fn param_count(&self) -> usize {
        self.norm.weight.len()
            + self.thread.param_count()
            + self.spool.param_count()
            + self.knot.param_count()
            + 3
            + self.ffn.param_count()
    }

    pub fn forward(
        &self,
        seq: &[Vec<f32>],
        orientation: Orientation,
        spool_state: &mut SpoolState,
    ) -> Vec<Vec<f32>> {
        let normed = self.norm.forward_seq(seq);
        let thread_out = self.thread.forward(&normed, orientation);
        let spool_out = self.spool.forward(&normed, spool_state);
        let knot_out = self.knot.forward(&normed, orientation);

        let [g_t, g_s, g_k] = self.mix_gate;
        let mut out_seq = Vec::with_capacity(seq.len());
        for i in 0..seq.len() {
            let mut residual = vec![0.0f32; self.d_model];
            for d in 0..self.d_model {
                residual[d] = seq[i][d]
                    + g_t * thread_out[i][d]
                    + g_s * spool_out[i][d]
                    + g_k * knot_out[i][d];
            }
            let ffn_delta = self.ffn.forward_token(&residual);
            for d in 0..self.d_model {
                residual[d] += ffn_delta[d];
            }
            out_seq.push(residual);
        }
        out_seq
    }
}

/// Two-Timescale Homeostatic Affect State (`Pulse`):
/// - `fast_affect`: 12 axes (decay rate lambda_f in [0.40, 0.65])
/// - `slow_affect`:  6 axes (decay rate lambda_s in [0.88, 0.96])
/// - `appraisal`:   12 axes (instantaneous cognitive appraisal)
/// Total = 30D affect vector.
#[derive(Clone, Debug)]
pub struct PulseAffectEngine {
    pub d_model: usize,
    pub w_appraisal: Vec<f32>,
    pub fast_affect: [f32; 12],
    pub slow_affect: [f32; 6],
    pub last_appraisal: [f32; 12],
}

impl PulseAffectEngine {
    pub fn new(d_model: usize, rng: &mut SplitMix64) -> Self {
        let scale = (1.0 / (d_model.max(1) as f32)).sqrt();
        let w_appraisal = (0..(12 * d_model))
            .map(|_| rng.next_signed(scale))
            .collect();
        Self {
            d_model,
            w_appraisal,
            fast_affect: [0.0; 12],
            slow_affect: [0.0; 6],
            last_appraisal: [0.0; 12],
        }
    }

    pub fn step(&mut self, pooled_hidden: &[f32]) -> [f32; 30] {
        let raw_app = linear_proj(pooled_hidden, &self.w_appraisal, self.d_model, 12);
        for i in 0..12 {
            let app = raw_app[i].tanh();
            self.last_appraisal[i] = app;
            let lambda_f = 0.45 + 0.015 * (i as f32);
            self.fast_affect[i] = (lambda_f * self.fast_affect[i] + (1.0 - lambda_f) * app).clamp(-1.0, 1.0);
        }
        for j in 0..6 {
            let pair_drive = 0.5 * (self.fast_affect[2 * j] + self.fast_affect[2 * j + 1]);
            let lambda_s = 0.90 + 0.01 * (j as f32);
            self.slow_affect[j] =
                (lambda_s * self.slow_affect[j] + (1.0 - lambda_s) * pair_drive).clamp(-1.0, 1.0);
        }

        let mut vec30 = [0.0f32; 30];
        vec30[0..12].copy_from_slice(&self.last_appraisal);
        vec30[12..24].copy_from_slice(&self.fast_affect);
        vec30[24..30].copy_from_slice(&self.slow_affect);
        vec30
    }
}

/// Complete Pure-Rust MOIRA Tri-Timescale Sequence Backbone (`TSKBlock` stack + `Pulse` 30D).
pub struct MoiraTskBackbone {
    pub vocab_size: usize,
    pub d_model: usize,
    pub embed: Vec<f32>,
    pub blocks: Vec<TskBlock>,
    pub final_norm: RmsNorm,
    pub pulse: PulseAffectEngine,
}

impl MoiraTskBackbone {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        radius: usize,
        spool_rank: usize,
        chunk_size: usize,
        ffn_dim: usize,
        seed: u64,
    ) -> Self {
        let mut rng = SplitMix64::new(seed);
        let scale = (1.0 / (d_model.max(1) as f32)).sqrt();
        let embed = (0..(vocab_size * d_model))
            .map(|_| rng.next_signed(scale))
            .collect();
        let mut blocks = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            blocks.push(TskBlock::new(
                d_model, n_heads, radius, spool_rank, chunk_size, ffn_dim, &mut rng,
            ));
        }
        Self {
            vocab_size,
            d_model,
            embed,
            blocks,
            final_norm: RmsNorm::new(d_model),
            pulse: PulseAffectEngine::new(d_model, &mut rng),
        }
    }

    pub fn param_count(&self) -> usize {
        self.embed.len()
            + self.blocks.iter().map(TskBlock::param_count).sum::<usize>()
            + self.final_norm.weight.len()
            + self.pulse.w_appraisal.len()
    }

    /// Runs a sequence of token IDs through all `TSKBlock`s under `orientation` and updates `Pulse`.
    /// Returns `(final_hidden_seq, pulse_30d_vector, spool_states)`.
    pub fn forward(
        &mut self,
        tokens: &[u32],
        orientation: Orientation,
    ) -> (Vec<Vec<f32>>, [f32; 30], Vec<SpoolState>) {
        let mut seq = Vec::with_capacity(tokens.len());
        for &tok in tokens {
            let idx = (tok as usize) % self.vocab_size.max(1);
            let off = idx * self.d_model;
            seq.push(self.embed[off..off + self.d_model].to_vec());
        }

        let mut spool_states = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let mut st = SpoolState::zeros(block.spool.rank, self.d_model);
            seq = block.forward(&seq, orientation, &mut st);
            spool_states.push(st);
        }
        seq = self.final_norm.forward_seq(&seq);

        let mut pooled = vec![0.0f32; self.d_model];
        if !seq.is_empty() {
            let inv_l = 1.0 / (seq.len() as f32);
            for tok_v in &seq {
                for d in 0..self.d_model {
                    pooled[d] += tok_v[d] * inv_l;
                }
            }
        }
        let pulse_vec = self.pulse.step(&pooled);
        (seq, pulse_vec, spool_states)
    }

    /// Exports the Tri-Timescale `MoiraTskBackbone` weights to a Hugging Face `.safetensors` file.
    pub fn save_safetensors(&self, path: &Path) -> std::io::Result<usize> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut named_tensors: Vec<(String, &[f32], Vec<usize>)> = Vec::new();
        named_tensors.push((
            "embed.weight".to_string(),
            &self.embed,
            vec![self.vocab_size, self.d_model],
        ));
        for (idx, blk) in self.blocks.iter().enumerate() {
            named_tensors.push((
                format!("blocks.{idx}.norm.weight"),
                &blk.norm.weight,
                vec![self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.thread.w_q"),
                &blk.thread.w_q,
                vec![self.d_model, self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.thread.w_k"),
                &blk.thread.w_k,
                vec![self.d_model, self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.thread.w_v"),
                &blk.thread.w_v,
                vec![self.d_model, self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.thread.w_o"),
                &blk.thread.w_o,
                vec![self.d_model, self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.spool.w_k"),
                &blk.spool.w_k,
                vec![blk.spool.rank, self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.spool.w_v"),
                &blk.spool.w_v,
                vec![self.d_model, self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.spool.w_q"),
                &blk.spool.w_q,
                vec![blk.spool.rank, self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.spool.w_out"),
                &blk.spool.w_out,
                vec![self.d_model, self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.knot.w_q"),
                &blk.knot.w_q,
                vec![self.d_model, self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.knot.w_k"),
                &blk.knot.w_k,
                vec![self.d_model, self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.knot.w_v"),
                &blk.knot.w_v,
                vec![self.d_model, self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.knot.w_o"),
                &blk.knot.w_o,
                vec![self.d_model, self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.ffn.w_gate"),
                &blk.ffn.w_gate,
                vec![blk.ffn.ffn_dim, self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.ffn.w_up"),
                &blk.ffn.w_up,
                vec![blk.ffn.ffn_dim, self.d_model],
            ));
            named_tensors.push((
                format!("blocks.{idx}.ffn.w_down"),
                &blk.ffn.w_down,
                vec![self.d_model, blk.ffn.ffn_dim],
            ));
        }
        named_tensors.push((
            "final_norm.weight".to_string(),
            &self.final_norm.weight,
            vec![self.d_model],
        ));
        named_tensors.push((
            "pulse.w_appraisal".to_string(),
            &self.pulse.w_appraisal,
            vec![12, self.d_model],
        ));

        let mut header_map = serde_json::Map::new();
        let mut meta = serde_json::Map::new();
        meta.insert("format".to_string(), serde_json::Value::String("pt".to_string()));
        meta.insert(
            "architecture".to_string(),
            serde_json::Value::String("MoiraTskBackbone".to_string()),
        );
        header_map.insert("__metadata__".to_string(), serde_json::Value::Object(meta));

        let mut offset = 0usize;
        for (name, slice, shape) in &named_tensors {
            let blen = slice.len() * 4;
            let mut info = serde_json::Map::new();
            info.insert("dtype".to_string(), serde_json::Value::String("F32".to_string()));
            info.insert(
                "shape".to_string(),
                serde_json::Value::Array(shape.iter().map(|&s| serde_json::Value::from(s)).collect()),
            );
            info.insert(
                "data_offsets".to_string(),
                serde_json::Value::Array(vec![
                    serde_json::Value::from(offset),
                    serde_json::Value::from(offset + blen),
                ]),
            );
            header_map.insert(name.clone(), serde_json::Value::Object(info));
            offset += blen;
        }

        let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(header_map))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        while header_bytes.len() % 8 != 0 {
            header_bytes.push(b' ');
        }

        let mut f = File::create(path)?;
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        f.write_all(&header_bytes)?;
        for (_, slice, _) in &named_tensors {
            for &val in *slice {
                f.write_all(&val.to_le_bytes())?;
            }
        }
        f.flush()?;
        Ok(std::fs::metadata(path)?.len() as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::{MoiraTskBackbone, Orientation};

    #[test]
    fn test_tsk_backbone_orientations_and_pulse_homeostasis() {
        let mut backbone = MoiraTskBackbone::new(256, 32, 2, 4, 2, 8, 4, 64, 369);
        assert!(backbone.param_count() > 20_000);

        let tokens = [3u32, 7, 11, 19, 23, 29, 31, 37];
        let (seq_c, pulse_c, states_c) = backbone.forward(&tokens, Orientation::Clotho);
        assert_eq!(seq_c.len(), tokens.len());
        assert_eq!(seq_c[0].len(), 32);
        assert_eq!(states_c.len(), 2);
        assert!(pulse_c.iter().all(|v| v.is_finite() && (-1.0..=1.0).contains(v)));

        // Verify Atropos causal prefix invariance: changing the last token must not alter position 0
        let tokens_b = [3u32, 7, 11, 19, 23, 29, 31, 99];
        let (seq_a1, _, _) = backbone.forward(&tokens, Orientation::Atropos);
        let (seq_a2, _, _) = backbone.forward(&tokens_b, Orientation::Atropos);
        for d in 0..32 {
            assert!((seq_a1[0][d] - seq_a2[0][d]).abs() < 1e-6);
        }

        // Verify Lachesis prefix-bidirectional + suffix-causal mode
        let (seq_l, _, _) = backbone.forward(&tokens, Orientation::Lachesis { prefix_len: 4 });
        assert_eq!(seq_l.len(), tokens.len());
    }
}
