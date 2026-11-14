// Copyright (C) 2026 ANISAYAK MITRA
// SPDX-License-Identifier: GPL-3.0-or-later

/// Deterministic structural token sifter for algorithmic & reasoning streams.
/// Filters natural-language filler and maps structural tokens into `[0, vocab_limit)`.
pub struct DataSifter {
    vocab_limit: usize,
}

impl Default for DataSifter {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSifter {
    pub fn new() -> Self {
        Self { vocab_limit: 4096 }
    }

    #[allow(dead_code)]
    pub fn with_vocab_limit(vocab_limit: usize) -> Self {
        Self {
            vocab_limit: vocab_limit.max(1),
        }
    }

    /// Sifts an input code/logic stream, discarding common filler words and hashing
    /// structural tokens into deterministic token IDs within `[0, vocab_limit)`.
    pub fn sift_stream_trajectory(&self, source_text: &str) -> Vec<u32> {
        const FILLER: &[&str] = &[
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "of", "to", "in",
        ];

        source_text
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|tok| !tok.is_empty())
            .filter(|tok| {
                let lower = tok.to_ascii_lowercase();
                !FILLER.contains(&lower.as_str())
            })
            .map(|tok| {
                // FNV-1a 64-bit deterministic hash into [0, vocab_limit)
                let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
                for &b in tok.as_bytes() {
                    hash ^= u64::from(b);
                    hash = hash.wrapping_mul(0x0100_0000_01b3);
                }
                (hash % (self.vocab_limit as u64)) as u32
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::DataSifter;

    #[test]
    fn test_data_sifter_filters_filler_and_bounds_tokens() {
        let sifter = DataSifter::with_vocab_limit(256);
        let tokens = sifter.sift_stream_trajectory("the fn forward(x) and a return x");
        assert!(!tokens.is_empty());
        assert!(tokens.iter().all(|&t| t < 256));
    }
}
