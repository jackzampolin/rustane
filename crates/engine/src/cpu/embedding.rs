//! Embedding lookup and gradient scatter.
//!
//! Forward: out[i] = table[token_ids[i]]  (row lookup)
//! Backward: d_table[token_ids[i]] += d_out[i]  (gradient scatter/accumulate)

/// Embedding forward: lookup rows from `table` by `token_ids`.
/// `table`: [vocab_size, dim], `token_ids`: [seq_len], `out`: [seq_len, dim].
pub fn forward(table: &[f32], dim: usize, token_ids: &[u32], out: &mut [f32]) {
    let seq_len = token_ids.len();
    assert_eq!(out.len(), seq_len * dim);

    for (s, &tok) in token_ids.iter().enumerate() {
        let src = &table[tok as usize * dim..(tok as usize + 1) * dim];
        out[s * dim..(s + 1) * dim].copy_from_slice(src);
    }
}

/// Embedding backward: scatter-add gradients back into `d_table`.
/// `d_out`: [seq_len, dim], `token_ids`: [seq_len], `d_table`: [vocab_size, dim] (accumulated).
pub fn backward(d_out: &[f32], dim: usize, token_ids: &[u32], d_table: &mut [f32]) {
    let seq_len = token_ids.len();
    assert_eq!(d_out.len(), seq_len * dim);

    for (s, &tok) in token_ids.iter().enumerate() {
        let offset = tok as usize * dim;
        for d in 0..dim {
            d_table[offset + d] += d_out[s * dim + d];
        }
    }
}

/// Channel-first embedding backward: scatter-add from [dim, seq] layout.
/// `d_out`: [dim, seq] channel-first, `token_ids`: [seq], `d_table`: [vocab, dim] (accumulated).
pub fn backward_channel_first(d_out: &[f32], dim: usize, token_ids: &[u32], d_table: &mut [f32]) {
    let seq = token_ids.len();
    assert_eq!(d_out.len(), dim * seq);

    for (s, &tok) in token_ids.iter().enumerate() {
        let offset = tok as usize * dim;
        for d in 0..dim {
            d_table[offset + d] += d_out[d * seq + s];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_lookup() {
        // vocab=3, dim=2
        let table = [
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
            5.0, 6.0, // token 2
        ];
        let ids = [2u32, 0, 1];
        let mut out = [0.0f32; 6]; // 3 tokens * dim 2
        forward(&table, 2, &ids, &mut out);
        assert_eq!(out, [5.0, 6.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn backward_scatter_add() {
        // vocab=3, dim=2
        let d_out = [
            0.1, 0.2, // grad for token at pos 0
            0.3, 0.4, // grad for token at pos 1
            0.5, 0.6, // grad for token at pos 2
        ];
        let ids = [1u32, 1, 2]; // token 1 appears twice
        let mut d_table = [0.0f32; 6]; // vocab=3, dim=2
        backward(&d_out, 2, &ids, &mut d_table);

        // token 0: no gradients
        assert_eq!(d_table[0], 0.0);
        assert_eq!(d_table[1], 0.0);
        // token 1: 0.1+0.3=0.4, 0.2+0.4=0.6
        assert!((d_table[2] - 0.4).abs() < 1e-6);
        assert!((d_table[3] - 0.6).abs() < 1e-6);
        // token 2: 0.5, 0.6
        assert!((d_table[4] - 0.5).abs() < 1e-6);
        assert!((d_table[5] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn roundtrip_identity() {
        let table = [10.0, 20.0, 30.0, 40.0]; // vocab=2, dim=2
        let ids = [0u32, 1];
        let mut out = [0.0f32; 4];
        forward(&table, 2, &ids, &mut out);
        assert_eq!(out, table);
    }
}
