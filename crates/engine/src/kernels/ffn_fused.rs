//! Fused FFN forward kernel: SwiGLU gate + up + down + residual.
//!
//! Input IOSurface: [1, DIM, 1, 2*SEQ + 3*HIDDEN]
//!   sp[0:SEQ]                     = x2norm [DIM, SEQ]  (post-RMSNorm input)
//!   sp[SEQ:2*SEQ]                 = x2 [DIM, SEQ]      (pre-norm residual)
//!   sp[2*SEQ:2*SEQ+HIDDEN]        = W1 [DIM, HIDDEN]   (gate projection)
//!   sp[2*SEQ+HIDDEN:2*SEQ+2*HID]  = W3 [DIM, HIDDEN]   (up projection)
//!   sp[2*SEQ+2*HID:...]           = W2 [DIM, HIDDEN]   (down projection, stored as [DIM,HIDDEN])
//!
//! Output: [1, DIM + 3*HIDDEN, 1, SEQ]
//!   = concat(x_next, h1, h3, gate_out)
//!   where x_next = x2 + alpha * (gate_out @ W2^T)
//!
//! h1 = xnorm @ W1 (gate), h3 = xnorm @ W3 (up)
//! gate_out = silu(h1) * h3
//! ffn_out = gate_out @ W2^T  (W2 stored as [DIM,HIDDEN], transposed inside kernel)

use ane_bridge::ane::{Graph, Shape};
use crate::model::ModelConfig;

/// Build the fused FFN forward graph.
pub fn build(cfg: &ModelConfig) -> Graph {
    let seq = cfg.seq;
    let dim = cfg.dim;
    let hidden = cfg.hidden;
    let nlayers = cfg.nlayers;

    let sp_in = 2 * seq + 3 * hidden;
    let alpha = 1.0 / (2.0 * nlayers as f32).sqrt(); // DeepNet residual scaling

    let mut g = Graph::new();
    let input = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: sp_in });

    // ── Slice inputs ──
    let x2norm = g.slice(input, [0, 0, 0, 0], [1, dim, 1, seq]);
    let x2 = g.slice(input, [0, 0, 0, seq], [1, dim, 1, seq]);
    let w1 = g.slice(input, [0, 0, 0, 2 * seq], [1, dim, 1, hidden]);
    let w3 = g.slice(input, [0, 0, 0, 2 * seq + hidden], [1, dim, 1, hidden]);
    let w2_raw = g.slice(input, [0, 0, 0, 2 * seq + 2 * hidden], [1, dim, 1, hidden]);

    // ── Gate and up projections: xnorm @ W1, xnorm @ W3 ──
    // Reshape for matmul: [1,DIM,1,SEQ] → [1,1,DIM,SEQ] → [1,1,SEQ,DIM]
    let xn2 = g.reshape(x2norm, Shape { batch: 1, channels: 1, height: dim, width: seq });
    let xnt = g.transpose(xn2, [0, 1, 3, 2]);

    let w12 = g.reshape(w1, Shape { batch: 1, channels: 1, height: dim, width: hidden });
    let w32 = g.reshape(w3, Shape { batch: 1, channels: 1, height: dim, width: hidden });

    // [1,1,SEQ,DIM] @ [1,1,DIM,HIDDEN] → [1,1,SEQ,HIDDEN]
    let h1m = g.matrix_multiplication(xnt, w12, false, false);
    let h3m = g.matrix_multiplication(xnt, w32, false, false);

    // Reshape back to [1,HIDDEN,1,SEQ]
    let h1t = g.transpose(h1m, [0, 1, 3, 2]);
    let h3t = g.transpose(h3m, [0, 1, 3, 2]);
    let h1 = g.reshape(h1t, Shape { batch: 1, channels: hidden, height: 1, width: seq });
    let h3 = g.reshape(h3t, Shape { batch: 1, channels: hidden, height: 1, width: seq });

    // ── SiLU gate: silu(h1) * h3 ──
    let sig = g.sigmoid(h1);
    let silu = g.multiplication(h1, sig);
    let gate = g.multiplication(silu, h3);

    // ── Down projection: gate @ W2^T ──
    // W2 stored as [DIM, HIDDEN], need transpose for gate[HIDDEN,SEQ] @ W2^T → [DIM,SEQ]
    let g2 = g.reshape(gate, Shape { batch: 1, channels: 1, height: hidden, width: seq });
    let gt = g.transpose(g2, [0, 1, 3, 2]); // [1,1,SEQ,HIDDEN]
    let w22 = g.reshape(w2_raw, Shape { batch: 1, channels: 1, height: dim, width: hidden });
    let w2t = g.transpose(w22, [0, 1, 3, 2]); // [1,1,HIDDEN,DIM]
    let fm = g.matrix_multiplication(gt, w2t, false, false); // [1,1,SEQ,DIM]
    let ft = g.transpose(fm, [0, 1, 3, 2]);
    let ffn_out = g.reshape(ft, Shape { batch: 1, channels: dim, height: 1, width: seq });

    // ── Residual: x_next = x2 + alpha * ffn_out ──
    let alpha_const = g.constant_with_scalar(alpha, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let ffn_scaled = g.multiplication(ffn_out, alpha_const);
    let x_next = g.addition(x2, ffn_scaled);

    // ── Output: concat(x_next, h1, h3, gate) for backward pass caching ──
    let _out = g.concat(&[x_next, h1, h3, gate], 1);

    g
}

/// Input spatial width for ffnFused (matmul path: activations + weights).
pub fn input_spatial_width(cfg: &ModelConfig) -> usize {
    2 * cfg.seq + 3 * cfg.hidden
}

/// Input spatial width for conv1x1 ffnFused (activations only, no weights).
pub fn input_spatial_width_conv1x1(cfg: &ModelConfig) -> usize {
    2 * cfg.seq
}

/// Output channel count for ffnFused.
pub fn output_channels(cfg: &ModelConfig) -> usize {
    cfg.dim + 3 * cfg.hidden
}

/// Build the fused FFN forward graph using conv1x1 (weights as compile-time constants).
/// Weights are baked into the compiled program — must recompile when weights change.
///
/// Input IOSurface: [1, DIM, 1, 2*SEQ] (activations only)
///   sp[0:SEQ]     = x2norm [DIM, SEQ]
///   sp[SEQ:2*SEQ] = x2 [DIM, SEQ]
///
/// Output: [1, DIM + 3*HIDDEN, 1, SEQ] (same as matmul version)
pub fn build_conv1x1(cfg: &ModelConfig, w1: &[f32], w3: &[f32], w2: &[f32]) -> Graph {
    let seq = cfg.seq;
    let dim = cfg.dim;
    let hidden = cfg.hidden;
    let nlayers = cfg.nlayers;

    let sp_in = 2 * seq;
    let alpha = 1.0 / (2.0 * nlayers as f32).sqrt();

    let mut g = Graph::new();
    let input = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: sp_in });

    // ── Slice inputs ──
    let x2norm = g.slice(input, [0, 0, 0, 0], [1, dim, 1, seq]);
    let x2 = g.slice(input, [0, 0, 0, seq], [1, dim, 1, seq]);

    // ── Weight constants (transposed to [OC, IC] order for conv1x1) ──
    // W1 [DIM, HIDDEN] → transpose to [HIDDEN, DIM]
    let w1_t = transpose_weights(w1, dim, hidden);
    let w1_c = g.constant(&w1_t, Shape::spatial(hidden, 1, 1));

    // W3 [DIM, HIDDEN] → transpose to [HIDDEN, DIM]
    let w3_t = transpose_weights(w3, dim, hidden);
    let w3_c = g.constant(&w3_t, Shape::spatial(hidden, 1, 1));

    // W2 [DIM, HIDDEN] = [OC=DIM, IC=HIDDEN] — already correct order
    let w2_c = g.constant(w2, Shape::spatial(dim, 1, 1));

    // ── Projections via conv1x1 (1 op each, replaces 6 reshape/transpose/matmul ops) ──
    let h1 = g.convolution_2d_1x1(x2norm, w1_c, None); // [1, HIDDEN, 1, SEQ]
    let h3 = g.convolution_2d_1x1(x2norm, w3_c, None); // [1, HIDDEN, 1, SEQ]

    // ── SiLU gate: silu(h1) * h3 ──
    let sig = g.sigmoid(h1);
    let silu = g.multiplication(h1, sig);
    let gate = g.multiplication(silu, h3);

    // ── Down projection via conv1x1 ──
    let ffn_out = g.convolution_2d_1x1(gate, w2_c, None); // [1, DIM, 1, SEQ]

    // ── Residual: x_next = x2 + alpha * ffn_out ──
    let alpha_const = g.constant_with_scalar(alpha, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let ffn_scaled = g.multiplication(ffn_out, alpha_const);
    let x_next = g.addition(x2, ffn_scaled);

    // ── Output: concat(x_next, h1, h3, gate) for backward pass caching ──
    let _out = g.concat(&[x_next, h1, h3, gate], 1);

    g
}

/// Transpose a [ROWS, COLS] row-major matrix to [COLS, ROWS] row-major.
fn transpose_weights(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}
