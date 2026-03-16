//! Numerical equivalence test: conv1x1 ffnFused vs matmul ffnFused.
//! Verifies that both graph builders produce identical output channels.
//!
//! Run: cargo test -p engine --test conv1x1_correctness --release -- --ignored --nocapture

use engine::kernels::ffn_fused;
use engine::layer::{CompiledKernels, LayerWeights, ForwardCache};
use engine::model::ModelConfig;
use engine::layer;
use std::time::Instant;

fn random_weights(n: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    let mut s = seed;
    for x in v.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *x = ((s >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }
    v
}

/// Compare all output channels between matmul and conv1x1 forward paths.
#[test]
#[ignore]
fn conv1x1_vs_matmul_equivalence() {
    println!("\n=== conv1x1 vs matmul Numerical Equivalence Test ===\n");

    for (name, cfg) in [
        ("gpt_karpathy (768)", ModelConfig::gpt_karpathy()),
    ] {
        println!("--- {} ---", name);
        let dim = cfg.dim;
        let seq = cfg.seq;
        let hidden = cfg.hidden;

        for seed in [42u64, 123, 999] {
            println!("  seed={}", seed);

            // Generate random weights and input
            let weights = LayerWeights {
                wq: random_weights(cfg.dim * cfg.q_dim, seed),
                wk: random_weights(cfg.dim * cfg.kv_dim, seed + 1),
                wv: random_weights(cfg.dim * cfg.kv_dim, seed + 2),
                wo: random_weights(cfg.q_dim * cfg.dim, seed + 3),
                w1: random_weights(cfg.dim * cfg.hidden, seed + 4),
                w3: random_weights(cfg.dim * cfg.hidden, seed + 5),
                w2: random_weights(cfg.dim * cfg.hidden, seed + 6),
                gamma1: vec![1.0; cfg.dim],
                gamma2: vec![1.0; cfg.dim],
            };

            let x = random_weights(cfg.dim * cfg.seq, seed + 10);

            // Run matmul path (no conv1x1 compiled)
            let kernels_mm = CompiledKernels::compile(&cfg);
            assert!(!kernels_mm.has_ffn_conv1x1());

            let t = Instant::now();
            let (x_next_mm, cache_mm) = layer::forward(&cfg, &kernels_mm, &weights, &x, 0);
            println!("    matmul forward: {:.2}ms", t.elapsed().as_secs_f32() * 1000.0);

            // Run conv1x1 path — compile with this layer's weights
            let mut kernels_conv = CompiledKernels::compile(&cfg);
            // Build conv1x1 for a single "layer 0" using the same weights
            kernels_conv.recompile_ffn_conv1x1(&cfg, std::slice::from_ref(&weights));
            assert!(kernels_conv.has_ffn_conv1x1());

            let t = Instant::now();
            let (x_next_conv, cache_conv) = layer::forward(&cfg, &kernels_conv, &weights, &x, 0);
            println!("    conv1x1 forward: {:.2}ms", t.elapsed().as_secs_f32() * 1000.0);

            // Compare x_next
            let max_diff_xnext = x_next_mm.iter().zip(x_next_conv.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("    x_next max diff: {:.6}", max_diff_xnext);

            // Compare h1 (gate projection)
            let max_diff_h1 = cache_mm.h1.iter().zip(cache_conv.h1.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("    h1 max diff: {:.6}", max_diff_h1);

            // Compare h3 (up projection)
            let max_diff_h3 = cache_mm.h3.iter().zip(cache_conv.h3.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("    h3 max diff: {:.6}", max_diff_h3);

            // Compare gate (silu*h3)
            let max_diff_gate = cache_mm.gate.iter().zip(cache_conv.gate.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("    gate max diff: {:.6}", max_diff_gate);

            // Tolerance: fp16 truncation accumulates over DIM multiplies.
            // sqrt(DIM) × 1e-3 ≈ 0.028 for DIM=768, use 5e-2 for safety.
            let tol = 5e-2;
            assert!(max_diff_xnext < tol,
                "x_next diverged: max_diff={max_diff_xnext} > tol={tol}");
            assert!(max_diff_h1 < tol,
                "h1 diverged: max_diff={max_diff_h1} > tol={tol}");
            assert!(max_diff_h3 < tol,
                "h3 diverged: max_diff={max_diff_h3} > tol={tol}");
            assert!(max_diff_gate < tol,
                "gate diverged: max_diff={max_diff_gate} > tol={tol}");

            println!("    PASS (all diffs < {tol})");
        }
        println!();
    }

    println!("=== Equivalence Test Complete ===\n");
}
