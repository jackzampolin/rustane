//! Sweep model configurations to find the ANE sweet spot.
//!
//! Tests different dim, hidden, seq, and layer combos to understand
//! how staging vs compute scales on M4 Max.
//!
//! Run: cargo test -p engine --test sweep_configs --release -- --ignored --nocapture

use engine::model::ModelConfig;
use engine::layer::CompiledKernels;
use engine::full_model::{self, ModelWeights, ModelGrads, ModelOptState, ModelForwardWorkspace, ModelBackwardWorkspace, TrainConfig};
use engine::metal_adam::MetalAdam;
use std::time::Instant;

fn bench_config(name: &str, cfg: &ModelConfig) {
    // Compile
    let t0 = Instant::now();
    let kernels = CompiledKernels::compile(cfg);
    let compile_ms = t0.elapsed().as_secs_f32() * 1000.0;

    let mut weights = ModelWeights::random(cfg);
    let mut grads = ModelGrads::zeros(cfg);
    let mut opt = ModelOptState::zeros(cfg);
    let tc = TrainConfig::default();
    let metal_adam = MetalAdam::new().expect("Metal required");
    let mut fwd_ws = ModelForwardWorkspace::new(cfg);
    let mut bwd_ws = ModelBackwardWorkspace::new(cfg);

    let tokens: Vec<u32> = (0..cfg.seq).map(|i| ((i * 31 + 7) % cfg.vocab) as u32).collect();
    let targets: Vec<u32> = (1..=cfg.seq).map(|i| ((i * 31 + 7) % cfg.vocab) as u32).collect();

    // Warmup
    {
        grads.zero_out();
        let _loss = full_model::forward_ws(cfg, &kernels, &weights, &tokens, &targets, 0.0, &mut fwd_ws);
        full_model::backward_ws(cfg, &kernels, &weights, &fwd_ws, &tokens, 0.0, 256.0, &mut grads, &mut bwd_ws);
    }

    // Bench 3 steps
    let mut fwd_total = 0.0f32;
    let mut bwd_total = 0.0f32;
    let mut step_total = 0.0f32;
    let runs = 3;

    for _ in 0..runs {
        grads.zero_out();

        let t_step = Instant::now();

        let t_fwd = Instant::now();
        let _loss = full_model::forward_ws(cfg, &kernels, &weights, &tokens, &targets, 0.0, &mut fwd_ws);
        fwd_total += t_fwd.elapsed().as_secs_f32() * 1000.0;

        let t_bwd = Instant::now();
        full_model::backward_ws(cfg, &kernels, &weights, &fwd_ws, &tokens, 0.0, 256.0, &mut grads, &mut bwd_ws);
        bwd_total += t_bwd.elapsed().as_secs_f32() * 1000.0;

        step_total += t_step.elapsed().as_secs_f32() * 1000.0;
    }

    let fwd_ms = fwd_total / runs as f32;
    let bwd_ms = bwd_total / runs as f32;
    let total_ms = step_total / runs as f32;
    let per_layer = total_ms / cfg.nlayers as f32;

    // Estimate params
    let params_m = (cfg.dim * cfg.dim * 4 + cfg.dim * cfg.hidden * 3 + cfg.dim * 2) * cfg.nlayers + cfg.vocab * cfg.dim;
    let params_m = params_m as f32 / 1e6;

    // Estimate matmul FLOPs
    let fwd_flops = (3 * cfg.dim * cfg.dim * cfg.seq * 2
        + cfg.dim * cfg.dim * cfg.seq * 2
        + 3 * cfg.dim * cfg.hidden * cfg.seq * 2) * cfg.nlayers;
    let total_flops = fwd_flops * 3; // fwd + 2x bwd
    let gflops = total_flops as f32 / 1e9;
    let compute_ms = gflops / 7.3; // at 7.3 TFLOPS single kernel

    println!("{:<25} {:>4}L {:>5}d {:>5}h {:>4}s | {:>6.1}ms fwd {:>6.1}ms bwd {:>6.1}ms total | {:>5.1}ms/L | {:>5.0}M params | {:>5.1}G FLOP | {:>5.1}ms compute | {:>4.0}% overhead",
        name, cfg.nlayers, cfg.dim, cfg.hidden, cfg.seq,
        fwd_ms, bwd_ms, total_ms,
        per_layer, params_m, gflops, compute_ms,
        (1.0 - compute_ms / total_ms) * 100.0);
}

#[test]
#[ignore]
fn sweep_model_configs() {
    println!("\n=== Rustane Config Sweep — M4 Max ANE ===\n");
    println!("{:<25} {:>4}  {:>5}  {:>5}  {:>4} | {:>8} {:>8} {:>8} | {:>7} | {:>11} | {:>10} | {:>12} | overhead",
        "config", "L", "dim", "hid", "seq", "fwd", "bwd", "total", "per-L", "params", "FLOP", "ANE compute");
    println!("{}", "-".repeat(170));

    // === Vary layers (dim=768, seq=256) ===
    for nl in [2, 4, 6, 8, 12] {
        let cfg = ModelConfig {
            dim: 768, hidden: 2048, heads: 6, kv_heads: 6, hd: 128,
            seq: 256, nlayers: nl, vocab: 8192, q_dim: 768, kv_dim: 768, gqa_ratio: 1,
        };
        bench_config(&format!("layers-{}", nl), &cfg);
    }
    println!();

    // === Vary dim (6 layers, seq=256) ===
    for (dim, hidden, heads) in [(384, 1024, 3), (512, 1408, 4), (768, 2048, 6), (1024, 2816, 8), (1536, 4096, 12)] {
        let cfg = ModelConfig {
            dim, hidden, heads, kv_heads: heads, hd: 128,
            seq: 256, nlayers: 6, vocab: 8192, q_dim: dim, kv_dim: dim, gqa_ratio: 1,
        };
        bench_config(&format!("dim-{}", dim), &cfg);
    }
    println!();

    // === Vary seq (dim=768, 6 layers) ===
    for seq in [64, 128, 256, 512, 1024] {
        let cfg = ModelConfig {
            dim: 768, hidden: 2048, heads: 6, kv_heads: 6, hd: 128,
            seq, nlayers: 6, vocab: 8192, q_dim: 768, kv_dim: 768, gqa_ratio: 1,
        };
        bench_config(&format!("seq-{}", seq), &cfg);
    }
    println!();

    // === Vary hidden ratio (dim=768, 6 layers, seq=256) ===
    for (ratio, hidden) in [("2x", 1536), ("2.67x", 2048), ("4x", 3072), ("5.3x", 4096)] {
        let cfg = ModelConfig {
            dim: 768, hidden, heads: 6, kv_heads: 6, hd: 128,
            seq: 256, nlayers: 6, vocab: 8192, q_dim: 768, kv_dim: 768, gqa_ratio: 1,
        };
        bench_config(&format!("hidden-{}", ratio), &cfg);
    }
    println!();

    // === "Sweet spot" candidates ===
    // Stories110M equivalent
    let cfg = ModelConfig {
        dim: 768, hidden: 2048, heads: 6, kv_heads: 6, hd: 128,
        seq: 256, nlayers: 12, vocab: 8192, q_dim: 768, kv_dim: 768, gqa_ratio: 1,
    };
    bench_config("stories110m-like", &cfg);

    // Our gpt_karpathy
    bench_config("gpt_karpathy", &ModelConfig::gpt_karpathy());

    // Our gpt_1024
    bench_config("gpt_1024", &ModelConfig::gpt_1024());

    // Higher dim, fewer layers
    let cfg = ModelConfig {
        dim: 1536, hidden: 4096, heads: 12, kv_heads: 12, hd: 128,
        seq: 256, nlayers: 4, vocab: 8192, q_dim: 1536, kv_dim: 1536, gqa_ratio: 1,
    };
    bench_config("wide-shallow", &cfg);

    // Lower dim, more layers
    let cfg = ModelConfig {
        dim: 512, hidden: 1408, heads: 4, kv_heads: 4, hd: 128,
        seq: 512, nlayers: 16, vocab: 8192, q_dim: 512, kv_dim: 512, gqa_ratio: 1,
    };
    bench_config("narrow-deep", &cfg);

    println!("\n=== Done ===\n");
}

#[test]
#[ignore]
fn sweep_large_configs() {
    println!("\n=== Large Config Sweep — Pushing Past 1024 ===\n");
    println!("{:<25} {:>4}  {:>5}  {:>5}  {:>4} | {:>8} {:>8} {:>8} | {:>7} | {:>11} | {:>10} | {:>12} | overhead",
        "config", "L", "dim", "hid", "seq", "fwd", "bwd", "total", "per-L", "params", "FLOP", "ANE compute");
    println!("{}", "-".repeat(170));

    let configs: Vec<(&str, usize, usize, usize, usize, usize)> = vec![
        // (name, dim, hidden, heads, seq, nlayers)
        ("1024-8L-seq256",  1024, 2816, 8,  256, 8),
        ("1024-8L-seq512",  1024, 2816, 8,  512, 8),
        ("1024-12L-seq256", 1024, 2816, 8,  256, 12),
        ("1536-8L-seq256",  1536, 4096, 12, 256, 8),
        ("1536-8L-seq512",  1536, 4096, 12, 512, 8),
        ("2048-6L-seq256",  2048, 5632, 16, 256, 6),
        ("2048-8L-seq256",  2048, 5632, 16, 256, 8),
        ("2048-4L-seq512",  2048, 5632, 16, 512, 4),
        ("3072-4L-seq256",  3072, 8192, 24, 256, 4),
        ("4096-2L-seq256",  4096, 11264,32, 256, 2),
    ];

    for (name, dim, hidden, heads, seq, nlayers) in configs {
        let cfg = ModelConfig {
            dim, hidden, heads, kv_heads: heads, hd: 128,
            seq, nlayers, vocab: 8192, q_dim: dim, kv_dim: dim, gqa_ratio: 1,
        };
        bench_config(name, &cfg);
    }

    println!("\n=== Done ===\n");
}

#[test]
#[ignore]
fn sweep_extreme_configs() {
    println!("\n=== EXTREME Config Sweep — 2048+ dims, various shapes ===\n");
    println!("{:<30} {:>4}  {:>5}  {:>5}  {:>4} | {:>8} {:>8} {:>8} | {:>7} | {:>11} | {:>10} | {:>12} | overhead",
        "config", "L", "dim", "hid", "seq", "fwd", "bwd", "total", "per-L", "params", "FLOP", "ANE compute");
    println!("{}", "-".repeat(175));

    let configs: Vec<(&str, usize, usize, usize, usize, usize)> = vec![
        // 2048 dim — vary everything
        ("2048-2L-seq128",  2048, 5632, 16, 128, 2),
        ("2048-2L-seq256",  2048, 5632, 16, 256, 2),
        ("2048-2L-seq512",  2048, 5632, 16, 512, 2),
        ("2048-4L-seq128",  2048, 5632, 16, 128, 4),
        ("2048-4L-seq256",  2048, 5632, 16, 256, 4),
        ("2048-6L-seq128",  2048, 5632, 16, 128, 6),
        ("2048-8L-seq128",  2048, 5632, 16, 128, 8),
        ("2048-12L-seq128", 2048, 5632, 16, 128, 12),

        // 2048 — vary hidden ratio
        ("2048-4L-h4096",   2048, 4096, 16, 256, 4),
        ("2048-4L-h5632",   2048, 5632, 16, 256, 4),
        ("2048-4L-h8192",   2048, 8192, 16, 256, 4),

        // 3072 — test SRAM cliff
        ("3072-2L-seq128",  3072, 8192, 24, 128, 2),
        ("3072-2L-seq256",  3072, 8192, 24, 256, 2),
        ("3072-4L-seq128",  3072, 8192, 24, 128, 4),
        ("3072-2L-seq512",  3072, 8192, 24, 512, 2),

        // 4096 — extreme
        ("4096-1L-seq64",   4096, 11264, 32, 64,  1),
        ("4096-1L-seq128",  4096, 11264, 32, 128, 1),
        ("4096-1L-seq256",  4096, 11264, 32, 256, 1),
        ("4096-2L-seq64",   4096, 11264, 32, 64,  2),
        ("4096-2L-seq128",  4096, 11264, 32, 128, 2),
        ("4096-2L-seq256",  4096, 11264, 32, 256, 2),
        ("4096-4L-seq64",   4096, 11264, 32, 64,  4),
        ("4096-4L-seq128",  4096, 11264, 32, 128, 4),

        // 4096 — vary hidden
        ("4096-2L-h8192",   4096, 8192,  32, 128, 2),
        ("4096-2L-h11264",  4096, 11264, 32, 128, 2),
        ("4096-2L-h16384",  4096, 16384, 32, 128, 2),

        // 6144/8192 — can ANE even handle this?
        ("6144-1L-seq64",   6144, 16384, 48, 64,  1),
        ("8192-1L-seq64",   8192, 22528, 64, 64,  1),
    ];

    for (name, dim, hidden, heads, seq, nlayers) in configs {
        let cfg = ModelConfig {
            dim, hidden, heads, kv_heads: heads, hd: 128,
            seq, nlayers, vocab: 8192, q_dim: dim, kv_dim: dim, gqa_ratio: 1,
        };
        bench_config(name, &cfg);
    }

    println!("\n=== Done ===\n");
}
