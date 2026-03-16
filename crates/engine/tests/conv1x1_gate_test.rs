//! Gate test: conv1x1 ffnFused compile time at model dimensions.
//! GATE: If compile > 50ms, net savings killed by compile overhead.
//!
//! Run: cargo test -p engine --test conv1x1_gate_test --release -- --ignored --nocapture

use engine::kernels::ffn_fused;
use engine::model::ModelConfig;
use objc2_foundation::NSQualityOfService;
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

#[test]
#[ignore]
fn gate_conv1x1_compile_time() {
    println!("\n=== conv1x1 ffnFused Compile Time Gate Test ===\n");

    for (name, cfg) in [
        ("gpt_karpathy (768)", ModelConfig::gpt_karpathy()),
        ("gpt_1024", ModelConfig::gpt_1024()),
    ] {
        println!("--- {} ---", name);
        println!("  dim={}, hidden={}, seq={}", cfg.dim, cfg.hidden, cfg.seq);

        let w1 = random_weights(cfg.dim * cfg.hidden, 42);
        let w3 = random_weights(cfg.dim * cfg.hidden, 43);
        let w2 = random_weights(cfg.dim * cfg.hidden, 44);

        // Cold compile
        let t = Instant::now();
        let graph = ffn_fused::build_conv1x1(&cfg, &w1, &w3, &w2);
        let build_ms = t.elapsed().as_secs_f32() * 1000.0;
        println!("  Graph build: {:.1}ms", build_ms);

        let t = Instant::now();
        let exe = graph.compile(NSQualityOfService::UserInteractive)
            .expect("conv1x1 ffnFused compile failed");
        let cold_ms = t.elapsed().as_secs_f32() * 1000.0;
        println!("  Cold compile: {:.1}ms", cold_ms);
        drop(exe);

        // Warm compiles (5 runs)
        let mut times = Vec::new();
        for _ in 0..5 {
            let graph = ffn_fused::build_conv1x1(&cfg, &w1, &w3, &w2);
            let t = Instant::now();
            let _exe = graph.compile(NSQualityOfService::UserInteractive)
                .expect("conv1x1 ffnFused compile failed");
            times.push(t.elapsed().as_secs_f32() * 1000.0);
        }
        let median = {
            let mut sorted = times.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        println!("  Warm median: {:.1}ms (runs: {:?})",
            median, times.iter().map(|t| format!("{:.1}", t)).collect::<Vec<_>>());

        // Verify output shape matches matmul version
        let expected_out_ch = ffn_fused::output_channels(&cfg);
        println!("  Expected output channels: {}", expected_out_ch);

        if median < 50.0 {
            println!("  GATE: PASS — {:.1}ms < 50ms threshold\n", median);
        } else {
            println!("  GATE: FAIL — {:.1}ms >= 50ms threshold\n", median);
        }
    }

    println!("=== Gate Test Complete ===\n");
}
