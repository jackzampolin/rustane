//! Benchmark ANE kernel recompile cost.
//! Gate test: if recompile < 5ms/kernel, conv1x1 with constant weights is viable for training.
//!
//! Run: cargo test -p engine --test bench_recompile --release -- --ignored --nocapture

use engine::model::ModelConfig;
use engine::layer::CompiledKernels;
use std::time::Instant;

#[test]
#[ignore]
fn bench_recompile_cost() {
    println!("\n=== ANE Kernel Recompile Benchmark ===");
    println!("Gate test: is recompile fast enough for per-step conv1x1?\n");
    
    let cfg = ModelConfig::gpt_1024();
    println!("Config: dim={}, hidden={}, seq={}, layers={}", cfg.dim, cfg.hidden, cfg.seq, cfg.nlayers);
    
    // Cold compile
    println!("\n--- Cold compile (all 10 kernels) ---");
    let t0 = Instant::now();
    let _kernels = CompiledKernels::compile(&cfg);
    let cold_ms = t0.elapsed().as_secs_f32() * 1000.0;
    println!("  {:.1}ms ({:.1}ms per kernel)", cold_ms, cold_ms / 10.0);
    
    // Warm recompile x5
    println!("\n--- Warm recompile (5 runs, all 10 kernels each) ---");
    let mut times = Vec::new();
    for i in 0..5 {
        let t = Instant::now();
        let _k = CompiledKernels::compile(&cfg);
        let ms = t.elapsed().as_secs_f32() * 1000.0;
        times.push(ms);
        println!("  Run {}: {:.1}ms ({:.1}ms/kernel)", i + 1, ms, ms / 10.0);
    }
    
    let avg = times.iter().sum::<f32>() / times.len() as f32;
    let per_kernel = avg / 10.0;
    
    println!("\n--- Results ---");
    println!("  Average: {:.1}ms total, {:.1}ms per kernel", avg, per_kernel);
    println!("  Per step (amortized over 10 microbatches): {:.1}ms", avg);
    
    if per_kernel < 5.0 {
        println!("\n  VERDICT: VIABLE — recompile is fast enough for per-step conv1x1");
        println!("  At {:.1}ms/step recompile cost, conv1x1 3x throughput gain would save ~30-40ms", avg);
    } else if per_kernel < 50.0 {
        println!("\n  VERDICT: MARGINAL — classifier conv1x1 only (amortized over microbatches)");
        println!("  Per-step cost {:.1}ms is too high for all kernels but OK for classifier alone", avg);
    } else {
        println!("\n  VERDICT: NOT VIABLE — recompile too slow for training");
    }
    
    // Also test 768 for comparison
    println!("\n--- 768 comparison ---");
    let cfg768 = ModelConfig::gpt_karpathy();
    let t = Instant::now();
    let _k = CompiledKernels::compile(&cfg768);
    let ms768 = t.elapsed().as_secs_f32() * 1000.0;
    println!("  768 compile: {:.1}ms ({:.1}ms/kernel)", ms768, ms768 / 10.0);
    
    println!("\n=== Done ===\n");
}
