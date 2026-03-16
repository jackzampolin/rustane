//! Benchmark: stage_all_weights timing
//! Run: cargo test -p engine --test bench_stage_weights --release -- --ignored --nocapture

use engine::model::ModelConfig;
use engine::layer::CompiledKernels;
use engine::full_model::ModelWeights;
use std::time::Instant;

#[test]
#[ignore]
fn bench_stage_weights() {
    let cfg = ModelConfig::gpt_1024();
    println!("\n=== Stage All Weights Benchmark (gpt_1024: 8L, 1024dim) ===\n");
    
    let mut kernels = CompiledKernels::compile(&cfg);
    let weights = ModelWeights::random(&cfg);
    
    // Measure stage_all_weights
    println!("Staging weights for {} layers...", cfg.nlayers);
    for run in 0..5 {
        let t = Instant::now();
        kernels.stage_all_weights(&cfg, &weights.layers);
        let ms = t.elapsed().as_secs_f32() * 1000.0;
        println!("  Run {}: {:.2}ms", run, ms);
    }
    
    println!("\nIf this is <5ms, we can afford to call it once per step (every 10 microbatches).");
    println!("Current staging overhead: ~50ms/step (102 calls × ~0.5ms each).");
    println!("With pre-staging: {:.2}ms/step (1 call per step).\n", {
        let t = Instant::now();
        kernels.stage_all_weights(&cfg, &weights.layers);
        t.elapsed().as_secs_f32() * 1000.0
    });
}
