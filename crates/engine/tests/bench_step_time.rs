//! Benchmark: full training step timing breakdown.
//!
//! Run manually:
//!   cargo test -p engine --test bench_step_time --release -- --ignored --nocapture

use engine::full_model::{self, ModelWeights, ModelGrads, ModelOptState, TrainConfig};
use engine::layer::CompiledKernels;
use engine::metal_adam::MetalAdam;
use engine::model::ModelConfig;
use std::time::Instant;

#[test]
#[ignore] // Run manually: cargo test -p engine --test bench_step_time --release -- --ignored --nocapture
fn bench_training_step() {
    let cfg = ModelConfig::gpt_karpathy();

    println!("\n=== Rustane Training Step Benchmark ===");
    println!("Model: gpt_karpathy (NL={}, DIM={}, SEQ={}, VOCAB={})",
             cfg.nlayers, cfg.dim, cfg.seq, cfg.vocab);

    // Compile all 10 ANE kernels
    println!("\nCompiling 10 ANE kernels...");
    let t0 = Instant::now();
    let kernels = CompiledKernels::compile(&cfg);
    println!("Compiled in {:.2}s", t0.elapsed().as_secs_f32());

    // Create random model state
    let mut weights = ModelWeights::random(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);
    let mut opt = ModelOptState::zeros(&cfg);
    let tc = TrainConfig::default();

    // Fixed tokens (deterministic, reproducible)
    let tokens: Vec<u32> = (0..cfg.seq).map(|i| ((i * 31 + 7) % cfg.vocab) as u32).collect();
    let targets: Vec<u32> = (1..=cfg.seq).map(|i| ((i * 31 + 7) % cfg.vocab) as u32).collect();

    // Init Metal Adam optimizer
    let metal_adam = MetalAdam::new().expect("Metal GPU required");

    // Warmup (1 step, not timed)
    println!("\nWarmup step...");
    {
        grads.zero_out();
        let fwd = full_model::forward(&cfg, &kernels, &weights, &tokens, &targets, tc.softcap);
        full_model::backward(&cfg, &kernels, &weights, &fwd, &tokens, tc.softcap, tc.loss_scale, &mut grads);
        let _gsc = 1.0 / tc.loss_scale;
        full_model::clip_grads(&mut grads, tc.grad_clip);
        let lr = full_model::learning_rate(0, &tc);
        full_model::update_weights(&cfg, &mut weights, &grads, &mut opt, 1, lr, &tc, &metal_adam);
    }

    // Benchmark (5 steps)
    println!("\n{:<6} {:>10} {:>10} {:>10} {:>10}   {}", "step", "total", "fwd", "bwd", "upd", "loss");
    println!("{}", "-".repeat(70));

    for step in 0..5u32 {
        grads.zero_out();

        let t0 = Instant::now();

        // Forward
        let t_fwd_start = Instant::now();
        let fwd = full_model::forward(&cfg, &kernels, &weights, &tokens, &targets, tc.softcap);
        let t_fwd = t_fwd_start.elapsed();

        // Backward
        let t_bwd_start = Instant::now();
        full_model::backward(&cfg, &kernels, &weights, &fwd, &tokens, tc.softcap, tc.loss_scale, &mut grads);
        let t_bwd = t_bwd_start.elapsed();

        // Fused scale+clip (single pass instead of two)
        let gsc = 1.0 / tc.loss_scale;
        let raw_norm = full_model::grad_norm(&grads);
        let scaled_norm = raw_norm * gsc;
        let combined_scale = if scaled_norm > tc.grad_clip {
            tc.grad_clip / raw_norm
        } else {
            gsc
        };
        scale_grads(&mut grads, combined_scale);

        // Update weights (Adam)
        let t_upd_start = Instant::now();
        let lr = full_model::learning_rate(step + 2, &tc); // +2 because warmup was step 1
        full_model::update_weights(&cfg, &mut weights, &grads, &mut opt, step + 2, lr, &tc, &metal_adam);
        let t_upd = t_upd_start.elapsed();

        let total = t0.elapsed();
        let loss = fwd.loss;

        println!("{:<6} {:>9.1}ms {:>9.1}ms {:>9.1}ms {:>9.1}ms   {:.4}",
                 step,
                 total.as_secs_f32() * 1000.0,
                 t_fwd.as_secs_f32() * 1000.0,
                 t_bwd.as_secs_f32() * 1000.0,
                 t_upd.as_secs_f32() * 1000.0,
                 loss);
    }

    println!("\n=== Benchmark complete ===\n");
}

/// Scale all gradient tensors by a scalar factor (in-place cblas_sscal).
fn scale_grads(grads: &mut ModelGrads, scale: f32) {
    engine::cpu::vdsp::sscal(&mut grads.dembed, scale);
    engine::cpu::vdsp::sscal(&mut grads.dgamma_final, scale);
    for lg in &mut grads.layers {
        engine::cpu::vdsp::sscal(&mut lg.dwq, scale);
        engine::cpu::vdsp::sscal(&mut lg.dwk, scale);
        engine::cpu::vdsp::sscal(&mut lg.dwv, scale);
        engine::cpu::vdsp::sscal(&mut lg.dwo, scale);
        engine::cpu::vdsp::sscal(&mut lg.dw1, scale);
        engine::cpu::vdsp::sscal(&mut lg.dw3, scale);
        engine::cpu::vdsp::sscal(&mut lg.dw2, scale);
        engine::cpu::vdsp::sscal(&mut lg.dgamma1, scale);
        engine::cpu::vdsp::sscal(&mut lg.dgamma2, scale);
    }
}
