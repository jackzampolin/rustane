//! Profile individual operations inside one layer forward+backward.
//! Run: cargo test -p engine --test profile_layer_ops --release -- --ignored --nocapture

use engine::full_model::{self, ModelWeights, ModelGrads, TrainConfig};
use engine::layer::CompiledKernels;
use engine::model::ModelConfig;
use std::time::Instant;

#[test]
#[ignore]
fn profile_full_step_breakdown() {
    let cfg = ModelConfig::gpt_karpathy();
    let kernels = CompiledKernels::compile(&cfg);
    let weights = ModelWeights::random(&cfg);
    let mut grads = ModelGrads::zeros(&cfg);
    let tc = TrainConfig::default();

    let tokens: Vec<u32> = (0..cfg.seq).map(|i| ((i * 31 + 7) % cfg.vocab) as u32).collect();
    let targets: Vec<u32> = (1..=cfg.seq).map(|i| ((i * 31 + 7) % cfg.vocab) as u32).collect();

    // Warmup
    {
        grads.zero_out();
        let fwd = full_model::forward(&cfg, &kernels, &weights, &tokens, &targets, tc.softcap);
        full_model::backward(&cfg, &kernels, &weights, &fwd, &tokens, tc.softcap, tc.loss_scale, &mut grads);
    }

    // Profile 3 forward+backward passes
    println!("\n=== Layer-Level Profiling (3 runs) ===");
    println!("Model: 6L 768D 512S 8192V");
    println!("{:<20} {:>10} {:>10} {:>10}", "operation", "run1", "run2", "run3");
    println!("{}", "-".repeat(52));

    for _run in 0..3 {
        grads.zero_out();

        let t0 = Instant::now();
        let fwd = full_model::forward(&cfg, &kernels, &weights, &tokens, &targets, tc.softcap);
        let fwd_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t1 = Instant::now();
        full_model::backward(&cfg, &kernels, &weights, &fwd, &tokens, tc.softcap, tc.loss_scale, &mut grads);
        let bwd_ms = t1.elapsed().as_secs_f32() * 1000.0;

        let t2 = Instant::now();
        let _gnorm = full_model::grad_norm(&grads);
        full_model::clip_grads(&mut grads, tc.grad_clip);
        let clip_ms = t2.elapsed().as_secs_f32() * 1000.0;

        println!("forward:  {fwd_ms:>8.1}ms");
        println!("backward: {bwd_ms:>8.1}ms");
        println!("clip:     {clip_ms:>8.1}ms");
        println!("total:    {:>8.1}ms\n", fwd_ms + bwd_ms + clip_ms);
    }

    // Now time the full_model::forward components individually
    println!("\n=== Forward Pass Breakdown ===");

    let dim = cfg.dim;
    let seq = cfg.seq;
    let vocab = cfg.vocab;

    // 1. Embedding
    let t0 = Instant::now();
    let mut x_row = vec![0.0f32; seq * dim];
    engine::cpu::embedding::forward(&weights.embed, dim, &tokens, &mut x_row);
    let mut x = vec![0.0f32; dim * seq];
    for s in 0..seq { for c in 0..dim { x[c * seq + s] = x_row[s * dim + c]; } }
    println!("embed + transpose: {:.1}ms", t0.elapsed().as_secs_f32() * 1000.0);

    // 2. Forward through 6 layers (total, not individual)
    let t0 = Instant::now();
    let mut caches = Vec::with_capacity(cfg.nlayers);
    for l in 0..cfg.nlayers {
        let (x_next, cache) = engine::layer::forward(&cfg, &kernels, &weights.layers[l], &x);
        caches.push(cache);
        x = x_next;
    }
    let layer_total = t0.elapsed().as_secs_f32() * 1000.0;
    println!("6 layers total: {layer_total:.1}ms ({:.1}ms/layer)", layer_total / 6.0);

    // 3. Final RMSNorm
    let t0 = Instant::now();
    let x_prenorm = x;
    let mut x_final = vec![0.0f32; dim * seq];
    let mut rms_inv_final = vec![0.0f32; seq];
    let mut x_pos = vec![0.0f32; dim];
    let mut out_pos = vec![0.0f32; dim];
    for s in 0..seq {
        for c in 0..dim { x_pos[c] = x_prenorm[c * seq + s]; }
        rms_inv_final[s] = engine::cpu::rmsnorm::forward(&x_pos, &weights.gamma_final, &mut out_pos);
        for c in 0..dim { x_final[c * seq + s] = out_pos[c]; }
    }
    println!("final rmsnorm: {:.1}ms", t0.elapsed().as_secs_f32() * 1000.0);

    // 4. Logits
    let t0 = Instant::now();
    let mut x_final_row = vec![0.0f32; seq * dim];
    for s in 0..seq { for c in 0..dim { x_final_row[s * dim + c] = x_final[c * seq + s]; } }
    let mut logits = vec![0.0f32; seq * vocab];
    engine::cpu::vdsp::sgemm_at(&x_final_row, seq, dim, &weights.embed, vocab, &mut logits);
    println!("logits matmul: {:.1}ms", t0.elapsed().as_secs_f32() * 1000.0);

    // 5. Softcap
    let t0 = Instant::now();
    if tc.softcap > 0.0 {
        let inv_cap = 1.0 / tc.softcap;
        let mut scaled = vec![0.0f32; seq * vocab];
        engine::cpu::vdsp::vsmul(&logits, inv_cap, &mut scaled);
        engine::cpu::vdsp::tanhf(&scaled, &mut logits);
        let mut capped = vec![0.0f32; seq * vocab];
        engine::cpu::vdsp::vsmul(&logits, tc.softcap, &mut capped);
        logits.copy_from_slice(&capped);
    }
    println!("softcap: {:.1}ms", t0.elapsed().as_secs_f32() * 1000.0);

    // 6. Cross-entropy
    let t0 = Instant::now();
    for s in 0..seq {
        let tok_logits = &logits[s * vocab..(s + 1) * vocab];
        let _ = engine::cpu::cross_entropy::forward(tok_logits, targets[s] as usize);
    }
    println!("cross-entropy: {:.1}ms", t0.elapsed().as_secs_f32() * 1000.0);
}
