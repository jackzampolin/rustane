//! Single transformer layer: compile kernels, run forward/backward on ANE + CPU.
//!
//! Forward: RMSNorm1(CPU) → sdpaFwd(ANE) → woFwd(ANE) → residual+RMSNorm2(CPU) → ffnFused(ANE)
//! Backward: Scale dy → ffnBwdW2t(ANE) → SiLU'(CPU) → ffnBwdW13t(ANE) → RMSNorm2 bwd(CPU)
//!           → wotBwd(ANE) → sdpaBwd1(ANE) → sdpaBwd2(ANE) → RoPE bwd(CPU)
//!           → qBwd(ANE) → kvBwd(ANE) → RMSNorm1 bwd(CPU)

use ane_bridge::ane::{Executable, Shape, TensorData};
use objc2_foundation::NSQualityOfService;
use crate::cpu::{rmsnorm, vdsp};
use crate::kernels::{dyn_matmul, sdpa_fwd, sdpa_bwd, ffn_fused};
use crate::model::ModelConfig;
use std::time::Instant;

/// Per-layer weights (f32, CPU-side).
pub struct LayerWeights {
    pub wq: Vec<f32>,      // [DIM * Q_DIM]
    pub wk: Vec<f32>,      // [DIM * KV_DIM]
    pub wv: Vec<f32>,      // [DIM * KV_DIM]
    pub wo: Vec<f32>,      // [Q_DIM * DIM]
    pub w1: Vec<f32>,      // [DIM * HIDDEN]
    pub w3: Vec<f32>,      // [DIM * HIDDEN]
    pub w2: Vec<f32>,      // [DIM * HIDDEN]
    pub gamma1: Vec<f32>,  // [DIM]
    pub gamma2: Vec<f32>,  // [DIM]
}

/// Weight gradients (same layout as weights).
pub struct LayerGrads {
    pub dwq: Vec<f32>,
    pub dwk: Vec<f32>,
    pub dwv: Vec<f32>,
    pub dwo: Vec<f32>,
    pub dw1: Vec<f32>,
    pub dw3: Vec<f32>,
    pub dw2: Vec<f32>,
    pub dgamma1: Vec<f32>,
    pub dgamma2: Vec<f32>,
}

/// Cached activations from forward pass, needed for backward.
pub struct ForwardCache {
    pub x: Vec<f32>,           // layer input [DIM * SEQ]
    pub xnorm: Vec<f32>,       // after RMSNorm1 [DIM * SEQ]
    pub rms_inv1: Vec<f32>,    // per-position rms_inv [SEQ]
    pub q_rope: Vec<f32>,      // [Q_DIM * SEQ]
    pub k_rope: Vec<f32>,      // [KV_DIM * SEQ]
    pub v: Vec<f32>,           // [KV_DIM * SEQ]
    pub attn_out: Vec<f32>,    // [Q_DIM * SEQ]
    pub o_out: Vec<f32>,       // woFwd output [DIM * SEQ]
    pub x2: Vec<f32>,          // post-attn residual [DIM * SEQ]
    pub x2norm: Vec<f32>,      // after RMSNorm2 [DIM * SEQ]
    pub rms_inv2: Vec<f32>,    // per-position rms_inv [SEQ]
    pub h1: Vec<f32>,          // gate projection [HIDDEN * SEQ]
    pub h3: Vec<f32>,          // up projection [HIDDEN * SEQ]
    pub gate: Vec<f32>,        // silu(h1) * h3 [HIDDEN * SEQ]
}


/// Pre-allocated IOSurface buffers for all 10 kernels (input + output each).
/// Eliminates ~100 IOSurface alloc/dealloc cycles per training step.
/// All writes use `TensorData::copy_from_f32(&self, ..)` which takes `&self`,
/// so no interior mutability wrapper is needed.
pub struct KernelBuffers {
    // Forward: sdpa_fwd, wo_fwd, ffn_fused
    sdpa_fwd_in: TensorData,
    sdpa_fwd_out: TensorData,
    wo_fwd_in: TensorData,
    wo_fwd_out: TensorData,
    ffn_fused_in: TensorData,
    ffn_fused_out: TensorData,
    // Backward: ffn_bwd_w2t, ffn_bwd_w13t, wot_bwd, sdpa_bwd1, sdpa_bwd2, q_bwd, kv_bwd
    ffn_bwd_w2t_in: TensorData,
    ffn_bwd_w2t_out: TensorData,
    ffn_bwd_w13t_in: TensorData,
    ffn_bwd_w13t_out: TensorData,
    wot_bwd_in: TensorData,
    wot_bwd_out: TensorData,
    sdpa_bwd1_in: TensorData,
    sdpa_bwd1_out: TensorData,
    sdpa_bwd2_in: TensorData,
    sdpa_bwd2_out: TensorData,
    q_bwd_in: TensorData,
    q_bwd_out: TensorData,
    kv_bwd_in: TensorData,
    kv_bwd_out: TensorData,
}

impl KernelBuffers {
    /// Pre-allocate all IOSurface buffers for the given model config.
    fn allocate(cfg: &ModelConfig) -> Self {
        let dim = cfg.dim;
        let seq = cfg.seq;
        let q_dim = cfg.q_dim;
        let kv_dim = cfg.kv_dim;
        let hidden = cfg.hidden;

        // Forward: sdpa_fwd
        let sdpa_sp = sdpa_fwd::input_spatial_width(cfg);
        let sdpa_out_ch = sdpa_fwd::output_channels(cfg);
        let sdpa_fwd_in = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: sdpa_sp });
        let sdpa_fwd_out = TensorData::new(Shape { batch: 1, channels: sdpa_out_ch, height: 1, width: seq });

        // Forward: wo_fwd
        let wo_sp = dyn_matmul::spatial_width(seq, dim);
        let wo_fwd_in = TensorData::new(Shape { batch: 1, channels: q_dim, height: 1, width: wo_sp });
        let wo_fwd_out = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });

        // Forward: ffn_fused
        let ffn_sp = ffn_fused::input_spatial_width(cfg);
        let ffn_out_ch = ffn_fused::output_channels(cfg);
        let ffn_fused_in = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: ffn_sp });
        let ffn_fused_out = TensorData::new(Shape { batch: 1, channels: ffn_out_ch, height: 1, width: seq });

        // Backward: ffn_bwd_w2t
        let w2t_sp = dyn_matmul::spatial_width(seq, hidden);
        let ffn_bwd_w2t_in = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: w2t_sp });
        let ffn_bwd_w2t_out = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: seq });

        // Backward: ffn_bwd_w13t
        let w13t_sp = dyn_matmul::dual_spatial_width(seq, dim);
        let ffn_bwd_w13t_in = TensorData::new(Shape { batch: 1, channels: hidden, height: 1, width: w13t_sp });
        let ffn_bwd_w13t_out = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });

        // Backward: wot_bwd
        let wot_sp = dyn_matmul::spatial_width(seq, q_dim);
        let wot_bwd_in = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: wot_sp });
        let wot_bwd_out = TensorData::new(Shape { batch: 1, channels: q_dim, height: 1, width: seq });

        // Backward: sdpa_bwd1
        let bwd1_in_ch = sdpa_bwd::bwd1_input_channels(cfg);
        let bwd1_out_ch = sdpa_bwd::bwd1_output_channels(cfg);
        let sdpa_bwd1_in = TensorData::new(Shape { batch: 1, channels: bwd1_in_ch, height: 1, width: seq });
        let sdpa_bwd1_out = TensorData::new(Shape { batch: 1, channels: bwd1_out_ch, height: 1, width: seq });

        // Backward: sdpa_bwd2
        let bwd2_in_ch = sdpa_bwd::bwd2_input_channels(cfg);
        let bwd2_out_ch = sdpa_bwd::bwd2_output_channels(cfg);
        let sdpa_bwd2_in = TensorData::new(Shape { batch: 1, channels: bwd2_in_ch, height: 1, width: seq });
        let sdpa_bwd2_out = TensorData::new(Shape { batch: 1, channels: bwd2_out_ch, height: 1, width: seq });

        // Backward: q_bwd
        let q_bwd_sp = dyn_matmul::spatial_width(seq, dim);
        let q_bwd_in = TensorData::new(Shape { batch: 1, channels: q_dim, height: 1, width: q_bwd_sp });
        let q_bwd_out = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });

        // Backward: kv_bwd
        let kv_bwd_sp = dyn_matmul::dual_spatial_width(seq, dim);
        let kv_bwd_in = TensorData::new(Shape { batch: 1, channels: kv_dim, height: 1, width: kv_bwd_sp });
        let kv_bwd_out = TensorData::new(Shape { batch: 1, channels: dim, height: 1, width: seq });

        Self {
            sdpa_fwd_in, sdpa_fwd_out,
            wo_fwd_in, wo_fwd_out,
            ffn_fused_in, ffn_fused_out,
            ffn_bwd_w2t_in, ffn_bwd_w2t_out,
            ffn_bwd_w13t_in, ffn_bwd_w13t_out,
            wot_bwd_in, wot_bwd_out,
            sdpa_bwd1_in, sdpa_bwd1_out,
            sdpa_bwd2_in, sdpa_bwd2_out,
            q_bwd_in, q_bwd_out,
            kv_bwd_in, kv_bwd_out,
        }
    }
}

/// Compiled kernels for one layer (shared across layers since same dims).
pub struct CompiledKernels {
    pub sdpa_fwd: Executable,
    pub wo_fwd: Executable,
    pub ffn_fused: Executable,
    pub ffn_bwd_w2t: Executable,
    pub ffn_bwd_w13t: Executable,
    pub wot_bwd: Executable,
    pub sdpa_bwd1: Executable,
    pub sdpa_bwd2: Executable,
    pub q_bwd: Executable,
    pub kv_bwd: Executable,
    /// Pre-allocated IOSurface buffers for all kernels (avoids alloc/dealloc per call).
    bufs: KernelBuffers,
}

impl CompiledKernels {
    /// Compile all 10 kernels for the given model config.
    pub fn compile(cfg: &ModelConfig) -> Self {
        let qos = NSQualityOfService::UserInteractive;

        // Forward kernels
        let sdpa_fwd = sdpa_fwd::build(cfg).compile(qos).expect("sdpaFwd compile");
        let wo_fwd = dyn_matmul::build(cfg.q_dim, cfg.dim, cfg.seq)
            .compile(qos).expect("woFwd compile");
        let ffn_fused = ffn_fused::build(cfg).compile(qos).expect("ffnFused compile");

        // Backward kernels
        let ffn_bwd_w2t = dyn_matmul::build(cfg.dim, cfg.hidden, cfg.seq)
            .compile(qos).expect("ffnBwdW2t compile");
        let ffn_bwd_w13t = dyn_matmul::build_dual(cfg.hidden, cfg.dim, cfg.seq)
            .compile(qos).expect("ffnBwdW13t compile");
        let wot_bwd = dyn_matmul::build(cfg.dim, cfg.q_dim, cfg.seq)
            .compile(qos).expect("wotBwd compile");
        let sdpa_bwd1 = sdpa_bwd::build_bwd1(cfg).compile(qos).expect("sdpaBwd1 compile");
        let sdpa_bwd2 = sdpa_bwd::build_bwd2(cfg).compile(qos).expect("sdpaBwd2 compile");
        let q_bwd = dyn_matmul::build(cfg.q_dim, cfg.dim, cfg.seq)
            .compile(qos).expect("qBwd compile");
        let kv_bwd = dyn_matmul::build_dual(cfg.kv_dim, cfg.dim, cfg.seq)
            .compile(qos).expect("kvBwd compile");

        // Pre-allocate IOSurface buffers for all kernels
        let bufs = KernelBuffers::allocate(cfg);

        Self {
            sdpa_fwd, wo_fwd, ffn_fused,
            ffn_bwd_w2t, ffn_bwd_w13t, wot_bwd,
            sdpa_bwd1, sdpa_bwd2, q_bwd, kv_bwd,
            bufs,
        }
    }
}

impl LayerWeights {
    /// Initialize to match Obj-C reference (train.m):
    /// Wq/Wk/Wv: 1/√DIM, Wo/W2: zero-init (DeepNet), W1/W3: 1/√HIDDEN.
    pub fn random(cfg: &ModelConfig) -> Self {
        let scale_qkv = 1.0 / (cfg.dim as f32).sqrt();
        let scale_ffn = 1.0 / (cfg.hidden as f32).sqrt();
        Self {
            wq: random_vec(cfg.dim * cfg.q_dim, scale_qkv),
            wk: random_vec(cfg.dim * cfg.kv_dim, scale_qkv),
            wv: random_vec(cfg.dim * cfg.kv_dim, scale_qkv),
            wo: vec![0.0; cfg.q_dim * cfg.dim],     // zero-init (DeepNet)
            w1: random_vec(cfg.dim * cfg.hidden, scale_ffn),
            w3: random_vec(cfg.dim * cfg.hidden, scale_ffn),
            w2: vec![0.0; cfg.dim * cfg.hidden],     // zero-init (DeepNet)
            gamma1: vec![1.0; cfg.dim],
            gamma2: vec![1.0; cfg.dim],
        }
    }
}

impl LayerGrads {
    pub fn zeros(cfg: &ModelConfig) -> Self {
        Self {
            dwq: vec![0.0; cfg.dim * cfg.q_dim],
            dwk: vec![0.0; cfg.dim * cfg.kv_dim],
            dwv: vec![0.0; cfg.dim * cfg.kv_dim],
            dwo: vec![0.0; cfg.q_dim * cfg.dim],
            dw1: vec![0.0; cfg.dim * cfg.hidden],
            dw3: vec![0.0; cfg.dim * cfg.hidden],
            dw2: vec![0.0; cfg.dim * cfg.hidden],
            dgamma1: vec![0.0; cfg.dim],
            dgamma2: vec![0.0; cfg.dim],
        }
    }

    pub fn zero_out(&mut self) {
        self.dwq.fill(0.0);
        self.dwk.fill(0.0);
        self.dwv.fill(0.0);
        self.dwo.fill(0.0);
        self.dw1.fill(0.0);
        self.dw3.fill(0.0);
        self.dw2.fill(0.0);
        self.dgamma1.fill(0.0);
        self.dgamma2.fill(0.0);
    }
}

/// Simple LCG pseudo-random for reproducible init (no external dep).
fn random_vec(n: usize, scale: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    let mut seed: u64 = 42 + n as u64;
    for x in v.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = ((seed >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        *x = r * scale;
    }
    v
}

// ── Helper: pack f32 data into ANE spatial layout ──

/// Stage activations into IOSurface spatial dimension.
/// `dst` is [channels * sp_width], `src` is [channels * src_width].
/// Writes src at spatial offset `sp_offset`.
/// Uses copy_from_slice for vectorized memcpy on inner dimension.
fn stage_spatial(dst: &mut [f32], channels: usize, sp_width: usize, src: &[f32], src_width: usize, sp_offset: usize) {
    for c in 0..channels {
        let d = c * sp_width + sp_offset;
        let s = c * src_width;
        dst[d..d + src_width].copy_from_slice(&src[s..s + src_width]);
    }
}

/// Read a slice of channels from ANE output buffer into a pre-allocated destination.
/// No-alloc version of the former `read_channels`.
/// Uses copy_from_slice for vectorized memcpy on inner dimension.
fn read_channels_into(src: &[f32], _total_ch: usize, seq: usize, ch_start: usize, ch_count: usize, dst: &mut [f32]) {
    for c in 0..ch_count {
        let d = c * seq;
        let s = (ch_start + c) * seq;
        dst[d..d + seq].copy_from_slice(&src[s..s + seq]);
    }
}

// ── Forward pass ──

/// Run forward pass for one transformer layer.
/// Returns (x_next, cache) where x_next is [DIM * SEQ].
pub fn forward(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    x: &[f32],
) -> (Vec<f32>, ForwardCache) {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let hidden = cfg.hidden;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    // 1. RMSNorm1 (CPU): bulk transpose → batch RMSNorm → transpose back
    let mut xnorm = vec![0.0f32; dim * seq];
    let mut rms_inv1 = vec![0.0f32; seq];
    {
        let mut x_t = vec![0.0f32; seq * dim];
        let mut xnorm_t = vec![0.0f32; seq * dim];
        vdsp::mtrans(x, seq, &mut x_t, dim, dim, seq);
        rmsnorm::forward_batch(&x_t, &weights.gamma1, &mut xnorm_t, &mut rms_inv1, dim, seq);
        vdsp::mtrans(&xnorm_t, dim, &mut xnorm, seq, seq, dim);
    }

    // 2. Stage sdpaFwd directly into IOSurface (skip scratch buffer)
    let sdpa_sp = sdpa_fwd::input_spatial_width(cfg);
    let sdpa_out_ch = sdpa_fwd::output_channels(cfg);
    {
        let mut locked = kernels.bufs.sdpa_fwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, sdpa_sp, &xnorm, seq, 0);
        stage_spatial(buf, dim, sdpa_sp, &weights.wq, q_dim, seq);
        stage_spatial(buf, dim, sdpa_sp, &weights.wk, kv_dim, seq + q_dim);
        stage_spatial(buf, dim, sdpa_sp, &weights.wv, kv_dim, seq + q_dim + kv_dim);
    }

    // 3. Run sdpaFwd (ANE)
    kernels.sdpa_fwd.run(&[&kernels.bufs.sdpa_fwd_in], &[&kernels.bufs.sdpa_fwd_out]).expect("ANE eval failed");

    // Extract: attn_out[Q_DIM,SEQ], Q_rope[Q_DIM,SEQ], K_rope[KV_DIM,SEQ], V[KV_DIM,SEQ]
    let mut attn_out = vec![0.0f32; q_dim * seq];
    let mut q_rope = vec![0.0f32; q_dim * seq];
    let mut k_rope = vec![0.0f32; kv_dim * seq];
    let mut v = vec![0.0f32; kv_dim * seq];
    {
        let locked = kernels.bufs.sdpa_fwd_out.as_f32_slice();
        read_channels_into(&locked, sdpa_out_ch, seq, 0, q_dim, &mut attn_out);
        read_channels_into(&locked, sdpa_out_ch, seq, q_dim, q_dim, &mut q_rope);
        read_channels_into(&locked, sdpa_out_ch, seq, 2 * q_dim, kv_dim, &mut k_rope);
        read_channels_into(&locked, sdpa_out_ch, seq, 2 * q_dim + kv_dim, kv_dim, &mut v);
    }

    // 4. Stage woFwd directly into IOSurface
    let wo_sp = dyn_matmul::spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.wo_fwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, q_dim, wo_sp, &attn_out, seq, 0);
        stage_spatial(buf, q_dim, wo_sp, &weights.wo, dim, seq);
    }

    // 5. Run woFwd (ANE)
    kernels.wo_fwd.run(&[&kernels.bufs.wo_fwd_in], &[&kernels.bufs.wo_fwd_out]).expect("ANE eval failed");

    // Read o_out directly from output IOSurface
    let mut o_out = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.wo_fwd_out.as_f32_slice();
        o_out.copy_from_slice(&locked[..dim * seq]);
    }

    // 6. Residual + RMSNorm2 (CPU)
    // x2 = x + alpha * o_out  (vDSP: vsma = o_out * alpha + x)
    let mut x2 = vec![0.0f32; dim * seq];
    vdsp::vsma(&o_out, alpha, x, &mut x2);
    let mut x2norm = vec![0.0f32; dim * seq];
    let mut rms_inv2 = vec![0.0f32; seq];
    {
        let mut x2_t = vec![0.0f32; seq * dim];
        let mut x2norm_t = vec![0.0f32; seq * dim];
        vdsp::mtrans(&x2, seq, &mut x2_t, dim, dim, seq);
        rmsnorm::forward_batch(&x2_t, &weights.gamma2, &mut x2norm_t, &mut rms_inv2, dim, seq);
        vdsp::mtrans(&x2norm_t, dim, &mut x2norm, seq, seq, dim);
    }

    // 7. Stage ffnFused directly into IOSurface
    let ffn_sp = ffn_fused::input_spatial_width(cfg);
    let ffn_out_ch = ffn_fused::output_channels(cfg);
    {
        let mut locked = kernels.bufs.ffn_fused_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, ffn_sp, &x2norm, seq, 0);
        stage_spatial(buf, dim, ffn_sp, &x2, seq, seq);
        stage_spatial(buf, dim, ffn_sp, &weights.w1, hidden, 2 * seq);
        stage_spatial(buf, dim, ffn_sp, &weights.w3, hidden, 2 * seq + hidden);
        stage_spatial(buf, dim, ffn_sp, &weights.w2, hidden, 2 * seq + 2 * hidden);
    }

    // 8. Run ffnFused (ANE)
    kernels.ffn_fused.run(&[&kernels.bufs.ffn_fused_in], &[&kernels.bufs.ffn_fused_out]).expect("ANE eval failed");

    // Extract: x_next[DIM,SEQ], h1[HIDDEN,SEQ], h3[HIDDEN,SEQ], gate[HIDDEN,SEQ]
    let mut x_next = vec![0.0f32; dim * seq];
    let mut h1 = vec![0.0f32; hidden * seq];
    let mut h3 = vec![0.0f32; hidden * seq];
    let mut gate = vec![0.0f32; hidden * seq];
    {
        let locked = kernels.bufs.ffn_fused_out.as_f32_slice();
        read_channels_into(&locked, ffn_out_ch, seq, 0, dim, &mut x_next);
        read_channels_into(&locked, ffn_out_ch, seq, dim, hidden, &mut h1);
        read_channels_into(&locked, ffn_out_ch, seq, dim + hidden, hidden, &mut h3);
        read_channels_into(&locked, ffn_out_ch, seq, dim + 2 * hidden, hidden, &mut gate);
    }

    let cache = ForwardCache {
        x: x.to_vec(), xnorm, rms_inv1, q_rope, k_rope, v, attn_out, o_out,
        x2, x2norm, rms_inv2, h1, h3, gate,
    };

    (x_next, cache)
}

/// Timing breakdown for forward pass.
#[derive(Debug, Clone)]
pub struct ForwardTimings {
    pub rmsnorm1_ms: f32,
    pub stage_sdpa_ms: f32,
    pub ane_sdpa_ms: f32,
    pub read_sdpa_ms: f32,
    pub stage_wo_ms: f32,
    pub ane_wo_ms: f32,
    pub read_wo_ms: f32,
    pub residual_rmsnorm2_ms: f32,
    pub stage_ffn_ms: f32,
    pub ane_ffn_ms: f32,
    pub read_ffn_ms: f32,
    pub total_ms: f32,
}

impl ForwardTimings {
    pub fn print(&self) {
        println!("  {:<30} {:>6.2}ms", "RMSNorm1 (CPU)", self.rmsnorm1_ms);
        println!("  {:<30} {:>6.2}ms", "stage sdpaFwd IOSurf", self.stage_sdpa_ms);
        println!("  {:<30} {:>6.2}ms", "ANE sdpaFwd", self.ane_sdpa_ms);
        println!("  {:<30} {:>6.2}ms", "read sdpaFwd output", self.read_sdpa_ms);
        println!("  {:<30} {:>6.2}ms", "stage woFwd IOSurf", self.stage_wo_ms);
        println!("  {:<30} {:>6.2}ms", "ANE woFwd", self.ane_wo_ms);
        println!("  {:<30} {:>6.2}ms", "read woFwd output", self.read_wo_ms);
        println!("  {:<30} {:>6.2}ms", "residual + RMSNorm2 (CPU)", self.residual_rmsnorm2_ms);
        println!("  {:<30} {:>6.2}ms", "stage ffnFused IOSurf", self.stage_ffn_ms);
        println!("  {:<30} {:>6.2}ms", "ANE ffnFused", self.ane_ffn_ms);
        println!("  {:<30} {:>6.2}ms", "read ffnFused output", self.read_ffn_ms);
        println!("  {:<30} {:>6.2}ms", "TOTAL", self.total_ms);
    }
}

/// Forward pass with per-operation timing (same output as `forward`).
pub fn forward_timed(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    x: &[f32],
) -> (Vec<f32>, ForwardCache, ForwardTimings) {
    let t_total = Instant::now();
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let hidden = cfg.hidden;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    // 1. RMSNorm1 (bulk transpose)
    let t = Instant::now();
    let mut xnorm = vec![0.0f32; dim * seq];
    let mut rms_inv1 = vec![0.0f32; seq];
    {
        let mut x_t = vec![0.0f32; seq * dim];
        let mut xnorm_t = vec![0.0f32; seq * dim];
        vdsp::mtrans(x, seq, &mut x_t, dim, dim, seq);
        rmsnorm::forward_batch(&x_t, &weights.gamma1, &mut xnorm_t, &mut rms_inv1, dim, seq);
        vdsp::mtrans(&xnorm_t, dim, &mut xnorm, seq, seq, dim);
    }
    let rmsnorm1_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 2. Stage sdpaFwd
    let t = Instant::now();
    let sdpa_sp = sdpa_fwd::input_spatial_width(cfg);
    let sdpa_out_ch = sdpa_fwd::output_channels(cfg);
    {
        let mut locked = kernels.bufs.sdpa_fwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, sdpa_sp, &xnorm, seq, 0);
        stage_spatial(buf, dim, sdpa_sp, &weights.wq, q_dim, seq);
        stage_spatial(buf, dim, sdpa_sp, &weights.wk, kv_dim, seq + q_dim);
        stage_spatial(buf, dim, sdpa_sp, &weights.wv, kv_dim, seq + q_dim + kv_dim);
    }
    let stage_sdpa_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 3. ANE sdpaFwd
    let t = Instant::now();
    kernels.sdpa_fwd.run(&[&kernels.bufs.sdpa_fwd_in], &[&kernels.bufs.sdpa_fwd_out]).expect("ANE eval failed");
    let ane_sdpa_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 4. Read output
    let t = Instant::now();
    let mut attn_out = vec![0.0f32; q_dim * seq];
    let mut q_rope = vec![0.0f32; q_dim * seq];
    let mut k_rope = vec![0.0f32; kv_dim * seq];
    let mut v = vec![0.0f32; kv_dim * seq];
    {
        let locked = kernels.bufs.sdpa_fwd_out.as_f32_slice();
        read_channels_into(&locked, sdpa_out_ch, seq, 0, q_dim, &mut attn_out);
        read_channels_into(&locked, sdpa_out_ch, seq, q_dim, q_dim, &mut q_rope);
        read_channels_into(&locked, sdpa_out_ch, seq, 2 * q_dim, kv_dim, &mut k_rope);
        read_channels_into(&locked, sdpa_out_ch, seq, 2 * q_dim + kv_dim, kv_dim, &mut v);
    }
    let read_sdpa_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 5. Stage woFwd
    let t = Instant::now();
    let wo_sp = dyn_matmul::spatial_width(seq, dim);
    {
        let mut locked = kernels.bufs.wo_fwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, q_dim, wo_sp, &attn_out, seq, 0);
        stage_spatial(buf, q_dim, wo_sp, &weights.wo, dim, seq);
    }
    let stage_wo_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 6. ANE woFwd
    let t = Instant::now();
    kernels.wo_fwd.run(&[&kernels.bufs.wo_fwd_in], &[&kernels.bufs.wo_fwd_out]).expect("ANE eval failed");
    let ane_wo_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 7. Read woFwd output
    let t = Instant::now();
    let mut o_out = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.wo_fwd_out.as_f32_slice();
        o_out.copy_from_slice(&locked[..dim * seq]);
    }
    let read_wo_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 8. Residual + RMSNorm2 (bulk transpose)
    let t = Instant::now();
    let mut x2 = vec![0.0f32; dim * seq];
    vdsp::vsma(&o_out, alpha, x, &mut x2);
    let mut x2norm = vec![0.0f32; dim * seq];
    let mut rms_inv2 = vec![0.0f32; seq];
    {
        let mut x2_t = vec![0.0f32; seq * dim];
        let mut x2norm_t = vec![0.0f32; seq * dim];
        vdsp::mtrans(&x2, seq, &mut x2_t, dim, dim, seq);
        rmsnorm::forward_batch(&x2_t, &weights.gamma2, &mut x2norm_t, &mut rms_inv2, dim, seq);
        vdsp::mtrans(&x2norm_t, dim, &mut x2norm, seq, seq, dim);
    }
    let residual_rmsnorm2_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 9. Stage ffnFused
    let t = Instant::now();
    let ffn_sp = ffn_fused::input_spatial_width(cfg);
    let ffn_out_ch = ffn_fused::output_channels(cfg);
    {
        let mut locked = kernels.bufs.ffn_fused_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, ffn_sp, &x2norm, seq, 0);
        stage_spatial(buf, dim, ffn_sp, &x2, seq, seq);
        stage_spatial(buf, dim, ffn_sp, &weights.w1, hidden, 2 * seq);
        stage_spatial(buf, dim, ffn_sp, &weights.w3, hidden, 2 * seq + hidden);
        stage_spatial(buf, dim, ffn_sp, &weights.w2, hidden, 2 * seq + 2 * hidden);
    }
    let stage_ffn_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 10. ANE ffnFused
    let t = Instant::now();
    kernels.ffn_fused.run(&[&kernels.bufs.ffn_fused_in], &[&kernels.bufs.ffn_fused_out]).expect("ANE eval failed");
    let ane_ffn_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 11. Read ffnFused output
    let t = Instant::now();
    let mut x_next = vec![0.0f32; dim * seq];
    let mut h1 = vec![0.0f32; hidden * seq];
    let mut h3 = vec![0.0f32; hidden * seq];
    let mut gate = vec![0.0f32; hidden * seq];
    {
        let locked = kernels.bufs.ffn_fused_out.as_f32_slice();
        read_channels_into(&locked, ffn_out_ch, seq, 0, dim, &mut x_next);
        read_channels_into(&locked, ffn_out_ch, seq, dim, hidden, &mut h1);
        read_channels_into(&locked, ffn_out_ch, seq, dim + hidden, hidden, &mut h3);
        read_channels_into(&locked, ffn_out_ch, seq, dim + 2 * hidden, hidden, &mut gate);
    }
    let read_ffn_ms = t.elapsed().as_secs_f32() * 1000.0;

    let total_ms = t_total.elapsed().as_secs_f32() * 1000.0;

    let cache = ForwardCache {
        x: x.to_vec(), xnorm, rms_inv1, q_rope, k_rope, v, attn_out, o_out,
        x2, x2norm, rms_inv2, h1, h3, gate,
    };

    let timings = ForwardTimings {
        rmsnorm1_ms, stage_sdpa_ms, ane_sdpa_ms, read_sdpa_ms,
        stage_wo_ms, ane_wo_ms, read_wo_ms, residual_rmsnorm2_ms,
        stage_ffn_ms, ane_ffn_ms, read_ffn_ms, total_ms,
    };

    (x_next, cache, timings)
}

/// Timing breakdown for backward pass.
#[derive(Debug, Clone)]
pub struct BackwardTimings {
    pub scale_dy_ms: f32,
    pub stage_run_ffn_bwd_w2t_ms: f32,
    pub silu_deriv_ms: f32,
    pub stage_ffn_bwd_w13t_ms: f32,
    pub async_ffn_bwd_w13t_plus_dw_ms: f32,
    pub rmsnorm2_bwd_ms: f32,
    pub stage_run_wot_bwd_ms: f32,
    pub stage_sdpa_bwd1_ms: f32,
    pub async_sdpa_bwd1_plus_dwo_ms: f32,
    pub read_sdpa_bwd1_ms: f32,
    pub stage_run_sdpa_bwd2_ms: f32,
    pub rope_bwd_ms: f32,
    pub stage_q_bwd_ms: f32,
    pub async_q_bwd_plus_dw_ms: f32,
    pub stage_run_kv_bwd_ms: f32,
    pub rmsnorm1_bwd_ms: f32,
    pub merge_dx_ms: f32,
    pub total_ms: f32,
}

impl BackwardTimings {
    pub fn print(&self) {
        println!("  {:<35} {:>6.2}ms", "scale dy (vDSP)", self.scale_dy_ms);
        println!("  {:<35} {:>6.2}ms", "stage+run ffnBwdW2t (ANE)", self.stage_run_ffn_bwd_w2t_ms);
        println!("  {:<35} {:>6.2}ms", "SiLU derivative (CPU)", self.silu_deriv_ms);
        println!("  {:<35} {:>6.2}ms", "stage ffnBwdW13t", self.stage_ffn_bwd_w13t_ms);
        println!("  {:<35} {:>6.2}ms", "async ffnBwdW13t + dW2+dW1+dW3", self.async_ffn_bwd_w13t_plus_dw_ms);
        println!("  {:<35} {:>6.2}ms", "RMSNorm2 backward (CPU)", self.rmsnorm2_bwd_ms);
        println!("  {:<35} {:>6.2}ms", "stage+run wotBwd (ANE)", self.stage_run_wot_bwd_ms);
        println!("  {:<35} {:>6.2}ms", "stage sdpaBwd1", self.stage_sdpa_bwd1_ms);
        println!("  {:<35} {:>6.2}ms", "async sdpaBwd1 + dWo", self.async_sdpa_bwd1_plus_dwo_ms);
        println!("  {:<35} {:>6.2}ms", "read sdpaBwd1 output", self.read_sdpa_bwd1_ms);
        println!("  {:<35} {:>6.2}ms", "stage+run sdpaBwd2 (ANE)", self.stage_run_sdpa_bwd2_ms);
        println!("  {:<35} {:>6.2}ms", "RoPE backward (CPU)", self.rope_bwd_ms);
        println!("  {:<35} {:>6.2}ms", "stage qBwd", self.stage_q_bwd_ms);
        println!("  {:<35} {:>6.2}ms", "async qBwd + dWq+dWk+dWv", self.async_q_bwd_plus_dw_ms);
        println!("  {:<35} {:>6.2}ms", "stage+run kvBwd (ANE)", self.stage_run_kv_bwd_ms);
        println!("  {:<35} {:>6.2}ms", "RMSNorm1 backward (CPU)", self.rmsnorm1_bwd_ms);
        println!("  {:<35} {:>6.2}ms", "merge dx (vDSP)", self.merge_dx_ms);
        println!("  {:<35} {:>6.2}ms", "TOTAL", self.total_ms);
    }
}

/// Backward pass with per-operation timing (same output as `backward`).
pub fn backward_timed(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    cache: &ForwardCache,
    dy: &[f32],
    grads: &mut LayerGrads,
) -> (Vec<f32>, BackwardTimings) {
    let t_total = Instant::now();
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let hidden = cfg.hidden;
    let heads = cfg.heads;
    let hd = cfg.hd;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    // 1. Scale dy
    let t = Instant::now();
    let mut dffn = vec![0.0f32; dim * seq];
    vdsp::vsmul(dy, alpha, &mut dffn);
    let scale_dy_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 2. ffnBwdW2t
    let t = Instant::now();
    let w2t_sp = dyn_matmul::spatial_width(seq, hidden);
    {
        let mut locked = kernels.bufs.ffn_bwd_w2t_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, w2t_sp, &dffn, seq, 0);
        stage_spatial(buf, dim, w2t_sp, &weights.w2, hidden, seq);
    }
    kernels.ffn_bwd_w2t.run(&[&kernels.bufs.ffn_bwd_w2t_in], &[&kernels.bufs.ffn_bwd_w2t_out]).expect("ANE eval failed");
    let mut dsilu_raw = vec![0.0f32; hidden * seq];
    {
        let locked = kernels.bufs.ffn_bwd_w2t_out.as_f32_slice();
        dsilu_raw.copy_from_slice(&locked[..hidden * seq]);
    }
    let stage_run_ffn_bwd_w2t_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 3. SiLU derivative (vvexpf for batch exp, fused scalar loop for cache locality)
    let t = Instant::now();
    let n = hidden * seq;
    let mut dh1 = vec![0.0f32; n];
    let mut dh3 = vec![0.0f32; n];
    {
        let mut neg_h1 = vec![0.0f32; n];
        let mut exp_neg = vec![0.0f32; n];
        vdsp::vsmul(&cache.h1, -1.0, &mut neg_h1);
        vdsp::expf(&neg_h1, &mut exp_neg);
        for i in 0..n {
            let sig = 1.0 / (1.0 + exp_neg[i]);
            let silu_val = cache.h1[i] * sig;
            let silu_deriv = sig * (1.0 + cache.h1[i] * (1.0 - sig));
            dh3[i] = dsilu_raw[i] * silu_val;
            dh1[i] = dsilu_raw[i] * cache.h3[i] * silu_deriv;
        }
    }
    let silu_deriv_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 4. Stage ffnBwdW13t (mtrans + stage_spatial)
    let t = Instant::now();
    let w13t_sp = dyn_matmul::dual_spatial_width(seq, dim);
    {
        let mut w1t = vec![0.0f32; hidden * dim];
        let mut w3t = vec![0.0f32; hidden * dim];
        vdsp::mtrans(&weights.w1, hidden, &mut w1t, dim, dim, hidden);
        vdsp::mtrans(&weights.w3, hidden, &mut w3t, dim, dim, hidden);
        let mut locked = kernels.bufs.ffn_bwd_w13t_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, hidden, w13t_sp, &dh1, seq, 0);
        stage_spatial(buf, hidden, w13t_sp, &dh3, seq, seq);
        stage_spatial(buf, hidden, w13t_sp, &w1t, dim, 2 * seq);
        stage_spatial(buf, hidden, w13t_sp, &w3t, dim, 2 * seq + dim);
    }
    let stage_ffn_bwd_w13t_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 5. ASYNC: ANE ffnBwdW13t || CPU dW
    let t = Instant::now();
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels.ffn_bwd_w13t.run(
                &[&kernels.bufs.ffn_bwd_w13t_in],
                &[&kernels.bufs.ffn_bwd_w13t_out],
            ).expect("ANE eval failed");
        });
        accumulate_dw(&dffn, dim, &cache.gate, hidden, seq, &mut grads.dw2);
        accumulate_dw(&cache.x2norm, dim, &dh1, hidden, seq, &mut grads.dw1);
        accumulate_dw(&cache.x2norm, dim, &dh3, hidden, seq, &mut grads.dw3);
        ane_handle.join().expect("ANE thread panicked");
    });
    let mut dx_ffn = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.ffn_bwd_w13t_out.as_f32_slice();
        dx_ffn.copy_from_slice(&locked[..dim * seq]);
    }
    let async_ffn_bwd_w13t_plus_dw_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 6. RMSNorm2 backward (bulk transpose)
    let t = Instant::now();
    let mut dx2 = vec![0.0f32; dim * seq];
    {
        let mut dy_t = vec![0.0f32; seq * dim];
        let mut x2_t = vec![0.0f32; seq * dim];
        let mut dx2_t = vec![0.0f32; seq * dim];
        vdsp::mtrans(&dx_ffn, seq, &mut dy_t, dim, dim, seq);
        vdsp::mtrans(&cache.x2, seq, &mut x2_t, dim, dim, seq);
        rmsnorm::backward_batch(&dy_t, &x2_t, &weights.gamma2, &cache.rms_inv2, &mut dx2_t, &mut grads.dgamma2, dim, seq);
        vdsp::mtrans(&dx2_t, dim, &mut dx2, seq, seq, dim);
    }
    let mut dx2_tmp = vec![0.0f32; dim * seq];
    vdsp::vadd(&dx2, dy, &mut dx2_tmp);
    dx2.copy_from_slice(&dx2_tmp);
    let rmsnorm2_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 7. wotBwd (mtrans Wo)
    let t = Instant::now();
    let mut dx2_scaled = vec![0.0f32; dim * seq];
    vdsp::vsmul(&dx2, alpha, &mut dx2_scaled);
    let wot_sp = dyn_matmul::spatial_width(seq, q_dim);
    {
        let mut wot = vec![0.0f32; dim * q_dim];
        vdsp::mtrans(&weights.wo, dim, &mut wot, q_dim, q_dim, dim);
        let mut locked = kernels.bufs.wot_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, wot_sp, &dx2_scaled, seq, 0);
        stage_spatial(buf, dim, wot_sp, &wot, q_dim, seq);
    }
    kernels.wot_bwd.run(&[&kernels.bufs.wot_bwd_in], &[&kernels.bufs.wot_bwd_out]).expect("ANE eval failed");
    let mut da = vec![0.0f32; q_dim * seq];
    {
        let locked = kernels.bufs.wot_bwd_out.as_f32_slice();
        da.copy_from_slice(&locked[..q_dim * seq]);
    }
    let stage_run_wot_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 8. Stage sdpaBwd1
    let t = Instant::now();
    let bwd1_in_ch = sdpa_bwd::bwd1_input_channels(cfg);
    let bwd1_out_ch = sdpa_bwd::bwd1_output_channels(cfg);
    {
        let mut locked = kernels.bufs.sdpa_bwd1_in.as_f32_slice_mut();
        let buf = &mut *locked;
        pack_channels(buf, bwd1_in_ch, seq, &cache.q_rope, q_dim, 0);
        pack_channels(buf, bwd1_in_ch, seq, &cache.k_rope, q_dim, q_dim);
        pack_channels(buf, bwd1_in_ch, seq, &cache.v, q_dim, 2 * q_dim);
        pack_channels(buf, bwd1_in_ch, seq, &da, q_dim, 3 * q_dim);
    }
    let stage_sdpa_bwd1_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 9. ASYNC: ANE sdpaBwd1 || CPU dWo
    let t = Instant::now();
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels.sdpa_bwd1.run(
                &[&kernels.bufs.sdpa_bwd1_in],
                &[&kernels.bufs.sdpa_bwd1_out],
            ).expect("ANE eval failed");
        });
        accumulate_dw(&cache.attn_out, q_dim, &dx2_scaled, dim, seq, &mut grads.dwo);
        ane_handle.join().expect("ANE thread panicked");
    });
    let async_sdpa_bwd1_plus_dwo_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 10. Read sdpaBwd1
    let t = Instant::now();
    let score_ch = heads * seq;
    let mut dv_full = vec![0.0f32; q_dim * seq];
    let mut probs_flat = vec![0.0f32; score_ch * seq];
    let mut dp_flat = vec![0.0f32; score_ch * seq];
    {
        let locked = kernels.bufs.sdpa_bwd1_out.as_f32_slice();
        read_channels_into(&locked, bwd1_out_ch, seq, 0, q_dim, &mut dv_full);
        read_channels_into(&locked, bwd1_out_ch, seq, q_dim, score_ch, &mut probs_flat);
        read_channels_into(&locked, bwd1_out_ch, seq, q_dim + score_ch, score_ch, &mut dp_flat);
    }
    let read_sdpa_bwd1_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 11. sdpaBwd2
    let t = Instant::now();
    let bwd2_in_ch = sdpa_bwd::bwd2_input_channels(cfg);
    let bwd2_out_ch = sdpa_bwd::bwd2_output_channels(cfg);
    {
        let mut locked = kernels.bufs.sdpa_bwd2_in.as_f32_slice_mut();
        let buf = &mut *locked;
        pack_channels(buf, bwd2_in_ch, seq, &probs_flat, score_ch, 0);
        pack_channels(buf, bwd2_in_ch, seq, &dp_flat, score_ch, score_ch);
        pack_channels(buf, bwd2_in_ch, seq, &cache.q_rope, q_dim, 2 * score_ch);
        pack_channels(buf, bwd2_in_ch, seq, &cache.k_rope, q_dim, 2 * score_ch + q_dim);
    }
    kernels.sdpa_bwd2.run(&[&kernels.bufs.sdpa_bwd2_in], &[&kernels.bufs.sdpa_bwd2_out]).expect("ANE eval failed");
    let mut dq = vec![0.0f32; q_dim * seq];
    let mut dk = vec![0.0f32; q_dim * seq];
    {
        let locked = kernels.bufs.sdpa_bwd2_out.as_f32_slice();
        read_channels_into(&locked, bwd2_out_ch, seq, 0, q_dim, &mut dq);
        read_channels_into(&locked, bwd2_out_ch, seq, q_dim, q_dim, &mut dk);
    }
    let dv = dv_full;
    let stage_run_sdpa_bwd2_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 12. RoPE backward
    let t = Instant::now();
    rope_backward_inplace(&mut dq, heads, hd, seq);
    rope_backward_inplace(&mut dk, heads, hd, seq);
    let rope_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 13. Stage qBwd (mtrans Wq)
    let t = Instant::now();
    let q_bwd_sp = dyn_matmul::spatial_width(seq, dim);
    {
        let mut wqt = vec![0.0f32; q_dim * dim];
        vdsp::mtrans(&weights.wq, q_dim, &mut wqt, dim, dim, q_dim);
        let mut locked = kernels.bufs.q_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, q_dim, q_bwd_sp, &dq, seq, 0);
        stage_spatial(buf, q_dim, q_bwd_sp, &wqt, dim, seq);
    }
    let stage_q_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 14. ASYNC: ANE qBwd || CPU dWq+dWk+dWv
    let t = Instant::now();
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels.q_bwd.run(
                &[&kernels.bufs.q_bwd_in],
                &[&kernels.bufs.q_bwd_out],
            ).expect("ANE eval failed");
        });
        accumulate_dw(&cache.xnorm, dim, &dq, q_dim, seq, &mut grads.dwq);
        accumulate_dw(&cache.xnorm, dim, &dk, kv_dim, seq, &mut grads.dwk);
        accumulate_dw(&cache.xnorm, dim, &dv, kv_dim, seq, &mut grads.dwv);
        ane_handle.join().expect("ANE thread panicked");
    });
    let mut dx_attn = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.q_bwd_out.as_f32_slice();
        dx_attn.copy_from_slice(&locked[..dim * seq]);
    }
    let async_q_bwd_plus_dw_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 15. kvBwd (mtrans Wk/Wv)
    let t = Instant::now();
    let kv_bwd_sp = dyn_matmul::dual_spatial_width(seq, dim);
    {
        let mut wkt = vec![0.0f32; kv_dim * dim];
        let mut wvt = vec![0.0f32; kv_dim * dim];
        vdsp::mtrans(&weights.wk, kv_dim, &mut wkt, dim, dim, kv_dim);
        vdsp::mtrans(&weights.wv, kv_dim, &mut wvt, dim, dim, kv_dim);
        let mut locked = kernels.bufs.kv_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, kv_dim, kv_bwd_sp, &dk, seq, 0);
        stage_spatial(buf, kv_dim, kv_bwd_sp, &dv, seq, seq);
        stage_spatial(buf, kv_dim, kv_bwd_sp, &wkt, dim, seq + seq);
        stage_spatial(buf, kv_dim, kv_bwd_sp, &wvt, dim, 2 * seq + dim);
    }
    kernels.kv_bwd.run(&[&kernels.bufs.kv_bwd_in], &[&kernels.bufs.kv_bwd_out]).expect("ANE eval failed");
    let mut dx_kv = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.kv_bwd_out.as_f32_slice();
        dx_kv.copy_from_slice(&locked[..dim * seq]);
    }
    let stage_run_kv_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 16. Merge + RMSNorm1 backward
    let t = Instant::now();
    let mut dx_merged = vec![0.0f32; dim * seq];
    vdsp::vadd(&dx_attn, &dx_kv, &mut dx_merged);
    let merge_dx_ms = t.elapsed().as_secs_f32() * 1000.0;

    let t = Instant::now();
    let mut dx_rms1 = vec![0.0f32; dim * seq];
    {
        let mut dy_t = vec![0.0f32; seq * dim];
        let mut x_t = vec![0.0f32; seq * dim];
        let mut dx_t = vec![0.0f32; seq * dim];
        vdsp::mtrans(&dx_merged, seq, &mut dy_t, dim, dim, seq);
        vdsp::mtrans(&cache.x, seq, &mut x_t, dim, dim, seq);
        rmsnorm::backward_batch(&dy_t, &x_t, &weights.gamma1, &cache.rms_inv1, &mut dx_t, &mut grads.dgamma1, dim, seq);
        vdsp::mtrans(&dx_t, dim, &mut dx_rms1, seq, seq, dim);
    }
    let rmsnorm1_bwd_ms = t.elapsed().as_secs_f32() * 1000.0;

    // 17. Final dx
    let mut dx = vec![0.0f32; dim * seq];
    vdsp::vadd(&dx_rms1, &dx2, &mut dx);

    let total_ms = t_total.elapsed().as_secs_f32() * 1000.0;

    let timings = BackwardTimings {
        scale_dy_ms, stage_run_ffn_bwd_w2t_ms, silu_deriv_ms,
        stage_ffn_bwd_w13t_ms, async_ffn_bwd_w13t_plus_dw_ms,
        rmsnorm2_bwd_ms, stage_run_wot_bwd_ms,
        stage_sdpa_bwd1_ms, async_sdpa_bwd1_plus_dwo_ms, read_sdpa_bwd1_ms,
        stage_run_sdpa_bwd2_ms, rope_bwd_ms,
        stage_q_bwd_ms, async_q_bwd_plus_dw_ms,
        stage_run_kv_bwd_ms, rmsnorm1_bwd_ms, merge_dx_ms, total_ms,
    };

    (dx, timings)
}

// ── Backward pass ──

/// Run backward pass for one transformer layer.
/// `dy` is gradient of loss w.r.t. layer output [DIM * SEQ].
/// Returns `dx` (gradient w.r.t. layer input) and fills `grads`.
pub fn backward(
    cfg: &ModelConfig,
    kernels: &CompiledKernels,
    weights: &LayerWeights,
    cache: &ForwardCache,
    dy: &[f32],
    grads: &mut LayerGrads,
) -> Vec<f32> {
    let dim = cfg.dim;
    let seq = cfg.seq;
    let q_dim = cfg.q_dim;
    let kv_dim = cfg.kv_dim;
    let hidden = cfg.hidden;
    let heads = cfg.heads;
    let hd = cfg.hd;
    let alpha = 1.0 / (2.0 * cfg.nlayers as f32).sqrt();

    // ── 1. Scale dy for FFN residual (vDSP vectorized) ──
    let mut dffn = vec![0.0f32; dim * seq];
    vdsp::vsmul(dy, alpha, &mut dffn);

    // ── 2. ffnBwdW2t(ANE): dffn @ W2 → dsilu_raw [HIDDEN, SEQ] ──
    let w2t_sp = dyn_matmul::spatial_width(seq, hidden);
    {
        let mut locked = kernels.bufs.ffn_bwd_w2t_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, w2t_sp, &dffn, seq, 0);
        stage_spatial(buf, dim, w2t_sp, &weights.w2, hidden, seq);
    }
    kernels.ffn_bwd_w2t.run(&[&kernels.bufs.ffn_bwd_w2t_in], &[&kernels.bufs.ffn_bwd_w2t_out]).expect("ANE eval failed");

    // Read dsilu_raw directly from output IOSurface
    let mut dsilu_raw = vec![0.0f32; hidden * seq];
    {
        let locked = kernels.bufs.ffn_bwd_w2t_out.as_f32_slice();
        dsilu_raw.copy_from_slice(&locked[..hidden * seq]);
    }

    // ── 3. SiLU derivative (vvexpf + fused scalar loop) ──
    let n = hidden * seq;
    let mut dh1 = vec![0.0f32; n];
    let mut dh3 = vec![0.0f32; n];
    {
        let mut neg_h1 = vec![0.0f32; n];
        let mut exp_neg = vec![0.0f32; n];
        vdsp::vsmul(&cache.h1, -1.0, &mut neg_h1);
        vdsp::expf(&neg_h1, &mut exp_neg);
        for i in 0..n {
            let sig = 1.0 / (1.0 + exp_neg[i]);
            let silu_val = cache.h1[i] * sig;
            let silu_deriv = sig * (1.0 + cache.h1[i] * (1.0 - sig));
            dh3[i] = dsilu_raw[i] * silu_val;
            dh1[i] = dsilu_raw[i] * cache.h3[i] * silu_deriv;
        }
    }

    // ── 4. Stage ffnBwdW13t: mtrans weights, then stage_spatial ──
    let w13t_sp = dyn_matmul::dual_spatial_width(seq, dim);
    {
        // Pre-transpose W1[dim,hidden] → W1t[hidden,dim] and W3 likewise
        let mut w1t = vec![0.0f32; hidden * dim];
        let mut w3t = vec![0.0f32; hidden * dim];
        vdsp::mtrans(&weights.w1, hidden, &mut w1t, dim, dim, hidden);
        vdsp::mtrans(&weights.w3, hidden, &mut w3t, dim, dim, hidden);

        let mut locked = kernels.bufs.ffn_bwd_w13t_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, hidden, w13t_sp, &dh1, seq, 0);
        stage_spatial(buf, hidden, w13t_sp, &dh3, seq, seq);
        stage_spatial(buf, hidden, w13t_sp, &w1t, dim, 2 * seq);
        stage_spatial(buf, hidden, w13t_sp, &w3t, dim, 2 * seq + dim);
    }

    // ── 5. ASYNC: ANE ffnBwdW13t || CPU dW2+dW1+dW3 accumulation ──
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels.ffn_bwd_w13t.run(
                &[&kernels.bufs.ffn_bwd_w13t_in],
                &[&kernels.bufs.ffn_bwd_w13t_out],
            ).expect("ANE eval failed");
        });
        // CPU: dW accumulation while ANE runs (no data race — different memory)
        accumulate_dw(&dffn, dim, &cache.gate, hidden, seq, &mut grads.dw2);
        accumulate_dw(&cache.x2norm, dim, &dh1, hidden, seq, &mut grads.dw1);
        accumulate_dw(&cache.x2norm, dim, &dh3, hidden, seq, &mut grads.dw3);
        ane_handle.join().expect("ANE thread panicked");
    });

    // Read dx_ffn directly from output IOSurface
    let mut dx_ffn = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.ffn_bwd_w13t_out.as_f32_slice();
        dx_ffn.copy_from_slice(&locked[..dim * seq]);
    }

    // ── 6. RMSNorm2 backward (CPU): bulk transpose → batch backward → transpose back ──
    let mut dx2 = vec![0.0f32; dim * seq];
    {
        let mut dy_t = vec![0.0f32; seq * dim];
        let mut x2_t = vec![0.0f32; seq * dim];
        let mut dx2_t = vec![0.0f32; seq * dim];
        vdsp::mtrans(&dx_ffn, seq, &mut dy_t, dim, dim, seq);
        vdsp::mtrans(&cache.x2, seq, &mut x2_t, dim, dim, seq);
        rmsnorm::backward_batch(&dy_t, &x2_t, &weights.gamma2, &cache.rms_inv2, &mut dx2_t, &mut grads.dgamma2, dim, seq);
        vdsp::mtrans(&dx2_t, dim, &mut dx2, seq, seq, dim);
    }
    // Add residual gradient: dx2 += dy (FFN residual branch, vDSP vectorized)
    let mut dx2_tmp = vec![0.0f32; dim * seq];
    vdsp::vadd(&dx2, dy, &mut dx2_tmp);
    dx2.copy_from_slice(&dx2_tmp);

    // ── 7. Scale dx2 for attention residual (vDSP vectorized) ──
    let mut dx2_scaled = vec![0.0f32; dim * seq];
    vdsp::vsmul(&dx2, alpha, &mut dx2_scaled);

    // ── 8. wotBwd(ANE): dx2_scaled @ Wo → da [Q_DIM, SEQ] ──
    let wot_sp = dyn_matmul::spatial_width(seq, q_dim);
    {
        let mut wot = vec![0.0f32; dim * q_dim];
        vdsp::mtrans(&weights.wo, dim, &mut wot, q_dim, q_dim, dim);
        let mut locked = kernels.bufs.wot_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, dim, wot_sp, &dx2_scaled, seq, 0);
        stage_spatial(buf, dim, wot_sp, &wot, q_dim, seq);
    }
    kernels.wot_bwd.run(&[&kernels.bufs.wot_bwd_in], &[&kernels.bufs.wot_bwd_out]).expect("ANE eval failed");

    // Read da directly from output IOSurface
    let mut da = vec![0.0f32; q_dim * seq];
    {
        let locked = kernels.bufs.wot_bwd_out.as_f32_slice();
        da.copy_from_slice(&locked[..q_dim * seq]);
    }

    // ── 9. Stage sdpaBwd1 directly into IOSurface, then overlap dWo with ANE ──
    let bwd1_in_ch = sdpa_bwd::bwd1_input_channels(cfg);
    let bwd1_out_ch = sdpa_bwd::bwd1_output_channels(cfg);
    {
        let mut locked = kernels.bufs.sdpa_bwd1_in.as_f32_slice_mut();
        let buf = &mut *locked;
        pack_channels(buf, bwd1_in_ch, seq, &cache.q_rope, q_dim, 0);
        pack_channels(buf, bwd1_in_ch, seq, &cache.k_rope, q_dim, q_dim);
        pack_channels(buf, bwd1_in_ch, seq, &cache.v, q_dim, 2 * q_dim);
        pack_channels(buf, bwd1_in_ch, seq, &da, q_dim, 3 * q_dim);
    }

    // ASYNC: ANE sdpaBwd1 || CPU dWo accumulation
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels.sdpa_bwd1.run(
                &[&kernels.bufs.sdpa_bwd1_in],
                &[&kernels.bufs.sdpa_bwd1_out],
            ).expect("ANE eval failed");
        });
        accumulate_dw(&cache.attn_out, q_dim, &dx2_scaled, dim, seq, &mut grads.dwo);
        ane_handle.join().expect("ANE thread panicked");
    });

    let score_ch = heads * seq;
    let mut dv_full = vec![0.0f32; q_dim * seq];
    let mut probs_flat = vec![0.0f32; score_ch * seq];
    let mut dp_flat = vec![0.0f32; score_ch * seq];
    {
        let locked = kernels.bufs.sdpa_bwd1_out.as_f32_slice();
        read_channels_into(&locked, bwd1_out_ch, seq, 0, q_dim, &mut dv_full);
        read_channels_into(&locked, bwd1_out_ch, seq, q_dim, score_ch, &mut probs_flat);
        read_channels_into(&locked, bwd1_out_ch, seq, q_dim + score_ch, score_ch, &mut dp_flat);
    }

    // ── 10. sdpaBwd2(ANE): probs, dp, Q_rope, K_rope → dQ, dK ──
    let bwd2_in_ch = sdpa_bwd::bwd2_input_channels(cfg);
    let bwd2_out_ch = sdpa_bwd::bwd2_output_channels(cfg);
    {
        let mut locked = kernels.bufs.sdpa_bwd2_in.as_f32_slice_mut();
        let buf = &mut *locked;
        pack_channels(buf, bwd2_in_ch, seq, &probs_flat, score_ch, 0);
        pack_channels(buf, bwd2_in_ch, seq, &dp_flat, score_ch, score_ch);
        pack_channels(buf, bwd2_in_ch, seq, &cache.q_rope, q_dim, 2 * score_ch);
        pack_channels(buf, bwd2_in_ch, seq, &cache.k_rope, q_dim, 2 * score_ch + q_dim);
    }
    kernels.sdpa_bwd2.run(&[&kernels.bufs.sdpa_bwd2_in], &[&kernels.bufs.sdpa_bwd2_out]).expect("ANE eval failed");

    let mut dq = vec![0.0f32; q_dim * seq];
    let mut dk = vec![0.0f32; q_dim * seq];
    {
        let locked = kernels.bufs.sdpa_bwd2_out.as_f32_slice();
        read_channels_into(&locked, bwd2_out_ch, seq, 0, q_dim, &mut dq);
        read_channels_into(&locked, bwd2_out_ch, seq, q_dim, q_dim, &mut dk);
    }
    // For MHA, dV_full = dV (no GQA reduce needed)
    let dv = dv_full;

    // ── 11. RoPE backward in-place (CPU) ──
    rope_backward_inplace(&mut dq, heads, hd, seq);
    rope_backward_inplace(&mut dk, heads, hd, seq);

    // ── 12. Stage qBwd: mtrans Wq, then stage_spatial ──
    let q_bwd_sp = dyn_matmul::spatial_width(seq, dim);
    {
        let mut wqt = vec![0.0f32; q_dim * dim];
        vdsp::mtrans(&weights.wq, q_dim, &mut wqt, dim, dim, q_dim);
        let mut locked = kernels.bufs.q_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, q_dim, q_bwd_sp, &dq, seq, 0);
        stage_spatial(buf, q_dim, q_bwd_sp, &wqt, dim, seq);
    }

    // ASYNC: ANE qBwd || CPU dWq+dWk+dWv accumulation
    std::thread::scope(|s| {
        let ane_handle = s.spawn(|| {
            kernels.q_bwd.run(
                &[&kernels.bufs.q_bwd_in],
                &[&kernels.bufs.q_bwd_out],
            ).expect("ANE eval failed");
        });
        accumulate_dw(&cache.xnorm, dim, &dq, q_dim, seq, &mut grads.dwq);
        accumulate_dw(&cache.xnorm, dim, &dk, kv_dim, seq, &mut grads.dwk);
        accumulate_dw(&cache.xnorm, dim, &dv, kv_dim, seq, &mut grads.dwv);
        ane_handle.join().expect("ANE thread panicked");
    });

    // Read dx_attn directly from output IOSurface
    let mut dx_attn = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.q_bwd_out.as_f32_slice();
        dx_attn.copy_from_slice(&locked[..dim * seq]);
    }

    // ── 14. kvBwd(ANE): mtrans Wk/Wv, then stage_spatial ──
    let kv_bwd_sp = dyn_matmul::dual_spatial_width(seq, dim);
    {
        let mut wkt = vec![0.0f32; kv_dim * dim];
        let mut wvt = vec![0.0f32; kv_dim * dim];
        vdsp::mtrans(&weights.wk, kv_dim, &mut wkt, dim, dim, kv_dim);
        vdsp::mtrans(&weights.wv, kv_dim, &mut wvt, dim, dim, kv_dim);
        let mut locked = kernels.bufs.kv_bwd_in.as_f32_slice_mut();
        let buf = &mut *locked;
        stage_spatial(buf, kv_dim, kv_bwd_sp, &dk, seq, 0);
        stage_spatial(buf, kv_dim, kv_bwd_sp, &dv, seq, seq);
        stage_spatial(buf, kv_dim, kv_bwd_sp, &wkt, dim, seq + seq);
        stage_spatial(buf, kv_dim, kv_bwd_sp, &wvt, dim, 2 * seq + dim);
    }
    kernels.kv_bwd.run(&[&kernels.bufs.kv_bwd_in], &[&kernels.bufs.kv_bwd_out]).expect("ANE eval failed");

    // Read dx_kv directly from output IOSurface
    let mut dx_kv = vec![0.0f32; dim * seq];
    {
        let locked = kernels.bufs.kv_bwd_out.as_f32_slice();
        dx_kv.copy_from_slice(&locked[..dim * seq]);
    }

    // ── 15. Merge: dx_attn + dx_kv (vDSP vectorized) ──
    let mut dx_merged = vec![0.0f32; dim * seq];
    vdsp::vadd(&dx_attn, &dx_kv, &mut dx_merged);

    // ── 16. RMSNorm1 backward (CPU): bulk transpose → batch backward → transpose back ──
    let mut dx_rms1 = vec![0.0f32; dim * seq];
    {
        let mut dy_t = vec![0.0f32; seq * dim];
        let mut x_t = vec![0.0f32; seq * dim];
        let mut dx_t = vec![0.0f32; seq * dim];
        vdsp::mtrans(&dx_merged, seq, &mut dy_t, dim, dim, seq);
        vdsp::mtrans(&cache.x, seq, &mut x_t, dim, dim, seq);
        rmsnorm::backward_batch(&dy_t, &x_t, &weights.gamma1, &cache.rms_inv1, &mut dx_t, &mut grads.dgamma1, dim, seq);
        vdsp::mtrans(&dx_t, dim, &mut dx_rms1, seq, seq, dim);
    }

    // ── 17. Final: dx = dx_rms1 + dx2 (residual from attention branch, vDSP vectorized) ──
    let mut dx = vec![0.0f32; dim * seq];
    vdsp::vadd(&dx_rms1, &dx2, &mut dx);

    dx
}

// ── CPU helpers ──

/// Accumulate weight gradient via BLAS: dW[a_ch, b_ch] += A[a_ch, seq] @ B[b_ch, seq]^T
/// `a` is [a_ch * seq] row-major, `b` is [b_ch * seq] row-major, `dw` is [a_ch * b_ch].
fn accumulate_dw(a: &[f32], a_ch: usize, b: &[f32], b_ch: usize, seq: usize, dw: &mut [f32]) {
    vdsp::sgemm_at(a, a_ch, seq, b, b_ch, dw);
}

/// Pack activation data into the channel dimension of an IOSurface.
/// Uses copy_from_slice for vectorized memcpy on inner dimension.
fn pack_channels(dst: &mut [f32], _total_ch: usize, seq: usize, src: &[f32], src_ch: usize, ch_offset: usize) {
    for c in 0..src_ch {
        let d = (ch_offset + c) * seq;
        let s = c * seq;
        dst[d..d + seq].copy_from_slice(&src[s..s + seq]);
    }
}

/// RoPE backward: inverse rotation applied in-place.
/// Uses precomputed cos/sin table to avoid powf/cos/sin in inner loop.
fn rope_backward_inplace(dx: &mut [f32], heads: usize, hd: usize, seq: usize) {
    // Precompute cos/sin table [hd/2, seq]
    let pairs = hd / 2;
    let mut cos_table = vec![0.0f32; pairs * seq];
    let mut sin_table = vec![0.0f32; pairs * seq];
    for i in 0..pairs {
        let freq = 1.0 / 10000.0f32.powf(2.0 * i as f32 / hd as f32);
        for p in 0..seq {
            let theta = p as f32 * freq;
            cos_table[i * seq + p] = theta.cos();
            sin_table[i * seq + p] = theta.sin();
        }
    }
    // Apply rotation using precomputed table
    for h in 0..heads {
        for i in 0..pairs {
            let base0 = (h * hd + 2 * i) * seq;
            let base1 = (h * hd + 2 * i + 1) * seq;
            let tbase = i * seq;
            for p in 0..seq {
                let c = cos_table[tbase + p];
                let s = sin_table[tbase + p];
                let d0 = dx[base0 + p];
                let d1 = dx[base1 + p];
                dx[base0 + p] = c * d0 + s * d1;
                dx[base1 + p] = -s * d0 + c * d1;
            }
        }
    }
}
