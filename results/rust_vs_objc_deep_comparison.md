# Rust vs Obj-C: Deep Performance Comparison

> Honest head-to-head analysis. Measured on M4 Max, same model config.
> Date: 2026-03-11

## The Honest Answer: Obj-C is 3.7x Faster (Per Microbatch)

| Metric | Obj-C | Rust | Ratio |
|--------|-------|------|-------|
| 1 microbatch (fwd+bwd) | ~89ms | ~326ms | **Obj-C 3.7x faster** |
| With Adam amortized | ~99ms | ~326ms + 5.3ms | Obj-C 3.3x faster |
| Full Adam step (10 micro) | ~1000ms | ~3313ms | Obj-C 3.3x faster |

Note: Obj-C uses vocab=32768, Rust uses vocab=8192. Adjusting Obj-C to 8192 vocab:
~80ms fwd+bwd → Rust is **4.1x slower**.

## Component-Level Breakdown

### Obj-C (from overnight_ane_nl6_s512_v3.log, steady-state average)

| Component | Time | Notes |
|-----------|------|-------|
| ANE forward | 16ms | 3 kernels/layer × 6 layers |
| IO forward staging | 3.7ms | Pre-allocated IOSurfaces |
| RMSNorm forward | 1.1ms | 12 calls, C code |
| ANE backward | 17.5ms | 7 kernels/layer × 6 layers |
| IO backward staging | 6.8ms | Pre-allocated IOSurfaces |
| SiLU derivative | 6.4ms | Scalar loop in C |
| RMSNorm backward | 5.1ms | 12 calls |
| Classifier + loss | 25ms | vocab=32768, BLAS matmul |
| CBLAS dW wait | **0.0ms** | **Async dispatch!** |
| dW copy | 7ms | Gradient accumulation |
| **Total** | **~89ms** | Overhead: +10ms |

### Rust (from bench_step_time, release mode)

| Component | Time | Notes |
|-----------|------|-------|
| Forward (total) | 111ms | All 6 layers + classifier |
| Backward (total) | 215ms | All 6 layers + grad accum |
| Adam update | 53ms | Full step, not amortized |
| **Total** | **397ms** | 326ms fwd+bwd |

## WHY Is Rust 3.7x Slower?

The ANE hardware time is identical (~33.5ms for fwd+bwd). Everything else is overhead.

| Overhead Type | Obj-C | Rust | Gap |
|--------------|-------|------|-----|
| Non-ANE time | ~55ms | ~293ms | **5.3x** |

### Root Cause #1: Synchronous ANE Dispatch (BIGGEST FACTOR)

**Obj-C**: `cblas_wait=0.0ms` — dispatches dW BLAS computations async via GCD, runs them while ANE executes the next kernel. CPU and ANE work in parallel.

**Rust**: `run_kernel_reuse()` calls `exe.run()` → blocks on `read_f32()` → copies to new Vec → does CPU work → next kernel. Fully serial.

With 60 kernel calls/step, even 2-3ms of wasted pipeline bubble per kernel = **120-180ms** of pure serialization waste.

### Root Cause #2: Vec Allocation on Every Kernel Read

`output_td.read_f32().to_vec()` allocates a fresh Vec and copies for every kernel output. 60 calls × ~1.5MB average = **90MB of allocation+copy per step**.

Obj-C reads directly from pre-allocated buffers.

### Root Cause #3: Classifier Cost Difference

Obj-C: 25ms for vocab=32768 (BLAS matmul). Rust: ~8-10ms for vocab=8192 (vDSP sgemm).
This actually favors Rust by ~15ms, partially masking the other gaps.

### Root Cause #4: Per-Tensor grad_norm Allocations

`grad_norm()` calls `.map(|x| x*x).collect::<Vec<_>>()` — allocates a temporary Vec per gradient tensor, 56 times per step. Pure waste.

## Path to Parity (and Beyond)

### Phase 1: Close the Gap (target: match Obj-C at ~100ms/microbatch)

| Optimization | Expected Savings | Complexity |
|-------------|-----------------|------------|
| Async ANE dispatch (split launch/wait) | 120-180ms | High (layer.rs refactor) |
| Eliminate .to_vec() (read into pre-alloc) | 15-25ms | Low |
| Fix grad_norm allocations | 8-12ms | Low |
| Metal Adam (GPU idle, could do 6ms) | 47ms off update | Medium |
| **Combined** | ~200-260ms | |

**Projected**: 326ms → ~70-130ms fwd+bwd. **This would match or beat Obj-C.**

### Phase 2: Beat Obj-C (target: <80ms/microbatch)

| Optimization | Expected Savings | Why Rust-Only |
|-------------|-----------------|---------------|
| Full pipelining (Rayon work-stealing) | 15-25ms | Borrow checker = no memcpy into closures |
| Zero-copy staging (LockedSliceMut) | 5-10ms | RAII guards safe at compile time |
| Fused softcap (single-pass loop) | 2-3ms | Both could do this |
| SiLU vvexpf vectorization | 3-5ms | Both could do this |
| Pre-pack transposed weights once/step | 10-16ms | Eliminates 55 redundant transposes |
| **Combined** | ~35-60ms | |

**Projected**: ~35-70ms/microbatch. **Faster than Obj-C.**

### Phase 3: Structural Wins (Rust-only advantages)

| Win | Impact |
|-----|--------|
| Safe concurrent IOSurface access | No runtime overhead for thread safety |
| Deterministic memory (no autoreleasepool) | Stable RSS over 72K steps |
| Portability to Jetson via candle-rs | Same training code, different backend |
| Cross-crate LTO | Dead code elimination across crate boundaries |

## What Obj-C Does Right (That We Must Match)

1. **Async dW dispatch**: `dispatch_group_async` for BLAS, runs CPU while ANE works
2. **Per-layer IOSurface pre-allocation**: Done in `ane_compile()`, reused every step ✓ (we now do this)
3. **Direct IOSurface writes**: No intermediate Vec, writes directly to surface address
4. **Amortized Adam**: Only runs every 10 steps, cost spread across microbatches
5. **Vocab compaction awareness**: Maps 32768→8144, avoids wasted compute on inactive tokens

## Immediate Quick Wins (Can Do Today)

1. **Eliminate `.to_vec()` in `run_kernel_reuse()`** — read into pre-allocated buffer, return slice
2. **Fix `grad_norm()`** — replace `.map(|x| x*x).collect()` with in-place `vdsp::sve`
3. **Fuse softcap** — single loop instead of 4 vDSP passes
4. **Switch SiLU derivative to `vvexpf`** — already have the binding

These alone could save 30-45ms (8-12% improvement) with minimal code changes.

## Key Insight

The Obj-C isn't faster because Objective-C is a faster language. It's faster because it **pipelines CPU and ANE work**. The `cblas_wait=0` tells the whole story — every BLAS computation runs behind ANE latency.

Rust can do this better (borrow checker eliminates the memcpy that Obj-C needs for closure safety), but we haven't implemented it yet. Async dispatch is the single highest-leverage optimization.
