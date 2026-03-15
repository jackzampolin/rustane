# AGENTS.md — Rustane Training Engine

Instructions for AI agents working on this codebase. Read this before writing any code.

## Build & Test

```bash
cargo build --release                                              # build
cargo test -p engine --release                                     # full test suite
cargo test -p engine --test phase3_kernels --release               # ANE kernel tests
cargo test -p engine --test phase4_training --release              # training integration
cargo test -p engine --test profile_single_layer --release -- --ignored --nocapture  # per-layer profiling
cargo test -p engine --test bench_step_time --release -- --ignored --nocapture       # step timing
```

## Architecture

- **3 crates**: `ane-bridge` (ANE FFI), `metal-decode` (Metal shaders), `engine` (training orchestrator)
- **Training loop**: `full_model.rs` → `layer.rs` (per-layer forward/backward) → ANE kernels via IOSurface
- **10 ANE kernels per layer**: sdpaFwd, woFwd, ffnFused (forward); ffnBwdW2t, ffnBwdW13t, wotBwd, sdpaBwd1, sdpaBwd2, qBwd, kvBwd (backward)
- **Weight gradients**: 7 `accumulate_dw` (sgemm_at) calls per layer in backward
- **Optimizer**: CPU fused Adam with 2-thread parallelism

## Apple M4 Max Hardware Facts

These are VERIFIED characteristics. Do not second-guess them.

- **ANE peak**: 7.3 TFLOPS per single kernel, up to 17.8 TFLOPS with 32+ fused ops
- **ANE dispatch overhead**: ~0.095ms per XPC + IOKit round-trip
- **GPU peak**: ~15 TFLOPS fp16 (80% of real 18.4T, NOT Apple's misleading 36.86T)
- **L2 cache**: Copies of buffers <16MB are effectively FREE (~0.01ms for 1.5MB)
- **Memory bandwidth**: ~400 GB/s unified, single P-core sees ~80 GB/s
- **IOSurface**: Stores fp32, ANE casts to fp16 internally. Channel-interleaved layout: `buf[c * SP + s]`
- **IOSurface writes**: Fully cached on Apple Silicon UMA. NOT write-combined. Double-buffering adds overhead with zero benefit.
- **IOSurface width**: Must be multiple of 16 (silent data corruption otherwise)
- **ANE compiler**: Fails on rsqrt/sqrt after reduce ops — use pow(-0.5)
- **sgemm_at**: Uses beta=1.0, always zero output buffer before calling
- **vDSP/vecLib FFI**: ~0.5μs per call overhead. Not worth batching for ops already <0.1ms.
- **LLVM auto-vectorization**: For fused compute on L2-resident data, LLVM's single-pass scalar loop BEATS explicit multi-pass vDSP decomposition (register reuse > memory bandwidth)

## Proven Dead Ends — DO NOT RETRY

These optimizations have been tried and PROVEN to not work on this hardware/model. Retrying them wastes time.

| Experiment | Why it failed | Iterations wasted |
|-----------|---------------|-------------------|
| **Parallelize IOSurface staging** | IOSurface writes serialize at memory level regardless of CPU thread count. IOKit bottleneck. | 3 |
| **Fuse SiLU with IOSurface staging** | Scalar fusion has 3 non-adjacent writes/iter. Separate auto-vectorized passes beat manual fusion. | 2 |
| **Replace vDSP_vadd with cblas_saxpy** | BLAS dispatch overhead cancels memory savings for L2-resident 1.5MB buffers | 1 |
| **Multi-pass vDSP for SiLU derivative** | 7 vDSP calls = 84MB traffic vs LLVM single-pass = 24MB. Register reuse wins. | 2 |
| **Double-buffer IOSurface through heap** | Apple Silicon UMA is fully cached. Double-buffering adds 20MB memcpy with zero benefit. | 1 |
| **Fuse residual vadd into RMSNorm** | L2-resident buffer copies are ~0.01ms each. Fusing saves nothing measurable. | 1 |
| **Pre-stage small IOSurface ops (<0.5ms)** | Thread::scope overhead (20μs) eats into savings. Only helps for ops >1ms. | 2 |
| **Parallelize SiLU backward** | Loop at 0.87ms/layer is too short for thread parallelism. Scope overhead dominates. | 2 |
| **Metal GPU for Adam optimizer** | 56 GPU dispatches = 99.8% driver overhead (0.03ms actual compute). CPU loop is faster. | 1 |
| **Cross-layer dW pipeline** | dW sgemm already fully hidden by within-layer thread::scope overlaps. mpsc channel overhead + wait_current() serialization + cache thrashing added 5ms. | 1 |

## What DOES Work

These patterns are proven effective on M4 Max:

- **thread::scope for ANE overlap**: Overlap CPU staging with ANE kernel execution (within same layer)
- **Rebalancing dW across async blocks**: Move sgemm calls from CPU-bound scopes to ANE-bound scopes with headroom
- **Threading for ops >2ms**: split_at_mut + thread::scope for Adam (10→6ms), grad_norm (2.3→1.6ms)
- **Pre-computing in async blocks**: vvexpf/vvrecf sigmoid chain hidden behind ANE dispatch
- **Deferred readback**: IOSurface reads deferred to next layer's ANE overlap
- **CPU over GPU for small ops**: CPU with LLVM auto-vectorization beats Metal for anything dispatch-dominated

## Current Performance Profile (102ms/step)

```
Forward:     31ms  (5.2ms/layer avg + classifier)
Backward:    57ms  (9.5ms/layer avg)
Adam:         7ms  (2-thread CPU fused)
Grad norm:  1.6ms  (2-thread parallel)
CE loss:    3.4ms  (not a bottleneck)
Other:      2ms    (thread sync, overhead)
```

**The hard wall**: ffnBwdW13t IOSurface staging = 2.27ms/layer (13.6ms total, 13% of step).
Cannot be parallelized, fused, overlapped, or reduced without architectural change.

## Remaining Optimization Paths

Ranked by feasibility:

1. **Cross-layer dW pipeline** — fire dW sgemm to background thread, overlap with next layer's ANE. Plan: `dev/plans/cross-layer-dw-pipeline.md`
2. **Pre-transpose weight cache** — transpose all weights once in begin_step() instead of 60x per step
3. **Async microbatch pipelining** — fwd(T+1) on ANE while bwd(T) on CPU
4. **Fused 2-layer ANE kernels** — 32+ chained ops for 94% ANE utilization (HIGH risk, requires ANE compiler)
5. **INT8 W8A8 quantization** — constexpr_affine_dequantize in MIL for 2x throughput

## Code Conventions

- One variable per experiment. Never combine changes.
- Test with: custom auto_*.rs test (proves YOUR change is safe) + full test suite
- Log EVERY experiment to `system/experiments.tsv` (even failures — they prevent re-attempts)
- Commit message: `"Phase 5: <what> — <before>ms → <after>ms (<X>% faster)"`
- Write status to `/tmp/rustane-status-{agent-id}` at each phase (READING, CODING, TESTING, BENCHMARKING, DONE)
