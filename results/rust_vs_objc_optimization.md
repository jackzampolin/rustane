# Rust vs Obj-C: Optimization Opportunities for ANE Training

> Where Rust can outperform the Obj-C reference implementation.
> Measured against: autoresearch-ANE/native/training/train.m

## Executive Summary

For this workload (ANE-dominated transformer training), the two codebases use identical Accelerate/vDSP calls and the same ANE hardware path. The Obj-C CPU ops are already plain C (not Obj-C message dispatch). **Rust's advantage is structural, not computational** — but the structural advantages unlock real, measurable speedups.

Total estimated wall-clock improvement: **15-25%** (4.4s/step → ~3.3-3.7s/step).

## The Big Three (80% of gains)

### 1. IOSurface Reuse (~5-10% of step time)

**Current**: `run_kernel()` allocates 2 new IOSurfaces per kernel call. 10 kernels forward + 7 backward = 34 IOSurface alloc/dealloc cycles per step.

**Obj-C does this better**: `ane_runtime.h` pre-allocates IOSurfaces in `ane_compile()` and reuses them. Our Rust code creates+destroys per call.

**Fix**: Add persistent `TensorData` fields to `CompiledKernels`. Write directly into them via `as_f32_slice_mut()`.

**Rust advantage**: Ownership model makes this *safe* — the borrow checker ensures no concurrent reads during writes, without runtime checks.

**Effort**: Medium (~100 lines). **Priority**: 1st.

### 2. Zero-Copy Staging (~3-5% of step time)

**Current**: Build `Vec<f32>` staging buffer on heap → copy into IOSurface via `TensorData::with_f32()`. Two allocations and one full memcpy per kernel.

**Obj-C alternative**: Writes directly into IOSurface address via `IOSurfaceGetBaseAddress()`.

**Fix**: Stage weights+activations directly into the persistent IOSurface's f32 buffer instead of building intermediate Vecs.

**Rust advantage**: `LockedSliceMut` RAII guard from ane-crate maps naturally to `&mut [f32]`. Borrow checker prevents use-after-unlock at compile time.

**Effort**: Medium (~80 lines). **Priority**: 2nd.

### 3. Eliminate Per-Position Allocations (~2-4% of step time)

**Current**: RMSNorm forward/backward allocates 2-4 `vec![0.0; dim]` per sequence position. With SEQ=512, 2 norms per layer, 6 layers, forward+backward: **~24,000 small heap allocations per step**.

**Obj-C has the same problem**: `cpu_ops.h` uses `calloc`/`free` per call (plus a thread-unsafe static `g_rms_tmp`).

**Fix**: Pre-allocate scratch buffers in `ForwardCache` or a per-layer workspace struct. Pass `&mut scratch` instead of allocating.

**Rust advantage**: The ownership model makes buffer reuse safe without the thread-unsafe global that Obj-C uses. With rayon parallelism, each thread gets its own scratch via `thread_local!`.

**Effort**: Low (~50 lines). **Priority**: 3rd.

## Medium Wins (15% of gains)

### 4. Async Gradient Dispatch (1-3%)

**Obj-C approach**: Uses `dispatch_group_async` for dW computation, but then `dispatch_group_wait` before each layer forward — serializing what could be pipelined.

**Rust advantage**: Rayon's work-stealing scheduler can overlap dW BLAS calls with the next layer's ANE kernel. The borrow checker guarantees non-aliasing at compile time, so you don't need `memcpy` into captured buffers (Obj-C copies gradients into blocks for safety).

**Savings**: Eliminates 3-5 MB of memcpy per layer × 6 layers = 18-30 MB/step of unnecessary copies, plus enables true pipeline parallelism.

**Effort**: Medium (~150 lines with crossbeam channels). **Priority**: 4th.

### 5. Vectorized Adam (1-2%)

**Current** (both codebases): Scalar loop with `sqrt()` per parameter. 48.8M parameters.

**Fix**: Use vDSP bulk operations:
- `vDSP_vsmsa` for momentum: `m = beta1*m + (1-beta1)*g`
- `vDSP_vmul` + `vDSP_vsmsa` for variance: `v = beta2*v + (1-beta2)*g*g`
- `vvsqrtf` for bulk sqrt
- `vDSP_vdiv` for `m_hat / sqrt(v_hat)`

**Rust advantage**: None vs Obj-C (both can call vDSP). But the Obj-C code *doesn't* do this — it's a scalar loop in `cpu_ops.h`. Fixing it in Rust means we beat the Obj-C reference.

**Effort**: Low (~30 lines). **Priority**: 5th.

### 6. SiLU NEON Vectorization (1-2%)

**Current**: Scalar loop with 2x `exp()` per element × hidden×seq = 393K exp calls.

**Fix**: Use `vvexpf()` (Accelerate's vectorized exp) on chunks, then NEON for the sigmoid/SiLU arithmetic:
```
sig = 1/(1+exp(-x))  →  bulk vvexpf(-x), then NEON vrecpeq for reciprocal
silu = x * sig
silu' = sig * (1 + x*(1-sig))
```

**Rust advantage**: `std::arch::aarch64` NEON intrinsics are first-class. Obj-C can also use them but the integration is less ergonomic.

**Effort**: High (~200 lines for correct fast-sigmoid). **Priority**: 6th.

### 7. LTO + PGO (2-5% on CPU portions)

**LTO**: Add `lto = "thin"` to `[profile.release]` in workspace Cargo.toml. Cross-crate inlining eliminates vDSP wrapper call frames. One line.

**PGO**: `cargo-pgo` automates instrumented build → profile → optimized rebuild. ~5-15% on branch-heavy code paths (staging logic, position sampling).

**Rust advantage**: Cross-crate LTO is a Rust/LLVM feature. Obj-C gets LTO within a single compilation unit but not across .m files without explicit setup.

**Effort**: Trivial (LTO) / Low (PGO, 30 min setup). **Priority**: 7th.

## Structural Advantages (Not Raw Speed)

These don't show up as TFLOPS but matter for the 72K-step training run:

### Compile-Time Safety for Concurrent IOSurface Access
Obj-C uses `dispatch_group_async` with manual `memcpy` into blocks to avoid data races. Rust's borrow checker eliminates this class of bug entirely. No runtime cost.

### Deterministic Memory Lifetime
Obj-C's `@autoreleasepool` in `train.m` wraps the entire training loop — temporary NSObjects accumulate until loop end. Rust drops values at scope exit, keeping memory stable.

### Portability to Jetson
The entire CPU ops layer (Adam, RMSNorm, SiLU, cross-entropy) ports to CUDA via candle-rs. The Obj-C code is permanently Apple-only.

### Safe Scratch Buffer Reuse
Obj-C uses a thread-unsafe static global (`g_rms_tmp` in cpu_ops.h line 25). Rust's `thread_local!` or explicit `&mut` passing makes multi-threaded scratch safe by construction.

## What Doesn't Help

| Myth | Reality |
|------|---------|
| "Rust FFI to Accelerate is faster" | Identical to C calls. Same `BL` instruction. 0% difference. |
| "No ARC overhead" | Obj-C CPU ops are already plain C, not NSObject. ARC only touches ANE wrapper. |
| "Better auto-vectorization" | Same LLVM backend, same IR, same NEON output. |
| "Faster memory allocation" | Both use jemalloc/system malloc. Rust's advantage is *avoiding* allocation, not faster allocation. |

## Implementation Priority

| # | Optimization | Est. Speedup | Effort | Rust-Specific? |
|---|-------------|-------------|--------|----------------|
| 1 | IOSurface reuse | 5-10% | Medium | Partially (safety) |
| 2 | Zero-copy staging | 3-5% | Medium | Yes (RAII guards) |
| 3 | Pre-alloc scratch | 2-4% | Low | Yes (borrow checker) |
| 4 | Async dW dispatch | 1-3% | Medium | Yes (no memcpy) |
| 5 | Vectorized Adam | 1-2% | Low | No (vDSP) |
| 6 | SiLU NEON | 1-2% | High | No (both can) |
| 7 | LTO + PGO | 2-5% | Trivial/Low | Partially (cross-crate) |
| **Total** | | **15-25%** | | |

## Honest Assessment

The ANE kernels are 70-80% of step time. No amount of CPU optimization changes that. The gains above target the other 20-30%.

The biggest single win (IOSurface reuse) is actually a **bug fix** — matching what the Obj-C reference already does. After that, the Rust-specific advantages (zero-copy staging, safe scratch reuse, async dW without memcpy) add up to genuine structural improvements that the Obj-C architecture makes harder to achieve safely.

For the 72K run at 4.4s/step: a 20% speedup saves ~17 hours (88h → 71h). Worth it.
