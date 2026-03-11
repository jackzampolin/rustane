# f32↔f16 Conversion Benchmark — M4 Max

**Date:** 2026-03-10
**Hardware:** Apple M4 Max
**Mode:** Release build, 100 iterations, black_box to prevent elision

## f32 → f16 Results

| Buffer | Elements | half crate (ms) | ane crate (ms) | Ratio |
|--------|----------|-----------------|----------------|-------|
| DIM×SEQ (768×512) | 393K | 0.145 | 0.472 | 0.31x |
| HIDDEN×SEQ (3072×512) | 1.6M | 0.455 | 1.780 | 0.26x |
| DIM² (768²) | 590K | 0.138 | 0.623 | 0.22x |
| DIM×HIDDEN (768×3072) | 2.4M | 0.729 | 2.723 | 0.27x |

**Winner: `half` crate — 3-4x faster than ane crate's software conversion.**

## f16 → f32 Results (half crate only)

| Buffer | Elements | ms |
|--------|----------|----|
| DIM×SEQ (768×512) | 393K | 0.091 |
| HIDDEN×SEQ (3072×512) | 1.6M | 0.580 |
| DIM² (768²) | 590K | 0.217 |
| DIM×HIDDEN (768×3072) | 2.4M | 0.544 |

## Throughput

- **Peak f32→f16:** 13.55 GB/s (2.4M elements in 0.697ms)
- **M4 Max memory bandwidth:** 546 GB/s
- **Utilization:** 2.5% of bandwidth

## Analysis

- `half` crate uses NEON `vcvt` under the hood on aarch64, already well-optimized
- ane crate's `f32_to_fp16_bytes` does scalar loop + byte packing, 3-4x slower
- **13.55 GB/s is fine for training** — conversion of a 2.4M weight matrix takes <1ms
- At 768→3072 FFN (2.4M params), weight conversion adds ~0.7ms per update
- ANE kernel execution takes ~330µs, so conversion is ~2x the compute time
- **Optimization opportunity:** NEON SIMD with `vcvt_f16_f32` could push toward 100+ GB/s

## Decision

Use `half` crate for f32↔f16 conversion. Consider NEON intrinsics later only if profiling shows conversion is a bottleneck in the training loop.
