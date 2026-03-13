# ANE TFLOPS Benchmark — M4 Max

**Date:** 2026-03-10
**Hardware:** Apple M4 Max
**Method:** conv1x1 kernel (≈ matmul), median of 1000 iterations, release mode
**FLOPs formula:** 2 × IC × OC × W

## Results

| Config | IC | OC | W | Median µs | TFLOPS |
|--------|----|----|---|-----------|--------|
| 64→64, w=64 (minimal) | 64 | 64 | 64 | 102.4 | 0.005 |
| 128→128, w=128 | 128 | 128 | 128 | 93.7 | 0.045 |
| 256→256, w=256 | 256 | 256 | 256 | 95.3 | 0.352 |
| 768→768, w=512 (Wo) | 768 | 768 | 512 | 183.2 | 3.297 |
| **768→3072, w=512 (FFN up)** | 768 | 3072 | 512 | 330.3 | **7.315** |
| 3072→768, w=512 (FFN down) | 3072 | 768 | 512 | 746.1 | 3.238 |

## Analysis

- **Peak measured: 7.3 TFLOPS** via isolated conv1x1 kernels
- **Orion paper claims ~19 TFLOPS** on M4 Max using fused mega-kernels
- **Gap (2.6x):** Single conv1x1 ops don't saturate ANE. Fused mega-kernels (FFN = up→gelu→down, SDPA = QKV→softmax→out) overlap operations and avoid per-kernel launch overhead.
- **Asymmetry:** 768→3072 (7.3 TFLOPS) vs 3072→768 (3.2 TFLOPS). ANE strongly prefers channel expansion over reduction.
- **Minimum spatial width:** 64 elements enforced by ane crate.

## Implications

1. Must fuse kernels to approach 19 TFLOPS — individual matmuls won't do it
2. DynMatmul with fused graphs should close the gap significantly
3. FFN mega-kernel (768→3072→gelu→768 fused) is the first target
4. Weight update overhead (memcpy to IOSurface) is the other bottleneck to measure
