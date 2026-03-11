#!/usr/bin/env python3
"""Phase 0.5.1: Measure actual GPU TFLOPS via MLX matmul.

Runs a known fp16 matmul (4096x4096 x 4096x4096), times 100 iterations
after warmup, reports median/p5/p95 TFLOPS.

Compare against Apple's claimed ~36.86 TFLOPS peak for M4 Max.
"""

import time
import mlx.core as mx
import numpy as np


def bench_matmul(M: int, N: int, K: int, dtype, warmup: int = 10, iters: int = 100):
    """Benchmark a single matmul configuration."""
    a = mx.random.normal((M, K)).astype(dtype)
    b = mx.random.normal((K, N)).astype(dtype)
    mx.eval(a, b)

    # Warmup
    for _ in range(warmup):
        c = a @ b
        mx.eval(c)

    # Timed iterations
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        c = a @ b
        mx.eval(c)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times_ms = np.array(times) * 1000
    flops_per_op = 2 * M * N * K  # matmul = 2*M*N*K FLOPs
    tflops = np.array([flops_per_op / t / 1e12 for t in times])

    return {
        "M": M, "N": N, "K": K,
        "dtype": str(dtype),
        "median_ms": float(np.median(times_ms)),
        "p5_ms": float(np.percentile(times_ms, 5)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "median_tflops": float(np.median(tflops)),
        "p5_tflops": float(np.percentile(tflops, 5)),
        "p95_tflops": float(np.percentile(tflops, 95)),
        "iters": iters,
    }


def main():
    print(f"MLX version: {mx.__version__}")
    print(f"Metal device: {mx.default_device()}")
    print()

    configs = [
        # Standard benchmark sizes
        (4096, 4096, 4096, mx.float16, "4K x 4K fp16"),
        (4096, 4096, 4096, mx.float32, "4K x 4K fp32"),
        # Training-relevant sizes (gpt_karpathy: DIM=768, FFN=3072, SEQ=512)
        (512, 3072, 768, mx.float16, "SEQ×FFN×DIM fp16 (FFN fwd)"),
        (512, 768, 3072, mx.float16, "SEQ×DIM×FFN fp16 (FFN proj)"),
        # Larger model sizes (Qwen3-0.6B: DIM=1024, FFN=3072)
        (512, 3072, 1024, mx.float16, "0.6B FFN fwd fp16"),
        # Big square matmul for peak TFLOPS
        (8192, 8192, 8192, mx.float16, "8K x 8K fp16 (peak test)"),
    ]

    print(f"{'Config':<30} {'Median ms':>10} {'p5 ms':>10} {'p95 ms':>10} {'TFLOPS':>10} {'p5':>8} {'p95':>8}")
    print("-" * 96)

    results = []
    for M, N, K, dtype, label in configs:
        r = bench_matmul(M, N, K, dtype, warmup=10, iters=100)
        results.append((label, r))
        print(f"{label:<30} {r['median_ms']:>10.3f} {r['p5_ms']:>10.3f} {r['p95_ms']:>10.3f} "
              f"{r['median_tflops']:>10.2f} {r['p5_tflops']:>8.2f} {r['p95_tflops']:>8.2f}")

    print()
    peak = max(r['median_tflops'] for _, r in results)
    print(f"Peak measured: {peak:.2f} TFLOPS")
    print(f"Apple claimed: ~36.86 TFLOPS (M4 Max fp16)")
    print(f"Utilization:   {peak / 36.86 * 100:.1f}% of claimed peak")


if __name__ == "__main__":
    main()
