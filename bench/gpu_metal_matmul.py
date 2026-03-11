#!/usr/bin/env python3
"""Investigate GPU utilization ceiling on M4 Max.

Tests different strategies to push past MLX's ~40% MFU:
1. Larger matrices (more work per dispatch)
2. Batched matmuls (amortize launch overhead)
3. Different data types (bfloat16 if supported)
4. Back-to-back dispatches (pipeline depth)
"""

import time
import mlx.core as mx
import numpy as np


def bench(label, fn, warmup=10, iters=100, flops_per_call=0):
    """Generic benchmark harness."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    med = np.median(times)
    tflops = flops_per_call / med / 1e12 if flops_per_call > 0 else 0
    print(f"  {label:<45} {np.median(times)*1000:>8.3f} ms  {tflops:>7.2f} TFLOPS")
    return tflops


def main():
    print(f"MLX {mx.__version__}, device: {mx.default_device()}")
    print()

    # =========================================================
    # Test 1: Matrix size sweep — find the sweet spot
    # =========================================================
    print("=== Matrix size sweep (square, fp16) ===")
    for N in [1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384]:
        a = mx.random.normal((N, N)).astype(mx.float16)
        b = mx.random.normal((N, N)).astype(mx.float16)
        mx.eval(a, b)
        flops = 2 * N * N * N
        def matmul(a=a, b=b):
            c = a @ b
            mx.eval(c)
        bench(f"{N}x{N}", matmul, flops_per_call=flops)

    # =========================================================
    # Test 2: Batched matmul — does batching help?
    # =========================================================
    print("\n=== Batched matmul (4096x4096, fp16) ===")
    N = 4096
    for batch in [1, 2, 4, 8]:
        a = mx.random.normal((batch, N, N)).astype(mx.float16)
        b = mx.random.normal((batch, N, N)).astype(mx.float16)
        mx.eval(a, b)
        flops = 2 * batch * N * N * N
        def matmul(a=a, b=b):
            c = a @ b
            mx.eval(c)
        bench(f"batch={batch}", matmul, flops_per_call=flops)

    # =========================================================
    # Test 3: Back-to-back dispatches (pipeline depth)
    # =========================================================
    print("\n=== Pipeline depth (4096x4096, fp16, multiple matmuls before eval) ===")
    N = 4096
    a = mx.random.normal((N, N)).astype(mx.float16)
    b = mx.random.normal((N, N)).astype(mx.float16)
    mx.eval(a, b)
    base_flops = 2 * N * N * N

    for depth in [1, 2, 4, 8]:
        def pipeline(a=a, b=b, d=depth):
            c = a
            for _ in range(d):
                c = c @ b
            mx.eval(c)
        bench(f"depth={depth} (chained)", pipeline, flops_per_call=base_flops * depth)

    # =========================================================
    # Test 4: Non-square (training-relevant, but bigger)
    # =========================================================
    print("\n=== Training-relevant shapes (fp16) ===")
    configs = [
        # (M, N, K, label)
        (512, 3072, 768, "gpt_karpathy FFN"),
        (512, 768, 3072, "gpt_karpathy proj"),
        (2048, 3072, 768, "4x batch FFN"),
        (2048, 768, 3072, "4x batch proj"),
        (4096, 3072, 768, "8x batch FFN"),
        (512, 3072, 1024, "qwen3-0.6B FFN"),
        (512, 5120, 1536, "1.5B FFN"),
        (2048, 5120, 1536, "1.5B 4x batch FFN"),
    ]
    for M, N, K, label in configs:
        a = mx.random.normal((M, K)).astype(mx.float16)
        b = mx.random.normal((K, N)).astype(mx.float16)
        mx.eval(a, b)
        flops = 2 * M * N * K
        def matmul(a=a, b=b):
            c = a @ b
            mx.eval(c)
        bench(label, matmul, flops_per_call=flops)

    print()
    print("Target: 25 TFLOPS = 68% MFU of Apple's claimed 36.86 TFLOPS")
    print("If MLX can't reach it, custom Metal shaders with hand-tuned tiling may be needed.")


if __name__ == "__main__":
    main()
