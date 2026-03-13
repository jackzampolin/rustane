# Phase 1: Training Probe Results — M4 Max

**Date:** 2026-03-10
**Hardware:** Apple M4 Max
**Mode:** Release build

## Critical Finding

**DynMatmul pattern (slice + transpose + matmul) COMPILES and RUNS on ANE.**

This validates the entire training approach: compile once, update weights via IOSurface writes, eval in a loop.

## Test Results

### Small Config (768×64 @ 768×64 → 64×64)
- **Compile:** 50.5ms ✓
- **Single eval:** output sum = -0.3037 (non-zero, correct) ✓
- **1000 iterations:** all passed ✓
- **Dynamic weight update:** outputs differ by 125.14 ✓
- **FLOPs/eval:** 6.3M (too small for meaningful TFLOPS)

### GPT-2 FFN Scale (768×64 @ 768×256 → 64×256)
- **Compile:** ✓
- **Compute:** 536.7 µs/eval
- **With staging:** 786.9 µs/eval (overhead: 250.2 µs, 31.8%)
- **FLOPs/eval:** 25.2M

## Graph Pattern

```
Input: [1, 768, 1, 128]  (packed activations + weights in spatial dim)
  ├── slice [0,0,0,0] size [1,768,1,64]  → activations
  ├── slice [0,0,0,64] size [1,768,1,64] → weights
  ├── transpose [0,2,1,3] both          → [1,1,768,64]
  └── matmul (transpose_x=true)          → [1,1,64,64]
```

## IOSurface Layout

Stores fp32 — ANE casts to fp16 internally. No manual conversion needed.

Channel-interleaved packing (matching autoresearch-ANE):
- For channel c, spatial pos s: `buf[c * SP + s]`
- Activations: s ∈ [0, SEQ)
- Weights: s ∈ [SEQ, SEQ+OC)

Weight update: `copy_from_f32()` or `as_f32_slice_mut()` — zero-copy, in-place.

## Observations

1. **TFLOPS are low** at these sizes because per-kernel launch overhead dominates. The conv1x1 benchmark hit 7.3T at 768→3072 w=512. DynMatmul adds slice+transpose overhead on top of that.

2. **Staging overhead (250µs at 768×256)** includes IOSurface lock/unlock + 768×256×4 = 786KB memcpy. At scale (768×3072), this would be ~9.4MB/update.

3. **The benchmark methodology needs improvement** — warm-up asymmetry caused negative overhead in the small config. Future benchmarks should use more careful interleaving.

4. **Both configs compiled on first try** — the ane crate's graph builder correctly handles slice+transpose+matmul chains.

## What This Unlocks

- Training loop: forward on ANE → backward on CPU/GPU → stage weights → repeat
- Fused mega-kernels: FFN = DynMatmul(up) → GELU → DynMatmul(down) as single graph
- KV-cache integration: concat + slice for attention with persistent state
