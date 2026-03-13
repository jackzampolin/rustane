# Phase 0: ANE Crate Analysis

**Date**: 2026-03-10
**Status**: Source read, tests drafted, awaiting ANE hardware for verification

## Critical Finding: IOSurface Layout Mismatch

### ane crate (inference pattern)
```
make_surface(byte_count):
  width = byte_count, height = 1, bytes_per_element = 1, bytes_per_row = byte_count
```
The crate creates **flat byte buffers** and writes fp32 data linearly. ANE casts to fp16 internally. This works for inference because the MIL signature declares `tensor<fp32, ...>` I/O.

### autoresearch-ANE (training pattern)
```
make_surface(bytes):
  SAME flat layout: width=bytes, height=1, bpe=1, bpr=bytes
```

**SURPRISE**: The Obj-C training code uses the **exact same flat IOSurface layout** as the ane crate! The difference is NOT in how IOSurfaces are created, but in **how data is written to them**.

### The Real Difference: Data Layout

Training kernels (DynMatmul pattern) pack **activations AND weights** into the same IOSurface using a **channel-interleaved** layout:

```
For a DynMatmul with input_channels=IC, spatial=SEQ+OC:
  IOSurface size = IC * (SEQ + OC) * sizeof(fp16) bytes

  For each channel d in 0..IC:
    buf[d * SP + 0..SEQ]     = activation data (row d)
    buf[d * SP + SEQ..SEQ+OC] = weight data (row d)
```

This is what `io_write_dyn()` does — it interleaves acts and weights per channel, converting f32→fp16 via NEON during the write.

### Kernel-specific layouts (from io.h)

| Kernel | Channels | Spatial (SP) | Layout per channel |
|--------|----------|--------|-----|
| sdpaFwd | DIM | SEQ + Q_DIM + KV_DIM + KV_DIM | acts \| Wq \| Wk \| Wv |
| woFwd | Q_DIM | SEQ + DIM | acts \| Wo |
| ffnFused | DIM | 2×SEQ + 3×HIDDEN | xnorm \| x2 \| W1 \| W3 \| W2 |
| ffnBwdW2t | DIM | SEQ + HIDDEN | dffn \| W2 |
| ffnBwdW13t | HIDDEN | 2×SEQ + 2×DIM | dh1 \| dh3 \| W1 \| W3 |
| wotBwd | DIM | SEQ + Q_DIM | dy \| Wo |
| qBwd | Q_DIM | SEQ + DIM | dq \| Wq |
| kvBwd | KV_DIM | 2×SEQ + 2×DIM | dk \| dv \| Wk \| Wv |

### What ane-bridge needs

1. **IOSurface creation**: Can reuse the ane crate's flat layout (they match!)
2. **fp16 write functions**: Need channel-interleaved write helpers that pack acts + weights per the above table
3. **NEON f32↔f16**: Need fast vectorized conversion (the ane crate uses software conversion)

## Weight Blob Handling

The ane crate packs ALL weight blobs into a single `weight.bin` with per-blob chunk headers (64-byte global header + 64-byte per-blob header + data). Each blob is referenced by offset in the MIL text via `BLOBFILE(path="weight.bin", offset=N)`.

The Obj-C code (`compile_kern_mil_w`) writes **separate files** per weight key:
```objc
for (NSString *path in weights) {
    NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
    [weights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
}
```

BUT — the ane crate's approach (single weight.bin, multiple offsets) is how CoreML/MIL actually works internally. The Obj-C code supports multiple weight files for convenience, but a single blob with offsets is equivalent and simpler.

**Verdict**: ane crate's weight handling works for multiple BLOBFILEs. sdpaFwd's 3 constant blobs (mask, rope_cos, rope_sin) are handled via `graph.constant()` calls, each getting its own offset.

Note: sdpaFwd's **dynamic weights** (Wq, Wk, Wv) are NOT in BLOBFILEs — they're packed into the input IOSurface spatial dimension at runtime. Only the **static** data (attention mask, RoPE tables) use BLOBFILEs.

## Temp Directory

Both implementations:
1. Get hex identifier from model
2. Create `$TMPDIR/<hex>/` and `$TMPDIR/<hex>/weights/`
3. Write `model.mil` and `weights/weight.bin`
4. Compile

The ane crate handles this correctly in `client.rs:90-98`.

## Summary

| Unknown | Status | Details |
|---------|--------|---------|
| Multiple BLOBFILEs | **WORKS** | Static constants (mask, RoPE) via `graph.constant()` → offsets in weight.bin |
| Temp directory | **WORKS** | Automatic via hex_id, identical pattern to Obj-C |
| IOSurface creation | **WORKS** | Same flat layout! The difference is in write helpers, not creation |

The main gap is: we need **channel-interleaved fp16 write functions** for each kernel pattern, and **NEON-accelerated f32↔fp16 conversion**. The IOSurface creation itself is compatible.
