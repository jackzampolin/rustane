# rustane

Rust-native hybrid inference engine for Apple Neural Engine + Metal GPU.

Train on ANE → export weights → run fast with hybrid ANE prefill + Metal decode.

## Status

Early scaffold. Not yet functional.

## Architecture

```
                 ┌─────────────┐
                 │   engine    │  hybrid orchestrator
                 └──────┬──────┘
                ┌───────┴───────┐
         ┌──────┴──────┐ ┌──────┴──────┐
         │  ane-bridge  │ │ metal-decode │
         │  (prefill)   │ │  (decode)    │
         └──────────────┘ └──────────────┘
              ANE              Metal GPU
         private APIs       custom shaders
```

**ane-bridge** — Safe Rust FFI to ANE private APIs via dlopen. Handles `_ANEInMemoryModel`, `_ANERequest`, IOSurface weight packing.

**metal-decode** — Custom Metal shaders for single-token decode. One command buffer per token, zero allocations. Kernels: q8_gemv, q4_gemv, rmsnorm, rope, sdpa_causal.

**engine** — Orchestrates ANE prefill (fused FFN mega-kernels) + Metal GPU decode + CPU fallback.

## Target Hardware

- Apple Silicon (M4 Max, 128GB) — primary dev platform
- NVIDIA Jetson — deployment target via candle-rs + CUDA

## Setup

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build
cargo build

# Test
cargo test
```

## Sister Projects

- [autoresearch-ANE](https://github.com/ncdrone/autoresearch-ANE) — ANE training (native Obj-C, private APIs)
- [autoresearch-mlx](https://github.com/ncdrone/autoresearch-mlx) — MLX training (Python)

## Credits

See [CREDITS.md](CREDITS.md).

## License

MIT
