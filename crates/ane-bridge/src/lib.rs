//! Safe Rust FFI bindings to Apple Neural Engine private APIs.
//!
//! Wraps the `ane` crate and extends it for training:
//! - Channel-interleaved IOSurface writes (dynamic weight pipeline)
//! - NEON-accelerated f32↔f16 conversion
//! - Training kernel patterns (DynMatmul, DualDynMatmul, fused SDPA/FFN)

// Re-export the ane crate's public API
pub use ane;
