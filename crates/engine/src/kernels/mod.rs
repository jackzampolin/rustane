//! ANE kernel generators for training.
//!
//! Each generator builds an `ane::Graph` from a `ModelConfig`.
//! Pattern: placeholder → slice(acts + weights) → reshape → matmul → reshape.
//! Compile once at startup, update weights via IOSurface memcpy at runtime.

pub mod dyn_matmul;
pub mod sdpa_fwd;
pub mod sdpa_bwd;
pub mod ffn_fused;
