//! CPU training operations using Apple Accelerate (vDSP + vecLib).

pub mod vdsp;
pub mod rmsnorm;
pub mod cross_entropy;
pub mod adam;
pub mod silu;
pub mod embedding;
