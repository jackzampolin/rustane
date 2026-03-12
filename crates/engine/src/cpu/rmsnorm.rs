//! RMSNorm — forward and backward passes using vDSP.
//!
//! Forward:  y = (x / rms(x)) * gamma
//!   where rms(x) = sqrt(mean(x²) + eps)
//!
//! Backward: dx = (gamma / rms) * (dy - x * dot(dy, x*gamma) / (n * rms²))

use super::vdsp;

const EPS: f32 = 1e-5;

/// RMSNorm forward pass.
/// `x`: input [dim], `gamma`: learned scale [dim], `out`: normalized output [dim].
/// Returns the inverse RMS (1/rms) needed for backward.
pub fn forward(x: &[f32], gamma: &[f32], out: &mut [f32]) -> f32 {
    let dim = x.len();
    assert_eq!(gamma.len(), dim);
    assert_eq!(out.len(), dim);

    // x² → out (reuse buffer temporarily)
    vdsp::vmul(x, x, out);

    // mean(x²)
    let mean_sq = vdsp::sve(out) / dim as f32;

    // 1 / sqrt(mean_sq + eps)
    let rms_inv = 1.0 / (mean_sq + EPS).sqrt();

    // out = x * rms_inv
    vdsp::vsmul(x, rms_inv, out);

    // out = out * gamma  (need scratch to avoid aliasing)
    let mut tmp = vec![0.0f32; dim];
    vdsp::vmul(out, gamma, &mut tmp);
    out.copy_from_slice(&tmp);

    rms_inv
}

/// RMSNorm backward pass.
/// `dy`: gradient from upstream [dim], `x`: original input [dim],
/// `gamma`: scale weights [dim], `rms_inv`: from forward pass.
/// Writes `dx` [dim] and accumulates into `dgamma` [dim].
pub fn backward(
    dy: &[f32],
    x: &[f32],
    gamma: &[f32],
    rms_inv: f32,
    dx: &mut [f32],
    dgamma: &mut [f32],
) {
    let dim = x.len();
    assert_eq!(dy.len(), dim);
    assert_eq!(gamma.len(), dim);
    assert_eq!(dx.len(), dim);
    assert_eq!(dgamma.len(), dim);

    // x_hat = x * rms_inv (normalized input, stored in dx temporarily)
    vdsp::vsmul(x, rms_inv, dx);

    // dgamma += dy * x_hat  (accumulate)
    let mut scratch = vec![0.0f32; dim];
    vdsp::vmul(dy, dx, &mut scratch); // scratch = dy * x_hat
    let mut tmp = vec![0.0f32; dim];
    vdsp::vadd(dgamma, &scratch, &mut tmp);
    dgamma.copy_from_slice(&tmp);

    // dx = rms_inv * (dy * gamma - x_hat * mean(dy * gamma * x_hat))
    vdsp::vmul(dy, gamma, &mut scratch); // scratch = dy * gamma
    vdsp::vmul(&scratch, dx, &mut tmp); // tmp = dy * gamma * x_hat
    let dot = vdsp::sve(&tmp) / dim as f32;

    vdsp::vsmul(dx, dot, &mut tmp); // tmp = x_hat * dot
    vdsp::vsub(&tmp, &scratch, dx); // dx = (dy * gamma) - (x_hat * dot)
    scratch.copy_from_slice(dx);
    vdsp::vsmul(&scratch, rms_inv, dx); // dx *= rms_inv
}

/// Batch RMSNorm forward on position-contiguous data.
/// `x_t` is [seq, dim] row-major (position-contiguous), `gamma` is [dim].
/// Writes `xnorm_t` [seq, dim] and `rms_inv` [seq].
/// Uses a single scratch allocation for all positions.
pub fn forward_batch(
    x_t: &[f32],
    gamma: &[f32],
    xnorm_t: &mut [f32],
    rms_inv: &mut [f32],
    dim: usize,
    seq: usize,
) {
    let mut scratch = vec![0.0f32; dim];
    for s in 0..seq {
        let x_pos = &x_t[s * dim..(s + 1) * dim];
        let out = &mut xnorm_t[s * dim..(s + 1) * dim];

        vdsp::vmul(x_pos, x_pos, &mut scratch);
        let mean_sq = vdsp::sve(&scratch) / dim as f32;
        let inv = 1.0 / (mean_sq + EPS).sqrt();
        rms_inv[s] = inv;

        vdsp::vsmul(x_pos, inv, out);
        vdsp::vmul(out, gamma, &mut scratch);
        out.copy_from_slice(&scratch);
    }
}

/// Batch RMSNorm backward on position-contiguous data.
/// All inputs/outputs are [seq, dim] row-major. Accumulates into `dgamma`.
pub fn backward_batch(
    dy_t: &[f32],
    x_t: &[f32],
    gamma: &[f32],
    rms_inv: &[f32],
    dx_t: &mut [f32],
    dgamma: &mut [f32],
    dim: usize,
    seq: usize,
) {
    let mut scratch = vec![0.0f32; dim];
    let mut tmp = vec![0.0f32; dim];
    for s in 0..seq {
        let dy_pos = &dy_t[s * dim..(s + 1) * dim];
        let x_pos = &x_t[s * dim..(s + 1) * dim];
        let dx_pos = &mut dx_t[s * dim..(s + 1) * dim];
        let inv = rms_inv[s];

        // x_hat = x * inv (stored in dx temporarily)
        vdsp::vsmul(x_pos, inv, dx_pos);

        // dgamma += dy * x_hat
        vdsp::vmul(dy_pos, dx_pos, &mut scratch);
        vdsp::vadd(dgamma, &scratch, &mut tmp);
        dgamma.copy_from_slice(&tmp);

        // dx = inv * (dy * gamma - x_hat * mean(dy * gamma * x_hat))
        vdsp::vmul(dy_pos, gamma, &mut scratch);
        vdsp::vmul(&scratch, dx_pos, &mut tmp);
        let dot = vdsp::sve(&tmp) / dim as f32;
        vdsp::vsmul(dx_pos, dot, &mut tmp);
        vdsp::vsub(&tmp, &scratch, dx_pos);
        scratch.copy_from_slice(dx_pos);
        vdsp::vsmul(&scratch, inv, dx_pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_unit_gamma() {
        // gamma=1 everywhere → pure normalization
        let x = [3.0, 4.0]; // rms = sqrt((9+16)/2) = sqrt(12.5)
        let gamma = [1.0, 1.0];
        let mut out = [0.0f32; 2];
        let rms_inv = forward(&x, &gamma, &mut out);

        let expected_rms = (12.5f32 + EPS).sqrt();
        assert!((rms_inv - 1.0 / expected_rms).abs() < 1e-5);
        assert!((out[0] - 3.0 / expected_rms).abs() < 1e-5);
        assert!((out[1] - 4.0 / expected_rms).abs() < 1e-5);
    }

    #[test]
    fn forward_with_gamma() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let gamma = [0.5, 0.5, 0.5, 0.5];
        let mut out = [0.0f32; 4];
        let rms_inv = forward(&x, &gamma, &mut out);

        let mean_sq = (1.0 + 4.0 + 9.0 + 16.0) / 4.0; // 7.5
        let expected_rms_inv = 1.0 / (mean_sq + EPS).sqrt();
        assert!((rms_inv - expected_rms_inv).abs() < 1e-5);

        for i in 0..4 {
            let expected = x[i] * expected_rms_inv * 0.5;
            assert!((out[i] - expected).abs() < 1e-5, "out[{i}] = {} vs {expected}", out[i]);
        }
    }

    #[test]
    fn backward_numerical_gradient() {
        // Numerical gradient check for dx
        let x = [1.0f32, -2.0, 3.0, -0.5];
        let gamma = [1.0, 0.5, 2.0, 1.5];
        let dy = [1.0, 1.0, 1.0, 1.0];
        let dim = x.len();

        let mut out = [0.0f32; 4];
        let rms_inv = forward(&x, &gamma, &mut out);

        let mut dx = [0.0f32; 4];
        let mut dgamma = [0.0f32; 4];
        backward(&dy, &x, &gamma, rms_inv, &mut dx, &mut dgamma);

        // Numerical gradient
        let eps = 1e-4;
        for i in 0..dim {
            let mut x_plus = x;
            let mut x_minus = x;
            x_plus[i] += eps;
            x_minus[i] -= eps;

            let mut out_plus = [0.0f32; 4];
            let mut out_minus = [0.0f32; 4];
            forward(&x_plus, &gamma, &mut out_plus);
            forward(&x_minus, &gamma, &mut out_minus);

            // loss = sum(out * dy) = sum(out) when dy=1
            let loss_plus: f32 = out_plus.iter().sum();
            let loss_minus: f32 = out_minus.iter().sum();
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);

            assert!(
                (dx[i] - numerical).abs() < 1e-2,
                "dx[{i}]: analytical={} vs numerical={numerical}", dx[i]
            );
        }
    }
}
