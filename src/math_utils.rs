use std::f32::consts::E;

/// Sigmoid Function
///
/// * `val` - Float value
pub fn sigf(val: f32) -> f32 {
    return 1.0 / (1.0 + E.powf(val * -1.0));
}

/// Sigmoid Prime Function
///
/// * `val` - Float value
pub fn sigf_prime(val: f32) -> f32 {
    let z: f32 = sigf(val);
    return z * (1.0 - z);
}

/// Sigmoid Function
///
/// * `val` - Double value
pub fn sig(val: f64) -> f64 {
    return 1.0 / (1.0 + std::f64::consts::E.powf(val))
}

