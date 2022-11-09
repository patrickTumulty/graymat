

/// Sigmoid Function
///
/// * `val` - Float value
pub fn sigf(val: f32) -> f32 {
    return 1.0 / (1.0 + std::f32::consts::E.powf(val));
}

/// Sigmoid Function
///
/// * `val` - Double value
pub fn sig(val: f64) -> f64 {
    return 1.0 / (1.0 + std::f64::consts::E.powf(val))
}

