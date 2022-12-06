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

/// Hyperbolic tangent
///
/// First derivative of the hyperbolic tangent function
///
/// * `val` - Float value
pub fn tanh_prime(val: f32) -> f32 {
    return 1.0 - val.tanh().powf(2.0);
}

/// Rectified Linear Unit
///
/// * `val` - Float value
pub fn relu(val: f32) -> f32 {
    return val.max(0.0);
}

/// Rectified Linear Unit Prime
///
/// First derivative of the Rectified Linear Unit function
///
/// Note: The derivative of ReLU is the slope of the curve at a particular value. The slope for
/// values less than 0 is 0 and the slot for values above 0 is 1. The function is
/// non-differentiable for value 0. For simplicity, this function assumes slop 0 for value 0.
///
/// * `val` - Float value
pub fn relu_prime(val: f32) -> f32 {
    return if val > 0.0 { 1.0 } else { 0.0 };
}
