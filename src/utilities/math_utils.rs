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

/// Float compare
/// ```
/// use graymat::utilities::math_utils::float_compare;
///
/// let result = float_compare(0.1234599999, 0.1234588888, 4); // todo investigate why precision 5 doesn't work here
/// assert_eq!(result, true);
/// ```
/// * `lhs` - left hand side float value
/// * `rhs` - right hand side float value
/// * `precision` - number of decimal points to compare
/// * `returns` - true if lhs and rhs are equal within the designated precision
pub fn float_compare(lhs: f32, rhs: f32, precision: u8) -> bool {
    let mult = 10f32.powi(precision as i32);
    return ((lhs * mult) as u32) == ((rhs * mult) as u32);
}

/// Double compare
/// ```
/// use graymat::utilities::math_utils::double_compare;
///
/// let result = double_compare(0.1234599999, 0.1234588888, 5);
/// assert_eq!(result, true);
/// ```
/// * `lhs` - left hand side double value
/// * `rhs` - right hand side double value
/// * `precision` - number of decimal points to compare
/// * `returns` - true if lhs and rhs are equal within the designated precision
pub fn double_compare(lhs: f64, rhs: f64, precision: u8) -> bool {
    let mult = 10f64.powi(precision as i32);
    return ((lhs * mult) as u64) == ((rhs * mult) as u64);
}
