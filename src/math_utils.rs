use std::f32::consts::E;

/// Sigmoid Function
///
/// * `val` - Float value
pub fn sig(val: f32) -> f32 {
    return 1.0 / (1.0 + E.powf(val));
}

