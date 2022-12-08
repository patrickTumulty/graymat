

use ndarray::{Array2, AssignElem};
use rand::Rng;
use rand::rngs::ThreadRng;


pub mod math {

    use ndarray::Array2;
    use crate::utilities::math_utils;

    /// Sigmoid
    ///
    /// Perform sigmoid function on a 2 dimensional array
    ///
    /// * `arr` - Array to process
    pub fn sig(arr: &Array2<f32>) -> Array2<f32> {
        let mut result = arr.to_owned();
        for element in result.iter_mut() {
            *element = math_utils::sigf(*element);
        }
        return result;
    }

    /// Sigmoid Prime
    ///
    /// Perform first derivative of sigmoid function on a 2 dimensional array
    ///
    /// * `arr` - Array to process
    pub fn sig_prime(arr: &Array2<f32>) -> Array2<f32> {
        let mut result: Array2<f32> = arr.to_owned();
        for element in result.iter_mut() {
            *element = math_utils::sigf_prime(*element);
        }
        return result;
    }

    /// Hyperbolic tangent
    ///
    /// * `arr` - Array to process
    pub fn tanh(arr: &Array2<f32>) -> Array2<f32> {
        let mut result: Array2<f32> = arr.to_owned();
        for element in result.iter_mut() {
            *element = element.tanh();
        }
        return result;
    }

    /// Hyperbolic tangent
    ///
    /// First derivative of the hyperbolic tangent function
    ///
    /// * `arr` - Array to process
    pub fn tanh_prime(arr: &Array2<f32>) -> Array2<f32> {
        let mut result: Array2<f32> = arr.to_owned();
        for element in result.iter_mut() {
            *element = math_utils::tanh_prime(*element);
        }
        return result;
    }

    /// Rectified Linear Unit
    ///
    /// * `arr` - Array to process
    pub fn relu(arr: &Array2<f32>) -> Array2<f32> {
        let mut result: Array2<f32> = arr.to_owned();
        for element in result.iter_mut() {
            *element = math_utils::relu(*element);
        }
        return result;
    }

    /// Rectified Linear Unit
    ///
    /// First derivative of the Rectified Linear Unit function
    ///
    /// Note: The derivative of ReLU is the slope of the curve at a particular value. The slope for
    /// values less than 0 is 0 and the slot for values above 0 is 1. The function is
    /// non-differentiable for value 0. For simplicity, this function assumes slop 0 for values
    /// 0.
    ///
    /// * `arr` - Array to process
    pub fn relu_prime(arr: &Array2<f32>) -> Array2<f32> {
        let mut result: Array2<f32> = arr.to_owned();
        for element in result.iter_mut() {
            *element = math_utils::relu_prime(*element);
        }
        return result;
    }

    /// Multiple an array2 by a certain power
    ///
    /// * `arr` - Array
    /// * `power` - Raise to the power of
    pub fn pow(arr: &Array2<f32>, power: f32) -> Array2<f32> {
        let mut result = arr.to_owned();
        for element in result.iter_mut() {
            *element = element.powf(power);
        }
        return result;
    }
}

/// Randomize all elements of a 2 dimensional array
///
/// * `arr` - 2 dimensional array to modify
/// * `lower` - lower bounds of random value (inclusive)
/// * `upper` - upper bounds of random value (inclusive)
pub fn randomize_array(arr: &mut Array2<f32>, lower: f32, upper: f32) {
    let mut rng: ThreadRng = rand::thread_rng();
    for val in arr.iter_mut() {
        val.assign_elem(rng.gen_range(lower..=upper))
    }
}

pub fn range_fill_offset(arr: &mut Array2<i32>, start: i32, step: i32) {
    let mut counter: i32 = start;
    for item in arr.iter_mut() {
        *item = counter;
        counter += step;
    }
}

pub fn range_fill(arr: &mut Array2<i32>) {
    range_fill_offset(arr, 0, 1);
}


