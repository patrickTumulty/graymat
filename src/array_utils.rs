use std::ops::Add;
use ndarray::{Array2};
use num::Zero;
use crate::math_utils;


/// Sum each element of two Array2 types.
///
/// * `lhs` - Left hand side 2 dimensional array
/// * `rhs` - Right hand side 2 dimensional array
/// * `returns` - the summed array
pub fn add<T>(lhs: &Array2<T>, rhs: &Array2<T>) -> Array2<T>
    where T: Add + Copy + Zero
{
    if lhs.shape() != rhs.shape() {
        panic!("Cannot add two arrays with difference dimensions");
    }
    let mut arr: Array2<T> = Array2::zeros((lhs.shape()[0], lhs.shape()[1]));
    for i in 0..arr.shape()[0] {
        for j in 0..arr.shape()[1] {
            arr[[i,j]] = lhs[[i,j]] + rhs[[i,j]];
        }
    }
    return arr;
}

/// Sigmoid
///
/// Perform sigmoid on an 2 dimensional array
///
/// * `arr` - Array to process
pub fn sig(arr: &mut Array2<f32>) {
    for i in 0..arr.shape()[0] {
        for j in 0..arr.shape()[1] {
            math_utils::sig(arr[[i, j]]);
        }
    }
}



