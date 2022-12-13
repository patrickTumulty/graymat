use std::path::Path;
use ndarray::array;
use crate::activation_function::ActivationFunction;
use crate::column_vector::ColumnVector;
use crate::cvec;
use crate::neural_network::{NeuralNetwork};

/// **Example Script: Binary to Int**
///
/// This example tests if a network can parse an array of 4 bits into an integer. The 4 bit array
/// can be interpreted as 16 different output values.
pub fn binary_to_int() {
    println!("Running Binary to Int Example!");

    let mut training_data: Vec<(ColumnVector, ColumnVector)> = Vec::with_capacity(16);

    for i in 0..16 {
        let mut col_in = ColumnVector::zeros(4);
        col_in[0] = (i & 0b1) as f32;
        col_in[1] = (i >> 1 & 0b1) as f32;
        col_in[2] = (i >> 2 & 0b1) as f32;
        col_in[3] = (i >> 3& 0b1) as f32;
        let mut col_out = ColumnVector::zeros(16);
        col_out[i] = 1.0;
        training_data.push((col_in.clone(), col_out.clone()));
    }

    let load_file_if_present = true;
    let model_filename = "binary_to_int_example.gnm";
    let mut nn: NeuralNetwork;

    if Path::new(model_filename).exists() && load_file_if_present {

        println!("Loading Network from File: {}", model_filename);
        nn = NeuralNetwork::from_file(".", model_filename);

    } else {
        nn = NeuralNetwork::new(4, 16, vec![12], ActivationFunction::SIGMOID);

        println!("Training Network...");
        nn.train(training_data, 100_000, 6, 0.4);

        println!("Saving to file: {}", model_filename);
        nn.to_file(".", model_filename);
    }

    let result1 = max(nn.evaluate(cvec![0, 1, 0, 0]));
    println!("Input: [2] | Network Output: [{}: {:3}]", result1.0, result1.1);
    let result2 = max(nn.evaluate(cvec![1, 1, 1, 0]));
    println!("Input: [7] | Network Output: [{}: {:3}]", result2.0, result2.1);
    let result3 = max(nn.evaluate(cvec![1, 0, 0, 1]));
    println!("Input: [9] | Network Output: [{}: {:3}]", result3.0, result3.1);

}

fn max(data: ColumnVector) -> (usize, f32) {

    let mut max_element = (0, f32::MIN);
    for (i, num) in data.get_data().iter().enumerate() {
        if num > &max_element.1 {
            max_element = (i, *num);
        }
    }
    return max_element;
}
