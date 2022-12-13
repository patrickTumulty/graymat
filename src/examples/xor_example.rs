use std::path::Path;
use ndarray::array;
use crate::activation_function::ActivationFunction;
use crate::column_vector::ColumnVector;
use crate::cvec;
use crate::neural_network::{NeuralNetwork};

pub fn xor() {
    println!("Running XOR Example!");

    let mut training_data = Vec::with_capacity(4);
    training_data.push((cvec![1, 0], cvec![1]));
    training_data.push((cvec![0, 1], cvec![1]));
    training_data.push((cvec![0, 0], cvec![0]));
    training_data.push((cvec![1, 1], cvec![0]));

    let load_file_if_present = true;
    let model_filename = "xor_example.gnm";
    let mut nn: NeuralNetwork;

    if Path::new(model_filename).exists() && load_file_if_present {

        println!("Loading Network from File: {}", model_filename);
        nn = NeuralNetwork::from_file(".", model_filename);

    } else {
        nn = NeuralNetwork::new(2, 1, vec![2], ActivationFunction::RELU);

        println!("Training Network...");
        nn.train(training_data, 1_000, 2, 0.3);

        println!("Saving to file: {}", model_filename);
        nn.to_file(".", model_filename);
    }

    println!("Input: [0, 1] | Network Output: [{}]", nn.evaluate(cvec![0, 1])[0]);
    println!("Input: [1, 0] | Network Output: [{}]", nn.evaluate(cvec![1, 0])[0]);
    println!("Input: [0, 0] | Network Output: [{}]", nn.evaluate(cvec![0, 0])[0]);
    println!("Input: [1, 1] | Network Output: [{}]", nn.evaluate(cvec![1, 1])[0]);
}
