use ndarray::array;
use rand::{Rng, thread_rng};
use crate::column_vector::ColumnVector;
use crate::cvec;
use crate::neural_network::NeuralNetwork;

pub fn xor() {
    println!("Running XOR Example!");

    let mut nn = NeuralNetwork::new(2, 1, vec![2]);

    let mut test_data = Vec::with_capacity(4);
    test_data.push((cvec![1, 0], cvec![1]));
    test_data.push((cvec![0, 1], cvec![1]));
    test_data.push((cvec![0, 0], cvec![0]));
    test_data.push((cvec![1, 1], cvec![0]));

    let mut rng = thread_rng();

    for _i in 0..50_000 {
        let index = rng.gen_range(0..4);
        nn.train(&vec![test_data[index].clone()], 0.1);
    }

    println!("Input: [0, 1] | Network Output: [{}]", nn.evaluate(cvec![0, 1])[0]);
    println!("Input: [1, 0] | Network Output: [{}]", nn.evaluate(cvec![1, 0])[0]);
    println!("Input: [0, 0] | Network Output: [{}]", nn.evaluate(cvec![0, 0])[0]);
    println!("Input: [1, 1] | Network Output: [{}]", nn.evaluate(cvec![1, 1])[0]);
}
