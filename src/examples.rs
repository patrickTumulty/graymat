use ndarray::array;
use rand::Rng;
use crate::{ColumnVector, NeuralNetwork};

pub fn xor() {
    println!("Running XOR Example!");

    let mut nn = NeuralNetwork::new(2, 1, vec![2]);

    let mut test_data = Vec::with_capacity(4);
    test_data.push((ColumnVector::from(&array![[0.0, 1.0]]), ColumnVector::from(&array![[1.0]])));
    test_data.push((ColumnVector::from(&array![[1.0, 0.0]]), ColumnVector::from(&array![[1.0]])));
    test_data.push((ColumnVector::from(&array![[0.0, 0.0]]), ColumnVector::from(&array![[0.0]])));
    test_data.push((ColumnVector::from(&array![[1.0, 1.0]]), ColumnVector::from(&array![[0.0]])));

    let mut rng = rand::thread_rng();

    for _i in 0..50_000 {
        let index = rng.gen_range(0..4);
        nn.train(&vec![test_data[index].clone()], 0.3);
    }

    println!("Input: [0, 1] | Network Output: [{}]", nn.evaluate(ColumnVector::from(&array![[0.0, 1.0]]))[0]);
    println!("Input: [1, 0] | Network Output: [{}]", nn.evaluate(ColumnVector::from(&array![[1.0, 0.0]]))[0]);
    println!("Input: [0, 0] | Network Output: [{}]", nn.evaluate(ColumnVector::from(&array![[0.0, 0.0]]))[0]);
    println!("Input: [1, 1] | Network Output: [{}]", nn.evaluate(ColumnVector::from(&array![[1.0, 1.0]]))[0]);
}
