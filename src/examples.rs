use ndarray::array;
use rand::{Rng, thread_rng};
use rand::seq::SliceRandom;
use crate::{ColumnVector, NeuralNetwork};

pub fn xor() {
    println!("Running XOR Example!");

    let mut nn = NeuralNetwork::new(2, 1, vec![2]);

    let mut test_data = Vec::with_capacity(4);
    test_data.push((ColumnVector::from(&array![[0.0, 1.0]]), ColumnVector::from(&array![[1.0]])));
    test_data.push((ColumnVector::from(&array![[1.0, 0.0]]), ColumnVector::from(&array![[1.0]])));
    test_data.push((ColumnVector::from(&array![[0.0, 0.0]]), ColumnVector::from(&array![[0.0]])));
    test_data.push((ColumnVector::from(&array![[1.0, 1.0]]), ColumnVector::from(&array![[0.0]])));

    let mut rng = thread_rng();

    for _i in 0..50_000 {
        let index = rng.gen_range(0..4);
        nn.train(&vec![test_data[index].clone()], 0.1);
    }

    println!("Input: [0, 1] | Network Output: [{}]", nn.evaluate(ColumnVector::from(&array![[0.0, 1.0]]))[0]);
    println!("Input: [1, 0] | Network Output: [{}]", nn.evaluate(ColumnVector::from(&array![[1.0, 0.0]]))[0]);
    println!("Input: [0, 0] | Network Output: [{}]", nn.evaluate(ColumnVector::from(&array![[0.0, 0.0]]))[0]);
    println!("Input: [1, 1] | Network Output: [{}]", nn.evaluate(ColumnVector::from(&array![[1.0, 1.0]]))[0]);
}

pub fn binary_to_int() {
    println!("Running Binary to Int Example!");

    let mut nn = NeuralNetwork::new(4, 16, vec![12]);
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

    for _i in 0..200_000 {
        training_data.shuffle(&mut thread_rng());
        let mut data = Vec::with_capacity(5);
        for j in 0..6 {
            data.push(training_data[j].clone());
        }
        nn.train(&data, 0.3);
    }

    let result1 = max(nn.evaluate(ColumnVector::from(&array![[0.0, 1.0, 0.0, 0.0]])));
    println!("Input: [2] | Network Output: [{}: {:3}]", result1.0, result1.1);
    let result2 = max(nn.evaluate(ColumnVector::from(&array![[1.0, 1.0, 1.0, 0.0]])));
    println!("Input: [7] | Network Output: [{}: {:3}]", result2.0, result2.1);
    let result3 = max(nn.evaluate(ColumnVector::from(&array![[1.0, 0.0, 0.0, 1.0]])));
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
