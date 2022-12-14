use std::fmt::{Display, Formatter};
use ndarray::{Array2};
use std::fmt::Write;
use rand::seq::SliceRandom;
use rand::thread_rng;
use crate::activation_function::ActivationFunction;
use crate::column_vector::ColumnVector;
use crate::neural_network_io::{check_gnm_filepath, from_file, to_file};
use crate::utilities::array2_utils;


// Cost Expression
pub type CE = fn(result: &Array2<f32>, target: &Array2<f32>) -> Array2<f32>;
// Activation Expression
pub type AE = fn(x: &Array2<f32>) -> Array2<f32>;

pub struct NeuralNetwork {
    input_neurons: usize,
    output_neurons: usize,
    hidden_layer_sizes: Vec<usize>,
    layers: Vec<NeuralNetworkLayer>,
    cost_function: CE,
    activation_function: ActivationFunction,
    activation_expression: AE,
    activation_expression_prime: AE
}

impl NeuralNetwork {
    /// Constructor
    ///
    /// This constructor produces a neural network with randomly generated weights and biases
    ///
    /// * `input_neurons` - Number of input neurons
    /// * `output_neurons` - Number of output neurons
    /// * `hidden_layer_sizes` - Vector defining how many hidden layers there should be and the
    ///                          size of each hidden layer. An empty vector results in the input
    ///                          neurons being linked directly to the output neurons.
    /// * `activation` - Activation function
    pub fn new(input_neurons: usize, output_neurons: usize, hidden_layer_sizes: Vec<usize>, activation: ActivationFunction) -> Self {
        let number_of_hidden_layers: usize = hidden_layer_sizes.len();
        let mut instance = NeuralNetwork {
            input_neurons,
            output_neurons,
            hidden_layer_sizes,
            layers: Vec::with_capacity(number_of_hidden_layers + 1),
            cost_function: NeuralNetwork::calculate_cost_default,
            activation_function: ActivationFunction::SIGMOID,
            activation_expression: array2_utils::math::sig,
            activation_expression_prime: array2_utils::math::sig_prime
        };
        Self::set_activation_function(&mut instance, activation);
        Self::init_network_layers(&mut instance);
        Self::randomize_weights_and_biases(&mut instance);
        return instance;
    }

    /// Return a neural network object from a known collection of weights and biases
    ///
    /// * `weights` - Neural Network weights in ascending order
    /// * `biases` - Neural Network biases in ascending order
    /// * `activation` - Activation function
    pub fn from(weights: Vec<Array2<f32>>, biases: Vec<Array2<f32>>, activation: ActivationFunction) -> Self {
        assert_eq!(weights.len(), biases.len());
        let number_of_hidden_layers: usize = weights.len();
        let mut instance = NeuralNetwork {
            input_neurons: weights[0].dim().1,
            output_neurons: weights[weights.len() - 1].dim().1,
            hidden_layer_sizes: Vec::with_capacity(number_of_hidden_layers),
            layers: Vec::with_capacity(number_of_hidden_layers),
            cost_function: NeuralNetwork::calculate_cost_default,
            activation_function: ActivationFunction::SIGMOID,
            activation_expression: array2_utils::math::sig,
            activation_expression_prime: array2_utils::math::sig_prime
        };
        Self::set_activation_function(&mut instance, activation);
        for i in 0..instance.layers.capacity() {
            instance.layers.push(NeuralNetworkLayer {
                weights: weights[i].clone(),
                biases: biases[i].clone()
            })
        }
        return instance;
    }

    /// Set the activation function for this network
    ///
    /// * `function` - Activation function type
    pub fn set_activation_function(&mut self, function: ActivationFunction) {
        self.activation_function = function.to_owned();
        match function {
            ActivationFunction::SIGMOID => {
                self.activation_expression = array2_utils::math::sig;
                self.activation_expression_prime = array2_utils::math::sig_prime;
            }
            ActivationFunction::TANH => {
                self.activation_expression = array2_utils::math::tanh;
                self.activation_expression_prime = array2_utils::math::tanh_prime;
            }
            ActivationFunction::RELU => {
                self.activation_expression = array2_utils::math::relu;
                self.activation_expression_prime = array2_utils::math::relu_prime;
            }
        }
    }

    /// Set the cost function for this neural network
    ///
    /// # Note:
    /// Custom cost functions will not be saved into a .gnm file.
    ///
    /// * `expression` - Cost expression
    pub fn set_cost_function(&mut self, expression: CE) {
        self.cost_function = expression;
    }

    ///
    /// Init net work layers
    ///
    fn init_network_layers(instance: &mut NeuralNetwork) {
        let mut layer_inputs = instance.input_neurons;
        for layer_size in &instance.hidden_layer_sizes {
            instance.layers.push(NeuralNetworkLayer::new(layer_inputs, *layer_size));
            layer_inputs = *layer_size;
        }
        instance.layers.push(NeuralNetworkLayer::new(layer_inputs, instance.output_neurons));
    }

    ///
    /// Init the network with random values
    ///
    fn randomize_weights_and_biases(instance: &mut NeuralNetwork) {
        for layer in instance.layers.iter_mut() {
            array2_utils::randomize_array(layer.weights_mut(), 0.0, 1.0);
            array2_utils::randomize_array(layer.biases_mut(), 0.0, 1.0);
        }
    }

    /// Forward propagate a column vector of inputs through the network to calculate a result
    ///
    /// * `inputs` - ColumnVector inputs
    /// * `returns` - ColumnVector outputs
    pub fn evaluate(&self, inputs: ColumnVector) -> ColumnVector {
        let mut activation: Array2<f32> = inputs.get_data().to_owned();
        for layer in self.layers.iter() {
            activation = self.non_linearity(&((layer.weights().dot(&activation)) + layer.biases()));
        }
        return ColumnVector::from(&activation);
    }

    /// Train the network using stochastic gradient descent
    ///
    /// * `training_data` - Training data is a list of tuples (x, y) where x is the input data, and
    ///                     y is the target output.
    /// * `iterations` - Number of times to iterate the training data
    /// * `batch_size` - Size of mini batches
    /// * `learning_rate` - The learning rate
    pub fn train(&mut self,
                 mut training_data: Vec<(ColumnVector, ColumnVector)>,
                 iterations: u32,
                 batch_size: usize,
                 learning_rate: f32)
    {
        let mut batch: Vec<(ColumnVector, ColumnVector)> = Vec::with_capacity(batch_size);
        for _i in 0..iterations {
            training_data.shuffle(&mut thread_rng());
            for j in 0..(training_data.len() / batch_size) {
                let lower = j * batch_size;
                let upper = lower + batch_size;
                training_data.as_slice()[lower..upper]
                             .clone_into(&mut batch);
                self.train_batch(&batch, learning_rate);
            }
        }
    }

    /// Train the network given a collection of inputs and expected outputs
    ///
    /// This method will update the networks weights and biases with the averaged result of
    /// all training data.
    ///
    /// * `training_data` - vector of (input, target) tuples. Input is the test data and target
    ///                     is the expected result.
    /// * `learning_rate` - learning rate
    fn train_batch(&mut self, training_data: &Vec<(ColumnVector, ColumnVector)>, learning_rate: f32) {

        let adjustment_vectors = self.init_zeroed_adjustment_matrices();
        let mut weight_adjustments: Vec<Array2<f32>> = adjustment_vectors.0;
        let mut bias_adjustments: Vec<Array2<f32>> = adjustment_vectors.1;

        for i in 0..training_data.len() {
            let result = self.back_propagate(training_data[i].0.get_data(), training_data[i].1.get_data());
            for j in 0..self.layers.len() {
                weight_adjustments[j] += &result.0[j];
                bias_adjustments[j] += &result.1[j];
            }
        }

        let number_of_examples = training_data.len() as f32;
        let lr_coef = learning_rate / number_of_examples;
        for i in 0..self.layers.len() {
            self.layers[i].weights = &self.layers[i].weights - (lr_coef * &weight_adjustments[i]);
            self.layers[i].biases = &self.layers[i].biases - (lr_coef * &bias_adjustments[i]);
        }
    }

    ///
    /// Init zeroed adjustment matrices
    ///
    fn init_zeroed_adjustment_matrices(&mut self) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {
        let mut weight_adjustments: Vec<Array2<f32>> = Vec::with_capacity(self.layers.len());
        let mut bias_adjustments: Vec<Array2<f32>> = Vec::with_capacity(self.layers.len());
        for layer in self.layers().iter() {
            weight_adjustments.push(Array2::zeros(layer.weights.dim()));
            bias_adjustments.push(Array2::zeros(layer.biases.dim()));
        }
        return (weight_adjustments, bias_adjustments);
    }

    /// Forward propagate an input vector through the network and evaluate cost of the networks
    /// output with an expected result. Back propagate the test error through the network.
    ///
    /// * `input` - Input vector
    /// * `expected` - Expected output vector
    /// * `returns` - A tuple of vectors. Tuple index 0 is the weight adjustments, tuple index 1
    ///               is the bias adjustments. // TODO make named tuple or struct
    pub fn back_propagate(&self, input: &Array2<f32>, expected: &Array2<f32>) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {

        let mut weight_adjustments: Vec<Array2<f32>> = Vec::with_capacity(self.layers.len());
        let mut bias_adjustments: Vec<Array2<f32>> = Vec::with_capacity(self.layers.len());

        self.back_prop_recursive(0, &input, &expected, &mut weight_adjustments, &mut bias_adjustments);

        return (weight_adjustments, bias_adjustments);
    }

    /// Back propagate recursively
    ///
    /// * `layer_index` - layer index
    /// * `x` - layer activations
    /// * `y` - expected result
    /// * `wam` - weight adjustment matrix
    /// * `bam` - bias adjustment matrix
    /// * `returns` - layer error
    fn back_prop_recursive(&self, layer_index: usize,
                           x: &Array2<f32>,
                           y: &Array2<f32>,
                           wam: &mut Vec<Array2<f32>>,
                           bam: &mut Vec<Array2<f32>>) -> Array2<f32>
    {
        if layer_index == self.layers.len() {
            return self.calculate_cost(x, y);
        }

        let w: &Array2<f32> = &self.layers[layer_index].weights;
        let b: &Array2<f32> = &self.layers[layer_index].biases;
        let z = w.dot(x) + b;
        let result: Array2<f32> = self.non_linearity(&z);

        let error: Array2<f32> = self.back_prop_recursive(layer_index + 1, &result, y, wam, bam);

        let x_prime: Array2<f32> = self.non_linearity_prime(&z);
        let delta = &error * x_prime;
        wam.insert(0, delta.dot(&x.t()));
        bam.insert(0, delta);

        return self.layers[layer_index].weights.clone().t().dot(&error);
    }

    /// Calculate network cost
    ///
    /// * `result` - network output result
    /// * `target` - target network result
    fn calculate_cost(&self, result: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
        return (self.cost_function)(result, target);
    }

    /// Calculate network cost
    ///
    /// * `result` - network output result
    /// * `target` - target network result
    fn calculate_cost_default(result: &Array2<f32>, target: &Array2<f32>) -> Array2<f32> {
        return result - target;
    }

    /// Network non-linearity
    ///
    /// * `x` - array2 to process
    fn non_linearity(&self, x: &Array2<f32>) -> Array2<f32> {
        return (self.activation_expression)(x);
    }

    /// Network non-linearity first derivative
    ///
    /// * `x` - array2 to process
    fn non_linearity_prime(&self, x: &Array2<f32>) -> Array2<f32> {
        return (self.activation_expression_prime)(x);
    }

    ///
    /// Get neural network layers
    ///
    pub fn layers(&self) -> &Vec<NeuralNetworkLayer> {
        &self.layers
    }

    /// Save the current network to a file
    ///
    /// * `path` - File path
    /// * `filename` - Filename. The .gnm file extension will be automatically added if not already set
    pub fn to_file(&self, path: &str, filename: &str) {
        let check_file_result = check_gnm_filepath(path, filename);
        let filepath = match check_file_result {
            Ok(filepath) => filepath,
            Err(error) => panic!("Error saving network to file: {:?}", error)
        };
        to_file(filepath, &self);
    }

    /// Load a neural network instance from a file
    ///
    /// * `path` - File path
    /// * `filename` - Filename. The .gnm file extension will be automatically added if not already set
    pub fn from_file(path: &str, filename: &str) -> Self {
        let check_file_result = check_gnm_filepath(path, filename);
        let filepath = match check_file_result {
            Ok(filepath) => filepath,
            Err(error) => panic!("Error saving network to file: {:?}", error)
        };
        return from_file(filepath);
    }

    ///
    /// Get network activation function
    ///
    pub fn activation_function(&self) -> ActivationFunction {
        self.activation_function
    }
}

impl Display for NeuralNetwork {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = "".to_string();
        write!(s, "Activation Function: {}\n", ActivationFunction::convert_to_string(self.activation_function)).unwrap();
        for (i, layer) in self.layers.iter().enumerate() {
            write!(s, "Layer {} ({}x{})\n", i + 1, layer.weights.shape()[0], layer.weights.shape()[1]).unwrap();
            write!(s, "{}", *layer).unwrap();
        }
        write!(f, "{}", s)
    }
}

pub struct NeuralNetworkLayer {
    weights: Array2<f32>,
    biases: Array2<f32>
}

impl NeuralNetworkLayer {
    /// New neural network layers
    ///
    /// * `inputs` - Number of inputs
    /// * `neurons` - Number of layer neurons
    pub fn new(inputs: usize, neurons: usize) -> Self {
        return NeuralNetworkLayer {
            weights: Array2::zeros((neurons, inputs)),
            biases: Array2::ones((neurons, 1))
        };
    }

    ///
    /// Layer weights
    ///
    pub fn weights(&self) -> &Array2<f32> {
        &self.weights
    }

    ///
    /// Layer weights (mutable)
    ///
    pub fn weights_mut(&mut self) -> &mut Array2<f32> {
        &mut self.weights
    }

    ///
    /// Layer biases
    ///
    pub fn biases(&self) -> &Array2<f32> {
        &self.biases
    }

    ///
    /// Layer biases (mutable)
    ///
    pub fn biases_mut(&mut self) -> &mut Array2<f32> {
        &mut self.biases
    }
}

impl Display for NeuralNetworkLayer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let rows = self.weights.shape()[0];
        let cols = self.weights.shape()[1];
        let mut s = "".to_string();
        for i in 0..rows {
            for j in 0..cols {
                write!(s, "{:2.4}(w{}:{}) ", self.weights[[i, j]], i, j).unwrap();
            }
            write!(s, "| {:2.4}(b{})\n", self.biases[[i, 0]], i).unwrap();
        }
        write!(f, "{}", s)
    }
}


