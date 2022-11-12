
pub mod mlrust {
    use std::fmt::{Display, Formatter};
    use std::ptr::write;
    use ndarray::{Array2};
    use num::pow;
    use std::fmt::Write;
    use crate::{array_utils, ColumnVector};

    pub struct NeuralNetwork {
        input_neurons: usize,
        output_neurons: usize,
        hidden_layer_sizes: Vec<usize>,
        layers: Vec<NeuralNetworkLayer>
    }

    impl NeuralNetwork {
        /// Constructor
        ///
        /// This constructor produces a nerual network with randomly generated weights and biases
        ///
        /// * `input_neurons` - Number of input neurons
        /// * `output_neurons` - Number of output neurons
        /// * `hidden_layer_sizes` - Vector defining how many hidden layers there should be and the
        ///                          size of each hidden layer. An empty vector results in the input
        ///                          neurons being linked directly to the output neurons.
        pub fn new(input_neurons: usize, output_neurons: usize, hidden_layer_sizes: Vec<usize>) -> Self {
            let number_of_hidden_layers: usize = hidden_layer_sizes.len();
            let mut instance = NeuralNetwork {
                input_neurons,
                output_neurons,
                hidden_layer_sizes,
                layers: Vec::with_capacity(number_of_hidden_layers + 1)
            };
            Self::init_network_layers(&mut instance);
            Self::randomize_weights_and_biases(&mut instance);
            return instance;
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
        pub fn randomize_weights_and_biases(instance: &mut NeuralNetwork) {
            for layer in instance.layers.iter_mut() {
                array_utils::randomize_array(layer.weights_mut(), 0.0, 1.0);
                array_utils::randomize_array(layer.biases_mut(), 0.0, 1.0);
            }
        }

        /// Forward propagate a column vector of inputs through the network to calculate a result
        ///
        /// * `inputs` - ColumnVector inputs
        /// * `returns` - ColumnVector outputs
        pub fn feed_forward(self, inputs: ColumnVector) -> ColumnVector {
            let mut activation: Array2<f32> = inputs.get_data().to_owned();
            for layer in self.layers {
                activation = (layer.weights() * activation) + layer.biases();
                array_utils::sig(&mut activation);
            }
            return ColumnVector::from(&activation);
        }

        fn cost_derivative(result: Array2<f32>, expected: Array2<f32>) -> Array2<f32> {
            return result - expected;
        }

        fn non_linearity(data: Array2<f32>) -> Array2<f32> {
            let mut arr: Array2<f32> = data.to_owned();
            array_utils::sig(&mut arr);
            return arr;
        }

        fn d_non_linearity(data: Array2<f32>) -> Array2<f32> {
            let mut arr: Array2<f32> = data.to_owned();
            array_utils::sig_prime(&mut arr);
            return arr;
        }

        fn back_prop(&mut self, input: ColumnVector, expected: ColumnVector) -> (Vec<Array2<f32>>, Vec<Array2<f32>>){
            let number_of_layers: usize = self.layers.len();
            let mut neg_gradient_w: Vec<Array2<f32>> = Vec::with_capacity(number_of_layers);
            let mut neg_gradient_b: Vec<Array2<f32>> = Vec::with_capacity(number_of_layers);
            for (i, layer) in neg_gradient_w.iter_mut().enumerate() {
                *layer = Array2::zeros(self.layers[i].weights().dim());
            }
            for (i, layer) in neg_gradient_b.iter_mut().enumerate() {
                *layer = Array2::zeros(self.layers[i].biases().dim());
            }

            let mut activation: Array2<f32> = input.get_data().to_owned();
            let mut activations: Vec<Array2<f32>> = Vec::with_capacity(number_of_layers);
            let mut zs: Vec<Array2<f32>> = Vec::with_capacity(number_of_layers);
            for layer in self.layers.iter() {
                activation = (layer.weights() * activation.to_owned()) + layer.biases();
                zs.push(activation.to_owned());
                activation = NeuralNetwork::non_linearity(activation);
                activations.push(activation.to_owned());
            }

            let mut delta: Array2<f32> = NeuralNetwork::cost_derivative(activations[activations.len() - 1].to_owned(), expected.get_data().to_owned());
            neg_gradient_b[number_of_layers - 1] = delta.to_owned();
            neg_gradient_w[number_of_layers - 1] = delta.to_owned() * activations[activations.len() - 2].to_owned();

            for i in 2..number_of_layers {
                let z: Array2<f32> = zs[number_of_layers - i].to_owned();
                let sp: Array2<f32> = NeuralNetwork::d_non_linearity(z);
                delta = (self.layers[number_of_layers - i + 1].weights() * delta.to_owned()) * sp;
                neg_gradient_b[number_of_layers - i] = delta.to_owned();
                neg_gradient_w[number_of_layers - i] = delta.to_owned() * activations[number_of_layers - i - 1].to_owned();
            }
            return (neg_gradient_b, neg_gradient_w);
        }

        ///
        /// Calculate the cost of the network
        ///
        pub fn calculate_cost(self, inputs: ColumnVector, expected: ColumnVector) -> (ColumnVector, f32) {
            let result = self.feed_forward(inputs);
            let mut cost: f32 = 0.0;
            for i in 0..result.size() {
                cost += pow(result[i] + expected[i], 2);
            }
            return (result, cost);
        }

        pub fn layers(&self) -> &Vec<NeuralNetworkLayer> {
            &self.layers
        }
    }

    impl Display for NeuralNetwork {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            let mut s = "".to_string();
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
        pub fn new(inputs: usize, neurons: usize) -> Self {
            return NeuralNetworkLayer {
                weights: Array2::zeros((neurons, inputs)),
                biases: Array2::ones((neurons, 1))
            };
        }

        pub fn weights(&self) -> &Array2<f32> {
            &self.weights
        }

        pub fn weights_mut(&mut self) -> &mut Array2<f32> {
            &mut self.weights
        }

        pub fn biases(&self) -> &Array2<f32> {
            &self.biases
        }

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
}

