
pub mod mlrust {
    use std::fmt::{Display, Formatter};

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

        pub fn from(weights: Vec<Array2<f32>>, biases: Vec<Array2<f32>>) -> Self {
            assert_eq!(weights.len(), biases.len());
            let number_of_hidden_layers: usize = weights.len();
            let mut instance = NeuralNetwork {
                input_neurons: weights[0].dim().1,
                output_neurons: weights[weights.len() - 1].dim().1,
                hidden_layer_sizes: Vec::with_capacity(number_of_hidden_layers),
                layers: Vec::with_capacity(number_of_hidden_layers)
            };
            for i in 0..instance.layers.capacity() {
                instance.layers.push(NeuralNetworkLayer {
                    weights: weights[i].clone(),
                    biases: biases[i].clone()
                })
            }
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
                // array_utils::sig(&mut activation);
            }
            return ColumnVector::from(&activation);
        }

        pub fn back_propagate(&mut self, input: ColumnVector, expected: ColumnVector) -> (Vec<Array2<f32>>, Vec<Array2<f32>>) {

            let mut activation: Array2<f32> = input.get_data().clone();
            let mut layer_activations: Vec<Array2<f32>> = Vec::with_capacity(self.layers.len());
            layer_activations.push(activation.clone());

            for layer in self.layers.iter() {
                activation = array_utils::math::sig(&(layer.weights.dot(&activation) + &layer.biases));
                layer_activations.push(activation.clone());
            }

            let mut layer_errors: Vec<Array2<f32>> = Vec::with_capacity(self.layers.len());
            layer_errors.push((expected.get_data() - &activation).clone());
            let mut j = self.layers.len() - 1;
            for i in 1..layer_errors.capacity() {
                layer_errors.insert(0, self.layers[j].weights.clone().t().dot(&layer_errors[i - 1]));
                j -= 1;
            }

            let mut weight_adjustments: Vec<Array2<f32>> = Vec::with_capacity(self.layers.len());
            let mut bias_adjustments: Vec<Array2<f32>> = Vec::with_capacity(self.layers.len());

            for i in (1..layer_activations.len()).rev() {
                let layer_output = &layer_activations[i];
                let layer_input = &layer_activations[i - 1];
                let x_prime = layer_output * (1.0 - layer_output);
                weight_adjustments.insert(0, (&layer_errors[i - 1] * &x_prime).dot(&layer_input.clone().t()));
                bias_adjustments.insert(0, &layer_errors[i - 1] * &x_prime);
            }

            return (weight_adjustments, bias_adjustments);
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

