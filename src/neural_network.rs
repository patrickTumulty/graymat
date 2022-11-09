
pub mod mlrust {
    use ndarray::{Array2};
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
        pub fn init_network(&mut self) {
            for layer in self.layers.iter_mut() {
                array_utils::randomize_array(layer.connection_weights_mut(), 0.0, 1.0);
            }
         }

        /// Evaluate inputs
        ///
        /// * `inputs` - ColumnVector inputs
        /// * `returns` - ColumnVector outputs
        pub fn evaluate(self, inputs: ColumnVector) -> ColumnVector {
            let mut prop: Array2<f32> = inputs.get_data().to_owned();
            for mut layer in self.layers {
                prop = array_utils::add(&layer.connection_weights().dot(&prop), layer.neuron_biases());
                array_utils::sig(&mut prop);
            }
            return ColumnVector::from(&prop);
        }


        pub fn layers(&self) -> &Vec<NeuralNetworkLayer> {
            &self.layers
        }
    }

    pub struct NeuralNetworkLayer {
        connection_weights: Array2<f32>,
        neuron_biases: Array2<f32>
    }

    impl NeuralNetworkLayer {
        pub fn new(inputs: usize, neurons: usize) -> Self {
            return NeuralNetworkLayer {
                connection_weights: Array2::zeros((neurons, inputs)),
                neuron_biases: Array2::zeros((neurons, 1))
            };
        }

        pub fn connection_weights(&self) -> &Array2<f32> {
            &self.connection_weights
        }

        pub fn connection_weights_mut(&mut self) -> &mut Array2<f32> {
            &mut self.connection_weights
        }

        pub fn neuron_biases(&mut self) -> &mut Array2<f32> {
            &mut self.neuron_biases
        }

    }
}

