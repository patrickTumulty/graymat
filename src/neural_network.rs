
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

        fn init_network_layers(instance: &mut NeuralNetwork) {
            let mut layer_inputs = instance.input_neurons;
            for layer_size in &instance.hidden_layer_sizes {
                instance.layers.push(NeuralNetworkLayer::new(layer_inputs, *layer_size));
                layer_inputs = *layer_size;
            }
            instance.layers.push(NeuralNetworkLayer::new(layer_inputs, instance.output_neurons));
        }

        pub fn evaluate(self, inputs: ColumnVector) -> ColumnVector {
            let mut prop: Array2<f32> = inputs.get_data().to_owned();
            for layer in self.layers {
                prop = array_utils::add(&layer.connection_weights().dot(&prop), layer.neuron_biases());
                array_utils::sig(&mut prop);
            }
            return ColumnVector::from(&prop);
        }
    }

    struct NeuralNetworkLayer {
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

        pub fn neuron_biases(&self) -> &Array2<f32> {
            &self.neuron_biases
        }
    }
}

