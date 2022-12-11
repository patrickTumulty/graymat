
#[cfg(test)]
mod tests {
    use graymat::neural_network::NeuralNetwork;

    #[test]
    fn test_network_io() {
        let path = "./";
        let filename = "network_io_test";

        let nn_original = NeuralNetwork::new(5, 10, vec![10, 20, 15]);

        nn_original.to_file(path, filename);

        let nn_loaded = NeuralNetwork::from_file(path, filename);

        assert_eq!(nn_original.layers().len(), nn_loaded.layers().len());

        for i in 0..nn_original.layers().len() {
            assert_eq!(nn_original.layers()[i].weights(), nn_loaded.layers()[i].weights());
            assert_eq!(nn_original.layers()[i].biases(), nn_loaded.layers()[i].biases());
        }
    }
}
