
use graymat::neural_network::NeuralNetwork;
use graymat::neural_network_io::{check_gnm_filepath, GRAYMAT_NETWORK_FILE_EXTENSION};

#[test]
fn test_network_io() {
    let path = "./";
    let filename = "network_io_test";

    let nn_original = NeuralNetwork::new(100, 10, vec![25, 20, 25]);

    nn_original.to_file(path, filename);

    let nn_loaded = NeuralNetwork::from_file(path, filename);

    assert_eq!(nn_original.layers().len(), nn_loaded.layers().len());

    for i in 0..nn_original.layers().len() {
        assert_eq!(nn_original.layers()[i].weights(), nn_loaded.layers()[i].weights());
        assert_eq!(nn_original.layers()[i].biases(), nn_loaded.layers()[i].biases());
    }
}

#[test]
fn test_filepath_handling1() {
    let path = "./";
    let filename = "network_io_test";
    let result = check_gnm_filepath(path, filename);
    let filepath = match result {
        Ok(filepath) => filepath,
        Err(err) => panic!("{:?}", err)
    };

    assert_eq!(filepath.ends_with(GRAYMAT_NETWORK_FILE_EXTENSION), true);
}

#[test]
fn test_filepath_handling2() {
    let path = ".";
    let filename = "network_io_test.gnm";
    let result = check_gnm_filepath(path, filename);
    let filepath = match result {
        Ok(filepath) => filepath,
        Err(err) => panic!("{:?}", err)
    };

    assert_eq!(filepath.ends_with("/network_io_test.gnm"), true);
}

#[test]
#[should_panic]
fn test_filepath_handling_error() {
    let path = "./does_not_exists";
    let filename = "network_io_test";
    let result = check_gnm_filepath(path, filename);
    let _filepath = match result {
        Ok(filepath) => filepath,
        Err(err) => panic!("{:?}", err)
    };
}

