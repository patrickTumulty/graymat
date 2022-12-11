use std::fs::File;
use std::io;
use std::io::{ErrorKind, Read, Write};
use std::path::Path;

use ndarray::{Array2};
use crate::neural_network::NeuralNetwork;
use serde::{Serialize, Deserialize};
use crate::utilities::string_utils::copy_string_into_byte_array;

const FILE_HEADER_SIZE_BYTES: u64 = 36;
const LAYER_HEADER_SIZE_BYTES: u64 = 28;
const META_SIZE: usize = 12;
const GRAYMAT_NETWORK_FILE_EXTENSION: &str = ".gnm"; // GrayMat Network Model

#[derive(Debug, Serialize, Deserialize)]
struct FileHeader {
    pub version: u32,
    meta: [u8; META_SIZE],
    pub header_size_bytes: u64,
    pub layer_header_size_bytes: u64,
    pub number_of_layers: u32
}

impl FileHeader {
    pub fn new(number_of_layers: u32) -> Self {
        let mut ca: [u8; META_SIZE] = [0; META_SIZE];
        copy_string_into_byte_array("GrayMay(0_0)", &mut ca);
        return Self {
            version: 0x01_00_00, // v1.00
            meta: ca,
            header_size_bytes: FILE_HEADER_SIZE_BYTES,
            layer_header_size_bytes: LAYER_HEADER_SIZE_BYTES,
            number_of_layers
        };
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct LayerHeader {
    weight_rows: u32,
    weight_cols: u32,
    biases: u32,
    weights_size_bytes: u64,
    biases_size_bytes: u64,
}

impl LayerHeader {
    pub fn new(weight_rows: u32, weight_cols: u32, biases: u32, weights_size_bytes: u64, biases_size_bytes: u64) -> Self {
        return Self {
            weight_rows,
            weight_cols,
            biases,
            weights_size_bytes,
            biases_size_bytes
        };
    }
}

/// Check .gnm filepath. This function verifies that the path exists. It also ensures that the
/// filename has the appropriate file extension.
///
/// # Arguments
/// * `path` - File path
/// * `filename` - Filename. The .gnm file extension will be automatically added if not already set
/// * `returns` - Result string if path exists, else Error
pub fn check_gnm_filepath(path: &str, filename: &str) -> Result<String, io::Error> {
    if Path::new(path).exists() {
        return Err(io::Error::new(ErrorKind::NotFound, "Invalid Filepath: Path does not exist"));
    }

    let mut full_filename = filename.to_owned();
    if !full_filename.ends_with(GRAYMAT_NETWORK_FILE_EXTENSION) {
        full_filename.push_str(GRAYMAT_NETWORK_FILE_EXTENSION);
    }

    let mut full_path = path.to_owned();
    if !full_path.ends_with("/") {
        full_path.push_str("/"); // TODO make platform independent
    }

    full_path.push_str(full_filename.as_str());
    return Ok(full_path);
}

/// Write a NeuralNetwork to a file
/// # Example
/// ```
/// use graymat::neural_network::NeuralNetwork;
/// use graymat::neural_network_io::{check_gnm_filepath, to_file};
///
/// let nn = NeuralNetwork::new(2, 2, vec![2]);
/// let path = "/home";
/// let filename = "test";
/// to_file(check_gnm_filepath(path, filename).unwrap(), &nn);
/// ```
/// # Arguments
/// * `path` - Full filepath with filename and extension
/// * `network` - The NeuralNetwork to save
pub fn to_file(path: String, network: &NeuralNetwork) {

    let mut file = File::create(path).unwrap();

    let file_header = FileHeader::new(network.layers().len() as u32);
    let file_header_bytes = bincode::serialize(&file_header).unwrap();

    file.write(&file_header_bytes).unwrap();

    for layer in network.layers() {

        let weights = layer.weights();
        let biases = layer.biases();
        let serialized_weights = bincode::serialize(&weights.to_owned().into_raw_vec()).unwrap();
        let serialized_biases = bincode::serialize(&biases.to_owned().into_raw_vec()).unwrap();

        let layer_header = LayerHeader::new(weights.shape()[0] as u32,
                                            weights.shape()[1] as u32,
                                            biases.shape()[0] as u32,
                                            serialized_weights.len() as u64,
                                            serialized_biases.len() as u64);
        let layer_header_bytes = bincode::serialize(&layer_header).unwrap();

        file.write(&layer_header_bytes).unwrap();
        file.write(&serialized_weights).unwrap();
        file.write(&serialized_biases).unwrap();
    }
}

/// Read a NeuralNetwork from a file
/// # Example
/// ```
/// use graymat::neural_network::NeuralNetwork;
/// use graymat::neural_network_io::{check_gnm_filepath, from_file, to_file};
///
/// let path = "/home";
/// let filename = "test"; // result /home/test.gnm
/// let nn: NeuralNetwork = from_file(check_gnm_filepath(path, filename).unwrap());
/// ```
/// # Arguments
/// * `path` - Filepath to network file
/// * `returns` - NeuralNetwork
pub fn from_file(path: String) -> NeuralNetwork {

    let mut file = File::open(path).unwrap();

    let mut file_header_buffer: [u8; FILE_HEADER_SIZE_BYTES as usize] = [0; FILE_HEADER_SIZE_BYTES as usize];
    file.read_exact(&mut file_header_buffer).unwrap();
    let file_header: FileHeader = bincode::deserialize(&file_header_buffer).unwrap();

    let mut loaded_weights: Vec<Array2<f32>> = Vec::with_capacity(file_header.number_of_layers as usize);
    let mut loaded_biases: Vec<Array2<f32>> = Vec::with_capacity(file_header.number_of_layers as usize);

    for _i in 0..file_header.number_of_layers {

        let layer_header = load_layer_header(&mut file);

        load_layer_weights(&mut file, &mut loaded_weights, &layer_header);

        load_layer_biases(&mut file, &mut loaded_biases, &layer_header);
    }

    return NeuralNetwork::from(loaded_weights, loaded_biases);
}

/// Load Layer Network
///
/// * `file` - Open file
fn load_layer_header(file: &mut File) -> LayerHeader {
    let mut layer_header_buffer: [u8; LAYER_HEADER_SIZE_BYTES as usize] = [0; LAYER_HEADER_SIZE_BYTES as usize];
    file.read_exact(&mut layer_header_buffer).unwrap();
    let layer_header: LayerHeader = bincode::deserialize(&layer_header_buffer).unwrap();
    layer_header
}

/// Load layer biases
///
/// * `file` - Open file
/// * `loaded_biases` - Vector of biases matrices that are loaded from file
/// * `layer_header` - Layer header
fn load_layer_biases(file: &mut File, loaded_biases: &mut Vec<Array2<f32>>, layer_header: &LayerHeader) {
    let biases_bytes: usize = layer_header.biases_size_bytes as usize;
    let mut biases_buffer: Vec<u8> = vec![0; biases_bytes];
    file.read_exact(&mut biases_buffer).unwrap();

    let biases_shape: [usize; 2] = [layer_header.biases as usize, 1];
    let biases_vec: Vec<f32> = bincode::deserialize(&biases_buffer).unwrap();
    loaded_biases.push(Array2::from_shape_vec(biases_shape, biases_vec).unwrap());
}

/// Load layer weights
///
/// * `file` - Open file
/// * `loaded_weights` - Vector of weights matrices that are loaded from file
/// * `layer_header` - Layer header
fn load_layer_weights(file: &mut File, loaded_weights: &mut Vec<Array2<f32>>, layer_header: &LayerHeader) {
    let weights_bytes: usize = layer_header.weights_size_bytes as usize;
    let mut weights_buffer: Vec<u8> = vec![0; weights_bytes];
    file.read_exact(&mut weights_buffer).unwrap();

    let weights_shape: [usize; 2] = [layer_header.weight_rows as usize, layer_header.weight_cols as usize];
    let weights_vec: Vec<f32> = bincode::deserialize(&weights_buffer).unwrap();
    loaded_weights.push(Array2::from_shape_vec(weights_shape, weights_vec).unwrap());
}
