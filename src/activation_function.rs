use crate::activation_function::ActivationFunction::{RELU, SIGMOID, TANH};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ActivationFunction {
    SIGMOID = 1,
    TANH = 2,
    RELU = 3,
}

impl ActivationFunction {
    /// Get activation function from u8
    ///
    /// * `val` - u8 value
    /// * `returns` - Option some if val is matched else None
    pub fn from_u8(val: u8) -> Option<Self> {
        let functions = [ SIGMOID, TANH, RELU ];
        for af in functions {
            if (af as u8) == val {
                return Some(af);
            }
        }
        None
    }

    /// Get the activation function as a string
    ///
    /// * `activation` - activation function
    pub fn convert_to_string(activation: ActivationFunction) -> String {
        match activation {
            SIGMOID => "Sigmoid".to_owned(),
            TANH => "Tanh".to_owned(),
            RELU => "ReLU".to_owned()
        };
        "Unknown".to_owned()
    }
}
