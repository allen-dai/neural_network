use serde::{Deserialize, Serialize};

pub mod convolution;
pub mod dense;

// Forward prop output
#[derive(Default, Debug, Serialize, Deserialize, PartialEq)]
pub enum LayerOutput {
    Conv(Vec<Vec<f32>>),
    Dense(Vec<f32>),
    #[default]
    None
}

#[derive(Serialize, Deserialize)]
pub enum LayerType {
    Dense(dense::DenseLayer),
    Conv(convolution::ConvolutionLayer),
}
