use serde::{Deserialize, Serialize};

pub mod convolution;
pub mod dense;

// Forward prop output
pub enum FOut {
    Conv(Vec<Vec<Vec<f32>>>),
    Dense(Vec<f32>),
}

#[derive(Serialize, Deserialize)]
pub enum LayerType {
    Dense(dense::DenseLayer),
    Conv(convolution::ConvolutionLayer),
}
