use serde::{Deserialize, Deserializer};

pub mod dense;
pub mod convolution;

// Forward prop output
pub enum FOut {
    Conv(Vec<Vec<Vec<f32>>>),
    Dense(Vec<f32>)
}

#[typetag::serde(tag = "type", content = "value")]
pub trait Layer {
    fn f_prop(&mut self, input: &Vec<f32>) -> FOut;
    fn b_prop(&mut self, output_gradient: &Vec<f32>, learning_rate: f32) -> Vec<f32>;
}

