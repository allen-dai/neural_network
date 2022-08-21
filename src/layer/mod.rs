use serde::{Deserialize, Deserializer};

pub mod dense;
pub mod convolution;

#[typetag::serde(tag = "type", content = "value")]
pub trait Layer {
    fn f_prop(&mut self, input: &Vec<f32>) -> Vec<f32>;
    fn b_prop(&mut self, output_gradient: &Vec<f32>, learning_rate: f32) -> Vec<f32>;
}
