pub mod dense;
pub mod convolution;

pub trait Layer : erased_serde::Serialize  {
    fn f_prop(&mut self, input: &Vec<f32>) -> Vec<f32>;
    fn b_prop(&mut self, output_gradient: &Vec<f32>, learning_rate: f32) -> Vec<f32>;
}

erased_serde::serialize_trait_object!(Layer);
