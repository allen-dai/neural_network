pub mod dense;

pub trait Layer{
    fn f_prop(&mut self, input: &Vec<f32>) -> Vec<f32>;
    fn b_prop(&mut self, output_error: &Vec<f32>, learning_rate: f32) -> Vec<f32>;
}
