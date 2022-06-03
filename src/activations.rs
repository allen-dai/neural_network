pub trait Activation {
    fn f_prop(&mut self, input: &Vec<f32>) -> Vec<f32>;
    fn b_prop(&self, output_gradient: &Vec<f32>, learning_rate: f32) -> Vec<f32>;
}

#[derive(Default, Clone)]
pub struct Tanh {
    input: Vec<f32>,
}

impl Tanh {
    fn activation(x: &f32) -> f32 {
        x.tanh()
    }

    fn activation_prime(x: &f32) -> f32 {
        1f32 - x.tanh().powi(2)
    }
}

impl Activation for Tanh {
    fn f_prop(&mut self, input: &Vec<f32>) -> Vec<f32> {
        self.input = input.clone();
        input.iter().map(|i| Self::activation(i)).collect()
    }

    fn b_prop(&self, output_gradient: &Vec<f32>, learning_rate: f32) -> Vec<f32> {
        self.input
            .iter()
            .map(|i| Self::activation_prime(i))
            .zip(output_gradient.iter())
            .map(|(i, og)| i * og)
            .collect()
    }
}
