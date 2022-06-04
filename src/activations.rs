pub trait Activation {
    fn f_prop(&mut self, input: &Vec<f32>) -> Vec<f32>;
    fn b_prop(&self, output_gradient: &Vec<f32>, learning_rate: f32) -> Vec<f32>;
}

#[derive(Default, Clone)]
pub struct Tanh {
    input: Vec<f32>,
}

impl Tanh {
    fn tanh(x: &f32) -> f32 {
        x.tanh()
    }

    // derivative
    fn tanh_prime(x: &f32) -> f32 {
        1f32 - x.tanh().powi(2)
    }
}

impl Activation for Tanh {
    fn f_prop(&mut self, input: &Vec<f32>) -> Vec<f32> {
        self.input = input.to_vec();
        input.iter().map(|i| Self::tanh(i)).collect()
    }

    fn b_prop(&self, output_gradient: &Vec<f32>, learning_rate: f32) -> Vec<f32> {
        assert_eq!(output_gradient.len(), self.input.len());
        self.input
            .iter()
            .zip(output_gradient.iter())
            .map(|(i, og)| Self::tanh_prime(i) * og)
            .collect()
    }
}

#[derive(Default, Clone)]
pub struct Sigmoid {
    input: Vec<f32>,
}

impl Sigmoid {
    fn sigmoid(x: &f32) -> f32 {
        1f32 / (1f32 + f32::exp(-x))
    }

    // derivative
    fn sigmoid_prime(x: &f32) -> f32 {
        Self::sigmoid(x) * (1f32 - Self::sigmoid(x))
    }
}

impl Activation for Sigmoid {
    fn f_prop(&mut self, input: &Vec<f32>) -> Vec<f32> {
        self.input = input.to_vec();
        input.iter().map(|i| Self::sigmoid(i)).collect()
    }

    fn b_prop(&self, output_gradient: &Vec<f32>, learning_rate: f32) -> Vec<f32> {
        assert_eq!(output_gradient.len(), self.input.len());
        self.input
            .iter()
            .zip(output_gradient.iter())
            .map(|(i, og)| Self::sigmoid_prime(i) * og)
            .collect()
    }
}
