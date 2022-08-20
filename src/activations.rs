use serde::{Deserialize, Serialize};

pub trait Activation : erased_serde::Serialize {
    fn activation(&self, x: f32) -> f32;
    fn derivative(&self, x: f32) -> f32;
    fn set_input(&mut self, input: &[f32]);
    fn get_input(&self) -> &Vec<f32>;

    fn f_prop(&mut self, input: &[f32]) -> Vec<f32> {
        self.set_input(&input.to_vec());
        input.iter().map(|i| self.activation(*i)).collect()
    }

    fn b_prop(&self, output_gradient: &[f32]) -> Vec<f32> {
        self.get_input()
            .iter()
            .zip(output_gradient.iter())
            .map(|(i, og)| self.derivative(*i) * og)
            .collect()
    }
}

erased_serde::serialize_trait_object!(Activation);

#[derive(Default, Serialize, Deserialize, PartialEq)]
pub struct Tanh {
    input: Vec<f32>,
}
impl Activation for Tanh {
    fn activation(&self, x: f32) -> f32 {
        x.tanh()
    }

    fn derivative(&self, x: f32) -> f32 {
        1f32 - x.tanh().powi(2)
    }

    fn set_input(&mut self, input: &[f32]) {
        self.input = input.to_vec();
    }

    fn get_input(&self) -> &Vec<f32> {
        &self.input
    }
}

#[derive(Default, Serialize, Deserialize, PartialEq)]
pub struct Sigmoid {
    input: Vec<f32>,
}
impl Activation for Sigmoid {
    fn activation(&self, x: f32) -> f32 {
        1f32 / (1f32 + f32::exp(-x))
    }

    fn derivative(&self, x: f32) -> f32 {
        self.activation(x) * (1f32 - self.activation(x))
    }

    fn set_input(&mut self, input: &[f32]) {
        self.input = input.to_vec();
    }

    fn get_input(&self) -> &Vec<f32> {
        &self.input
    }
}

#[derive(Default, Serialize, Deserialize, PartialEq)]
pub struct Relu {
    input: Vec<f32>,
}
impl Activation for Relu {
    fn activation(&self, x: f32) -> f32 {
        f32::max(0f32, x)
    }

    fn derivative(&self, x: f32) -> f32 {
        if x > 0f32 {
            return 1f32
        }
        return 0f32
    }

    fn set_input(&mut self, input: &[f32]) {
        self.input = input.to_vec();
    }

    fn get_input(&self) -> &Vec<f32> {
        &self.input
    }
}
