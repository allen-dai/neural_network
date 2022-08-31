use super::layer::LayerOutput;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum ActivationFn {
    Tanh(Tanh),
    Sigmoid(Sigmoid),
    Relu(Relu),
}

pub trait Activation {
    fn activation(&self, x: f32) -> f32;
    fn derivative(&self, x: f32) -> f32;
    fn set_input(&mut self, input: LayerOutput);
    fn get_input(&self) -> &LayerOutput;

    fn f_prop(&mut self, layer_out: &LayerOutput) -> LayerOutput {
        match layer_out {
            LayerOutput::Conv(input) => {
                self.set_input(LayerOutput::Conv(input.clone()));
                LayerOutput::Conv(
                    input
                        .iter()
                        .map(|row| row.iter().map(|i| self.activation(*i)).collect())
                        .collect(),
                )
            }
            LayerOutput::Dense(input) => {
                self.set_input(LayerOutput::Dense(input.clone()));
                LayerOutput::Dense(input.iter().map(|i| self.activation(*i)).collect())
            }
            LayerOutput::None => unreachable!(),
        }
    }

    fn b_prop(&self, output_gradient: &[f32]) -> Vec<f32> {
        match self.get_input() {
            LayerOutput::Conv(_) => todo!(),
            LayerOutput::Dense(input) => input
                .iter()
                .zip(output_gradient.iter())
                .map(|(i, og)| self.derivative(*i) * og)
                .collect(),
            LayerOutput::None => todo!(),
        }
    }
}

#[derive(Default, Serialize, Deserialize, PartialEq)]
pub struct Tanh {
    input: LayerOutput,
}

impl Activation for Tanh {
    fn activation(&self, x: f32) -> f32 {
        x.tanh()
    }

    fn derivative(&self, x: f32) -> f32 {
        1f32 - x.tanh().powi(2)
    }

    fn set_input(&mut self, input: LayerOutput) {
        self.input = input;
    }

    fn get_input(&self) -> &LayerOutput {
        &self.input
    }
}

#[derive(Default, Serialize, Deserialize, PartialEq)]
pub struct Sigmoid {
    input: LayerOutput,
}

impl Activation for Sigmoid {
    fn activation(&self, x: f32) -> f32 {
        1f32 / (1f32 + f32::exp(-x))
    }

    fn derivative(&self, x: f32) -> f32 {
        self.activation(x) * (1f32 - self.activation(x))
    }

    fn set_input(&mut self, input: LayerOutput) {
        self.input = input;
    }

    fn get_input(&self) -> &LayerOutput {
        &self.input
    }
}

#[derive(Default, Serialize, Deserialize, PartialEq)]
pub struct Relu {
    input: LayerOutput,
}

impl Activation for Relu {
    fn activation(&self, x: f32) -> f32 {
        f32::max(0f32, x)
    }

    fn derivative(&self, x: f32) -> f32 {
        if x > 0f32 {
            return 1f32;
        }
        return 0f32;
    }

    fn set_input(&mut self, input: LayerOutput) {
        self.input = input;
    }

    fn get_input(&self) -> &LayerOutput {
        &self.input
    }
}
