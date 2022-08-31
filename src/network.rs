use crate::activations::{Activation, ActivationFn};
use crate::layer::{LayerOutput, LayerType};
use crate::loss::Loss;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Serialize, Deserialize, Default)]
pub struct Network {
    layers: Vec<LayerType>,
    activations: Vec<ActivationFn>,
}

impl Network {
    pub fn new(layers: Vec<LayerType>, activations: Vec<ActivationFn>) -> Self {
        Network {
            layers,
            activations,
        }
    }

    pub fn predict(&mut self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = LayerOutput::Dense(input.clone());
        for (layer_type, activation_fn) in self.layers.iter_mut().zip(self.activations.iter_mut()) {
            match layer_type {
                LayerType::Dense(layer) => {
                    match output {
                        // currenty just flattening the vector. not sure if this is the proper way
                        // to do it.
                        LayerOutput::Conv(v) => {
                            output = layer.f_prop(&v.into_iter().flatten().collect());
                        }
                        LayerOutput::Dense(v) => output = layer.f_prop(&v),
                        LayerOutput::None => unreachable!(),
                    }
                    match activation_fn {
                        ActivationFn::Tanh(tanh) => {
                            output = tanh.f_prop(&output);
                        }
                        ActivationFn::Sigmoid(sigmoid) => {
                            output = sigmoid.f_prop(&output);
                        }
                        ActivationFn::Relu(relu) => {
                            output = relu.f_prop(&output);
                        }
                    }
                }
                LayerType::Conv(layer) => match output {
                    LayerOutput::Conv(v) => output = layer.f_prop(&v),
                    LayerOutput::Dense(v) => output = layer.f_prop(&vec![v]),
                    LayerOutput::None => unreachable!(),
                },
            }
        }

        match output {
            LayerOutput::Conv(_) | LayerOutput::None => unreachable!(),
            LayerOutput::Dense(prediction) => prediction,
        }
    }

    pub fn train<L: Loss>(
        &mut self,
        loss_fn: L,
        train_set: &Vec<Vec<f32>>,
        train_answer: &Vec<Vec<f32>>,
        learning_rate: f32,
        epoch: usize,
        verbose: bool,
    ) {
        let (mut loss, mut output, mut gradient);
        for i in 0..epoch {
            loss = 0f32;
            for (x, y) in train_set.iter().zip(train_answer.iter()) {
                output = self.predict(&x);

                loss += loss_fn.loss(&y, &output);
                gradient = loss_fn.loss_prime(&y, &output);

                for (layer_type, activation_fn) in self
                    .layers
                    .iter_mut()
                    .zip(self.activations.iter_mut())
                    .rev()
                {
                    match activation_fn {
                        ActivationFn::Tanh(_) => todo!(),
                        ActivationFn::Sigmoid(sigmoid) => {
                            gradient = sigmoid.b_prop(&gradient);
                        }
                        ActivationFn::Relu(_) => todo!(),
                    }
                    match layer_type {
                        LayerType::Dense(layer) => {
                            gradient = layer.b_prop(&gradient, learning_rate);
                        }
                        LayerType::Conv(_) => todo!(),
                    }
                }
            }

            if verbose {
                println!("epoch: {} | loss: {}", i, loss / train_set.len() as f32);
            }
        }
    }

    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = fs::File::create(path)?;
        let serialized: Vec<u8> = serde_cbor::to_vec(&self)?;
        file.write_all(&serialized)?;
        Ok(())
    }

    pub fn load_from_file(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let f = fs::File::open(path)?;
        let network: Network = serde_cbor::from_reader(f)?;
        *self = network;
        Ok(())
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let f = fs::File::open(path)?;
        let network: Network = serde_cbor::from_reader(f)?;
        Ok(network)
    }
}
