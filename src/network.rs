use crate::activations::{Activation, ActivationFn};
use crate::layer::{LayerOutput, LayerType};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Serialize, Deserialize, Clone)]
pub enum Net {
    Layer(LayerType),
    Activation(ActivationFn),
}

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct Network {
    pub(crate) layers: Vec<LayerType>,
    pub(crate) activations: Vec<ActivationFn>,
}

impl Network {
    pub fn new(net: Vec<Net>) -> Self {
        let mut layers = Vec::new();
        let mut activations = Vec::new();
        for n in net {
            match n {
                Net::Layer(layer) => layers.push(layer),
                Net::Activation(activation) => activations.push(activation),
            }
        }
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
                        LayerOutput::Dense(v) => {
                            output = layer.f_prop(&v);
                        }
                        _ => unreachable!(),
                    }
                }
                LayerType::Conv(layer) => match output {
                    LayerOutput::Conv(v) => output = layer.f_prop(&v),
                    LayerOutput::Dense(v) => output = layer.f_prop(&vec![v]),
                    _ => unreachable!(),
                },
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

        match output {
            LayerOutput::Conv(_) | LayerOutput::None => {
                unreachable!("Last layer need to be a dense layer")
            }
            LayerOutput::Dense(prediction) => prediction,
        }
    }


    pub fn predict_ref(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = LayerOutput::Dense(input.clone());
        for (layer_type, activation_fn) in self.layers.iter().zip(self.activations.iter()) {
            match layer_type {
                LayerType::Dense(layer) => {
                    match output {
                        // currenty just flattening the vector. not sure if this is the proper way
                        // to do it.
                        LayerOutput::Conv(v) => {
                            output = layer.f_prop_ref(&v.into_iter().flatten().collect());
                        }
                        LayerOutput::Dense(v) => {
                            output = layer.f_prop_ref(&v);
                        }
                        _ => unreachable!(),
                    }
                }
                LayerType::Conv(layer) => match output {
                    LayerOutput::Conv(v) => output = layer.f_prop_ref(&v),
                    LayerOutput::Dense(v) => output = layer.f_prop_ref(&vec![v]),
                    _ => unreachable!(),
                },
            }

            match activation_fn {
                ActivationFn::Tanh(tanh) => {
                    output = tanh.f_prop_ref(&output);
                }
                ActivationFn::Sigmoid(sigmoid) => {
                    output = sigmoid.f_prop_ref(&output);
                }
                ActivationFn::Relu(relu) => {
                    output = relu.f_prop_ref(&output);
                }
            }
        }

        match output {
            LayerOutput::Conv(_) | LayerOutput::None => {
                unreachable!("Last layer need to be a dense layer")
            }
            LayerOutput::Dense(prediction) => prediction,
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

    pub fn from_slice(&mut self, slice: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        let network: Network = serde_cbor::from_reader(slice)?;
        *self = network;
        Ok(())
    }
}
