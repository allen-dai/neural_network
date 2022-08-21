use crate::activations::Activation;
use crate::layer::Layer;
use crate::loss::Loss;
use std::fs;
use std::io::{BufReader, Read, Write};
use std::path::Path;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Network {
    layers: Vec<Box<dyn Layer>>,
    activations: Vec<Box<dyn Activation>>,
}

impl<'a> Network {
    pub fn new(layers: Vec<Box<dyn Layer>>, activations: Vec<Box<dyn Activation>>) -> Self {
        Network {
            layers,
            activations,
        }
    }

    pub fn predict(&mut self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = input.clone();
        for (layer, activation_fn) in self.layers.iter_mut().zip(self.activations.iter_mut()) {
            output = layer.f_prop(&output);
            output = activation_fn.f_prop(&output);
        }
        output
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

                for (layer, activation_fn) in self
                    .layers
                    .iter_mut()
                    .zip(self.activations.iter_mut())
                    .rev()
                {
                    gradient = activation_fn.b_prop(&gradient);
                    gradient = layer.b_prop(&gradient, learning_rate);
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

    pub fn load_from_file(&mut self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let mut bytes = Vec::new();
        let _ = {
            let mut f = fs::File::open(path)?;
            let _ = f.read_to_end(&mut bytes)?;
            &bytes[..]
        };
        let cbor = &mut serde_cbor::Deserializer::from_slice(&bytes);
        let mut erased = Box::new(<dyn erased_serde::Deserializer>::erase(cbor));
        let network: Self = erased_serde::deserialize(erased.as_mut())?;
        *self = network;
        Ok(())
    }
}
