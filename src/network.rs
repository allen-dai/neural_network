use crate::activations::Activation;
use crate::layer::Layer;
use crate::loss::Loss;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufReader, Read, Write};
use std::path::Path;

#[derive(Serialize, Deserialize, PartialEq)]
pub struct Network<T, U> {
    layers: Vec<T>,
    activations: Vec<U>,
}

impl<'a, T, U> Network<T, U>
where
    T: Layer + Serialize + Deserialize<'a>,
    U: Activation + Serialize + Deserialize<'a>,
{
    pub fn new(layers: Vec<T>, activations: Vec<U>) -> Self {
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
        let serialized: Vec<u8> = bincode::serialize(self.clone())?;
        file.write_all(&serialized)?;
        Ok(())
    }

    pub fn load_from_file(&mut self, bytes: &'a Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
        //TODO Type infer - instead of needing a '&mut self', return Self.
        //I don't know how to implement this.
        let deserialized: Self = bincode::deserialize(bytes)?;
        self.layers = deserialized.layers;
        self.activations = deserialized.activations;
        Ok(())
    }
}
