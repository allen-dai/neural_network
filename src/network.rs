use crate::activations::Activation;
use crate::layer::Layer;
use crate::loss::Loss;

pub struct Network<'a, T, U> where T: Layer, U: Activation {
    layers: &'a mut Vec<T>,
    activations: &'a mut Vec<U>,
}

impl<'a, T, U> Network<'a, T, U> where T: Layer, U: Activation{
    pub fn new(layers: &'a mut Vec<T>, activations: &'a mut Vec<U>) -> Self {
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
                for (layer, activation_fn) in
                    self.layers.iter_mut().zip(self.activations.iter_mut())
                {
                    gradient = activation_fn.b_prop(&gradient, learning_rate);
                    gradient = layer.b_prop(&gradient, learning_rate);
                }
            }

            if verbose {
                println!("epoch: {} | loss: {}", i, loss / train_set.len() as f32);
            }
        }
    }
}