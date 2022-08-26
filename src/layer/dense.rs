use super::FOut;
use rand::thread_rng;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq)]
pub struct DenseLayer {
    pub input: Vec<f32>,
    pub weights: Vec<Vec<f32>>, //vec[nth neuron][weights]
    pub biases: Vec<f32>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut weights = Vec::with_capacity(input_size * output_size);
        let mut biases = Vec::with_capacity(output_size);
        let mut rng = thread_rng();
        let mut temp: Vec<f32> = vec![0f32; input_size];
        for _y in 0..output_size {
            for _x in 0..input_size {
                temp[_x] = rng.gen_range(-1f32..1f32);
            }
            weights.push(temp.clone());
        }
        for _y in 0..output_size {
            biases.push(rng.gen_range(-1f32..1f32));
        }
        thread_rng().try_fill(&mut biases[..]).unwrap();

        Self {
            input: vec![0f32; input_size],
            weights,
            biases,
        }
    }

    pub fn f_prop(&mut self, input: &Vec<f32>) -> FOut {
        assert_eq!(self.input.len(), input.len());

        //println!("{} {}", self.biases.len(), self.weights[0].len());
        self.input = input.to_vec();
        FOut::Dense(
            self.weights
                .iter()
                //y.j = x.i * w.j.i
                .map(|neuron| {
                    neuron
                        .iter()
                        .zip(input.iter())
                        .fold(0f32, |p, (w, i)| p + w * i)
                })
                //y.j += b.j
                .zip(self.biases.iter())
                .map(|(o, b)| o + b)
                .collect(),
        )
    }

    pub fn b_prop(&mut self, output_gradient: &Vec<f32>, learning_rate: f32) -> Vec<f32> {
        // dot product of output_gradient vec[_] * vec[input]
        let mut weight_grad: Vec<Vec<f32>> =
            Vec::with_capacity(output_gradient.len() * self.input.len());
        let mut temp: Vec<f32> = vec![0f32; self.input.len()];
        for r in output_gradient.iter() {
            for (i, c) in self.input.iter().enumerate() {
                temp[i] = c * r;
            }
            weight_grad.push(temp.clone());
        }

        let mut weight_t: Vec<Vec<f32>> =
            Vec::with_capacity(self.weights[0].len() * self.weights.len());
        let mut temp: Vec<f32> = vec![0f32; self.weights.len()];
        for col in 0..self.weights[0].len() {
            for (i, row) in self.weights.iter().enumerate() {
                temp[i] = row[col];
            }
            weight_t.push(temp.clone());
        }

        // dot product of vec[neurons][weights].t * vec[out_grad]
        let input_grad: Vec<f32> = weight_t
            .iter()
            .map(|row| {
                row.iter()
                    .zip(output_gradient.iter())
                    .fold(0f32, |p, (w, og)| p + w * og)
            })
            .collect();

        self.weights = self
            .weights
            .iter()
            .zip(weight_grad.iter())
            .map(|(neuron, nwg)| {
                neuron
                    .iter()
                    .zip(nwg.iter())
                    .map(|(w, wg)| w - wg * learning_rate)
                    .collect()
            })
            .collect();

        self.biases = self
            .biases
            .iter()
            .zip(output_gradient)
            .map(|(b, og)| b - og * learning_rate)
            .collect();

        input_grad
    }
}
