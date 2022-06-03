use crate::layer::Layer;

use rand::thread_rng;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub input: Vec<f32>,
    pub output: Vec<f32>,
    pub weights: Vec<Vec<f32>>, //vec[nth neuron][weights]
    pub biases: Vec<f32>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut rng = thread_rng();
        for _y in 0..output_size {
            let mut temp = Vec::new();
            for _x in 0..input_size {
                temp.push(rng.gen_range(0f32..1f32));
            }
            weights.push(temp);
        }
        for _y in 0..input_size {
            biases.push(rng.gen_range(0f32..1f32));
        }
        thread_rng().try_fill(&mut biases[..]).unwrap();

        DenseLayer {
            input: vec![0f32; input_size],
            output: vec![0f32; input_size],
            weights,
            biases,
        }
    }
}
impl Layer for DenseLayer {
    fn f_prop(&mut self, input: &Vec<f32>) -> Vec<f32> {
        //assert_eq!(self.input.len(), input.len());

        self.input = input.clone();
        self.weights
            .iter()
            //y.j = x.i * w.j.i
            .map(|neuron| neuron.iter().zip(input).fold(0f32, |p, (w, i)| p + w * i))
            .zip(self.biases.iter())
            //y.j += b.j
            .map(|(w, b)| w + b)
            .collect()
    }

    fn b_prop(&mut self, output_gradient: &Vec<f32>, learning_rate: f32) -> Vec<f32> {
        // dot product of output_gradient vec[_] * vec[input]
        let weight_grad: Vec<f32> = self
            .input
            .iter()
            .zip(output_gradient.iter())
            .map(|(i, og)| i * og)
            .collect();

        let weight_t: Vec<Vec<f32>> = (0..self.weights[0].len())
            .map(|i| self.weights.iter().map(|inner| inner[i]).collect())
            .collect();

        // dot product of vec[neurons][weights].t * vec[out_grad]
        let input_grad: Vec<f32> = weight_t
        .iter()
        .map(|neuron| {
            neuron
                .iter()
                .zip(output_gradient.iter())
                .fold(0f32, |p, (w, og)| p + w * og)
        } / neuron.len() as f32)
        .collect();

        self.weights = self
            .weights
            .iter()
            .map(|neuron| {
                neuron
                    .iter()
                    .zip(weight_grad.iter())
                    .map(|(w, g)| w - g * learning_rate)
                    .collect()
            })
            .collect();

        self.biases = self
            .biases
            .iter()
            .zip(output_gradient.iter())
            .map(|(b, og)| b - og * learning_rate)
            .collect();

        input_grad
    }
}
