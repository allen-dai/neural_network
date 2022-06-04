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
        assert_eq!(self.input.len(), input.len());

        //println!("{} {}", self.biases.len(), self.weights[0].len());
        self.input = input.clone();
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
            .collect()
    }

    fn b_prop(&mut self, output_gradient: &Vec<f32>, learning_rate: f32) -> Vec<f32> {
        // dot product of output_gradient vec[_] * vec[input]
        let mut weight_grad: Vec<Vec<f32>> = Vec::new();
        for r in output_gradient.iter() {
            let mut temp = Vec::new();
            for i in self.input.iter() {
                temp.push(i * r);
            }
            weight_grad.push(temp);
        }

        let mut weight_t: Vec<Vec<f32>> = Vec::new();
        for col in 0..self.weights[0].len() {
            let mut temp = Vec::new();
            for row in self.weights.iter() {
                temp.push(row[col]);
            }
            weight_t.push(temp);
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
