use super::{FOut, Layer};
use rand::thread_rng;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq)]
pub struct ConvolutionLayer {
    input_shape: (usize, usize, usize), // (depth, width, height)
    output_shape: (usize, usize, usize),
    kernel_shape: (usize, usize),
    kernels: Vec<Vec<Vec<Vec<f32>>>>, // [ input_depth [ kernel_depth [ kernel=2d ] ] ] = 4d vec
    biases: Vec<Vec<Vec<f32>>>, // Only 3d vec because u only need one block per depth of kernel,
}

impl ConvolutionLayer {
    pub fn new(input_shape: (usize, usize, usize), kernel_shape: (usize, usize)) -> Self {
        let (input_depth, input_height, input_width) = input_shape;
        let (kernel_depth, kernel_size) = kernel_shape;
        assert!(kernel_size > input_depth || kernel_size > input_width);
        let mut rng = thread_rng();

        let output_shape = (
            kernel_depth,
            input_height - kernel_size + 1,
            input_width - kernel_size + 1,
        );
        let mut kernels = Vec::with_capacity(input_depth);
        let mut biases = Vec::with_capacity(input_depth);
        for _ in 0..input_depth {
            let mut kernel = Vec::with_capacity(kernel_depth);
            for _ in 0..kernel_depth {
                let mut kernel_block = Vec::with_capacity(kernel_size);
                let mut bias_block = Vec::with_capacity(kernel_size);
                for _ in 0..kernel_size {
                    let mut kernel_row = Vec::with_capacity(kernel_size);
                    let mut bias_row = Vec::with_capacity(kernel_size);
                    for _ in 0..kernel_size {
                        kernel_row.push(rng.gen_range(-1f32..1f32));
                        bias_row.push(rng.gen_range(-1f32..1f32));
                    }
                    kernel_block.push(kernel_row);
                    bias_block.push(bias_row);
                }
                biases.push(bias_block);
                kernel.push(kernel_block);
            }
            kernels.push(kernel);
        }

        Self {
            input_shape,
            output_shape,
            kernel_shape,
            kernels,
            biases,
        }
    }
}

#[typetag::serde(name = "ConvolutionLayer")]
impl Layer for ConvolutionLayer {
    fn f_prop(&mut self, input: &Vec<f32>) -> FOut {
        todo!()
    }

    fn b_prop(&mut self, output_gradient: &Vec<f32>, learning_rate: f32) -> Vec<f32> {
        todo!()
    }
}
