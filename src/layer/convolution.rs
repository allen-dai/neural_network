use super::LayerOutput;
use rand::thread_rng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Serialize, Deserialize, PartialEq)]
pub struct ConvolutionLayer {
    input: Vec<Vec<f32>>,
    input_shape: (usize, usize, usize), // (depth, width, height)
    output_shape: (usize, usize, usize),
    kernel_shape: (usize, usize), // (depth, size)
    kernels: Vec<Vec<Vec<f32>>>,  // [ input_depth [ kernel_depth [ kernel ] ] ] = 3d vec
    biases: Vec<Vec<f32>>,        // 2d vec because u only need one block per depth of kernel,
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
                for _ in 0..kernel_size * kernel_size {
                    kernel_block.push(rng.gen_range(-1f32..1f32));
                    bias_block.push(rng.gen_range(-1f32..1f32));
                }
                biases.push(bias_block);
                kernel.push(kernel_block);
            }
            kernels.push(kernel);
        }
        Self {
            input: vec![],
            input_shape,
            output_shape,
            kernel_shape,
            kernels,
            biases,
        }
    }

    pub fn f_prop(&mut self, input: &Vec<Vec<f32>>) -> LayerOutput {
        self.input = input.clone();
        let mut out: Vec<Vec<f32>> =
            vec![vec![0f32; self.output_shape.1 * self.output_shape.2]; self.output_shape.0];
        let kernel_size = self.kernel_shape.1;
        let chunks = Self::correlation_chunks(&input, &self.input_shape, kernel_size);
        for (depth, (kernel, chunk)) in self.kernels.iter().zip(chunks).enumerate() {
            for block in kernel {
                // chunk is all the movements ( all the slidings ) for the current kernel
                for (i, mov) in chunk.iter().enumerate() {
                    out[depth][i] = mov.iter().zip(block).map(|(m, k)| m * k).sum::<f32>();
                }
            }
        }
        LayerOutput::Conv(out)
    }

    pub fn b_prop(&mut self, output_gradient: &Vec<f32>, learning_rate: f32) -> Vec<f32> {
        // output_gradient =  dE / dY
        //
        // the output gradient is going to be 1d ( since we flatten our output to 1d array in forward
        // propagation ), as long as we know our kernel size, we can get the individual block of
        // output of our kernels.
        //
        // since the activation layer store its input as an enum (output of layer), the activation layer
        // can also decipher the output base on the type of layer it is responsible to. However, we
        // might not need to do it in the activation layer since we can just flatten the input.
        let mut og = Vec::new();
        for chunk in output_gradient
            .chunks(output_gradient.len() / self.output_shape.0)
            .into_iter()
        {
            og.push(chunk.to_vec());
        }

        // cross correlation between output_gradient and input
        let mut kernel_gradient = vec![vec![]; self.input_shape.0];
        let kg_chunks =
            Self::correlation_chunks(&self.input, &self.input_shape, self.output_shape.1);
        for (depth, (chunk, grad)) in kg_chunks.iter().zip(og.iter()).enumerate() {
            for mov in chunk {
                kernel_gradient[depth].push(mov.iter().zip(grad).map(|(m, k)| m * k).sum::<f32>());
            }
        }
        self.kernels = self
            .kernels
            .iter()
            .zip(og.iter())
            .map(|(kernel_depth, gradients)| {
                kernel_depth
                    .iter()
                    .map(|kernel| {
                        kernel
                            .iter()
                            .zip(gradients.iter())
                            .map(|(k, g)| k - g * learning_rate)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        //biases
        self.biases = self
            .biases
            .iter()
            //depth
            .zip(og.iter())
            //biases and gradients
            .map(|(bias, grad)| {
                bias.iter()
                    .zip(grad.iter())
                    //bias and grad
                    .map(|(b, g)| b - g * learning_rate)
                    .collect()
            })
            .collect();

        // Full correlation between output_gradient and kernel
        //let mut input_grendient = Vec::new();
        Self::full_correlation_chunks(&og, &self.output_shape, self.kernel_shape.1);

        todo!()
    }

    fn correlation_chunks(
        input: &Vec<Vec<f32>>,
        input_shape: &(usize, usize, usize),
        size: usize,
    ) -> Vec<Vec<Vec<f32>>> {
        let mut out = Vec::new();
        let mut indices = Vec::new();
        for depth in 0..input_shape.0 {
            let mut d = Vec::new();
            for row in 0..size {
                for col in 0..size {
                    d.push(col + row * input_shape.0);
                }
            }
            indices.push(d);
        }
        for depth in 0..input_shape.0 {
            let mut at_depth = Vec::new();
            for row in 0..input_shape.2 - size + 1 {
                for col in 0..input_shape.1 - size + 1 {
                    let mut tmp = Vec::new();
                    for i in indices[depth].iter() {
                        tmp.push(input[depth][*i + col]);
                    }
                    at_depth.push(tmp);
                }
                indices[depth] = indices[depth].iter().map(|i| i + input_shape.1).collect();
            }
            out.push(at_depth);
        }
        out
    }

    fn full_correlation_chunks(
        input: &Vec<Vec<f32>>,
        input_shape: &(usize, usize, usize),
        size: usize,
    ) -> Vec<Vec<Vec<f32>>> {
        let mut col_indices = Vec::new();
        let mut tmp = VecDeque::new();
        let mut index = 0;
        //for _ in 0..input_shape.1 - size + 2 {
        for _ in 0..input_shape.1 {
            //println!("{:?}", tmp);
            if tmp.is_empty() || tmp.len() < size {
                tmp.push_back(index);
                index += 1;
            } else if tmp.len() == size {
                tmp.pop_front().expect("size should not be 0");
                tmp.push_back(tmp[tmp.len() - 1] + 1);
            }
            col_indices.push(tmp.clone());
        }
        for _ in 0..2 {
            tmp.pop_front().expect("size should not be 0");
            if !tmp.is_empty() {
                col_indices.push(tmp.clone());
            }
        }
        let row_indcies: Vec<usize> = col_indices.iter().map(|col| col.len()).collect();
        let mut d: Vec<Vec<Vec<usize>>> = Vec::new();
        let mut col_idx = 0;
        let mut count = 1;
        for R in row_indcies.iter() {
            let mut tmp: Vec<Vec<usize>> = vec![vec![]; col_indices.len()];
            for row in 0..*R {
                for (tmp, col) in tmp.iter_mut().zip(col_indices.iter()) {
                    for c in col {
                        tmp.push(c + input_shape.1 * col_idx)
                    }
                }
                col_idx += 1;
            }
            col_idx = 0;
                println!("{} {}", R, size);
            if R == &size || count > 1 {
                col_idx = count;
                count += 1;
            }
            d.push(tmp);
        }
        //println!("{:?}", col_indices);
        //println!("{:?}", row_indcies);
        for x in d {
            println!("{:?}", x);
        }
        todo!()
    }
}

#[test]
fn conv_init_f_prop() {
    let mut l1 = ConvolutionLayer::new((1, 8, 8), (1, 3));
    let test: Vec<f32> = (0..8 * 8).into_iter().map(|i| i as f32).collect();
    let l1_out = l1.f_prop(&vec![test]);
    //println!("{:?}", l1_out);

    let mut l2 = ConvolutionLayer::new((2, 28, 28), (2, 5));
    let test28: Vec<f32> = (0..28 * 28).into_iter().map(|i| i as f32).collect();
    //println!("{:?}", l2.f_prop(&vec![test28.clone(), test28.clone()]));

    let mut l3 = ConvolutionLayer::new((3, 28, 28), (3, 5));
    /* println!(
        "{:?}",
        l3.f_prop(&vec![test28.clone(), test28.clone(), test28.clone()])
    ); */

    match l1_out {
        LayerOutput::Conv(out) => {
            l1.b_prop(&out.into_iter().flatten().collect(), 0.1);
        }
        LayerOutput::Dense(_) => todo!(),
        LayerOutput::None => todo!(),
    }
}
