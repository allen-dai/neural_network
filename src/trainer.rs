use crate::{
    activations::{Activation, ActivationFn},
    layer::LayerType,
    loss::Loss,
    network::Network,
};
use num_cpus;
use scoped_threadpool::Pool;
use std::sync::mpsc;

pub struct Trainer;

impl Trainer {
    pub fn cpu(
        network: &mut Network,
        loss_fn: impl Loss + std::marker::Send + Copy,
        train_set: &Vec<Vec<f32>>,
        train_ans: &Vec<Vec<f32>>,
        learning_rate: f32,
        epoch: usize,
        verbose: bool,
        path: &str,
    ) {
        let num_thread = num_cpus::get();
        let mut pool = Pool::new(num_thread as u32);
        let batch_size = train_set.len() / num_thread;
        pool.scoped(|s| {
            for e in 0..epoch {
                let (layer_tx, layer_rx) = mpsc::channel();
                let (loss_tx, loss_rx) = mpsc::channel();
                for n in 0..num_thread {
                    let mut net = network.clone();
                    let layer_tx_clone = layer_tx.clone();
                    let loss_tx_clone = loss_tx.clone();
                    s.execute(move || {
                        let (mut loss, mut output, mut gradient);
                        loss = 0f32;
                        for i in n * batch_size..(n + 1) * batch_size {
                            let x = &train_set[i];
                            let y = &train_ans[i];
                            output = net.predict(&x);
                            loss += loss_fn.loss(&y, &output);
                            gradient = loss_fn.loss_prime(&y, &output);
                            for (layer_type, activation_fn) in
                                net.layers.iter_mut().zip(net.activations.iter_mut()).rev()
                            {
                                match activation_fn {
                                    ActivationFn::Tanh(tanh) => {
                                        gradient = tanh.b_prop(&gradient);
                                    }
                                    ActivationFn::Sigmoid(sigmoid) => {
                                        gradient = sigmoid.b_prop(&gradient);
                                    }
                                    ActivationFn::Relu(relu) => {
                                        gradient = relu.b_prop(&gradient);
                                    }
                                }

                                match layer_type {
                                    LayerType::Dense(layer) => {
                                        gradient = layer.b_prop(&gradient, learning_rate);
                                    }
                                    LayerType::Conv(layer) => {
                                        gradient = layer.b_prop(&gradient, learning_rate);
                                    }
                                }
                            }
                        }
                        layer_tx_clone.send(net.layers).unwrap();
                        loss_tx_clone.send(loss / train_set.len() as f32).unwrap();
                    });
                }

                drop(layer_tx);
                drop(loss_tx);

                for layers in layer_rx {
                    for (train_layer, layer) in layers.iter().zip(network.layers.iter_mut()) {
                        match (train_layer, layer) {
                            (LayerType::Dense(train_dense), LayerType::Dense(dense)) => {
                                for (train_w, w) in
                                    train_dense.weights.iter().zip(dense.weights.iter_mut())
                                {
                                    for col in 0..w.len() {
                                        w[col] = (w[col] + train_w[col]) / 2.0;
                                    }
                                }

                                for col in 0..dense.biases.len() {
                                    dense.biases[col] =
                                        (dense.biases[col] + train_dense.biases[col]) / 2.0;
                                }
                            }
                            (LayerType::Conv(train_conv), LayerType::Conv(conv)) => {
                                // z_depth. At the time I wrote this 3d conv, I did not have a
                                // complete picture, I called this depth in its implementation
                                for (train_z, z) in
                                    train_conv.kernels.iter().zip(conv.kernels.iter_mut())
                                {
                                    for (train_r, r) in train_z.iter().zip(z.iter_mut()) {
                                        for col in 0..train_r.len() {
                                            r[col] = (r[col] + train_r[col]) / 2.0;
                                        }
                                    }
                                }

                                //biases
                                for (train_r, r) in
                                    train_conv.biases.iter().zip(conv.biases.iter_mut())
                                {
                                    for col in 0..train_r.len() {
                                        r[col] = (r[col] + train_r[col]) / 2.0;
                                    }
                                }
                            }
                            _ => (),
                        }
                    }
                }
                let loss: f32 = loss_rx.iter().sum::<f32>() / num_thread as f32;
                if verbose {
                    println!("epoch: {} to {}", e + 1, loss)
                }
                network.save_to_file(&path).unwrap();
            }
        });
    }
}
