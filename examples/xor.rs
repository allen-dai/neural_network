use neural_network::activations::Tanh;
use neural_network::layer::dense::DenseLayer;
use neural_network::loss::MSE;
use neural_network::network::Network;

fn main() {
    let layer_1 = DenseLayer::new(2, 4);
    let layer_2 = DenseLayer::new(4, 1);
    let activation_1 = Tanh::default();
    let activation_2 = Tanh::default();

    let layers = vec![layer_1, layer_2];
    let activations = vec![activation_1, activation_2];

    let mut network = Network::new(layers, activations);
    let train_set = vec![
        vec![0f32, 0f32],
        vec![0f32, 1f32],
        vec![1f32, 0f32],
        vec![1f32, 1f32],
    ];
    let train_answer = vec![vec![0f32], vec![1f32], vec![1f32], vec![0f32]];

    println!("Training started...");
    network.train(MSE {}, &train_set, &train_answer, 0.1f32, 10000, true);
    println!("Training finished...\n\n");

    println!("---------- Against original train set ----------");
    let mut correct = 0f32;
    for i in 0..train_set.len() {
        let out = network.predict(&train_set[i]);
        let answer = &train_answer[i];

        let pred = out[0];
        let truth = answer[0];
        if (pred - truth).abs() < 0.5 {
            correct += 1f32;
        }
        println!("Prediction: {:>5.2} | Truth: {}", pred, truth);
    }
    println!(
        "\nCorrect: {:>5}  Incorrect: {:>5}   Accuracy: {:>5.2}%",
        correct as usize,
        train_set.len() - correct as usize,
        correct / train_set.len() as f32 * 100.0,
    );
}
