use neural_network::activations::ActivationFn;
use neural_network::activations::Sigmoid;
use neural_network::layer::dense::DenseLayer;
use neural_network::layer::LayerType;
use neural_network::loss::MSE;
use neural_network::network::Network;

use mnist::MnistLoader;

use rand::thread_rng;
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let MnistLoader {
        train_images,
        train_labels,
        test_images,
        test_labels,
    } = MnistLoader::new(
        "./examples/mnist_data/train-images-idx3-ubyte",
        "./examples/mnist_data/train-labels-idx1-ubyte",
        "./examples/mnist_data/t10k-images-idx3-ubyte",
        "./examples/mnist_data/t10k-labels-idx1-ubyte",
    )?;

    let train_size = 1000;
    let test_size = 100;

    let mut rng = thread_rng();
    let train_bound: usize = 60000 - train_size;
    let test_bound: usize = 10000 - test_size;
    let train_left = rng.gen_range(train_size..train_bound);
    let test_left = rng.gen_range(test_size..test_bound);
    let train_right = train_left + train_size;
    let test_right = test_left + test_size;

    let train_range = train_left..train_right;
    let test_range = test_left..test_right;

    let mut train_set = Vec::new();
    let mut train_answer = Vec::new();

    let mut test_set = Vec::new();
    let mut test_answer = Vec::new();
    println!("Loading dataset...");
    for i in train_range.clone() {
        let img = train_images[i * 28 * 28..(i + 1) * 28 * 28].to_vec();
        let mut temp: Vec<f32> = Vec::new();
        for pixel in img.iter() {
            temp.push(*pixel as f32 / 255.0);
        }
        train_set.push(temp);
    }

    for i in train_range.clone() {
        let l = train_labels[i];
        let mut temp = [0f32; 10];
        temp[l as usize] = 1f32;
        train_answer.push(temp.to_vec());
    }

    for i in test_range.clone() {
        let img = test_images[i * 28 * 28..(i + 1) * 28 * 28].to_vec();
        let mut temp: Vec<f32> = Vec::new();
        for pixel in img.iter() {
            temp.push(*pixel as f32 / 255.0)
        }
        test_set.push(temp);
    }

    for i in test_range.clone() {
        let l = test_labels[i];
        let mut temp = [0f32; 10];
        temp[l as usize] = 1f32;
        test_answer.push(temp.to_vec());
    }

    let layer_1 = DenseLayer::new(28 * 28, 50);
    let layer_2 = DenseLayer::new(50, 10);
    let activation_1 = Sigmoid::default();
    let activation_2 = Sigmoid::default();
    let layers: Vec<LayerType> = vec![LayerType::Dense(layer_1), LayerType::Dense(layer_2)];
    let activations: Vec<ActivationFn> = vec![
        ActivationFn::Sigmoid(activation_1),
        ActivationFn::Sigmoid(activation_2),
    ];

    let mut network = Network::new(layers, activations);

    println!("Training started...");

    network.train(MSE {}, &train_set, &train_answer, 3f32, 100, true);

    println!("Training finished...\n\n");

    println!("---------- Against original train set ----------");
    let mut correct = 0f32;
    for img_num in 0..test_size {
        let out = network.predict(&train_set[img_num]);
        let answer = &train_answer[img_num];

        let pred = max_f32(&out)?;
        let truth = max_f32(&answer)?;
        if pred.0 == truth.0 {
            correct += 1f32;
        }
        println!(
            "Prediction: {:>1}  Confidence: {:>6.2}% | Truth: {}",
            pred.0,
            pred.1 * 100f32,
            truth.0
        );
    }
    println!(
        "\nCorrect: {:>5}  Incorrect: {:>5}   Accuracy: {:>5.2}%",
        correct as usize,
        test_size - correct as usize,
        correct / test_size as f32 * 100.0,
    );

    println!("\n\n---------- Against Test set ----------");
    let mut correct = 0f32;
    for img_num in 0..test_size {
        let out = network.predict(&test_set[img_num]);
        let answer = &test_answer[img_num];

        let pred = max_f32(&out)?;
        let truth = max_f32(&answer)?;
        if pred.0 == truth.0 {
            correct += 1f32;
        }
        println!(
            "Prediction: {:>1}  Confidence: {:>6.2}% | Truth: {}",
            pred.0,
            pred.1 * 100f32,
            truth.0
        );
    }
    println!(
        "\nCorrect: {:>5}  Incorrect: {:>5}   Accuracy: {:>5.2}%",
        correct as usize,
        test_size - correct as usize,
        correct / test_size as f32 * 100.0,
    );

    println!("Model saved to ./models/mnist");
    network.save_to_file("./models/mnist").unwrap();

    Ok(())
}

fn max_f32(v: &Vec<f32>) -> Result<(usize, &f32), Box<dyn std::error::Error>> {
    let mut iter = v.iter().enumerate();
    let init = iter.next().ok_or("Need at least one input")?;
    let result = iter.try_fold(init, |acc, x| {
        let cmp = x.1.partial_cmp(acc.1)?;
        let max = if let std::cmp::Ordering::Greater = cmp {
            x
        } else {
            acc
        };
        Some(max)
    });

    if !result.is_some() {
        return Err("NaN value exists".into());
    }
    Ok(result.unwrap())
}
