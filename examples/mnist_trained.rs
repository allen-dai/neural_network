use mnist::MnistLoader;
use neural_network::network::Network;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let MnistLoader {
        train_images: _,
        train_labels: _,
        test_images,
        test_labels,
    } = MnistLoader::new(
        "./examples/mnist_data/train-images-idx3-ubyte",
        "./examples/mnist_data/train-labels-idx1-ubyte",
        "./examples/mnist_data/t10k-images-idx3-ubyte",
        "./examples/mnist_data/t10k-labels-idx1-ubyte",
    )?;

    let test_size = 10000;
    let test_range = 0..10000;

    let mut test_set = Vec::new();
    let mut test_answer = Vec::new();
    println!("Loading dataset...");

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

    println!("Loading Model...");

    let mut network = Network::from_file("./models/mnist").unwrap();

    println!("Model loaded, starting to test...\n\n");

    println!("\n\n---------- Test set ----------");
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
