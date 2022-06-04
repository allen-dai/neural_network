mod activations;
mod layer;
mod loss;
mod network;

use activations::{Sigmoid, Tanh};
use layer::dense::DenseLayer;
use loss::Mse;
use network::Network;

use mnist::MnistLoader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    /* let MnistLoader {
        train_images,
        train_labels,
        ..
    } = MnistLoader::new(
        "./data/train-images-idx3-ubyte",
        "./data/train-labels-idx1-ubyte",
        "./data/t10k-images-idx3-ubyte",
        "./data/t10k-labels-idx1-ubyte",
    )?; */

    let layer_1 = DenseLayer::new(2, 2);
    let layer_2 = DenseLayer::new(2, 2);
    let layer_3 = DenseLayer::new(2, 1);
    let activation_1 = Sigmoid::default();
    let activation_2 = Tanh::default();
    let activation_3 = Sigmoid::default();

    let mut layers = vec![layer_1, layer_3];
    let mut activations = vec![activation_1, activation_3];

    let mut network = Network::new(&mut layers, &mut activations);
    let train_set = vec![
        vec![0f32, 0f32],
        vec![0f32, 1f32],
        vec![1f32, 0f32],
        vec![1f32, 1f32],
    ];
    let train_answer = vec![vec![0f32], vec![1f32], vec![1f32], vec![0f32]];

    network.train(Mse {}, &train_set, &train_answer, 0.1, 10000, true);

    let input = vec![0f32, 0f32];
    let output = network.predict(&input);
    println!("\n\n{:?}\n{:?}", input, output);

    let input = vec![0f32, 1f32];
    let output = network.predict(&input);
    println!("\n\n{:?}\n{:?}", input, output);

    let input = vec![1f32, 0f32];
    let output = network.predict(&input);
    println!("\n\n{:?}\n{:?}", input, output);

    let input = vec![1f32, 1f32];
    let output = network.predict(&input);
    println!("\n\n{:?}\n{:?}", input, output);

    Ok(())
}

/* use mnist::MnistLoader;
use plotters::prelude::*;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let MnistLoader { train_images, .. } = MnistLoader::new(
        "./data/train-images-idx3-ubyte",
        "./data/train-labels-idx1-ubyte",
        "./data/t10k-images-idx3-ubyte",
        "./data/t10k-labels-idx1-ubyte",
    )?;

    let img_num = 20;

    let image1 = &train_images[img_num * 28 * 28..(img_num + 1) * 28 * 28];
    //let image1 = train_images[0];
    for y in 0..28 {
        print!("\n");
        for x in 0..28 {
            let n = image1[y * 28 + x];
            if n == 0 {
                print!("   ");
            } else {
                print!("{}", n);
            }
        }
    }

    let root = BitMapBackend::new("./images/01.png", (28, 28)).into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root).build_cartesian_2d(28..0, 28..0)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;

    let plotting_area = chart.plotting_area();

    for y in 0..28 {
        let mut row = Vec::new();
        for x in 0..28 {
            row.push(image1[y * 28 + x]);
        }
        row.reverse();
        for (x, v) in row.iter().enumerate() {
            plotting_area.draw_pixel((x as i32, y as i32), &RGBColor(*v, *v, *v))?;
        }
    }
    Ok(())
}
*/
