use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
enum LossFunction {
    Mse(MSE)
}
pub trait Loss {
    fn loss(&self, truth: &Vec<f32>, prediction: &Vec<f32>) -> f32;
    fn loss_prime(&self, truth: &Vec<f32>, prediction: &Vec<f32>) -> Vec<f32>;
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct MSE;
impl Loss for MSE {
    fn loss(&self, truth: &Vec<f32>, prediction: &Vec<f32>) -> f32 {
        let len = truth.len() as f32;
        truth
            .iter()
            .zip(prediction.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f32>()
            / len
    }
    fn loss_prime(&self, truth: &Vec<f32>, prediction: &Vec<f32>) -> Vec<f32> {
        let len = truth.len() as f32;
        truth
            .iter()
            .zip(prediction.iter())
            .map(|(t, p)| 2f32 * (p - t) / len)
            .collect()
    }
}
