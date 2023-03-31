use neural_network::network::Network;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
struct WasmNN(Network);

#[wasm_bindgen]
impl WasmNN {
    #[wasm_bindgen(constructor)]
    pub async fn new(url: &str) -> Result<WasmNN, JsError> {
        let mut wasmNN = WasmNN(Network::default());
        wasmNN.0 = Network::default();
        let response = reqwest::get(url).await?.bytes().await?;
        let model: Vec<u8> = response.into();
        wasmNN.0.from_slice(&model).unwrap();
        Ok(wasmNN)
    }

    pub fn predict(&self, input: Vec<f32>) -> Vec<f32> {
        self.0.predict_ref(&input)
    }
}
