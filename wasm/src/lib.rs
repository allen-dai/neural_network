#![feature(generic_arg_infer)]
#![feature(generic_const_exprs)]

use std::sync::atomic::AtomicPtr;

use neural_network::network::Network;
use wasm_bindgen::prelude::*;

use ::safetensors::{
    serialize_to_file,
    tensor::{Dtype as SDtype, SafeTensors, TensorView},
    SafeTensorError,
};
use dfdx::prelude::*;

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

type M = (
    (Conv2D<1, 3, 3>, ReLU),
    (Conv2D<3, 2, 3>, ReLU),
    Flatten2D,
    Linear<1152, 10>,
);

type Model = <M as BuildOnDevice<AutoDevice, f32>>::Built;

#[wasm_bindgen]
struct DfdxNN {
    model: Model,
}

#[wasm_bindgen]
impl DfdxNN {
    #[wasm_bindgen(constructor)]
    pub async fn new(url: &str) -> Result<DfdxNN, JsError> {
        let dev = AutoDevice::default();
        let mut model = dev.build_module::<M, f32>();
        let response = reqwest::get(url).await?.bytes().await?;
        let bytes: Vec<u8> = response.into();
        model.load_safetensor_from_bytes(&bytes).unwrap();
        Ok(Self { model })
    }

    pub fn predict(&self, input: Vec<f32>) -> Vec<f32> {
        let dev = AutoDevice::default();
        let tensor: Tensor<(Const<1>, Const<28>, Const<28>), f32, Cpu> = dev.tensor(input);
        let out = self.model.forward(tensor);
        out.as_vec()
    }
}
