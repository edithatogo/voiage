// SPDX-License-Identifier: Apache-2.0
use innovate_rust::KernelBinding;
use serde_json::{json, Value};
use std::io;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input: Value = serde_json::from_reader(io::stdin())?;
    let binding = KernelBinding::new();
    let mut output = Vec::new();
    for draw in input.as_array().ok_or("expected draws")? {
        let request = binding.predict_model_request(
            "bass",
            json!({
                "inputs":{"time":[0.0,1.0,2.0,3.0]},
                "state":{"model_key":"bass","model_name":"BassModel","constructor_kwargs":{},
                         "predict_kwargs":{},"parameters":draw["parameters"]}
            }),
        );
        let response = binding.predict_model_native(&request)?;
        assert_eq!(response.metadata["runtime"], "rust_native");
        output.push(
            json!({"scenario_id":draw["scenario_id"],"draw_id":draw["draw_id"],
                          "prediction":response.result.ok_or("empty Rust response")?["values"]}),
        );
    }
    serde_json::to_writer(io::stdout(), &output)?;
    Ok(())
}
