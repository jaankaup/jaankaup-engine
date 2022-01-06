#!/bin/bash

RUSTFLAGS=--cfg=web_sys_unstable_apis cargo build --features input_debug --no-default-features --target wasm32-unknown-unknown --example "$1"
wasm-bindgen --out-dir target/generated --web target/wasm32-unknown-unknown/debug/examples/$1.wasm
#RUSTFLAGS=--cfg=web_sys_unstable_apis cargo build -p "$1" --target wasm32-unknown-unknown
#wasm-bindgen --out-dir target/generated --web "target/wasm32-unknown-unknown/debug/$1.wasm"
