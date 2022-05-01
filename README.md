# jaankaup-engine
A wgpu graphics/compute engine.

This project is a huge mess. The main goal is to implement a parallel algorithm for the fast marching method (GPU).
When I finish the this project, I'll rewrite the whole engine.

## Examples

some_examples directory contains some small projects that use jaankaup_engine. For more information, see some_examples directory.

The projects can be occasionally broken.

$ cargo run --example fmm (not finished)

$ cargo run --example curves (some space-filling curve visualizations)

$ cargo run --example mc (marching cubes test, compiles to wasm)

$ cargo run --example basic (a textured cube, compiles to wasm)

$ cargo run --example visualizer (just testing aabb:s, arrows and numbers)
