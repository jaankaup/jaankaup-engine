struct VertexOutput {
    @builtin(position) final_pos: vec4<f32>,
    @location(0) @interpolate(flat) col: vec4<f32>,
};

struct Camera {
    u_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

@group(0)
@binding(0)
var<uniform> camerauniform: Camera;

fn decode_color(c: u32) -> vec4<f32> {
  let a: f32 = f32(c & 256u) / 255.0;
  let b: f32 = f32((c & 65280u) >> 8u) / 255.0;
  let g: f32 = f32((c & 16711680u) >> 16u) / 255.0;
  let r: f32 = f32((c & 4278190080u) >> 24u) / 255.0;
  return vec4<f32>(r,g,b,a);
}

@stage(vertex)
fn vs_main(@location(0) pos: vec3<f32>, @location(1) col: u32) -> VertexOutput {
    return VertexOutput(
                 camerauniform.u_view_proj * vec4<f32>(pos, 1.0),
                 decode_color(col)
    );
}

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.col;
}
