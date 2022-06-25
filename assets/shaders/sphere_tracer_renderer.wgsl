@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;

@group(0) @binding(1)
var s_diffuse: sampler;

struct VertexOutput {
    @builtin(position) my_pos: vec4<f32>,
    @location(0) pos: vec4<f32>,
};

@vertex
//fn vs_main(@location(0) gl_pos: vec4<f32>, @location(1) point_pos: vec4<f32>) -> VertexOutput {
fn vs_main(@location(0) gl_pos: vec4<f32>) -> VertexOutput {
    var out_next_stage: VertexOutput;
    out_next_stage.my_pos  = gl_pos;
    out_next_stage.pos  = vec4<f32>(gl_pos.xy * 0.5, 0.0, 1.0) + vec4<f32>(0.5, 0.5, 0.0, 0.0);
    return out_next_stage;
    //return VertexOutput(
    //    gl_pos,
    //    //vec4<f32>(0.0, 0.0, 0.0, 0.0), //gl_pos,
    //    // vec4<f32>(0.0, 0.0, 0.0, 0.0) //gl_pos,
    //    // point_pos,
    //);
    // var out: VertexOutput;
    // out.my_pos = gl_pos;
    // out.pos_out = point_pos;
    // return out;
}

@fragment
fn fs_main(inp: VertexOutput) -> @location(0) vec4<f32> {
    // return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    return textureSample(t_diffuse, s_diffuse, inp.pos.xy);
}
