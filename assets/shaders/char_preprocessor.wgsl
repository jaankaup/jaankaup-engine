struct Camera {
    u_view_proj: mat4x4<f32>;
    pos: vec4<f32>;
};

struct CharParams{
    iterator_start: atomic<u32>;
    itreator_end: u32;
    number_of_threads: u32;
    draw_index: atomic<u32>; 
};

struct Char {
    start_pos: vec4<f32>; // encode start.pos.w the decimal_count.
    value: vec4<f32>;
    font_size: f32;
    vec_dim_count: u32; // 1 => f32, 2 => vec3<f32>, 3 => vec3<f32>, 4 => vec4<f32>
    color: u32;
    draw_index: u32;
};

// TODO: change all z_offets in all files! 
struct Char {
    start_pos: vec4<f32>; // encode start.pos.w the decimal_count. TODO: something better.
    value: vec4<f32>;
    font_size: f32;
    vec_dim_count: u32; // 1 => f32, 2 => vec3<f32>, 3 => vec3<f32>, 4 => vec4<f32>
    color: u32;
    number_of_points: u32;
};

struct DrawIndirect {
    vertex_count: atomic<u32>;
    instance_count: u32;
    base_vertex: u32;
    base_instance: u32;
};

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(0) @binding(1)
var<storage, read_write> char_params: array<CharParams>;

@group(0) @binding(2)
var<storage, read_write> indirect: array<DrawIndirect>;

@group(0) @binding(3)
var<storage,read_write> chars_array: array<Char>;

@stage(compute)
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

}
