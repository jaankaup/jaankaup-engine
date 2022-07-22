struct AABB {
    min: vec4<f32>, 
    max: vec4<f32>, 
};

struct Arrow {
    start_pos: vec4<f32>,
    end_pos:   vec4<f32>,
    color: u32,
    size:  f32,
};

struct Char {
    start_pos: vec3<f32>,
    font_size: f32,
    value: vec4<f32>,
    vec_dim_count: u32,
    color: u32,
    decimal_count: u32,
    auxiliary_data: u32,
};

// struct PrivateData {
//     ai: u32,
//     bi: u32,
//     ai_bcf: u32,
//     bi_bcf: u32,
//     global_ai: u32,
//     global_bi: u32,
// };
// 
// // Thread private data for prefix sum.
// struct PrivatePrefixSum {
//     ai: u32,         // Thread index
//     bi: u32,         // Thread index x 2
//     ai_bcf: u32,     // Bank conflict free thread index.
//     bi_bcf: u32,     // Bank conflict free thread index x 2.
//     global_ai: u32,  // Global index. 
//     global_bi: u32,  // Global index x 2. 
// };

struct PrefixParams {
    data_start_index: u32,
    data_end_index: u32,
    exclusive_parts_start_index: u32,
    exclusive_parts_end_index: u32,
};

struct FmmParams {
    global_dimension: vec3<u32>,
    future_usage: u32,
    local_dimension: vec3<u32>,
    future_usage2: u32,
};

struct FimCellPc {
    tag: u32,
    value: f32,
    color: u32,
};

let FAR      = 0u;
let BAND_NEW = 1u;
let BAND     = 2u;
let KNOWN    = 3u;
let OUTSIDE  = 4u;

// @group(0) @binding(0) var<uniform>            prefix_params: PrefixParams;
@group(0) @binding(0) var<uniform>            fmm_params:    FmmParams;
@group(0) @binding(1) var<storage,read_write> active_list: array<u32>; //fmm_blocks
// @group(0) @binding(3) var<storage,read_write> temp_prefix_sum: array<u32>;
@group(0) @binding(2) var<storage,read_write> fim_data: array<FimCellPc>;
@group(0) @binding(3) var<storage,read_write> fim_counter: array<atomic<u32>>; // 5 placeholders

// GpuDebugger.
// @group(1) @binding(0) var<storage,read_write> counter: array<atomic<u32>>;
// @group(1) @binding(1) var<storage,read_write> output_char: array<Char>;
// @group(1) @binding(2) var<storage,read_write> output_arrow: array<Arrow>;
// @group(1) @binding(3) var<storage,read_write> output_aabb: array<AABB>;
// @group(1) @binding(4) var<storage,read_write> output_aabb_wire: array<AABB>;

// fn dummy() {
//   _ = &active_list;
//   // _ = t;
//   // _ = s;
// }

@compute
//@workgroup_size(1024,1,1)
@workgroup_size(256,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

}
