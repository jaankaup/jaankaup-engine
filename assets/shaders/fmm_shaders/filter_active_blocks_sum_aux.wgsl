let FAR      = 0u;
let BAND_NEW = 1u;
let BAND     = 2u;
let KNOWN    = 3u;
let OUTSIDE  = 4u;

struct FmmCellPc {
    tag: atomic<u32>,
    value: f32,
    color: u32,
};

struct FmmBlock {
    index: u32,
    number_of_band_points: atomic<u32>,
};

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

struct PrivateData {
    ai: u32,
    bi: u32,
    ai_bcf: u32,
    bi_bcf: u32,
    global_ai: u32,
    global_bi: u32,
};

let THREAD_COUNT = 128u;

@group(0) @binding(0) var<uniform>            prefix_params: PrefixParams;
@group(0) @binding(1) var<uniform>            fmm_params:    FmmParams;
@group(0) @binding(2) var<storage,read_write> fmm_blocks: array<FmmBlock>;
@group(0) @binding(3) var<storage,read_write> temp_prefix_sum: array<u32>;
@group(0) @binding(4) var<storage,read_write> temp_data: array<FmmBlock>;
@group(0) @binding(5) var<storage,read_write> fmm_data: array<FmmCellPc>;
@group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>; // 5 placeholders

fn total_cell_count() -> u32 {

    return fmm_params.global_dimension.x * 
           fmm_params.global_dimension.y * 
           fmm_params.global_dimension.z * 
           fmm_params.local_dimension.x * 
           fmm_params.local_dimension.y * 
           fmm_params.local_dimension.z; 
};

@compute
@workgroup_size(128,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    let data_count = total_cell_count();

    let index_a = local_index + workgroup_id.x * THREAD_COUNT * 2u;
    let index_b = local_index + THREAD_COUNT + workgroup_id.x * THREAD_COUNT * 2u;

    let number_to_add = temp_prefix_sum[prefix_params.exclusive_parts_start_index + workgroup_id.x / 8u + 2048u]; // shared_aux[i + (i >> 4u)];

    var a = temp_prefix_sum[index_a];
    temp_prefix_sum[index_a] = a + number_to_add;

    var b = temp_prefix_sum[index_b];
    temp_prefix_sum[index_b] = b + number_to_add;
};
