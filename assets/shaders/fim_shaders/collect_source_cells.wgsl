/// Kernel that finds all source cells. 
/// The number of found cells is stored to fim_counter[0]. The fim_counter[0] must be equal to zero. 
/// The output (cell indices) are saved to temp_prefix_sum array.

let OTHER   = 0u;
let REMEDY  = 1u;
let ACTIVE  = 2u;
let SOURCE  = 3u;
let OUTSIDE = 4u;

struct FimCellPc {
    tag: atomic<u32>,
    value: f32,
    color: u32,
};

// struct FmmBlock {
//     index: u32,
//     number_of_band_points: atomic<u32>,
// };

struct TempData {
    data0: u32,
    data1: u32,
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

@group(0) @binding(0) var<uniform>            prefix_params: PrefixParams;
@group(0) @binding(1) var<uniform>            fmm_params:    FmmParams;
@group(0) @binding(2) var<storage,read_write> active_list: array<TempData>; //fmm_blocks
@group(0) @binding(3) var<storage,read_write> temp_prefix_sum: array<u32>;
// @group(0) @binding(4) var<storage,read_write> remedy_list: array<TempData>; // temp_data
// @group(0) @binding(5) var<storage,read_write> source_list: array<TempData>; // temp_data
@group(0) @binding(4) var<storage,read_write> fim_data: array<FimCellPc>;
@group(0) @binding(5) var<storage,read_write> fim_counter: array<atomic<u32>>; // 5 placeholders

// Workgroup counter to keep track of count of found cells.
var<workgroup> shared_counter: atomic<u32>;
var<workgroup> offset: u32;
var<workgroup> temp_indices: array<u32, 1024>; 

fn total_cell_count() -> u32 {

    return fmm_params.global_dimension.x *
           fmm_params.global_dimension.y *
           fmm_params.global_dimension.z *
           fmm_params.local_dimension.x *
           fmm_params.local_dimension.y *
           fmm_params.local_dimension.z;
};

fn gather_cells_to_temp_data(thread_index: u32) {

    var fim_cell = fim_data[thread_index];

    if (fim_cell.tag == 3u) {

        let index = atomicAdd(&shared_counter, 1u);

        // Save the cell index to workgroup memory.
        temp_indices[index] = thread_index;
    }
}

@compute
@workgroup_size(128,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

        // Initialize shader_counter;
	if (local_index == 0u) { shared_counter = 0u; offset = 0u; }
        workgroupBarrier();

        let cell_count = total_cell_count();

        if (global_id.x < cell_count) {
            gather_cells_to_temp_data(global_id.x);
	}
        workgroupBarrier();

	if (local_index == 0u) {
	    offset = atomicAdd(&fim_counter[0], shared_counter);
        }
        storageBarrier();
        //workgroupBarrier();

	// Scatter data.
        if (local_index < shared_counter) {
            temp_prefix_sum[offset + local_index] = temp_indices[local_index];
	}
}
