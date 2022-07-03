/// Kernel that finds all cells of given tag. 
/// The number of found cells is stored to fmm_fmm_counter[0]. The fmm_counter[pc.tag] must be equal to zero. 
/// The output (cell indices) are saved to temp_prefix_sum array.

let FAR      = 0u;
let BAND_NEW = 1u;
let BAND     = 2u;
let KNOWN    = 3u;
let OUTSIDE  = 4u;
let KNOWN_NEW = 5u;

struct FmmCellPc {
    tag: atomic<u32>,
    value: f32,
    color: u32,
};

struct FmmBlock {
    index: u32,
    number_of_band_points: u32,
};

struct TempData {
    index: u32,
    tag: u32,
    value: f32,
};

struct PushConstants {
    tag: u32,    
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
@group(0) @binding(2) var<storage,read_write> fmm_blocks: array<FmmBlock>;
@group(0) @binding(3) var<storage,read_write> temp_prefix_sum: array<u32>;
@group(0) @binding(4) var<storage,read_write> temp_data: array<FmmBlock>;
@group(0) @binding(5) var<storage,read_write> fmm_data: array<FmmCellPc>;
@group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>; // 5 placeholders

// Workgroup counter to keep track of count of found cells.
// var<workgroup> shared_counter: atomic<u32>;
// var<workgroup> offset: u32;
var<workgroup> wg_cells: array<TempData, 64>; 

// Push constants.
//var<push_constant> pc: PushConstants;

fn total_cell_count() -> u32 {

    return fmm_params.global_dimension.x * 
           fmm_params.global_dimension.y * 
           fmm_params.global_dimension.z * 
           fmm_params.local_dimension.x * 
           fmm_params.local_dimension.y * 
           fmm_params.local_dimension.z; 
};

// fn gather_cells_to_temp_data(thread_index: u32) {
// 
//     var fmm_cell = fmm_data[thread_index];
// 
//     if (fmm_cell.tag == pc.tag) {
// 
//         let index = atomicAdd(&shared_counter, 1u);
// 
//         // Save the cell index to workgroup memory.
//         temp_indices[index] = thread_index;
//     }
// }

fn reduce(local_index: u32) {

    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
        if (local_index < s) {

	    var values: array<TempData, 2> = array<TempData, 2>(wg_cells[local_index], wg_cells[local_index + s]);
	    var choice = select(0, 1, ((values[0].tag == BAND) && (values[1].tag == BAND) && (values[0].value > values[1].value)) ||
	                 ((values[0].tag != BAND) && (values[1].tag == BAND)));
	    // var choice = ((values[0].tag == BAND) && (values[1].tag == BAND) && (values[0].value > values[1].value)) ||
	    //              ((values[0].tag != BAND) && (values[1].tag == BAND));
            //wg_cells[local_index] = values[u32(choice)]; 
            wg_cells[local_index] = values[choice]; 
	}
            // if (choice) { wg_cells[local_index] = values[1]; }
	    // else {
            //     wg_cells[local_index] = values[0]; 
	    // }

            // let ta = wg_cells[local_index];
            // let tb = wg_cells[local_index + s];

	// var values: array<TempData, 2> = array<TempData, 2>(wg_cells[local_index], wg_cells[local_index + s]);
	// var choice = ((ta.tag == BAND) && (tb.tag == BAND) && (ta.value > tb.value)) ||
	//              ((ta.tag != BAND) && (tb.tag == BAND));

	// var choice = (ta.tag == BAND) && (tb.tag == BAND) && (ta.value > tb.value);
	// choice = select(choice, true, (ta.tag != BAND) && (tb.tag == BAND));

        // var temp = select(ta, tb, (ta.tag == BAND) && (tb.tag == BAND) && (ta.value > tb.value));
	// temp =     select(temp, tb, (ta.tag != BAND) && (tb.tag == BAND));
        
        // wg_cells[local_index] = temp; 
        workgroupBarrier();
    }

    // for (uint s = LOCAL_X_DIM/2 ; s > 0 ; s >>= 1) {
    //     if (gl_LocalInvocationID.x < s && true) {

    //         uint a_index = shared_fmm_indices[gl_LocalInvocationIndex]; // + 160 * gl_LocalInvocationID.y;
    //         uint b_index = shared_fmm_indices[gl_LocalInvocationIndex + s]; // + 160 * gl_LocalInvocationID.y;

    //         FMM_Node a = shared_fmm_nodes[a_index + 160 * gl_LocalInvocationID.y];
    //         FMM_Node b = shared_fmm_nodes[b_index + 160 * gl_LocalInvocationID.y];

    //         // TODO: optimize

    //         uint temp_index = (a.tag == BAND) &&
    //                           (b.tag == BAND) &&
    //                           (a.value > b.value) ? b_index : a_index;

    //         temp_index      = (a.tag != BAND) &&
    //                           (b.tag == BAND) ? b_index : temp_index;

    //         shared_fmm_indices[gl_LocalInvocationIndex] = shared_fmm_indices[temp_index];

    //     }
    // }
}

@compute
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(num_workgroups) num_workgroups: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

        let block_count = fmm_counter[1];

	let actual_index = workgroup_id.x + workgroup_id.y * num_workgroups.x;
        if (actual_index < block_count) {

	    let b = temp_data[actual_index]; // num_workgroups.x)];

            // Load cell to workgroup memory.
            //let cell_index = (b.index << 6u) + local_index;
            let cell_index = b.index * 64u + local_index;
	    var cell = fmm_data[cell_index];
            wg_cells[local_index] = TempData(cell_index, cell.tag, cell.value);

            workgroupBarrier(); // This is safe because all threads, or none of them, comes here.

            reduce(local_index);

	    if (local_index == 0u) {

                // A new known point!
	        fmm_data[wg_cells[0].index].tag = KNOWN; 

	        // Reduce band count by one.
                fmm_blocks[b.index].number_of_band_points = u32(i32(b.number_of_band_points) - 1);

		// Add new found known point to 
                let known_index = atomicAdd(&fmm_counter[KNOWN], 1u); // TODO: remove.
                temp_prefix_sum[actual_index] = wg_cells[0].index;
            }
	}
}
