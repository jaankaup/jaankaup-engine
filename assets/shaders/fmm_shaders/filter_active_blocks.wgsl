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

@group(0) @binding(0) var<uniform>            prefix_params: PrefixParams;
@group(0) @binding(1) var<uniform>            fmm_params:    FmmParams;
@group(0) @binding(2) var<storage,read_write> fmm_blocks: array<FmmBlock>;
@group(0) @binding(3) var<storage,read_write> temp_prefix_sum: array<u32>;
@group(0) @binding(4) var<storage,read_write> temp_data: array<FmmCellPc>;
@group(0) @binding(5) var<storage,read_write> fmm_data: array<FmmCellPc>;
@group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>; // 5 placeholders

let THREAD_COUNT = 1024u;
let SCAN_BLOCK_SIZE = 2176u;

// thread count :: bcf memory count
// 128          :: 272
// 192          :: 408
// 256          :: 544
// 320          :: 680
// 384          :: 816
// 448          :: 952
// 512          :: 1088
// 576          :: 1224
// 640          :: 1360
// 704          :: 1496
// 768          :: 1632
// 832          :: 1768
// 896          :: 1904
// 960          :: 2040
// 1024         :: 2176

var<workgroup> shared_prefix_sum: array<u32, SCAN_BLOCK_SIZE>; 
var<workgroup> shared_aux: array<u32, SCAN_BLOCK_SIZE>;
var<workgroup> stream_compaction_count: u32;

var<private> private_data: PrivateData;

fn create_prefix_sum_private_data(local_index: u32, workgroup_index: u32) {

    let ai = local_index; 
    let bi = ai + THREAD_COUNT;
    let ai_bcf = ai + (ai >> 4u); 
    let bi_bcf = bi + (bi >> 4u);
    let global_ai = ai + workgroup_index * (THREAD_COUNT * 2u);
    let global_bi = bi + workgroup_index * (THREAD_COUNT * 2u);

    private_data = PrivateData (
        ai,
        bi, 
        ai_bcf,
        bi_bcf,
        global_ai,
        global_bi
    );
}

fn total_cell_count() -> u32 {

    return fmm_params.global_dimension.x * 
           fmm_params.global_dimension.y * 
           fmm_params.global_dimension.z * 
           fmm_params.local_dimension.x * 
           fmm_params.local_dimension.y * 
           fmm_params.local_dimension.z; 
};

//////////////////////////////////
////// Prefix scan part 1   //////
//////////////////////////////////

fn store_block_to_global_temp() {

    temp_prefix_sum[private_data.global_ai] = shared_prefix_sum[private_data.ai_bcf];
    temp_prefix_sum[private_data.global_bi] = shared_prefix_sum[private_data.bi_bcf];
}

fn copy_block_to_shared_temp() {

    var a = fmm_blocks[private_data.global_ai];
    var b = fmm_blocks[private_data.global_bi];

    let end_index = total_cell_count();

    // Add prediates here {0 :: false, 1 :: true).
    shared_prefix_sum[private_data.ai_bcf] = select(0u, 1u, private_data.global_ai < end_index && a.number_of_band_points > 0u);
    shared_prefix_sum[private_data.bi_bcf] = select(0u, 1u, private_data.global_bi < end_index && b.number_of_band_points > 0u);
}

// Perform prefix sum for one dispatch.
fn local_prefix_sum(workgroup_index: u32) {

    // Up sweep.

    let n = THREAD_COUNT * 2u;
    var offset = 1u;

    for (var d = n >> 1u ; d > 0u; d = d >> 1u) {

        workgroupBarrier();

        if (private_data.ai < d) {

            var ai_temp = offset * (private_data.ai * 2u + 1u) - 1u;
            var bi_temp = offset * (private_data.ai * 2u + 2u) - 1u;

            ai_temp = ai_temp + (ai_temp >> 4u);
            bi_temp = bi_temp + (bi_temp >> 4u);

            shared_prefix_sum[bi_temp] = shared_prefix_sum[bi_temp] + shared_prefix_sum[ai_temp];

        }
        offset = offset * 2u; // bit shift?
    }
      
    // Clear the last item. 
    if (private_data.ai == 0u) {

        let last_index = (THREAD_COUNT * 2u) - 1u + ((THREAD_COUNT * 2u - 1u) >> 4u);
        temp_prefix_sum[prefix_params.exclusive_parts_start_index + workgroup_index] = shared_prefix_sum[last_index];
        shared_prefix_sum[last_index] = 0u;
    }

    // storageBarrier(); // ???
    
    // Down sweep.
    for (var d = 1u; d < n ; d = d * 2u) {
        offset = offset >> 1u;
        workgroupBarrier();

        if (private_data.ai < d) {

            var ai_temp = offset * (private_data.ai * 2u + 1u) - 1u;
            var bi_temp = offset * (private_data.ai * 2u + 2u) - 1u;
            ai_temp = ai_temp + (ai_temp >> 4u);
            bi_temp = bi_temp + (bi_temp >> 4u);
            var t = shared_prefix_sum[ai_temp];

            shared_prefix_sum[ai_temp] = shared_prefix_sum[bi_temp];
            shared_prefix_sum[bi_temp] = shared_prefix_sum[bi_temp] + t;
        }
    }
}

@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {
		
        
	// Clear the counters from previous steps.
	if (global_id.x == 0u) {
            fmm_counter[0] = 0u;
            fmm_counter[1] = 0u;
            fmm_counter[2] = 0u;
        }

        create_prefix_sum_private_data(local_index, workgroup_id.x);
        copy_block_to_shared_temp();
        local_prefix_sum(workgroup_id.x);
	workgroupBarrier();
        store_block_to_global_temp();
}
