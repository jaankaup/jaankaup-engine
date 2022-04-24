// 64 :: 136
// 128 :: 272
// 192 :: 408
// 256 :: 544
// 320 :: 680
// 384 :: 816
// 448 :: 952
// 512 :: 1088
// 576 :: 1224
// 640 :: 1360
// 704 :: 1496
// 768 :: 1632
// 832 :: 1768
// 896 :: 1904
// 960 :: 2040
// 1024 :: 2176

struct PrivateData {
    ai: u32,
    bi: u32,
    ai_bcf: u32,
    bi_bcf: u32,
    global_ai: u32,
    global_bi: u32,
};

struct PrefixParams {
    data_start_index: u32,
    data_end_index: u32,
    exclusive_parts_start_index: u32,
    exclusive_parts_end_index: u32,
    temp_prefix_data_start_index: u32,
    temp_prefix_data_end_index: u32,
    stage: u32,
};

struct FmmBlock {
    index: u32,
    band_points_count: u32,
};

@group(0)
@binding(0)
var<uniform> fmm_prefix_params: PrefixParams;

@group(0)
@binding(1)
var<storage, read_write> fmm_blocks: array<FmmBlock>;

@group(0)
@binding(2)
var<storage, read_write> temp_prefix_sum: array<u32>;

@group(0)
@binding(3)
var<storage,read_write> filtered_blocks: array<FmmBlock>;

// let THREAD_COUNT = 64u;
// let SCAN_BLOCK_SIZE = 136; 

let THREAD_COUNT = 1024u;
let SCAN_BLOCK_SIZE = 2176u; 

//let THREAD_COUNT = 256u;
//let SCAN_BLOCK_SIZE = 544u; 

// The output of global active fmm block scan.
var<workgroup> shared_prefix_sum: array<u32, SCAN_BLOCK_SIZE>; 
var<workgroup> shared_aux: array<u32, SCAN_BLOCK_SIZE>;

// The counter for active fmm blocks.
var<workgroup> stream_compaction_count: u32;
var<workgroup> local_exclusive_part: u32;

var<private> private_data: PrivateData;

fn udiv_up_safe32(x: u32, y: u32) -> u32 {
    let tmp = (x + y - 1u) / y;
    return select(tmp, 0u, y == 0u); 
}

// Encode "rgba" to u32.
fn rgba_u32(r: u32, g: u32, b: u32, a: u32) -> u32 {
  return (r << 24u) | (g << 16u) | (b  << 8u) | a;
}

///////////////////////////
////// Prefix scan   //////
///////////////////////////

fn store_block_to_global_temp() {

    temp_prefix_sum[private_data.global_ai] = shared_prefix_sum[private_data.ai_bcf];
    temp_prefix_sum[private_data.global_bi] = shared_prefix_sum[private_data.bi_bcf];
}

// Copies FMM_Block to shared_prefix_sum {0,1}. "Add padding 0 if necessery." not implemented.
fn copy_block_to_shared_temp() {

    let a = fmm_blocks[private_data.global_ai];
    let b = fmm_blocks[private_data.global_bi];

    shared_prefix_sum[private_data.ai_bcf] = select(0u, 1u, private_data.global_ai < fmm_prefix_params.data_end_index && a.band_points_count > 0u);
    shared_prefix_sum[private_data.bi_bcf] = select(0u, 1u, private_data.global_bi < fmm_prefix_params.data_end_index && b.band_points_count > 0u);
}

fn copy_exclusive_data_to_shared_aux() {

    let data_count = fmm_prefix_params.exclusive_parts_end_index -
                     fmm_prefix_params.exclusive_parts_start_index;

    let data_a = temp_prefix_sum[fmm_prefix_params.exclusive_parts_start_index + private_data.ai];
    let data_b = temp_prefix_sum[fmm_prefix_params.exclusive_parts_start_index + private_data.bi];

    // TODO: remove select.
    shared_aux[private_data.ai_bcf] = select(0u, data_a, private_data.ai < data_count);
    shared_aux[private_data.bi_bcf] = select(0u, data_b, private_data.bi < data_count);
}

// Perform prefix sum for one dispatch.
fn local_prefix_sum(workgroup_index: u32) {

    // var exclusive_part: u32 = 0u;

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
    //++ workgroupBarrier();
      
    // Clear the last item. 
    if (private_data.ai == 0u) {

        let last_index = (THREAD_COUNT * 2u) - 1u + ((THREAD_COUNT * 2u - 1u) >> 4u);
        temp_prefix_sum[fmm_prefix_params.exclusive_parts_start_index + workgroup_index] = shared_prefix_sum[last_index];
        shared_prefix_sum[last_index] = 0u;

    }
    
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

// Perform prefix sum for shared_aux.
fn local_prefix_sum_aux() {

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

            shared_aux[bi_temp] = shared_aux[bi_temp] + shared_aux[ai_temp];
        }
        offset = offset * 2u;
    }
    workgroupBarrier();
      
    // Clear the last item. 
    if (private_data.ai == 0u) {

        let last_index = (THREAD_COUNT * 2u) - 1u + ((THREAD_COUNT * 2u - 1u) >> 4u);

        // Update stream compaction count.
        stream_compaction_count = stream_compaction_count + shared_aux[last_index];

        shared_aux[last_index] = 0u;
    }
    
    // Down sweep.
    for (var d = 1u; d < n ; d = d * 2u) {
        offset = offset >> 1u;
        workgroupBarrier();

        if (private_data.ai < d) {

            var ai_temp = offset * (private_data.ai * 2u + 1u) - 1u;
            var bi_temp = offset * (private_data.ai * 2u + 2u) - 1u;
            ai_temp = ai_temp + (ai_temp >> 4u);
            bi_temp = bi_temp + (bi_temp >> 4u);
            let t = shared_aux[ai_temp];

            shared_aux[ai_temp] = shared_aux[bi_temp];
            shared_aux[bi_temp] = shared_aux[bi_temp] + t;
        }
    }
}

fn sum_auxiliar() {

    let data_count = fmm_prefix_params.data_end_index -
                     fmm_prefix_params.data_start_index;

    let chunks = udiv_up_safe32(data_count, THREAD_COUNT * 2u);

    if (private_data.ai == 0u) {
        let last_index = chunks + 1u;
        stream_compaction_count = shared_aux[last_index + (last_index >> 4u)];
    } 
    workgroupBarrier();

    for (var i: u32 = 0u; i < chunks ; i = i + 1u) {

        let a = temp_prefix_sum[private_data.ai + i * THREAD_COUNT * 2u];
        temp_prefix_sum[private_data.ai + i * THREAD_COUNT * 2u] = a + shared_aux[i + (i >> 4u)];

        let b = temp_prefix_sum[private_data.bi + i * THREAD_COUNT * 2u];
        temp_prefix_sum[private_data.bi + i * THREAD_COUNT * 2u] = b + shared_aux[i + (i >> 4u)];
     }
};

fn gather_data() {

    let data_count = fmm_prefix_params.data_end_index -
                     fmm_prefix_params.data_start_index;

    let chunks = udiv_up_safe32(data_count, THREAD_COUNT * 2u);

    for (var i: u32 = 0u; i < chunks ; i = i + 1u) {

        let index_a = private_data.ai + i * THREAD_COUNT * 2u;
        let index_b = private_data.bi + i * THREAD_COUNT * 2u;

        let a = fmm_blocks[index_a];
        let a_offset = temp_prefix_sum[index_a];
        let predicate_a = index_a < data_count && a.band_points_count > 0u;
        if (predicate_a) { filtered_blocks[a_offset] = a; }

        let b = fmm_blocks[index_b];
        let b_offset = temp_prefix_sum[index_b];
        let predicate_b = index_b < data_count && b.band_points_count > 0u;
        if (predicate_b) { filtered_blocks[b_offset] = b; }
    }
}

@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    let ai = local_index; 
    let bi = ai + THREAD_COUNT;
    let ai_bcf = ai + (ai >> 4u); 
    let bi_bcf = bi + (bi >> 4u);
    let global_ai = ai + work_group_id.x * (THREAD_COUNT * 2u);
    let global_bi = bi + work_group_id.x * (THREAD_COUNT * 2u);

    private_data = PrivateData (
        ai,
        bi, 
        ai_bcf,
        bi_bcf,
        global_ai,
        global_bi
    );

    // STAGE 1.
    // Create scan data and store it to global memory.
    if (fmm_prefix_params.stage == 1u) {

        copy_block_to_shared_temp();
        local_prefix_sum(work_group_id.x);
        store_block_to_global_temp();

    }

    // STAGE 2.
    // Do the actual prefix sum and the filtered data to the final destination array.
    else if (fmm_prefix_params.stage == 2u) {

      if (local_index == 0u) {
          stream_compaction_count = 0u;
      }
      workgroupBarrier();

      // 2a. Load the exclusive data to the shared_aux. 
      copy_exclusive_data_to_shared_aux();

      // 2b. Perform prefix sum on shared_aux data.  
      local_prefix_sum_aux();
      workgroupBarrier();

      sum_auxiliar();
      workgroupBarrier();

      gather_data();

    }

    // STAGE 3.
    // else if (fmm_prefix_params.stage == 3u) {

    //     // 3a. Sum the exclusive parts to the actual prefix sum.
    //     // sum_auxiliar();

    //     // 3b. Gather the filtered data.
    // }
}
