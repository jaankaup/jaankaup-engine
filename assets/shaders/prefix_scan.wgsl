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
    ai: u32;
    bi: u32;
    ai_bcf: u32;
    bi_bcf: u32;
    global_ai: u32;
    global_bi: u32;
};

struct AABB {
    min: vec4<f32>; 
    max: vec4<f32>; 
};

struct Arrow {
    start_pos: vec4<f32>;
    end_pos:   vec4<f32>;
    color: u32;
    size:  f32;
};

struct Char {
    start_pos: vec4<f32>;
    value: vec4<f32>;
    font_size: f32;
    vec_dim_count: u32; // 1 => f32, 2 => vec3<f32>, 3 => vec3<f32>, 4 => vec4<f32>
    color: u32;
    z_offset: f32;
};

struct PrefixParams {
    data_start_index: u32;
    data_end_index: u32;
    exclusive_parts_start_index: u32;
    exclusive_parts_end_index: u32;
    temp_prefix_data_start_index: u32;
    temp_prefix_data_end_index: u32;
    stage: u32;
};

struct FmmBlock {
    index: u32;
    band_points_count: u32;
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
var<storage, read_write> counter: array<atomic<u32>>;

@group(0)
@binding(4)
var<storage,read_write> output_char: array<Char>;

@group(0)
@binding(5)
var<storage,read_write> output_arrow: array<Arrow>;

@group(0)
@binding(6)
var<storage,read_write> output_aabb: array<AABB>;

@group(0)
@binding(7)
var<storage,read_write> output_aabb_wire: array<AABB>;

@group(0)
@binding(8)
var<storage,read_write> filtered_blocks: array<FmmBlock>;

// let THREAD_COUNT = 64u;
// let SCAN_BLOCK_SIZE = 136; 
// let X_OFFSET = vec4<f32>(64.0, 0.0, 0.0, 0.0);
let THREAD_COUNT = 1024u;
let SCAN_BLOCK_SIZE = 2176u; 
let X_OFFSET = vec4<f32>(1024.0, 0.0, 0.0, 0.0);
let X_FACTOR = 1.0;

// The output of global active fmm block scan.
var<workgroup> shared_prefix_sum: array<u32, SCAN_BLOCK_SIZE>; 
var<workgroup> shared_aux: array<u32, SCAN_BLOCK_SIZE>;

// The counter for active fmm blocks.
var<workgroup> stream_compaction_count: u32;
var<workgroup> local_exclusive_part: u32;

var<private> base_position: vec4<f32>;
var<private> base_position_bcf: vec4<f32>;
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
////// Logging       //////
///////////////////////////

fn create_arrow(pos_a: vec4<f32>, pos_b: vec4<f32>) {

    output_arrow[atomicAdd(&counter[1], 1u)] =  
          Arrow (
              pos_a,
              pos_b,
              rgba_u32(0u, 0u, 2550u, 255u),
              0.1
    );
}

fn create_number(value: f32, pos: vec4<f32>) {

    output_aabb[atomicAdd(&counter[2], 1u)] =  
          AABB (
              vec4<f32>(pos.x - 0.01,
                        pos.y - 0.1,
                        pos.z - 0.1,
                        f32(rgba_u32(255u, 0u, 0u, 255u))),
              vec4<f32>(pos.x + 0.25,
                        pos.y + 0.2, 
                        pos.z - 0.01,
                        1.0)
    );

    output_char[atomicAdd(&counter[0], 1u)] =  
    
          Char (
              vec4<f32>(pos.x,
                        pos.y,
                        pos.z,
                        1.0 // check do we need this
              ),
              vec4<f32>(value, 0.0, 0.0, 0.0),
              0.05,
              1u,
              rgba_u32(255u, 0u, 2550u, 255u),
              0.1, // not used 
    );
}

///////////////////////////
////// Prefix scan   //////
///////////////////////////

fn store_block_to_global_temp(number_of_items: u32) {

    temp_prefix_sum[private_data.global_ai] = shared_prefix_sum[private_data.ai_bcf];
    temp_prefix_sum[private_data.global_bi] = shared_prefix_sum[private_data.bi_bcf];
}

// Copies FMM_Block to shared_prefix_sum {0,1}. "Add padding 0 if necessery." not implemented.
fn copy_block_to_shared_temp(number_of_items: u32) {

    let a = fmm_blocks[private_data.global_ai];
    let b = fmm_blocks[private_data.global_bi];

    shared_prefix_sum[private_data.ai_bcf] = select(0u, 1u, private_data.ai < number_of_items && a.band_points_count > 0u);
    shared_prefix_sum[private_data.bi_bcf] = select(0u, 1u, private_data.bi < number_of_items && b.band_points_count > 0u);
}

fn copy_prefix_sum_to_temp() {

    temp_prefix_sum[private_data.global_ai] = shared_prefix_sum[private_data.ai_bcf];
    temp_prefix_sum[private_data.global_bi] = shared_prefix_sum[private_data.bi_bcf];
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

    var exclusive_part: u32 = 0u;

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

        // Global last index.
        let last_index = (THREAD_COUNT * 2u) - 1u + ((THREAD_COUNT * 2u - 1u) >> 4u);

        // Copy the last prefix sum to the shared_aux. 
        // local_exclusive_part = shared_prefix_sum[last_index];
        //++ temp_prefix_sum[fmm_prefix_params.exclusive_parts_start_index + private_data.global_ai] = shared_prefix_sum[last_index];
        temp_prefix_sum[fmm_prefix_params.exclusive_parts_start_index + workgroup_index] = shared_prefix_sum[last_index];

        // Add zero to the last index.
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

        // Global last index.
        let last_index = (THREAD_COUNT * 2u) - 1u + ((THREAD_COUNT * 2u - 1u) >> 4u);

        // Copy the last prefix sum to the shared_aux. 
        stream_compaction_count = stream_compaction_count + shared_aux[last_index];

        // Add zero to the last index.
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

// The stream_compaction_count must be solved before calling this function.
fn gather_data() {

    let data_count = fmm_prefix_params.data_end_index -
                     fmm_prefix_params.data_start_index;

    let chunks = udiv_up_safe32(data_count, THREAD_COUNT * 2u);

    for (var i: u32 = 0u; i < chunks ; i = i + 1u) {

        let index_a = private_data.ai + i * THREAD_COUNT * 2u;
        let index_b = private_data.bi + i * THREAD_COUNT * 2u;

        let a = fmm_blocks[index_a];
        let b = fmm_blocks[index_b];

        let a_offset = temp_prefix_sum[index_a];
        let b_offset = temp_prefix_sum[index_b];

        let predicate_a = index_a < data_count && a.band_points_count > 0u;
        let predicate_b = index_b < data_count && b.band_points_count > 0u;

        if (predicate_a) {
            filtered_blocks[a_offset] = a;
        }
        if (predicate_b) {
            filtered_blocks[b_offset] = b;
        }
    }
}

@stage(compute)
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    // if (local_index == 0u) {
    // }
    // workgroupBarrier();

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

    //++ let fmm_block = fmm_blocks[global_ai];
    //++ let fmm_block2 = fmm_blocks[global_bi];

    //++ let start_pos = vec4<f32>(f32(private_data.global_ai) * X_FACTOR, 0.0, 22.0, 0.0);
    //++ base_position = start_pos;

    //++ create_number(f32(fmm_block.index),              base_position - vec4<f32>(0.0, 0.5, 0.0, 0.0));
    //++ create_number(f32(fmm_block.band_points_count),  base_position - vec4<f32>(0.0, 0.75, 0.0, 0.0));

    //++ create_number(f32(fmm_block2.index),             base_position - vec4<f32>(0.0, 0.5,  0.0, 0.0) + X_OFFSET);
    //++ create_number(f32(fmm_block2.band_points_count), base_position - vec4<f32>(0.0, 0.75, 0.0, 0.0) + X_OFFSET);

    //++ base_position = start_pos + vec4<f32>(0.0, 1.0, 0.0, 0.0);

    //++ // TODO: min of ... 
    //++ copy_block_to_shared_temp(THREAD_COUNT * 2u);

    //++ // copy_prefix_sum_to_temp();
    //++ local_prefix_sum();

    //++ store_block_to_global_temp(THREAD_COUNT * 2u);

    //++ let shared_sum  = shared_prefix_sum[private_data.ai_bcf]; 
    //++ let shared_sum2 = shared_prefix_sum[private_data.bi_bcf]; 

    //++ base_position = base_position + vec4<f32>(0.0, 0.5, 0.0, 0.0);
    //++ create_number(f32(shared_sum),   base_position); 
    //++ create_number(f32(shared_sum2),  base_position + X_OFFSET); 

    //++ base_position = base_position + vec4<f32>(0.0, 0.5, 0.0, 0.0);

    // if (local_index == 0u) {

    //     create_number(f32(local_exclusive_part), base_position);

    // }
    // Prefix sum

    // reduce

    // STAGE 1.
    if (fmm_prefix_params.stage == 1u) {

        // 1a. This is done using multiple dispatches. This creates the 0,1 data (has band points) to shared_prefix_sum.
        // This could also be done using loop.
        copy_block_to_shared_temp(THREAD_COUNT * 2u);

        // 1b. Perform prefix sum on shared_temp for the previous data. Copies the exclusive part to the global array (temp_prefix_sum).   
        local_prefix_sum(work_group_id.x);
        
        // 1c. Save the prefix sum to global array (temp_prefix_sum).
        store_block_to_global_temp(THREAD_COUNT * 2u);
    }

    // STAGE 2.
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

      //if (local_index == 0u) {
      //    // create_number(123.3, vec4<f32>(0.0, 0.0, 0.0, 0.0)); 
      //    create_number(f32(stream_compaction_count), vec4<f32>(0.0, 0.0, 0.0, 0.0)); 
      //}
    }

    // STAGE 3.
    // else if (fmm_prefix_params.stage == 3u) {

    //     // 3a. Sum the exclusive parts to the actual prefix sum.
    //     // sum_auxiliar();

    //     // 3b. Gather the filtered data.
    // }
}
