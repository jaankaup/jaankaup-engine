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
    blaah: u32;
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

let THREAD_COUNT = 64u;
let SCAN_BLOCK_SIZE = 136; 
let X_OFFSET = vec4<f32>(64.0, 0.0, 0.0, 0.0);
let X_FACTOR = 1.0;

// The output of global active fmm block scan.
var<workgroup> shared_prefix_sum: array<u32, SCAN_BLOCK_SIZE>; 
var<workgroup> shared_aux: array<u32, 136>;

// The counter for active fmm blocks.
var<workgroup> stream_compaction_count: u32;
var<workgroup> local_exclusive_part: u32;

var<private> base_position: vec4<f32>;
var<private> base_position_bcf: vec4<f32>;
var<private> private_data: PrivateData;

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

// Add a chunk of block data to a temporary shared_perix_sum buffer.
// Copies FMM_Block to shared_prefix_sum {0,1}. Add padding 0 if necessery.
fn copy_block_to_temp(number_of_items: u32) {

    let a = fmm_blocks[private_data.global_ai];
    let b = fmm_blocks[private_data.global_bi];

    shared_prefix_sum[private_data.ai_bcf] = select(0u, 1u, private_data.ai < number_of_items && a.band_points_count > 0u);
    shared_prefix_sum[private_data.bi_bcf] = select(0u, 1u, private_data.bi < number_of_items && b.band_points_count > 0u);

    create_number(f32(shared_prefix_sum[private_data.ai_bcf]), base_position);
    create_number(f32(shared_prefix_sum[private_data.bi_bcf]), base_position + X_OFFSET);
}

fn copy_prefix_sum_to_temp() {

    temp_prefix_sum[private_data.global_ai] = shared_prefix_sum[private_data.ai_bcf];
    temp_prefix_sum[private_data.global_bi] = shared_prefix_sum[private_data.bi_bcf];
}

// Perform prefix sum. Return the exclusive part of the prefix sum.
// NOTE: only the thread 0 returns the actual exclusive part. NO RETURN!!!
// Instead, the exclusive part is stored to workgroup exclusive_part
fn local_prefix_sum() {

    var exclusive_part: u32 = 0u;

    // Up sweep.

    // let n = (THREAD_COUNT * 2u) >> 1u;
    let n = THREAD_COUNT * 2u;
    var offset = 1u;

    for (var d = n >> 1u ; d > 0u; d = d >> 1u) {
      workgroupBarrier();

      base_position = base_position + vec4<f32>(0.0, 0.5, 0.0, 0.0);

      if (private_data.ai < d) {

          var ai_temp = offset * (private_data.ai * 2u + 1u) - 1u;
          var bi_temp = offset * (private_data.ai * 2u + 2u) - 1u;

          var ai_temp_debug = offset * (private_data.ai * 2u + 1u) - 1u;
          var bi_temp_debug = offset * (private_data.ai * 2u + 2u) - 1u;

          ai_temp = ai_temp + (ai_temp >> 4u);
          bi_temp = bi_temp + (bi_temp >> 4u);

          shared_prefix_sum[bi_temp] = shared_prefix_sum[bi_temp] + shared_prefix_sum[ai_temp];

          create_number(f32(shared_prefix_sum[bi_temp]),
                        vec4<f32>(f32(bi_temp_debug) * X_FACTOR, base_position.y, 22.0, 0.0)
          ); 
          create_arrow(vec4<f32>(f32(ai_temp_debug) * X_FACTOR, base_position.y, 22.0, 0.0),
                       vec4<f32>(f32(bi_temp_debug) * X_FACTOR, base_position.y, 22.0, 0.0)
          );
      }
      offset = offset * 2u; // bit shift?
    }
    workgroupBarrier();
      
    base_position = base_position + vec4<f32>(0.0, 1.5, 0.0, 0.0);
    // Clear the last item. 
    if (private_data.ai == 0u) {

        // Global last index.
        let last_index = (THREAD_COUNT * 2u) - 1u + ((THREAD_COUNT * 2u - 1u) >> 4u);

        // Copy the last prefix sum to the shared_aux. 
        local_exclusive_part = shared_prefix_sum[last_index];

        // Add zero to the last index.
        shared_prefix_sum[last_index] = 0u;


        create_number(f32(shared_prefix_sum[last_index]),
                      vec4<f32>(f32(private_data.global_ai + 2u * THREAD_COUNT - 1u) * X_FACTOR, base_position.y, 22.0, 0.0)
        ); 
        create_arrow(vec4<f32>(f32(private_data.global_ai + 2u * THREAD_COUNT - 1u) * X_FACTOR, base_position.y - 0.5, 22.0, 0.0),
                     vec4<f32>(f32(private_data.global_ai + 2u * THREAD_COUNT - 1u) * X_FACTOR, base_position.y, 22.0, 0.0)
        );
    }
    // workgroupBarrier();
    // BARRIER?
    
    // Down sweep.
    for (var d = 1u; d < n ; d = d * 2u) {
        offset = offset >> 1u;
        workgroupBarrier();
        base_position = base_position + vec4<f32>(0.0, 0.5, 0.0, 0.0);

        if (private_data.ai < d) {
            var ai_temp = offset * (private_data.ai * 2u + 1u) - 1u;
            var bi_temp = offset * (private_data.ai * 2u + 2u) - 1u;
            var ai_temp_debug = offset * (private_data.ai * 2u + 1u) - 1u;
            var bi_temp_debug = offset * (private_data.ai * 2u + 2u) - 1u;
            ai_temp = ai_temp + (ai_temp >> 4u);
            bi_temp = bi_temp + (bi_temp >> 4u);
            var t = shared_prefix_sum[ai_temp];

            shared_prefix_sum[ai_temp] = shared_prefix_sum[bi_temp];
            shared_prefix_sum[bi_temp] = shared_prefix_sum[bi_temp] + t;

            create_number(f32(shared_prefix_sum[ai_temp]),
                          vec4<f32>(f32(ai_temp_debug) * X_FACTOR, base_position.y, 22.0, 0.0)
            ); 
            create_number(f32(shared_prefix_sum[bi_temp]),
                          vec4<f32>(f32(bi_temp_debug) * X_FACTOR, base_position.y, 22.0, 0.0)
            ); 
            create_arrow(vec4<f32>(f32(bi_temp_debug) * X_FACTOR, base_position.y, 21.9, 0.0),
                         vec4<f32>(f32(ai_temp_debug) * X_FACTOR, base_position.y, 21.9, 0.0)
            );
        }
    }
}

// Perform prefix sum for shared_aux. DONT't Return the exclusive part of the prefix sum.
fn local_prefix_sum_aux(thread_id: u32) {

    // Create the local indices. TODO: add these to private attribute.
    let ai = thread_id; 
    let bi = ai + THREAD_COUNT;

    // Create the bank conflict free local indices.
    let ai_bcf = ai + (ai >> 4u); 
    let bi_bcf = bi + (bi >> 4u);

    // var exclusive_part = 0; // TODO: not used! remove.

    // Up sweep.

    let n = THREAD_COUNT * 2u;
    var offset = 1u;
    for (var d = n >> 1u ; d > 0u; d = d >> 1u) {
        workgroupBarrier();
        if (ai < d) {

            var ai_temp = offset * (ai * 2u + 1u) - 1u;
            var bi_temp = offset * (ai * 2u + 2u) - 1u;

            ai_temp = ai_temp + (ai_temp >> 4u);
            bi_temp = bi_temp + (bi_temp >> 4u);

            shared_aux[bi_temp] += shared_aux[ai_temp];
        }
        offset = offset * 2u;
    }
    workgroupBarrier();
      
    // Clear the last item. 
    if (thread_id == 0u) {

        // Global last index.
        let last_index = (THREAD_COUNT * 2u) - 1u + ((THREAD_COUNT * 2u - 1u) >> 4u);

        // Copy the last prefix sum to the shared_aux. 
        stream_compaction_count = shared_aux[last_index];

        // Add zero to the last index.
        shared_aux[last_index] = 0u;
    }
    // BARRIER?
    
    // Down sweep.
    for (var d = 1u; d < n ; d = d * 2u) {
      offset = offset >> 1u;
      workgroupBarrier();
      if (ai < d) {
          var ai_temp = offset * (ai * 2u + 1u) - 1u;
          var bi_temp = offset * (ai * 2u + 2u) - 1u;
          ai_temp = ai_temp + (ai_temp >> 4u);
          bi_temp = bi_temp + (bi_temp >> 4u);
          let t = shared_aux[ai_temp];

          shared_aux[ai_temp] = shared_aux[bi_temp];
          shared_aux[bi_temp] += t;
      }
  }
  // return exclusive_part;
}

//++ fn global_solver(thread_id: u32) {
//++ 
//++   let number_of_blocks = 16u * 16u * 16u;
//++ 
//++   // Reset the counter for active blocks. 
//++   if (thread_id == 0u) { stream_compaction_count = 0u; }
//++   workgroupBarrier();
//++   
//++   // Number of data chunks. There must be at least one iteration.
//++   // neeeded. TODO: check number_of_blocks == 0. Return immediatelly if it
//++   // true.
//++ 
//++   let number_of_scan_blocks = (number_of_blocks - 1u) / (THREAD_COUNT * 2u) + 1u;
//++ 
//++   // Available items count.
//++   var items_available: u32 = number_of_blocks;
//++ 
//++   // The number of items on current chunk.
//++   var number_of_taken_items = 0u; 
//++ 
//++   // Create the local indices.
//++   let ai = thread_id; 
//++   let bi = ai + THREAD_COUNT;
//++ 
//++   // Create the local indices.
//++   let ai_bcf = ai + (ai >> 4u); 
//++   let bi_bcf = bi + (bi >> 4u);
//++   
//++   // Add zeroes to the shared_aux (the conflict free indexes).
//++   shared_aux[ai_bcf] = 0u;
//++   shared_aux[bi_bcf] = 0u;
//++   workgroupBarrier();
//++ 
//++   // Perfom prefix sum on boolean array.
//++   for (var j = 0u ; j < number_of_scan_blocks ; j = j + 1u) {
//++ 
//++       // Determine the count of items in this iteration.
//++       number_of_taken_items = min(THREAD_COUNT * 2u, items_available);
//++ 
//++       // Decrease the number of available items.
//++       items_available = items_available - number_of_taken_items;
//++ 
//++       // Copy data from global FMM_Block array to
//++       // shared_prefix_sum. [{0,1}]
//++       copy_block_to_temp(j, number_of_taken_items, thread_id);
//++       workgroupBarrier(); // ???
//++ 
//++       // Perform the prefix_sum.
//++       local_prefix_sum();
//++       workgroupBarrier(); // ???
//++ 
//++       // Add the exclusive part to the shared aux.
//++       if (thread_id == 0u) { shared_aux[j + (j >> 4u)] = local_exclusive_part; }
//++       workgroupBarrier(); 
//++ 
//++       // Copy local prefix sum data to temp_prefix_sum buffer.
//++       copy_prefix_sum_to_temp(j, thread_id);
//++       workgroupBarrier(); // ???
//++   }
//++ 
//++   // Perform the prefix_sum for shared_aux.
//++   local_prefix_sum_aux(thread_id);
//++   workgroupBarrier(); // ??
//++ 
//++   // Add the shared_aux to the final result.
//++   for (var j = 0u ; j < number_of_scan_blocks ; j = j + 1u) {
//++     let value_to_add = shared_aux[j + (j >> 4u)]; 
//++ 
//++     // Create the global indices to access the global memory.
//++     let global_ai = ai + j * (THREAD_COUNT * 2u);
//++     let global_bi = bi + j * (THREAD_COUNT * 2u);
//++ 
//++     temp_prefix_sum[global_ai] = temp_prefix_sum[global_ai] + value_to_add; 
//++     temp_prefix_sum[global_bi] = temp_prefix_sum[global_bi] + value_to_add; 
//++   }
//++   workgroupBarrier(); // ??
//++ 
//++   // Finally. Update the number of active FMM_Blocks.
//++   // solve_active_fmm_block(number_of_scan_blocks);
//++   // workgroupBarrier(); //??
//++ 
//++   // return stream_compaction_count;
//++ }


@stage(compute)
@workgroup_size(64,1,1)
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

    let fmm_block = fmm_blocks[global_ai];
    let fmm_block2 = fmm_blocks[global_bi];

    let start_pos = vec4<f32>(f32(private_data.global_ai) * X_FACTOR, 0.0, 22.0, 0.0);
    base_position = start_pos;

    create_number(f32(fmm_block.index),              base_position - vec4<f32>(0.0, 0.5, 0.0, 0.0));
    create_number(f32(fmm_block.band_points_count),  base_position - vec4<f32>(0.0, 0.75, 0.0, 0.0));

    create_number(f32(fmm_block2.index),             base_position - vec4<f32>(0.0, 0.5,  0.0, 0.0) + X_OFFSET);
    create_number(f32(fmm_block2.band_points_count), base_position - vec4<f32>(0.0, 0.75, 0.0, 0.0) + X_OFFSET);

    base_position = start_pos + vec4<f32>(0.0, 1.0, 0.0, 0.0);

    // TODO: min of ... 
    copy_block_to_temp(THREAD_COUNT * 2u);

    // copy_prefix_sum_to_temp();
    local_prefix_sum();

    let shared_sum  = shared_prefix_sum[private_data.ai_bcf]; 
    let shared_sum2 = shared_prefix_sum[private_data.bi_bcf]; 

    base_position = base_position + vec4<f32>(0.0, 0.5, 0.0, 0.0);
    create_number(f32(shared_sum),   base_position); 
    create_number(f32(shared_sum2),  base_position + X_OFFSET); 

    // if (local_index == 0u) {


    // }
    // Prefix sum

    // reduce
}
