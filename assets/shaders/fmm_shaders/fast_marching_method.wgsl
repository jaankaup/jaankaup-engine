// Band conflict free memory counts.

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

// Algorithm states.

// Phase 0 -> Prefix Sum for active blocks part 1 
// Phase 1 -> Prefix Sum for active blocks part 2

let AABB_SIZE = 16.00;

struct PushConstants {
    phase: u32,    
};

// GpuDebugger

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

struct PrivateData {
    ai: u32,
    bi: u32,
    ai_bcf: u32,
    bi_bcf: u32,
    global_ai: u32,
    global_bi: u32,
};

// Thread private data for prefix sum.
struct PrivatePrefixSum {
    ai: u32,         // Thread index
    bi: u32,         // Thread index x 2
    ai_bcf: u32,     // Bank conflict free thread index.
    bi_bcf: u32,     // Bank conflict free thread index x 2.
    global_ai: u32,  // Global index. 
    global_bi: u32,  // Global index x 2. 
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

struct FmmCellPc {
    tag: atomic<u32>,
    value: f32,
    color: u32,
};

struct FmmBlock {
    index: u32,
    number_of_band_points: atomic<u32>,
};

struct TempData {
    data0: u32,
    data1: u32,
};

let FAR      = 0u;
let BAND_NEW = 1u;
let BAND     = 2u;
let KNOWN    = 3u;
let OUTSIDE  = 4u;

@group(0) @binding(0) var<uniform>            prefix_params: PrefixParams;
@group(0) @binding(1) var<uniform>            fmm_params:    FmmParams;
@group(0) @binding(2) var<storage,read_write> fmm_blocks: array<FmmBlock>;
@group(0) @binding(3) var<storage,read_write> temp_prefix_sum: array<u32>;
@group(0) @binding(4) var<storage,read_write> temp_data: array<TempData>;
@group(0) @binding(5) var<storage,read_write> fmm_data: array<FmmCellPc>;
@group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>; // 5 placeholders

// GpuDebugger.
@group(1) @binding(0) var<storage,read_write> counter: array<atomic<u32>>;
@group(1) @binding(1) var<storage,read_write> output_char: array<Char>;
@group(1) @binding(2) var<storage,read_write> output_arrow: array<Arrow>;
@group(1) @binding(3) var<storage,read_write> output_aabb: array<AABB>;
@group(1) @binding(4) var<storage,read_write> output_aabb_wire: array<AABB>;

let THREAD_COUNT = 1024u;
let SCAN_BLOCK_SIZE = 2176u;

// Push constants.
var<push_constant> pc: PushConstants;

// The output of global active fmm block scan.
var<workgroup> shared_prefix_sum: array<u32, SCAN_BLOCK_SIZE>; 
var<workgroup> shared_aux: array<u32, SCAN_BLOCK_SIZE>;
var<workgroup> workgroup_cells: array<FmmCellPc, 1024u>;
var<workgroup> shared_counter: atomic<u32>; //, SCAN_BLOCK_SIZE>;

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

fn total_block_count() -> u32 {

    return fmm_params.global_dimension.x * 
           fmm_params.global_dimension.y * 
           fmm_params.global_dimension.z; 

}

fn total_local_block_size() -> u32 {

    return fmm_params.local_dimension.x * 
           fmm_params.local_dimension.y * 
           fmm_params.local_dimension.z; 

}

fn total_cell_count() -> u32 {

    return fmm_params.global_dimension.x * 
           fmm_params.global_dimension.y * 
           fmm_params.global_dimension.z * 
           fmm_params.local_dimension.x * 
           fmm_params.local_dimension.y * 
           fmm_params.local_dimension.z; 
};


///////////////////////////
////// Prefix scan   //////
///////////////////////////

fn store_block_to_global_temp() {

    temp_prefix_sum[private_data.global_ai] = shared_prefix_sum[private_data.ai_bcf];
    temp_prefix_sum[private_data.global_bi] = shared_prefix_sum[private_data.bi_bcf];
}

// Copies FMM_Block to shared_prefix_sum {0,1}. "Add padding 0 if necessery." not implemented.
fn copy_block_to_shared_temp() {

    var a = fmm_blocks[private_data.global_ai];
    var b = fmm_blocks[private_data.global_bi];

    let end_index = total_cell_count();

    // Add prediates here {0 :: false, 1 :: true).
    shared_prefix_sum[private_data.ai_bcf] = select(0u, 1u, private_data.global_ai < end_index && a.number_of_band_points > 0u);
    shared_prefix_sum[private_data.bi_bcf] = select(0u, 1u, private_data.global_bi < end_index && b.number_of_band_points > 0u);
}

fn copy_exclusive_data_to_shared_aux() {

    let data_count = prefix_params.exclusive_parts_end_index -
                     prefix_params.exclusive_parts_start_index;

    let data_a = temp_prefix_sum[prefix_params.exclusive_parts_start_index + private_data.ai];
    let data_b = temp_prefix_sum[prefix_params.exclusive_parts_start_index + private_data.bi];

    // TODO: remove select.
    shared_aux[private_data.ai_bcf] = select(0u, data_a, private_data.ai < data_count);
    shared_aux[private_data.bi_bcf] = select(0u, data_b, private_data.bi < data_count);
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

// Perform prefix sum for shared_aux.
fn local_prefix_sum_aux() {

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
	fmm_counter[1] = stream_compaction_count; 

        shared_aux[last_index] = 0u;
    }
    
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

    let data_count = total_cell_count(); //  prefix_params.data_end_index -
                                         // prefix_params.data_start_index;

    let chunks = udiv_up_safe32(data_count, THREAD_COUNT * 2u);

    if (private_data.ai == 0u) {
        let last_index = chunks + 1u;
        stream_compaction_count = shared_aux[last_index + (last_index >> 4u)];

	// Store total number of filtered objects to fmm_counter[1].
	atomicStore(&fmm_counter[1], stream_compaction_count);
    } 
    storageBarrier();
    //workgroupBarrier();

    for (var i: u32 = 0u; i < chunks ; i = i + 1u) {

        var a = temp_prefix_sum[private_data.ai + i * THREAD_COUNT * 2u];
        temp_prefix_sum[private_data.ai + i * THREAD_COUNT * 2u] = a + shared_aux[i + (i >> 4u)];

        var b = temp_prefix_sum[private_data.bi + i * THREAD_COUNT * 2u];
        temp_prefix_sum[private_data.bi + i * THREAD_COUNT * 2u] = b + shared_aux[i + (i >> 4u)];
     }
};

fn gather_data() {

    let data_count = total_cell_count(); // prefix_params.data_end_index -
                                         // prefix_params.data_start_index;

    let chunks = udiv_up_safe32(data_count, THREAD_COUNT * 2u);

    for (var i: u32 = 0u; i < chunks ; i = i + 1u) {

        let index_a = private_data.ai + i * THREAD_COUNT * 2u;
        let index_b = private_data.bi + i * THREAD_COUNT * 2u;

        var a = fmm_blocks[index_a];
        let a_offset = temp_prefix_sum[index_a];
        var predicate_a = index_a < data_count && a.number_of_band_points > 0u;
        if (predicate_a) { temp_data[a_offset] = TempData(a.index, a.number_of_band_points); }

        var b = fmm_blocks[index_b];
        let b_offset = temp_prefix_sum[index_b];
        var predicate_b = index_b < data_count && b.number_of_band_points > 0u;
        if (predicate_b) { temp_data[b_offset] = TempData(b.index, b.number_of_band_points); }
    }
}

fn encode3Dmorton32(x: u32, y: u32, z: u32) -> u32 {
    var x_temp = (x      | (x      << 16u)) & 0x030000FFu;
        x_temp = (x_temp | (x_temp <<  8u)) & 0x0300F00Fu;
        x_temp = (x_temp | (x_temp <<  4u)) & 0x030C30C3u;
        x_temp = (x_temp | (x_temp <<  2u)) & 0x09249249u;

    var y_temp = (y      | (y      << 16u)) & 0x030000FFu;
        y_temp = (y_temp | (y_temp <<  8u)) & 0x0300F00Fu;
        y_temp = (y_temp | (y_temp <<  4u)) & 0x030C30C3u;
        y_temp = (y_temp | (y_temp <<  2u)) & 0x09249249u;

    var z_temp = (z      | (z      << 16u)) & 0x030000FFu;
        z_temp = (z_temp | (z_temp <<  8u)) & 0x0300F00Fu;
        z_temp = (z_temp | (z_temp <<  4u)) & 0x030C30C3u;
        z_temp = (z_temp | (z_temp <<  2u)) & 0x09249249u;

    return x_temp | (y_temp << 1u) | (z_temp << 2u);
}

fn get_third_bits32(m: u32) -> u32 {
    var x = m & 0x9249249u;
    x = (x ^ (x >> 2u))  & 0x30c30c3u;
    x = (x ^ (x >> 4u))  & 0x0300f00fu;
    x = (x ^ (x >> 8u))  & 0x30000ffu;
    x = (x ^ (x >> 16u)) & 0x000003ffu;

    return x;
}

fn decode3Dmorton32(m: u32) -> vec3<u32> {
    return vec3<u32>(
        get_third_bits32(m),
        get_third_bits32(m >> 1u),
        get_third_bits32(m >> 2u)
   );
}

/// xy-plane indexing. (x,y,z) => index
fn index_to_uvec3(index: u32, dim_x: u32, dim_y: u32) -> vec3<u32> {
  var x  = index;
  let wh = dim_x * dim_y;
  let z  = x / wh;
  x  = x - z * wh; // check
  let y  = x / dim_x;
  x  = x - y * dim_x;
  return vec3<u32>(x, y, z);
}

fn gather_cells_to_temp_data(tag: u32, thread_index: u32) {

    var fmm_cell = fmm_data[thread_index];

    if (fmm_cell.tag == tag) {

        let index = atomicAdd(&shared_counter, 1u);

        // Save the cell index to temp_data.
        temp_data[index] = TempData(thread_index, 777u);
    }
}

/// A function that checks if a given coordinate is within the global computational domain. 
fn isInside(coord: vec3<i32>) -> bool {
    return (coord.x >= 0 && coord.x < i32(fmm_params.local_dimension.x * fmm_params.global_dimension.x)) &&
           (coord.y >= 0 && coord.y < i32(fmm_params.local_dimension.y * fmm_params.global_dimension.y)) &&
           (coord.z >= 0 && coord.z < i32(fmm_params.local_dimension.z * fmm_params.global_dimension.z)); 
}

/// Get group coordinate based on cell memory index.
fn get_group_coordinate(global_index: u32) -> vec3<u32> {

    let stride = fmm_params.local_dimension.x * fmm_params.local_dimension.y * fmm_params.local_dimension.z;
    let block_index = global_index / stride;

    return index_to_uvec3(block_index, fmm_params.global_dimension.x, fmm_params.global_dimension.y);
}

/// Get memory index from given cell coordinate.
fn get_cell_mem_location(v: vec3<u32>) -> u32 {

    let stride = fmm_params.local_dimension.x * fmm_params.local_dimension.y * fmm_params.local_dimension.z;

    // let yOffset = fmm_params.global_dimension.x;
    // let zOffset = fmm_params.global_dimension.x * fmm_params.global_dimension.y;

    let global_coordinate = vec3<u32>(v) / fmm_params.local_dimension;

    let global_index = (global_coordinate.x +
                        global_coordinate.y * fmm_params.global_dimension.x +
                        global_coordinate.z * fmm_params.global_dimension.x * fmm_params.global_dimension.y) * stride;

    let local_coordinate = vec3<u32>(v) - global_coordinate * fmm_params.local_dimension;

    let local_index = encode3Dmorton32(local_coordinate.x, local_coordinate.y, local_coordinate.z);

    return global_index + local_index;
}

/// Get cell index based on domain dimension.
fn get_cell_index(global_index: u32) -> vec3<u32> {

    let stride = fmm_params.local_dimension.x * fmm_params.local_dimension.y * fmm_params.local_dimension.z;
    let block_index = global_index / stride;
    let block_position = index_to_uvec3(block_index, fmm_params.global_dimension.x, fmm_params.global_dimension.y) * fmm_params.local_dimension;

    // Calculate local position.
    let local_index = global_index - block_index * stride;

    let local_position = decode3Dmorton32(local_index);

    let cell_position = block_position + local_position;

    return cell_position; 
}

fn load_neighbors_6(coord: vec3<u32>) -> array<u32, 6> {

    var neighbors: array<vec3<i32>, 6> = array<vec3<i32>, 6>(
        vec3<i32>(coord) + vec3<i32>(-1,  0,  0),
        vec3<i32>(coord) + vec3<i32>(1,   0,  0),
        vec3<i32>(coord) + vec3<i32>(0,   1,  0),
        vec3<i32>(coord) + vec3<i32>(0,  -1,  0),
        vec3<i32>(coord) + vec3<i32>(0,   0,  1),
        vec3<i32>(coord) + vec3<i32>(0,   0, -1)
    );

    let i0 = get_cell_mem_location(vec3<u32>(neighbors[0]));
    let i1 = get_cell_mem_location(vec3<u32>(neighbors[1]));
    let i2 = get_cell_mem_location(vec3<u32>(neighbors[2]));
    let i3 = get_cell_mem_location(vec3<u32>(neighbors[3]));
    let i4 = get_cell_mem_location(vec3<u32>(neighbors[4]));
    let i5 = get_cell_mem_location(vec3<u32>(neighbors[5]));

    // The index of the "outside" cell.
    let tcc = total_cell_count();

    return array<u32, 6>(
        select(tcc ,i0, isInside(neighbors[0])),
        select(tcc ,i1, isInside(neighbors[1])),
        select(tcc ,i2, isInside(neighbors[2])),
        select(tcc ,i3, isInside(neighbors[3])),
        select(tcc ,i4, isInside(neighbors[4])),
        select(tcc ,i5, isInside(neighbors[5])) 
    );
}

fn solve_quadratic(coord: vec3<u32>) -> f32 {

    var memory_locations = load_neighbors_6(coord);

    var n0 = fmm_data[memory_locations[0]];
    var n1 = fmm_data[memory_locations[1]];
    var n2 = fmm_data[memory_locations[2]];
    var n3 = fmm_data[memory_locations[3]];
    var n4 = fmm_data[memory_locations[4]];
    var n5 = fmm_data[memory_locations[5]];

    var phis: array<f32, 6> = array<f32, 6>(
                                  select(1000000.0, n0.value, n0.tag == KNOWN),
                                  select(1000000.0, n1.value, n1.tag == KNOWN),
                                  select(1000000.0, n2.value, n2.tag == KNOWN),
                                  select(1000000.0, n3.value, n3.tag == KNOWN),
                                  select(1000000.0, n4.value, n4.tag == KNOWN),
                                  select(1000000.0, n5.value, n5.tag == KNOWN) 
    );


    var p = vec3<f32>(min(phis[0], phis[1]), min(phis[2], phis[3]), min(phis[4], phis[5]));
    var p_sorted: vec3<f32>;

    var tmp: f32;

    if p[1] < p[0] {
        tmp = p[1];
        p[1] = p[0];
        p[0] = tmp;
    }
    if p[2] < p[0] {
        tmp = p[2];
        p[2] = p[0];
        p[0] = tmp;
    }
    if p[2] < p[1] {
        tmp = p[2];
        p[2] = p[1];
        p[1] = tmp;
    }

    var result = 777.0;

    if (abs(p[0] - p[2]) < 1.0) {
        var phi_sum = p[0] + p[1] + p[2];
        var phi_sum_pow2 = pow(phi_sum, 2.0);
        result = 1.0/6.0 * (2.0 * phi_sum + sqrt(4.0 * phi_sum_pow2 - 12.0 * (phi_sum_pow2 - 1.0))); 
        // result = 123.0;
    }

    else if (abs(p[0] - p[1]) < 1.0) {
        result = 0.5 * (p[0] + p[1] + sqrt(2.0 * 1.0 - pow((p[0] - p[1]), 2.0)));
        // result = 555.0;
    }
 
    else {
        result = p[0] + 1.0;
    }

    return result;
}

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

fn visualize_block(position: vec3<f32>, color: u32) {
    output_aabb_wire[atomicAdd(&counter[3], 1u)] =
          AABB (
              //vec4<f32>(position, bitcast<f32>(color)),
              //vec4<f32>(12.0 * position,  2.5)
              4.0 * vec4<f32>(position, 0.0) + vec4<f32>(vec3<f32>(0.002), bitcast<f32>(color)),
              4.0 * vec4<f32>(position, 0.0) + vec4<f32>(vec3<f32>(AABB_SIZE - 0.002), 0.2)
          );
}

@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {


    // STAGE 1.
    // Create scan data and store it to global memory.
    if (pc.phase == 0u) {

        create_prefix_sum_private_data(local_index, workgroup_id.x);
        copy_block_to_shared_temp();
        local_prefix_sum(workgroup_id.x);
	workgroupBarrier();
        store_block_to_global_temp();
    }

    // STAGE 2.
    // Do the actual prefix sum and the filtered data to the final destination array.
    else if (pc.phase == 1u) {

        create_prefix_sum_private_data(local_index, workgroup_id.x);
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

    else if (pc.phase == 2u || pc.phase == 4u) {

        // Initialize shader_counter;
	if (local_index == 0u) { shared_counter = 0u; fmm_counter[0] = 0u; }
        workgroupBarrier();

        let cell_count = total_cell_count();
        let chuncks = udiv_up_safe32(cell_count, 1024u);

        // if (global_id.x < cell_count) {
        //     gather_cells_to_temp_data(select(BAND,KNOWN, pc.phase == 2u) , global_id.x);
	// }

        for (var i: u32 = 0u; i < chuncks; i = i + 1u) {

            let actual_index = local_index + 1024u * i;

            if (actual_index < cell_count) {
                gather_cells_to_temp_data(select(BAND,KNOWN, pc.phase == 2u) , actual_index);
	    }
        }
        //workgroupBarrier();
        //storageBarrier();

	if (local_index == 0u) {
	    atomicStore(&fmm_counter[0], shared_counter);
        }
    }

    // Expand interface from "known" cells. Add new band points to fmm_temp.
    else if (pc.phase == 3u) {

        // Get the number of known points;
	let known_point_count = fmm_counter[0];
        let chuncks = udiv_up_safe32(known_point_count, 1024u);

        for (var i: u32 = 0u; i < chuncks; i = i + 1u) {

            let actual_index = local_index + 1024u * i;

	    if (actual_index < known_point_count) {

                // Get the cell index.
	        let t = temp_data[actual_index];  

	        // get_neighbors.
		var this_coord = get_cell_index(t.data0);

                var memory_locations: array<u32, 6> = load_neighbors_6(this_coord);

		// Neighbors.
		var n0 = fmm_data[memory_locations[0]];
		var n1 = fmm_data[memory_locations[1]];
		var n2 = fmm_data[memory_locations[2]];
		var n3 = fmm_data[memory_locations[3]];
		var n4 = fmm_data[memory_locations[4]];
		var n5 = fmm_data[memory_locations[5]];
		if (n0.tag == FAR) {
		    let old_tag = atomicExchange(&fmm_data[memory_locations[0]].tag, BAND);
		    if (old_tag == FAR) { atomicAdd(&fmm_blocks[memory_locations[0] / 64u].number_of_band_points, 1u); }
	        }
		if (n1.tag == FAR) {
		    let old_tag = atomicExchange(&fmm_data[memory_locations[1]].tag, BAND);
		    if (old_tag == FAR) { atomicAdd(&fmm_blocks[memory_locations[1] / 64u].number_of_band_points, 1u); }
	        }
		if (n2.tag == FAR) {
		    let old_tag = atomicExchange(&fmm_data[memory_locations[2]].tag, BAND);
		    if (old_tag == FAR) { atomicAdd(&fmm_blocks[memory_locations[2] / 64u].number_of_band_points, 1u); }
	        }
		if (n3.tag == FAR) {
		    let old_tag = atomicExchange(&fmm_data[memory_locations[3]].tag, BAND);
		    if (old_tag == FAR) { atomicAdd(&fmm_blocks[memory_locations[3] / 64u].number_of_band_points, 1u); }
	        }
		if (n4.tag == FAR) {
		    let old_tag = atomicExchange(&fmm_data[memory_locations[4]].tag, BAND);
		    if (old_tag == FAR) { atomicAdd(&fmm_blocks[memory_locations[4] / 64u].number_of_band_points, 1u); }
	        }
		if (n5.tag == FAR) {
		    let old_tag = atomicExchange(&fmm_data[memory_locations[5]].tag, BAND);
		    if (old_tag == FAR) { atomicAdd(&fmm_blocks[memory_locations[5] / 64u].number_of_band_points, 1u); }
	        }
		// if (n1.tag == FAR) { fmm_data[memory_locations[1]].tag = BAND; }
		// if (n2.tag == FAR) { fmm_data[memory_locations[2]].tag = BAND; }
		// if (n3.tag == FAR) { fmm_data[memory_locations[3]].tag = BAND; }
		// if (n4.tag == FAR) { fmm_data[memory_locations[4]].tag = BAND; }
		// if (n5.tag == FAR) { fmm_data[memory_locations[5]].tag = BAND; }
	    }
        }
    }

    // Solve quadratic.
    else if (pc.phase == 5u) {

	let band_point_count = fmm_counter[0];
        let chuncks = udiv_up_safe32(band_point_count, 1024u);

        for (var i: u32 = 0u; i < chuncks; i = i + 1u) {

            let actual_index = local_index + 1024u * i;

	    if (actual_index < band_point_count) {

                // Get the cell index.
	        let t = temp_data[actual_index];  
		var this_coord = get_cell_index(t.data0);
                let fmm_value = solve_quadratic(this_coord);
		fmm_data[t.data0].value = fmm_value;
	    }
	}
    }
    // Visualize block
    else if (pc.phase == 15u) {

        // Initialize shared_counter;
	// if (local_index == 0u) { shared_counter = 0u; }
        // workgroupBarrier();

	let block_count = fmm_counter[0];
        let chuncks = udiv_up_safe32(block_count, 1024u);

        for (var i: u32 = 0u; i < chuncks; i = i + 1u) {

            let actual_index = local_index + 1024u * i;

	    if (actual_index < block_count) {
                let color_band = rgba_u32(222u, 55u, 150u, 255u);
		let block_index = temp_data[actual_index].data0;

                let block_position = index_to_uvec3(block_index, fmm_params.global_dimension.x, fmm_params.global_dimension.y) * fmm_params.local_dimension;
                visualize_block(vec3<f32>(block_position), color_band); 
            }
	}
    }

        
        // let block_count = total_block_count();
        // let stride = total_local_block_size();
	// let blocks_per_thread_group = 1024u / stride; 
        // let chuncks = udiv_up_safe32(block_count, 1024u);

        // for (var i: u32 = 0u; i < chunks; i = i + 1u) {

        //     // Load max 1024 cells to shared memory.
	//     let index = local_index + i * 1024u;
	//     let block_index = index / stride;
	//     let fmm_cell = fmm_data[index];
	//     if (fmm_cell.tag == BAND) {
        //         atomicAdd(&fmm_blocks[block_index].number_of_band_points, 1u);
	//     }
	//     // workgroup_cells[local_index] = fmm_data[index];
	//     // workgroupBarier();

	//     // let actual_index = local_index + stride * chunks; // & (stride - 1); 

        //     // let block = fmm_blocks[local_index
        // }

        //++ let fmm_cell = fmm_data[local_index + workgroup_id.x * 64u];    
        //++ if (fmm_cell.tag == BAND) {
        //++     atomicAdd(&fmm_blocks[workgroup_id.x].number_of_band_points, 1u);
	//++ }
	//++ storageBarrier();

	//++ if (local_index == 0u && fmm_blocks[workgroup_id.x].number_of_band_points > 0u) {
        //++     var position = vec3<f32>(get_group_coordinate(global_id.x)) * 4.0;
        //++     let color_band = rgba_u32(222u, 55u, 150u, 255u);
        //++     visualize_cell(position, color_band);
	//++ }
}
