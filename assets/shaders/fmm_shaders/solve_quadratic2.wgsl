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

// Binding id from where to load data.
struct PushConstants {
    buffer_id: u32,    
};

@group(0) @binding(0) var<uniform>            prefix_params: PrefixParams;
@group(0) @binding(1) var<uniform>            fmm_params:    FmmParams;
@group(0) @binding(2) var<storage,read_write> fmm_blocks: array<FmmBlock>;
@group(0) @binding(3) var<storage,read_write> temp_prefix_sum: array<u32>;
@group(0) @binding(4) var<storage,read_write> temp_data: array<u32>;
@group(0) @binding(5) var<storage,read_write> fmm_data: array<FmmCellPc>;
@group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>; // 5 placeholders

// Push constants.
var<push_constant> pc: PushConstants;

fn total_cell_count() -> u32 {

    return fmm_params.global_dimension.x * 
           fmm_params.global_dimension.y * 
           fmm_params.global_dimension.z * 
           fmm_params.local_dimension.x * 
           fmm_params.local_dimension.y * 
           fmm_params.local_dimension.z; 
};

/// A function that checks if a given coordinate is within the global computational domain. 
fn isInside(coord: vec3<i32>) -> bool {
    return (coord.x >= 0 && coord.x < i32(fmm_params.local_dimension.x * fmm_params.global_dimension.x)) &&
           (coord.y >= 0 && coord.y < i32(fmm_params.local_dimension.y * fmm_params.global_dimension.y)) &&
           (coord.z >= 0 && coord.z < i32(fmm_params.local_dimension.z * fmm_params.global_dimension.z)); 
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

/// Get memory index from given cell coordinate.
fn get_cell_mem_location(v: vec3<u32>) -> u32 {

    let stride = fmm_params.local_dimension.x * fmm_params.local_dimension.y * fmm_params.local_dimension.z;

    let global_coordinate = vec3<u32>(v) / fmm_params.local_dimension;

    let global_index = (global_coordinate.x +
                        global_coordinate.y * fmm_params.global_dimension.x +
                        global_coordinate.z * fmm_params.global_dimension.x * fmm_params.global_dimension.y) * stride;

    let local_coordinate = vec3<u32>(v) - global_coordinate * fmm_params.local_dimension;

    let local_index = encode3Dmorton32(local_coordinate.x, local_coordinate.y, local_coordinate.z);

    return global_index + local_index;
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
                                  select(10000.0, n0.value, n0.tag == KNOWN),
                                  select(10000.0, n1.value, n1.tag == KNOWN),
                                  select(10000.0, n2.value, n2.tag == KNOWN),
                                  select(10000.0, n3.value, n3.tag == KNOWN),
                                  select(10000.0, n4.value, n4.tag == KNOWN),
                                  select(10000.0, n5.value, n5.tag == KNOWN) 
    );


    var p = vec3<f32>(min(phis[0], phis[1]), min(phis[2], phis[3]), min(phis[4], phis[5]));
    // var p_sorted: vec3<f32>;

    var tmp: f32;

    if p[1] < p[0] {
        tmp = p[0];
        p[0] = p[1];
        p[1] = tmp;
    }
    if p[2] < p[0] {
        tmp = p[0];
        p[0] = p[2];
        p[2] = tmp;
    }
    if p[2] < p[1] {
        tmp = p[1];
        p[1] = p[2];
        p[2] = tmp;
    }
    // if p[1] < p[0] {
    //     tmp = p[1];
    //     p[1] = p[0];
    //     p[0] = tmp;
    // }
    // if p[2] < p[0] {
    //     tmp = p[2];
    //     p[2] = p[0];
    //     p[0] = tmp;
    // }
    // if p[2] < p[1] {
    //     tmp = p[2];
    //     p[2] = p[1];
    //     p[1] = tmp;
    // }

    var result: f32 = 777.0;

    if (abs(p[0] - p[2]) < 1.0) {
        var phi_sum = p[0] + p[1] + p[2];
        var phi_sum2 = pow(p[0], 2.0) + pow(p[1], 2.0) + pow(p[2], 2.0);
        var phi_sum_pow2 = pow(phi_sum, 2.0);
        result = 1.0/6.0 * (2.0 * phi_sum + sqrt(4.0 * phi_sum_pow2 - 12.0 * (phi_sum2 - 1.0)));
        //result = 1.0/6.0 * (2.0 * phi_sum + sqrt(4.0 * phi_sum_pow2 - 12.0 * (phi_sum_pow2 - 1.0)));
    }

    else if (abs(p[0] - p[1]) < 1.0) {
        result = 0.5 * (p[0] + p[1] + sqrt(2.0 * 1.0 - pow((p[0] - p[1]), 2.0)));
    }
 
    else {
        result = p[0] + 1.0;
    }

    return result;
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

@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

	let band_point_count = fmm_counter[1];

        if (global_id.x < band_point_count) {
	    var t = temp_data[global_id.x]; // The second time. TODO: something better.

	    var fmm_cell = fmm_data[t];
	    var this_coord = get_cell_index(t);
            fmm_cell.value = solve_quadratic(this_coord);
            fmm_cell.tag = BAND;
            //let fmm_value = solve_quadratic(this_coord);
	    fmm_data[t] = fmm_cell; 

    //tag: atomic<u32>,
    //value: f32,
    //color: u32,
	}
}
