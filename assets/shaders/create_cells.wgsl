/// Create the initial fast marching method cells.

let MORTON32 = 0u;  
let HILBERT = 1u;  
let CUBE_SLICE = 2u; 

struct FmmCreationParams {
    fmm_global_dimension: vec3<u32>, 
    fmm_global_indexing: u32,
    fmm_inner_dimension: vec3<u32>, 
    fmm_inner_indexing: u32,
};

struct FmmCell {
    tag: atomic<u32>,
    value: f32,
    update: atomic<u32>,
    misc: u32,
};

// Debugging.

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

@group(0)
@binding(0)
var<uniform> fmm_creation_params: FmmCreationParams;

@group(0)
@binding(1)
var<storage, read_write> fmm_data: array<FmmCell>;

// Debugging.
@group(0)
@binding(2)
var<storage, read_write> counter: array<atomic<u32>>;

@group(0)
@binding(3)
var<storage,read_write> output_char: array<Char>;

@group(0)
@binding(4)
var<storage,read_write> output_arrow: array<Arrow>;

@group(0)
@binding(5)
var<storage,read_write> output_aabb: array<AABB>;

@group(0)
@binding(6)
var<storage,read_write> output_aabb_wire: array<AABB>;

// Debugging.
fn rgba_u32(r: u32, g: u32, b: u32, a: u32) -> u32 {
  return (r << 24u) | (g << 16u) | (b  << 8u) | a;
}

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

fn udiv_up_safe32(x: u32, y: u32) -> u32 {
    let tmp = (x + y - 1u) / y;
    return select(tmp, 0u, y == 0u); 
}

fn isInside(coord: ptr<function, vec3<i32>>) -> bool {
    let max_coord = ___fmm_global_dimension * ___fmm_local_dimension;
    return ((*coord).x >= 0 && (*coord).x < max_coord.x) &&
           ((*coord).y >= 0 && (*coord).y < max_coord.y) &&
           ((*coord).z >= 0 && (*coord).z < max_coord.z); 
}

@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    // Total number of fmm blocks.
    let block_count = fmm_global_dimension.x * fmm_global_dimension.y * fmm_global_dimension.z;

    let stride = fmm_inner_dimension.x * fmm_inner_dimension.y * fmm_inner_dimension.z;

    // Total number of fmm cells.
    let cell_count = block_count * stride;

    if (global_id < cell_count) { return; }

    let block_index = global_id / stride; 

    // Calculate the block position. 
    var block_position : vec32<u32>;

    if (___fmm_global_indexing == MORTON32) {
        block_position = decode3Dmorton32(block_index);
    }
    else if (___fmm_global_indexing == CUBE_SLICE) {
        block_position = index_to_uvec3(___fmm_global_dimension.x, ___fmm_global_dimension.y, block_index);
    }
    
    // Calculate local position.
    let local_index = global_id - block_index * stride;

    // Calculate the local position. 
    var local_position : vec32<u32>;

    if (___fmm_local_indexing == MORTON32) {
        local_position = decode3Dmorton32(local_index);
    }

    else if (___fmm_local_indexing == CUBE_SLICE) {
        local_position = index_to_uvec3(___fmm_local_dimension.x, ___fmm_local_dimension.y, local_index);
    }

    let fmm_cell_position = block_position + local_position * ___fmm_local_dimension;

    // Check the neighbors. 
    var neighbor_0_coord = vec3<i32>(fmm_cell_position) + vec3<i32>(1, 0, 0);
    var neighbor_1_coord = vec3<i32>(fmm_cell_position) - vec3<i32>(1, 0, 0);
    var neighbor_2_coord = vec3<i32>(fmm_cell_position) + vec3<i32>(0, 1, 0);
    var neighbor_3_coord = vec3<i32>(fmm_cell_position) - vec3<i32>(0, 1, 0);
    var neighbor_4_coord = vec3<i32>(fmm_cell_position) + vec3<i32>(0, 0, 1);
    var neighbor_5_coord = vec3<i32>(fmm_cell_position) - vec3<i32>(0, 0, 1);

    let isInside0 = isInside(&neighbor_0_coord);
    let isInside1 = isInside(&neighbor_1_coord);
    let isInside2 = isInside(&neighbor_2_coord);
    let isInside3 = isInside(&neighbor_3_coord);
    let isInside4 = isInside(&neighbor_4_coord);
    let isInside5 = isInside(&neighbor_5_coord);
}
