// GpuDebugger

struct AABB {
    min: vec4<f32>, 
    max: vec4<f32>, 
};

// struct AabbWire {
//     min: vec3<f32>, 
//     color: u32,
//     max: vec3<f32>, 
//     thickness: f32,
// };

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

//let AABB_SIZE = 1.04;
let AABB_SIZE = 16.00;

// Fmm
struct FmmParams {
    global_dimension: vec3<u32>,
    future_usage: u32,
    local_dimension: vec3<u32>,
    future_usage2: u32,
};

struct FmmCellPc {
    tag: u32,
    value: u32,
    color: u32,
};

struct FmmBlock {
    index: u32,
    number_of_band_points: atomic<u32>,
};

let FAR      = 0u;
let BAND_NEW = 1u;
let BAND     = 2u;
let KNOWN    = 3u;
let OUTSIDE  = 4u;

@group(0) @binding(0) var<uniform>            fmm_params:    FmmParams;
@group(0) @binding(1) var<storage,read_write> fmm_data: array<FmmCellPc>;
@group(0) @binding(2) var<storage,read_write> fmm_blocks: array<FmmBlock>;

// GpuDebugger.
@group(1) @binding(0) var<storage,read_write> counter: array<atomic<u32>>;
@group(1) @binding(1) var<storage,read_write> output_char: array<Char>;
@group(1) @binding(2) var<storage,read_write> output_arrow: array<Arrow>;
@group(1) @binding(3) var<storage,read_write> output_aabb: array<AABB>;
@group(1) @binding(4) var<storage,read_write> output_aabb_wire: array<AABB>;

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

/// A function that checks if a given coordinate is within the global computational domain. 
// fn isInside(coord: vec3<i32>) -> bool {
//     return (coord.x >= 0 && coord.x < i32(fmm_params.local_dimension.x * fmm_params.global_dimension.x)) &&
//            (coord.y >= 0 && coord.y < i32(fmm_params.local_dimension.y * fmm_params.global_dimension.y)) &&
//            (coord.z >= 0 && coord.z < i32(fmm_params.local_dimension.z * fmm_params.global_dimension.z)); 
// }

/// Get group coordinate based on cell memory index.
fn get_group_coordinate(global_index: u32) -> vec3<u32> {

    let stride = fmm_params.local_dimension.x * fmm_params.local_dimension.y * fmm_params.local_dimension.z;
    let block_index = global_index / stride;

    return index_to_uvec3(block_index, fmm_params.global_dimension.x, fmm_params.global_dimension.y);
}

/// Get memory index from given cell coordinate.
fn get_cell_mem_location(v: vec3<u32>) -> u32 {

    let stride = fmm_params.local_dimension.x * fmm_params.local_dimension.y * fmm_params.local_dimension.z;

    let xOffset = 1u;
    let yOffset = fmm_params.global_dimension.x;
    let zOffset = yOffset * fmm_params.global_dimension.y;

    let global_coordinate = vec3<u32>(v) / fmm_params.local_dimension;

    let global_index = (global_coordinate.x * xOffset +
                        global_coordinate.y * yOffset +
                        global_coordinate.z * zOffset) * stride;

    let local_coordinate = vec3<u32>(v) - global_coordinate * fmm_params.local_dimension;

    let local_index = encode3Dmorton32(local_coordinate.x, local_coordinate.y, local_coordinate.z);

    return global_index + local_index; 
}

// Encode "rgba" to u32.
fn rgba_u32(r: u32, g: u32, b: u32, a: u32) -> u32 {
  return (r << 24u) | (g << 16u) | (b  << 8u) | a;
}

fn visualize_cell(position: vec3<f32>, color: u32) {
    output_aabb_wire[atomicAdd(&counter[3], 1u)] =
          AABB (
              //vec4<f32>(position, bitcast<f32>(color)),
              //vec4<f32>(12.0 * position,  2.5)
              4.0 * vec4<f32>(position, 0.0) + vec4<f32>(vec3<f32>(0.002), bitcast<f32>(color)),
              4.0 * vec4<f32>(position, 0.0) + vec4<f32>(vec3<f32>(AABB_SIZE - 0.002), 0.2)
          );
}

@compute
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

        if (global_id.x >= fmm_params.global_dimension.x *
	                   fmm_params.global_dimension.y *
	                   fmm_params.global_dimension.z *
	                   fmm_params.local_dimension.x *
	                   fmm_params.local_dimension.y *
	                   fmm_params.local_dimension.z) { return; } 

        let fmm_cell = fmm_data[local_index + workgroup_id.x * 64u];    
        if (fmm_cell.tag == BAND) {
            atomicAdd(&fmm_blocks[workgroup_id.x].number_of_band_points, 1u);
	}
	storageBarrier();

	if (local_index == 0u && fmm_blocks[workgroup_id.x].number_of_band_points > 0u) {
            var position = vec3<f32>(get_group_coordinate(global_id.x)) * 4.0;
            let color_band = rgba_u32(222u, 55u, 150u, 255u);
            visualize_cell(position, color_band);
	}
}
