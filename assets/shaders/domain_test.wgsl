// TODO: add padding to Rust struct.

struct ComputationalDomain {
    global_dimension: vec3<u32>,
    aabb_size: f32,
    local_dimension:  vec3<u32>,
    font_size: f32,
    current_cell: vec3<u32>,
    //permutation_index: u32,
};

/// parameters for permutations.
struct Permutation {
    modulo: u32,
    x_factor: u32,  
    y_factor: u32,  
    z_factor: u32,  
};

/// Struct which contains the data of a single thread.
struct PrivateData {
    global_position: array<u32, 3>, 
    memory_location: u32,
};

// Debugging.

struct AABB {
    min: vec4<f32>, 
    max: vec4<f32>, 
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

struct Arrow {
    start_pos: vec4<f32>,
    end_pos:   vec4<f32>,
    color: u32,
    size:  f32,
};

let FONT_SIZE = 0.015;
let AABB_SIZE = 0.26;

@group(0) @binding(0) var<uniform> computational_domain: ComputationalDomain;
// @group(0) @binding(1) var<storage,read_write> permutations: array<Permutation>;
@group(0) @binding(1) var<storage,read_write> counter: array<atomic<u32>>;
@group(0) @binding(2) var<storage,read_write> output_char: array<Char>;
@group(0) @binding(3) var<storage,read_write> output_arrow: array<Arrow>;
@group(0) @binding(4) var<storage,read_write> output_aabb: array<AABB>;
@group(0) @binding(5) var<storage,read_write> output_aabb_wire: array<AABB>;

var<private> MAGIC_NUMBERS: array<u32, 8> = array<u32, 8>(0u, 1u, 2u, 3u, 3u, 2u, 1u, 0u);
var<private> GROUP_COLORS: array<u32, 4> = array<u32, 4>(4278190335u, 4294902015u, 10223615u, 65535u);

/// DEBUG FUNCTION. 
struct ModF {
    fract: f32,
    whole: f32,
};

/// DEBUG FUNCTION. 
fn myTruncate(f: f32) -> f32 {
    return select(f32( i32( floor(f) ) ), f32( i32( ceil(f) ) ), f < 0.0); 
}

/// DEBUG FUNCTION. 
fn my_modf(f: f32) -> ModF {
    let iptr = trunc(f);
    let fptr = f - iptr;
    return ModF (
        select(fptr, (-1.0)*fptr, f < 0.0),
        iptr
    );
}

/// DEBUG FUNCTION. 
/// if the given integer is < 0, return 0. Otherwise 1 is returned.
fn the_sign_of_i32(n: i32) -> u32 {
    return (u32(n) >> 31u);
    //return 1u ^ (u32(n) >> 31u);
}

/// DEBUG FUNCTION. 
fn abs_i32(n: i32) -> u32 {
    let mask = u32(n) >> 31u;
    return (u32(n) + mask) ^ mask;
}

/// DEBUG FUNCTION. 
fn log2_u32(n: u32) -> u32 {

    var v = n;
    var r = (u32((v > 0xFFFFu))) << 4u;
    var shift = 0u;

    v  = v >> r;
    shift = (u32((v > 0xFFu))) << 3u;

    v = v >> shift;
    r = r | shift;

    shift = (u32(v > 0xFu)) << 2u;
    v = v >> shift;
    r = r | shift;

    shift = (u32(v > 0x3u)) << 1u;
    v = v >> shift;
    r = r | shift;
    r = r | (v >> 1u);
    return r;
}

/// DEBUG FUNCTION. 
var<private> PowersOf10: array<u32, 10> = array<u32, 10>(1u, 10u, 100u, 1000u, 10000u, 100000u, 1000000u, 10000000u, 100000000u, 1000000000u);

/// DEBUG FUNCTION. 
// NOT defined if u == 0u.
fn log10_u32(n: u32) -> u32 {
    
    var v = n;
    var r: u32;
    var t: u32;
    
    t = (log2_u32(v) + 1u) * 1233u >> 12u;
    r = t - u32(v < PowersOf10[t]);
    return r;
}

/// DEBUG FUNCTION. 
fn number_of_chars_i32(n: i32) -> u32 {

    if (n == 0) { return 1u; }
    return the_sign_of_i32(n) + log10_u32(u32(abs(n))) + 1u;
} 

/// DEBUG FUNCTION. 
fn number_of_chars_f32(f: f32, number_of_decimals: u32) -> u32 {

    let m = my_modf(f);
    return number_of_chars_i32(i32(m.whole)) + number_of_decimals + 1u;
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

/// DEBUG FUNCTION. 
var<private> NumberOfExtraChars: array<u32, 5> = array<u32, 5>(0u, 0u, 2u, 2u, 2u);

/// DEBUG FUNCTION. 
fn number_of_chars_data(data: vec4<f32>, vec_dim_count: u32, number_of_decimals: u32) -> u32 {
    
    // Calculate all possible char counts to avoid branches.
    let a = number_of_chars_f32(data.x, number_of_decimals) * u32(vec_dim_count >= 1u);
    let b = number_of_chars_f32(data.y, number_of_decimals) * u32(vec_dim_count >= 2u);
    let c = number_of_chars_f32(data.z, number_of_decimals) * u32(vec_dim_count >= 3u);
    let d = number_of_chars_f32(data.w, number_of_decimals) * u32(vec_dim_count >= 4u);

    return a + b + c + d + NumberOfExtraChars[vec_dim_count]; 
}

/// A function that checks if a given coordinate is within the global computational domain. 
fn isInside(coord: vec3<i32>) -> bool {
    return (coord.x >= 0 && coord.x < i32(computational_domain.local_dimension.x * computational_domain.global_dimension.x)) &&
           (coord.y >= 0 && coord.y < i32(computational_domain.local_dimension.y * computational_domain.global_dimension.y)) &&
           (coord.z >= 0 && coord.z < i32(computational_domain.local_dimension.z * computational_domain.global_dimension.z)); 
}

/// Get group coordinate based on cell memory index.
fn get_group_coordinate(global_index: u32) -> vec3<u32> {

    let stride = computational_domain.local_dimension.x * computational_domain.local_dimension.y * computational_domain.local_dimension.z;
    let block_index = global_index / stride;

    return index_to_uvec3(block_index, computational_domain.global_dimension.x, computational_domain.global_dimension.y);
}

/// Calculate group number {0, 1, 2, 3} based on cell memory location.
fn get_group_number(global_index: u32) -> u32 {

    // Get the group coordinate.
    let group_coordinate = get_group_coordinate(global_index);

    return MAGIC_NUMBERS[(group_coordinate.x & 1u) | ((group_coordinate.y & 1u) << 1u) | ((group_coordinate.z & 1u) << 2u)];
} 

/// Get memory index from given cell coordinate.
fn get_cell_mem_location(v: vec3<u32>) -> u32 {

    let stride = computational_domain.local_dimension.x * computational_domain.local_dimension.y * computational_domain.local_dimension.z;

    let xOffset = 1u;
    let yOffset = computational_domain.global_dimension.x;
    let zOffset = yOffset * computational_domain.global_dimension.y;

    let global_coordinate = vec3<u32>(v) / computational_domain.local_dimension;

    let global_index = (global_coordinate.x * xOffset +
                        global_coordinate.y * yOffset +
                        global_coordinate.z * zOffset) * stride;

    let local_coordinate = vec3<u32>(v) - global_coordinate * computational_domain.local_dimension;

    let local_index = encode3Dmorton32(local_coordinate.x, local_coordinate.y, local_coordinate.z);

    return global_index + local_index; 
}

/// Get cell index based on domain dimension.
fn get_cell_index(global_index: u32) -> vec3<u32> {

    let stride = computational_domain.local_dimension.x * computational_domain.local_dimension.y * computational_domain.local_dimension.z;
    let block_index = global_index / stride;
    let block_position = index_to_uvec3(block_index, computational_domain.global_dimension.x, computational_domain.global_dimension.y) * computational_domain.local_dimension;

    // Calculate local position.
    let local_index = global_index - block_index * stride;

    let local_position = decode3Dmorton32(local_index);

    let cell_position = block_position + local_position;

    return cell_position; 
}

// Encode "rgba" to u32.
fn rgba_u32(r: u32, g: u32, b: u32, a: u32) -> u32 {
  return (r << 24u) | (g << 16u) | (b  << 8u) | a;
}

/// DEBUG.
fn visualize_cell(position: vec3<f32>, color: u32, value: vec4<f32>, dimension: u32, draw_number: bool) {
    output_aabb[atomicAdd(&counter[2], 1u)] =
          AABB (
              vec4<f32>(position - vec3<f32>(computational_domain.aabb_size), bitcast<f32>(color)),
              vec4<f32>(position + vec3<f32>(computational_domain.aabb_size), 0.0),
          );

    if (draw_number) {
        let color_text = rgba_u32(255u, 255u, 255u, 255u);

        // let value = vec4<f32>(f32(global_id), 0.0, 0.0, 0.0);
        let total_number_of_chars = number_of_chars_data(value, 1u, 2u);
        let element_position = position - vec3<f32>(f32(total_number_of_chars) * computational_domain.font_size * 0.5, 0.0, (-1.0) * computational_domain.aabb_size - 0.001);
        let renderable_element = Char (
                        element_position,
                        computational_domain.font_size,
                        value,
                        dimension,
                        color_text,
                        1u,
                        0u
        );
        output_char[atomicAdd(&counter[0], 1u)] = renderable_element; 
    }
}

fn load_neighbors_6(coord: vec3<u32>, global_index: u32) {

    var neighbors: array<vec3<i32>, 6> = array<vec3<i32>, 6>(
        vec3<i32>(coord) + vec3<i32>(-1,  0,  0),
        vec3<i32>(coord) + vec3<i32>(1,   0,  0),
        vec3<i32>(coord) + vec3<i32>(0,   1,  0),
        vec3<i32>(coord) + vec3<i32>(0,  -1,  0),
        vec3<i32>(coord) + vec3<i32>(0,   0,  1),
        vec3<i32>(coord) + vec3<i32>(0,   0, -1)
    );

    var memory_locations: array<u32, 6> = array<u32, 6>(
        get_cell_mem_location(vec3<u32>(neighbors[0])),
        get_cell_mem_location(vec3<u32>(neighbors[1])),
        get_cell_mem_location(vec3<u32>(neighbors[2])),
        get_cell_mem_location(vec3<u32>(neighbors[3])),
        get_cell_mem_location(vec3<u32>(neighbors[4])),
        get_cell_mem_location(vec3<u32>(neighbors[5]))
    );

    let this_group_number = get_group_number(global_index);

    var col = select(
              select(
              select(
              select(0u, GROUP_COLORS[3], this_group_number == 3u),
                     GROUP_COLORS[2], this_group_number == 2u),
                     GROUP_COLORS[1], this_group_number == 1u),
                     GROUP_COLORS[0], this_group_number == 0u);

    if (coord.x == computational_domain.current_cell.x &&
        coord.y == computational_domain.current_cell.y &&
        coord.z == computational_domain.current_cell.z) {

            visualize_cell(4.0 * vec3<f32>(coord), col, vec4<f32>(f32(global_index), 0.0, 0.0, 0.0), 1u, true);

            // Draw arrows.
            for (var i: i32 = 0 ; i < 6; i = i + 1) {

                let n_group_number = get_group_number(memory_locations[i]);

		var col_arrow = select(
                                select(
                                select(
                                select(0u, GROUP_COLORS[3], n_group_number == 3u),
                                       GROUP_COLORS[2], n_group_number == 2u),
                                       GROUP_COLORS[1], n_group_number == 1u),
                                       GROUP_COLORS[0], n_group_number == 0u); 

                // Draw an arrow if the neighbor is inside computational domain.
                if (isInside(neighbors[i])) {
                    output_arrow[atomicAdd(&counter[1], 1u)] =  
                          Arrow (
                              4.0 * vec4<f32>(vec3<f32>(coord), 0.0),
                              4.0 * vec4<f32>(vec3<f32>(neighbors[i]), 0.0),
                              col_arrow,
                              0.1
                    );
                }
            }
        
    }
    else {
        visualize_cell(4.0 * vec3<f32>(coord), col, vec4<f32>(f32(global_index), 0.0, 0.0, 0.0), 1u, true);
    }

}

// Load 16 neighbors with debug.
fn load_neighbors_18(coord: vec3<u32>, global_index: u32) {

    var neighbors: array<vec3<i32>, 18> = array<vec3<i32>, 18>(
        vec3<i32>(coord) + vec3<i32>(-1, 0, 0),
        vec3<i32>(coord) + vec3<i32>(-1, 1, 0),
        vec3<i32>(coord) + vec3<i32>(-1,-1, 0),
        vec3<i32>(coord) + vec3<i32>(1,  0, 0),
        vec3<i32>(coord) + vec3<i32>(1,  1, 0),
        vec3<i32>(coord) + vec3<i32>(1, -1, 0),
        vec3<i32>(coord) + vec3<i32>(0,  1, 0),
        vec3<i32>(coord) + vec3<i32>(0, -1, 0),
        vec3<i32>(coord) + vec3<i32>(0,  0, -1),
        vec3<i32>(coord) + vec3<i32>(1,  0, -1),
        vec3<i32>(coord) + vec3<i32>(-1, 0, -1),
        vec3<i32>(coord) + vec3<i32>(0,  1, -1),
        vec3<i32>(coord) + vec3<i32>(0, -1, -1),
        vec3<i32>(coord) + vec3<i32>(0,  0, 1),
        vec3<i32>(coord) + vec3<i32>(1,  0, 1),
        vec3<i32>(coord) + vec3<i32>(-1, 0, 1),
        vec3<i32>(coord) + vec3<i32>(0,  1, 1),
        vec3<i32>(coord) + vec3<i32>(0, -1, 1)
    );

    var memory_locations: array<u32, 18> = array<u32, 18>(
        get_cell_mem_location(vec3<u32>(neighbors[0])),
        get_cell_mem_location(vec3<u32>(neighbors[1])),
        get_cell_mem_location(vec3<u32>(neighbors[2])),
        get_cell_mem_location(vec3<u32>(neighbors[3])),
        get_cell_mem_location(vec3<u32>(neighbors[4])),
        get_cell_mem_location(vec3<u32>(neighbors[5])),
        get_cell_mem_location(vec3<u32>(neighbors[6])),
        get_cell_mem_location(vec3<u32>(neighbors[7])),
        get_cell_mem_location(vec3<u32>(neighbors[8])),
        get_cell_mem_location(vec3<u32>(neighbors[9])),
        get_cell_mem_location(vec3<u32>(neighbors[10])),
        get_cell_mem_location(vec3<u32>(neighbors[11])),
        get_cell_mem_location(vec3<u32>(neighbors[12])),
        get_cell_mem_location(vec3<u32>(neighbors[13])),
        get_cell_mem_location(vec3<u32>(neighbors[14])),
        get_cell_mem_location(vec3<u32>(neighbors[15])),
        get_cell_mem_location(vec3<u32>(neighbors[16])),
        get_cell_mem_location(vec3<u32>(neighbors[17]))
    );

    let this_group_number = get_group_number(global_index);

    var col = select(
              select(
              select(
              select(0u, GROUP_COLORS[3], this_group_number == 3u),
                     GROUP_COLORS[2], this_group_number == 2u),
                     GROUP_COLORS[1], this_group_number == 1u),
                     GROUP_COLORS[0], this_group_number == 0u);

    if (coord.x == computational_domain.current_cell.x &&
        coord.y == computational_domain.current_cell.y &&
        coord.z == computational_domain.current_cell.z) {

            visualize_cell(4.0 * vec3<f32>(coord), col, vec4<f32>(f32(global_index), 0.0, 0.0, 0.0), 1u, true);

            // Draw arrows.
            for (var i: i32 = 0 ; i < 18; i = i + 1) {

                let n_group_number = get_group_number(memory_locations[i]);

		var col_arrow = select(
                                select(
                                select(
                                select(0u, GROUP_COLORS[3], n_group_number == 3u),
                                       GROUP_COLORS[2], n_group_number == 2u),
                                       GROUP_COLORS[1], n_group_number == 1u),
                                       GROUP_COLORS[0], n_group_number == 0u); 

                // Draw an arrow if the neighbor is inside computational domain.
                if (isInside(neighbors[i])) {
                    output_arrow[atomicAdd(&counter[1], 1u)] =  
                          Arrow (
                              4.0 * vec4<f32>(vec3<f32>(coord), 0.0),
                              4.0 * vec4<f32>(vec3<f32>(neighbors[i]), 0.0),
                              col_arrow,
                              0.1
                    );
                }
            }
        
    }
    else {
        visualize_cell(4.0 * vec3<f32>(coord), col, vec4<f32>(f32(global_index), 0.0, 0.0, 0.0), 1u, true);
    }
}

@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    let total_block_count = computational_domain.global_dimension.x *
                            computational_domain.global_dimension.y *
                            computational_domain.global_dimension.z;

    let total_count = computational_domain.local_dimension.x *
                      computational_domain.local_dimension.y *
                      computational_domain.local_dimension.z *
                      computational_domain.global_dimension.x *
                      computational_domain.global_dimension.y *
                      computational_domain.global_dimension.z;

    if (global_id.x >= total_count) { return; }

    let index = get_cell_index(global_id.x);
    // load_neighbors_18(index, global_id.x);
    load_neighbors_6(index, global_id.x);
}
