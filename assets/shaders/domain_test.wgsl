// TODO: add padding to Rust struct.
struct ComputationalDomain {
    global_dimension: vec3<u32>,
    padding: u32,
    local_dimension:  vec3<u32>,
    padding2: u32,
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

let AABB_SIZE = 0.26;

@group(0) @binding(0) var<uniform> computational_domain: ComputationalDomain;
@group(0) @binding(1) var<storage,read_write> permutations: array<Permutation>;
@group(0) @binding(2) var<storage,read_write> counter: array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> output_char: array<Char>;
@group(0) @binding(4) var<storage,read_write> output_arrow: array<Arrow>;
@group(0) @binding(5) var<storage,read_write> output_aabb: array<AABB>;
@group(0) @binding(6) var<storage,read_write> output_aabb_wire: array<AABB>;

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
fn isInside(coord: ptr<function, vec3<i32>>) -> bool {
    return ((*coord).x >= 0 && (*coord).x < 64) &&
           ((*coord).y >= 0 && (*coord).y < 64) &&
           ((*coord).z >= 0 && (*coord).z < 64); 
}

fn get_group_coordinate(global_index: u32) -> vec3<u32> {
    let stride = computational_domain.local_dimension.x * computational_domain.local_dimension.y * computational_domain.local_dimension.z;
    let block_index = global_index / stride;
    return index_to_uvec3(block_index, computational_domain.global_dimension.x, computational_domain.global_dimension.y);
}

/// Get cell index based on domain dimension.
fn get_cell_index(global_index: u32) -> vec3<u32> {

    let stride = computational_domain.local_dimension.x * computational_domain.local_dimension.y * computational_domain.local_dimension.z;
    let block_index = global_index / stride;
    let block_position = index_to_uvec3(block_index, computational_domain.global_dimension.x, computational_domain.global_dimension.y) * computational_domain.local_dimension;
    // let block_position = index_to_uvec3(block_index, computational_domain.global_dimension.x, computational_domain.global_dimension.y) * computational_domain.local_dimension;

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

fn visualize_cell(position: vec3<f32>, color: u32) {
    output_aabb[atomicAdd(&counter[2], 1u)] =
          AABB (
              vec4<f32>(position - vec3<f32>(AABB_SIZE), bitcast<f32>(color)),
              vec4<f32>(position + vec3<f32>(AABB_SIZE), 0.0),
          );
}

@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    let total_count = computational_domain.local_dimension.x *
                      computational_domain.local_dimension.y *
                      computational_domain.local_dimension.z *
                      computational_domain.global_dimension.x *
                      computational_domain.global_dimension.y *
                      computational_domain.global_dimension.z;

    if (global_id.x >= total_count) { return; }

    let index = get_cell_index(global_id.x);
    
    let group_coord = get_group_coordinate(global_id.x);

    let permutation_number = (2u * group_coord.x +  13u * group_coord.y + 17u * group_coord.z) % 4u; 
    // let permutation_number = (2u * position_u32_temp.x +  3u * position_u32_temp.y + 5u * position_u32_temp.z) & 3u; 
    // let permutation_number = (3u * position_u32_temp.x +  5u * position_u32_temp.y + 7u * position_u32_temp.z) & 2u; 

    let permutation_color0 = rgba_u32(255u, 0u, 0u, 255u);
    let permutation_color1 = rgba_u32(0u, 255u, 0u, 255u);
    let permutation_color2 = rgba_u32(0u, 0u, 255u, 255u);
    let permutation_color3 = rgba_u32(0u, 155u, 155u, 255u);

    var col = 0u;
    if (permutation_number == 0u) { col =  permutation_color0; }
    else if (permutation_number == 1u) { col = permutation_color1; }
    else if (permutation_number == 2u) { col = permutation_color2; }
    else if (permutation_number == 3u) { col = permutation_color3; }

//    var col = select(select(select(select(0u, permutation_color3, permutation_number == 3u), permutation_color2, permutation_number == 2u), permutation_color1, permutation_number == 1u), permutation_color0, permutation_number == 0u);

    // col = select(0u, permutation_color1, permutation_number == 1u);
    // col = select(0u, permutation_color2, permutation_number == 2u);
    // col = select(0u, permutation_color3, permutation_number == 3u);

    // var col = select(0u, permutation_color0, permutation_number == 0u);
    // col = select(0u, permutation_color1, permutation_number == 1u);
    // col = select(0u, permutation_color2, permutation_number == 2u);
    // col = select(0u, permutation_color3, permutation_number == 3u);

    visualize_cell(vec3<f32>(index), col);
    
}
