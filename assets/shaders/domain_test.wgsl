// TODO: add padding to Rust struct.
struct ComputationalDomain {
    global_dimension: array<u32, 3>,
    local_dimension:  array<u32, 3>,
};

/// Parameters for permutations.
struct Permutation {
    mod: u32,
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

@group(0) @binding(0) var<storage,read_write> counter: array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> output_char: array<Char>;
@group(0) @binding(2) var<storage,read_write> output_arrow: array<Arrow>;
@group(0) @binding(3) var<storage,read_write> output_aabb: array<AABB>;
@group(0) @binding(4) var<storage,read_write> output_aabb_wire: array<AABB>;

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

/// Get cell index based on domain dimension.
fn get_cell_index(global_index: u32) -> vec3<u32> {

    let stride = mc_uniform.noise_local_dimension.x * mc_uniform.noise_local_dimension.y * mc_uniform.noise_local_dimension.z;
    let block_index = global_index / stride;
    let block_position = index_to_uvec3(block_index, mc_uniform.noise_global_dimension.x, mc_uniform.noise_global_dimension.y) * mc_uniform.noise_local_dimension;

    // Calculate local position.
    let local_index = global_index - block_index * stride;

    let local_position = decode3Dmorton32(local_index);

    let cell_position = block_position + local_position;

    return cell_position; 
}

@compute
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

}
