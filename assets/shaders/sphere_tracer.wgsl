struct FmmCellPc {
    tag: atomic<u32>,
    value: f32,
    color: u32,
};

// struct FmmBlock {
//     index: u32,
//     number_of_band_points: atomic<u32>,
// };

struct FmmParams {
    global_dimension: vec3<u32>,
    future_usage: u32,
    local_dimension: vec3<u32>,
    future_usage2: u32,
};

// // Binding id from where to load data.
// struct PushConstants {
//     buffer_id: u32,    
// };

struct SphereTracerParams {
    
};

struct RayOutput {
    origin: vec3<f32>,
    visibility: f32,
    intersection_point: vec3<f32>,
    opacity: f32,
    normal: vec3<f32>,
    diffuce_color: u32,
};

struct Ray {
  origin: vec3<f32>,
  rMin: f32,
  direction;: vec3<f32>,
  rMax: f32,
};

struct RayPayload {
  intersection_point: vec3<f32>,
  visibility: f32,
  normal: vec3<f32>,
  opacity: f32,
  color: u32,
};

struct RayCamera {
    pos: vec3<f32>,
    apertureRadius: f32,
    view: vec3<f32>,
    focalDistance: f32,
    up: vec3<f32>,
    fov: vec2<f32>,
};

struct SphereTracerParams {
    inner_dim: vec32<u32>,
    outer_dim: vec32<u32>,
};

@group(0) @binding(0) var<uniform>            fmm_params: FmmParams;
@group(0) @binding(1) var<uniform>            sphere_tracer_params: SphereTracerParams;
@group(0) @binding(2) var<uniform>            camera: RayCamera;
@group(0) @binding(3) var<storage,read_write> fmm_data: array<FmmCellPc>;
@group(0) @binding(4) var<storage,read_write> screen_output: array<RayOutput>;

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

fn index_to_2d(index: u32, width: u32) -> vec2<u32> {
    var x = index;
    let y = x / width;
    x = x - y * width;
    return vec2<u32>(x, y);
}

fn d2_to_index(x: u32, y: u32, width: u32) -> u32 {
    x + y * width
}

fn inner_coord_d2(index: u32, dim_x: u32, dim_y: u32) -> vec2<u32> {
    var x = index & (dim_x - 1);
    let y = (index / dim_x) & (dim_y - 1);
    return vec2<u32>(x, y);
}

fn index_to_screen(index: u32, dim_x: u32, dim_y: u32, global_dim_x:u32) -> vec2<u32> {

        // Outer coords.
        let block_index = index / (dim_x*dim_y);
        let outer_coord = index_to_2d(block_index, global_dim_x);

        // inner coords.
        let x = index & (dim_x - 1);
        let y = (index / dim_x) & (dim_y - 1);

        // Global coordinate.
	return vec2<u32>(outer_coord[0] * dim_x + x, outer_coord[1] * dim_y + y);

        // let gx = outer_coord[0] * dim_x + x;
        // let gy = outer_coord[1] * dim_y + y;

        // [gx, gy]
}

@compute
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    let screen_coord = vec2<f32)(index_to_screen(global_id.x,
                                       sphere_tracer_params.inner_dim.x,
				       sphere_tracer_params.inner_dim.y,
				       sphere_tracer_params.outer_dim.x));

    let x_coord = f32((local_index.x & 7u) * workgroup_id.x);
    let y_coord = f32((local_index.x / 4u) * workgroup_id.x);
    let screen_width = f32(sphere_tracer_params.inner_dim.x * sphere_tracer_params.outer_dim.x);
    let screen_height = f32(sphere_tracer_params.inner_dim.y * sphere_tracer_params.outer_dim.y);

    vec3 right = normalize(cross(camera.view, camera.up.xyz));
    vec3 y = normalize(cross(camera.view, right));

    float d = camera.focalDistance;

    vec3 u = (d * tan(camera.fov.x*0.5)) * right;
    vec3 v = (d * tan(camera.fov.y*0.5)) * y;

    let alpha = 2.0 * (x_coord + 0.5) / screen_width - 1.0;
    let beta  = 1.0 - 2.0 * (y_coord + 0.5) / screen_height;

    let point_on_plane = alpha * u + beta * v;

    Ray ray;
    ray.origin = point_on_plane + camera.position.xyz;
    ray.direction = normalize(point_on_plane + d*camera.view.xyz);
    ray.rMin = 0.0f;
    ray.rMax = 300.0f;

    RayPayload payload;
    payload.color = encode_color(vec4(1.0f,0.0f,0.0f,1.0f));
    payload.visibility = 1.0;

    // traceRay(ray, payload);

    // RayOutput result;
    // result.origin = vec4(ray.origin, 1.0);
    // result.intersection_point = payload.intersection_point;
    // result.normal = payload.normal;
}
