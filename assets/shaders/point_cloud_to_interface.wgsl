/// Create an initial fmm interface from point data. 

struct VVVC {
    position: vec3<f32>,
    color: u32,
};

struct FmmParams {
    global_dimension: vec3<u32>,
    local_dimension: vec3<u32>,
};

struct PointCloudParams {
    min_point: vec3<f32>,
    point_count: u32,
    max_point: vec3<f32>,
    padding: u32,
};

struct FmmCellPc {
    tag: u32,
    value: f32,
    color: u32,
}

// Debugging.
struct AABB {
    min: vec4<f32>, 
    max: vec4<f32>, 
};

// Debugging.
struct Char {
    start_pos: vec3<f32>,
    font_size: f32,
    value: vec4<f32>,
    vec_dim_count: u32,
    color: u32,
    decimal_count: u32,
    auxiliary_data: u32,
};

// Debugging.
struct Arrow {
    start_pos: vec4<f32>,
    end_pos:   vec4<f32>,
    color: u32,
    size:  f32,
};

@group(0) @binding(0) var<uniform> fmm_params: FmmParams;
@group(0) @binding(1) var<uniform> point_cloud_params: PointCloudParams;
@group(0) @binding(2) var<storage, read_write> fmm_data: array<FmmCellPc>;
@group(0) @binding(3) var<storage, read_write> point_data: array<VVVC>;

// Debug.
@group(0) @binding(4) var<storage,read_write> counter: array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> output_char: array<Char>;
@group(0) @binding(6) var<storage,read_write> output_arrow: array<Arrow>;
@group(0) @binding(7) var<storage,read_write> output_aabb: array<AABB>;
@group(0) @binding(8) var<storage,read_write> output_aabb_wire: array<AABB>;

/// A function that checks if a given coordinate is within the global computational domain. 
fn isInside(coord: vec3<i32>) -> bool {
    return (coord.x >= 0 && coord.x < i32(fmm_params.local_dimension.x * fmm_params.global_dimension.x)) &&
           (coord.y >= 0 && coord.y < i32(fmm_params.local_dimension.y * fmm_params.global_dimension.y)) &&
           (coord.z >= 0 && coord.z < i32(fmm_params.local_dimension.z * fmm_params.global_dimension.z)); 
}

// Encode "rgba" to u32.
fn rgba_u32(r: u32, g: u32, b: u32, a: u32) -> u32 {
  return (r << 24u) | (g << 16u) | (b  << 8u) | a;
}

@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    let scene_aabb_color = bitcast<f32>(rgba_u32(222u, 0u, 150u, 255u));

    // Visualize the global fmm domain aabb.
    if (global_id.x == 0u) {
        output_aabb_wire[atomicAdd(&counter[3], 1u)] =  
              AABB (
                  vec4<f32>(0.0, 0.0, 0.0, scene_aabb_color),
                  vec4<f32>(vec3<f32>(fmm_params.global_dimension * fmm_params.local_dimension), 0.1)
              );
        output_aabb_wire[atomicAdd(&counter[3], 1u)] =  
              AABB (
                  vec4<f32>(point_cloud_params.min_point, scene_aabb_color),
                  vec4<f32>(point_cloud_params.max_point, 0.1)
              );
    }

    if (global_id.x >= point_cloud_params.point_count) { return; }

    let p = point_data[global_id.x];
    let nearest_cell = vec3<u32>(u32(round(f32(p.position.x))),
                                 u32(round(f32(p.position.y))),
                                 u32(round(f32(p.position.z))));
    let dist = distance(vec3<f32>(p.position), vec3<f32>(nearest_cell));

    output_arrow[atomicAdd(&counter[1], 1u)] =  
          Arrow (
              4.0 * vec4<f32>(vec3<f32>(p.position), 0.0),
              4.0 * vec4<f32>(vec3<f32>(nearest_cell), 0.0),
              p.color,
              0.05
    );
}
