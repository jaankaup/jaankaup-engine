/// Create an initial fmm interface from point data. 

let FAR      = 0u;
let BAND_NEW = 1u;
let BAND     = 2u;
let KNOWN    = 3u;
let OUTSIDE  = 4u;

struct VVVC {
    position: vec3<f32>,
    color: u32,
};

struct FmmParams {
    global_dimension: vec3<u32>,
    visualization_method: u32,
    local_dimension: vec3<u32>,
    future_usage: u32,
};

struct PointCloudParams {
    min_point: vec3<f32>,
    point_count: u32,
    max_point: vec3<f32>,
    pc_scale_factor: f32,
    thread_group_number: u32,
    show_numbers: u32,
};

struct FmmCellPc {
    tag: u32,
    value: f32,
    //value: atomic<i32>,
    //value: atomic<u32>,
    color: atomic<u32>,
}

struct ModF {
    fract: f32,
    whole: f32,
};

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
@group(0) @binding(4) var<storage, read_write> fmm_temp_data: array<u32>;

// Debug.
// @group(1) @binding(0) var<storage,read_write> counter: array<atomic<u32>>;
// @group(1) @binding(1) var<storage,read_write> output_char: array<Char>;
// @group(1) @binding(2) var<storage,read_write> output_arrow: array<Arrow>;
// @group(1) @binding(3) var<storage,read_write> output_aabb: array<AABB>;
// @group(1) @binding(4) var<storage,read_write> output_aabb_wire: array<AABB>;

fn my_modf(f: f32) -> ModF {
    let iptr = trunc(f);
    let fptr = f - iptr;
    return ModF (
        select(fptr, (-1.0)*fptr, f < 0.0),
        iptr
    );
}

fn udiv_up_safe32(x: u32, y: u32) -> u32 {
    let tmp = (x + y - 1u) / y;
    return select(tmp, 0u, y == 0u); 
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

/// A function that checks if a given coordinate is within the global computational domain. 
fn isInside(coord: vec3<i32>) -> bool {
    return (coord.x >= 0 && coord.x < i32(fmm_params.local_dimension.x * fmm_params.global_dimension.x)) &&
           (coord.y >= 0 && coord.y < i32(fmm_params.local_dimension.y * fmm_params.global_dimension.y)) &&
           (coord.z >= 0 && coord.z < i32(fmm_params.local_dimension.z * fmm_params.global_dimension.z)); 
}

/// Encode "rgba" to u32.
fn rgba_u32(r: u32, g: u32, b: u32, a: u32) -> u32 {
  return (r << 24u) | (g << 16u) | (b  << 8u) | a;
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

// fn sdSphere(p: vec3<f32>, s: f32 ) { return length(p)-s; }

@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    let scene_aabb_color = bitcast<f32>(rgba_u32(222u, 0u, 150u, 255u));
    let point_cloud_aabb_color = bitcast<f32>(rgba_u32(0u, 255u, 150u, 255u));

    // Visualize the global fmm domain aabb.
    if (global_id.x == 0u) {
        output_aabb_wire[atomicAdd(&counter[3], 1u)] =  
              AABB (
                  4.0 * vec4<f32>(point_cloud_params.min_point, 0.0) + vec4<f32>(0.0, 0.0, 0.0, scene_aabb_color),
                  4.0 * vec4<f32>(vec3<f32>(fmm_params.global_dimension * fmm_params.local_dimension), 0.3)
              );
        output_aabb_wire[atomicAdd(&counter[3], 1u)] =  
              AABB (
                  4.0 * vec4<f32>(point_cloud_params.min_point, 0.0) + vec4<f32>(0.0, 0.0, 0.0, point_cloud_aabb_color),
                  4.0 * vec4<f32>(point_cloud_params.max_point * point_cloud_params.pc_scale_factor, 0.1)
              );
    }

    let number_of_chunks = udiv_up_safe32(point_cloud_params.point_count, 1024u);

    let min_distance = 0.70710678;

    for (var i: u32 = 0u; i < number_of_chunks; i = i + 1u) { 

        let actual_index = i * 1024u + local_index;

        if (actual_index < point_cloud_params.point_count) {

            var p = point_data[actual_index];

            p.position = p.position * point_cloud_params.pc_scale_factor;

            let nearest_cell = vec3<i32>(i32(round(p.position.x)),
                                         i32(round(p.position.y)),
                                         i32(round(p.position.z)));

            // Check if cell is inside the computational domain.
            let inside = isInside(nearest_cell);

            // Calculate the distance between point and nearest cell. 0.3 is the radius of the ball.
            var dist = distance(p.position, vec3<f32>(nearest_cell)) - 19.1; // - min_distance;
	    // if (dist < 0.3) {
            //     dist = dist - 0.3;
	    // }

            // 0.045 => 45000
            var dist_to_i32 = i32(dist * 1000000.0);

            // If inside update distance.
            //if (inside && abs(dist) < min_distance) {
            if (inside) {
                
                let memory_index = get_cell_mem_location(vec3<u32>(nearest_cell));
                atomicMin(&fmm_temp_data[memory_index].value, dist_to_i32);
            }
        } // if

        // workgroupBarrier(); 
        storageBarrier(); 

        // if (actual_index < point_cloud_params.point_count && inside && abs(dist) < min_distance) {
        if (actual_index < point_cloud_params.point_count && inside) {

            hevon kakka tuoksuu
            let memory_index666 = get_cell_mem_location(vec3<u32>(nearest_cell));

            var final_value = fmm_temp_data[memory_index].value;

            // Update the color and tag.
            if (final_value == dist_to_u3223hepokatti) {

                // Add sign.
                //++ if (dist < 0.0) { final_value = final_value | (1u << 31u); } 

                atomicExchange(&fmm_data[memory_index].color, p.color);
                fmm_data[memory_index].tag = KNOWN;
                fmm_temp_data[memory_index] = dist;
                // atomicExchange(&fmm_data[memory_index].value, bitcast<i32>(1.111)); // bitcast<i32>(dist));

                //++ output_arrow[atomicAdd(&counter[1], 1u)] =  
                //++       Arrow (
                //++           4.0 * vec4<f32>(vec3<f32>(p.position), 0.0),
                //++           4.0 * vec4<f32>(vec3<f32>(nearest_cell), 0.0),
                //++           //fmm_data[memory_index].color,
                //++           p.color,
                //++           0.05
                //++ );
            }
        }
    }
}
