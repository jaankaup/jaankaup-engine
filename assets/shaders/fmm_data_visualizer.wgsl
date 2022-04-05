struct AABB {
    min: vec4<f32>, 
    // color: u32,
    max: vec4<f32>, 
    // padding: u32,
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
    vec_dim_count: u32, // 1 => f32, 2 => vec3<f32>, 3 => vec3<f32>, 4 => vec4<f32>
    color: u32,
    decimal_count: u32,
    auxiliary_data: u32,
};

// FMM tags
// 0 :: Far
// 1 :: Band
// 2 :: Band New
// 3 :: Known
// 4 :: Outside

let FAR      = 0u;
let BAND_NEW = 1u;
let BAND     = 2u;
let KNOWN    = 3u;
let OUTSIDE  = 4u;

struct FmmCell {
    tag: u32,
    value: f32,
};

struct FmmBlock {
    index: u32,
    band_points_count: u32,
};

struct FmmVisualizationParams {
    fmm_global_dimension: vec3<u32>, 
    visualization_method: u32, // a bit mask. 1 :: far, 2 :: band, 4 :: known 
    fmm_inner_dimension: vec3<u32>, 
    future_usage: u32,
};

@group(0) @binding(0) var<uniform> fmm_visualization_params: FmmVisualizationParams;
@group(0) @binding(1) var<storage, read_write> fmm_data: array<FmmCell>;
@group(0) @binding(2) var<storage, read_write> fmm_blocks: array<FmmBlock>;
@group(0) @binding(3) var<storage, read_write> isotropic_data: array<f32>;
@group(0) @binding(4) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> output_char: array<Char>;
@group(0) @binding(6) var<storage,read_write> output_arrow: array<Arrow>;
@group(0) @binding(7) var<storage,read_write> output_aabb: array<AABB>;
@group(0) @binding(8) var<storage,read_write> output_aabb_wire: array<AABB>;

let THREAD_COUNT = 64u;

//////// Common function  ////////

/**
 * [a1] from range min
 * [a2] from range max
 * [b1] to range min
 * [b2] to range max
 * [s] value to scale
 *
 * a1 != a2
 */
fn mapRange(a1: f32, a2: f32, b1: f32, b2: f32, s: f32) -> f32 {
    return b1 + (s - a1) * (b2 - b1) / (a2 - a1);
}

// Encode "rgba" to u32.
fn rgba_u32(r: u32, g: u32, b: u32, a: u32) -> u32 {
  return (r << 24u) | (g << 16u) | (b  << 8u) | a;
}

// fn u32_rgba(c: u32) -> vec4<f32> {
//   let a: f32 = f32(c & 256u) / 255.0;
//   let b: f32 = f32((c & 65280u) >> 8u) / 255.0;
//   let g: f32 = f32((c & 16711680u) >> 16u) / 255.0;
//   let r: f32 = f32((c & 4278190080u) >> 24u) / 255.0;
//   return vec4<f32>(r,g,b,a);
// }

///////////////////////////
////// MORTON CODE   //////
///////////////////////////

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

@stage(compute)
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    let color = bitcast<f32>(rgba_u32(222u, 0u, 150u, 255u));

    // Visualize the global fmm domain aabb.
    if (global_id.x == 0u) {
        output_aabb_wire[global_id.x] =  
              AABB (
                  vec4<f32>(0.0, 0.0, 0.0, color),
                  vec4<f32>(vec3<f32>(fmm_visualization_params.fmm_global_dimension), 0.1)
              );
        atomicAdd(&counter[3], 1u);
    }

    let cell = fmm_data[global_id.x];
    let position = vec3<f32>(decode3Dmorton32(global_id.x)) * 0.25;

    let color_far = rgba_u32(255u, 255u, 255u, 255u);
    let color_band = rgba_u32(255u, 0u, 0u, 255u);
    let color_known = rgba_u32(0u, 255u, 0u, 255u);
    var colors: array<u32, 5> = array<u32, 5>(color_far, 0u, color_known, color_far, 0u);  

    var col = colors[cell.tag]; 
    //var col = colors[3]; 

    // Visualize far points.
    if ((fmm_visualization_params.visualization_method & 1u) != 0u && cell.tag == FAR) {
        output_aabb[atomicAdd(&counter[2], 1u)] =
              AABB (
                  vec4<f32>(position - vec3<f32>(0.008), bitcast<f32>(col)),
                  vec4<f32>(position + vec3<f32>(0.008), 0.0),
              );
    }

    // Visualize band points.
    if ((fmm_visualization_params.visualization_method & 2u) != 0u && cell.tag == BAND) {
        output_aabb[atomicAdd(&counter[2], 1u)] =
              AABB (
                  vec4<f32>(position - vec3<f32>(0.008), bitcast<f32>(col)),
                  vec4<f32>(position + vec3<f32>(0.008), 0.0),
              );
    }

    // Visualize known points.
    if ((fmm_visualization_params.visualization_method & 4u) != 0u && cell.tag == KNOWN) {
        output_aabb[atomicAdd(&counter[2], 1u)] =
              AABB (
                  vec4<f32>(position - vec3<f32>(0.008), bitcast<f32>(col)),
                  vec4<f32>(position + vec3<f32>(0.008), 0.0),
              );
    }
}
