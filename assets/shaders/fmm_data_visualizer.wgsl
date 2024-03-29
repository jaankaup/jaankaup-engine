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

let FONT_SIZE = 0.15;
let AABB_SIZE = 0.16;

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
let KNOWN_NEW = 5u;

struct FmmCellPc {
    tag: u32,
    value: f32,
    color: u32,
};

struct FmmBlock {
    index: u32,
    band_points_count: u32,
};

struct FmmVisualizationParams {
    fmm_global_dimension: vec3<u32>, 
    visualization_method: u32, // a bit mask. 1 :: far, 2 :: band, 4 :: known, // 64 :: numbers
    fmm_inner_dimension: vec3<u32>, 
    future_usage: u32,
};

@group(0) @binding(0) var<uniform> fmm_visualization_params: FmmVisualizationParams;
@group(0) @binding(1) var<storage, read_write> fmm_data: array<FmmCellPc>;
// @group(0) @binding(2) var<storage, read_write> fmm_blocks: array<FmmBlock>;
// @group(0) @binding(3) var<storage, read_write> isotropic_data: array<f32>;
@group(0) @binding(2) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> output_char: array<Char>;
@group(0) @binding(4) var<storage,read_write> output_arrow: array<Arrow>;
@group(0) @binding(5) var<storage,read_write> output_aabb: array<AABB>;
@group(0) @binding(6) var<storage,read_write> output_aabb_wire: array<AABB>;

//////// Common function  ////////

fn total_cell_count() -> u32 {

    return fmm_visualization_params.fmm_global_dimension.x *
           fmm_visualization_params.fmm_global_dimension.y *
           fmm_visualization_params.fmm_global_dimension.z *
           fmm_visualization_params.fmm_inner_dimension.x *
           fmm_visualization_params.fmm_inner_dimension.y *
           fmm_visualization_params.fmm_inner_dimension.z;
};

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

struct ModF {
    fract: f32,
    whole: f32,
};

///////////////////////////
////// CHAR COUNT    //////
///////////////////////////

fn myTruncate(f: f32) -> f32 {
    return select(f32( i32( floor(f) ) ), f32( i32( ceil(f) ) ), f < 0.0); 
}

fn my_modf(f: f32) -> ModF {
    let iptr = trunc(f);
    let fptr = f - iptr;
    return ModF (
        select(fptr, (-1.0)*fptr, f < 0.0),
        iptr
    );
}

/// if the given integer is < 0, return 0. Otherwise 1 is returned.
fn the_sign_of_i32(n: i32) -> u32 {
    return (u32(n) >> 31u);
    //return 1u ^ (u32(n) >> 31u);
}

fn abs_i32(n: i32) -> u32 {
    let mask = u32(n) >> 31u;
    return (u32(n) + mask) ^ mask;
}

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

var<private> PowersOf10: array<u32, 10> = array<u32, 10>(1u, 10u, 100u, 1000u, 10000u, 100000u, 1000000u, 10000000u, 100000000u, 1000000000u);

// NOT defined if u == 0u.
fn log10_u32(n: u32) -> u32 {
    
    var v = n;
    var r: u32;
    var t: u32;
    
    t = (log2_u32(v) + 1u) * 1233u >> 12u;
    r = t - u32(v < PowersOf10[t]);
    return r;
}

fn number_of_chars_i32(n: i32) -> u32 {

    if (n == 0) { return 1u; }
    return the_sign_of_i32(n) + log10_u32(u32(abs(n))) + 1u;
} 

fn number_of_chars_f32(f: f32, number_of_decimals: u32) -> u32 {

    let m = my_modf(f);
    let minus = select(0u, 1u, m.fract < 0.0 && i32(m.whole) == 0); // New
    return number_of_chars_i32(i32(m.whole)) + number_of_decimals + 1u + minus;
}

//++ fn number_of_chars_f32(f: f32, number_of_decimals: u32) -> u32 {
//++ 
//++     let m = my_modf(f);
//++     return number_of_chars_i32(i32(m.whole)) + number_of_decimals + 1u;
//++ }


// dots, brachets
// vec_dim_count == 0   _  not used
// vec_dim_count == 1   a
// vec_dim_count == 2   (a , b)
// vec_dim_count == 3   (a , b , c)
// vec_dim_count == 4   (a , b , c , d)
// vec_dim_count == 5   undefined
//var<private> NumberOfExtraChars: array<u32, 5> = array<u32, 5>(0u, 0u, 3u, 4u, 5u);
var<private> NumberOfExtraChars: array<u32, 5> = array<u32, 5>(0u, 0u, 2u, 2u, 2u);

// TODO: Instead of data, Element as reference.
fn number_of_chars_data(data: vec4<f32>, vec_dim_count: u32, number_of_decimals: u32) -> u32 {
    
    // Calculate all possible char counts to avoid branches.
    let a = number_of_chars_f32(data.x, number_of_decimals) * u32(vec_dim_count >= 1u);
    let b = number_of_chars_f32(data.y, number_of_decimals) * u32(vec_dim_count >= 2u);
    let c = number_of_chars_f32(data.z, number_of_decimals) * u32(vec_dim_count >= 3u);
    let d = number_of_chars_f32(data.w, number_of_decimals) * u32(vec_dim_count >= 4u);

    return a + b + c + d + NumberOfExtraChars[vec_dim_count]; 
}

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

/// xy-plane indexing. (x,y,z) => index
fn index_to_uvec3(index: u32, dim_x: u32, dim_y: u32) -> vec3<u32> {
  var x  = index;
  let wh = dim_x * dim_y;
  let z  = x / wh;
  x  = x - z * wh;
  let y  = x / dim_x;
  x  = x - y * dim_x;
  return vec3<u32>(x, y, z);
}

/// Get memory index from given cell coordinate.
fn get_cell_mem_location(v: vec3<u32>) -> u32 {

    let stride = fmm_visualization_params.fmm_inner_dimension.x * fmm_visualization_params.fmm_inner_dimension.y * fmm_visualization_params.fmm_inner_dimension.z;

    let xOffset = 1u;
    let yOffset = fmm_visualization_params.fmm_global_dimension.x;
    let zOffset = yOffset * fmm_visualization_params.fmm_global_dimension.y;

    let global_coordinate = vec3<u32>(v) / fmm_visualization_params.fmm_inner_dimension;

    let global_index = (global_coordinate.x * xOffset +
                        global_coordinate.y * yOffset +
                        global_coordinate.z * zOffset) * stride;

    let local_coordinate = vec3<u32>(v) - global_coordinate * fmm_visualization_params.fmm_inner_dimension;

    let local_index = encode3Dmorton32(local_coordinate.x, local_coordinate.y, local_coordinate.z);

    return global_index + local_index; 
}

fn get_cell_index(global_index: u32) -> vec3<u32> { 

    let stride = fmm_visualization_params.fmm_inner_dimension.x * fmm_visualization_params.fmm_inner_dimension.y * fmm_visualization_params.fmm_inner_dimension.z;
    let block_index = global_index / stride;
    let block_position = index_to_uvec3(block_index, fmm_visualization_params.fmm_global_dimension.x, fmm_visualization_params.fmm_global_dimension.y) * fmm_visualization_params.fmm_inner_dimension;

    // calculate local position.
    let local_index = global_index - block_index * stride;

    let local_position = decode3Dmorton32(local_index);

    let cell_position = block_position + local_position;

    return cell_position; 
}

fn visualize_cell(position: vec3<f32>, color: u32) {
    output_aabb[atomicAdd(&counter[2], 1u)] =
          AABB (
              4.0 * vec4<f32>(position - vec3<f32>(AABB_SIZE), 0.0) + vec4<f32>(0.0, 0.0, 0.0, bitcast<f32>(color)),
              4.0 * vec4<f32>(position + vec3<f32>(AABB_SIZE), 0.0),
          );
}

@compute
@workgroup_size(128,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        // @builtin(workgroup_id) workgroup_id: vec3<u32>,
        // @builtin(num_workgroups) num_workgroups: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    // let color = bitcast<f32>(rgba_u32(222u, 0u, 150u, 255u));
    let cell_count = total_cell_count();

    if (global_id.x >= cell_count) { return; }

    let cell = fmm_data[global_id.x];

    var position = vec3<f32>(get_cell_index(global_id.x)); // * 0.25;

    let color_far = rgba_u32(255u, 255u, 255u, 255u);
    let color_band = rgba_u32(255u, 0u, 0u, 255u);
    let color_band_new = rgba_u32(255u, 0u, 0u, 255u);
    // let color_band_new = rgba_u32(255u, 255u, 50u, 255u);
    let color_known = rgba_u32(0u, 255u, 0u, 255u);
    let color_text = rgba_u32(255u, 255u, 255u, 255u);
    let color_known_new = rgba_u32(123u, 123u, 50u, 255u);
    //var colors: array<u32, 5> = array<u32, 5>(color_far, color_band, color_band_new, color_known, color_outside, color_known_new);
    var colors: array<u32, 6> = array<u32, 6>(color_far, color_band_new, color_band, color_known, 0u, color_known_new);
    //let colors_ptr = &colors;

    var col = colors[cell.tag];

    // TODO: spaces!
    let value = vec4<f32>(cell.value, 0.0, 0.0, 0.0);
    let total_number_of_chars = number_of_chars_data(value, 1u, 2u);
    //let element_position = position - vec3<f32>(f32(total_number_of_chars) * FONT_SIZE * 0.5, 0.0, (-1.0) * AABB_SIZE - FONT_SIZE*4.0 + 0.01);
    let element_position = 4.0 * position - vec3<f32>(f32(total_number_of_chars) * FONT_SIZE * 0.5, 0.0, -AABB_SIZE - 4.0 * FONT_SIZE - 0.01);
    let renderable_element = Char (
                    // element_position,
                    element_position,
                    FONT_SIZE,
                    value,
                    1u,
                    color_text,
                    4u,
                    0u
    );

    if ((fmm_visualization_params.visualization_method & 1u) != 0u && cell.tag == FAR) {

        visualize_cell(position, col);

        if ((fmm_visualization_params.visualization_method & 64u) != 0u) {

            output_char[atomicAdd(&counter[0], 1u)] = renderable_element; 
        }
    }

    // if ((fmm_visualization_params.visualization_method & 2u) != 0u && (cell.tag == BAND || cell.tag == BAND_NEW) ) {
    //if ((fmm_visualization_params.visualization_method & 2u) != 0u && cell.tag == BAND ) {
    if ((fmm_visualization_params.visualization_method & 2u) != 0u && cell.tag == BAND_NEW) {

        visualize_cell(position, col);

        if ((fmm_visualization_params.visualization_method & 64u) != 0u) {

            output_char[atomicAdd(&counter[0], 1u)] = renderable_element; 
        }
    }

    if ((fmm_visualization_params.visualization_method & 4u) != 0u && cell.tag == KNOWN) {
                           
        // visualize_cell(position, cell.color);
	// if (cell.value > 100000.0) { visualize_cell(position, color_known_new); }  
        // visualize_cell(position, cell.color);
	// else { visualize_cell(position, col); }
        visualize_cell(position, col);

        if ((fmm_visualization_params.visualization_method & 64u) != 0u) {

            output_char[atomicAdd(&counter[0], 1u)] = renderable_element; 
        }
    }

    //++ let ranged_color = mapRange(0.0, 100.0, 0.0, 255.0, speed);
    //++ let ranged_color_rgba = rgba_u32(u32(ranged_color), 10u, 0u, 255u);

    //++ // Visualize isotropic speed data.
    //++ if ((fmm_visualization_params.visualization_method & 8u) != 0u) {
    //++     visualize_cell(position, ranged_color_rgba);
    //++     if ((fmm_visualization_params.visualization_method & 64u) != 0u) {

    //++         output_char[atomicAdd(&counter[0], 1u)] =  
    //++             
    //++             Char (
    //++                 element_position,
    //++                 FONT_SIZE,
    //++                 vec4<f32>(speed, 0.0, 0.0, 0.0),
    //++                 1u,
    //++                 color_text,
    //++                 2u,
    //++                 0u
    //++             );
    //++      }
    //++ }
}
