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

// vec_dim_count 
//
//  31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9  8  7  6  5  4  3  2  1  0
// +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
// |d |d |d |d |d |d |d |d |d |d |n |n |n |n |n |n |n |n |pc|pc|pc|pc|pc|pc|pc|pc|pc|pc|pc|pc|pc|pc| 
// +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
// |-----------|------- d -------|--------- n -----------|----------------- pc --------------------|
//
//  0-13 [0..16384] points_per_char 
// 14-21 [0..512]   number_of_chars
// 22-31 [0..1024]    draw_index
//
// mask points_per_char = 3FFF 
// mask number_of_chars = 3fc000 ( >> 14) 
// mask draw_index = ffc00000 ( >> 22)

struct Char {
    start_pos: vec3<f32>,
    font_size: f32,
    value: vec4<f32>,
    vec_dim_count: u32, // 1 => f32, 2 => vec3<f32>, 3 => vec3<f32>, 4 => vec4<f32>
    color: u32,
    decimal_count: u32,
    auxiliary_data: u32,
};

// fn get_points_per_char(aux_data: u32) -> u32 {
//     return aux_data & 0x3FFFu;
// }
// 
// fn get_number_of_chars(aux_data: u32) -> u32 {
//     return (aux_data & 0x3fc000u) >> 14u;
// }
// 
// fn get_draw_index(aux_data: u32) -> u32 {
//     return (aux_data & 0xffc00000u) >> 22u;
// }
// 
// fn set_points_per_char(v: u32, ch: ptr<function, Char>) {
//     (*ch).auxiliary_data = ((*ch).auxiliary_data & (!0x3FFFu)) | v;
// }
// 
// fn set_number_of_chars(v: u32, ch: ptr<function, Char>) {
//     (*ch).auxiliary_data = ((*ch).auxiliary_data & (!0x3fc000u)) | (v << 14u);
// }
// 
// fn set_draw_index(v: u32, ch: ptr<function, Char>) {
//     (*ch).auxiliary_data = ((*ch).auxiliary_data & (!0xffc00000u)) | (v << 22u);
// }

fn udiv_up_safe32(x: u32, y: u32) -> u32 {
    let tmp = (x + y - 1u) / y;
    return select(tmp, 0u, y == 0u); 
}

struct ModF {
    fract: f32,
    whole: f32,
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

struct FmmParams {
    fmm_global_dimension: vec3<u32>, 
    generation_method: u32, // 0 -> generate speed information, 1 -> generate fmm interface
    fmm_inner_dimension: vec3<u32>, 
    triangle_count: u32,
};

@group(0)
@binding(0)
var<uniform> fmm_params: FmmParams;

@group(0)
@binding(1)
var<storage, read_write> fmm_data: array<FmmCell>;

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

let THREAD_COUNT = 64u;
// let SCAN_BLOCK_SIZE = (THREAD_COUNT * 2u + ((THREAD_COUNT * 2u) >> 4u)); 
// 
// // The output of global active fmm block scan.
// shared uint[SCAN_BLOCK_SIZE] shared_prefix_sum;
// var<workgroup> shared_prefix_sum: array<FmmCell, SCAN_BLOCK_SIZE>; 

//////// ModF ////////

// fn myTruncate(f: f32) -> f32 {
//     return select(f32( i32( floor(f) ) ), f32( i32( ceil(f) ) ), f < 0.0); 
// }

fn my_modf(f: f32) -> ModF {
    //let iptr = myTruncate(f);
    let iptr = trunc(f);
    let fptr = f - iptr;
    return ModF (
        select(fptr, (-1.0)*fptr, f < 0.0),
        iptr
    );
}

///////////////////////////
////// Grid curve    //////
///////////////////////////

// Map index to 3d coordinate (hexahedron). The x and y dimensions are chosen. The curve goes from left to right, row by row.
// The z direction is "unlimited".
fn index_to_uvec3(index: u32, dim_x: u32, dim_y: u32) -> vec3<u32> {
  var x  = index;
  let wh = dim_x * dim_y;
  let z  = x / wh;
  x  = x - z * wh; // check
  let y  = x / dim_x;
  x  = x - y * dim_x;
  return vec3<u32>(x, y, z);
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

fn u32_rgba(c: u32) -> vec4<f32> {
  let a: f32 = f32(c & 256u) / 255.0;
  let b: f32 = f32((c & 65280u) >> 8u) / 255.0;
  let g: f32 = f32((c & 16711680u) >> 16u) / 255.0;
  let r: f32 = f32((c & 4278190080u) >> 24u) / 255.0;
  return vec4<f32>(r,g,b,a);
}

@compute
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    // Precondition. Interface is determined. (There should exists atleast one known point.)

    // Generate the initial first active blocks.

    // 1. Load one active block.

    // 2. Perform fmm.

    // 3. Synchronization.

    // 4. Reduction. 

    // 5. Prefix sum if necessery.
     
    let color = f32(rgba_u32(222u, 0u, 150u, 255u));

    if (global_id.x == 0u) {
        output_aabb_wire[global_id.x] =  
              AABB (
                  vec4<f32>(0.0, 0.0, 0.0, color),
                  vec4<f32>(vec3<f32>(fmm_params.fmm_global_dimension) * 4.0, 0.1)
              );
        atomicAdd(&counter[3], 1u);
    }

    // let cell = fmm_data[global_id.x];
    // let position = vec3<f32>(decode3Dmorton32(global_id.x)) * 0.25;

    // if (cell.tag == FAR) { return; }

    // var col = select(f32(rgba_u32(0u  , 255u, 0u, 255u)),
    //                  f32(rgba_u32(0u, 255u,   0u, 255u)),
    //                  cell.tag == KNOWN); 
    // output_aabb[atomicAdd(&counter[2], 1u)] =
    //       AABB (
    //           vec4<f32>(position - vec3<f32>(0.02), col),
    //           vec4<f32>(position + vec3<f32>(0.02), 0.0),
    //       );

    // output_char[atomicAdd(&counter[0], 1u)] =  
    //     
    //           Char (
    //               position.xyz + vec3<f32>(-0.018, 0.0, 0.022),
    //               0.01,
    //               vec4<f32>(f32(cell.value), 0.0, 0.0, 0.0),
    //               1u,
    //               rgba_u32(255u, 0u, 2550u, 255u),
    //               2u,
    //               0u
    //           );
    


















        //let min_box = vec3<f32>(global_id) + vec3<f32>(6.0); 
        //let max_box = vec3<f32>(global_id) + vec3<f32>(6.5); 

        //output_arrow[atomicAdd(&counter[1], 1u)] =  
        //      Arrow (
        //          vec4<f32>(f32(global_id.x), f32(global_id.y), f32(global_id.z), 0.0),
        //          vec4<f32>(f32(global_id.x), f32(global_id.y) + 4.0, f32(global_id.z), 0.0),
        //          rgba_u32(255u, 0u, 0u, 255u),
        //          0.5
        //);

        //output_aabb[atomicAdd(&counter[2], 1u)] =  
        //      AABB (
        //          vec4<f32>(min_box.x,
        //                    min_box.y,
        //                    min_box.z,
        //                    f32(rgba_u32(255u, 0u, 2550u, 255u))),
        //          vec4<f32>(max_box.x,
        //                    max_box.y, 
        //                    max_box.z,
        //                    1.0)
        //);
        //
        //output_aabb_wire[atomicAdd(&counter[3], 1u)] =  
        //      AABB (
        //          vec4<f32>(min_box.x,
        //                    min_box.y + 3.0,
        //                    min_box.z,
        //                    f32(rgba_u32(255u, 0u, 2550u, 255u))),
        //          vec4<f32>(max_box.x,
        //                    max_box.y + 3.0, 
        //                    max_box.z,
        //                    0.1)
        //);

        // for (var i: u32 = 0u ; i < 100u ; i = i + 1u ) {

        //     output_char[atomicAdd(&counter[0], 1u)] =  
        //     
        //           Char (
        //               vec3<f32>(min_box.x * 9.0,
        //                         min_box.y + f32(i) * 3.0,
        //                         min_box.z + 2.0
        //               ),
        //               0.3,
        //               vec4<f32>(f32(global_id.x) * 13.0 + 0.03 - 15.0, 0.0, f32(global_id.x) * 100.0, 0.0),
        //               3u,
        //               rgba_u32(255u, 0u, 2550u, 255u),
        //               4u,
        //               0u
        //           );
        //     }


    // if (local_index == 0u) {
    // }
    // workgroupBarrier();

    // Prefix sum

    // reduce
    
}
