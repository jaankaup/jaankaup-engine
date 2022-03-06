struct AABB {
    min: vec4<f32>; 
    max: vec4<f32>; 
};

struct Arrow {
    start_pos: vec4<f32>;
    end_pos:   vec4<f32>;
    color: u32;
    size:  f32;
};

struct Char {
    start_pos: vec4<f32>;
    value: vec4<f32>;
    font_size: f32;
    vec_dim_count: u32; // 1 => f32, 2 => vec3<f32>, 3 => vec3<f32>, 4 => vec4<f32>
    color: u32;
    z_offset: f32;
};

struct ModF {
    fract: f32;
    whole: f32;
};

struct FmmCell {
    tag: u32;
    value: f32;
};

struct FmmParams {
    blah: f32;
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

fn myTruncate(f: f32) -> f32 {
    return select(f32( i32( floor(f) ) ), f32( i32( ceil(f) ) ), f < 0.0); 
}

fn my_modf(f: f32) -> ModF {
    let iptr = myTruncate(f);
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

@stage(compute)
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    // if (local_index == 0u) {
    // }
    // workgroupBarrier();

    // Prefix sum

    // reduce
    
}
