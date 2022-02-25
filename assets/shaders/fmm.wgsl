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

var<workgroup> wg_arrow_count: atomic<u32>;

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

fn reset_workgroup_counters() {
    wg_arrow_count = 0u;
}

fn update_workgroup_counters(wg_arr_count: u32,
                             wg_aabb_count: u32,
                             wg_aabb_wire_count: u32,
                             wg_char_count: u32) {
    atomicAdd(&counter[0], wg_char_count);
    atomicAdd(&counter[1], wg_arr_count);
    atomicAdd(&counter[2], wg_aabb_count);
    atomicAdd(&counter[3], wg_aabb_wire_count);
}

@stage(compute)
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    if (local_index == 0u) {
        reset_workgroup_counters();
    }
    workgroupBarrier();
    
    for (var i: i32 = 0; i < 100 ; i = i + 1) {
        output_arrow[local_index + 64u * u32(i)] =  
              Arrow (
                  vec4<f32>(f32(i * 64 + i32(local_index)) * 0.2 + 0.5,
                            0.0,
                            4.0 * f32(local_index),
                            1.0),
                  vec4<f32>(f32(i * 64 + i32(local_index)) * 0.2,
                            4.0,
                            4.0 * f32(local_index+1u),
                            1.0),
                  rgba_u32(255u, 0u, 0u, 255u),
                  0.5
              );
    }

    for (var i: i32 = 0; i < 100 ; i = i + 1) {
        output_aabb[local_index + 64u * u32(i)] =  
              AABB (
                  vec4<f32>(f32(i * 64 + i32(local_index)) * 0.2 + 0.5,
                            15.0,
                            4.0 * f32(local_index+1u),
                            f32(rgba_u32(255u, 0u, 2550u, 255u))),
                  vec4<f32>(f32(i * 64 + i32(local_index)) * 0.2,
                            16.0,
                            4.0 * f32(local_index),
                            1.0)
              );
    }

    for (var i: i32 = 0; i < 100 ; i = i + 1) {
        output_aabb_wire[local_index + 64u * u32(i)] =  
              AABB (
                  vec4<f32>(f32(i * 64 + i32(local_index)) * 0.2 + 0.5,
                            15.0,
                            40.0 + 4.0 * f32(local_index+1u),
                            f32(rgba_u32(255u, 0u, 2550u, 255u))),
                  vec4<f32>(f32(i * 64 + i32(local_index)) * 0.2,
                            40.0 + 16.0,
                            4.0 * f32(local_index),
                            0.2)
              );
    }

    for (var i: i32 = 0; i < 100 ; i = i + 1) {
        var vec_dimension = 1u;

        if (i % 4 == 0) {
            vec_dimension = 4u; 
        }
        if (i % 3 == 0) {
            vec_dimension = 3u; 
        }
        if (i % 2 == 0) {
            vec_dimension = 1u; 
        }
        if (i % 1 == 0) {
            vec_dimension = 1u; 
        }

        output_char[local_index + 64u * u32(i)] =  

              // struct Char {
              //     start_pos: vec4<f32>;
              //     value: vec4<f32>;
              //     font_size: f32;
              //     vec_dim_count: u32; // 1 => f32, 2 => vec3<f32>, 3 => vec3<f32>, 4 => vec4<f32>
              //     color: u32;
              //     z_offset: f32;
              // };


              Char (
                  vec4<f32>(f32(i * 64 + i32(local_index)) * 0.2 + 0.5,
                            65.0,
                            40.0 + 4.0 * f32(local_index+1u),
                            2.0
                  ),
                  // vec4<f32>(1.0, 1.0, 1.0, 0.0),
                  vec4<f32>(f32(i), f32(i+5*i), f32(-i), 0.0),
                  0.5,
                  vec_dimension,
                  rgba_u32(255u, 0u, 2550u, 255u),
                  0.1
              );
    }

    if (local_index == 0u) {
        update_workgroup_counters(100u * 64u, 100u * 64u, 100u * 64u, 100u * 64u);
    }
}
