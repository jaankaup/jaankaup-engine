/// Apply triangle mesh to fmm data. The interface creation.

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

struct Vertex {
    v: vec4<f32>;
    n: vec4<f32>;
};

struct Triangle {
    a: Vertex;
    b: Vertex;
    c: Vertex;
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
var<storage, read_write> triangle_mesh_in: array<Triangle>;

@group(0)
@binding(3)
var<storage, read_write> counter: array<atomic<u32>>;

@group(0)
@binding(4)
var<storage,read_write> output_char: array<Char>;

@group(0)
@binding(5)
var<storage,read_write> output_arrow: array<Arrow>;

@group(0)
@binding(6)
var<storage,read_write> output_aabb: array<AABB>;

@group(0)
@binding(7)
var<storage,read_write> output_aabb_wire: array<AABB>;

// Do we need this?
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

//// Intersection things. ////

// struct Triangle ?
fn closest_point_to_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {

    let ab = b - a;
    let ac = c - a;
    let bc = c - b;

    let d1 = dot(b-a, p-a);
    let d2 = dot(c-a, p-a);
    let d3 = dot(b-a, p-b);
    let d4 = dot(c-a, p-b);
    let d5 = dot(b-a, p-c);
    let d6 = dot(c-a, p-c);

    let va = d3*d6 - d5*d4;
    let vb = d5*d2 - d1*d6;
    let vc = d1*d4 - d3*d2;

    let snom = d1;
    let sdenom = -d3;
    let tnom = d2;
    let tdenom = -d6;
    let unom = d4 - d3;
    let udenom = d5 - d6;

    // TODO: optimize.

    if (snom <= 0.0 && tnom <= 0.0) { return a; }
    if (sdenom <= 0.0 && unom <= 0.0) { return b; }
    if (tdenom <= 0.0 && udenom <= 0.0) { return c; }

    let n = cross(b-a, c-a);

    if (vc <= 0.0 && snom >= 0.0 && sdenom >= 0.0) { return a + snom / (snom + sdenom) * ab; }
    if (va <= 0.0 && unom >= 0.0 && udenom >= 0.0) { return b + unom / (unom + udenom) * bc; }
    if (vb <= 0.0 && tnom >= 0.0 && tdenom >= 0.0) { return a + tnom / (tnom + tdenom) * ac; }

    let u = va / (va + vb + vc);
    let v = vb / (va + vb + vc);
    let w = 1.0 - u - v;
    return u * a + v * b + w * c;
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

@stage(compute)
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {



//++    // Initialize fmm values.
//++    let color = f32(rgba_u32(222u, 0u, 150u, 255u));
//++    let coordinate = decode3Dmorton32(global_id.x) * 3u;
//++    let coordinate2 = decode3Dmorton32(global_id.x + 1u) * 3u;
//++    let min_coord = vec4<f32>(vec3<f32>(coordinate) - vec3<f32>(0.3), color); 
//++    let max_coord = vec4<f32>(vec3<f32>(coordinate) + vec3<f32>(0.3), 1.0); 
//++
//++    output_arrow[global_id.x] =
//++        Arrow (
//++            vec4<f32>(vec3<f32>(coordinate), 1.0), 
//++            vec4<f32>(vec3<f32>(coordinate2), 1.0), 
//++            rgba_u32(0u, 0u, 255u, 255u),
//++            0.1
//++        );
//++    atomicAdd(&counter[1], 1u);
//++    
//++    output_aabb[global_id.x] =  
//++          AABB (
//++              min_coord,
//++              max_coord
//++          );
//++    atomicAdd(&counter[2], 1u);
//++
//++    output_char[global_id.x] =  
//++          Char (
//++              vec4<f32>(vec3<f32>(coordinate) + vec3<f32>(-0.3, 0.0, 0.31), 1.5), 
//++              vec4<f32>(f32(global_id.x), 0.0, 0.0, 0.0),
//++              0.1,
//++              1u,
//++              rgba_u32(0u, 0u, 2550u, 255u),
//++              0.1
//++          );
//++    atomicAdd(&counter[0], 1u);
}
