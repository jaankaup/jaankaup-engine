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

@stage(compute)
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

}
