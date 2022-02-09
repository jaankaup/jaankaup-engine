// https://glmatrix.net/docs/quat.js.html

struct AABB {
    min: vec4<f32>; 
    max: vec4<f32>; 
};

struct VisualizationParams{
    triangle_start: u32;
    triangle_end: u32;
};

struct Counter {
    counter: atomic<u32>;
};

struct Vertex {
    v: vec4<f32>;
    n: vec4<f32>;
};

struct DebugArray {
    start: vec3<f32>;
    end:   vec3<f32>;
    color: u32;
    size:  f32;
};

struct TempAABB {
    aabb: AABB;
    triangle_data: array<Vertex, 36>;
};

let STRIDE: u32 = 64u;

// Lookup table for ctz. Maybe this should be a uniform.
var<private> lookup: array<u32, 32> = array<u32, 32>(
    0u, 1u, 28u, 2u, 29u, 14u, 24u, 3u, 30u, 22u, 20u, 15u, 25u, 17u, 4u, 8u,
    31u, 27u, 13u, 23u, 21u, 19u, 16u, 7u, 26u, 12u, 18u, 6u, 11u, 5u, 10u, 9u
);

// AABB triangulation.
var<private> vertex_positions: array<u32, 36> = array<u32, 36>(
    4u, 5u, 6u, 4u, 6u, 7u,
    5u, 1u, 2u, 5u, 2u, 6u,
    1u, 0u, 3u, 1u, 3u, 2u,
    0u, 4u, 7u, 0u, 7u, 3u,
    6u, 3u, 7u, 6u, 2u, 3u,
    4u, 0u, 5u, 0u, 1u, 5u
);
    
var<workgroup> temp_vertex_data: array<Vertex, 2304>; // 36 * 256 

@group(0)
@binding(0)
var<uniform> visualization_params: VisualizationParams;

@group(0)
@binding(1)
var<storage, read_write> counter: Counter;

@group(0)
@binding(2)
var<storage, read> debug_arrays: array<DebugArray>;

@group(0)
@binding(3)
var<storage,read_write> output: array<Vertex>;

// Get the bit value from postition i.
fn get_bit(x: u32, i: u32) -> u32 {
    return (x & (1u << i)) >> i;
}

// Modulo 3 for u32.
fn mod3_32(x: u32) -> u32 {
    var a = x;
    a = (a >> 16u) + (a & 0xFFFFu);
    a = (a >>  8u) + (a & 0xFFu);
    a = (a >>  4u) + (a & 0xFu);
    a = (a >>  2u) + (a & 0x3u);
    a = (a >>  2u) + (a & 0x3u);
    a = (a >>  2u) + (a & 0x3u);
    if (a > 2u) { a = a - 3u; }
    return a;
}

// Rotate first N bits in place to the right.
fn rotate_right(x: u32, amount: u32) -> u32 {
    let i = mod3_32(amount);
    return (x >> i)^(x << (3u-i)) & !(0xFFFFFFFFu<<3u);
}

// Rotate first N bits in place to the left.
fn rotate_left(x: u32, amount: u32) -> u32 {
    let i = mod3_32(amount);
    return (x << i) ^ (x >> (3u-i)) & !(0xFFFFFFFFu<<3u);
}

// Calculate the gray code.
fn gc(i: u32) -> u32 {
    return i ^ (i >> 1u);
}

// Calculate the entry point of hypercube i.
fn e(i: u32) -> u32 {
    if (i == 0u) { return 0u; } else { return gc(2u * ((i - 1u) / 2u)); }
}

// Calculate the exit point of hypercube i.
fn f(i: u32) -> u32 {
   return e((1u << 3u) - 1u - i) ^ (1u << 2u);
}

// Calculate the inverse gray code.
fn inverse_gc(g: u32) -> u32 {
    return (g ^ (g >> 1u)) ^ (g >> 2u);
}

// TODO: what happend when u32 is a really big number, larger than i32 can hold?
fn countTrailingZeros(x: u32) -> u32 {
    return lookup[
 	u32((i32(x) & (-(i32(x))))) * 0x077CB531u >> 27u
    ];
}

// Calculate the direction between i and the next one.
fn g(i: u32) -> u32 {
    return countTrailingZeros(!i);
}

// Calculate the direction of the arrow whitin a subcube.
fn d(i: u32) -> u32 {
    if (i == 0u) { return 0u; }
    else if ((i & 1u) == 0u) {
        return mod3_32(g(i - 1u));
    }
    else {
        return mod3_32(g(i));
    }
}

// Transform b.
fn t(e: u32, d: u32, b: u32) -> u32 {
    return rotate_right(b^e, d + 1u);
}

// Inverse transform.
fn t_inv(e: u32, d: u32, b: u32) -> u32 {
    return rotate_left(b, d + 1u) ^ e;
}

// Calculate the hilbert index from 3d point p.
fn to_hilbert_index(p: vec3<u32>, m: u32) -> u32 {

    var h = 0u;

    var ve: u32 = 0u;
    var vd: u32 = 0u;

    var i: u32 = m - 1u;
    loop {
        if (i == 0u) { break; }

        let l = get_bit(p.x, i) | (get_bit(p.y, i) << 1u) | (get_bit(p.z, i) << 2u);
    	let w = inverse_gc(t(l, ve, vd));
    	ve = ve ^ (rotate_left(e(w), vd + 1u));
    	vd = mod3_32(vd + d(w) + 1u);
        h = (h << 3u) | w;

        i = i - 1u;
    }
    return h;
}

/// Calculate 3d point from hilbert index.
fn from_hilber_index(h: u32, m: u32) -> vec3<u32> {
    
    var ve: u32 = 0u;
    var vd: u32 = 0u;
    var p = vec3<u32>(0u, 0u, 0u);

    var i: u32 = m - 1u;
    loop {
        if (i == 0u) { break; }

        let w = get_bit(h, i*3u) | (get_bit(h, i*3u + 1u) << 1u) | (get_bit(h, i*3u + 2u) << 2u);
    	let l = t_inv(ve, vd, gc(w)); 
    	p.x = (p.x << 1u) | ((l >> 0u) & 1u);
    	p.y = (p.y << 1u) | ((l >> 1u) & 1u);
    	p.z = (p.z << 1u) | ((l >> 2u) & 1u);
    	ve = ve ^ rotate_left(e(w), vd + 1u);
    	vd = mod3_32(vd + d(w) + 1u);

        i = i - 1u;
    }
    return p;
}

// Map index to 3d coordinate.
fn index_to_uvec3(index: u32, dim_x: u32, dim_y: u32) -> vec3<u32> {
  var x  = index;
  let wh = dim_x * dim_y;
  let z  = x / wh;
  x  = x - z * wh; // check
  let y  = x / dim_x;
  x  = x - y * dim_x;
  return vec3<u32>(x, y, z);
}

fn create_aabb(aabb: AABB, offset: u32, r: u32, g: u32, b: u32) {

    // Global start position.
    let index = atomicAdd(&counter.counter, 36u);

    let delta: vec4<f32> = aabb.max - aabb.min;

    let n_front  = vec4<f32>(0.0, 0.0, -1.0, 0.0);
    let n_back   = vec4<f32>(0.0, 0.0, 1.0, 0.0);
    let n_right  = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    let n_left   = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    let n_top    = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    let n_bottom = vec4<f32>(0.0, -1.0, 0.0, 0.0);

    var normals = array<vec4<f32>, 6>(
    	vec4<f32>(0.0, 0.0, -1.0, 0.0),
    	vec4<f32>(0.0, 0.0, 1.0, 0.0),
    	vec4<f32>(1.0, 0.0, 0.0, 0.0),
    	vec4<f32>(-1.0, 0.0, 0.0, 0.0),
    	vec4<f32>(0.0, 1.0, 0.0, 0.0),
    	vec4<f32>(0.0, -1.0, 0.0, 0.0)
    );


    //          p3
    //          +-----------------+p2
    //         /|                /|
    //        / |               / |
    //       /  |           p6 /  |
    //   p7 +-----------------+   |
    //      |   |             |   |
    //      |   |p0           |   |p1
    //      |   +-------------|---+  
    //      |  /              |  /
    //      | /               | /
    //      |/                |/
    //      +-----------------+
    //      p4                p5
    //

    // Decode color information to the fourth component.

    var positions = array<vec4<f32>, 8>(
    	aabb.min,
    	aabb.min + vec4<f32>(delta.x , 0.0     , 0.0, 0.0),
    	aabb.min + vec4<f32>(delta.x , delta.y , 0.0, 0.0),
    	aabb.min + vec4<f32>(0.0     , delta.y , 0.0, 0.0),
    	aabb.min + vec4<f32>(0.0     , 0.0     , delta.z, 0.0),
    	aabb.min + vec4<f32>(delta.x , 0.0     , delta.z, 0.0),
    	aabb.min + vec4<f32>(delta.x , delta.y , delta.z, 0.0),
    	aabb.min + vec4<f32>(0.0     , delta.y , delta.z, 0.0)
    );
    
    var i: u32 = 0u;

    loop {
        if (i == 36u) { break; }
        temp_vertex_data[offset + STRIDE * i]  = Vertex(positions[vertex_positions[i]], normals[i/6u]);
        i = i + 1u;
    }

    workgroupBarrier();

    var i: u32 = 0u;

    loop {
        if (i == 36u) { break; }
    	output[index+i] = temp_vertex_data[offset + STRIDE * i];
        i = i + 1u;
    }
}

// Create a renderable array.
fn create_array(index: u32) {
    
}

@stage(compute)
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    // let mapping = index_to_uvec3(global_id.x, 64u, 64u);
    var mapping =  from_hilber_index(global_id.x, 5u);
    var mapping2 = from_hilber_index(global_id.x + 1u, 5u);

    if ((global_id.x + 1u ) >= 32768u) { mapping2 = mapping; }
    
    let ma = max(mapping, mapping2);
    let mi = min(mapping, mapping2);
    
    let v0 = vec4<f32>(f32(mapping.x),
		       f32(mapping.y),
		       f32(mapping.z),
                       1.0
    );
    let v1 = vec4<f32>(f32(mapping2.x), 
		       f32(mapping2.y),
		       f32(mapping2.z),
                       1.0
    );
    // let aabb = AABB(vec4<f32>(min(v0.x, v1.x) - 0.01,
    //     		      min(v0.y, v1.y) - 0.01,
    //     		      min(v0.z, v1.z) - 0.01,
    //                           1.0),
    //                 vec4<f32>(max(v0.x, v1.x) + 0.01,
    //     		      max(v0.y, v1.y) + 0.01,
    //     		      max(v0.z, v1.z) + 0.01,
    //                           1.0)
    // );

    let aabb = AABB(vec4<f32>(f32(mi.x) - 0.05,
			      f32(mi.y) - 0.05,
			      f32(mi.z) - 0.05,
                              1.0),
                    vec4<f32>(f32(ma.x) + 0.05,
			      f32(ma.y) + 0.05,
			      f32(ma.z) + 0.05,
                              1.0)
    );
}
