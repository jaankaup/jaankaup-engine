struct AABB {
    min: vec4<f32>, 
    max: vec4<f32>, 
};

struct Quaternion {
    w: f32,
    x: f32,
    y: f32,
    z: f32,
};

struct VisualizationParams{
    curve_number: u32,
    iterator_start_index: u32,
    iterator_end_index: u32,
    arrow_size: f32,
    thread_mode: u32,
    thread_mode_start_index: u32,
    thread_mode_end_index: u32,
};

struct Vertex {
    v: vec4<f32>,
    n: vec4<f32>,
};

struct Triangle {
    a: Vertex,
    b: Vertex,
    c: Vertex,
};

type Hexaedra = array<Vertex , 8>;

struct Arrow {
    start_pos: vec4<f32>,
    end_pos:   vec4<f32>,
    color: u32,
    size:  f32,
};

let THREAD_COUNT: u32 = 256u;
let PI: f32 = 3.14159265358979323846;

// AABB triangulation.
var<private> vertex_positions: array<u32, 36> = array<u32, 36>(
    4u, 5u, 6u, 4u, 6u, 7u,
    5u, 1u, 2u, 5u, 2u, 6u,
    1u, 0u, 3u, 1u, 3u, 2u,
    0u, 4u, 7u, 0u, 7u, 3u,
    6u, 3u, 7u, 6u, 2u, 3u,
    4u, 0u, 5u, 0u, 1u, 5u
);

var<private> aabb_normals: array<vec4<f32>, 6> = array<vec4<f32>, 6>(
    vec4<f32>(0.0, 0.0, -1.0, 0.0),
    vec4<f32>(0.0, 0.0, 1.0, 0.0),
    vec4<f32>(1.0, 0.0, 0.0, 0.0),
    vec4<f32>(-1.0, 0.0, 0.0, 0.0),
    vec4<f32>(0.0, 1.0, 0.0, 0.0),
    vec4<f32>(0.0, -1.0, 0.0, 0.0)
);
    
// var<workgroup> temp_vertex_data: array<Vertex, 2304>; // 36 * 256 

var<workgroup> thread_group_counter: u32 = 0; 

@group(0)
@binding(0)
var<uniform> visualization_params: VisualizationParams;

// 0 :: total number of written vertices.
@group(0)
@binding(1)
var<storage, read_write> counter: array<atomic<u32>>;

@group(0)
@binding(2)
var<storage, read_write> arrows: array<Arrow>;

@group(0)
@binding(3)
var<storage,read_write> output_data: array<Triangle>;

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

// ROSENBERG-STRONG

fn r2(x: i32, y: i32) -> i32 {
    let max2 = max(x,y);
    return max2 * max2 + max2 + x - y;
}

fn r3(x: i32, y: i32, z: i32) -> i32 {
    let max2 = max(x,y);
    let max3  = max(z, max2);
    let max3_2 = max3 * max3;
    let max3_3 = max3 * max3_2;
    return max2 * max2 + max2 + x - y + max3_3 + (max3 - z) * (2 * max3 + 1);
}

fn r2_reverse(z: f32) -> vec2<i32> {
    let m = floor(sqrt(z));
    let t = z - m * m;
    if (t < m) { return vec2<i32>(i32(t), i32(m)); }
    else {
        return vec2<i32>(i32(m), i32( m * m + 2.0 * m - z));
    }
}

fn cbrt(x: f32) -> f32 {

    // CHEATING! The precision of f32 and log2 are not enought.
    let lg2 = log2(x) + 0.00001;

    if (x > 0.0) { return exp2(lg2 / 3.0); }
    return 0.0;
}

fn r3_reverse(z: f32) -> vec3<i32> {
    let m = floor(cbrt(z));
    let t = z - m*m*m - m*m;
    let v = max(0.0, t);
    let x3 = m - floor(
                     v / ((m + 1.0) * (m + 1.0) - m*m)
                 );
    let xy = r2_reverse(z - m*m*m - (m - x3) * ((m + 1.0) * (m + 1.0) - m*m));
    return vec3<i32>(xy.x, xy.y, i32(x3));
}

// THE HILBERT STUFF.

// Lookup table for ctz. Maybe this should be a uniform.
var<private> lookup: array<u32, 32> = array<u32, 32>(
    0u, 1u, 28u, 2u, 29u, 14u, 24u, 3u, 30u, 22u, 20u, 15u, 25u, 17u, 4u, 8u,
    31u, 27u, 13u, 23u, 21u, 19u, 16u, 7u, 26u, 12u, 18u, 6u, 11u, 5u, 10u, 9u
);

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
    return select(a, a - 3u, a > 2u);
}

// Rotate first N bits in place to the right. This is now tuned for 3-bits.
// For more general function, implement function that has number of bits as parameter.
fn rotate_right(x: u32, amount: u32) -> u32 {
    let i = mod3_32(amount);
    return (x >> i)^(x << (3u-i)) & !(0xFFFFFFFFu<<3u);
}

// Rotate first N bits in place to the left. This is now tuned for 3-bits.
// For more general function, implement function that has number of bits as parameter.
fn rotate_left(x: u32, amount: u32) -> u32 {
    let i = mod3_32(amount);
    return (x << i) ^ (x >> (3u-i)) & !(0xFFFFFFFFu<<3u);
}

// Calculate the gray code.
fn gc(i: u32) -> u32 {
    return i ^ (i >> 1u);
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

///////////////////////////
////// HILBERT INDEX //////
///////////////////////////

// Calculate the entry point of hypercube i.
fn e(i: u32) -> u32 {
    let f = gc(2u * ((i - 1u) / 2u));
    return select(f, 0u, i == 0u);
    // if (i == 0u) { return 0u; } else { return gc(2u * ((i - 1u) / 2u)); }
}

// Calculate the exit point of hypercube i.
fn f(i: u32) -> u32 {
   return e((1u << 3u) - 1u - i) ^ (1u << 2u);
}

// Calculate the direction between i and the next one.
fn g(i: u32) -> u32 {
    return countTrailingZeros(!i);
}

// Calculate the direction of the arrow whitin a subcube.
fn d(i: u32) -> u32 {

    // Avoid branches.
    let s0 = 0u;
    let s1 = mod3_32(g(i - 1u));
    let s2 = mod3_32(g(i));

    return select(select(s1, 0u, i == 0u), s2, (i & 1u) != 0u);
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
// m is the number of bits that represent the value of single coordinate.
// If m is 2, then the compute domain is: m X m X m => 2x2x2
fn to_hilbert_index(p: vec3<u32>, m: u32) -> u32 {

    var h = 0u;

    var ve: u32 = 1u;
    var vd: u32 = 0u;

    var i: u32 = m - 1u;
    loop {

        let l = get_bit(p.x, i) | (get_bit(p.y, i) << 1u) | (get_bit(p.z, i) << 2u);
    	let w = inverse_gc(t(l, ve, vd));
    	ve = ve ^ (rotate_left(e(w), vd + 1u));
    	vd = mod3_32(vd + d(w) + 1u);
        h = (h << 3u) | w;

        i = i - 1u;

        if (i == 0u) { break; }
    }
    return h;
}

// Calculate 3d point from hilbert index.
// m is the number of bits that represent the value of single coordinate.
// If m is 2, then the compute domain is: m X m X m => 2x2x2
fn from_hilbert_index(h: u32, m: u32) -> vec3<u32> {
    
    var ve: u32 = 0u;
    var vd: u32 = 0u;
    var p = vec3<u32>(0u, 0u, 0u);

    var i: u32 = m - 1u;
    loop {

        let w = get_bit(h, i*3u) | (get_bit(h, i*3u + 1u) << 1u) | (get_bit(h, i*3u + 2u) << 2u);
    	let l = t_inv(ve, vd, gc(w)); 
    	p.x = (p.x << 1u) | ((l >> 0u) & 1u);
    	p.y = (p.y << 1u) | ((l >> 1u) & 1u);
    	p.z = (p.z << 1u) | ((l >> 2u) & 1u);
    	ve = ve ^ rotate_left(e(w), vd + 1u);
    	vd = mod3_32(vd + d(w) + 1u);

        if (i == 0u) { break; }

        i = i - 1u;
    }
    return p;
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


fn quaternion_id() -> Quaternion {
    return Quaternion(1.0, 0.0, 0.0, 0.0);
}

fn quaternion_add(a: Quaternion, b: Quaternion) -> Quaternion {
    return Quaternion(
        a.w + b.w,
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    );
}

fn quaternion_mul(a: Quaternion, b: Quaternion) -> Quaternion {
    return Quaternion(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w
    );
}

fn quaternion_scale(q: Quaternion, s: f32) -> Quaternion {
    return Quaternion(
        q.w * s,
        q.x * s, 
        q.y * s, 
        q.z * s 
    );
}

fn quaternion_dot(a: Quaternion, b: Quaternion) -> f32 {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z; 
}

fn quaternion_conjugate(q: Quaternion) -> Quaternion {
    return Quaternion(
         q.w,
        -q.x,
        -q.y,
        -q.z
    );
}

// n == 0?
fn quaternion_normalize(q: Quaternion) -> Quaternion {
    let n = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w); 
    return Quaternion(
        q.w / n,
        q.x / n,
        q.y / n,
        q.z / n
    );
} 

fn rotate_vector(q: Quaternion, v: vec3<f32>) -> vec3<f32> {
    let t = cross(vec3<f32>(q.x, q.y, q.z), v) * 2.0; 
    return v + q.w * t + cross(vec3<f32>(q.x, q.y, q.z), t);
}

fn axis_angle(axis: vec3<f32>, angle: f32) -> Quaternion {
    let half_angle = angle * 0.5;
    let bl = axis * sin(half_angle);
    return Quaternion(cos(half_angle), bl.x, bl.y, bl.z); 
}

fn vec3_cross(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
   );
}

fn rotation_from_to(a: vec3<f32>, b: vec3<f32>) -> Quaternion {

    let v = cross(a,b);  

    let a_len = length(a);
    let b_len = length(b);

    let w = sqrt(a_len * a_len * b_len * b_len) + dot(a,b);

    if ( a.x == 1.0 && b.x == -1.0 && a.y == 0.0 && b.y == 0.0 && a.z == 0.0 && b.z == 0.0) { 
        
        var axis = cross(vec3<f32>(1.0, 0.0, 0.0), a);
        if (length(axis) == 0.0) {
            axis = cross(vec3<f32>(0.0, 1.0, 0.0), a);
        }
        axis = normalize(axis);
        return axis_angle(axis, PI);
    }
     
    return quaternion_normalize(Quaternion(w, v.x, v.y, v.z));
}

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

//++ fn store_hexaedron(positions: ptr<function, array<vec4<f32>, 8>>,
//++                    normals:   ptr<function, array<vec4<f32>, 6>>,
//++                    offset: u32) {
//++ 
//++     let index = atomicAdd(&counter[0], 36u);
//++ 
//++     var i: u32 = 0u;
//++ 
//++     loop {
//++         if (i == 36u) { break; }
//++         temp_vertex_data[offset + STRIDE * i]  =
//++             Vertex(
//++                 (*positions)[vertex_positions[i]],
//++                 (*normals)[i/6u]
//++             );
//++ 
//++         i = i + 1u;
//++     }
//++ 
//++     workgroupBarrier();
//++ 
//++     var i: u32 = 0u;
//++ 
//++     loop {
//++         if (i == 36u) { break; }
//++     	output_data[index+i] = temp_vertex_data[offset + STRIDE * i];
//++         i = i + 1u;
//++     }
//++ 
//++     workgroupBarrier();
//++ } 

//++ fn create_aabb(aabb: AABB, offset: u32, color: u32) {
//++ 
//++     // Global start position.
//++     let index = atomicAdd(&counter[0], 36u);
//++ 
//++     //          p3
//++     //          +-----------------+p2
//++     //         /|                /|
//++     //        / |               / |
//++     //       /  |           p6 /  |
//++     //   p7 +-----------------+   |
//++     //      |   |             |   |
//++     //      |   |p0           |   |p1
//++     //      |   +-------------|---+  
//++     //      |  /              |  /
//++     //      | /               | /
//++     //      |/                |/
//++     //      +-----------------+
//++     //      p4                p5
//++     //
//++ 
//++     // Decode color information to the fourth component.
//++ 
//++     let c = f32(color);
//++ 
//++     let delta = aabb.max - aabb.min;
//++ 
//++     var positions = array<vec4<f32>, 8>(
//++     	vec4<f32>(aabb.min.xyz, c),
//++     	vec4<f32>(aabb.min.xyz, c) + vec4<f32>(delta.x , 0.0     , 0.0, 0.0),
//++     	vec4<f32>(aabb.min.xyz, c) + vec4<f32>(delta.x , delta.y , 0.0, 0.0),
//++     	vec4<f32>(aabb.min.xyz, c) + vec4<f32>(0.0     , delta.y , 0.0, 0.0),
//++     	vec4<f32>(aabb.min.xyz, c) + vec4<f32>(0.0     , 0.0     , delta.z, 0.0),
//++     	vec4<f32>(aabb.min.xyz, c) + vec4<f32>(delta.x , 0.0     , delta.z, 0.0),
//++     	vec4<f32>(aabb.min.xyz, c) + vec4<f32>(delta.x , delta.y , delta.z, 0.0),
//++     	vec4<f32>(aabb.min.xyz, c) + vec4<f32>(0.0     , delta.y , delta.z, 0.0)
//++     );
//++     
//++     var i: u32 = 0u;
//++ 
//++     loop {
//++         if (i == 36u) { break; }
//++         temp_vertex_data[offset + STRIDE * i]  = Vertex(
//++                                                     positions[vertex_positions[i]],
//++                                                     vec4<f32>(aabb_normals[i/6u])
//++         );
//++         i = i + 1u;
//++     }
//++ 
//++     workgroupBarrier();
//++ 
//++     var i: u32 = 0u;
//++ 
//++     loop {
//++         if (i == 36u) { break; }
//++     	output_data[index+i] = temp_vertex_data[offset + STRIDE * i];
//++         i = i + 1u;
//++     }
//++ 
//++     workgroupBarrier();
//++ }

fn create_arrow(arr: Arrow, offset: u32, local_index: u32) {

    var direction = arr.end_pos.xyz - arr.start_pos.xyz;
    let array_length = length(direction);
    direction = normalize(direction);

    let head_size = min(0.5 * array_length, 2.0 * arr.size);

    //          array_length
    //    +----------------------+
    // s  |                      | ------> x
    //    +-----------+----------+
    //              x = 0

    let from_origo_x = vec3<f32>(vec3<f32>(array_length, arr.size, arr.size)) - vec3<f32>(head_size, 0.0, 0.0);

    let aabb = AABB(
                   vec4<f32>((-0.5) * from_origo_x, 0.0),
                   vec4<f32>(0.5    * from_origo_x, 0.0)
    );

    var delta = aabb.max - aabb.min;

    let q = rotation_from_to(vec3<f32>(1.0, 0.0, 0.0), direction);
    let the_pos = arr.start_pos.xyz + rotate_vector(q, vec3<f32>(1.0, 0.0, 0.0)) * 0.5 * array_length;

    let c = bitcast<f32>(arr.color);

    var positions = array<vec4<f32>, 8>(
        vec4<f32>(the_pos - 0.5 * head_size * direction + rotate_vector(q, aabb.min.xyz), c),
        vec4<f32>(the_pos - 0.5 * head_size * direction + rotate_vector(q, aabb.min.xyz + vec3<f32>(delta.x , 0.0     , 0.0)), c),
        vec4<f32>(the_pos - 0.5 * head_size * direction + rotate_vector(q, aabb.min.xyz + vec3<f32>(delta.x , delta.y , 0.0)), c),
        vec4<f32>(the_pos - 0.5 * head_size * direction + rotate_vector(q, aabb.min.xyz + vec3<f32>(0.0     , delta.y , 0.0)), c),
        vec4<f32>(the_pos - 0.5 * head_size * direction + rotate_vector(q, aabb.min.xyz + vec3<f32>(0.0     , 0.0     , delta.z)), c),
        vec4<f32>(the_pos - 0.5 * head_size * direction + rotate_vector(q, aabb.min.xyz + vec3<f32>(delta.x , 0.0     , delta.z)), c),
        vec4<f32>(the_pos - 0.5 * head_size * direction + rotate_vector(q, aabb.min.xyz + vec3<f32>(delta.x , delta.y , delta.z)), c),
        vec4<f32>(the_pos - 0.5 * head_size * direction + rotate_vector(q, aabb.min.xyz + vec3<f32>(0.0     , delta.y , delta.z)), c)
    );

    var n0 = normalize(cross(positions[5u].xyz - positions[6u].xyz,
                             positions[4u].xyz - positions[6u].xyz));
    var n1 = normalize(cross(positions[1u].xyz - positions[2u].xyz,
                             positions[5u].xyz - positions[2u].xyz));
    var n2 = normalize(cross(positions[0u].xyz - positions[3u].xyz,
                             positions[1u].xyz - positions[3u].xyz));
    var n3 = normalize(cross(positions[4u].xyz - positions[7u].xyz,
                             positions[0u].xyz - positions[7u].xyz));
    var n4 = normalize(cross(positions[3u].xyz - positions[7u].xyz,
                             positions[6u].xyz - positions[7u].xyz));
    var n5 = normalize(cross(positions[0u].xyz - positions[5u].xyz,
                             positions[4u].xyz - positions[5u].xyz));

    var normals = array<vec4<f32>, 6> (
        vec4<f32>(n0, 0.0),
        vec4<f32>(n1, 0.0),
        vec4<f32>(n2, 0.0),
        vec4<f32>(n3, 0.0),
        vec4<f32>(n4, 0.0),
        vec4<f32>(n5, 0.0)
    );

    var i: u32 = 0u;

    loop {
        if (i == 12u) { break; }
        output_data[thread_group_counter + i * offset + local_index]  = 
            Triangle(
            	Vertex(
            	    positions[vertex_positions[i*3u]],
            	    normals[(i*3u)/6u]
            	),
            	Vertex(
            	    positions[vertex_positions[i*3u+1u]],
            	    normals[(i*3u)/6u]
            	),
            	Vertex(
            	    positions[vertex_positions[i*3u+2u]],
            	    normals[(i*3u)/6u]
            	)
        );

        i = i + 1u;
    }

    let from_origo_x_top_arr = vec3<f32>(1.0, 3.0 * arr.size, 3.0 * arr.size);

    let aabb_top = AABB(
        vec4<f32>((-0.5) * from_origo_x_top_arr, 0.0),
        vec4<f32>(0.5    * from_origo_x_top_arr, 0.0)
    );

    let delta_head = aabb_top.max - aabb_top.min;
    
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

    let hf = 0.45;

    let p0 = aabb_top.min.xyz;
    let p1 = aabb_top.min.xyz + vec3<f32>(delta_head.x , delta_head.y * hf , delta_head.z * hf);
    let p2 = aabb_top.min.xyz + vec3<f32>(delta_head.x , delta_head.y * (1.0 - hf) , delta_head.z * hf);
    let p3 = aabb_top.min.xyz + vec3<f32>(0.0     , delta_head.y , 0.0);
    let p4 = aabb_top.min.xyz + vec3<f32>(0.0     , 0.0     , delta_head.z);
    let p5 = aabb_top.min.xyz + vec3<f32>(delta_head.x , delta_head.y * hf , delta_head.z * (1.0 - hf));
    let p6 = aabb_top.min.xyz + vec3<f32>(delta_head.x , delta_head.y * (1.0 - hf) , delta_head.z * (1.0 - hf));
    let p7 = aabb_top.min.xyz + vec3<f32>(0.0     , delta_head.y , delta_head.z);

    let the_pos_head = arr.start_pos.xyz;

    positions = array<vec4<f32>, 8>(
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p0) - direction * 0.5, c),
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p1) - direction * 0.5, c),
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p2) - direction * 0.5, c),
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p3) - direction * 0.5, c),
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p4) - direction * 0.5, c),
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p5) - direction * 0.5, c),
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p6) - direction * 0.5, c),
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p7) - direction * 0.5, c)
    );

    n0 = normalize(cross(positions[5u].xyz - positions[6u].xyz,
                         positions[4u].xyz - positions[6u].xyz));
    n1 = normalize(cross(positions[1u].xyz - positions[2u].xyz,
                         positions[5u].xyz - positions[2u].xyz));
    n2 = normalize(cross(positions[0u].xyz - positions[3u].xyz,
                         positions[1u].xyz - positions[3u].xyz));
    n3 = normalize(cross(positions[4u].xyz - positions[7u].xyz,
                         positions[0u].xyz - positions[7u].xyz));
    n4 = normalize(cross(positions[3u].xyz - positions[7u].xyz,
                         positions[6u].xyz - positions[7u].xyz));
    n5 = normalize(cross(positions[0u].xyz - positions[5u].xyz,
                         positions[4u].xyz - positions[5u].xyz));

    normals = array<vec4<f32>, 6> (
    	vec4<f32>(n0, 0.0),
    	vec4<f32>(n1,  0.0),
    	vec4<f32>(n2,  0.0),
    	vec4<f32>(n3, 0.0),
    	vec4<f32>(n4,  0.0),
    	vec4<f32>(n5, 0.0)
    );

    var j: u32 = 0u;

    let stride = offset * 12u;

    loop {
        if (j == 12u) { break; }
        output_data[thread_group_counter + offset * 12u + j * offset + local_index]  = 
            Triangle(
            	Vertex(
            	    positions[vertex_positions[j*3u]],
            	    normals[(j*3u)/6u]
            	),
            	Vertex(
            	    positions[vertex_positions[j*3u+1u]],
            	    normals[(j*3u)/6u]
            	),
            	Vertex(
            	    positions[vertex_positions[j*3u+2u]],
            	    normals[(j*3u)/6u]
            	)
        );

        j = j + 1u;
    }
}

@compute
@workgroup_size(256,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32) {

    let actual_index = global_id.x + counter[1]; 

    if (actual_index >= visualization_params.iterator_end_index - 1u) { return; }
    // if (counter[1] >= visualization_params.iterator_end_index) || counter[1] +  { return; }
    // if (actual_index < visualization_params.iterator_start_index) { return; }

    var curve_coord   = vec3<f32>(0.0);
    var curve_coord_n = vec3<f32>(0.0);

    if (visualization_params.curve_number == 1u) {

        let h1 =   from_hilbert_index(actual_index, 4u);
        let h2 = from_hilbert_index(actual_index + 1u, 4u);

        curve_coord   = vec3<f32>(f32(h1.x), f32(h1.y), f32(h1.z));
        curve_coord_n = vec3<f32>(f32(h2.x), f32(h2.y), f32(h2.z));
    }

    if (visualization_params.curve_number == 2u) {

        let r1 = r3_reverse(f32(actual_index));
        let r2 = r3_reverse(f32(actual_index + 1u));

        curve_coord   = vec3<f32>(f32(r1.x), f32(r1.y), f32(r1.z));
        curve_coord_n = vec3<f32>(f32(r2.x), f32(r2.y), f32(r2.z));
    }

    if (visualization_params.curve_number == 3u) {

        let i1 = index_to_uvec3(actual_index,      16u, 16u);
        let i2 = index_to_uvec3(actual_index + 1u, 16u, 16u);

        curve_coord   = vec3<f32>(f32(i1.x), f32(i1.y), f32(i1.z));
        curve_coord_n = vec3<f32>(f32(i2.x), f32(i2.y), f32(i2.z));
    }

    if (visualization_params.curve_number == 4u) {

        let m1 = decode3Dmorton32(actual_index);
        let m2 = decode3Dmorton32(actual_index + 1u);

        curve_coord   = vec3<f32>(f32(m1.x), f32(m1.y), f32(m1.z));
        curve_coord_n = vec3<f32>(f32(m2.x), f32(m2.y), f32(m2.z));
    }

    let c = mapRange(0.0, 
                     4096.0,
        	     0.0,
        	     255.0, 
        	     f32(actual_index)
    );

    let work_group_start_index = work_group_id.x * THREAD_COUNT;
    // if (work_group_start_index >= visualization_params.iterator_end_index) { return; }
    // if (work_group_start_index + THREAD_COUNT < visualization_params.iterator_end_index) { return; }

    // Check!
    var delta = abs(
                    min(
                        i32(visualization_params.iterator_end_index) - i32(work_group_id.x * THREAD_COUNT), i32(THREAD_COUNT)
                        // i32(visualization_params.iterator_end_index) - i32(counter[0]), i32(THREAD_COUNT)
                    )
    );

    if (visualization_params.thread_mode == 1u) {
        delta = i32(min(visualization_params.iterator_end_index - counter[1] - 1u, THREAD_COUNT));
    }


    if (local_index == 0u) {
    	thread_group_counter = atomicAdd(&counter[0], 24u * u32(delta));
    }
    workgroupBarrier();

    create_arrow(
        Arrow (
    	    5.0 * vec4<f32>(f32(curve_coord.x), f32(curve_coord.y), f32(curve_coord.z), 0.0),
    	    5.0 * vec4<f32>(f32(curve_coord_n.x), f32(curve_coord_n.y), f32(curve_coord_n.z), 0.0),
    	    //rgba_u32(255u, 0u, 0u, 255u),
    	    rgba_u32(u32(255u- u32(c)), 0u, u32(c), 255u),
    	    visualization_params.arrow_size,
        ),
        u32(delta),
        local_index
    );
}
