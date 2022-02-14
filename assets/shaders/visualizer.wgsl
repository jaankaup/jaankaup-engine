// https://glmatrix.net/docs/quat.js.html

struct AABB {
    min: vec4<f32>; 
    max: vec4<f32>; 
};

struct Quaternion {
    w: f32;
    x: f32;
    y: f32;
    z: f32;
};

struct VisualizationParams{
    max_vertex_capacity: u32;
    iterator_start_index: u32;
    iterator_end_index: u32;
    arrow_size: f32;
};

struct Vertex {
    v: vec4<f32>;
    n: vec4<f32>;
};

type Hexaedra = array<Vertex , 8>;

struct Arrow {
    start_pos: vec4<f32>;
    end_pos:   vec4<f32>;
    color: u32;
    size:  f32;
};

let STRIDE: u32 = 64u;
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
    
var<workgroup> temp_vertex_data: array<Vertex, 2304>; // 36 * 256 

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
var<storage,read_write> output: array<Vertex>;

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

fn rotation_from_to(a: vec3<f32>, b: vec3<f32>) -> Quaternion {
    // let a_norm = normalize(a);    
    // let b_norm = normalize(b);    

    let v = cross(a,b);  
    let a_len = length(a);
    let b_len = length(b);
    let w = sqrt(a_len * a_len * b_len * b_len) + dot(a,b);
     
    return quaternion_normalize(Quaternion(w, v.x, v.y, v.z));
}

// fn from_euler_angles(x: f32, y: f32, z: 32) -> Quaternion {
// 
// 
// }

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

fn store_hexaedron(positions: ptr<function, array<vec4<f32>, 8>>,
                   normals:   ptr<function, array<vec4<f32>, 6>>,
                   offset: u32) {

    let index = atomicAdd(&counter[0], 36u);

    var i: u32 = 0u;

    loop {
        if (i == 36u) { break; }
        temp_vertex_data[offset + STRIDE * i]  =
            Vertex(
                (*positions)[vertex_positions[i]],
                (*normals)[i/6u]
            );

        i = i + 1u;
    }

    workgroupBarrier();

    var i: u32 = 0u;

    loop {
        if (i == 36u) { break; }
    	output[index+i] = temp_vertex_data[offset + STRIDE * i];
        i = i + 1u;
    }

    workgroupBarrier();
} 

fn create_aabb(aabb: AABB, offset: u32, color: u32) {

    // Global start position.
    let index = atomicAdd(&counter[0], 36u);

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

    let c = f32(color);

    let delta = aabb.max - aabb.min;

    // var positions = array<vec4<f32>, 8>(
    // 	vec4<f32>(original_pos + rotate_vector(q, aabb.min.xyz), c),
    // 	vec4<f32>(original_pos + rotate_vector(q, aabb.min.xyz + vec3<f32>(delta.x , 0.0     , 0.0)), c),
    // 	vec4<f32>(original_pos + rotate_vector(q, aabb.min.xyz + vec3<f32>(delta.x , delta.y , 0.0)), c),
    // 	vec4<f32>(original_pos + rotate_vector(q, aabb.min.xyz + vec3<f32>(0.0     , delta.y , 0.0)), c),
    // 	vec4<f32>(original_pos + rotate_vector(q, aabb.min.xyz + vec3<f32>(0.0     , 0.0     , delta.z)), c),
    // 	vec4<f32>(original_pos + rotate_vector(q, aabb.min.xyz + vec3<f32>(delta.x , 0.0     , delta.z)), c),
    // 	vec4<f32>(original_pos + rotate_vector(q, aabb.min.xyz + vec3<f32>(delta.x , delta.y , delta.z)), c),
    // 	vec4<f32>(original_pos + rotate_vector(q, aabb.min.xyz + vec3<f32>(0.0     , delta.y , delta.z)), c)
    // );

    var positions = array<vec4<f32>, 8>(
    	vec4<f32>(aabb.min.xyz, c),
    	vec4<f32>(aabb.min.xyz, c) + vec4<f32>(delta.x , 0.0     , 0.0, 0.0),
    	vec4<f32>(aabb.min.xyz, c) + vec4<f32>(delta.x , delta.y , 0.0, 0.0),
    	vec4<f32>(aabb.min.xyz, c) + vec4<f32>(0.0     , delta.y , 0.0, 0.0),
    	vec4<f32>(aabb.min.xyz, c) + vec4<f32>(0.0     , 0.0     , delta.z, 0.0),
    	vec4<f32>(aabb.min.xyz, c) + vec4<f32>(delta.x , 0.0     , delta.z, 0.0),
    	vec4<f32>(aabb.min.xyz, c) + vec4<f32>(delta.x , delta.y , delta.z, 0.0),
    	vec4<f32>(aabb.min.xyz, c) + vec4<f32>(0.0     , delta.y , delta.z, 0.0)
    );
    
    var i: u32 = 0u;

    loop {
        if (i == 36u) { break; }
        temp_vertex_data[offset + STRIDE * i]  = Vertex(
                                                    positions[vertex_positions[i]],
                                                    vec4<f32>(aabb_normals[i/6u])
        );
        i = i + 1u;
    }

    workgroupBarrier();

    var i: u32 = 0u;

    loop {
        if (i == 36u) { break; }
    	output[index+i] = temp_vertex_data[offset + STRIDE * i];
        i = i + 1u;
    }

    workgroupBarrier();
}

fn create_arrow(arr: Arrow, offset: u32) {

    var direction = arr.end_pos.xyz - arr.start_pos.xyz;
    let array_length = length(direction);
    direction = normalize(direction);

    let head_size = min(array_length, 2.0 * arr.size);

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

    let delta = aabb.max - aabb.min;

    let q = rotation_from_to(vec3<f32>(1.0, 0.0, 0.0), direction);
    let the_pos = arr.start_pos.xyz + rotate_vector(q, vec3<f32>(1.0, 0.0, 0.0)) * 0.5 * array_length * 1.0;

    let c = f32(arr.color);

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

    // var normals = array<vec4<f32>, 6> (
    //     vec4<f32>(normalize(rotate_vector(q, vec3<f32>(0.0, 0.0, -1.0))), 0.0),
    // 	vec4<f32>(normalize(rotate_vector(q, vec3<f32>(0.0, 0.0,  1.0))),  0.0),
    // 	vec4<f32>(normalize(rotate_vector(q, vec3<f32>(1.0, 0.0, 0.0))),  0.0),
    // 	vec4<f32>(normalize(rotate_vector(q, vec3<f32>(-1.0, 0.0, 0.0))), 0.0),
    // 	vec4<f32>(normalize(rotate_vector(q, vec3<f32>(0.0, 1.0, 0.0))),  0.0),
    // 	vec4<f32>(normalize(rotate_vector(q, vec3<f32>(0.0, -1.0, 0.0))), 0.0)
    // );

    store_hexaedron(&positions, &normals, offset);

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

    var positions_head = array<vec4<f32>, 8>(
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p0) - direction * 0.5, c),
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p1) - direction * 0.5, c),
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p2) - direction * 0.5, c),
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p3) - direction * 0.5, c),
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p4) - direction * 0.5, c),
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p5) - direction * 0.5, c),
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p6) - direction * 0.5, c),
        vec4<f32>(arr.end_pos.xyz + rotate_vector(q, p7) - direction * 0.5, c)
    );

    // var positions_head = array<vec4<f32>, 8>(
    //     vec4<f32>(arr.end_pos.xyz - direction + rotate_vector(q, p0), c),
    //     vec4<f32>(arr.end_pos.xyz - direction + rotate_vector(q, p1), c),
    //     vec4<f32>(arr.end_pos.xyz - direction + rotate_vector(q, p2), c),
    //     vec4<f32>(arr.end_pos.xyz - direction + rotate_vector(q, p3), c),
    //     vec4<f32>(arr.end_pos.xyz - direction + rotate_vector(q, p4), c),
    //     vec4<f32>(arr.end_pos.xyz - direction + rotate_vector(q, p5), c),
    //     vec4<f32>(arr.end_pos.xyz - direction + rotate_vector(q, p6), c),
    //     vec4<f32>(arr.end_pos.xyz - direction + rotate_vector(q, p7), c)
    // );

    // 4u, 5u, 6u,
    // 5u, 1u, 2u,
    // 1u, 0u, 3u,
    // 0u, 4u, 7u,
    // 6u, 3u, 7u,
    // 4u, 0u, 5u,

    // 6u, 5u, 4u,
    // 2u, 1u, 5u,
    // 3u, 0u, 1u,
    // 7u, 4u, 0u,
    // 7u, 3u, 6u,
    // 5u, 0u, 4u,

    // let n0 = normalize(cross(positions_head[5u].xyz - positions_head[4u].xyz,
    //                          positions_head[6u].xyz - positions_head[4u].xyz));
    // let n1 = normalize(cross(positions_head[1u].xyz - positions_head[5u].xyz,
    //                          positions_head[2u].xyz - positions_head[5u].xyz));
    // let n2 = normalize(cross(positions_head[0u].xyz - positions_head[1u].xyz,
    //                          positions_head[3u].xyz - positions_head[1u].xyz));
    // let n3 = normalize(cross(positions_head[4u].xyz - positions_head[0u].xyz,
    //                          positions_head[7u].xyz - positions_head[0u].xyz));
    // let n4 = normalize(cross(positions_head[3u].xyz - positions_head[6u].xyz,
    //                          positions_head[7u].xyz - positions_head[6u].xyz));
    // let n5 = normalize(cross(positions_head[0u].xyz - positions_head[4u].xyz,
    //                          positions_head[5u].xyz - positions_head[4u].xyz));

    n0 = normalize(cross(positions_head[5u].xyz - positions_head[6u].xyz,
                         positions_head[4u].xyz - positions_head[6u].xyz));
    n1 = normalize(cross(positions_head[1u].xyz - positions_head[2u].xyz,
                         positions_head[5u].xyz - positions_head[2u].xyz));
    n2 = normalize(cross(positions_head[0u].xyz - positions_head[3u].xyz,
                         positions_head[1u].xyz - positions_head[3u].xyz));
    n3 = normalize(cross(positions_head[4u].xyz - positions_head[7u].xyz,
                         positions_head[0u].xyz - positions_head[7u].xyz));
    n4 = normalize(cross(positions_head[3u].xyz - positions_head[7u].xyz,
                         positions_head[6u].xyz - positions_head[7u].xyz));
    n5 = normalize(cross(positions_head[0u].xyz - positions_head[5u].xyz,
                         positions_head[4u].xyz - positions_head[5u].xyz));
    // TODO: recalculate.
    var normals_head = array<vec4<f32>, 6> (
    	vec4<f32>(n0, 0.0),
    	vec4<f32>(n1,  0.0),
    	vec4<f32>(n2,  0.0),
    	vec4<f32>(n3, 0.0),
    	vec4<f32>(n4,  0.0),
    	vec4<f32>(n5, 0.0)
    );

    // var normals_head = array<vec4<f32>, 6> (
    // 	vec4<f32>(normalize(rotate_vector(q, n0)), 0.0),
    // 	vec4<f32>(normalize(rotate_vector(q, n1)),  0.0),
    // 	vec4<f32>(normalize(rotate_vector(q, n2)),  0.0),
    // 	vec4<f32>(normalize(rotate_vector(q, n3)), 0.0),
    // 	vec4<f32>(normalize(rotate_vector(q, n4)),  0.0),
    // 	vec4<f32>(normalize(rotate_vector(q, n5)), 0.0)
    // );

    // var normals_head = array<vec4<f32>, 6> (
    // 	vec4<f32>(rotate_vector(q, vec3<f32>(0.0, 0.0, -1.0)), 0.0),
    // 	vec4<f32>(rotate_vector(q, vec3<f32>(0.0, 0.0, 1.0)),  0.0),
    // 	vec4<f32>(rotate_vector(q, vec3<f32>(1.0, 0.0, 0.0)),  0.0),
    // 	vec4<f32>(rotate_vector(q, vec3<f32>(-1.0, 0.0, 0.0)), 0.0),
    // 	vec4<f32>(rotate_vector(q, vec3<f32>(0.0, 1.0, 0.0)),  0.0),
    // 	vec4<f32>(rotate_vector(q, vec3<f32>(0.0, -1.0, 0.0)), 0.0)
    // );


    store_hexaedron(&positions_head, &normals_head, offset);

    // create_aabb(aabb_top,
    //             64u,
    //             arr.color,
    //             q_top3,
    //             arr.end_pos.xyz - rotate_vector(q_top3, vec3<f32>(1.0, 0.0, 0.0)) * 0.5 * 1.1 );
}

@stage(compute)
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    // struct VisualizationParams{
    //     max_vertex_capacity: u32;
    //     iterator_start_index: u32;
    //     iterator_end_index: u32;
    // };

    let actual_index = global_id.x + counter[1]; 

    if (actual_index >= visualization_params.iterator_end_index - 1u) { return; }
    // if (global_id.x >= 5u) { return; }

    let curve_coord =   index_to_uvec3(actual_index,      32u, 32u);
    let curve_coord_n = index_to_uvec3(actual_index + 1u, 32u, 32u);

    let c = mapRange(0.0, 
                     f32(visualization_params.iterator_end_index),
		     0.0,
		     255.0, 
		     f32(actual_index)
    );

    create_arrow(
        Arrow (
    	    3.0 * vec4<f32>(f32(curve_coord.x), f32(curve_coord.y), f32(curve_coord.z), 0.0),
    	    3.0 * vec4<f32>(f32(curve_coord_n.x), f32(curve_coord_n.y), f32(curve_coord_n.z), 0.0),
    	    rgba_u32(u32(c), 0u, 255u - u32(c), 255u),
    	    visualization_params.arrow_size
        ),
        local_id.x,
    );
    
    // if (global_id.x > 0u) { return; }

    // let arr = arrows[global_id.x];

    // var direction = arr.end_pos - arr.start_pos;
    // let array_length = length(direction);
    // direction = normalize(direction);

    // create_arrow(arr);

    // create_aabb(
    //     AABB(
    //         arr.start_pos - vec4<f32>(arr.size * 1.1, arr.size * 1.1, arr.size * 1.1, 0.0),
    //         arr.start_pos + vec4<f32>(arr.size * 1.1, arr.size * 1.1, arr.size * 1.1, 0.0)
    //     ),
    //     64u,
    //     rgba_u32(255u, 0u, 0u, 255u)
    // );
    // create_aabb(
    //     AABB(
    //         arr.end_pos - vec4<f32>(arr.size * 1.1, arr.size * 1.1, arr.size * 1.1, 0.0),
    //         arr.end_pos + vec4<f32>(arr.size * 1.1, arr.size * 1.1, arr.size * 1.1, 0.0)
    //     ),
    //     64u,
    //     rgba_u32(0u, 255u, 0u, 255u)
    // );
}
