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

struct ArrowAabbParams{
    max_number_of_vertices: u32,
    iterator_start_index: u32,
    iterator_end_index: u32,
    element_type: u32, // 0 :: arrow, 1 :: aabb, 2 :: aabb wire
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

// A struct for errors.
// vertex_overflow: 0 :: OK, n :: amount of overflow.
struct Errors {
    vertex_buffer_overflow: u32,
};

@group(0) @binding(0)
var<uniform> arrow_aabb_params: ArrowAabbParams;

@group(0) @binding(1)
var<storage, read_write> arrows: array<Arrow>;

@group(0) @binding(2)
var<storage, read_write> aabbs: array<AABB>;

@group(0) @binding(3)
var<storage, read_write> aabb_wires: array<AABB>;

@group(0) @binding(4)
var<storage,read_write> output_data: array<Triangle>;

var<workgroup> thread_group_counter: u32 = 0; 

let THREAD_COUNT: u32 = 64u;
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
    
// var<workgroup> temp_vertex_data: array<Vertex, 2304>; // 36 * 256 
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

// QUATERNION STUFF

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

fn create_aabb(aabb: AABB, offset: u32, local_index: u32, color: f32, start_index: u32) {

    // Global start position.
    //let index = atomicAdd(&counter[0], 36u);

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

    // Encode color information to the fourth component.

    // let c = f32(color);
    let c = color; // bitcast<f32>(color);

    let delta = aabb.max - aabb.min;

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

    var normals: array<vec4<f32>, 6> = array<vec4<f32>, 6>(
        vec4<f32>(0.0, 0.0, -1.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(-1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, -1.0, 0.0, 0.0)
    );

    var i: u32 = 0u;

    loop {
        if (i == 12u) { break; }
        // output_data[thread_group_counter + i * offset + local_index] = 
        output_data[start_index + i * offset + local_index] = 
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
}
 
fn create_aabb_wire(aabb: AABB, t: f32, col: f32, offset: u32, local_index: u32, start_index: u32) {

    var aabbs = array<AABB, 12>(
        AABB(aabb.min, vec4<f32>(aabb.max.x, aabb.min.y + t, aabb.min.z + t, 1.0)),
        AABB(aabb.min, vec4<f32>(aabb.min.x + t, aabb.min.y + t, aabb.max.z, 1.0)),
        AABB(vec4<f32>(aabb.max.x - t, aabb.min.y, aabb.min.z, 1.0), vec4<f32>(aabb.max.x, aabb.min.y + t,  aabb.max.z, 1.0)),
        AABB(vec4<f32>(aabb.min.x, aabb.min.y, aabb.max.z - t, 1.0), vec4<f32>(aabb.max.x, aabb.min.y + t,  aabb.max.z, 1.0)),
        AABB(vec4<f32>(aabb.min.x, aabb.max.y - t, aabb.min.z, 1.0),  vec4<f32>(aabb.max.x, aabb.max.y,     aabb.min.z + t, 1.0)),
        AABB(vec4<f32>(aabb.min.x, aabb.max.y - t, aabb.min.z, 1.0),  vec4<f32>(aabb.min.x + t, aabb.max.y, aabb.max.z, 1.0)),
        AABB(vec4<f32>(aabb.max.x - t, aabb.max.y - t, aabb.min.z, 1.0),  vec4<f32>(aabb.max.x, aabb.max.y, aabb.max.z, 1.0)),
        AABB(vec4<f32>(aabb.min.x, aabb.max.y - t, aabb.max.z - t, 1.0),  vec4<f32>(aabb.max.x, aabb.max.y, aabb.max.z, 1.0)),
        AABB(vec4<f32>(aabb.min.x, aabb.min.y, aabb.min.z, 1.0), vec4<f32>(aabb.min.x + t, aabb.max.y, aabb.min.z + t, 1.0)),
        AABB(vec4<f32>(aabb.max.x - t, aabb.min.y, aabb.min.z, 1.0),  vec4<f32>(aabb.max.x    , aabb.max.y, aabb.min.z + t, 1.0)),
        AABB(vec4<f32>(aabb.min.x,  aabb.min.y, aabb.max.z - t, 1.0), vec4<f32>(aabb.min.x + t, aabb.max.y, aabb.max.z, 1.0)),
        AABB(vec4<f32>(aabb.max.x - t, aabb.min.y, aabb.max.z - t, 1.0),  vec4<f32>(aabb.max.x    , aabb.max.y, aabb.max.z, 1.0))
    );

    var normals: array<vec4<f32>, 6> = array<vec4<f32>, 6>(
        vec4<f32>(0.0, 0.0, -1.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(-1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, -1.0, 0.0, 0.0)
    );

    for (var i: i32 = 0; i<12 ; i = i + 1) {
        
        let delta = aabbs[i].max - aabbs[i].min;
        var positions = array<vec4<f32>, 8>(
        	vec4<f32>(aabbs[i].min.xyz, col),
        	vec4<f32>(aabbs[i].min.xyz, col) + vec4<f32>(delta.x , 0.0     , 0.0, 0.0),
        	vec4<f32>(aabbs[i].min.xyz, col) + vec4<f32>(delta.x , delta.y , 0.0, 0.0),
        	vec4<f32>(aabbs[i].min.xyz, col) + vec4<f32>(0.0     , delta.y , 0.0, 0.0),
        	vec4<f32>(aabbs[i].min.xyz, col) + vec4<f32>(0.0     , 0.0     , delta.z, 0.0),
        	vec4<f32>(aabbs[i].min.xyz, col) + vec4<f32>(delta.x , 0.0     , delta.z, 0.0),
        	vec4<f32>(aabbs[i].min.xyz, col) + vec4<f32>(delta.x , delta.y , delta.z, 0.0),
        	vec4<f32>(aabbs[i].min.xyz, col) + vec4<f32>(0.0     , delta.y , delta.z, 0.0)
        );

    	var j: u32 = 0u;

        //output_data[start_index + i * offset + local_index] = 

    	loop {
    	    if (j == 12u) { break; }
    	    output_data[start_index + j * offset + local_index + u32(i) * offset * 12u ]  = 
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
}

fn create_arrow(arr: Arrow, offset: u32, local_index: u32, start_index: u32) {

    var direction = arr.end_pos.xyz - arr.start_pos.xyz;
    let array_length = length(direction);
    direction = normalize(direction);

    let head_size = min(0.5 * array_length, 2.0 * arr.size);
    // let head_size = min(0.1 * array_length, 0.5 * arr.size);

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
        output_data[start_index + i * offset + local_index]  = 
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
        output_data[start_index + offset * 12u + j * offset + local_index]  = 
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
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    var delta = min(arrow_aabb_params.iterator_end_index - (arrow_aabb_params.iterator_start_index + THREAD_COUNT * work_group_id.x), THREAD_COUNT); 

    let actual_index = arrow_aabb_params.iterator_start_index + global_id.x;

    if (local_index >= delta) { return; } 

    if (arrow_aabb_params.element_type == 0u) { 
        create_arrow(arrows[actual_index], u32(delta), local_index, THREAD_COUNT * work_group_id.x * 24u);
    }
    else if (arrow_aabb_params.element_type == 1u) {
        let aabb = aabbs[actual_index];
        create_aabb(aabb, u32(delta), local_index, aabb.min.w, THREAD_COUNT * work_group_id.x * 12u);
    }
    else if (arrow_aabb_params.element_type == 2u) {
        let aabb = aabb_wires[actual_index];
        create_aabb_wire(aabb, aabb.max.w, aabb.min.w, u32(delta), local_index, THREAD_COUNT * work_group_id.x * 144u);
    }
}
