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

// var<private> cube: Cube;

//let edge_info: array<vec2<i32>, 12> = array<vec2<i32>, 12>(
// TODO: uniform!
// var<private> edge_info: array<vec2<i32>, 12> = array<vec2<i32>, 12>(
//         vec2<i32>(0,1), vec2<i32>(1,2), vec2<i32>(2,3), vec2<i32>(3,0), 
//         vec2<i32>(4,5), vec2<i32>(5,6), vec2<i32>(6,7), vec2<i32>(7,4), 
//         vec2<i32>(0,4), vec2<i32>(1,5), vec2<i32>(2,6), vec2<i32>(3,7)
// ); 

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

    // temp_data[local_id.x] = TempAABB(aabb, array<Vertex, 36>);

    // atomicAdd(&temp_aabb_data_index, 36u);

    // Global start position.
    let index = atomicAdd(&counter.counter, 36u);

    let delta: vec3<f32> = aabb.max.xyz - aabb.min.xyz;

    let n_front = vec4<f32>(0.0, 0.0, -1.0, 0.0);
    let n_back  = vec4<f32>(0.0, 0.0, 1.0, 0.0);
    let n_right = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    let n_left = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    let n_top = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    let n_bottom = vec4<f32>(0.0, -1.0, 0.0, 0.0);

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

    let p0 = aabb.min.xyz;
    let p1 = aabb.min.xyz + vec3<f32>(delta.x , 0.0     , 0.0);
    let p2 = aabb.min.xyz + vec3<f32>(delta.x , delta.y , 0.0);
    let p3 = aabb.min.xyz + vec3<f32>(0.0     , delta.y , 0.0);
    let p4 = aabb.min.xyz + vec3<f32>(0.0     , 0.0     , delta.z);
    let p5 = aabb.min.xyz + vec3<f32>(delta.x , 0.0     , delta.z);
    let p6 = aabb.min.xyz + vec3<f32>(delta.x , delta.y , delta.z);
    let p7 = aabb.min.xyz + vec3<f32>(0.0     , delta.y , delta.z);

    // FRONT.

    // output[index]    = Vertex(vec4<f32>(p4, 1.0), n_front);
    // output[index+1u] = Vertex(vec4<f32>(p5, 1.0), n_front);
    // output[index+2u] = Vertex(vec4<f32>(p6, 1.0), n_front);
    // output[index+3u] = Vertex(vec4<f32>(p4, 1.0), n_front);
    // output[index+4u] = Vertex(vec4<f32>(p6, 1.0), n_front);
    // output[index+5u] = Vertex(vec4<f32>(p7, 1.0), n_front);

    temp_vertex_data[offset]    = Vertex(vec4<f32>(p4, 1.0), n_front);
    temp_vertex_data[offset + STRIDE * 1u] = Vertex(vec4<f32>(p5, 1.0), n_front);
    temp_vertex_data[offset + STRIDE * 2u] = Vertex(vec4<f32>(p6, 1.0), n_front);
    temp_vertex_data[offset + STRIDE * 3u] = Vertex(vec4<f32>(p4, 1.0), n_front);
    temp_vertex_data[offset + STRIDE * 4u] = Vertex(vec4<f32>(p6, 1.0), n_front);
    temp_vertex_data[offset + STRIDE * 5u] = Vertex(vec4<f32>(p7, 1.0), n_front);

    //// RIGHT.

    // output[index+6u]  = Vertex(vec4<f32>(p5, 1.0), n_right);
    // output[index+7u]  = Vertex(vec4<f32>(p1, 1.0), n_right);
    // output[index+8u]  = Vertex(vec4<f32>(p2, 1.0), n_right);
    // output[index+9u]  = Vertex(vec4<f32>(p5, 1.0), n_right);
    // output[index+10u] = Vertex(vec4<f32>(p2, 1.0), n_right);
    // output[index+11u] = Vertex(vec4<f32>(p6, 1.0), n_right);

    temp_vertex_data[offset + STRIDE * 6u]  = Vertex(vec4<f32>(p5, 1.0), n_right);
    temp_vertex_data[offset + STRIDE * 7u]  = Vertex(vec4<f32>(p1, 1.0), n_right);
    temp_vertex_data[offset + STRIDE * 8u]  = Vertex(vec4<f32>(p2, 1.0), n_right);
    temp_vertex_data[offset + STRIDE * 9u]  = Vertex(vec4<f32>(p5, 1.0), n_right);
    temp_vertex_data[offset + STRIDE * 10u] = Vertex(vec4<f32>(p2, 1.0), n_right);
    temp_vertex_data[offset + STRIDE * 11u] = Vertex(vec4<f32>(p6, 1.0), n_right);

    // //// BACK.

    // output[index+12u] = Vertex(vec4<f32>(p1, 1.0), n_back);
    // output[index+13u] = Vertex(vec4<f32>(p0, 1.0), n_back);
    // output[index+14u] = Vertex(vec4<f32>(p3, 1.0), n_back);
    // output[index+15u] = Vertex(vec4<f32>(p1, 1.0), n_back);
    // output[index+16u] = Vertex(vec4<f32>(p3, 1.0), n_back);
    // output[index+17u] = Vertex(vec4<f32>(p2, 1.0), n_back);

    temp_vertex_data[offset + STRIDE * 12u] = Vertex(vec4<f32>(p1, 1.0), n_back);
    temp_vertex_data[offset + STRIDE * 13u] = Vertex(vec4<f32>(p0, 1.0), n_back);
    temp_vertex_data[offset + STRIDE * 14u] = Vertex(vec4<f32>(p3, 1.0), n_back);
    temp_vertex_data[offset + STRIDE * 15u] = Vertex(vec4<f32>(p1, 1.0), n_back);
    temp_vertex_data[offset + STRIDE * 16u] = Vertex(vec4<f32>(p3, 1.0), n_back);
    temp_vertex_data[offset + STRIDE * 17u] = Vertex(vec4<f32>(p2, 1.0), n_back);

    // //// LEFT.

    // output[index+18u] = Vertex(vec4<f32>(p0, 1.0), n_left);
    // output[index+19u] = Vertex(vec4<f32>(p4, 1.0), n_left);
    // output[index+20u] = Vertex(vec4<f32>(p7, 1.0), n_left);
    // output[index+21u] = Vertex(vec4<f32>(p0, 1.0), n_left);
    // output[index+22u] = Vertex(vec4<f32>(p7, 1.0), n_left);
    // output[index+23u] = Vertex(vec4<f32>(p3, 1.0), n_left);

    temp_vertex_data[offset + STRIDE * 18u] = Vertex(vec4<f32>(p0, 1.0), n_left);
    temp_vertex_data[offset + STRIDE * 19u] = Vertex(vec4<f32>(p4, 1.0), n_left);
    temp_vertex_data[offset + STRIDE * 20u] = Vertex(vec4<f32>(p7, 1.0), n_left);
    temp_vertex_data[offset + STRIDE * 21u] = Vertex(vec4<f32>(p0, 1.0), n_left);
    temp_vertex_data[offset + STRIDE * 22u] = Vertex(vec4<f32>(p7, 1.0), n_left);
    temp_vertex_data[offset + STRIDE * 23u] = Vertex(vec4<f32>(p3, 1.0), n_left);

    // //// TOP.

    // output[index+24u] = Vertex(vec4<f32>(p6, 1.0), n_top);
    // output[index+25u] = Vertex(vec4<f32>(p3, 1.0), n_top);
    // output[index+26u] = Vertex(vec4<f32>(p7, 1.0), n_top);
    // output[index+27u] = Vertex(vec4<f32>(p6, 1.0), n_top);
    // output[index+28u] = Vertex(vec4<f32>(p2, 1.0), n_top);
    // output[index+29u] = Vertex(vec4<f32>(p3, 1.0), n_top);

    temp_vertex_data[offset + STRIDE * 24u] = Vertex(vec4<f32>(p6, 1.0), n_top);
    temp_vertex_data[offset + STRIDE * 25u] = Vertex(vec4<f32>(p3, 1.0), n_top);
    temp_vertex_data[offset + STRIDE * 26u] = Vertex(vec4<f32>(p7, 1.0), n_top);
    temp_vertex_data[offset + STRIDE * 27u] = Vertex(vec4<f32>(p6, 1.0), n_top);
    temp_vertex_data[offset + STRIDE * 28u] = Vertex(vec4<f32>(p2, 1.0), n_top);
    temp_vertex_data[offset + STRIDE * 29u] = Vertex(vec4<f32>(p3, 1.0), n_top);

    // //// BOTTOM.

    // output[index+30u] = Vertex(vec4<f32>(p4, 1.0), n_bottom);
    // output[index+31u] = Vertex(vec4<f32>(p0, 1.0), n_bottom);
    // output[index+32u] = Vertex(vec4<f32>(p5, 1.0), n_bottom);
    // output[index+33u] = Vertex(vec4<f32>(p0, 1.0), n_bottom);
    // output[index+34u] = Vertex(vec4<f32>(p1, 1.0), n_bottom);
    // output[index+35u] = Vertex(vec4<f32>(p5, 1.0), n_bottom);

    temp_vertex_data[offset + STRIDE * 30u] = Vertex(vec4<f32>(p4, 1.0), n_bottom);
    temp_vertex_data[offset + STRIDE * 31u] = Vertex(vec4<f32>(p0, 1.0), n_bottom);
    temp_vertex_data[offset + STRIDE * 32u] = Vertex(vec4<f32>(p5, 1.0), n_bottom);
    temp_vertex_data[offset + STRIDE * 33u] = Vertex(vec4<f32>(p0, 1.0), n_bottom);
    temp_vertex_data[offset + STRIDE * 34u] = Vertex(vec4<f32>(p1, 1.0), n_bottom);
    temp_vertex_data[offset + STRIDE * 35u] = Vertex(vec4<f32>(p5, 1.0), n_bottom);

    workgroupBarrier();

    output[index]     = temp_vertex_data[offset];
    output[index+1u]  = temp_vertex_data[offset + STRIDE * 1u];
    output[index+2u]  = temp_vertex_data[offset + STRIDE * 2u];
    output[index+3u]  = temp_vertex_data[offset + STRIDE * 3u];
    output[index+4u]  = temp_vertex_data[offset + STRIDE * 4u];
    output[index+5u]  = temp_vertex_data[offset + STRIDE * 5u];
    output[index+6u]  = temp_vertex_data[offset + STRIDE * 6u];
    output[index+7u]  = temp_vertex_data[offset + STRIDE * 7u];
    output[index+8u]  = temp_vertex_data[offset + STRIDE * 8u];
    output[index+9u]  = temp_vertex_data[offset + STRIDE * 9u];
    output[index+10u] = temp_vertex_data[offset + STRIDE * 10u];
    output[index+11u] = temp_vertex_data[offset + STRIDE * 11u];
    output[index+12u] = temp_vertex_data[offset + STRIDE * 12u];
    output[index+13u] = temp_vertex_data[offset + STRIDE * 13u];
    output[index+14u] = temp_vertex_data[offset + STRIDE * 14u];
    output[index+15u] = temp_vertex_data[offset + STRIDE * 15u];
    output[index+16u] = temp_vertex_data[offset + STRIDE * 16u];
    output[index+17u] = temp_vertex_data[offset + STRIDE * 17u];
    output[index+18u] = temp_vertex_data[offset + STRIDE * 18u];
    output[index+19u] = temp_vertex_data[offset + STRIDE * 19u];
    output[index+20u] = temp_vertex_data[offset + STRIDE * 20u];
    output[index+21u] = temp_vertex_data[offset + STRIDE * 21u];
    output[index+22u] = temp_vertex_data[offset + STRIDE * 22u];
    output[index+23u] = temp_vertex_data[offset + STRIDE * 23u];
    output[index+24u] = temp_vertex_data[offset + STRIDE * 24u];
    output[index+25u] = temp_vertex_data[offset + STRIDE * 25u];
    output[index+26u] = temp_vertex_data[offset + STRIDE * 26u];
    output[index+27u] = temp_vertex_data[offset + STRIDE * 27u];
    output[index+28u] = temp_vertex_data[offset + STRIDE * 28u];
    output[index+29u] = temp_vertex_data[offset + STRIDE * 29u];
    output[index+30u] = temp_vertex_data[offset + STRIDE * 30u];
    output[index+31u] = temp_vertex_data[offset + STRIDE * 31u];
    output[index+32u] = temp_vertex_data[offset + STRIDE * 32u];
    output[index+33u] = temp_vertex_data[offset + STRIDE * 33u];
    output[index+34u] = temp_vertex_data[offset + STRIDE * 34u];
    output[index+35u] = temp_vertex_data[offset + STRIDE * 35u];
}

fn create_array(index: u32) {
    
}

@stage(compute)
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    let mapping = index_to_uvec3(global_id.x, 64u, 64u);

    let aabb = AABB(vec4<f32>(2.0 * f32(mapping.x), 
			      2.0 * f32(mapping.y),
			      2.0 * f32(mapping.z),
                              1.0),
                    vec4<f32>(2.0 * f32(mapping.x) + 0.3, 
			      2.0 * f32(mapping.y) + 0.3,
			      2.0 * f32(mapping.z) + 0.3,
                              1.0)
    );

    // workgroupBarrier();

    create_aabb(aabb, local_index, 0u, local_index, 0u);

    //var i: u32 = 0u;
    // loop {
    //     if (i == 5u) { break; }

    //     let base_index: u32 = triTable[cube_case * STRIDE + i];

    //     if (base_index != 16777215u) { 


    //         // Create the triangle vertices and normals.
    //         createVertex(i32((base_index & 0xff0000u) >> 16u), i32(index));
    //         createVertex(i32((base_index & 0xff00u) >> 8u)   , i32(index+1u));
    //         createVertex(i32( base_index & 0xffu),            i32(index+2u));
    //     }
    //     i = i + 1u;
    // }
}
