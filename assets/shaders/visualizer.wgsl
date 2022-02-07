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

// struct VertexBuffer {
//     data: @stride(32) array<Vertex>;
//     //data: [[stride(32)]] array<Vertex>;
// };

@group(0)
@binding(0)
var<uniform> visualization_params: VisualizationParams;

@group(0)
@binding(1)
var<storage, read_write> counter: Counter;

@group(0)
@binding(2)
var<storage,read_write> output: array<Vertex>;

// var<private> cube: Cube;

//let edge_info: array<vec2<i32>, 12> = array<vec2<i32>, 12>(
// TODO: uniform!
// var<private> edge_info: array<vec2<i32>, 12> = array<vec2<i32>, 12>(
//         vec2<i32>(0,1), vec2<i32>(1,2), vec2<i32>(2,3), vec2<i32>(3,0), 
//         vec2<i32>(4,5), vec2<i32>(5,6), vec2<i32>(6,7), vec2<i32>(7,4), 
//         vec2<i32>(0,4), vec2<i32>(1,5), vec2<i32>(2,6), vec2<i32>(3,7)
// ); 


@stage(compute)
@workgroup_size(4,4,4)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    var i: u32 = 0u;

    let index = atomicAdd(&counter.counter, 3u);

    // loop {
    //     if (i == 5u) { break; }

    //     let base_index: u32 = triTable[cube_case * OFFSET + i];

    //     if (base_index != 16777215u) { 


    //         // Create the triangle vertices and normals.
    //         createVertex(i32((base_index & 0xff0000u) >> 16u), i32(index));
    //         createVertex(i32((base_index & 0xff00u) >> 8u)   , i32(index+1u));
    //         createVertex(i32( base_index & 0xffu),            i32(index+2u));
    //     }
    //     i = i + 1u;
    // }
}
