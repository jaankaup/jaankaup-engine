struct Camera {
    u_view_proj: mat4x4<f32>,
    pos: vec4<f32>,
};

// struct CharParams{
//     vertices_so_far: atomic<u32>,
//     iterator_end: u32,
//     number_of_threads: u32,
//     draw_index: atomic<u32>, 
//     max_points_per_char: u32,
//     max_number_of_vertices: u32, // The maximum capacity of vexter buffer.
// };

struct CharParams{
    vertices_so_far: atomic<u32>,
    iterator_end: u32, 
    draw_index: u32, 
    max_points_per_char: u32,
    max_number_of_vertices: u32,
    dispatch_indirect_prefix_sum: array<atomic<u32>, 64>, // [u32; 64],
};

struct Char {
    start_pos: vec3<f32>,
    font_size: f32,
    value: vec4<f32>,
    vec_dim_count: u32,
    color: u32,
    decimal_count: u32,
    auxiliary_data: u32,
};

struct DrawIndirect {
    vertex_count: atomic<u32>,
    instance_count: u32,
    base_vertex: u32,
    base_instance: u32,
};

struct DispatchIndirect {
    x: atomic<u32>,
    y: atomic<u32>,
    z: atomic<u32>,
};

struct ModF {
    fract: f32,
    whole: f32,
};

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(0) @binding(1)
var<storage, read_write> char_params: array<CharParams>;

@group(0) @binding(2)
var<storage, read_write> indirect: array<DispatchIndirect>;

@group(0) @binding(3)
var<storage, read_write> draw_indirect: array<DrawIndirect>;

@group(0) @binding(4)
var<storage,read_write> chars_array: array<Char>;

var<workgroup> wg_total_vertice_count: atomic<u32>; // Is atomic necessery?

var<workgroup> wg_char_params: CharParams;

let NUMBER_OF_DRAW_INDIRECTS = 1024u;
let NUMBER_OF_THREADS = 1024u;

// Temporary storage for elements.
var<workgroup> workgroup_chars: array<Char, NUMBER_OF_THREADS>; 

// The number of DispatchIndirects is 1024.
var<workgroup> wg_indirect_dispatchs: array<DispatchIndirect, 1024>; 

let DUMMY_CHAR = Char(
    vec3<f32>(0.0, 0.0, 0.0),
    0.0,
    vec4<f32>(0.0, 0.0, 0.0, 0.0),
    0u,
    0u,
    0u,
    0xfc00000u, // draw index 64.
);

fn get_points_per_char(aux_data: u32) -> u32 {
    return aux_data & 0x3FFFu;
}

fn get_number_of_chars(aux_data: u32) -> u32 {
    return (aux_data & 0x3fc000u) >> 14u;
}

fn get_draw_index(aux_data: u32) -> u32 {
    return (aux_data & 0xffc00000u) >> 22u;
}

fn set_points_per_char(v: u32, ch: ptr<function, Char>) {
    (*ch).auxiliary_data = ((*ch).auxiliary_data & (!0x3FFFu)) | v;
}

fn set_number_of_chars(v: u32, ch: ptr<function, Char>) {
    (*ch).auxiliary_data = ((*ch).auxiliary_data & (!0x3fc000u)) | (v << 14u);
}

fn set_draw_index(v: u32, ch: ptr<function, Char>) {
    (*ch).auxiliary_data = ((*ch).auxiliary_data & (!0xffc00000u)) | (v << 22u);
}

fn udiv_up_safe32(x: u32, y: u32) -> u32 {
    let tmp = (x + y - 1u) / y;
    return select(tmp, 0u, y == 0u); 
}

fn myTruncate(f: f32) -> f32 {
    return select(f32( i32( floor(f) ) ), f32( i32( ceil(f) ) ), f < 0.0); 
}

fn my_modf(f: f32) -> ModF {
    let iptr = trunc(f);
    let fptr = f - iptr;
    return ModF (
        select(fptr, (-1.0)*fptr, f < 0.0),
        iptr
    );
}

/// if the given integer is < 0, return 0. Otherwise 1 is returned.
fn the_sign_of_i32(n: i32) -> u32 {
    return (u32(n) >> 31u);
    //return 1u ^ (u32(n) >> 31u);
}

fn abs_i32(n: i32) -> u32 {
    let mask = u32(n) >> 31u;
    return (u32(n) + mask) ^ mask;
}

fn log2_u32(n: u32) -> u32 {

    var v = n;
    var r = (u32((v > 0xFFFFu))) << 4u;
    var shift = 0u;

    v  = v >> r;
    shift = (u32((v > 0xFFu))) << 3u;

    v = v >> shift;
    r = r | shift;

    shift = (u32(v > 0xFu)) << 2u;
    v = v >> shift;
    r = r | shift;

    shift = (u32(v > 0x3u)) << 1u;
    v = v >> shift;
    r = r | shift;
    r = r | (v >> 1u);
    return r;
}

var<private> PowersOf10: array<u32, 10> = array<u32, 10>(1u, 10u, 100u, 1000u, 10000u, 100000u, 1000000u, 10000000u, 100000000u, 1000000000u);

// NOT defined if u == 0u.
fn log10_u32(n: u32) -> u32 {
    
    var v = n;
    var r: u32;
    var t: u32;
    
    t = (log2_u32(v) + 1u) * 1233u >> 12u;
    r = t - u32(v < PowersOf10[t]);
    return r;
}

fn number_of_chars_i32(n: i32) -> u32 {

    if (n == 0) { return 1u; }
    return the_sign_of_i32(n) + log10_u32(u32(abs(n))) + 1u;
} 


fn number_of_chars_f32(f: f32, number_of_decimals: u32) -> u32 {

    let m = my_modf(f);
    let minus = select(0u, 1u, m.fract < 0.0 && i32(m.whole) == 0); // New
    return number_of_chars_i32(i32(m.whole)) + number_of_decimals + 1u + minus;
}

//++ fn number_of_chars_f32(f: f32, number_of_decimals: u32) -> u32 {
//++ 
//++     let m = my_modf(f);
//++     return number_of_chars_i32(i32(m.whole)) + number_of_decimals + 1u;
//++ }


// dots, brachets
// vec_dim_count == 0   _  not used
// vec_dim_count == 1   a
// vec_dim_count == 2   (a , b)
// vec_dim_count == 3   (a , b , c)
// vec_dim_count == 4   (a , b , c , d)
// vec_dim_count == 5   undefined
//var<private> NumberOfExtraChars: array<u32, 5> = array<u32, 5>(0u, 0u, 3u, 4u, 5u);
var<private> NumberOfExtraChars: array<u32, 5> = array<u32, 5>(0u, 0u, 2u, 2u, 2u);

// TODO: Instead of data, Element as reference.
fn number_of_chars_data(data: vec4<f32>, vec_dim_count: u32, number_of_decimals: u32) -> u32 {
    
    // Calculate all possible char counts to avoid branches.
    let a = number_of_chars_f32(data.x, number_of_decimals) * u32(vec_dim_count >= 1u);
    let b = number_of_chars_f32(data.y, number_of_decimals) * u32(vec_dim_count >= 2u);
    let c = number_of_chars_f32(data.z, number_of_decimals) * u32(vec_dim_count >= 3u);
    let d = number_of_chars_f32(data.w, number_of_decimals) * u32(vec_dim_count >= 4u);

    return a + b + c + d + NumberOfExtraChars[vec_dim_count]; 
}

// Bitonic sort.
fn bitonic(thread_index: u32) {

  for (var k: u32 = 2u; k <= NUMBER_OF_THREADS; k = k << 1u) {
  for (var j: u32 = k >> 1u ; j > 0u; j = j >> 1u) {
    workgroupBarrier();

    let index = thread_index; 
    let ixj = index ^ j;
    let a = workgroup_chars[index];
    let b = workgroup_chars[ixj];
    let draw_index_a = get_draw_index(a.auxiliary_data);
    let draw_index_b = get_draw_index(b.auxiliary_data);

    if (ixj > index && (((index & k) == 0u && draw_index_a > draw_index_b) || ((index & k) != 0u && draw_index_a < draw_index_b)) ) {
            workgroup_chars[index] = b;
            workgroup_chars[ixj] = a;
    }
  }};
}


@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    // Initialize workgroup attributes.
    if (local_id.x == 0u) {
        wg_char_params = char_params[0];
        // wg_total_vertice_count = 0u;
    }
    workgroupBarrier();

    // Initialize the workgroup DrawIndirect structs.
    if (local_index < NUMBER_OF_DRAW_INDIRECTS) {
        // wg_indirect_dispatchs[local_index] = DispatchIndirect(0u, 1u, 1u);
        wg_indirect_dispatchs[local_index].x = 0u;
        wg_indirect_dispatchs[local_index].y = 1u;
        wg_indirect_dispatchs[local_index].z = 1u;
        draw_indirect[local_index].vertex_count = 0u;
    }
    workgroupBarrier();

    //let number_of_chunks = udiv_up_safe32(NUMBER_OF_THREADS, wg_char_params.iterator_end);
    let number_of_chunks = udiv_up_safe32(wg_char_params.iterator_end, NUMBER_OF_THREADS);

    for (var i: u32 = 0u; i < number_of_chunks; i = i + 1u) {
        workgroupBarrier();

        let global_index = local_index + i * NUMBER_OF_THREADS;
        
        if (global_index < wg_char_params.iterator_end) {

            // Load element. 
            var ch = chars_array[global_index];

            // Calculate the distance between camera and element
            // let dist = min(max(1.0, distance(camera.pos, ch.start_pos.xyz)), 255.0);
            let dist = max(1.0, distance(camera.pos.xyz, ch.start_pos.xyz));

            // Calculate the vertex count per char.
            let vertex_count_per_char = min(u32(f32(wg_char_params.max_points_per_char) / f32(pow(dist, 1.2))), char_params[0].max_points_per_char);
            // let vertex_count_per_char = min(u32(f32(wg_char_params.max_points_per_char) / dist), char_params[0].max_points_per_char);
            set_points_per_char(vertex_count_per_char, &ch);

            // Calculate the total vertex count. 
            let total_number_of_chars = number_of_chars_data(ch.value, ch.vec_dim_count, ch.decimal_count);
            set_number_of_chars(total_number_of_chars, &ch);

            // Update the total vertice count.
            let vertices_so_far = atomicAdd(&(wg_char_params.vertices_so_far), total_number_of_chars * vertex_count_per_char);

            // Determine in which indirect draw group this element belongs to. 
            //let draw_index = (vertices_so_far + total_number_of_chars * vertex_count_per_char) / wg_char_params.max_number_of_vertices;
            let draw_index = vertices_so_far / wg_char_params.max_number_of_vertices;
            set_draw_index(draw_index, &ch); 

            // Update indirect dispatch x.
            atomicAdd(&wg_indirect_dispatchs[draw_index].x, 1u);

            // Copy data to workgroup memory.
            workgroup_chars[local_index] = ch;
        }

        // Add some dummy padding values. The is necessery because we use bitonic sort. Avoid buffer overflow.
        // Prefix scan could be a better choice.
        if (global_index >= wg_char_params.iterator_end) {
             workgroup_chars[local_index] = DUMMY_CHAR;
        }
        // if (global_index >= wg_char_params.iterator_end) {
        //     workgroup_chars[local_index] = 
        //         Char(
        //            vec3<f32>(5.0, 0.0, 0.0),
        //            0.0,
        //            vec4<f32>(4.0, 0.0, 0.0, 0.0),
        //            1u,
        //            1u,
        //            0xfc00000u, // draw index 64.
        //            0u
        //         );
        // }

        // Sort data by draw index.
        bitonic(local_index);
        workgroupBarrier();
 
        if (global_index < wg_char_params.iterator_end) {

            // Scatter the data back to global memory.
            chars_array[global_index] = workgroup_chars[local_index];
        }
    }

    // Scatter indirect results.
    if (local_index < NUMBER_OF_DRAW_INDIRECTS) {
        indirect[local_index] = wg_indirect_dispatchs[local_index];
    }

    // Finally update the final output parameters.
    if (local_index == 0u) {
        // TODO: can this overflow? Implement prefix scan.
        wg_char_params.draw_index = wg_char_params.vertices_so_far / wg_char_params.max_number_of_vertices;
        char_params[0] = wg_char_params;
    }
}
