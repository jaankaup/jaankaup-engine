struct Camera {
    u_view_proj: mat4x4<f32>,
    pos: vec4<f32>,
};

struct CharParams{
    iterator_start: atomic<u32>,
    iterator_end: u32,
    number_of_threads: u32,
    draw_index: atomic<u32>, 
    max_points_per_char: u32,
    max_number_of_vertices: u32, // The maximum capacity of vexter buffer.
};

struct Char {
    start_pos: vec3<f32>, // encode start.pos.w the decimal_count. TODO: something better.
    font_size: f32,
    value: vec4<f32>,
    vec_dim_count: u32, // 1 => f32, 2 => vec3<f32>, 3 => vec3<f32>, 4 => vec4<f32>
    color: u32,
    draw_index: u32,
    point_count: u32,
};

struct DrawIndirect {
    vertex_count: atomic<u32>,
    instance_count: u32,
    base_vertex: u32,
    base_instance: u32,
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
var<storage, read_write> indirect: array<DrawIndirect>;

@group(0) @binding(3)
var<storage,read_write> chars_array: array<Char>;

var<workgroup> wg_total_char_count: atomic<u32>; // Is atomic necessery?
var<workgroup> wg_current_draw_indirect_number: atomic<u32>;

var<workgroup> char_params_wg: CharParams;
var<workgroup> wg_start_index: u32; 

let NUMBER_OF_DRAW_INDIRECTS = 64u;
let NUMBER_OF_THREADS = 1024u;

// Temporary storage for elements.
var<workgroup> workgroup_chars: array<Char, NUMBER_OF_THREADS>; 

// The number of DrawIndirects is 64.
var<workgroup> wg_indirect_draws: array<DrawIndirect, 64>; 

let DUMMY_CHAR = Char(
    vec3<f32>(0.0, 0.0, 0.0),
    0.0,
    vec4<f32>(0.0, 0.0, 0.0, 0.0),
    0u,
    0u,
    5000000u,
    0u,
);

fn udiv_up_safe32(x: u32, y: u32) -> u32 {
    let tmp = (x + y - 1u) / y;
    return select(tmp, 0u, y == 0u); 
}

fn myTruncate(f: f32) -> f32 {
    return select(f32( i32( floor(f) ) ), f32( i32( ceil(f) ) ), f < 0.0); 
}

fn my_modf(f: f32) -> ModF {
    let iptr = myTruncate(f);
    let fptr = f - iptr;
    return ModF (
        select(fptr, (-1.0)*fptr, f < 0.0),
        iptr
        // copysign (isinf( f ) ? 0.0 : f - iptr, f)  
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

// Calculates the number of digits + minus.
fn number_of_chars_i32(n: i32) -> u32 {

    if (n == 0) { return 1u; }
    return the_sign_of_i32(n) + log10_u32(abs_i32(n)) + 1u;
} 

fn number_of_chars_f32(f: f32, number_of_decimals: u32) -> u32 {

    let m = my_modf(f);
    return number_of_chars_i32(i32(m.whole)) + number_of_decimals + 1u;
}

// dots, brachets
// vec_dim_count == 0   _  not used
// vec_dim_count == 1   a
// vec_dim_count == 2   (a , b)
// vec_dim_count == 3   (a , b , c)
// vec_dim_count == 4   (a , b , c , d)
// vec_dim_count == 5   undefined
var<private> NumberOfExtraChars: array<u32, 5> = array<u32, 5>(0u, 0u, 3u, 4u, 5u);

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
    if (ixj > index && (((index & k) == 0u && a.draw_index > b.draw_index) || ((index & k) != 0u && a.draw_index < b.draw_index)) ) {
            workgroup_chars[index] = b;
            workgroup_chars[ixj] = a;
    }
  }};
}


@stage(compute)
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    // Initialize workgroup attributes.
    if (local_id.x == 0u) {
        wg_start_index = 0u;
        wg_total_char_count = 0u;
        wg_current_draw_indirect_number = 0u;
    }
    workgroupBarrier();

    // Initialize the workgroup DrawIndirect structs.
    if (local_index < NUMBER_OF_DRAW_INDIRECTS) {
        atomicAnd(&(wg_indirect_draws[local_index].vertex_count), 0u);
        wg_indirect_draws[local_index].instance_count = 1u;
        wg_indirect_draws[local_index].base_vertex = 0u;
        wg_indirect_draws[local_index].base_instance = 0u;
    }
    workgroupBarrier();

    let number_of_chunks = udiv_up_safe32(NUMBER_OF_THREADS, char_params[0].iterator_end);

    for (var i: u32 = 0u; i < number_of_chunks ; i = i + 1u) {
        workgroupBarrier();

        let global_index = local_index + i * NUMBER_OF_THREADS;
        
        // Add some dummy padding values. Avoid buffer overflow.
        if (global_index >= char_params[0].iterator_end) {
             workgroup_chars[local_index] = DUMMY_CHAR;
             continue;
        }

        //++ total_char_count = 0u;

        // Load element. 
        var ch = chars_array[global_index];

        // Calculate the distance between camera and element
        let dist = min(max(1.0, distance(camera.pos.xyz, ch.start_pos.xyz)), 255.0);

        // Calculate the vertex count per char.
        let vertex_count_per_char = min(u32(f32(char_params[0].max_points_per_char) / f32(dist)), char_params[0].max_points_per_char);

        // Calculate the total vertex count. 
        let total_number_of_vertices = number_of_chars_data(ch.value, ch.vec_dim_count, 2u); // 2u, number of cedimals.

        // Update the total vertice count.
        let vertices_so_far = atomicAdd(&wg_total_char_count, total_number_of_vertices * vertex_count_per_char);

        ch.point_count = total_number_of_vertices * vertex_count_per_char;
        ch.draw_index = vertices_so_far / 7000u; // char_params[0].max_number_of_vertices;

        workgroup_chars[local_index] = ch;

        bitonic(local_index);
 
        chars_array[global_index] = workgroup_chars[local_index]; //ch;

        // workgroup_chars[local_index] = ch;
    }

    //++ // Determine the draw start offset and the number of vertices for drawing the element.
}
