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
    max_number_of_vertices: u32,
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

var<private> total_char_count: u32;

var<workgroup> char_params_wg: CharParams;

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
// vec_dim_count == 1   a
// vec_dim_count == 2   (a , b)
// vec_dim_count == 3   (a , b , c)
// vec_dim_count == 4   (a , b , c , d)
var<private> NumberOfExtraChars: array<u32, 4> = array<u32, 4>(0u, 3u, 4u, 5u);

fn number_of_chars_data(data: ptr<function, vec4<f32>>, vec_dim_count: u32, number_of_decimals: u32) -> u32 {
    
    // Calculate all possible char counts to avoid branches.
    let a = number_of_chars_f32((*data).x, number_of_decimals) * u32(vec_dim_count >= 1u);
    let b = number_of_chars_f32((*data).y, number_of_decimals) * u32(vec_dim_count >= 2u);
    let c = number_of_chars_f32((*data).z, number_of_decimals) * u32(vec_dim_count >= 3u);
    let d = number_of_chars_f32((*data).w, number_of_decimals) * u32(vec_dim_count >= 4u);

    return a + b + c + d + NumberOfExtraChars[vec_dim_count]; 
}

// @param n : number that should be rendered.
// @ignore_first : ignore the first number. This should be false when rendering u32. 
// @total_vertex_count : total number of vertices per cha for rendering. 
fn log_number(n: i32, ignore_first: bool, total_vertex_count: u32) {

    var found = false;
    var ignore = ignore_first;
    var temp_n = abs(n);

    //++ // Only one char are produced.
    //++ if (n == 0) {

    //++     create_char(0u, thread_index, total_vertex_count, *base_pos, col, font_size);
    //++     *base_pos = *base_pos + vec4<f32>(1.0, 0.0, 0.0, 0.0) * font_size;
    //++ }

    //++ // Add minus if negative.
    //++ if (n < 0) {
    //++     create_char(10u, thread_index, total_vertex_count, *base_pos, col, font_size);
    //++     *base_pos = *base_pos + vec4<f32>(1.0, 0.0, 0.0, 0.0) * font_size;
    //++ }

    //++ for (var i: i32 = 9 ; i>=0 ; i = i - 1) {
    //++     let remainder = temp_n / i32(joo[i]);  
    //++     temp_n = temp_n - remainder * joo[i];
    //++     found = select(found, true, remainder != 0);
    //++     if (found == true) {
    //++         if (ignore == true) { ignore = false; continue; }
    //++         
    //++         create_char(u32(remainder), thread_index, total_vertex_count, *base_pos, col, font_size);
    //++         *base_pos = *base_pos + vec4<f32>(1.0, 0.0, 0.0, 0.0) * font_size;
    //++     }
    //++ }
}

fn numb_vertices_float(f: f32, max_decimals: u32, total_vertex_count: u32) {

    //++ if (f == 0.0) {
    //++     log_number(0, thread_index, false, base_pos, total_vertex_count, col, font_size);
    //++     *base_pos = *base_pos - vec4<f32>(0.1, 0.0, 0.0, 0.0) * font_size;
    //++     create_char(13u, thread_index, total_vertex_count, *base_pos, col, font_size);
    //++     *base_pos = *base_pos + vec4<f32>(0.3, 0.0, 0.0, 0.0) * font_size;

    //++     for (var i: i32 = 0; i<i32(max_decimals); i = i + 1) {
    //++         log_number(0, thread_index, true, base_pos, total_vertex_count, col, font_size);
    //++     }
    //++     return;
    //++ }

    //++ let fract_and_whole = my_modf(f); 

    //++ // Multiply fractional part. 
    //++ let fract_part = i32((abs(fract_and_whole.fract) + 1.0) * f32(joo[max_decimals]));

    //++ // Cast integer part to uint.
    //++ let integer_part = i32(fract_and_whole.whole);

    //++ // Parse the integer part.
    //++ log_number(integer_part, thread_index, false, base_pos, total_vertex_count, col, font_size);

    //++ // TODO: create_dot function for this.
    //++ *base_pos = *base_pos - vec4<f32>(0.1, 0.0, 0.0, 0.0) * font_size;
    //++ create_char(13u, thread_index, total_vertex_count, *base_pos, col, font_size);
    //++ *base_pos = *base_pos + vec4<f32>(0.3, 0.0, 0.0, 0.0) * font_size;

    //++ // Parse the frag part.
    //++ log_number(abs(fract_part), thread_index, true, base_pos, total_vertex_count, col, font_size);
}

@stage(compute)
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    if (global_id.x >= char_params[0].iterator_end) { return; } 

    total_char_count = 0u;

    // Load element. 
    let ch = chars_array[global_id.x];

    // Calculate distance to the element
    let dist = min(max(1.0, distance(camera.pos.xyz, ch.start_pos.xyz)), 255.0);

    // Calculate the vertex count per char.
    let total_vertex_count = min(u32(f32(char_params[0].max_points_per_char) / f32(dist)), char_params[0].max_points_per_char);

    // Determine the draw start offset and the number of vertices for drawing the element.

}
