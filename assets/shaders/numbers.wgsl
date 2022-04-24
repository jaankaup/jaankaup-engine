struct Camera {
    u_view_proj: mat4x4<f32>,
    pos: vec4<f32>,
};

struct AABB {
    min: vec4<f32>, 
    max: vec4<f32>, 
};

struct ModF {
    fract: f32,
    whole: f32,
};

struct Quaternion {
    w: f32,
    x: f32,
    y: f32,
    z: f32,
};

struct Vertex {
    v: vec4<f32>,
    n: vec4<f32>,
};

struct VVVC {
    pos: vec3<f32>,
    col: u32,
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

struct WorkGroupParams {
    wg_char: Char,
    start_index: u32,
    char_id: u32,
};

struct PirateParams {
    this_id: u32,
};

var<workgroup> workgroup_params: WorkGroupParams; 
//var<workgroup> wg_char: Char;

var<private> private_params: PirateParams; 

// A struct for errors.
// vertex_overflow: 0 :: OK, n :: amount of overflow.

// struct Errors {
//     vertex_buffer_overflow: u32,
// };

struct DrawIndirect {
    vertex_count: atomic<u32>,
    instance_count: u32,
    base_vertex: u32,
    base_instance: u32,
};

@group(0) @binding(0)
var<storage, read_write> indirect: array<DrawIndirect>;

@group(0) @binding(1)
var<storage, read_write> dispatch_counter: array<atomic<u32>>;

@group(0) @binding(2)
var<storage,read_write> input_data: array<Char>;

@group(0) @binding(3)
var<storage,read_write> output_data: array<VVVC>;

let STRIDE: u32 = 64u;
let PI: f32 = 3.14159265358979323846;

// CHAR FUNCTIONS.

fn get_points_per_char(aux_data: u32) -> u32 {
    return aux_data & 0x3FFFu;
}

fn get_number_of_chars(aux_data: u32) -> u32 {
    return (aux_data & 0x3fc000u) >> 14u;
}

fn get_draw_index(aux_data: u32) -> u32 {
    return (aux_data & 0xffc00000u) >> 22u;
}

// FONT STUFF

// Unsafe for y == 0.
fn udiv_up_32(x: u32, y: u32) -> u32 {
  return (x + y - 1u) / y;
}

var<private> bez_indices: array<u32, 17> = array<u32, 17>(
	1953524840u,
	4278978564u,
	4279768080u,
	4294909980u,
	4281083940u,
	4281873456u,
	4282662972u,
	4294920240u,
	1481920588u,
	4284768348u,
	4294967040u,
	4294930536u,
	4294931564u,
	2694617236u,
	2223012984u,
	4287663240u,
	4294967295u,
);

var<private> bez_table: array<vec4<f32>, 164> = array<vec4<f32>, 164>(
    // Bez minus. 
    vec4<f32>(0.3, 0.5, 0.0, 0.25),     // 0
    vec4<f32>(0.433333, 0.5, 0.0, 0.0), // 1
    vec4<f32>(0.566666, 0.5, 0.0, 0.0), // 2
    vec4<f32>(0.7, 0.5, 0.0, 0.0),      // 3 

    // Bez 1.
    vec4<f32>(0.3, 0.1, 0.0, 0.28571428571), // 4
    vec4<f32>(0.433333, 0.1, 0.0, 0.0), // 5
    vec4<f32>(0.566666, 0.1, 0.0, 0.0), // 6
    vec4<f32>(0.7, 0.1, 0.0, 0.0),      // 7 
    vec4<f32>(0.5, 0.1, 0.0, 0.57142857142 ), // 8
    vec4<f32>(0.5, 0.366666, 0.0, 0.0), // 9
    vec4<f32>(0.5, 0.633333, 0.0, 0.0), // 10 
    vec4<f32>(0.5, 0.9, 0.0, 0.0),      // 11 
    vec4<f32>(0.5, 0.9, 0.0, 0.14285714285 ), // 12 
    vec4<f32>(0.4, 0.8, 0.0, 0.0),      // 13 
    vec4<f32>(0.35, 0.8, 0.0, 0.0),     // 14 
    vec4<f32>(0.3, 0.8, 0.0, 0.0),      // 15 

    // Bez 2.
    vec4<f32>(0.3, 0.1, 0.0, 0.33333333333 ),  // 16 
    vec4<f32>(0.433333, 0.1, 0.0, 0.0),  // 17 
    vec4<f32>(0.566666, 0.1, 0.0, 0.0),  // 18 
    vec4<f32>(0.7, 0.1, 0.0, 0.0),       // 19 
    vec4<f32>(0.3, 0.1, 0.0, 0.36666666666),  // 20 
    vec4<f32>(0.416666, 0.3, 0.0, 0.0),  // 21 
    vec4<f32>(0.533333, 0.4, 0.0, 0.0),  // 22
    vec4<f32>(0.65, 0.55, 0.0, 0.0),     // 23 
    vec4<f32>(0.65, 0.55, 0.0, 0.43333333333),// 24 
    vec4<f32>(0.8, 0.7, 0.0, 0.0),       // 25
    vec4<f32>(0.55, 1.08, 0.0, 0.0),     // 26
    vec4<f32>(0.3, 0.8, 0.0, 0.0),       // 27

    // Number 3
    vec4<f32>(0.3, 0.8, 0.0, 0.5),      // 28
    vec4<f32>(0.5, 1.1, 0.0, 0.0),      // 29
    vec4<f32>(0.95, 0.7, 0.0, 0.0),     // 30
    vec4<f32>(0.45, 0.55, 0.0, 0.0),    // 31
    vec4<f32>(0.45, 0.55, 0.0, 0.5),    // 32
    vec4<f32>(1.0, 0.45, 0.0, 0.0),     // 33
    vec4<f32>(0.5, -0.15, 0.0, 0.0),    // 34
    vec4<f32>(0.3, 0.2, 0.0, 0.0),      // 35

    // Number 4
    vec4<f32>(0.6, 0.1, 0.0, 0.47058823529),   // 36
    vec4<f32>(0.6, 0.3666666, 0.0, 0.0),  // 37
    vec4<f32>(0.6, 0.6333333, 0.0, 0.0),  // 38
    vec4<f32>(0.6, 0.9, 0.0, 0.0),        // 39
    vec4<f32>(0.6, 0.9, 0.0, 0.29411764705),   // 40
    vec4<f32>(0.466666, 0.75, 0.0, 0.0),  // 41
    vec4<f32>(0.333333, 0.6, 0.0, 0.0),   // 42
    vec4<f32>(0.2, 0.45, 0.0, 0.0),       // 43
    
    vec4<f32>(0.2, 0.45, 0.0, 0.23529411764),  // 44
    vec4<f32>(0.3666666, 0.45, 0.0, 0.0), // 45
    vec4<f32>(0.5333333, 0.45, 0.0, 0.0), // 46
    vec4<f32>(0.7, 0.45, 0.0, 0.0),       // 47
    
    // Number 5.
    vec4<f32>(0.3, 0.9, 0.0, 0.15384615384),  // 48
    vec4<f32>(0.433333, 0.9, 0.0, 0.0),  // 49
    vec4<f32>(0.566666, 0.9, 0.0, 0.0),  // 50
    vec4<f32>(0.7, 0.9, 0.0, 0.0),       // 51
    vec4<f32>(0.3, 0.5, 0.0, 0.15384615384),  // 52
    vec4<f32>(0.3, 0.633333, 0.0, 0.0),  // 53
    vec4<f32>(0.3, 0.766666, 0.0, 0.0),  // 54
    vec4<f32>(0.3, 0.9, 0.0, 0.0),       // 55
    vec4<f32>(0.3, 0.5, 0.0, 0.6923076923),  // 56
    vec4<f32>(1.0, 0.75, 0.0, 0.0),      // 57
    vec4<f32>(0.7, -0.2, 0.0, 0.0),      // 58
    vec4<f32>(0.3, 0.2, 0.0, 0.0),       // 59

    // Number 6
    vec4<f32>(0.7, 0.8, 0.0, 0.23809523809),  // 60
    vec4<f32>(0.5, 1.05, 0.0, 0.0),      // 61
    vec4<f32>(0.3, 0.8, 0.0, 0.0),       // 62
    vec4<f32>(0.3, 0.5, 0.0, 0.0),       // 63
    vec4<f32>(0.3, 0.5, 0.0, 0.42857142857 ),  // 64
    vec4<f32>(0.3, -0.05, 0.0, 0.0),     // 65
    vec4<f32>(0.7, 0.0, 0.0, 0.0),       // 66
    vec4<f32>(0.72, 0.4, 0.0, 0.0),      // 67
    vec4<f32>(0.72, 0.4, 0.0, 0.23809523809), // 68
    vec4<f32>(0.72, 0.6, 0.0, 0.0),      // 69
    vec4<f32>(0.5, 0.7, 0.0, 0.0),       // 70
    vec4<f32>(0.3, 0.5, 0.0, 0.0),       // 71
    
    // Number 7 .
    // five_bez_a
    vec4<f32>(0.4, 0.1, 0.0, 0.46153846153 ), // 72
    vec4<f32>(0.5, 0.366666, 0.0, 0.0), // 73
    vec4<f32>(0.6, 0.633333, 0.0, 0.0), // 74
    vec4<f32>(0.7, 0.9, 0.0, 0.0),      // 75
    
    // Number 8
    vec4<f32>(0.5, 0.9, 0.0, 0.25), // 76
    vec4<f32>(0.2, 0.85, 0.0, 0.0), // 77
    vec4<f32>(0.2, 0.55, 0.0, 0.0), // 78
    vec4<f32>(0.5, 0.5, 0.0, 0.0),  // 79
    vec4<f32>(0.5, 0.9, 0.0, 0.25), // 80
    vec4<f32>(0.8, 0.85, 0.0, 0.0), // 81
    vec4<f32>(0.8, 0.55, 0.0, 0.0), // 82
    vec4<f32>(0.5, 0.5, 0.0, 0.0),  // 83
    vec4<f32>(0.5, 0.1, 0.0, 0.25), // 84
    vec4<f32>(0.8, 0.15, 0.0, 0.0), // 85
    vec4<f32>(0.8, 0.45, 0.0, 0.0), // 86
    vec4<f32>(0.5, 0.5, 0.0, 0.0),  // 87
    vec4<f32>(0.5, 0.1, 0.0, 0.25), // 88
    vec4<f32>(0.2, 0.15, 0.0, 0.0), // 89
    vec4<f32>(0.2, 0.45, 0.0, 0.0), // 90
    vec4<f32>(0.5, 0.5, 0.0, 0.0),  // 91
    
    // Number 9
    vec4<f32>(0.3, 0.2, 0.0, 0.4), // 92
    vec4<f32>(0.5, -0.05, 0.0, 0.0),    // 93
    vec4<f32>(0.7, 0.2, 0.0, 0.0),      // 94
    vec4<f32>(0.7, 0.6, 0.0, 0.0),      // 95
    vec4<f32>(0.7, 0.6, 0.0, 0.3), // 96
    vec4<f32>(0.7, 0.95, 0.0, 0.0),     // 97
    vec4<f32>(0.4, 1.0, 0.0, 0.0),      // 98
    vec4<f32>(0.28, 0.8, 0.0, 0.0),     // 99
    vec4<f32>(0.28, 0.8, 0.0, 0.3),// 100
    vec4<f32>(0.1, 0.4, 0.0, 0.0),      // 101
    vec4<f32>(0.6, 0.4, 0.0, 0.0),      // 102
    vec4<f32>(0.7, 0.6, 0.0, 0.0),      // 103
    
    // Number 0 
    vec4<f32>(0.5, 0.9, 0.0, 0.25),   // 104
    vec4<f32>(0.25, 0.85, 0.0, 0.0),  // 105
    vec4<f32>(0.25, 0.55, 0.0, 0.0),  // 106
    vec4<f32>(0.25, 0.5, 0.0, 0.0),   // 107
    vec4<f32>(0.5, 0.9, 0.0, 0.25),   // 108
    vec4<f32>(0.75, 0.85, 0.0, 0.0),  // 109
    vec4<f32>(0.75, 0.55, 0.0, 0.0),  // 110
    vec4<f32>(0.75, 0.5, 0.0, 0.0),   // 111
    vec4<f32>(0.5, 0.1, 0.0, 0.25),   // 112
    vec4<f32>(0.25, 0.15, 0.0, 0.0),  // 113
    vec4<f32>(0.25, 0.45, 0.0, 0.0),  // 114
    vec4<f32>(0.25, 0.5, 0.0, 0.0),   // 115
    vec4<f32>(0.5, 0.1, 0.0, 0.25),   // 116
    vec4<f32>(0.75, 0.15, 0.0, 0.0),  // 117
    vec4<f32>(0.75, 0.45, 0.0, 0.0),  // 118
    vec4<f32>(0.75, 0.5, 0.0, 0.0),   // 119
    
    // Number inf 
    vec4<f32>(0.5, 0.5, 0.0, 0.25),   // 120
    vec4<f32>(0.4, 0.7, 0.0, 0.0),    // 121
    vec4<f32>(0.2, 0.7, 0.0, 0.0),    // 122
    vec4<f32>(0.1, 0.5, 0.0, 0.0),    // 123  
    vec4<f32>(0.1, 0.5, 0.0, 0.25),   // 124
    vec4<f32>(0.2, 0.3, 0.0, 0.0),    // 125
    vec4<f32>(0.4, 0.3, 0.0, 0.0),    // 126
    vec4<f32>(0.5, 0.5, 0.0, 0.0),    // 127  
    vec4<f32>(0.5, 0.5, 0.0, 0.25),   // 128
    vec4<f32>(0.6, 0.7, 0.0, 0.0),    // 129
    vec4<f32>(0.8, 0.7, 0.0, 0.0),    // 130
    vec4<f32>(0.9, 0.5, 0.0, 0.0),    // 131  
    vec4<f32>(0.9, 0.5, 0.0, 0.25),   // 132
    vec4<f32>(0.8, 0.3, 0.0, 0.0),    // 133
    vec4<f32>(0.6, 0.3, 0.0, 0.0),    // 134
    vec4<f32>(0.5, 0.5, 0.0, 0.0),    // 135  
    
    // Nan
    vec4<f32>(0.2, 0.1, 0.0, 0.30769230769), // 136
    vec4<f32>(0.2, 0.366666, 0.0, 0.0), // 137
    vec4<f32>(0.2, 0.633333, 0.0, 0.0), // 138
    vec4<f32>(0.2, 0.9, 0.0, 0.0),      // 139
    
    vec4<f32>(0.2, 0.9, 0.0, 0.38461538461),// 140
    vec4<f32>(0.4, 0.633333, 0.0, 0.0), // 141
    vec4<f32>(0.6, 0.366666, 0.0, 0.0), // 142
    vec4<f32>(0.8, 0.1, 0.0, 0.0),      // 143
    
    vec4<f32>(0.8, 0.1, 0.0, 0.30769230769), // 144
    vec4<f32>(0.8, 0.366666, 0.0, 0.0), // 145
    vec4<f32>(0.8, 0.633333, 0.0, 0.0), // 146
    vec4<f32>(0.8, 0.9, 0.0, 0.0),      // 147

    vec4<f32>(0.23, 0.089999996, 0.0, 0.015000001),
    vec4<f32>(0.205, 0.085, 0.0, 0.0),
    vec4<f32>(0.205, 0.055000003, 0.0, 0.0),
    vec4<f32>(0.205, 0.05, 0.0, 0.0),
    vec4<f32>(0.23, 0.089999996, 0.0, 0.015000001),
    vec4<f32>(0.255, 0.085, 0.0, 0.0),
    vec4<f32>(0.255, 0.055000003, 0.0, 0.0),
    vec4<f32>(0.255, 0.05, 0.0, 0.0),
    vec4<f32>(0.23, 0.010000001, 0.0, 0.015000001),
    vec4<f32>(0.205, 0.015000001, 0.0, 0.0),
    vec4<f32>(0.205, 0.044999998, 0.0, 0.0),
    vec4<f32>(0.205, 0.05, 0.0, 0.0),
    vec4<f32>(0.23, 0.010000001, 0.0, 0.015000001),
    vec4<f32>(0.255, 0.015000001, 0.0, 0.0),
    vec4<f32>(0.255, 0.044999998, 0.0, 0.0),
    vec4<f32>(0.255, 0.05, 0.0, 0.0),

);

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

/// FONT STUFF


// @param thread_index
fn bezier_4c(vertices_per_thread: u32,
             total_num_points: u32,
             c0: vec3<f32>,
             c1: vec3<f32>,
             c2: vec3<f32>,
             c3: vec3<f32>,
             color: u32,
             start_offset: u32) {

    for (var i: u32 = 0u ; i < vertices_per_thread ; i = i + 1u) {
        let t = f32(i * 64u + private_params.this_id) / (f32(total_num_points) - 1.0);
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;
        let result = c0.xyz * mt3 + c1.xyz * 3.0 * mt2*t + c2.xyz * 3.0 * mt*t2 + c3.xyz * t3;
        let ind = 64u * i + private_params.this_id;
        if (ind < total_num_points) {
            output_data[start_offset + ind] = VVVC(result, color);
        }
    }
}

// @param char_index: the index of char.
// @param local_index: thread_id
// @param num_poinst: total number of vertices for rendering
// @param offset: the start position of char.
// @param color: the color of the char.
fn create_char(char_index: u32,
               num_points: u32,
               offset: vec3<f32>,
               color: u32,
               font_size: f32) {

    let index = bez_indices[char_index];

    var indices = vec4<u32>(
        index & 0xffu,
        (index & 0xff00u) >> 8u,
        (index & 0xff0000u) >> 16u,
        (index & 0xff000000u) >> 24u
    );

    var bez_values = array<vec4<f32>, 4>(
        bez_table[indices[0]],
        bez_table[indices[1]],
        bez_table[indices[2]],
        bez_table[indices[3]]
    );

    var number_of_render_points = vec4<u32>(
        u32(bez_values[0].w * f32(num_points)),
        u32(bez_values[1].w * f32(num_points)),
        u32(bez_values[2].w * f32(num_points)),
        u32(bez_values[3].w * f32(num_points))
    );

    let actual_number_of_render_points = number_of_render_points[0] + number_of_render_points[1] + number_of_render_points[2] + number_of_render_points[3];

    if (private_params.this_id == 0u) {
        workgroup_params.start_index = atomicAdd(&indirect[get_draw_index(workgroup_params.wg_char.auxiliary_data)].vertex_count, actual_number_of_render_points);
    }
    workgroupBarrier();

    let base_offset = workgroup_params.start_index;

    var start_offsets = vec4<u32>(
        base_offset,
        base_offset + number_of_render_points[0],
        base_offset + number_of_render_points[0] + number_of_render_points[1],
        base_offset + number_of_render_points[0] + number_of_render_points[1] + number_of_render_points[2]
    );

    for (var i: u32 = 0u ; i < 4u ; i = i + 1u) {
        //++ if (indices[i] == 255u) { break; }

        // Create per thead workload (how many vertices to create and store).
        let vertices_per_thread = udiv_up_32(number_of_render_points[i], 64u);

        bezier_4c(
            vertices_per_thread,
            number_of_render_points[i],
            font_size * bez_table[indices[i]].xyz + offset,
            font_size * bez_table[indices[i] + 1u].xyz + offset,
            font_size * bez_table[indices[i] + 2u].xyz + offset,
            font_size * bez_table[indices[i] + 3u].xyz + offset,
            color,
            start_offsets[i]
        );
    }
}

fn myTruncate(f: f32) -> f32 {
    return select(f32( i32( floor(f) ) ), f32( i32( ceil(f) ) ), f < 0.0); 
}

fn my_modf(f: f32) -> ModF {
    let iptr = trunc(f);
    let fptr = f - iptr;
    return ModF (
        select(fptr, (-1.0)*fptr, f < 0.0),
        iptr,
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
    return the_sign_of_i32(n) + log10_u32(u32(abs(n))) + 1u;
    //return the_sign_of_i32(n) + log10_u32(abs_i32(n)) + 1u;
} 

fn number_of_chars_f32(f: f32, number_of_decimals: u32) -> u32 {

    let m = my_modf(f);
    return number_of_chars_i32(i32(m.whole)) + number_of_decimals + 1u;
}

fn log_number_new(
    n: i32,
    base_pos: ptr<function, vec3<f32>>,
    vertices_per_char: u32,
    col: u32,
    font_size: f32) {

    // Render minus.
    if (n < 0) {
        create_char(10u, vertices_per_char, *base_pos, col, font_size);
        *base_pos = *base_pos + vec3<f32>(1.0, 0.0, 0.0) * font_size;
    }

    // Calculate the number of rendereable chars without minus.
    let number_of_chars = number_of_chars_i32(abs(n));

    var temp_n = u32(abs(n)); 

    for (var i: u32 = 0u; i < number_of_chars ; i = i + 1u) {

        var head = temp_n / PowersOf10[number_of_chars - 1u - i]; //temp_n - ((temp_n / 10u) * 10u);
        head = head - ((head / 10u) * 10u);
        create_char(head,
                    vertices_per_char,
                    *base_pos,
                    col,
                    font_size
        );
        *base_pos = *base_pos + vec3<f32>(1.0, 0.0, 0.0) * font_size;
    }
}

fn log_f32_fract(
    f: f32,
    base_pos: ptr<function, vec3<f32>>,
    vertices_per_char: u32,
    col: u32,
    font_size: f32,
    number_of_decimals: u32) {
        
        for (var i: u32 = 0u ; i < min(number_of_decimals,6u); i = i + 1u) {
 
            // OVERFLOW RISK! TODO: fix.
            var head = u32(round(f * f32(PowersOf10[i + 1u])));
            head = head - ((head / 10u) * 10u);

            create_char(
                head,
                vertices_per_char,
                *base_pos,
                col,
                font_size
            );

            *base_pos = *base_pos + vec3<f32>(1.0, 0.0, 0.0) * font_size;
        }
}

fn log_float(f: f32, max_decimals: u32, base_pos: ptr<function, vec3<f32>>, total_vertex_count: u32, col: u32, font_size: f32) {

    let m = my_modf(f);
    log_number_new(i32(m.whole), base_pos, total_vertex_count, col, font_size);

    *base_pos = *base_pos - vec3<f32>(0.1, 0.0, 0.0) * font_size;
    create_char(13u, total_vertex_count, *base_pos, col, font_size);
    *base_pos = *base_pos + vec3<f32>(0.4, 0.0, 0.0) * font_size;

    log_f32_fract(abs(m.fract), base_pos, total_vertex_count, col, font_size, max_decimals);
}

fn log_vec3(v: vec3<f32>, max_decimals: u32, base_pos: ptr<function, vec3<f32>>, total_vertex_count: u32, col: u32, font_size: f32) {

    create_char(11u, total_vertex_count, *base_pos, col, font_size);
    *base_pos = *base_pos + vec3<f32>(0.5, 0.0, 0.0) * font_size;

    log_float(v.x, max_decimals, base_pos, total_vertex_count, col, font_size);
    *base_pos = *base_pos + vec3<f32>(0.5, 0.0, 0.0) * font_size;

    log_float(v.y, max_decimals, base_pos, total_vertex_count, col, font_size);
    *base_pos = *base_pos + vec3<f32>(0.5, 0.0, 0.0) * font_size;

    log_float(v.z, max_decimals, base_pos, total_vertex_count, col, font_size);
    *base_pos = *base_pos + vec3<f32>(0.5, 0.0, 0.0) * font_size;

    create_char(12u, total_vertex_count, *base_pos, col, font_size);
    *base_pos = *base_pos + vec3<f32>(0.2, 0.0, 0.0) * font_size;
}

fn log_vec4(v: vec4<f32>, max_decimals: u32, base_pos: ptr<function, vec3<f32>>, total_vertex_count: u32, col: u32, font_size: f32) {

    create_char(11u, total_vertex_count, *base_pos, col, font_size);
    *base_pos = *base_pos + vec3<f32>(0.5, 0.0, 0.0) * font_size;

    log_float(v.x, max_decimals, base_pos, total_vertex_count, col, font_size);
    *base_pos = *base_pos + vec3<f32>(0.5, 0.0, 0.0) * font_size;

    log_float(v.y, max_decimals, base_pos, total_vertex_count, col, font_size);
    *base_pos = *base_pos + vec3<f32>(0.5, 0.0, 0.0) * font_size;

    log_float(v.z, max_decimals, base_pos, total_vertex_count, col, font_size);
    *base_pos = *base_pos + vec3<f32>(0.5, 0.0, 0.0) * font_size;

    log_float(v.w, max_decimals, base_pos, total_vertex_count, col, font_size);
    *base_pos = *base_pos + vec3<f32>(0.5, 0.0, 0.0) * font_size;

    create_char(12u, total_vertex_count, *base_pos, col, font_size);
    *base_pos = *base_pos + vec3<f32>(0.2, 0.0, 0.0) * font_size;
}

// fn log_vec4_f32(v: vec4<f32>, max_decimals: u32, thread_index: u32, base_pos: ptr<function, vec3<f32>>, total_vertex_count: u32, col: u32, font_size: f32) {
// 
//     // TODO: create_dot function for this.
//     create_char(11u, thread_index, total_vertex_count, *base_pos, col, font_size);
//     *base_pos = *base_pos + vec3<f32>(0.5, 0.0, 0.0) * font_size;
// 
//     log_float(v.x, max_decimals, thread_index, base_pos, total_vertex_count, col, font_size);
// 
//     *base_pos = *base_pos + vec3<f32>(2.0, 0.0, 0.0) * font_size;
// 
//     log_float(v.y, max_decimals, thread_index, base_pos, total_vertex_count, col, font_size);
// 
//     *base_pos = *base_pos + vec3<f32>(2.0, 0.0, 0.0) * font_size;
// 
//     log_float(v.w, max_decimals, thread_index, base_pos, total_vertex_count, col, font_size);
// 
//     *base_pos = *base_pos + vec3<f32>(2.0, 0.0, 0.0) * font_size;
// 
//     log_float(v.z, max_decimals, thread_index, base_pos, total_vertex_count, col, font_size);
// 
//     // TODO: create_dot function for this.
//     *base_pos = *base_pos + vec3<f32>(-0.5, 0.0, 0.0) * font_size;
//     create_char(12u, thread_index, total_vertex_count, *base_pos, col, font_size);
//     *base_pos = *base_pos + vec3<f32>(0.5, 0.0, 0.0) * font_size;
// }

// fn log_vec3_f32(v: vec3<f32>, max_decimals: u32, thread_index: u32, base_pos: ptr<function, vec3<f32>>, total_vertex_count: u32, col: u32, font_size: f32) {
// 
//     // TODO: create_dot function for this.
//     create_char(11u, thread_index, total_vertex_count, *base_pos, col, font_size);
//     *base_pos = *base_pos + vec3<f32>(0.5, 0.0, 0.0) * font_size;
// 
//     log_float(v.x, max_decimals, thread_index, base_pos, total_vertex_count, col, font_size);
// 
//     *base_pos = *base_pos + vec3<f32>(2.0, 0.0, 0.0) * font_size;
// 
//     log_float(v.y, max_decimals, thread_index, base_pos, total_vertex_count, col, font_size);
// 
//     *base_pos = *base_pos + vec3<f32>(2.0, 0.0, 0.0) * font_size;
// 
//     log_float(v.z, max_decimals, thread_index, base_pos, total_vertex_count, col, font_size);
// 
//     // TODO: create_dot function for this.
//     *base_pos = *base_pos + vec3<f32>(-0.5, 0.0, 0.0) * font_size;
//     create_char(12u, thread_index, total_vertex_count, *base_pos, col, font_size);
//     *base_pos = *base_pos + vec3<f32>(0.5, 0.0, 0.0) * font_size;
// }

// bit_count must be in range 1..32. Otherwise there will be undefined behaviour. TODO: fix.
// fn log_u32_b2(n: u32, bit_count: u32, thread_index: u32, base_pos: ptr<function, vec3<f32>>, total_vertex_count: u32, col: u32, font_size: f32) {
// 
//     var bit_c = select(bit_count, 32u, n > 32u);
//     bit_c = select(bit_count, 1u, n == 0u);
// 
//     for (var i: i32 = i32(bit_count) - 1 ; i >= 0 ; i = i - 1) {
//         log_number(select(1, 0, ((n & (1u << u32(i))) >> u32(i)) == 0u),
//                    thread_index,
//                    false,
//                    base_pos,
//                    total_vertex_count,
//                    col,
//                    font_size
//         );
//     } 
// }

@compute
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    if (local_index == 0u) {
        let char_number = atomicAdd(&dispatch_counter[0], 1u);
        workgroup_params.char_id = char_number;
        workgroup_params.wg_char = input_data[char_number]; 
    }
    workgroupBarrier();

    // Load element. TODO: from hash array. TODO: use workgroup Char instead?
    var ch = input_data[workgroup_params.char_id]; 
    // private_char = wg_char;

    // Store thread id.
    private_params.this_id = local_index;

    var jooo = ch.start_pos;

    //log_float(ch.value.x,
    //          decimals, //workgroup_params.wg_char.decimal_count,
    //          &jooo,
    //          get_points_per_char(ch.auxiliary_data),
    //          ch.color,
    //          ch.font_size);
    if (ch.vec_dim_count == 1u) {
       log_float(ch.value.x,
                 workgroup_params.wg_char.decimal_count, //u32(the_start_position.w),
                 &jooo,
                 get_points_per_char(ch.auxiliary_data),
                 ch.color,
                 ch.font_size);
    }

    if (ch.vec_dim_count == 3u) {
        log_vec3(ch.value.xyz,
                  workgroup_params.wg_char.decimal_count,
                  &jooo,
                  get_points_per_char(ch.auxiliary_data),
                  ch.color,
                  ch.font_size);
    }

    if (ch.vec_dim_count == 4u) {
        log_vec4(ch.value,
                  workgroup_params.wg_char.decimal_count,
                  &jooo,
                  get_points_per_char(ch.auxiliary_data),
                  ch.color,
                  ch.font_size);
    }

   //++ // Log f32.
   //++ if (the_char.vec_dim_count == 1u) {
   //++     log_float(the_char.value.x,
   //++               2u, //u32(the_start_position.w),
   //++               local_index,
   //++               &the_start_position,
   //++               total_vertex_count,
   //++               the_char.color,
   //++               the_char.font_size);
   //++ }
   //++ // Log vec3<f32>.
   //++ else if (the_char.vec_dim_count == 3u) {
   //++     log_vec3_f32(the_char.value.xyz,
   //++                  2u, //u32(the_start_position.w),
   //++                  local_index,
   //++                  &the_start_position,
   //++                  total_vertex_count,
   //++                  the_char.color,
   //++                  the_char.font_size);
   //++ }
   //++ // Log vec4<f32>.
   //++ else if (the_char.vec_dim_count == 4u) {
   //++     log_vec4_f32(the_char.value,
   //++                  2u, //u32(the_start_position.w),
   //++                  local_index,
   //++                  &the_start_position,
   //++                  total_vertex_count,
   //++                  the_char.color,
   //++                  the_char.font_size);
   //++ }
}
