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

struct VisualizationParams{
    max_vertex_capacity: u32,
    iterator_start_index: u32,
    iterator_end_index: u32,
    arrow_size: f32,
};

struct Vertex {
    v: vec4<f32>,
    n: vec4<f32>,
};

struct VVVC {
    pos: vec3<f32>,
    col: u32,
};

struct WorkGroupParams {
    start_index: u32, // Start position of the output buffer.
    last_index: u32,  // Last legal position.
};

struct PirateParams {
    this_offset: u32,      // This thread offset in the current work group.
    number_of_writes: u32, // Number of points to calculate/store.
};

var<workgroup> workgroup_params: WorkGroupParams; 
var<private> private_params: PirateParams; 

// struct Char {
//     start_position: vec3<f32>;
//     char_index: u32;
// }

// var<private> char_sequency: array<Char, 32>;

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

@group(0)
@binding(0)
var<uniform> camera: Camera;

@group(0)
@binding(1)
var<uniform> visualization_params: VisualizationParams;

@group(0)
@binding(2)
var<storage, read_write> counter: array<atomic<u32>>;

@group(0)
@binding(3)
var<storage, read_write> arrows: array<Arrow>;

@group(0)
@binding(4)
var<storage,read_write> output_data: array<VVVC>;

let STRIDE: u32 = 64u;
let PI: f32 = 3.14159265358979323846;

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

    // vec4<f32>(0.46, 0.17999999, 0.0, 0.030000001),
    // vec4<f32>(0.41, 0.17, 0.0, 0.0),
    // vec4<f32>(0.41, 0.11000001, 0.0, 0.0),
    // vec4<f32>(0.41, 0.1, 0.0, 0.0),
    // vec4<f32>(0.46, 0.17999999, 0.0, 0.030000001),
    // vec4<f32>(0.51, 0.17, 0.0, 0.0),
    // vec4<f32>(0.51, 0.11000001, 0.0, 0.0),
    // vec4<f32>(0.51, 0.1, 0.0, 0.0),
    // vec4<f32>(0.46, 0.020000001, 0.0, 0.030000001),
    // vec4<f32>(0.41, 0.030000001, 0.0, 0.0),
    // vec4<f32>(0.41, 0.089999996, 0.0, 0.0),
    // vec4<f32>(0.41, 0.1, 0.0, 0.0),
    // vec4<f32>(0.46, 0.020000001, 0.0, 0.030000001),
    // vec4<f32>(0.51, 0.030000001, 0.0, 0.0),
    // vec4<f32>(0.51, 0.089999996, 0.0, 0.0),
    // vec4<f32>(0.51, 0.1, 0.0, 0.0)
);

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

/// FONT STUFF


// @param thread_index
fn bezier_4c(thread_index: u32, 
             vertices_per_thread: u32,
             total_num_points: u32,
             c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, c3: vec4<f32>, color: u32) {

    // if (num_points < 1u) { return; }

    for (var i: u32 = 0u ; i < vertices_per_thread ; i = i + 1u) {
        let t = f32(i * 64u + thread_index) / (f32(total_num_points) - 1.0);
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;
        let result = c0.xyz * mt3 + c1.xyz * 3.0 * mt2*t + c2.xyz * 3.0 * mt*t2 + c3.xyz * t3;
        output_data[workgroup_params.start_index + 64u * i + thread_index] = VVVC(result, color);
    }
}

// @param char_index: the index of char.
// @param local_index: thread_id
// @param num_poinst: total number of vertices for rendering
// @param offset: the start position of char.
// @param color: the color of the char.
fn create_char(char_index: u32,
               local_index: u32,
               num_points: u32,
               offset: vec4<f32>,
               color: u32,
               font_size: f32) {

    let index = bez_indices[char_index];

    var indices = array<u32, 4>(
        index & 0xffu,
        (index & 0xff00u) >> 8u,
        (index & 0xff0000u) >> 16u,
        (index & 0xff000000u) >> 24u,
    );

    for (var i: i32 = 0 ; i<4 ; i = i + 1) {
        let bez_index = indices[u32(i)];
        if (bez_index == 255u) { break; }

        // Relative factor for rendering the bezier curve.
        let bi = bez_table[bez_index];

        // Number of points for this bezier curve. At least one vertex per bezier.
        let count = u32(max(1.0, f32(num_points) * bi.w));

	// Allocate memory from output_data buffer.
        if (local_index == 0u) {

            // The start position for storing the vertices.
            let start_index: u32 = atomicAdd(&counter[0], count);

            // Update the thread group params.
            workgroup_params = WorkGroupParams (
                start_index,
                start_index + u32(count), // the end index. do we need this?
            );
        }
        workgroupBarrier();

        // Create per thead workload (how many vertices to create and store).
        let chunks_per_thread = i32(udiv_up_32(count, 64u));

    	// // Offset and chunk size (number of vertices to create and store).
        let vertices_per_thread = select(u32(chunks_per_thread),
    	                                 u32(max(chunks_per_thread - 1, 0)),
    	                                 u32(max(chunks_per_thread - 1, 0)) * 64u + local_index >= count);
        bezier_4c(
            local_index,
            vertices_per_thread,
            count,
            font_size * bez_table[bez_index    ]  + offset,
            font_size * bez_table[bez_index + 1u] + offset,
            font_size * bez_table[bez_index + 2u] + offset,
            font_size * bez_table[bez_index + 3u] + offset,
            color
        );
    }
}

var<private> joo: array<i32, 10> =  array<i32, 10> (
                      1,
                      10,
                      100,
                      1000,
                      10000,
                      100000,
                      1000000,
                      10000000,
                      100000000,
                      1000000000
);

// @param n : number that should be rendered.
// @thread_index : local_thread_index.
// @ignore_first : ignore the first number. This should be false when rendering u32. 
// @base_pos : the starting position for rendering. 
// @total_vertex_count : total number of vertices for rendering. 
// @cola : the color of the vertices. 
fn log_number(n: i32, thread_index: u32, ignore_first: bool, base_pos: ptr<function, vec4<f32>>, total_vertex_count: u32, col: u32, font_size: f32) {

    var found = false;
    var ignore = ignore_first;
    var temp_n = abs(n);

    if (n == 0) {
        create_char(0u, thread_index, total_vertex_count, *base_pos, col, font_size);
        *base_pos = *base_pos + vec4<f32>(1.0, 0.0, 0.0, 0.0) * font_size;
    }

    // Add minus if negative.
    if (n < 0) {
        create_char(10u, thread_index, total_vertex_count, *base_pos, col, font_size);
        *base_pos = *base_pos + vec4<f32>(1.0, 0.0, 0.0, 0.0) * font_size;
    }

    for (var i: i32 = 9 ; i>=0 ; i = i - 1) {
        let remainder = temp_n / i32(joo[i]);  
        temp_n = temp_n - remainder * joo[i];
        found = select(found, true, remainder != 0);
        if (found == true) {
            if (ignore == true) { ignore = false; continue; }
            
            create_char(u32(remainder), thread_index, total_vertex_count, *base_pos, col, font_size);
            *base_pos = *base_pos + vec4<f32>(1.0, 0.0, 0.0, 0.0) * font_size;
        }
    }
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

fn log_float(f: f32, max_decimals: u32, thread_index: u32, base_pos: ptr<function, vec4<f32>>, total_vertex_count: u32, col: u32, font_size: f32) {

    if (f == 0.0) {
        log_number(0, thread_index, false, base_pos, total_vertex_count, col, font_size);
        *base_pos = *base_pos - vec4<f32>(0.1, 0.0, 0.0, 0.0) * font_size;
        create_char(13u, thread_index, total_vertex_count, *base_pos, col, font_size);
        *base_pos = *base_pos + vec4<f32>(0.3, 0.0, 0.0, 0.0) * font_size;

        for (var i: i32 = 0; i<i32(max_decimals); i = i + 1) {
            log_number(0, thread_index, true, base_pos, total_vertex_count, col, font_size);
        }
        return;
    }

    // // Add minus if negative.
    // if (f < 0.0) {
    //     create_char(10u, thread_index, total_vertex_count, *base_pos, col);
    //     *base_pos = *base_pos + vec4<f32>(1.0, 0.0, 0.0, 0.0);
    // }

    // Get rid of the minus. Necessery?
    // let f_positive = abs(f);

    let fract_and_whole = my_modf(f); 

    // Multiply fractional part. 
    let fract_part = i32((abs(fract_and_whole.fract) + 1.0) * f32(joo[max_decimals]));

    // Cast integer part to uint.
    let integer_part = i32(fract_and_whole.whole);

    // Parse the integer part.
    log_number(integer_part, thread_index, false, base_pos, total_vertex_count, col, font_size);

    // TODO: create_dot function for this.
    *base_pos = *base_pos - vec4<f32>(0.1, 0.0, 0.0, 0.0) * font_size;
    create_char(13u, thread_index, total_vertex_count, *base_pos, col, font_size);
    *base_pos = *base_pos + vec4<f32>(0.3, 0.0, 0.0, 0.0) * font_size;

    // Parse the frag part.
    log_number(abs(fract_part), thread_index, true, base_pos, total_vertex_count, col, font_size);
}

fn log_vec4_f32(v: vec4<f32>, max_decimals: u32, thread_index: u32, base_pos: ptr<function, vec4<f32>>, total_vertex_count: u32, col: u32, font_size: f32) {

    // TODO: create_dot function for this.
    create_char(11u, thread_index, total_vertex_count, *base_pos, col, font_size);
    *base_pos = *base_pos + vec4<f32>(0.5, 0.0, 0.0, 0.0) * font_size;

    log_float(v.x, max_decimals, thread_index, base_pos, total_vertex_count, col, font_size);

    *base_pos = *base_pos + vec4<f32>(2.0, 0.0, 0.0, 0.0) * font_size;

    log_float(v.y, max_decimals, thread_index, base_pos, total_vertex_count, col, font_size);

    *base_pos = *base_pos + vec4<f32>(2.0, 0.0, 0.0, 0.0) * font_size;

    log_float(v.w, max_decimals, thread_index, base_pos, total_vertex_count, col, font_size);

    *base_pos = *base_pos + vec4<f32>(2.0, 0.0, 0.0, 0.0) * font_size;

    log_float(v.z, max_decimals, thread_index, base_pos, total_vertex_count, col, font_size);

    // TODO: create_dot function for this.
    *base_pos = *base_pos + vec4<f32>(-0.5, 0.0, 0.0, 0.0) * font_size;
    create_char(12u, thread_index, total_vertex_count, *base_pos, col, font_size);
    *base_pos = *base_pos + vec4<f32>(0.5, 0.0, 0.0, 0.0) * font_size;
}

fn log_vec3_f32(v: vec3<f32>, max_decimals: u32, thread_index: u32, base_pos: ptr<function, vec4<f32>>, total_vertex_count: u32, col: u32, font_size: f32) {

    // TODO: create_dot function for this.
    create_char(11u, thread_index, total_vertex_count, *base_pos, col, font_size);
    *base_pos = *base_pos + vec4<f32>(0.5, 0.0, 0.0, 0.0) * font_size;

    log_float(v.x, max_decimals, thread_index, base_pos, total_vertex_count, col, font_size);

    *base_pos = *base_pos + vec4<f32>(2.0, 0.0, 0.0, 0.0) * font_size;

    log_float(v.y, max_decimals, thread_index, base_pos, total_vertex_count, col, font_size);

    *base_pos = *base_pos + vec4<f32>(2.0, 0.0, 0.0, 0.0) * font_size;

    log_float(v.z, max_decimals, thread_index, base_pos, total_vertex_count, col, font_size);

    // TODO: create_dot function for this.
    *base_pos = *base_pos + vec4<f32>(-0.5, 0.0, 0.0, 0.0) * font_size;
    create_char(12u, thread_index, total_vertex_count, *base_pos, col, font_size);
    *base_pos = *base_pos + vec4<f32>(0.5, 0.0, 0.0, 0.0) * font_size;
}

// bit_count must be in range 1..32. Otherwise there will be undefined behaviour. TODO: fix.
fn log_u32_b2(n: u32, bit_count: u32, thread_index: u32, base_pos: ptr<function, vec4<f32>>, total_vertex_count: u32, col: u32, font_size: f32) {

    var bit_c = select(bit_count, 32u, n > 32u);
    bit_c = select(bit_count, 1u, n == 0u);

    for (var i: i32 = i32(bit_count) - 1 ; i >= 0 ; i = i - 1) {
        log_number(select(1, 0, ((n & (1u << u32(i))) >> u32(i)) == 0u),
                   thread_index,
                   false,
                   base_pos,
                   total_vertex_count,
                   col,
                   font_size
        );
    } 
}

@compute
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    let actual_index = global_id.x + counter[1]; 

    let c = mapRange(0.0, 
                     4096.0,
        	     0.0,
        	     255.0, 
        	     f32(actual_index)
    );

    //++ let max_vertex_count = 5000u;
    //++ let offset = udiv_up_32(max_vertex_count, 64u);
    //++ let col = rgba_u32(255u, 0u, 0u, 255u);
    //++ let start_position = vec3<f32>(0.1, 0.1, 0.1);

    //++ let dist = min(max(1.0, distance(camera.pos.xyz, start_position)), 255.0);
    //++ let total_vertex_count = u32(f32(max_vertex_count) / f32(dist));

    // if (local_index == 0) {
    //     let start_index = atomicAdd(&counter[0], total_vertex_count);
    //     workgroup_params = WorkGroupParams {
    //         start_index,
    //         start_index + total_vertex_count,
    //     };
    // }
    // workgroupBarrier();

    // Calculate the thread stuff.

    // let mut count = if std::cmp::max(chunks_per_thread - 1, 0)  * (64 as i32) + i >= total_vertex_count { std::cmp::max(chunks_per_thread - 1, 0) } else { chunks_per_thread };
    
 
    // Create number 1.
    // local_index == thread_id
    // total_vertex_count == 400
    // start position of number 1.
    // col.

    //var number = 0u;
    var number = -12347.0;

    ////////////////////// log_vec3_f32 test ////////////////////////////
    
    var col = rgba_u32(255u, 0u, 222u, 255u);    
    var base_pos = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    
    log_vec3_f32(
        vec3<f32>(-9999.0,-1232.0093, 5.0),      
        4u,                                  
        local_index,                             
        &base_pos,                          
        800u,                              
        col,
        0.5,                                   
    );

    ////////////////////// log_vec4_f32 test ////////////////////////////
    
    col = rgba_u32(0u, 0u, 252u, 255u);    
    base_pos = vec4<f32>(0.0, 2.0, 0.0, 1.0);
    
    log_vec4_f32(
        vec4<f32>(89.893, -9999.0,-1232.0093, 5.0),      
        4u,                                  
        local_index,                             
        &base_pos,                          
        800u,                              
        col,
        1.0,                                   
    );
    
    ////////////////////// log_u32_b2 test ////////////////////////////

    let numb = 0x7463u;
    // 111010001100011

    base_pos = vec4<f32>(0.0, 4.0, 0.0, 1.0);
    log_u32_b2(numb, 32u, local_index, &base_pos, 800u, col, 2.0);

    /////////////////////////////////////////////////////////////////////

    //log_float(-99.0045, 5u, local_index, &base_pos, 400u, col);

    // for (var i: i32 = 0; i < 200 ; i = i + 1) {

    //     var base_pos = vec4<f32>(0.0, f32(i), 0.0, 1.0); 
    // 	let max_vertex_count = 5000u;
    // 	let offset = udiv_up_32(max_vertex_count, 64u);
    // 	let col = rgba_u32(255u, 0u, 0u, 255u);
    // 	let start_position = vec3<f32>(0.1, 0.1, 0.1);

    // 	let dist = min(max(1.0, distance(camera.pos.xyz, base_pos.xyz)), 255.0);
    // 	let total_vertex_count = u32(f32(max_vertex_count) / f32(dist));

    //     log_float(number, 3u, local_index, &base_pos, total_vertex_count, col);
    //     number = number + 12347.0;
    //     base_pos = base_pos + vec4<f32>(0.0, 1.0, 0.0, 0.0);
    // }
}
