/// Apply triangle mesh to fmm data. The interface creation.

// FMM tags
// 0 :: Far
// 1 :: Band
// 2 :: Band New
// 3 :: Known
// 4 :: Outside

let FAR      = 0u;
let BAND_NEW = 1u;
let BAND     = 2u;
let KNOWN    = 3u;
let OUTSIDE  = 4u;

let THREAD_COUNT: u32 = 64u;

struct AABB {
    min: vec4<f32>, 
    max: vec4<f32>, 
};

struct AABB_Uvec3 {
    min: vec3<u32>, 
    max: vec3<u32>, 
};

struct Arrow {
    start_pos: vec4<f32>,
    end_pos:   vec4<f32>,
    color: u32,
    size:  f32,
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

struct FmmParams {
    fmm_global_dimension: vec3<u32>, 
    generation_method: u32, // 0 -> generate speed information, 1 -> generate fmm interface
    fmm_inner_dimension: vec3<u32>, 
    triangle_count: u32,
};

struct Char {
    start_pos: vec4<f32>,
    value: vec4<f32>,
    font_size: f32,
    vec_dim_count: u32, // 1 => f32, 2 => vec3<f32>, 3 => vec3<f32>, 4 => vec4<f32>
    color: u32,
    z_offset: f32,
};

struct ModF {
    fract: f32,
    whole: f32,
};

struct FmmCell {
    tag: u32,
    value: f32,
    queue_value: u32,
};

struct FmmParams {
    fmm_global_dimension: vec3<u32>, 
    generation_method: u32, // 0 -> generate speed information, 1 -> generate fmm interface
    fmm_inner_dimension: vec3<u32>, 
    triangle_count: u32,
};

@group(0) @binding(0) var<uniform> fmm_params: FmmParams;
@group(0) @binding(1) var<storage, read_write> fmm_data: array<FmmCell>;
@group(0) @binding(2) var<storage, read_write> triangle_mesh_in: array<Triangle>;
@group(0) @binding(3) var<storage, read_write> isotropic_data: array<f32>;

//debug @group(0)
//debug @binding(4)
//debug var<storage, read_write> counter: array<atomic<u32>>;
//debug 
//debug @group(0)
//debug @binding(5)
//debug var<storage,read_write> output_char: array<Char>;
//debug 
//debug @group(0)
//debug @binding(6)
//debug var<storage,read_write> output_arrow: array<Arrow>;
//debug 
//debug @group(0)
//debug @binding(7)
//debug var<storage,read_write> output_aabb: array<AABB>;
//debug 
//debug @group(0)
//debug @binding(8)
//debug var<storage,read_write> output_aabb_wire: array<AABB>;

// Do we need this?
//////// ModF ////////

fn myTruncate(f: f32) -> f32 {
    return select(f32( i32( floor(f) ) ), f32( i32( ceil(f) ) ), f < 0.0); 
}

fn my_modf(f: f32) -> ModF {
    let iptr = myTruncate(f);
    let fptr = f - iptr;
    return ModF (
        select(fptr, (-1.0)*fptr, f < 0.0),
        iptr
    );
}

///////////////////////////
////// Grid curve    //////
///////////////////////////

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

//////// Common function  ////////

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

//// Intersection things. ////

// struct Triangle ?
fn closest_point_to_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {

    let ab = b - a;
    let ac = c - a;
    let bc = c - b;

    let d1 = dot(b-a, p-a);
    let d2 = dot(c-a, p-a);
    let d3 = dot(b-a, p-b);
    let d4 = dot(c-a, p-b);
    let d5 = dot(b-a, p-c);
    let d6 = dot(c-a, p-c);

    let va = d3*d6 - d5*d4;
    let vb = d5*d2 - d1*d6;
    let vc = d1*d4 - d3*d2;

    let snom = d1;
    let sdenom = -d3;
    let tnom = d2;
    let tdenom = -d6;
    let unom = d4 - d3;
    let udenom = d5 - d6;

    // TODO: optimize.

    if (snom <= 0.0 && tnom <= 0.0) { return a; }
    if (sdenom <= 0.0 && unom <= 0.0) { return b; }
    if (tdenom <= 0.0 && udenom <= 0.0) { return c; }

    let n = cross(b-a, c-a);

    if (vc <= 0.0 && snom >= 0.0 && sdenom >= 0.0) { return a + snom / (snom + sdenom) * ab; }
    if (va <= 0.0 && unom >= 0.0 && udenom >= 0.0) { return b + unom / (unom + udenom) * bc; }
    if (vb <= 0.0 && tnom >= 0.0 && tdenom >= 0.0) { return a + tnom / (tnom + tdenom) * ac; }

    let u = va / (va + vb + vc);
    let v = vb / (va + vb + vc);
    let w = 1.0 - u - v;
    return u * a + v * b + w * c;
}

// Calculate the aabb in fmm global dimensions for triangle.
fn triangle_to_aabb(tr: Triangle) -> AABB_Uvec3 {

    // Calculate aabb.
    let aabb_min_x = min(tr.c.v.x, min(tr.a.v.x, tr.b.v.x));
    let aabb_min_y = min(tr.c.v.y, min(tr.a.v.y, tr.b.v.y));
    let aabb_min_z = min(tr.c.v.z, min(tr.a.v.z, tr.b.v.z));
    
    let aabb_max_x = max(tr.c.v.x, max(tr.a.v.x, tr.b.v.x));
    let aabb_max_y = max(tr.c.v.y, max(tr.a.v.y, tr.b.v.y));
    let aabb_max_z = max(tr.c.v.z, max(tr.a.v.z, tr.b.v.z));

    // Calculate extended aabb.
    let min_x_expanded = u32(min(f32(fmm_params.fmm_global_dimension.x - 1u), max(0.0, floor(aabb_min_x))));
    let min_y_expanded = u32(min(f32(fmm_params.fmm_global_dimension.y - 1u), max(0.0, floor(aabb_min_y))));
    let min_z_expanded = u32(min(f32(fmm_params.fmm_global_dimension.z - 1u), max(0.0, floor(aabb_min_z)))); 

    let max_x_expanded = u32(min(f32(fmm_params.fmm_global_dimension.x), max(0.0, ceil(aabb_max_x + 4.0))));
    let max_y_expanded = u32(min(f32(fmm_params.fmm_global_dimension.y), max(0.0, ceil(aabb_max_y + 4.0))));
    let max_z_expanded = u32(min(f32(fmm_params.fmm_global_dimension.z), max(0.0, ceil(aabb_max_z + 4.0))));

    return AABB_Uvec3(vec3<u32>(min_x_expanded, min_y_expanded, min_z_expanded),
                      vec3<u32>(max_x_expanded, max_y_expanded, max_z_expanded)
    );
} 

fn update_fmm_interface(aabb: AABB_Uvec3, tr: Triangle, thread_index: u32) {

    let dim_x = aabb.max.x - aabb.min.x;
    let dim_y = aabb.max.y - aabb.min.y;
    let dim_z = aabb.max.z - aabb.min.z;

    let number_of_cubes = dim_x * dim_y * dim_z;

    // Calculate the smallest distances between triangle and the cube grid points.
    for (var i: i32 = 0 ; i < i32(number_of_cubes) ; i = i + 1) {

        // TODO: Filter those cubes that are close enought the triangle.
        // Prefix sum?

        // The global index for cube.
        let base_coordinate = (index_to_uvec3(u32(i), dim_x, dim_y) + aabb.min) * 4u;
        let base_index = encode3Dmorton32(base_coordinate.x, base_coordinate.y, base_coordinate.z);
        let actual_index = base_index + thread_index; 
        let actual_coordinate = decode3Dmorton32(actual_index);

        let cp = closest_point_to_triangle(vec3<f32>(actual_coordinate) * 0.25, tr.a.v.xyz, tr.b.v.xyz, tr.c.v.xyz);
        let dist = distance(cp, vec3<f32>(actual_coordinate) * 0.25);

    	let color = f32(rgba_u32(222u, 200u, 150u, 255u));

        // Update the initial fmm interface (KNONW cell).
        if (fmm_params.generation_method == 1u) {

            // Load the fmm cell.
            let cell = fmm_data[actual_index];

                if (dist < 0.25 && dist < cell.value) {

                   // Is this safe. Do we need atomic operations?
                   fmm_data[actual_index] = FmmCell(KNOWN, dist, 0u);
                }
        }

        // Update speed data.
        if (fmm_params.generation_method == 0u) {

            // Load speed information.
            let speed = isotropic_data[actual_index];

            // Update the isotropic speed data. TODO: repalce 15.0 with some actual speed data. 
            if (dist < 0.25 && 15.0 / max(0.05, dist) > speed) {

               // Is this safe. Do we need atomic operations?
               isotropic_data[actual_index] = 15.0 / max(0.05, dist);
            }
        }
    }
}

@stage(compute)
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    // Print the global aabb.
    let color = f32(rgba_u32(222u, 0u, 150u, 255u));

    // One triangle per dispatch.
    let tr = triangle_mesh_in[work_group_id.x];
    let tr_aabb = triangle_to_aabb(tr);
    update_fmm_interface(tr_aabb, tr, local_index);

    // else if (fmm_params.visualize == 1u) {

    //     if (global_id.x == 0u) {
    //         output_aabb_wire[global_id.x] =  
    //               AABB (
    //                   vec4<f32>(0.0, 0.0, 0.0, color),
    //                   vec4<f32>(vec3<f32>(fmm_params.fmm_global_dimension), 0.1)
    //               );
    //         atomicAdd(&counter[3], 1u);
    //     }

    //     let cell = fmm_data[global_id.x];
    //     let position = vec3<f32>(decode3Dmorton32(global_id.x)) * 0.25;

    //     if (cell.tag == FAR) { return; }

    //     var col = select(f32(rgba_u32(0u  , 255u, 0u, 255u)),
    //                      f32(rgba_u32(0u, 255u,   0u, 255u)),
    //                      cell.tag == KNOWN); 
    //     output_aabb[atomicAdd(&counter[2], 1u)] =
    //           AABB (
    //               vec4<f32>(position - vec3<f32>(0.008), col),
    //               vec4<f32>(position + vec3<f32>(0.008), 0.0),
    //           );
    // }

    // Initalize fmm data to far points.  
    // fmm_data[global_id.x] = FmmCell (FAR, 10000000.0);

    // Triangle to fmm interface.

    // let global_coordinate = vec3<f32>(decode3Dmorton32(work_group_id.x));
    // let local_coordinate  = vec3<f32>(decode3Dmorton32(local_index)) * 0.25;
    

    // output_aabb[atomicAdd(&counter[2], 1u)] = 
    //       AABB (
    //           vec4<f32>(global_coordinate + local_coordinate - vec3<f32>(0.008), color),
    //           vec4<f32>(global_coordinate + local_coordinate + vec3<f32>(0.008), 0.0),
    //       );
}
