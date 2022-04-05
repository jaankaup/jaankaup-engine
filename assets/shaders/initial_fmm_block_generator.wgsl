struct AABB {
    min: vec4<f32>, 
    max: vec4<f32>, 
};

struct Arrow {
    start_pos: vec4<f32>,
    end_pos:   vec4<f32>,
    color: u32,
    size:  f32,
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

struct FmmCell {
    tag: u32,
    value: f32,
};

struct FmmBlock {
    index: u32,
    band_points_count: u32,
};

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

let THREAD_COUNT = 64u;

@group(0) @binding(0) var<storage, read_write> fmm_data: array<FmmCell>;
@group(0) @binding(1) var<storage, read_write> fmm_blocks: array<FmmBlock>;

// Debugging.
@group(0) @binding(2) var<storage,read_write> counter: array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write>  output_char: array<Char>;
@group(0) @binding(4) var<storage,read_write>  output_arrow: array<Arrow>;
@group(0) @binding(5) var<storage,read_write>  output_aabb: array<AABB>;
@group(0) @binding(6) var<storage,read_write>  output_aabb_wire: array<AABB>;

struct PrivateParams {
    index: u32,
    g_index: u32,
};

var<private> private_params: PrivateParams; 
var<workgroup> temp_fmm_cells: array<FmmCell, THREAD_COUNT>; 

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

fn reduce() {

    for (var i: u32 = THREAD_COUNT / 2u; i > 0u ; i = i >> 1u) {

        workgroupBarrier();
        if (private_params.index < i) {

            // Load data a.
            let ptr_a = &temp_fmm_cells[private_params.index]; 

            // Load data b.
            let ptr_b = &temp_fmm_cells[private_params.index + i]; 

            if (((*ptr_a).tag == BAND && (*ptr_b).tag == BAND && (*ptr_a).value > (*ptr_b).value) || ((*ptr_a).tag != BAND && (*ptr_b).tag == BAND)) {
                *ptr_a = *ptr_b;
            }
            // Not working.
            // *ptr_a = select(*ptr_a,
            //                 *ptr_b,
            //                 ((*ptr_a).tag == BAND && (*ptr_b).tag == BAND && (*ptr_a).value > (*ptr_b).value) || ((*ptr_a).tag != BAND && (*ptr_b).tag == BAND));
        }
    }
}

fn load_chunk_to_workgroup() {
    fmm_data[private_params.index * private_params.g_index] = temp_fmm_cells[private_params.index]; 
}

@stage(compute)
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {
}
