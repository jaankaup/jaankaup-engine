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

@stage(compute)
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {
}
