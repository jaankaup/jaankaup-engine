///
/// Fast marching method kernel.
///

// Fmm tags.
let FAR      = 0u;
let BAND     = 1u;
let KNOWN    = 2u;

/// A struct for one fast marching methow cell.
struct FmmCellPc {
    tag: atomic<u32>,
    value: f32,
    color: u32,
};

/// A struct that holds information for a single computational domain area.
struct FmmBlock {
    index: u32,
    band_point_count: u32,
};
@group(0) @binding(0) var<uniform> fmm_params: FmmParams;
@group(0) @binding(1) var<storage, read_write> fmm_data: array<FmmCellPc>;

@group(0) @binding(2)
var<storage, read_write> temp_prefix_sum: array<u32>;

// debug.
@group(0) @binding(2) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> output_char: array<Char>;
@group(0) @binding(4) var<storage,read_write> output_arrow: array<Arrow>;
@group(0) @binding(5) var<storage,read_write> output_aabb: array<AABB>;
@group(0) @binding(6) var<storage,read_write> output_aabb_wire: array<AABB>;

@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

}
