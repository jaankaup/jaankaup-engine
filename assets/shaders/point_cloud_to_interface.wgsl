/// Create an initial fmm interface from point data. 


struct VVVC {
    position: [f32; 3],
    color: u32,
};

struct FmmParams {
    global_dimension: vec3<u32>,
    local_dimension: vec3<u32>,
};

struct PointCloudParams {
    point_count: u32,
};

struct FmmCellPc {
    tag: u32,
    value: f32,
    color: u32,
}

@group(0) @binding(0) var<uniform> fmm_params: FmmParams;
@group(0) @binding(1) var<uniform> point_cloud_params: PointCloudParams;
@group(0) @binding(2) var<storage, read_write> fmm_data: array<FmmCell>;
@group(0) @binding(3) var<storage, read_write> point_data: array<VVVC>;

// Debug.
@group(0) @binding(4) var<storage,read_write> counter: array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> output_char: array<Char>;
@group(0) @binding(6) var<storage,read_write> output_arrow: array<Arrow>;
@group(0) @binding(7) var<storage,read_write> output_aabb: array<AABB>;
@group(0) @binding(8) var<storage,read_write> output_aabb_wire: array<AABB>;

@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {


    if (global_id.x >= point_cloud_params.point_count) { return; }

    let point = point_cloud_params[global_id.x];
    let nearest_cell = vec3<u32>(round(point.x), round(point.y), round(point.z));
    let distance = distance(point, nearest_cell);
}
