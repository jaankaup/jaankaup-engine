struct PointCloudParams {
    min_point: vec3<f32>,
    point_count: u32,
    max_point: vec3<f32>,
    pc_scale_factor: f32,
    thread_group_number: u32, // Not used. Only for debugging purposes.
    show_numbers: u32, // Not used. Only for debugging purposes.
};

struct FmmParams {
    global_dimension: vec3<u32>,
    visualization_method: u32,
    local_dimension: vec3<u32>,
    future_usage: u32,
};

struct TempPoint {
    original_mem_location: u32,
    block_index: u32,
};

@group(0) @binding(0) var<uniform> fmm_params: FmmParams;
@group(0) @binding(1) var<uniform> point_cloud_params: PointCloudParams;
@group(0) @binding(2) var<storage, read_write> fmm_data: array<FmmCellPc>;
@group(0) @binding(3) var<storage, read_write> point_data: array<VVVC>;

fn isInside(coord: vec3<i32>) -> bool {
    return (coord.x >= 0 && coord.x < i32(fmm_params.local_dimension.x * fmm_params.global_dimension.x)) &&
           (coord.y >= 0 && coord.y < i32(fmm_params.local_dimension.y * fmm_params.global_dimension.y)) &&
           (coord.z >= 0 && coord.z < i32(fmm_params.local_dimension.z * fmm_params.global_dimension.z)); 
}

@compute
@workgroup_size(256,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    if (global_id.x < point_cloud_params.point_count) {

        var p = point_data[global_id.x];

	let radius = 0.1;

        let p0 = vec3<i32>(floor(p.position)); 
        let p1 = p0 + vec3(1, 0, 0); 
        let p2 = p0 + vec3(0, 1, 0); 
        let p3 = p0 + vec3(1, 1, 0); 
        let p4 = p0 + vec3(0, 0, 1); 
        let p5 = p0 + vec3(1, 0, 1); 
        let p6 = p0 + vec3(0, 1, 1); 
        let p7 = p0 + vec3(1, 1, 1); 
        
        // Check if cell is inside the computational domain.
        let inside = isInside(nearest_cell);

        if (isInside(p0) && distance(p0, p.position) < radius {
            let memory_index_p0 = get_cell_mem_location(vec3<u32>(p0));
        }
            
            let memory_index = get_cell_mem_location(vec3<u32>(nearest_cell));
            atomicMin(&fmm_data[memory_index].value, dist_to_i32);
            // atomicMin(&fmm_data[memory_index].value, dist_to_i32);
        }
    }
}
