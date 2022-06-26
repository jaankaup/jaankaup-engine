struct FmmParams {
    global_dimension: vec3<u32>,
    local_dimension: vec3<u32>,
};

struct FmmCellPc {
    tag: u32,
    //value: f32,
    //value: atomic<i32>,
    value: i32,
    color: u32,
}

let FAR      = 0u;
let BAND_NEW = 1u;
let BAND     = 2u;
let KNOWN    = 3u;
let OUTSIDE  = 4u;

@group(0) @binding(0) var<uniform> fmm_params: FmmParams;
@group(0) @binding(1) var<storage, read_write> fmm_data: array<FmmCellPc>;
@group(0) @binding(2) var<storage, read_write> fmm_temp_data: array<f32>;

@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) work_group_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    let total_count = fmm_params.local_dimension.x *
                      fmm_params.local_dimension.y *
                      fmm_params.local_dimension.z *
                      fmm_params.global_dimension.x *
                      fmm_params.global_dimension.y *
                      fmm_params.global_dimension.z;

    if (global_id.x >= total_count) { return; }

    var cell = fmm_data[global_id.x];
    //++ let the_sign = select(1.0, -1.0, (cell.value >> 31u) != 0u); 
    // let the_sign = select(1.0, -1.0, (cell.value & (1u << 31u)) != 0u); 

    //mc // Convert value back to f32.
    //mc if (cell.tag == KNOWN) {
    //mc     cell.value = bitcast<u32>(f32(cell.value) * (-1.0) * 0.0001);
    //mc }
    //mc else {
    //mc     cell.value = bitcast<u32>(1.0);
    //mc }

    // Convert value back to f32.
    if (cell.tag == KNOWN) {
        // cell.value = bitcast<i32>(-0.005);
        //cell.value = fmm_temp_data[global_id.x];
        fmm_data[global_id.x].value = bitcast<i32>(f32(cell.value) * 0.000001); 
        //fmm_data[global_id.x].value = bitcast<i32>(-0.666); 
        //cell.value = bitcast<u32>(f32(cell.value & (!(1u << 31u))) * 0.000001 * the_sign);
    }
    // else {
    //     cell.value = 100000.0;
    // }

    // atomicExchange(&fmm_data[global_id.x].value, bitcast<i32>(0.666)); //cell.value; 
}
