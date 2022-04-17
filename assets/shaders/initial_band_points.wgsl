/// Create the initial band points and update fmm blocks.

struct AABB {
    min: vec4<f32>, 
    max: vec4<f32>, 
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

struct Arrow {
    start_pos: vec4<f32>,
    end_pos:   vec4<f32>,
    color: u32,
    size:  f32,
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

struct FmmCell {
    tag: u32,
    value: f32,
    queue_value: atomic<u32>,
};

struct FmmCellSync {
    tag: u32,
    value: f32,
    queue_value: u32,
};

struct FmmBlock {
    index: u32,
    band_points_count: u32,
};

let THREAD_COUNT: u32 = 1024u;
let NEIGHBORS: u32 = 6144u;

//var<workgroup> neighbor_cells: array<FmmCell, THREAD_COUNT * 6u>;
var<workgroup> neighbor_cell_indices: array<FmmCell, NEIGHBORS>;
var<workgroup> neighbor_cells_counter: atomic<u32>;
var<workgroup> known_points_counter: atomic<u32>;

//   0       1     2      3     4      5
// +------+-----+-----+------+-----+------+
// | left |right| top |bottom| far | near |
// |      |     |     |      |     |      |
// +------+-----+-----+------+-----+------+

// var<private> neighbor_cells: array<FmmCell, 6>;
// var<private> neighbor_cell_indices: array<u32, 6>;

@group(0) @binding(0) var<storage, read_write> fmm_data: array<FmmCell>;
@group(0) @binding(1) var<storage, read_write> fmm_blocks: array<FmmBlock>;
@group(0) @binding(2) var<storage, read_write> synchronization_data: array<FmmCellSync>;
@group(0) @binding(3) var<storage, read_write> sync_point_counter: array<atomic<u32>>;

// Debugging.
@group(0) @binding(4) var<storage,read_write> counter: array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> output_char: array<Char>;
@group(0) @binding(6) var<storage,read_write> output_arrow: array<Arrow>;
@group(0) @binding(7) var<storage,read_write> output_aabb: array<AABB>;
@group(0) @binding(8) var<storage,read_write> output_aabb_wire: array<AABB>;

// Encode "rgba" to u32.
fn rgba_u32(r: u32, g: u32, b: u32, a: u32) -> u32 {
  return (r << 24u) | (g << 16u) | (b  << 8u) | a;
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

fn isInside(coord: ptr<function, vec3<i32>>) -> bool {
    return ((*coord).x >= 0 && (*coord).x < 64) &&
           ((*coord).y >= 0 && (*coord).y < 64) &&
           ((*coord).z >= 0 && (*coord).z < 64); 
}

fn isInside2(x: i32, y: i32, z: i32) -> bool {
    return (x >= 0 && x < 64) &&
           (y >= 0 && y < 64) &&
           (z >= 0 && z < 64); 
}

fn load_neighbors(coord: vec3<u32>) {

    // TODO: Do not load from outside the computational domain.
    // TODO: Implement cached loads/saves. 
   
    var neighbor_0_coord = vec3<i32>(coord) + vec3<i32>(1, 0, 0);
    var neighbor_1_coord = vec3<i32>(coord) - vec3<i32>(1, 0, 0);
    var neighbor_2_coord = vec3<i32>(coord) + vec3<i32>(0, 1, 0);
    var neighbor_3_coord = vec3<i32>(coord) - vec3<i32>(0, 1, 0);
    var neighbor_4_coord = vec3<i32>(coord) + vec3<i32>(0, 0, 1);
    var neighbor_5_coord = vec3<i32>(coord) - vec3<i32>(0, 0, 1);

    var neighbor_0_coord_u32 = coord + vec3<u32>(1u, 0u, 0u);
    var neighbor_1_coord_u32 = coord - vec3<u32>(1u, 0u, 0u);
    var neighbor_2_coord_u32 = coord + vec3<u32>(0u, 1u, 0u);
    var neighbor_3_coord_u32 = coord - vec3<u32>(0u, 1u, 0u);
    var neighbor_4_coord_u32 = coord + vec3<u32>(0u, 0u, 1u);
    var neighbor_5_coord_u32 = coord - vec3<u32>(0u, 0u, 1u);

    var cell_index0 = encode3Dmorton32(neighbor_0_coord_u32.x, neighbor_0_coord_u32.y, neighbor_0_coord_u32.z);
    var cell_index1 = encode3Dmorton32(neighbor_1_coord_u32.x, neighbor_1_coord_u32.y, neighbor_1_coord_u32.z);
    var cell_index2 = encode3Dmorton32(neighbor_2_coord_u32.x, neighbor_2_coord_u32.y, neighbor_2_coord_u32.z);
    var cell_index3 = encode3Dmorton32(neighbor_3_coord_u32.x, neighbor_3_coord_u32.y, neighbor_3_coord_u32.z);
    var cell_index4 = encode3Dmorton32(neighbor_4_coord_u32.x, neighbor_4_coord_u32.y, neighbor_4_coord_u32.z);
    var cell_index5 = encode3Dmorton32(neighbor_5_coord_u32.x, neighbor_5_coord_u32.y, neighbor_5_coord_u32.z);

    // TODO: fmm_data[262144] == Outside Cell
    cell_index0 = select(262144u, cell_index0, cell_index0 < 262144u);
    cell_index1 = select(262144u, cell_index1, cell_index1 < 262144u);
    cell_index2 = select(262144u, cell_index2, cell_index2 < 262144u);
    cell_index3 = select(262144u, cell_index3, cell_index3 < 262144u);
    cell_index4 = select(262144u, cell_index4, cell_index4 < 262144u);
    cell_index5 = select(262144u, cell_index5, cell_index5 < 262144u);

    var queue_val0 = atomicAdd(&fmm_data[cell_index0].queue_value, 1u);
    var queue_val1 = atomicAdd(&fmm_data[cell_index1].queue_value, 1u);
    var queue_val2 = atomicAdd(&fmm_data[cell_index2].queue_value, 1u);
    var queue_val3 = atomicAdd(&fmm_data[cell_index3].queue_value, 1u);
    var queue_val4 = atomicAdd(&fmm_data[cell_index4].queue_value, 1u);
    var queue_val5 = atomicAdd(&fmm_data[cell_index5].queue_value, 1u);

    // TODO: do not access the data outside buffer range.
    var neighbor_cell0 = fmm_data[cell_index0];
    var neighbor_cell1 = fmm_data[cell_index1];
    var neighbor_cell2 = fmm_data[cell_index2];
    var neighbor_cell3 = fmm_data[cell_index3];
    var neighbor_cell4 = fmm_data[cell_index4];
    var neighbor_cell5 = fmm_data[cell_index5];

    var inside0 = isInside(&neighbor_0_coord);
    var inside1 = isInside(&neighbor_1_coord);
    var inside2 = isInside(&neighbor_2_coord);
    var inside3 = isInside(&neighbor_3_coord);
    var inside4 = isInside(&neighbor_4_coord);
    var inside5 = isInside(&neighbor_5_coord);
    let index = atomicAdd(&sync_point_counter[0], 1u);

    if (inside0 && neighbor_cell0.tag != KNOWN && queue_val0 == 0u) {
        let index = atomicAdd(&sync_point_counter[0], 1u);
        synchronization_data[index] = FmmCellSync(BAND, 100000.0, 0u);
    }
    if (inside0 && neighbor_cell1.tag != KNOWN && queue_val1 == 0u) {
        let index = atomicAdd(&sync_point_counter[0], 1u);
        synchronization_data[index] = FmmCellSync(BAND, 100000.0, 0u);
    }
    if (inside0 && neighbor_cell2.tag != KNOWN && queue_val2 == 0u) {
        let index = atomicAdd(&sync_point_counter[0], 1u);
        synchronization_data[index] = FmmCellSync(BAND, 100000.0, 0u);
    }
    if (inside0 && neighbor_cell3.tag != KNOWN && queue_val3 == 0u) {
        let index = atomicAdd(&sync_point_counter[0], 1u);
        synchronization_data[index] = FmmCellSync(BAND, 100000.0, 0u);
    }
    if (inside0 && neighbor_cell4.tag != KNOWN && queue_val4 == 0u) {
        let index = atomicAdd(&sync_point_counter[0], 1u);
        synchronization_data[index] = FmmCellSync(BAND, 100000.0, 0u);
    }
    if (inside0 && neighbor_cell5.tag != KNOWN && queue_val5 == 0u) {
        let index = atomicAdd(&sync_point_counter[0], 1u);
        synchronization_data[index] = FmmCellSync(BAND, 100000.0, 0u);
    }
}

// fn collect_known_points(thread_index: u32) {
//    let fc = fmm_data[thread_index];
// }



@stage(compute)
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    var this_coordinate_u32 = decode3Dmorton32(global_id.x);
    var this_coordinate_i32 = vec3<i32>(this_coordinate_u32);
    var this_cell = fmm_data[global_id.x];
    if (this_cell.tag == KNOWN) {
        load_neighbors(this_coordinate_u32);
    }
}

@stage(compute)
@workgroup_size(1024,1,1)
fn sync_and_calculate_all_bands(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    var this_coordinate_u32 = decode3Dmorton32(global_id.x);
    var this_coordinate_i32 = vec3<i32>(this_coordinate_u32);
    var this_cell = fmm_data[global_id.x];
    if (this_cell.tag == KNOWN) {
        load_neighbors(this_coordinate_u32);
    }
}
