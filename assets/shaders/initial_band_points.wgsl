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
    mem_location: u32,
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

var<private> neighbor_cells: array<FmmCellSync, 6>;
// var<private> neighbor_cell_indices: array<u32, 6>;

@group(0) @binding(0) var<storage, read_write> fmm_data: array<FmmCell>;
@group(0) @binding(1) var<storage, read_write> fmm_blocks: array<FmmBlock>;
@group(0) @binding(2) var<storage, read_write> synchronization_data: array<FmmCellSync>;
@group(0) @binding(3) var<storage, read_write> isotropic_data: array<f32>;
@group(0) @binding(4) var<storage, read_write> sync_point_counter: array<atomic<u32>>;

// Debugging.
@group(0) @binding(5) var<storage,read_write> counter: array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> output_char: array<Char>;
@group(0) @binding(7) var<storage,read_write> output_arrow: array<Arrow>;
@group(0) @binding(8) var<storage,read_write> output_aabb: array<AABB>;
@group(0) @binding(9) var<storage,read_write> output_aabb_wire: array<AABB>;

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

    neighbor_cells[0] = FmmCellSync(OUTSIDE, 100000.0, 0u, 262144u);
    neighbor_cells[1] = FmmCellSync(OUTSIDE, 100000.0, 0u, 262144u);
    neighbor_cells[2] = FmmCellSync(OUTSIDE, 100000.0, 0u, 262144u);
    neighbor_cells[3] = FmmCellSync(OUTSIDE, 100000.0, 0u, 262144u);
    neighbor_cells[4] = FmmCellSync(OUTSIDE, 100000.0, 0u, 262144u);
    neighbor_cells[5] = FmmCellSync(OUTSIDE, 100000.0, 0u, 262144u);

    if (inside0) {
        neighbor_cells[0] = FmmCellSync(neighbor_cell0.tag, neighbor_cell0.value, queue_val0, cell_index0);
    } 
    if (inside1) {
        neighbor_cells[1] = FmmCellSync(neighbor_cell1.tag, neighbor_cell1.value, queue_val1, cell_index1);
    } 
    if (inside2) {
        neighbor_cells[2] = FmmCellSync(neighbor_cell2.tag, neighbor_cell2.value, queue_val2, cell_index2);
    } 
    if (inside3) {
        neighbor_cells[3] = FmmCellSync(neighbor_cell3.tag, neighbor_cell3.value, queue_val3, cell_index3);
    } 
    if (inside4) {
        neighbor_cells[4] = FmmCellSync(neighbor_cell4.tag, neighbor_cell4.value, queue_val4, cell_index4);
    } 
    if (inside5) {
        neighbor_cells[5] = FmmCellSync(neighbor_cell5.tag, neighbor_cell5.value, queue_val5, cell_index5);
    } 

    // neighbor_cells[1] = if (inside1) {
    //     neighbor_cells[1] = FmmCellSync(neighbor_cell1.tag, neighbor_cell1.value, queue_val1, cell_index1);
    // ); 
    // neighbor_cells[2] = if (inside2) {
    //     neighbor_cells[2] = FmmCellSync(neighbor_cell2.tag, neighbor_cell2.value, queue_val2, cell_index2);
    // ); 
    // neighbor_cells[3] = if (inside3) {
    //     neighbor_cells[3] = FmmCellSync(neighbor_cell3.tag, neighbor_cell3.value, queue_val3, cell_index3);
    // ); 
    // neighbor_cells[4] = if (inside4) {
    //     neighbor_cells[4] = FmmCellSync(neighbor_cell4.tag, neighbor_cell4.value, queue_val4, cell_index4);
    // ); 
    // neighbor_cells[5] = if (inside5) {
    //     neighbor_cells[5] = FmmCellSync(neighbor_cell5.tag, neighbor_cell5.value, queue_val5, cell_index5);
    // ); 

    // neighbor_cells[1] = neighbor_cell1; 
    // neighbor_cells[2] = neighbor_cell2; 
    // neighbor_cells[3] = neighbor_cell3; 
    // neighbor_cells[4] = neighbor_cell4; 
    // neighbor_cells[5] = neighbor_cell5; 

    // if (inside0 && neighbor_cell0.tag != KNOWN && queue_val0 == 0u) {
    //     let index = atomicAdd(&sync_point_counter[0], 1u);
    //     synchronization_data[index] = FmmCellSync(BAND, 100000.0, 0u, cell_index0);
    // }
    // if (inside0 && neighbor_cell1.tag != KNOWN && queue_val1 == 0u) {
    //     let index = atomicAdd(&sync_point_counter[0], 1u);
    //     synchronization_data[index] = FmmCellSync(BAND, 100000.0, 0u, cell_index1);
    // }
    // if (inside0 && neighbor_cell2.tag != KNOWN && queue_val2 == 0u) {
    //     let index = atomicAdd(&sync_point_counter[0], 1u);
    //     synchronization_data[index] = FmmCellSync(BAND, 100000.0, 0u, cell_index2);
    // }
    // if (inside0 && neighbor_cell3.tag != KNOWN && queue_val3 == 0u) {
    //     let index = atomicAdd(&sync_point_counter[0], 1u);
    //     synchronization_data[index] = FmmCellSync(BAND, 100000.0, 0u, cell_index3);
    // }
    // if (inside0 && neighbor_cell4.tag != KNOWN && queue_val4 == 0u) {
    //     let index = atomicAdd(&sync_point_counter[0], 1u);
    //     synchronization_data[index] = FmmCellSync(BAND, 100000.0, 0u, cell_index4);
    // }
    // if (inside0 && neighbor_cell5.tag != KNOWN && queue_val5 == 0u) {
    //     let index = atomicAdd(&sync_point_counter[0], 1u);
    //     synchronization_data[index] = FmmCellSync(BAND, 100000.0, 0u, cell_index5);
    // }
}

fn add_neighbors_to_sync() {

    for (var i: i32 = 0 ; i < 6 ; i = i + 1) {
        if (neighbor_cells[i].tag != OUTSIDE && neighbor_cells[i].tag != KNOWN && neighbor_cells[i].queue_value == 0u) {
            let index = atomicAdd(&sync_point_counter[0], 1u);
            synchronization_data[index] = neighbor_cells[index]; 
            // var temp_cell: FmmCell;
            // temp_cell.tag = BAND;
            // temp_cell.value = 100000.0;
            // temp_cell.queue_value = 0u;
            // fmm_data[neighbor_cells[i].mem_location] = temp_cell;
        }
    }
}

fn add_neighbors_to_band() {

    for (var i: i32 = 0 ; i < 6 ; i = i + 1) {
        if (neighbor_cells[i].tag != OUTSIDE && neighbor_cells[i].tag != KNOWN && neighbor_cells[i].queue_value == 0u) {
            var temp_cell: FmmCell;
            temp_cell.tag = BAND;
            temp_cell.value = 100000.0;
            temp_cell.queue_value = 0u;
            fmm_data[neighbor_cells[i].mem_location] = temp_cell;
        }
    }
}

// fn collect_known_points(thread_index: u32) {
//    let fc = fmm_data[thread_index];
// }



/////////////
//   FMM   // 
/////////////

fn solve_quadratic_2(speed: f32) -> f32 {

    var phis: array<f32, 6> = array<f32, 6>(
                                  select(1000000.0, neighbor_cells[0].value, neighbor_cells[0].tag == KNOWN),
                                  select(1000000.0, neighbor_cells[1].value, neighbor_cells[1].tag == KNOWN),
                                  select(1000000.0, neighbor_cells[2].value, neighbor_cells[2].tag == KNOWN),
                                  select(1000000.0, neighbor_cells[3].value, neighbor_cells[3].tag == KNOWN),
                                  select(1000000.0, neighbor_cells[4].value, neighbor_cells[4].tag == KNOWN),
                                  select(1000000.0, neighbor_cells[5].value, neighbor_cells[5].tag == KNOWN) 
    );

    var p = vec3<f32>(min(phis[0], phis[1]), min(phis[2], phis[3]), min(phis[4], phis[5]));
    var p_sorted: vec3<f32>;

    var tmp: f32;

    if p[1] < p[0] {
        tmp = p[1];
        p[1] = p[0];
        p[0] = tmp;
    }
    if p[2] < p[0] {
        tmp = p[2];
        p[2] = p[0];
        p[0] = tmp;
    }
    if p[2] < p[1] {
        tmp = p[2];
        p[2] = p[1];
        p[1] = tmp;
    }

    var result = 777.0;

    if (abs(p[0] - p[2]) < 1.0) {
        var phi_sum = p[0] + p[1] + p[2];
        var phi_sum_pow2 = pow(phi_sum, 2.0);
        result = 1.0/6.0 * (2.0 * phi_sum + sqrt(4.0 * phi_sum_pow2 - 12.0 * (phi_sum_pow2 - 1.0/pow(speed, 2.0)))); 
        // result = 123.0;
    }

    else if (abs(p[0] - p[1]) < 1.0) {
        result = 0.5 * (p[0] + p[1] + sqrt(2.0 * 1.0/pow(speed, 2.0) - pow((p[0] - p[1]), 2.0)));
        // result = 555.0;
    }
 
    else {
        result = p[0] + 1.0/speed;
    }

    return result;
}

fn solve_quadratic(speed: f32) -> f32 {

        var phis: array<f32, 3> = array<f32, 3>(0.0, 0.0, 0.0);

        // Deltas. 
        var hs: array<f32, 3> = array<f32, 3>(0.0, 0.0, 0.0);
 
        // x dir.
        phis[0] = select(phis[0], min(neighbor_cells[0].value, neighbor_cells[1].value), neighbor_cells[0].tag == KNOWN && neighbor_cells[1].tag == KNOWN);
        phis[0] = select(phis[0], neighbor_cells[0].value, neighbor_cells[0].tag == KNOWN && neighbor_cells[1].tag != KNOWN);
        phis[0] = select(phis[0], neighbor_cells[1].value, neighbor_cells[0].tag != KNOWN && neighbor_cells[1].tag == KNOWN);

        // y dir.
        phis[1] = select(phis[1], min(neighbor_cells[2].value, neighbor_cells[3].value), neighbor_cells[2].tag == KNOWN && neighbor_cells[3].tag == KNOWN);
        phis[1] = select(phis[1], neighbor_cells[2].value, neighbor_cells[2].tag == KNOWN && neighbor_cells[3].tag != KNOWN);
        phis[1] = select(phis[1], neighbor_cells[3].value, neighbor_cells[2].tag != KNOWN && neighbor_cells[3].tag == KNOWN);

        // z dir.
        phis[2] = select(phis[2], min(neighbor_cells[4].value, neighbor_cells[5].value), neighbor_cells[4].tag == KNOWN && neighbor_cells[5].tag == KNOWN);
        phis[2] = select(phis[2], neighbor_cells[4].value, neighbor_cells[4].tag == KNOWN && neighbor_cells[5].tag != KNOWN);
        phis[2] = select(phis[2], neighbor_cells[5].value, neighbor_cells[4].tag != KNOWN && neighbor_cells[5].tag == KNOWN);


        if (phis[0] != 0.0) { hs[0] = 1.0; }
        if (phis[1] != 0.0) { hs[1] = 1.0; }
        if (phis[2] != 0.0) { hs[2] = 1.0; }

        var final_distance = 777.0;

        for (var j: i32 = 0 ; j<3 ; j = j + 1) {
            let h0 = hs[0] * hs[0];
            let h1 = hs[1] * hs[1];
            let h2 = hs[2] * hs[2];

            let a = h0 + h1 + h2;
            let b = (-2.0) * (h0*phis[0] + h1*phis[1] + h2*phis[2]); 
            let c = h0 * phis[0]*phis[0] + h1 * phis[1]*phis[1] + h2 * phis[2]*phis[2] - 1.0;

            let discriminant = pow(b, 2.0) - (4.0*a*c);

            if (discriminant >= 0.0) {
                let t_phi = ((-1.0) * b + sqrt(discriminant)) / (2.0*a); 
                if (phis[0] < t_phi && phis[1] < t_phi && phis[2] < t_phi) {
                    final_distance = min(t_phi, final_distance);
                }
            }

            var max_j = select(0, 1, phis[0] < phis[1]);
            max_j = select(max_j, 2, phis[max_j] < phis[2]);
            phis[max_j] = 0.0;
            hs[max_j] = 0.0;
        }

        return final_distance;
}

fn add_to_band(cell: FmmCellSync) {
    var temp_cell: FmmCell;
    temp_cell.tag = BAND;
    temp_cell.value = cell.value;
    temp_cell.queue_value = 0u;
    fmm_data[cell.mem_location] = temp_cell; 
}

@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    var this_coordinate_u32 = decode3Dmorton32(global_id.x);
    var this_coordinate_i32 = vec3<i32>(this_coordinate_u32);
    var this_cell = fmm_data[global_id.x];
    if (this_cell.tag == KNOWN) {
        load_neighbors(this_coordinate_u32);
        output_arrow[atomicAdd(&counter[1], 1u)] =  
              Arrow (
                  vec4<f32>(vec3<f32>(this_coordinate_i32), 0.0),
                  vec4<f32>(vec3<f32>(decode3Dmorton32(neighbor_cells[0].mem_location)), 0.0),
                  rgba_u32(255u, 0u, 0u, 255u),
                  0.01
        );
        add_neighbors_to_sync();
        //add_neighbors_to_band();
    }
}

@compute
@workgroup_size(1024,1,1)
fn sync_and_calculate_all_bands(@builtin(local_invocation_id)    local_id: vec3<u32>,
                                @builtin(local_invocation_index) local_index: u32,
                                @builtin(global_invocation_id)   global_id: vec3<u32>) {

    let sync_point_count = sync_point_counter[0];
    if (local_index < sync_point_count) {
        var sync_cell = synchronization_data[local_index];
        var speed = isotropic_data[sync_cell.mem_location];
        let this_coordinate_u32 = decode3Dmorton32(sync_cell.mem_location);
        load_neighbors(this_coordinate_u32);
        var value = solve_quadratic_2(speed);
        sync_cell.value = value;
        add_to_band(sync_cell);
            
        // add_neihgbors_to_band();
    }
}
