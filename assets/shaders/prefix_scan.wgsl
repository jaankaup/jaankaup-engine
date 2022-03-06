// 64 :: 136
// 128 :: 272
// 192 :: 408
// 256 :: 544
// 320 :: 680
// 384 :: 816
// 448 :: 952
// 512 :: 1088
// 576 :: 1224
// 640 :: 1360
// 704 :: 1496
// 768 :: 1632
// 832 :: 1768
// 896 :: 1904
// 960 :: 2040
// 1024 :: 2176

struct PrefixParams {
    blaah: u32;
};

struct FmmBlock {
    index: u32;
    band_points_count: u32;
};

@group(0)
@binding(0)
var<uniform> fmm_prefix_params: PrefixParams;

@group(0)
@binding(1)
var<storage, read_write> fmm_blocks: array<FmmBlock>;

let THREAD_COUNT = 64u;
let SCAN_BLOCK_SIZE = 136; 

// The output of global active fmm block scan.
var<workgroup> shared_prefix_sum: array<u32, SCAN_BLOCK_SIZE>; 

///////////////////////////
////// Prefix scan   //////
///////////////////////////

// Add a chunk of block data to a temporary shared_perix_sum buffer.
// Copies FMM_Block to shared_prefix_sum {0,1}. Add padding 0 if necessery.
fn copy_block_to_temp(chunck_id: u32, number_of_items: u32, thread_id: u32) {

    // Create the local indices.
    let ai = thread_id; 
    let bi = ai + THREAD_COUNT;

    // Create the bank conflict free local indices.
    let ai_bcf = ai + (ai >> 4u); 
    let bi_bcf = bi + (bi >> 4u);

    // Create the global indices to access the global memory.
    let global_ai = ai + chunck_id * (THREAD_COUNT * 2u);
    let global_bi = bi + chunck_id * (THREAD_COUNT * 2u);

    // Create {0:1} array from the global fmm_block array.
    // 0 :: no band cells.
    // 1 :: band cells > 0.

    let a = fmm_blocks[global_ai];
    let b = fmm_blocks[global_bi];

    shared_prefix_sum[ai_bcf] = select(0u, 1u, ai < number_of_items && a.band_points_count > 0u);
    shared_prefix_sum[bi_bcf] = select(0u, 1u, bi < number_of_items && b.band_points_count > 0u);
}

@stage(compute)
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    if (local_index == 0u) {
    }
    workgroupBarrier();

    // Prefix sum

    // reduce
    
}
