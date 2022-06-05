// Keys per thread.
let KPT = 8u;
let KP_BITONIC = 3u;
//let NUMBER_OF_THREADS = 256u;
let NUMBER_OF_THREADS = 1024u;

// Phase 0 :: initial count

struct PushConstants {
    phase: u32,    
};

struct Bucket {
    bucket_id: u32,
    rank: u32,
    bucket_offset: u32,
    size: u32,
}; 

struct KeyMemoryIndex {
    key: u32,
    memory_location: u32,
};

struct KeyBlock {
    key_offset: u32,
    key_count: u32,
    bucket_id: u32,
    bucket_offset: u32,
};

struct PrivateData {
    ai: u32,
    bi: u32,
    ai_bcf: u32,
    bi_bcf: u32,
    global_ai: u32,
    global_bi: u32,
};

var<push_constant> pc: PushConstants;

// @group(0) @binding(0)
// var<uniform> radix_sort_params: u32;

@group(0) @binding(0)
var<storage, read_write> data1: array<KeyMemoryIndex>;

@group(0) @binding(1)
var<storage, read_write> data2: array<KeyMemoryIndex>;

@group(0) @binding(2)
var<storage, read_write> global_histogram: array<atomic<u32>>;

// @group(0) @binding(3)
// var<storage, read_write> global_histogram: array<atomic<u32>>;

var<private> private_data: PrivateData;
var<workgroup> local_radix256_histogram: array<atomic<u32>, 256>;
var<workgroup> bitonic_temp: array<KeyMemoryIndex , 3072>;

// var<workgroup> bitonic_temp: array<u32 ; NUMBER_OF_THREADS * KP_BITONIC>;

fn udiv_up_safe32(x: u32, y: u32) -> u32 {
    let tmp = (x + y - 1u) / y;
    return select(tmp, 0u, y == 0u); 
}

fn bitonic(thread_index: u32) {

  for (var k: u32 = 2u; k <= NUMBER_OF_THREADS * KP_BITONIC; k = k << 1u) {
  for (var j: u32 = k >> 1u ; j > 0u; j = j >> 1u) {
    workgroupBarrier();
    for (var i: u32 = 0u ; i<KP_BITONIC; i = i + 1u)  {
        let index = thread_index + i * NUMBER_OF_THREADS; 
        let ixj = index ^ j;
        let a = bitonic_temp[index];
        let b = bitonic_temp[ixj];

        if (ixj > index &&
            (((index & k) == 0u && a.key > b.key) || 
            ((index & k) != 0u && a.key < b.key)) ) {
                bitonic_temp[index] = b;
                bitonic_temp[ixj] = a;
        }
  }}};
}

fn load_keys_to_bitonic_temp(bucket: ptr<function, Bucket>, local_index: u32) {

    for (var i:u32 = 0u ; i < KP_BITONIC ; i = i + 1u) {

        let key_index = local_index + NUMBER_OF_THREADS * i;

        // Load key/pair to the workgroup memory.
        if (key_index < (*bucket).size) {
            bitonic_temp[key_index] = data1[key_index + (*bucket).bucket_offset];
        }

        // Add dummy key/pairt to the workgroup memory.
        else {
            bitonic_temp[key_index] = KeyMemoryIndex(0xffffffffu, 0xffffffffu);
        }
    } 
}

fn save_keys_from_bitonic_temp(bucket: ptr<function, Bucket>, local_index: u32) {

    for (var i:u32 = 0u ; i < KP_BITONIC ; i = i + 1u) {

        let key_index = local_index + NUMBER_OF_THREADS * i; 

        if (key_index < (*bucket).size) {
            data1[key_index + (*bucket).bucket_offset] = bitonic_temp[key_index]; 
        }
    } 
}
             
// group :: 0 0xff000000
// group :: 1 0x00ff0000
// group :: 2 0x0000ff00
// group :: 3 0x000000ff
fn extract_digit_8(key: u32, group: u32) -> u32 {
    let shift = (3u - group) * 8u; 
    return (key & (0x000000ffu << shift)) >> shift;
}

// fn format_local_histogram(local_index: ptr<function, u32>) {
fn format_local_histogram(local_index: u32) {

        if (local_index < 256u) {
            local_radix256_histogram[local_index] = 0u;
        }

        workgroupBarrier();
}

fn local_count(bucket: ptr<function, Bucket>, local_index: u32, workgroup_index: u32) {

        for (var i: u32 = 0u; i < KPT ; i = i + 1u) {

            // without bucket offset.
            let key_index = NUMBER_OF_THREADS * KPT * workgroup_index + local_index + NUMBER_OF_THREADS * i; 

            // If the thread index is smaller than key count, load the key and do the count.
            if (key_index < (*bucket).size) { //key_block.key_count) 

                // Get the key-value pair.
                let data = data1[key_index + (*bucket).bucket_offset]; // + key_block.bucket_offset];
                atomicAdd(&local_radix256_histogram[extract_digit_8(data.key, 3u)], 1u);
            }
        }

        // Do we need this?
        workgroupBarrier();
}

fn update_global_histogram(local_index: u32) {

    if (local_index < 256u) {
        atomicAdd(&global_histogram[local_index], local_radix256_histogram[local_index]);
    }
}

fn counting_sort(bucket: ptr<function, Bucket>, local_index: u32, workgroup_index: u32) {

        format_local_histogram(local_index);
        local_count(bucket, local_index, workgroup_index);
        update_global_histogram(local_index);
}

fn bitonic_sort(bucket: ptr<function, Bucket>, local_index: u32) {

    load_keys_to_bitonic_temp(bucket, local_index);
    bitonic(local_index);
    save_keys_from_bitonic_temp(bucket, local_index);
}

    // let ai = local_index; 
    // let bi = ai + NUMBER_OF_THREADS;
    // let ai_bcf = ai + (ai >> 4u); 
    // let bi_bcf = bi + (bi >> 4u);
    // let global_ai = ai + workgroup_id.x * (NUMBER_OF_THREADS * 2u);
    // let global_bi = bi + workgroup_id.x * (NUMBER_OF_THREADS * 2u);


@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    var test_bucket = Bucket(0u, 0u, 0u, 1000000u);
    var test_bucket_bitonic = Bucket(0u, 0u, 0u, 1800u);

    if (pc.phase == 0u) {
        counting_sort(&test_bucket, local_index, workgroup_id.x);
        bitonic_sort(&test_bucket_bitonic, local_index);
    }
}
