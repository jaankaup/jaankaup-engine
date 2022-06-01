// Keys per thread.
let KPT = 8u;

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

@group(0) @binding(0)
var<storage, read_write> data1: array<KeyMemoryIndex>;

@group(0) @binding(1)
var<storage, read_write> data2: array<KeyMemoryIndex>;

@group(0) @binding(2)
var<storage, read_write> global_histogram: array<atomic<u32>>;

var<workgroup> local_radix256_histogram: array<atomic<u32>, 256>;
//++ var<private> private_keys: array<KeyMemoryIndex, KPT>;

fn udiv_up_safe32(x: u32, y: u32) -> u32 {
    let tmp = (x + y - 1u) / y;
    return select(tmp, 0u, y == 0u); 
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

fn local_count(local_index: u32, workgroup_index: u32) {

        for (var i: u32 = 0u; i < KPT ; i = i + 1u) {

            // without bucket offset.
            let key_index = 1024u * KPT * workgroup_index + local_index + 1024u * i; 

            // If the thread index is smaller than key count, load the key and do the count.
            if (key_index < 130000u) { //key_block.key_count) 

                // Get the key-value pair.
                let data = data1[key_index]; // + key_block.bucket_offset];
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

fn counting_sort(local_index: u32, workgroup_index: u32) {

        format_local_histogram(local_index);
        local_count(local_index, workgroup_index);
        update_global_histogram(local_index);
}

@compute
@workgroup_size(1024,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

        // let key_block = KeyBlock(0u, 1024u, 5u, 2096u);
        counting_sort(local_index, workgroup_id.x);

}
