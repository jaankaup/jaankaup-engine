
// TODO: implement RadixSort Struct using Rust 


//++ // #define THREADS 256 
//++ // #define KPT 16 
//++ // #define RADIX 256 
//++ // #define KPB 4096
//++ 
//++ struct KeyBlock {
//++     key_offset: u32,
//++     key_count: u32,
//++     bucket_id: u32,
//++     bucket_offset: u32,
//++ };
//++ 
//++ struct LocalSortBlock {
//++     bucket_id: u32,
//++     bucket_offset: u32,
//++     is_merged: u32,
//++ };
//++ 
//++ 
//++ // Phase 0
//++ 
//++ // Create initial key blocks.
//++ 
//++ //    kb0           kb1        kb2          kb3        kb4         kb5         kb6         kb7         kb8
//++ // +-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------|---+
//++ // |           |           |           |           |           |           |           |           |       |   |
//++ // |           |           |           |           |           |           |           |           |       |   |
//++ // +-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-------|---+
//++ 
//++ @group(0) @binding(0) var<storage, read_write> input:  array<u32>;
//++ @group(0) @binding(1) var<storage, read_write> input2: array<u32>;
//++ 
//++ var<workgroup> histogram: array<u32, 256>;
//++ var<workgroup> prefix_sum: array<u32, 1024>;
//++ 
//++ // @amount == 0 (ff000000)
//++ // @amount == 1 (  ff0000)
//++ // @amount == 2 (    ff00)
//++ // @amount == 3 (      ff)
//++ fn extract_digit_msd(key: u32, amount: u32) { return key & (0xff000000 >> (amount * 8u)); }
//++ 
//++ // Bitonic sort.
//++ fn bitonic(thread_index: u32) {
//++ 
//++   for (var k: u32 = 2u; k <= NUMBER_OF_THREADS; k = k << 1u) {
//++   for (var j: u32 = k >> 1u ; j > 0u; j = j >> 1u) {
//++     workgroupBarrier();
//++ 
//++     let index = thread_index; 
//++     let ixj = index ^ j;
//++     let a = // elem a
//++     let b = // elem b
//++ 
//++     if (ixj > index &&
//++            (
//++                ((index & k) == 0u && a > b) ||
//++                ((index & k) != 0u && a < b)
//++            )
//++        )
//++        // Swap.
//++        {
//++             workgroup_chars[index] = b;
//++             workgroup_chars[ixj] = a;
//++        }
//++   }};
//++ }
//++ 
//++ @compute
//++ @workgroup_size(64,1,1)
//++ fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
//++         @builtin(local_invocation_index) local_index: u32,
//++         @builtin(workgroup_id) work_group_id: vec3<u32>,
//++         @builtin(global_invocation_id)   global_id: vec3<u32>) {
//++ 
//++ }
