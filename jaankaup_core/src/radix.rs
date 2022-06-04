use std::borrow::Cow;
use crate::misc::Convert2Vec;
use crate::impl_convert;
use crate::common_functions::{create_uniform_bindgroup_layout, udiv_up_safe32};
use crate::render_object::create_bind_groups;
use crate::render_object::ComputeObject;
use crate::common_functions::create_buffer_bindgroup_layout;
use crate::buffer::{buffer_from_data};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RadixSortParams {
    phase: u32,    
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Bucket {
    pub bucket_id: u32,
    pub rank: u32,
    pub bucket_offset: u32,
    pub size: u32,
} 

struct KeyBlock {
	key_offset: u32,
	key_count: u32,
	bucket_id: u32,
	bucket_offset: u32,
}

// #[repr(C)]
// #[derive(Debug, Clone, Copy, Pod, Zeroable)]
// pub struct LocalSortBlock {
// 	pub bucket_id: u32,
// 	pub bucket_offset: u32,
// 	pub is_merged: u32,
// }

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LocalSortBlock {
	pub local_sort_id: u32,
	pub local_offset: u32,
	pub local_key_count: u32,
	pub is_merged: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct KeyMemoryIndex {
    pub key: u32,
    pub memory_location: u32,
}

impl_convert!{KeyMemoryIndex}

pub struct RadixSort {
    aux_buffer: wgpu::Buffer, 
    histogram_buffer: wgpu::Buffer,
    bucket_buffer: wgpu::Buffer,
    radix_sort_compute_object: ComputeObject,
    bind_groups: Vec<wgpu::BindGroup>,
    n: u32,
}

const KPT: u32 = 8;
const LOCAL_SORT_THRESHOLD: u32 = 4096;
const KEYBLOCK_SIZE: u32 = 1024 * KPT;

impl RadixSort {

    pub fn init(device: &wgpu::Device, input_data: &wgpu::Buffer, input_buffer_size: u32, n: u32) -> Self {

        // let buffer= buffer_from_data::<u32>(
        //     &device,
        //     &vec![0 ; buffer_size as usize],
        //     wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        //     None
        // );
        //
        let aux_buffer = 
                device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("Radix sort aux_buffer"),
                size: (input_buffer_size as usize * std::mem::size_of::<KeyMemoryIndex>()) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
        );

        // Max number of buckets. floor(n/local_sort_threshold)

        let max_number_of_buckets = 256 * input_buffer_size / LOCAL_SORT_THRESHOLD;

        println!("Number of buckets :: {}", max_number_of_buckets);

        let histogram_buffer = 
                device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("Radix sort histogram buffer."),
                size: (256 as usize * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
        );

        let bucket_buffer = 
                device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("Bucket buffer."),
                size: (max_number_of_buckets as usize * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
        );

        let radix_sort_compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("counting.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/radix/counting.wgsl"))),
                    
                    }),
                    Some("Fmm radix sort compute object"),
                    &vec![
                        vec![
                            // // @group(0) @binding(0) var<uniform> fmm_params: FmmVisualizationParams;
                            // create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(0) var<storage, read_write> data1: array<KeyMemoryIndex>;
                            create_buffer_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(1) var<storage, read_write> data2: array<KeyMemoryIndex>;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(2) var<storage, read_write> global_histogram: array<u32>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string()
        );

        let bind_groups =
            create_bind_groups(
                &device,
                &radix_sort_compute_object.bind_group_layout_entries,
                &radix_sort_compute_object.bind_group_layouts,
                &vec![
                    vec![
                         &input_data.as_entire_binding(),
                         &aux_buffer.as_entire_binding(),
                         &histogram_buffer.as_entire_binding(),

                         // &gpu_debugger.get_element_counter_buffer().as_entire_binding(),
                         // &gpu_debugger.get_output_chars_buffer().as_entire_binding(),
                         // &gpu_debugger.get_output_arrows_buffer().as_entire_binding(),
                         // &gpu_debugger.get_output_aabbs_buffer().as_entire_binding(),
                         // &gpu_debugger.get_output_aabb_wires_buffer().as_entire_binding(),
                    ]
                ]
        );

        Self {
            aux_buffer: aux_buffer,
            histogram_buffer: histogram_buffer, 
            bucket_buffer: bucket_buffer, 
            radix_sort_compute_object: radix_sort_compute_object,
            bind_groups: bind_groups,
            n: n,
        }
    }

    pub fn get_global_histogram(&self) -> &wgpu::Buffer {
        &self.histogram_buffer
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {

    }

    /// Sorts the global data for msb 8 bits and creates the initial buckets.
    pub fn initial_counting_sort(&self, encoder: &mut wgpu::CommandEncoder) {

        // The GPU counting sort.
          
        println!("radix sort dispatch count :: {}", udiv_up_safe32(self.n, 1024 * KPT)); 

        self.radix_sort_compute_object.dispatch_push_constants::<u32>(
            &self.bind_groups,
            encoder,
            udiv_up_safe32(self.n, 1024 * KPT) + 1, 1, 1,
            0,
            0,
            Some("Radix dispatch.")
        );
    }
}
