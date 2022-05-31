use std::borrow::Cow;
use crate::render_object::create_bind_groups;
use crate::render_object::ComputeObject;
use crate::common_functions::create_buffer_bindgroup_layout;
use crate::buffer::{buffer_from_data};
use bytemuck::{Pod, Zeroable};

struct Bucket {
    bucket_id: u32,
    size: u32,
}

struct KeyBlock {
	key_offset: u32,
	key_count: u32,
	bucket_id: u32,
	bucket_offset: u32,
}

struct LocalSortBlock {
	bucket_id: u32,
	bucket_offset: u32,
	is_merged: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct KeyMemoryIndex {
    pub key: u32,
    pub memory_location: u32,
}

pub struct RadixSort {
    aux_buffer: wgpu::Buffer, 
    radix_sort_compute_object: ComputeObject,
    bind_groups: Vec<wgpu::BindGroup>,
}

impl RadixSort {

    pub fn init(device: &wgpu::Device, input_data: &wgpu::Buffer, input_buffer_size: u32) -> Self {

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
                            //++ // @group(0) @binding(0) var<uniform> fmm_params: FmmVisualizationParams;
                            //++ create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(0) var<storage, read_write> data1: array<KeyMemoryIndex>;
                            create_buffer_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(1) var<storage, read_write> data1: array<KeyMemoryIndex>;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),
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
            radix_sort_compute_object: radix_sort_compute_object,
            bind_groups: bind_groups,
        }
    }

    /// Sorts the global data for msb 8 bits and creates the initial buckets.
    pub fn initial_counting_sort(&self) {

        // The GPU counting sort.
        
    }
}
