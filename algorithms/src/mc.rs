use jaankaup_core::wgpu;
use bytemuck::{Zeroable, Pod};
use jaankaup_core::buffer::buffer_from_data;
use jaankaup_core::render_object::{ComputeObject, create_bind_groups, DrawIndirect};
use jaankaup_core::histogram::Histogram;
use jaankaup_core::common_functions::{
    create_uniform_bindgroup_layout,
    create_buffer_bindgroup_layout
};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct McParams {
    pub base_position: [f32; 4],
    pub isovalue: f32,
    pub cube_length: f32,
    pub future_usage1: f32,
    pub future_usage2: f32,
    pub noise_global_dimension: [u32; 4], 
    pub noise_local_dimension: [u32; 4], 
}

pub struct MarchingCubes {
    // The mc pipeline.
    compute_object: ComputeObject,
    mc_params: McParams,
    mc_params_buffer: wgpu::Buffer,
    buffer_counter: Histogram,
    indirect_buffer: wgpu::Buffer,
    bind_groups: Vec<wgpu::BindGroup>,
}

impl MarchingCubes {

    pub fn get_counter_value(&self, device:&wgpu::Device, queue: &wgpu::Queue) -> u32 {
        self.buffer_counter.get_values(device, queue)[0]
    }
    pub fn get_draw_indirect_buffer(&self) -> &wgpu::Buffer {
        &self.indirect_buffer
    }
    pub fn reset_counter_value(&self, queue: &wgpu::Queue) {
        self.buffer_counter.set_values_cpu_version(queue, &vec![0]);
        
        queue.write_buffer(
            &self.indirect_buffer,
            0,
            bytemuck::cast_slice(&[
                DrawIndirect {
                    vertex_count: 0,
                    instance_count: 1,
                    base_vertex: 0,
                    base_instance: 0,
                }
            ])
        );
    }

    pub fn update_mc_params(&mut self, queue: &wgpu::Queue, mc_params: McParams) {

        // self.mc_params.isovalue = isovalue;
        self.mc_params = mc_params;

        queue.write_buffer(
            &self.mc_params_buffer,
            0,
            bytemuck::cast_slice(&[self.mc_params ])
        );
    }

    pub fn get_mc_params(&self) -> McParams {
        self.mc_params
    }

    pub fn init_with_noise_buffer(device: &wgpu::Device,
                                  mc_params: &McParams,
                                  mc_shader: &wgpu::ShaderModule,
                                  noise_buffer: &wgpu::Buffer,
                                  output_buffer: &wgpu::Buffer,
                                  ) -> Self {

        let histogram = Histogram::init(device, &vec![0]);

        // Initialize the draw indirect data.
        let indirect_data =  
            DrawIndirect {
                vertex_count: 0,
                instance_count: 1,
                base_vertex: 0,
                base_instance: 0,
            };

        let indirect_buffer = 
            buffer_from_data::<DrawIndirect>(
            &device,
            &[indirect_data],
            // wgpu::BufferUsages::VERTEX |
            wgpu::BufferUsages::COPY_SRC |
            wgpu::BufferUsages::COPY_DST |
            wgpu::BufferUsages::INDIRECT |
            wgpu::BufferUsages::STORAGE,
            None
        );

        let compute_object =
                ComputeObject::init(
                    &device,
                    &mc_shader,
                    Some("Marching cubes Compute object"), // TODO: from parameter
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> mc_uniform: McParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<storage, read_write> indirect: array<DrawIndirect>;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(2) var<storage, read_write> counter: atomic<u32>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(3) var<storage, write> noise_values: array<Vertex>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, true),

                            // @group(0) @binding(4) var<storage, write> output: array<Vertex>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None
        );

        let mc_params_buffer = buffer_from_data::<McParams>(
                &device,
                &[*mc_params],
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
                None
        );

        let bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                         &mc_params_buffer.as_entire_binding(),
                         &indirect_buffer.as_entire_binding(),
                         &histogram.get_histogram_buffer().as_entire_binding(),
                         &noise_buffer.as_entire_binding(),
                         &output_buffer.as_entire_binding()
                    ]
                ]
        );

        Self {
            compute_object: compute_object,
            mc_params: *mc_params,
            mc_params_buffer: mc_params_buffer,
            buffer_counter: histogram,
            indirect_buffer: indirect_buffer,
            bind_groups: bind_groups,
        }
    }

    pub fn dispatch(&self,
                    encoder: &mut wgpu::CommandEncoder,
                    x: u32, y: u32, z: u32) {

        // let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Marhing cubes command encoder descriptor.") });

        self.compute_object.dispatch(
            &self.bind_groups,
            encoder,
            x, y, z, Some("mc dispatch")
        );
    }
}
