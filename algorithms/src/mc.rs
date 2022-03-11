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

/// Uniform data for marching cubes (set=0, binding=0).
// pub struct McParams {
//     params: McUniform,
//     // buffer: wgpu::Buffer,
//     pub counter_buffer: Histogram,
//     pub bind_groups: Option<Vec<wgpu::BindGroup>>,
// }
// 
// impl McParams {
// 
//     /// Create an instance of McParams.
//     pub fn init(device: &wgpu::Device, base_position: &cgmath::Vector4<f32>, isovalue: f32, cube_length: f32) -> Self {
// 
//         assert!(cube_length > 0.0, "{}", format!("cube_length ==  {} > 0.0", cube_length));
// 
//         let uniform = McUniform {
//                 base_position: *base_position,
//                 isovalue: isovalue,
//                 cube_length: cube_length,
//                 future_usage1: 0.0,
//                 future_usage2: 0.0,
//         };
// 
//         Self {
//             params: uniform,
//             buffer: buffer_from_data::<McUniform>(
//                 &device,
//                 &[uniform],
//                 wgpu::BufferUsages::COPY_DST |wgpu::BufferUsages::UNIFORM,
//                 None),
//             counter_buffer: buffer_from_data::<u32>(
//                 &device,
//                 &[0 as u32],
//                 wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST |wgpu::BufferUsages::COPY_SRC,
//                 None),
//             bind_groups: None,
//         }
//     }
// 
//     pub fn get_uniform_buffer(&self) -> &wgpu::Buffer {
//         &self.buffer
//     }
// 
//     pub fn get_params(&self) -> &McUniform {
//         &self.params
//     }
// 
//     pub fn reset_counter(&self, queue: &wgpu::Queue) {
//         queue.write_buffer(
//             &self.counter_buffer,
//             0,
//             bytemuck::cast_slice(&[0 as u32])
//         );
//     }
// 
//     /// Updates the given mc-parameters and updates the buffer.
//     pub fn update_params(
//         &mut self,
//         queue: &wgpu::Queue,
//         base_position: &Option<cgmath::Vector4<f32>>,
//         isovalue: &Option<f32>,
//         cube_length: &Option<f32>,
//         future1: &Option<f32>) {
// 
//         // Update params.
//         if let Some(position) = *base_position {
//             self.params.base_position = position;
//         }
//         if let Some(iso) = *isovalue {
//             self.params.isovalue = iso;
//         }
//         if let Some(length) = *cube_length {
//             assert!(length > 0.0, "{}", format!("length ==  {} > 0.0", length));
//             self.params.cube_length = length;
//         }
//         if let Some(f) = *future1 {
//             self.params.future_usage1 = f;
//         }
// 
//         // Update the buffer.
//         queue.write_buffer(
//             &self.buffer,
//             0,
//             bytemuck::cast_slice(&[self.params])
//         );
//     }
// }

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

    pub fn reset_counter_value(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.buffer_counter.set_values_cpu_version(queue, &vec![0]);
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
                instance_count: 0,
                base_vertex: 0,
                base_instance: 0,
            };

        let indirect_buffer = 
            buffer_from_data::<DrawIndirect>(
            &device,
            &[indirect_data],
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
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

                            // @group(0) @binding(1) var<storage, read_write> counter: atomic<u32>;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // READ_ONLY == true
                            // @group(0) @binding(2) var<storage, write> noise_values: array<Vertex>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, true),

                            // @group(0) @binding(3) var<storage, write> output: array<Vertex>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ]
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

