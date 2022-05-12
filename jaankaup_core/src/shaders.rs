use core::mem::size_of;
use std::borrow::Cow;
use crate::impl_convert;
use crate::misc::Convert2Vec;
use crate::render_object::{RenderObject, ComputeObject, create_bind_groups};
use crate::render_things::{LightBuffer, RenderParamBuffer};
use crate::camera::Camera;
use crate::buffer::buffer_from_data;
use crate::common_functions::{
    create_uniform_bindgroup_layout,
    create_buffer_bindgroup_layout,
};
use bytemuck::{Zeroable, Pod};

/// RenderObject for renderer_v4n4_debug_visualizator.wgsl.
pub struct Render_VVVVNNNN_camera {
    render_object: RenderObject,
}

impl Render_VVVVNNNN_camera {

    pub fn init(device: &wgpu::Device, sc_desc: &wgpu::SurfaceConfiguration) -> Self {
        Self {
            render_object: RenderObject::init(
                               &device,
                               &sc_desc,
                               &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                                   label: Some("renderer_v4n4_debug_visualizator.wgsl"),
                                   source: wgpu::ShaderSource::Wgsl(
                                       Cow::Borrowed(include_str!("../../assets/shaders/renderer_v4n4_debug_visualizator.wgsl"))),

                               }),
                               &vec![wgpu::VertexFormat::Float32x4, wgpu::VertexFormat::Float32x4],
                               &vec![
                                   vec![
                                   // @group(0) @binding(0) var<uniform> camerauniform: Camera;
                                   create_uniform_bindgroup_layout(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),

                                   // @group(0) @binding(1) var<uniform> light: Light;
                                   create_uniform_bindgroup_layout(1, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),

                                   // @group(0) @binding(2) var<uniform> other_params: OtherParams;
                                   create_uniform_bindgroup_layout(2, wgpu::ShaderStages::VERTEX)
                                   ]
                               ],
                               Some("Debug visualizator vvvvnnnn renderer with camera."),
                               true,
                               wgpu::PrimitiveTopology::TriangleList
            ),
        }
    }

    pub fn get_render_object(&self) -> &RenderObject {
        &self.render_object
    }

    pub fn create_bingroups(&self, device: &wgpu::Device, camera: &mut Camera, light: &LightBuffer, render_params: &RenderParamBuffer) -> Vec<wgpu::BindGroup> {
        create_bind_groups(
                &device,
                &self.render_object.bind_group_layout_entries,
                &self.render_object.bind_group_layouts,
                &vec![
                    vec![
                    &camera.get_camera_uniform(&device).as_entire_binding(),
                    &light.get_buffer().as_entire_binding(),
                    &render_params.get_buffer().as_entire_binding(),
                    ],
                ]
        )
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct NoiseParams {
    pub global_dim: [u32; 3],
    pub time: f32,
    pub local_dim: [u32; 3],
    pub value: f32,
    pub position: [f32; 3],
    pub value2: f32,
}

impl_convert!{NoiseParams}

pub struct NoiseParamBuffer {
    pub noise_params: NoiseParams, // TODO: getter
    buffer: wgpu::Buffer,
}

impl NoiseParamBuffer {

    pub fn create(device: &wgpu::Device,
        global_dim: [u32; 3],
        time: f32,
        local_dim: [u32; 3],
        value: f32,
        position: [f32; 3],
        value2: f32,
        ) -> Self {

        let params = NoiseParams {
            global_dim: global_dim,
            time,
            local_dim,
            value,
            position,
            value2,
        };

        let buf = buffer_from_data::<NoiseParams>(
                  &device,
                  &vec![params],
                  wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                  Some("noise params buffer.")
        );

        Self {
            noise_params: params,
            buffer: buf,
        }
    }

    pub fn get_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn get_noise_params(&self) -> NoiseParams {
        self.noise_params
    }

    pub fn update(&self, queue: &wgpu::Queue) {
        
        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&[self.noise_params])
        );
    }
}

/// Noise maker.
pub struct NoiseMaker {
    compute_object: ComputeObject,
    pub noise_params: NoiseParamBuffer, // TODO: getter
    buffer: wgpu::Buffer,
    bind_groups: Vec<wgpu::BindGroup>,
}

impl NoiseMaker {

    pub fn init(device: &wgpu::Device,
                entry_point: &String,
                global_dimension: [u32; 3],
                local_dimension: [u32; 3],
                position: [f32; 3]) -> Self {

        let buf_size = (global_dimension[0] *
                        global_dimension[1] *
                        global_dimension[2] *
                        local_dimension[0] *
                        local_dimension[1] *
                        local_dimension[2] *
                        size_of::<f32>() as u32) as u64;
            
        let buf = device.create_buffer(&wgpu::BufferDescriptor{
                      label: Some("noise buffer"),
                      size: buf_size,
                      usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                      mapped_at_creation: false,
                      }
        );

        let compute_obj = ComputeObject::init(
                    &device,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("noise compute object"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/noise_to_buffer.wgsl"))),
                    
                    }),
                    Some("Noise compute object"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> noise_params: NoiseParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<storage, read_write> noise_output_data: Output;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    entry_point
        );
        
        let params = NoiseParamBuffer::create(
                        device,
                        global_dimension,
                        0.0,
                        local_dimension,
                        0.0,
                        position,
                        0.0
        );

        let bind_groups = create_bind_groups(
                &device,
                &compute_obj.bind_group_layout_entries,
                &compute_obj.bind_group_layouts,
                &vec![
                    vec![
                    &params.get_buffer().as_entire_binding(),
                    &buf.as_entire_binding(),
                    ],
                ]
        );

        Self {
            compute_object: compute_obj,
            noise_params: params,
            buffer: buf,
            bind_groups: bind_groups, 
        }
    }

    pub fn get_compute_object(&self) -> &ComputeObject {
        &self.compute_object
    }

    // fn create_bingroups(&self, device: &wgpu::Device) {
    //     self.bind_groups = create_bind_groups(
    //             &device,
    //             &self.compute_object.bind_group_layout_entries,
    //             &self.compute_object.bind_group_layouts,
    //             &vec![
    //                 vec![
    //                 &self.noise_params.get_buffer().as_entire_binding(),
    //                 &self.buffer.as_entire_binding(),
    //                 ],
    //             ]
    //     );
    // }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {

        let global_dimension = self.noise_params.noise_params.global_dim;
        let local_dimension  = self.noise_params.noise_params.local_dim;

        let total_grid_count =
                        global_dimension[0] *
                        global_dimension[1] *
                        global_dimension[2] *
                        local_dimension[0] *
                        local_dimension[1] *
                        local_dimension[2];

        self.compute_object.dispatch(
            &self.bind_groups,
            encoder,
            total_grid_count / 1024, 1, 1, Some("noise dispatch")
        );
    }

    pub fn update_time(&mut self, queue: &wgpu::Queue, time: f32) {
        self.noise_params.noise_params.time = time;
        self.noise_params.update(queue);
        // println!("self.noise_params.noise_params == {:?}", self.noise_params.noise_params);
    }

    pub fn update_value(&mut self, queue: &wgpu::Queue, value: f32) {
        self.noise_params.noise_params.value = value;
        self.noise_params.update(queue);
    }

    pub fn get_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}
