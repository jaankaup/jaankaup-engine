use std::borrow::Cow;
use crate::render_object::{RenderObject, ComputeObject, create_bind_groups};
use crate::render_things::{LightBuffer, RenderParamBuffer};
use crate::camera::Camera;
use crate::common_functions::{
    create_uniform_bindgroup_layout,
    create_buffer_bindgroup_layout,
};

/// RenderObject for renderer_v4n4_debug_visualizator.wgsl.
pub struct Render_VVVVNNNN_camera {
    render_object: RenderObject,
}

impl Render_VVVVNNNN_camera {

    fn init(device: &wgpu::Device, sc_desc: &wgpu::SurfaceConfiguration) -> Self {
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

    fn get_render_object(&self) -> &RenderObject {
        &self.render_object
    }

    fn create_bingroups(&self, device: &wgpu::Device, camera: &mut Camera, light: &LightBuffer, render_params: &RenderParamBuffer) -> Vec<wgpu::BindGroup> {
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
