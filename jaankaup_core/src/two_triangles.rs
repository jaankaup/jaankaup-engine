use std::borrow::Cow;
use crate::render_object::{RenderObject, create_bind_groups};
use crate::buffer::buffer_from_data;
use crate::texture::Texture as JaankaupTexture;

use crate::common_functions::{
    create_texture,
    create_texture_sampler,
};

pub struct TwoTriangles {
    two_triangles_ro: RenderObject,
    two_triangles_bg: Vec<wgpu::BindGroup>,
    two_triangles_texture: JaankaupTexture,
    two_triangles_vertex_buffer: wgpu::Buffer,
}

impl TwoTriangles {
    
    pub fn init(device: &wgpu::Device, sc_desc: &wgpu::SurfaceConfiguration, texture_dimension: [u32; 2]) -> Self {

        let two_triangles_vertex_buffer  = buffer_from_data::<f32>(
            &device,
            // gl_Position     |    point_pos
            &[-1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0,
               1.0,  1.0, 0.0, 1.0, 1.0,  1.0, 0.0, 1.0,
              -1.0,  1.0, 0.0, 1.0,-1.0, -1.0, 0.0, 1.0],
            // &[-1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            //    1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            //    1.0,  1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
            //    1.0,  1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0,
            //   -1.0,  1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            //   -1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            // ],
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC,
            None
        );

        let two_triangles_ro = RenderObject::init(
                                   &device,
                                   &sc_desc,
                                   &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                                       label: Some("sphere_tracer_renderer.wgsl"),
                                       source: wgpu::ShaderSource::Wgsl(
                                           Cow::Borrowed(include_str!("../../assets/shaders/sphere_tracer_renderer.wgsl"))),

                                   }),
                                   &vec![wgpu::VertexFormat::Float32x4],
                                   //&vec![wgpu::VertexFormat::Float32x4, wgpu::VertexFormat::Float32x4],
                                   &vec![
                                       vec![
                                       // @group(0) @binding(0) var t_diffuse1: texture_2d<f32>;
                                       create_texture(0, wgpu::ShaderStages::FRAGMENT),

                                       // @group(0) @binding(1) var s_diffuse1: sampler;
                                       create_texture_sampler(1, wgpu::ShaderStages::FRAGMENT),
                                       ],
                                   ],
                                   Some("Sphere tracer renderer."),
                                   true,
                                   wgpu::PrimitiveTopology::TriangleList
        );

        let two_triangle_texture = JaankaupTexture::create_texture2d(&device, &sc_desc, 1, texture_dimension[0], texture_dimension[1]);

        let two_triangles_bg = create_bind_groups(
                &device,
                &two_triangles_ro.bind_group_layout_entries,
                &two_triangles_ro.bind_group_layouts,
                &vec![
                    vec![&wgpu::BindingResource::TextureView(&two_triangle_texture.view),
                         &wgpu::BindingResource::Sampler(&two_triangle_texture.sampler),
                    ]
                ]
        );

        Self {
            two_triangles_ro: two_triangles_ro,
            two_triangles_bg: two_triangles_bg,
            two_triangles_texture: two_triangle_texture,
            two_triangles_vertex_buffer: two_triangles_vertex_buffer,
        }
    }

    pub fn get_draw_buffer(&self) -> &wgpu::Buffer {
        &self.two_triangles_vertex_buffer
    }

    pub fn get_render_object(&self) -> &RenderObject {
        &self.two_triangles_ro
    }

    pub fn get_bind_groups(&self) -> &Vec<wgpu::BindGroup> {
        &self.two_triangles_bg
    }

    pub fn get_texture(&self) -> &wgpu::Texture {
        &self.two_triangles_texture.texture
    }
}
