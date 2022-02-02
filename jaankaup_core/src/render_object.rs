use core::ops::Range;
use crate::texture::Texture;

pub struct TestRenderObject {
    pub bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout_entries: Vec<Vec<wgpu::BindGroupLayoutEntry>>
}

impl TestRenderObject {
    pub fn init(device: &wgpu::Device,
                sc_desc: &wgpu::SurfaceConfiguration,
                wgsl_module: &wgpu::ShaderModule,
                vertex_attributes: &Vec<wgpu::VertexFormat>,
                bind_group_layout_entries: Vec<Vec<wgpu::BindGroupLayoutEntry>>,
                ) -> Self {

        let mut bind_group_layouts: Vec<wgpu::BindGroupLayout> = Vec::new();

        // Create bind group layout descriptors. This is used to create bind group layout.
        //for ve in layout_data.get_bind_group_layout_entries().iter() {
        for ve in bind_group_layout_entries.iter() {
            bind_group_layouts.push(
                device.create_bind_group_layout(
                    &wgpu::BindGroupLayoutDescriptor {
                        entries: &ve,
                        label: None,
                    }
                )
            );
        }

        let (stride, attributes) =  create_vb_descriptor(
            vertex_attributes
        );

        // Create pipeline layout.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &bind_group_layouts.iter().collect::<Vec<_>>(), // &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create the render pipeline.
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &wgsl_module,
                entry_point: "vs_main",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: stride,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &attributes,
                    }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // cull_mode: Some(wgpu::Face::Front),
                unclipped_depth: false, // ???
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState {
                    front: wgpu::StencilFaceState::IGNORE,
                    back: wgpu::StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: wgpu::DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &wgsl_module,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: sc_desc.format,
                    blend: None, //Some(wgpu::BlendState {
                           //     color: wgpu::BlendComponent {
                           //          src_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                           //          dst_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                           //          operation: wgpu::BlendOperation::Max,
                           //     },
                           //     alpha: wgpu::BlendComponent {
                           //          src_factor: wgpu::BlendFactor::SrcAlpha,
                           //          dst_factor: wgpu::BlendFactor::One,
                           //          operation: wgpu::BlendOperation::Add,
                           //     },
                           // }),
                    // alpha_blend: wgpu::BlendState::REPLACE,
                    // color_blend: wgpu::BlendState::REPLACE,
                    write_mask: wgpu::ColorWrites::COLOR,
                }],
            }),
            multiview: None,
        });

        Self {
            bind_group_layouts: bind_group_layouts,
            pipeline: pipeline,
            bind_group_layout_entries: bind_group_layout_entries,
        }
    }

}

/// Takes wgpu::VertexFormats as input and returns (stride, Vec<wgpu::VertexBufferDescriptor>)
pub fn create_vb_descriptor(formats: &Vec<wgpu::VertexFormat>) -> (u64, Vec<wgpu::VertexAttribute>) { 

    let mut attribute_descriptors: Vec<wgpu::VertexAttribute> = Vec::new();
    let mut stride: u64 = 0;
    for (i, format) in formats.iter().enumerate() {
        let size = match format {
                wgpu::VertexFormat::Uint8x2 => 2 * std::mem::size_of::<u8>() as u64, 
                wgpu::VertexFormat::Uint8x4 => 4 * std::mem::size_of::<u8>() as u64,
                wgpu::VertexFormat::Sint8x2 => 2 * std::mem::size_of::<i8>() as u64,
                wgpu::VertexFormat::Sint8x4 => 4 * std::mem::size_of::<i8>() as u64,
                wgpu::VertexFormat::Unorm8x2 => 2 * std::mem::size_of::<u8>() as u64,
                wgpu::VertexFormat::Unorm8x4 => 4 * std::mem::size_of::<u8>() as u64,
                wgpu::VertexFormat::Snorm8x2 => 2 * std::mem::size_of::<u8>() as u64,
                wgpu::VertexFormat::Snorm8x4 => 4 * std::mem::size_of::<u8>() as u64,
                wgpu::VertexFormat::Uint16x2 => 2 * std::mem::size_of::<u16>() as u64,
                wgpu::VertexFormat::Uint16x4 => 4 * std::mem::size_of::<u16>() as u64,
                wgpu::VertexFormat::Sint16x2 => 2 * std::mem::size_of::<i16>() as u64,
                wgpu::VertexFormat::Sint16x4 => 4 * std::mem::size_of::<i16>() as u64,
                wgpu::VertexFormat::Unorm16x2 => 2 * std::mem::size_of::<u16>() as u64,
                wgpu::VertexFormat::Unorm16x4 => 4 * std::mem::size_of::<u16>() as u64,
                wgpu::VertexFormat::Snorm16x2 => 2 * std::mem::size_of::<i16>() as u64,
                wgpu::VertexFormat::Snorm16x4 => 4 * std::mem::size_of::<i16>() as u64,
                wgpu::VertexFormat::Float16x2 => unimplemented!(),
                wgpu::VertexFormat::Float16x4 => unimplemented!(),
                wgpu::VertexFormat::Float32 => std::mem::size_of::<f32>() as u64,
                wgpu::VertexFormat::Float32x2 => 2 * std::mem::size_of::<f32>() as u64,
                wgpu::VertexFormat::Float32x3 => 3 * std::mem::size_of::<f32>() as u64,
                wgpu::VertexFormat::Float32x4 => 4 * std::mem::size_of::<f32>() as u64,
                wgpu::VertexFormat::Uint32 => std::mem::size_of::<u32>() as u64,
                wgpu::VertexFormat::Uint32x2 => 2 * std::mem::size_of::<u32>() as u64,
                wgpu::VertexFormat::Uint32x3 => 3 * std::mem::size_of::<u32>() as u64,
                wgpu::VertexFormat::Uint32x4 => 4 * std::mem::size_of::<u32>() as u64,
                wgpu::VertexFormat::Sint32 => std::mem::size_of::<i32>() as u64,
                wgpu::VertexFormat::Sint32x2 => 2 * std::mem::size_of::<i32>() as u64,
                wgpu::VertexFormat::Sint32x3 => 3 * std::mem::size_of::<i32>() as u64,
                wgpu::VertexFormat::Sint32x4 => 4 * std::mem::size_of::<i32>() as u64,
                wgpu::VertexFormat::Float64
                | wgpu::VertexFormat::Float64x2
                | wgpu::VertexFormat::Float64x3
                | wgpu::VertexFormat::Float64x4
                => panic!("VERTEX_ATTRIBUTE_64BIT must be enabled to use Double formats")
        };
        attribute_descriptors.push(
            wgpu::VertexAttribute {
                format: *format,
                offset: stride,
                shader_location: i as u32, 
            }
        );
        stride = stride + size;
    }

    (stride, attribute_descriptors)
}

pub fn create_bind_groups(device: &wgpu::Device,
                          entry_layouts: &Vec<Vec<wgpu::BindGroupLayoutEntry>>,
                          bing_group_layouts: &Vec<wgpu::BindGroupLayout>,
                          bindings: &Vec<Vec<&wgpu::BindingResource>>)
                        -> Vec<wgpu::BindGroup> {

    // The created bindgroups.
    let mut result: Vec<wgpu::BindGroup> = Vec::new();

    // Add Binding resources to the bind group.
    for i in 0..entry_layouts.len() {

        let mut inner_group: Vec<wgpu::BindGroupEntry> = Vec::new();

        // Create the bind groups.

        for j in 0..entry_layouts[i].len() {

            // Create bind group entry from rresource.
            inner_group.push(
                wgpu::BindGroupEntry {
                    binding: j as u32,
                    resource: bindings[i][j].clone(),
                }
            );

            // If all bind group entries has been created, create BindGroup.
            if j == entry_layouts[i].len() - 1 {
                result.push(device.create_bind_group(
                    &wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &bing_group_layouts[i],
                        entries: &inner_group,
                    })
                );
            }
        } // j
    } // i
    result
}

pub fn create_bind_group_layouts(device: &wgpu::Device, layout_entries: &Vec<Vec<wgpu::BindGroupLayoutEntry>>) -> Vec<wgpu::BindGroupLayout> {

    let mut bind_group_layouts: Vec<wgpu::BindGroupLayout> = Vec::new();
    for e in layout_entries.iter() {
        bind_group_layouts.push(device.create_bind_group_layout(
                &wgpu::BindGroupLayoutDescriptor {
                    entries: &e,
                    label: None,
                }
                ));
    }
    bind_group_layouts
}

pub fn draw(encoder: &mut wgpu::CommandEncoder,
            view: &wgpu::TextureView,
            depth_texture: &Texture,
            bind_groups: &Vec<wgpu::BindGroup>,
            pipeline: &wgpu::RenderPipeline,
            draw_buffer: &wgpu::Buffer,
            range: Range<u32>,
            clear: bool) {

    let mut render_pass = encoder.begin_render_pass(
            &wgpu::RenderPassDescriptor {
                label: Some("two_triangles_rendes_pass_descriptor"),
                color_attachments: &[
                    wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: match clear {
                                    true => {
                                        wgpu::LoadOp::Clear(wgpu::Color {
                                            r: 0.0,
                                            g: 0.0,
                                            b: 0.0,
                                            a: 1.0,
                                        })
                                    }
                                    false => {
                                        wgpu::LoadOp::Load
                                    }
                                },
                                store: true,
                            },
                    }
                ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                        load: match clear { true => wgpu::LoadOp::Clear(1.0), false => wgpu::LoadOp::Load },
                        store: true,
                }),
                stencil_ops: None,
                }),
    });
    
    render_pass.set_pipeline(&pipeline);
    // Set bind groups.
    for (e, bgs) in bind_groups.iter().enumerate() {
        render_pass.set_bind_group(e as u32, &bgs, &[]);
    }
    
    // Set vertex buffer.
    render_pass.set_vertex_buffer(
        0,
        draw_buffer.slice(..)
    );
    
    render_pass.draw(range, 0..1);
}
