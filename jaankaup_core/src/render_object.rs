use core::ops::Range;
use crate::texture::Texture;

pub trait RenderObject: Sized + 'static {
    fn init(device: &wgpu::Device,
            sc_desc: &wgpu::SurfaceConfiguration,
            wgsl_module: &wgpu::ShaderModule,
            ) -> Self; 

    fn draw(
        label: wgpu::Label,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_texture: Option<&Texture>,
        bind_groups: &Vec<wgpu::BindGroup>,
        pipeline: &wgpu::RenderPipeline,
        draw_buffer: &wgpu::Buffer,
        range: Range<u32>,
        clear: bool) {

            let mut render_pass = encoder.begin_render_pass(
                    &wgpu::RenderPassDescriptor {
                        label: label,
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
                    depth_stencil_attachment: if depth_texture.is_none() { None } else {
                            Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth_texture.unwrap().view,
                            depth_ops: Some(wgpu::Operations {
                                    load: match clear { true => wgpu::LoadOp::Clear(1.0), false => wgpu::LoadOp::Load },
                                    store: true,
                            }),
                            stencil_ops: None,
                            })},
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

    fn create_bind_group_layouts(device: &wgpu::Device, layout_entries: &Vec<Vec<wgpu::BindGroupLayoutEntry>>) -> Vec<wgpu::BindGroupLayout> {

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

    fn create_bind_groups(device: &wgpu::Device,
                          entry_layouts: &Vec<Vec<wgpu::BindGroupLayoutEntry>>,
                          bindings: &Vec<Vec<&wgpu::BindingResource>>)
                        -> Vec<wgpu::BindGroup> {

        // The created bindgroups.
        let mut result: Vec<wgpu::BindGroup> = Vec::new();

        // Add Binding resources to the bind group.
        for i in 0..entry_layouts.len() {

            let mut inner_group: Vec<wgpu::BindGroupEntry> = Vec::new();

            // Create the bind groups. TODO: this should be created only once (add to struct
            // attribute).
            let layouts = Self::create_bind_group_layouts(&device, &entry_layouts);

            for j in 0..entry_layouts[i].len() {

                // Create bind group entry from resource.
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
                            layout: &layouts[i],
                            entries: &inner_group,
                        })
                    );
                }
            } // j
        } // i
        result
    }
}
