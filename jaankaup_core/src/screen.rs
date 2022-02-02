use std::mem;
use crate::texture::Texture;

/// A struct that owns the current wgpu::SurfaceTexture and the optional depth texture.
/// TODO: getter_functions for attributes
pub struct ScreenTexture {
    pub surface_texture: Option<wgpu::SurfaceTexture>,
    #[allow(dead_code)]
    pub depth_texture: Option<Texture>,
}

impl ScreenTexture {

    /// Creates ScreenTexture without current wgpu::SurfaceTexture.
    /// A depth texture is created if create_depth_texture is true.
    pub fn init(
             device: &wgpu::Device,
             sc_desc: &wgpu::SurfaceConfiguration,
             create_depth_texture: bool) -> Self {

        log::info!("Screen::init.");

        let depth_texture = if create_depth_texture {
                Some(Texture::create_depth_texture(
                    &device,
                    &sc_desc,
                    Some("depth_texture")
                    )
                )
            } else { None };

        log::info!("Created depth_texture.");

        Self {
            surface_texture: None,
            depth_texture: depth_texture,
        }
    }

    /// Acquire the current screen texture.
    pub fn acquire_screen_texture(
            &mut self,
            device: &wgpu::Device,
            sc_desc: &wgpu::SurfaceConfiguration,
            surface: &wgpu::Surface) {

        let frame = match surface.get_current_texture() {
            Ok(frame) => {frame},
            Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                surface.configure(&device, &sc_desc);
                let surface_texture = surface.get_current_texture().expect("Failed to acquire next texture");
                surface_texture
            },
            Err(wgpu::SurfaceError::Timeout) => panic!("Timeout occurred while acquiring the next frame texture."),
            Err(wgpu::SurfaceError::OutOfMemory) => panic!("OutOfMemory occurred while acquiring the next frame texture."),
        };
        self.surface_texture = Some(frame);
    }

    /// This must be called so the texture can be actually rendered to the screen. Call this method
    /// after wgpu::Queue::submit.
    //pub fn prepare_for_rendering(&mut self) {
    pub fn prepare_for_rendering(&mut self) {

        if self.surface_texture.is_none() {
            panic!("ScreenTexture doesn't have a surface_texture. Consider calling the ScreenTexture::acquire_screen_texture before this method.");
        }

        mem::take(&mut self.surface_texture).unwrap().present();
    }
    
    
    // pub fn draw(encoder: &mut wgpu::CommandEncoder,
    //         //frame: &wgpu::SurfaceFrame,
    //         view: &wgpu::TextureView,
    //         depth_texture: &jaankaup::Texture,
    //         bind_groups: &Vec<wgpu::BindGroup>,
    //         pipeline: &wgpu::RenderPipeline,
    //         draw_buffer: &wgpu::Buffer,
    //         range: Range<u32>,
    //         clear: bool) {

    //     let mut render_pass = encoder.begin_render_pass(
    //             &wgpu::RenderPassDescriptor {
    //                 label: Some("two_triangles_rendes_pass_descriptor"),
    //                 color_attachments: &[
    //                     wgpu::RenderPassColorAttachment {
    //                             view: &view,
    //                             resolve_target: None,
    //                             ops: wgpu::Operations {
    //                                 load: match clear {
    //                                     true => {
    //                                         wgpu::LoadOp::Clear(wgpu::Color {
    //                                             r: 0.0,
    //                                             g: 0.0,
    //                                             b: 0.0,
    //                                             a: 1.0,
    //                                         })
    //                                     }
    //                                     false => {
    //                                         wgpu::LoadOp::Load
    //                                     }
    //                                 },
    //                                 store: true,
    //                             },
    //                     }
    //                 ],
    //             depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
    //                 view: &depth_texture.view,
    //                 depth_ops: Some(wgpu::Operations {
    //                         load: match clear { true => wgpu::LoadOp::Clear(1.0), false => wgpu::LoadOp::Load },
    //                         store: true,
    //                 }),
    //                 stencil_ops: None,
    //                 }),
    //         });

    //         render_pass.set_pipeline(&pipeline);
    //         // Set bind groups.
    //         for (e, bgs) in bind_groups.iter().enumerate() {
    //             render_pass.set_bind_group(e as u32, &bgs, &[]);
    //         }

    //         // Set vertex buffer.
    //         render_pass.set_vertex_buffer(
    //             0,
    //             draw_buffer.slice(..)
    //         );

    //         render_pass.draw(range, 0..1);
    // }
}
