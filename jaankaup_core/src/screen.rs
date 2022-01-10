use std::mem;
use crate::texture::Texture;

/// A struct that owns the current wgpu::SurfaceTexture and the optional depth texture.
pub struct ScreenTexture {
    surface_texture: Option<wgpu::SurfaceTexture>,
    #[allow(dead_code)]
    depth_texture: Option<Texture>,
}

impl ScreenTexture {

    /// Creates ScreenTexture without current wgpu::SurfaceTexture.
    /// A depth texture is created is create_depth_texture is true.
    pub fn init(
             &self,
             device: &wgpu::Device,
             sc_desc: &wgpu::SurfaceConfiguration,
             create_depth_texture: bool) -> Self {

        let depth_texture = if create_depth_texture {
                Some(Texture::create_depth_texture(
                    &device,
                    &sc_desc,
                    Some("depth_texture")
                    )
                )
            } else { None };

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
            panic!("ScreenTexture doesn't have a surface_testure. Consider calling the ScreenTexture::acquire_screen_texture before this method.");
        }

        mem::take(&mut self.surface_texture).unwrap().present();
    }
}
