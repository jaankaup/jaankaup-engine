use crate::texture::Texture;

pub struct ScreenTexture {
    surface_texture: Option<wgpu::SurfaceTexture>,
    depth_texture: Option<Texture>,
}

impl ScreenTexture {
    pub fn init(device: &wgpu::Device,
             sc_desc: &wgpu::SurfaceConfiguration,
             surface: &wgpu::Surface,
             depth_texture: bool) -> Self {

        Self {
            surface_texture: None,
            depth_texture: None,
        }

    }
}
