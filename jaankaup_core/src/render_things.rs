// use jaankaup_core::wgpu;
use bytemuck::{Pod, Zeroable};
use crate::impl_convert;
use crate::misc::Convert2Vec;
use crate::buffer::buffer_from_data;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RenderParams {
    scale_factor: f32,
}

pub struct RenderParamBuffer {
    _params: RenderParams,
    buffer: wgpu::Buffer,
}

impl RenderParamBuffer {
    pub fn create(device: &wgpu::Device, scale_factor: f32 ) -> Self {

        let params = RenderParams { scale_factor: scale_factor, };

        let buf = buffer_from_data::<RenderParams>(
                  &device,
                  &vec![params],
                  wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                  Some("render params buffer.")
        );

        Self {
            _params: params,
            buffer: buf,
        }
    }

    pub fn get_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}


#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Light {
    light_pos: [f32; 3],
    material_shininess: f32,
    material_spec_color: [f32; 3],
    ambient_coeffience: f32,
    light_color: [f32; 3],
    attentuation_factor: f32,
}

pub struct LightBuffer {
    _light: Light,
    buffer: wgpu::Buffer,
}

impl LightBuffer {
    pub fn create(device: &wgpu::Device,
                  position: [f32; 3],
                  material_spec_color: [u8; 3], 
                  light_color: [u8; 3], 
                  material_shininess: f32,
                  ambient_coeffience: f32,
                  attentuation_factor: f32) -> Self {

        let light = Light {
            light_pos: position,
            material_shininess: material_shininess,
            material_spec_color: [material_spec_color[0] as f32 / 255.0,
                                  material_spec_color[1] as f32 / 255.0,
                                  material_spec_color[2] as f32 / 255.0],
            ambient_coeffience: ambient_coeffience,
            light_color: [light_color[0] as f32 / 255.0, light_color[1] as f32 / 255.0, light_color[2] as f32 / 255.0],
            attentuation_factor: attentuation_factor,
        };

        let buf = buffer_from_data::<Light>(
                  &device,
                  &vec![light],
                  wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, // | wgpu::BufferUsages::STORAGE 
                  Some("light buffer.")
        );

        Self {
            _light: light,
            buffer: buf,
        }
    }

    pub fn get_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

impl_convert!{Light}

// TODO: Replace with RenderParams
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ScaleFactor {
    factor: f32,
}
