use bytemuck::{Zeroable, Pod};
use crate::buffer::buffer_from_data;

#[repr(C)]
#[derive(Clone, Copy,Zeroable,Pod)]
pub struct McUniform {
    pub base_position: cgmath::Vector4<f32>,
    pub isovalue: f32,
    pub cube_length: f32,
    pub future_usage1: f32,
    pub future_usage2: f32,
}
