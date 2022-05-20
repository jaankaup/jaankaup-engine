use bytemuck::{Pod, Zeroable};
use crate::impl_convert;
use crate::misc::Convert2Vec;
use crate::buffer::buffer_from_data;

/// Tag value for Far cell.
const FAR: u32      = 0;

/// Tag value for band cell whose value is not yet known.
const BAND_NEW: u32 = 1;

/// Tag value for a band cell.
const BAND: u32     = 2;

/// Tag value for a known cell.
const KNOWN: u32    = 3;

/// Tag value for a cell outside the computational domain.
const OUTSIDE: u32  = 4;

/// Basic data for the fast marching method.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FMMCell {
    tag: u32,
    value: f32,
}

/// A struct for 3d grid general information.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ComputationalDomain {
    global_dimension: [u32; 3],
    local_dimension:  [u32; 3],
}

impl_convert!{FMMCell}
impl_convert!{ComputationalDomain}

/// A struct for Computational domain data, operations and buffer.
pub struct ComputationalDomainBuffer {
    computational_domain: ComputationalDomain,
    buffer: wgpu::Buffer,
}

impl ComputationalDomainBuffer {
    
    /// Initialize and create ComputationalDomainBuffer object.
    pub fn create(device: &wgpu::Device,
                  global_dimension: [u32; 3],
                  local_dimension: [u32; 3]) -> Self {

        // TODO: asserts

        let domain = ComputationalDomain {
            global_dimension,
            local_dimension,
        };

        let buf = buffer_from_data::<ComputationalDomain>(
                  &device,
                  &vec![domain],
                  wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                  Some("Computational domain wgpu::buffer.")
        );

        Self {
            computational_domain: domain,
            buffer: buf,
        }
    }

    /// Update buffer.
    fn update(&self, queue: &wgpu::Queue) {
        
        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&[self.computational_domain])
        );
    }

    pub fn update_global_dimension(&mut self, queue: &wgpu::Queue, global_dimension: [u32; 3]) {
        // TODO: asserts
        self.computational_domain.global_dimension = global_dimension;
    }
}
