use bytemuck::{Pod, Zeroable};
use crate::impl_convert;
use crate::misc::Convert2Vec;

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

struct ComputationalDomainBuffer {
    computational_domain: ComputationalDomain,
    buffer: wgpu::Buffer,
}
