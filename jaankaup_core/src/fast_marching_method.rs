use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FmmCellPointCloud {
    tag: u32, 
    value: u32, 
    color: u32, 
    update: u32, 
}

/// Struct for parallel fast marching method. 
struct FastMarchingMethod {

}

impl FastMarchingMethod {

    pub fn init(device: &wgpu::Device,
                fmm_params_buffer: &wgpu::Buffer,
                fmm_data: &wgpu::Buffer,
                ) -> Self {

        Self {

        }

    }

    /// Define the initial band points and update the band values.
    pub fn create_initial_band_points(&self) {

    }

    /// 
    pub fn refresh_neighbor_hood(&self) {

    }

    /// From each active fmm block, find smallest band point and change the tag to known. 
    /// Add the known cells to update list.
    pub fn poll_band(&self) {

    }
}
