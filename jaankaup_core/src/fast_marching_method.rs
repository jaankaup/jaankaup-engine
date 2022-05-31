use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FmmCellPointCloud {
    tag: u32,
    value: f32,
    color: u32,
    update: u32,
}

/// Global dimension. 
const FMM_GLOBAL_DIMENSION: [usize ; 3] = [32, 8, 32]; 

/// Inner dimension.
const FMM_INNER_DIMENSION: [usize ; 3] = [4, 4, 4]; 

/// Computational domain dimension. 
const FMM_DOMAIN_DIMENSION: [usize ; 3] = [FMM_GLOBAL_DIMENSION[0] * FMM_INNER_DIMENSION[0],
                                           FMM_GLOBAL_DIMENSION[1] * FMM_INNER_DIMENSION[1],
                                           FMM_GLOBAL_DIMENSION[2] * FMM_INNER_DIMENSION[2]]; 
/// The total fmm cell count. 
const FMM_CELL_COUNT: usize = FMM_GLOBAL_DIMENSION[0] * FMM_INNER_DIMENSION[0] *
                              FMM_GLOBAL_DIMENSION[1] * FMM_INNER_DIMENSION[1] *
                              FMM_GLOBAL_DIMENSION[2] * FMM_INNER_DIMENSION[2];

/// Max number of arrows for gpu debugger.
const MAX_NUMBER_OF_ARROWS:     usize = 262144;

/// Max number of aabbs for gpu debugger.
const MAX_NUMBER_OF_AABBS:      usize = FMM_CELL_COUNT;

/// Max number of box frames for gpu debugger.
const MAX_NUMBER_OF_AABB_WIRES: usize =  40960;

/// Max number of renderable char elements (f32, vec3, vec4, ...) for gpu debugger.
const MAX_NUMBER_OF_CHARS:      usize = FMM_CELL_COUNT;

/// Max number of vvvvnnnn vertices reserved for gpu draw buffer.
const MAX_NUMBER_OF_VVVVNNNN: usize =  2000000;

// 
// Temp prefix sum array.
//   
// 
//
//


// TODO list:
//
// * initial wave front
// *  

/// Struct for parallel fast marching method. 
struct FastMarchingMethod {

    /// Store general buffers here.
    buffers: HashMap<String, wgpu::Buffer>,
}

impl FastMarchingMethod {

    pub fn init(device: &wgpu::Device,
                fmm_params_buffer: &wgpu::Buffer,
                fmm_data: &wgpu::Buffer,
                ) -> Self {

        // Buffer hash_map.
        let mut buffers: HashMap<String, wgpu::Buffer> = HashMap::new();

        // create_buffers(device: &wgpu::Device)
        // let pc_sample_data = 
        //     buffers.insert(
        //         "pc_sample_data".to_string(),
        //         buffer_from_data::<FmmCellPc>(
        //         &configuration.device,
        //         &vec![FmmCellPc {
        //             tag: 0,
        //             value: 10000000,
        //             color: 0,
        //             // padding: 0,
        //         } ; FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z * FMM_INNER_X * FMM_INNER_Y * FMM_INNER_Z],
        //         wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        //         None)
        //     );

        Self {
            buffers: buffers,

        }
    }

    /// Create the initial interface from given data.
    pub fn initialize_initial_interface(&self) {

    }

    /// Define the initial band cells and update the band values.
    pub fn create_initial_band_cells(&self) {

    }

    /// Update all band point counts from global computational domain. 
    pub fn calculate_band_point_counts(&self) {

    }

    /// 
    pub fn refresh_neighbors(&self) {

    }

    /// From each active fmm block, find smallest band point and change the tag to known. 
    /// Add the known cells to update list.
    pub fn poll_band(&self) {

    }

    fn create_buffers(device: &wgpu::Device, buffers: &HashMap<String, wgpu::Buffer>) {

        // let pc_sample_data = 
        //     buffers.insert(
        //         "pc_sample_data".to_string(),
        //         buffer_from_data::<FmmCellPc>(
        //         &configuration.device,
        //         &vec![FmmCellPc {
        //             tag: 0,
        //             value: 10000000,
        //             color: 0,
        //             // padding: 0,
        //         } ; FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z * FMM_INNER_X * FMM_INNER_Y * FMM_INNER_Z],
        //         wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        //         None)
        //     );
    }
}
