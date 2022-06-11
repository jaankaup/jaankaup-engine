use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::collections::HashMap;
use crate::render_object::ComputeObject;
use crate::common_functions::{create_uniform_bindgroup_layout, create_buffer_bindgroup_layout};
use crate::fmm_things::{FmmParamsBuffer, PointCloud};
use crate::gpu_debugger::GpuDebugger;

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
#[allow(dead_code)]
const FMM_DOMAIN_DIMENSION: [usize ; 3] = [FMM_GLOBAL_DIMENSION[0] * FMM_INNER_DIMENSION[0],
                                           FMM_GLOBAL_DIMENSION[1] * FMM_INNER_DIMENSION[1],
                                           FMM_GLOBAL_DIMENSION[2] * FMM_INNER_DIMENSION[2]]; 
/// The total fmm cell count. 
#[allow(dead_code)]
const FMM_CELL_COUNT: usize = FMM_GLOBAL_DIMENSION[0] * FMM_INNER_DIMENSION[0] *
                              FMM_GLOBAL_DIMENSION[1] * FMM_INNER_DIMENSION[1] *
                              FMM_GLOBAL_DIMENSION[2] * FMM_INNER_DIMENSION[2];

/// Max number of arrows for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_ARROWS:     usize = 262144;

/// Max number of aabbs for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_AABBS:      usize = FMM_CELL_COUNT;

/// Max number of box frames for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_AABB_WIRES: usize =  40960;

/// Max number of renderable char elements (f32, vec3, vec4, ...) for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_CHARS:      usize = FMM_CELL_COUNT;

/// Max number of vvvvnnnn vertices reserved for gpu draw buffer.
#[allow(dead_code)]
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
pub struct FastMarchingMethod {

    /// The fmm compute object. 
    #[allow(dead_code)]
    compute_object: ComputeObject,

    /// Store general buffers here.
    #[allow(dead_code)]
    buffers: HashMap<String, wgpu::Buffer>,

    /// Fast marching method params.
    #[allow(dead_code)]
    fmm_params_buffer: FmmParamsBuffer,

    /// Global dimension for computational domain.
    #[allow(dead_code)]
    global_dimension: [u32; 3],

    /// Local dimension for computational domain.
    #[allow(dead_code)]
    local_dimension: [u32; 3],
}

impl FastMarchingMethod {

    /// Creates an instance of FastMarchingMethod.
    #[allow(dead_code)]
    pub fn init(device: &wgpu::Device,
                global_dimension: [u32; 3],
                local_dimension: [u32; 3],
                gpu_debugger: &Option<&GpuDebugger>,
                ) -> Self {

        // TODO: assertions for local and global dimension.

        // Buffer hash_map.
        let buffers: HashMap<String, wgpu::Buffer> = HashMap::new();

        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("fast_marching_method.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/fast_marching_method.wgsl"))),

                    }),
                    Some("FastMarchingMethod ComputeObject"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> prefix_params: PrefixParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<uniform> fmm_params: FmmParams;
                            create_uniform_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(2) var<storage, read_write> fmm_blocks: array<FmmBlock>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),
                              
                            // @group(0) @binding(3) var<storage, read_write> temp_prefix_sum: array<u32>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),
                              
                            // @group(0) @binding(4) var<storage,read_write> filtered_blocks: array<FmmBlock>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
                        ],
                        vec![

                            // @group(1) @binding(0) var<storage, read_write> counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE, false),

                            // @group(1) @binding(1) var<storage,read_write>  output_char: array<Char>;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // @group(1) @binding(2) var<storage,read_write>  output_arrow: array<Arrow>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                            // @group(1) @binding(3) var<storage,read_write>  output_aabb: array<AABB>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            // @group(1) @binding(4) var<storage,read_write>  output_aabb_wire: array<AABB>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
                        ],

                    ],
                    &"main".to_string(),
                    Some(vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::COMPUTE,
                        range: 0..4,
                    }]),
        );

        let fmm_params_buffer = FmmParamsBuffer::create(device, global_dimension, local_dimension);
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
            compute_object: compute_object,
            buffers: buffers,
            fmm_params_buffer: fmm_params_buffer, 
            global_dimension: global_dimension,
            local_dimension: local_dimension,
        }
    }

    /// Create the initial interface using point cloud data.
    /// TODO: point sampling method.
    #[allow(dead_code)]
    pub fn initialize_interface_pc(&self, _pc: &PointCloud) {

    }

    /// Define the initial band cells and update the band values.
    #[allow(dead_code)]
    pub fn create_initial_band_cells(&self) {

    }

    /// Update all band point counts from global computational domain. 
    #[allow(dead_code)]
    pub fn calculate_band_point_counts(&self) {

    }

    // 
    #[allow(dead_code)]
    pub fn update_neighbors(&self) {

    }

    #[allow(dead_code)]
    pub fn expand_interface(&self) {

    }

    // /// From each active fmm block, find smallest band point and change the tag to known. 
    // /// Add the known cells to update list.
    // pub fn poll_band(&self) {

    // }

    // fn create_buffers(device: &wgpu::Device, buffers: &HashMap<String, wgpu::Buffer>) {

    //     // let pc_sample_data = 
    //     //     buffers.insert(
    //     //         "pc_sample_data".to_string(),
    //     //         buffer_from_data::<FmmCellPc>(
    //     //         &configuration.device,
    //     //         &vec![FmmCellPc {
    //     //             tag: 0,
    //     //             value: 10000000,
    //     //             color: 0,
    //     //             // padding: 0,
    //     //         } ; FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z * FMM_INNER_X * FMM_INNER_Y * FMM_INNER_Z],
    //     //         wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    //     //         None)
    //     //     );
    // }
}
