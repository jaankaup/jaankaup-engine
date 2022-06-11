use crate::common_functions::udiv_up_safe32;
use crate::fmm_things::FmmValueFixer;
use crate::fmm_things::PointCloudParamsBuffer;
// use crate::fmm_things::PointCloudParams;
use std::convert::TryInto;
use crate::buffer::buffer_from_data;
use crate::fmm_things::FmmCellPc;
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::collections::HashMap;
use crate::render_object::{ComputeObject, create_bind_groups};
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

    /// Fmm cell data.
    #[allow(dead_code)]
    fmm_data: wgpu::Buffer,

    /// Point data to fmm interface compute object.
    #[allow(dead_code)]
    pc_to_interface_compute_object: ComputeObject,

    /// Point data to fmm interface bing groups.
    #[allow(dead_code)]
    point_to_interface_bind_groups: Option<Vec<wgpu::BindGroup>>,

    /// Preprocessor for fmm values.
    #[allow(dead_code)]
    fmm_value_fixer: FmmValueFixer,
}

impl FastMarchingMethod {

    /// Creates an instance of FastMarchingMethod.
    #[allow(dead_code)]
    pub fn init(device: &wgpu::Device,
                global_dimension: [u32; 3],
                local_dimension: [u32; 3],
                _gpu_debugger: &Option<&GpuDebugger>,
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
                                                
        let fmm_data = buffer_from_data::<FmmCellPc>(
                &device,
                &vec![FmmCellPc {
                    tag: 0,
                    value: 10000000,
                    color: 0,
                    // padding: 0,
                } ; (global_dimension[0] * global_dimension[1] * global_dimension[2] * local_dimension[0] * local_dimension[1] * local_dimension[2]).try_into().unwrap() ],
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                None
        );

        let pc_to_interface_compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("pc_to_interface_fmm.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/pc_to_interface_fmm.wgsl"))),

                    }),
                    Some("Cloud data to fmm Compute object fmm"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> fmm_params: FmmParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),
 
                            // @group(0) @binding(1) var<uniform> point_cloud_params: PointCloudParams;
                            create_uniform_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),
 
                            // @group(0) @binding(2) var<storage, read_write> fmm_data: array<FmmCellPc>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),
 
                            // @group(0) @binding(3) var<storage, read_write> point_data: array<VVVC>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),
 
                            // // @group(0) @binding(4) var<storage,read_write> counter: array<atomic<u32>>;
                            // create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
 
                            // // @group(0) @binding(5) var<storage,read_write> output_char: array<Char>;
                            // create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),
 
                            // // @group(0) @binding(6) var<storage,read_write> output_arrow: array<Arrow>;
                            // create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
 
                            // // @group(0) @binding(7) var<storage,read_write> output_aabb: array<AABB>;
                            // create_buffer_bindgroup_layout(7, wgpu::ShaderStages::COMPUTE, false),
 
                            // // @group(0) @binding(8) var<storage,read_write> output_aabb_wire: array<AABB>;
                            // create_buffer_bindgroup_layout(8, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None
        );

        let fmm_value_fixer = FmmValueFixer::init(&device,
                                                  &fmm_params_buffer.get_buffer(),
                                                  &fmm_data,
        );

        Self {
            compute_object: compute_object,
            buffers: buffers,
            fmm_params_buffer: fmm_params_buffer, 
            global_dimension: global_dimension,
            local_dimension: local_dimension,
            fmm_data: fmm_data,
            pc_to_interface_compute_object: pc_to_interface_compute_object,
            point_to_interface_bind_groups: None,
            fmm_value_fixer: fmm_value_fixer,
        }
    }

    /// Create the initial interface using point cloud data.
    /// TODO: point sampling method.
    #[allow(dead_code)]
    pub fn initialize_interface_pc(&self, encoder: &mut wgpu::CommandEncoder, pc: &PointCloud) {

        assert!(!self.point_to_interface_bind_groups.is_none(), "Consider calling add_point_cloud_data before this function."); 

         self.pc_to_interface_compute_object.dispatch(
             &self.point_to_interface_bind_groups.as_ref().unwrap(),
             encoder,
             1, 1, 1, Some("Point data to interface dispatch")
         );

        self.fmm_value_fixer.dispatch(encoder, [udiv_up_safe32(pc.get_point_count(), 1024) + 1, 1, 1]);
    }

    pub fn add_point_cloud_data(&mut self,
                                device: &wgpu::Device,
                                pc_data: &wgpu::Buffer,
                                point_cloud_params_buffer: &PointCloudParamsBuffer) {

        let point_to_interface_bind_groups = create_bind_groups(
                &device,
                &self.pc_to_interface_compute_object.bind_group_layout_entries,
                &self.pc_to_interface_compute_object.bind_group_layouts,
                &vec![
                    vec![
                    &self.fmm_params_buffer.get_buffer().as_entire_binding(),
                    &point_cloud_params_buffer.get_buffer().as_entire_binding(),
                    &self.fmm_data.as_entire_binding(),
                    &pc_data.as_entire_binding(),
                    // &gpu_debugger.get_element_counter_buffer().as_entire_binding(),
                    // &gpu_debugger.get_output_chars_buffer().as_entire_binding(),
                    // &gpu_debugger.get_output_arrows_buffer().as_entire_binding(),
                    // &gpu_debugger.get_output_aabbs_buffer().as_entire_binding(),
                    // &gpu_debugger.get_output_aabb_wires_buffer().as_entire_binding(),
                    ],
                ]
        );
        self.point_to_interface_bind_groups = Some(point_to_interface_bind_groups);

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

    pub fn get_fmm_data_buffer(&self) -> &wgpu::Buffer {
        &self.fmm_data
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
