use std::mem::size_of;
use crate::fmm_things::{FmmPrefixParams, FmmCellPc, FmmBlock};
use crate::common_functions::{udiv_up_safe32, encode_rgba_u32};
use crate::fmm_things::FmmValueFixer;
use crate::fmm_things::PointCloudParamsBuffer;
use std::convert::TryInto;
use crate::buffer::buffer_from_data;
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use std::collections::HashMap;
use crate::render_object::{ComputeObject, create_bind_groups};
use crate::common_functions::{create_uniform_bindgroup_layout, create_buffer_bindgroup_layout};
use crate::fmm_things::{FmmParamsBuffer, PointCloud};
use crate::gpu_debugger::GpuDebugger;
use crate::histogram::Histogram;
use crate::gpu_timer::GpuTimer;

#[allow(dead_code)]
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum FmmState {
    FindNeighbors,
    SolveQuadratic,
    FilterActiveBlocks,
    Reduce,
}

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

/// Struct for parallel fast marching method. 
pub struct FastMarchingMethod {

    /// The fmm compute object. 
    #[allow(dead_code)]
    compute_object: ComputeObject,

    ///fmm compute object bindgroups. 
    #[allow(dead_code)]
    compute_object_bind_groups: Vec<wgpu::BindGroup>,

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

    /// Temporary data for prefix sum.
    #[allow(dead_code)]
    fmm_blocks: wgpu::Buffer,

    /// Temporary data for prefix sum.
    #[allow(dead_code)]
    temporary_fmm_data: wgpu::Buffer,

    /// Point data to fmm interface compute object.
    #[allow(dead_code)]
    pc_to_interface_compute_object: ComputeObject,

    /// Point data to fmm interface bing groups.
    #[allow(dead_code)]
    pc_to_interface_bind_groups: Option<Vec<wgpu::BindGroup>>,

    /// Preprocessor for fmm values.
    #[allow(dead_code)]
    fmm_value_fixer: FmmValueFixer,

    /// Prefix params.
    #[allow(dead_code)]
    fmm_prefix_params: wgpu::Buffer,

    /// Temporary data for prefix sum.
    #[allow(dead_code)]
    prefix_temp_array: wgpu::Buffer,

    /// Temporary data for prefix sum.
    #[allow(dead_code)]
    update_band_counts_compute_object: ComputeObject,

    /// Temporary data for prefix sum.
    #[allow(dead_code)]
    update_band_counts_compute_object_bind_groups: Vec<wgpu::BindGroup>,

    /// A histogram for fmm purposes.
    #[allow(dead_code)]
    fmm_histogram: Histogram,

    first_time: bool,
    fmm_state: FmmState,

    count_known_cells: ComputeObject,
    _count_known_cells_bg: Vec<wgpu::BindGroup>,
    count_band_cells: ComputeObject,
    _count_band_cells_bg: Vec<wgpu::BindGroup>,
    create_init_band: ComputeObject,
    _create_init_band_gb: Vec<wgpu::BindGroup>,
    solve_quadratic: ComputeObject,
    solve_quadratic2: ComputeObject,
    prefix1: ComputeObject,
    prefix2: ComputeObject,
    prefix_gather: ComputeObject,
    prefix_sum_aux: ComputeObject,
    reduce: ComputeObject,
    find_neighbors: ComputeObject,
    fmm_alg_visualizer: ComputeObject, 
    iteration_counter: u32,
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

        let iteration_counter = 0;
        let fmm_state = FmmState::Reduce;

        // Buffer hash_map. DO we need this?
        let buffers: HashMap<String, wgpu::Buffer> = HashMap::new();

        // Fmm histogram.
        let fmm_histogram = Histogram::init(&device, &vec![0; 5]);

        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
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

                            // @group(0) @binding(5) var<storage,read_write>  fmm_data: array<TempData>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
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

        let number_of_fmm_blocks: usize = (global_dimension[0] * global_dimension[1] * global_dimension[2]).try_into().unwrap();
        let number_of_fmm_cells: usize = (number_of_fmm_blocks as u32 * local_dimension[0] * local_dimension[1] * local_dimension[2]).try_into().unwrap();

        let mut fmm_data = vec![FmmCellPc { tag: 0, value: 100000.0, color: encode_rgba_u32(0, 0, 0, 255), } ; number_of_fmm_cells + 1];

        // The outside value.
        fmm_data[number_of_fmm_cells] = FmmCellPc { tag: 4, value: 100000.0, color: encode_rgba_u32(0, 0, 0, 255), };
                                                
        // Fast marching method cell data.
        print!("Creating fmm_data buffer.  ");
        let fmm_data = buffer_from_data::<FmmCellPc>(
                &device,
                &fmm_data,
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                None
        );
        println!("OK");

        // Fast marching method cell data.
        let mut fmm_blocks_vec: Vec<FmmBlock> = Vec::with_capacity(number_of_fmm_blocks);

        // println!("number_of_fmm_blocks == {:?}", number_of_fmm_blocks);
        
        // Create the initial fmm_blocks.
        for i in 0..number_of_fmm_blocks {
            fmm_blocks_vec.push(FmmBlock { index: i as u32, number_of_band_points: 0, });
        }

        let fmm_blocks = buffer_from_data::<FmmBlock>(
                &device,
                &fmm_blocks_vec,
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                None
        );

        print!("Creating temporary fmm data buffer.  ");
        // Create temp data buffer.
        let temporary_fmm_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fmm temporary data buffer"),
            size: (number_of_fmm_cells * size_of::<FmmBlock>()) as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        println!("OK");


        print!("Creating prefix_temp_array.  ");
        // Prefix temp array.
        let prefix_temp_array = buffer_from_data::<u32>(
                &device,
                &vec![0 ; number_of_fmm_cells + 4096 as usize],
                //&vec![0 ; number_of_fmm_cells + 2048 as usize],
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                None
        );
        println!("OK");

        // Prefix sum params.
        let fmm_prefix_params = buffer_from_data::<FmmPrefixParams>(
                &device,
                &vec![FmmPrefixParams {
                     data_start_index: 0,
                     data_end_index: (number_of_fmm_cells - 1) as u32,
                     exclusive_parts_start_index: number_of_fmm_cells as u32,
                     exclusive_parts_end_index: number_of_fmm_cells as u32 + 2048,
                 }],
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                None
        );
        println!("fmm_prefix_params = {:?}", FmmPrefixParams { data_start_index: 0,
                                                               data_end_index: (number_of_fmm_cells - 1) as u32,
                                                               exclusive_parts_start_index: number_of_fmm_cells as u32,
                                                               exclusive_parts_end_index: number_of_fmm_cells as u32 + 2048,});

        let compute_object_bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &fmm_prefix_params.as_entire_binding(),
                        &fmm_params_buffer.get_buffer().as_entire_binding(),
                        &fmm_blocks.as_entire_binding(),
                        &prefix_temp_array.as_entire_binding(),
                        &temporary_fmm_data.as_entire_binding(),
                        &fmm_data.as_entire_binding(),
                        &fmm_histogram.get_histogram_buffer().as_entire_binding(),
                    ],
                    vec![
                        &gpu_debugger.unwrap().get_element_counter_buffer().as_entire_binding(),
                        &gpu_debugger.unwrap().get_output_chars_buffer().as_entire_binding(),
                        &gpu_debugger.unwrap().get_output_arrows_buffer().as_entire_binding(),
                        &gpu_debugger.unwrap().get_output_aabbs_buffer().as_entire_binding(),
                        &gpu_debugger.unwrap().get_output_aabb_wires_buffer().as_entire_binding(),
                    ],
                ]
        );

        let pc_to_interface_compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
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
                                                  &prefix_temp_array,
        );

        // Update band counts shader.

        let update_band_counts_compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("calculate_band_point_counts.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/calculate_band_point_counts.wgsl"))),

                    }),
                    Some("calculate_band_point_counts Compute objec"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> fmm_params: FmmParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),
 
                            // @group(0) @binding(1) var<storage,read_write> fmm_data: array<FmmCellPc>;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),
 
                            // @group(0) @binding(2) var<storage, read_write> fmm_blocks: array<FmmBlock>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),
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
                    None
        );

        let update_band_counts_compute_object_bind_groups = create_bind_groups(
                &device,
                &update_band_counts_compute_object.bind_group_layout_entries,
                &update_band_counts_compute_object.bind_group_layouts,
                &vec![
                    vec![
                    &fmm_params_buffer.get_buffer().as_entire_binding(),
                    &fmm_data.as_entire_binding(),
                    &fmm_blocks.as_entire_binding(),
                    ],
                    vec![
                    &gpu_debugger.unwrap().get_element_counter_buffer().as_entire_binding(),
                    &gpu_debugger.unwrap().get_output_chars_buffer().as_entire_binding(),
                    &gpu_debugger.unwrap().get_output_arrows_buffer().as_entire_binding(),
                    &gpu_debugger.unwrap().get_output_aabbs_buffer().as_entire_binding(),
                    &gpu_debugger.unwrap().get_output_aabb_wires_buffer().as_entire_binding(),
                    ],
                ]
        );

        let (count_known_cells, count_known_cells_bg) = Self::create_gather_known_cells_co(
                                        &device,
                                        &fmm_prefix_params,
                                        &fmm_params_buffer.get_buffer(),
                                        &fmm_blocks,
                                        &prefix_temp_array,
                                        &temporary_fmm_data,
                                        &fmm_data,
                                        &fmm_histogram.get_histogram_buffer(),
        );

        let (count_band_cells, count_band_cells_bg) = Self::create_gather_band_cells_co(
                                        &device,
                                        &fmm_prefix_params,
                                        &fmm_params_buffer.get_buffer(),
                                        &fmm_blocks,
                                        &prefix_temp_array,
                                        &temporary_fmm_data,
                                        &fmm_data,
                                        &fmm_histogram.get_histogram_buffer()
        );

        let (create_init_band, create_init_band_gb) = Self::create_initial_band_co(
                                        &device,
                                        &fmm_prefix_params,
                                        &fmm_params_buffer.get_buffer(),
                                        &fmm_blocks,
                                        &prefix_temp_array,
                                        &temporary_fmm_data,
                                        &fmm_data,
                                        &fmm_histogram.get_histogram_buffer()
        );

        let (solve_quadratic, _) = Self::create_solve_quadratic_co(
                                        &device,
                                        &fmm_prefix_params,
                                        &fmm_params_buffer.get_buffer(),
                                        &fmm_blocks,
                                        &prefix_temp_array,
                                        &temporary_fmm_data,
                                        &fmm_data,
                                        &fmm_histogram.get_histogram_buffer()
        );

        let (solve_quadratic2, _) = Self::create_solve_quadratic2_co(
                                        &device,
                                        &fmm_prefix_params,
                                        &fmm_params_buffer.get_buffer(),
                                        &fmm_blocks,
                                        &prefix_temp_array,
                                        &temporary_fmm_data,
                                        &fmm_data,
                                        &fmm_histogram.get_histogram_buffer()
        );

        let (prefix1, _) = Self::create_prefix_sum_1(
                                        &device,
                                        &fmm_prefix_params,
                                        &fmm_params_buffer.get_buffer(),
                                        &fmm_blocks,
                                        &prefix_temp_array,
                                        &temporary_fmm_data,
                                        &fmm_data,
                                        &fmm_histogram.get_histogram_buffer()
        );

        let (prefix2, _) = Self::create_prefix_sum_2(
                                        &device,
                                        &fmm_prefix_params,
                                        &fmm_params_buffer.get_buffer(),
                                        &fmm_blocks,
                                        &prefix_temp_array,
                                        &temporary_fmm_data,
                                        &fmm_data,
                                        &fmm_histogram.get_histogram_buffer()
        );

        let (prefix_gather, _) = Self::create_prefix_gather(
                                        &device,
                                        &fmm_prefix_params,
                                        &fmm_params_buffer.get_buffer(),
                                        &fmm_blocks,
                                        &prefix_temp_array,
                                        &temporary_fmm_data,
                                        &fmm_data,
                                        &fmm_histogram.get_histogram_buffer()
        );

        let (prefix_sum_aux, _) = Self::create_prefix_sum_aux(
                                        &device,
                                        &fmm_prefix_params,
                                        &fmm_params_buffer.get_buffer(),
                                        &fmm_blocks,
                                        &prefix_temp_array,
                                        &temporary_fmm_data,
                                        &fmm_data,
                                        &fmm_histogram.get_histogram_buffer()
        );

        let (reduce, _) = Self::create_reduce(
                                        &device,
                                        &fmm_prefix_params,
                                        &fmm_params_buffer.get_buffer(),
                                        &fmm_blocks,
                                        &prefix_temp_array,
                                        &temporary_fmm_data,
                                        &fmm_data,
                                        &fmm_histogram.get_histogram_buffer()
        );

        let (find_neighbors, _) = Self::create_find_neighbors(
                                        &device,
                                        &fmm_prefix_params,
                                        &fmm_params_buffer.get_buffer(),
                                        &fmm_blocks,
                                        &prefix_temp_array,
                                        &temporary_fmm_data,
                                        &fmm_data,
                                        &fmm_histogram.get_histogram_buffer()
        );
        
        let fmm_alg_visualizer = Self::create_fmm_alg_visualizer(
                                        &device,
                                        &fmm_prefix_params,
                                        &fmm_params_buffer.get_buffer(),
                                        &fmm_blocks,
                                        &prefix_temp_array,
                                        &temporary_fmm_data,
                                        &fmm_data,
                                        &fmm_histogram.get_histogram_buffer()
        );

        Self {
            compute_object: compute_object,
            compute_object_bind_groups: compute_object_bind_groups,
            buffers: buffers,
            fmm_params_buffer: fmm_params_buffer, 
            global_dimension: global_dimension,
            local_dimension: local_dimension,
            fmm_data: fmm_data,
            fmm_blocks: fmm_blocks,
            temporary_fmm_data: temporary_fmm_data,
            pc_to_interface_compute_object: pc_to_interface_compute_object,
            pc_to_interface_bind_groups: None,
            fmm_value_fixer: fmm_value_fixer,
            fmm_prefix_params: fmm_prefix_params,
            prefix_temp_array: prefix_temp_array,
            update_band_counts_compute_object: update_band_counts_compute_object,
            update_band_counts_compute_object_bind_groups: update_band_counts_compute_object_bind_groups,
            fmm_histogram: fmm_histogram,
            first_time: true,
            fmm_state: fmm_state,
            count_known_cells: count_known_cells,
            _count_known_cells_bg: count_known_cells_bg,
            count_band_cells: count_band_cells,
            _count_band_cells_bg: count_band_cells_bg,
            create_init_band: create_init_band,
            _create_init_band_gb: create_init_band_gb,
            solve_quadratic: solve_quadratic,
            solve_quadratic2: solve_quadratic2,
            prefix1: prefix1,
            prefix2: prefix2,
            prefix_gather: prefix_gather,
            prefix_sum_aux: prefix_sum_aux,
            reduce: reduce,
            find_neighbors: find_neighbors,
            fmm_alg_visualizer: fmm_alg_visualizer,
            iteration_counter: iteration_counter,
        }
    }

    /// Create the initial interface using point cloud data.
    /// TODO: point sampling method.
    #[allow(dead_code)]
    pub fn initialize_interface_pc(&self, encoder: &mut wgpu::CommandEncoder, pc: &PointCloud) {

        assert!(!self.pc_to_interface_bind_groups.is_none(), "Consider calling add_point_cloud_data before this function."); 

         self.pc_to_interface_compute_object.dispatch(
             &self.pc_to_interface_bind_groups.as_ref().unwrap(),
             encoder,
             1, 1, 1, Some("Point data to interface dispatch")
         );

         // println!("pc.get_point_count() == {}", pc.get_point_count());
         // println!("udiv_up_safe32(pc.get_point_count(), 1024) == {}", udiv_up_safe32(pc.get_point_count(), 1024));
         self.fmm_value_fixer.dispatch(encoder, [udiv_up_safe32(pc.get_point_count(), 1024), 1, 1]);
    }

    // pub fn create_compute_pass_fmm<'a>(&'a self, encoder: &'a mut wgpu::CommandEncoder) -> wgpu::ComputePass<'a> {

    //     let mut pass = encoder.begin_compute_pass(
    //         &wgpu::ComputePassDescriptor { label: Some("Fmm compute pass.")}
    //     );
    //     pass.set_pipeline(&self.compute_object.pipeline);
    //     for (e, bgs) in self.compute_object_bind_groups.iter().enumerate() {
    //         pass.set_bind_group(e as u32, &bgs, &[]);
    //     }
    //     pass
    // }

    pub fn fmm_iteration(&mut self, encoder: &mut wgpu::CommandEncoder, gpu_timer: &mut Option<GpuTimer>) {

        let number_of_dispatches = udiv_up_safe32(self.calculate_cell_count(), 1024);
        let number_of_dispatches_64 = udiv_up_safe32(self.calculate_cell_count(), 64);
        let number_of_dispatches_128 = udiv_up_safe32(self.calculate_cell_count(), 128);
        let number_of_dispatches_256 = udiv_up_safe32(self.calculate_cell_count(), 256);
        let number_of_dispatches_2048 = udiv_up_safe32(self.calculate_cell_count(), 2048);
        println!("number_of_dispatches == {}", number_of_dispatches);
        println!("number_of_dispatches_64 == {}", number_of_dispatches_64);
        println!("number_of_dispatches_128 == {}", number_of_dispatches_128);
        println!("self.calculate_block_count() == {}", self.calculate_block_count());
        let number_of_dispatches_prefix = udiv_up_safe32((self.global_dimension[0] * self.global_dimension[1] * self.global_dimension[2]) as u32, 1024 * 2);
        // let number_of_dispatches_prefix = udiv_up_safe32(self.calculate_cell_count(), 1024 * 2);
        // pass.set_pipeline(&self.count_cells.pipeline);

        let mut pass = encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor { label: Some("Fmm compute pass.")}
        );
        // pass.set_pipeline(&self.compute_object.pipeline);
        for (e, bgs) in self.compute_object_bind_groups.iter().enumerate() {
            pass.set_bind_group(e as u32, &bgs, &[]);
        }

        if !gpu_timer.is_none() {
            gpu_timer.as_mut().unwrap().start_pass(&mut pass);
        }

        if self.first_time {

            // Collect all known cells.
            pass.set_pipeline(&self.count_known_cells.pipeline);
            //pass.set_push_constants(0, bytemuck::cast_slice(&[3]));
            pass.dispatch_workgroups(number_of_dispatches_128, 1, 1); // TODO: create dispatch_indirect
            //timer gpu_timer.end_pass(&mut pass);

            // Create initial band.
            //timer gpu_timer.start_pass(&mut pass);
            pass.set_pipeline(&self.create_init_band.pipeline);
            pass.dispatch_workgroups(number_of_dispatches_128, 1, 1); // TODO: dispatch_indirect
            //timer gpu_timer.end_pass(&mut pass);

            // Collect all band cells.
            //timer gpu_timer.start_pass(&mut pass);
            pass.set_pipeline(&self.count_band_cells.pipeline);
            //pass.set_push_constants(0, bytemuck::cast_slice(&[2])); // TODO: create dispatch_indirect
            pass.dispatch_workgroups(number_of_dispatches_128, 1, 1);
            //timer gpu_timer.end_pass(&mut pass);

            // Solve quadratic on band cells.
            //timer gpu_timer.start_pass(&mut pass);
            pass.set_pipeline(&self.solve_quadratic.pipeline);
            //pass.set_push_constants(0, bytemuck::cast_slice(&[3])); // TODO: no push_constants.
            pass.dispatch_workgroups(number_of_dispatches, 1, 1); // TODO: dispatch_indirect!
            //timer gpu_timer.end_pass(&mut pass);

            // Scan active blocks part 1.
            //timer gpu_timer.start_pass(&mut pass);
            pass.set_pipeline(&self.prefix1.pipeline);
            pass.dispatch_workgroups(number_of_dispatches_prefix, 1, 1);
            //timer gpu_timer.end_pass(&mut pass);

            // Scan active blocks part 2.
            //timer gpu_timer.start_pass(&mut pass);
            pass.set_pipeline(&self.prefix2.pipeline);
            pass.dispatch_workgroups(1, 1, 1);

            pass.set_pipeline(&self.prefix_sum_aux.pipeline);
            pass.dispatch_workgroups(number_of_dispatches_256, 1, 1);

            //timer gpu_timer.end_pass(&mut pass);
            pass.set_pipeline(&self.prefix_gather.pipeline);
            pass.dispatch_workgroups(number_of_dispatches_256, 1, 1);

            // Recude and add new known point.
            //timer gpu_timer.start_pass(&mut pass);
            pass.set_pipeline(&self.reduce.pipeline);
            pass.dispatch_workgroups(number_of_dispatches_64 / 4, 4, 1);
            //timer gpu_timer.end_pass(&mut pass);
        }

        //for _ in 0..416 {
        //for _ in 0..417 {
        //for _ in 0..670 {
        for _ in 0..818 {
        //for _ in 0..1 {
        //for _ in 0..1300 {
        //for _ in 0..100 {
        // else {
                // let mut pass = encoder.begin_compute_pass(
                //     &wgpu::ComputePassDescriptor { label: Some("Fmm compute pass.")}
                // );
                // // pass.set_pipeline(&self.compute_object.pipeline);
                // for (e, bgs) in self.compute_object_bind_groups.iter().enumerate() {
                //     pass.set_bind_group(e as u32, &bgs, &[]);
                // }
                 pass.set_pipeline(&self.find_neighbors.pipeline);
                 pass.dispatch_workgroups(number_of_dispatches_128, 1, 1);
                 pass.set_pipeline(&self.solve_quadratic2.pipeline);
                 pass.set_push_constants(0, bytemuck::cast_slice(&[4])); // No push_constants!
                 pass.dispatch_workgroups(number_of_dispatches, 1, 1); // TODO: dispatch_indirect!
                 pass.set_pipeline(&self.prefix1.pipeline);
                 pass.dispatch_workgroups(number_of_dispatches_prefix, 1, 1);
                 pass.set_pipeline(&self.prefix2.pipeline);
                 pass.dispatch_workgroups(1, 1, 1);
                 pass.set_pipeline(&self.prefix_sum_aux.pipeline);
                 pass.dispatch_workgroups(number_of_dispatches_256, 1, 1);
                 pass.set_pipeline(&self.prefix_gather.pipeline);
                 pass.dispatch_workgroups(number_of_dispatches_256, 1, 1);
                 pass.set_pipeline(&self.reduce.pipeline);
                 pass.dispatch_workgroups(number_of_dispatches_64 / 4, 4, 1);
                 //pass.dispatch_workgroups(self.calculate_block_count() / 2, 2, 1);

             //++ if self.fmm_state == FmmState::FindNeighbors {
             //++     // Collect all band cells.
             //++     //timer gpu_timer.start_pass(&mut pass);
             //++     pass.set_pipeline(&self.find_neighbors.pipeline);
             //++     pass.dispatch_workgroups(number_of_dispatches * 8, 1, 1);
             //++     //timer gpu_timer.end_pass(&mut pass);
             //++ }

             //++ else if self.fmm_state == FmmState::SolveQuadratic {

             //++     //++ // Solve quadratic on band cells.
             //++     //++ //timer gpu_timer.start_pass(&mut pass);
             //++     pass.set_pipeline(&self.solve_quadratic.pipeline);
             //++     pass.set_push_constants(0, bytemuck::cast_slice(&[4]));
             //++     pass.dispatch_workgroups(number_of_dispatches, 1, 1); // TODO: dispatch_indirect!
             //++     //++ //timer gpu_timer.end_pass(&mut pass);
             //++ }

             //++ else if self.fmm_state == FmmState::FilterActiveBlocks {
             //++     //++ // Scan active blocks part 1.
             //++     //++ //timer gpu_timer.start_pass(&mut pass);
             //++     pass.set_pipeline(&self.prefix1.pipeline);
             //++     pass.dispatch_workgroups(number_of_dispatches_prefix, 1, 1);
             //++     //++ //timer gpu_timer.end_pass(&mut pass);
             //++       
             //++     //++ // Scan active blocks part 2.
             //++     //++ //timer gpu_timer.start_pass(&mut pass);
             //++     pass.set_pipeline(&self.prefix2.pipeline);
             //++     pass.dispatch_workgroups(1, 1, 1);
             //++     //++ //timer gpu_timer.end_pass(&mut pass);
             //++ }

             //++ else if self.fmm_state == FmmState::Reduce {

             //++     //++ // Recude and add new known point.
             //++     //++ //timer gpu_timer.start_pass(&mut pass);
             //++     pass.set_pipeline(&self.reduce.pipeline);
             //++     pass.dispatch_workgroups(self.calculate_block_count(), 1, 1);
             //++     //++ //timer gpu_timer.end_pass(&mut pass);
             //++     self.iteration_counter = self.iteration_counter + 1;
             //++     println!("iteration_counter = {}", self.iteration_counter);
             //++ }
        }


        // if self.fmm_state == FmmState::FilterActiveBlocks {
        //     pass.set_pipeline(&self.fmm_alg_visualizer.pipeline);
        //     pass.dispatch_workgroups(1, 1, 1);
        // }

        // }

        if !gpu_timer.is_none() {
            gpu_timer.as_mut().unwrap().end_pass(&mut pass);
        }
        drop(pass);
        self.first_time = false;
        // println!("{:?}", self.fmm_state);
        // self.next_fmm_state();
          
        if !gpu_timer.is_none() {
            gpu_timer.as_mut().unwrap().resolve_timestamps(encoder);
        }
    }

    pub fn next_fmm_state(&mut self) {
        match self.fmm_state {
            FmmState::FindNeighbors => { self.fmm_state = FmmState::SolveQuadratic; }
            FmmState::SolveQuadratic => { self.fmm_state = FmmState::FilterActiveBlocks; }
            FmmState::FilterActiveBlocks => { self.fmm_state = FmmState::Reduce; }
            FmmState::Reduce => { self.fmm_state = FmmState::FindNeighbors; }
        }
    }

    pub fn get_fmm_state(&self) -> FmmState {
        self.fmm_state
    }

    pub fn print_fmm_histogram(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        print!("{:?} ", self.fmm_histogram.get_values(&device, queue));
    }

    pub fn add_point_cloud_data(&mut self,
                                device: &wgpu::Device,
                                pc_data: &wgpu::Buffer,
                                point_cloud_params_buffer: &PointCloudParamsBuffer) {

        let pc_to_interface_bind_groups = create_bind_groups(
                &device,
                &self.pc_to_interface_compute_object.bind_group_layout_entries,
                &self.pc_to_interface_compute_object.bind_group_layouts,
                &vec![
                    vec![
                    &self.fmm_params_buffer.get_buffer().as_entire_binding(),
                    &point_cloud_params_buffer.get_buffer().as_entire_binding(),
                    &self.fmm_data.as_entire_binding(),
                    &pc_data.as_entire_binding(),
                    &self.prefix_temp_array.as_entire_binding(),
                    //&gpu_debugger.get_element_counter_buffer().as_entire_binding(),
                    // &gpu_debugger.get_output_chars_buffer().as_entire_binding(),
                    // &gpu_debugger.get_output_arrows_buffer().as_entire_binding(),
                    // &gpu_debugger.get_output_aabbs_buffer().as_entire_binding(),
                    // &gpu_debugger.get_output_aabb_wires_buffer().as_entire_binding(),
                    ],
                ]
        );
        self.pc_to_interface_bind_groups = Some(pc_to_interface_bind_groups);

    }

    fn calculate_cell_count(&self) -> u32 {
        self.global_dimension[0] * self.global_dimension[1] * self.global_dimension[2] * self.local_dimension[0]  * self.local_dimension[1]  * self.local_dimension[2] 
    }

    fn calculate_block_count(&self) -> u32 {
        self.global_dimension[0] * self.global_dimension[1] * self.global_dimension[2]
    }

    pub fn get_fmm_data_buffer(&self) -> &wgpu::Buffer {
        &self.fmm_data
    }

    pub fn get_fmm_temp_buffer(&self) -> &wgpu::Buffer {
        &self.temporary_fmm_data
    }

    pub fn get_fmm_prefix_temp_buffer(&self) -> &wgpu::Buffer {
        &self.prefix_temp_array
    }

    pub fn get_fmm_params_buffer(&self) -> &wgpu::Buffer {
        &self.fmm_params_buffer.get_buffer()
    }

    fn create_gather_known_cells_co(device: &wgpu::Device,
                                    prefix_params: &wgpu::Buffer,
                                    fmm_params: &wgpu::Buffer,
                                    fmm_blocks: &wgpu::Buffer,
                                    temp_prefix_sum: &wgpu::Buffer,
                                    filtered_blocks: &wgpu::Buffer,
                                    fmm_data: &wgpu::Buffer,
                                    fmm_counter: &wgpu::Buffer) -> (ComputeObject, Vec<wgpu::BindGroup>) {
        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("collect_known_points.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/collect_known_points.wgsl"))),

                    }),
                    Some("Collect known points ComputeObject"),
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

                            // @group(0) @binding(5) var<storage,read_write>  fmm_data: array<TempData>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    Some(vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::COMPUTE,
                        range: 0..4,
                    }]),
        );
        let bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &prefix_params.as_entire_binding(),
                        &fmm_params.as_entire_binding(),
                        &fmm_blocks.as_entire_binding(),
                        &temp_prefix_sum.as_entire_binding(),
                        &filtered_blocks.as_entire_binding(),
                        &fmm_data.as_entire_binding(),
                        &fmm_counter.as_entire_binding(),
                    ],
                ]
        );
        (compute_object, bind_groups)
    }

    fn create_gather_band_cells_co(device: &wgpu::Device,
                                   prefix_params: &wgpu::Buffer,
                                   fmm_params: &wgpu::Buffer,
                                   fmm_blocks: &wgpu::Buffer,
                                   temp_prefix_sum: &wgpu::Buffer,
                                   filtered_blocks: &wgpu::Buffer,
                                   fmm_data: &wgpu::Buffer,
                                   fmm_counter: &wgpu::Buffer) -> (ComputeObject, Vec<wgpu::BindGroup>) {
        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("collect_band_points.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/collect_band_points.wgsl"))),

                    }),
                    Some("Collect band points ComputeObject"),
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

                            // @group(0) @binding(5) var<storage,read_write>  fmm_data: array<TempData>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    Some(vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::COMPUTE,
                        range: 0..4,
                    }]),
        );
        let bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &prefix_params.as_entire_binding(),
                        &fmm_params.as_entire_binding(),
                        &fmm_blocks.as_entire_binding(),
                        &temp_prefix_sum.as_entire_binding(),
                        &filtered_blocks.as_entire_binding(),
                        &fmm_data.as_entire_binding(),
                        &fmm_counter.as_entire_binding(),
                    ],
                ]
        );
        (compute_object, bind_groups)
    }

    fn create_initial_band_co(device: &wgpu::Device,
                                    prefix_params: &wgpu::Buffer,
                                    fmm_params: &wgpu::Buffer,
                                    fmm_blocks: &wgpu::Buffer,
                                    temp_prefix_sum: &wgpu::Buffer,
                                    filtered_blocks: &wgpu::Buffer,
                                    fmm_data: &wgpu::Buffer,
                                    fmm_counter: &wgpu::Buffer) -> (ComputeObject, Vec<wgpu::BindGroup>) {
        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("create_initial_band_points.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/create_initial_band_points.wgsl"))),

                    }),
                    Some("Create initial band points ComputeObject"),
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

                            // @group(0) @binding(5) var<storage,read_write>  fmm_data: array<TempData>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None,
                    // Some(vec![wgpu::PushConstantRange {
                    //     stages: wgpu::ShaderStages::COMPUTE,
                    //     range: 0..4,
                    // }]),
        );
        let bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &prefix_params.as_entire_binding(),
                        &fmm_params.as_entire_binding(),
                        &fmm_blocks.as_entire_binding(),
                        &temp_prefix_sum.as_entire_binding(),
                        &filtered_blocks.as_entire_binding(),
                        &fmm_data.as_entire_binding(),
                        &fmm_counter.as_entire_binding(),
                    ],
                ]
        );
        (compute_object, bind_groups)
    }

    fn create_solve_quadratic_co(device: &wgpu::Device,
                                    prefix_params: &wgpu::Buffer,
                                    fmm_params: &wgpu::Buffer,
                                    fmm_blocks: &wgpu::Buffer,
                                    temp_prefix_sum: &wgpu::Buffer,
                                    filtered_blocks: &wgpu::Buffer,
                                    fmm_data: &wgpu::Buffer,
                                    fmm_counter: &wgpu::Buffer) -> (ComputeObject, Vec<wgpu::BindGroup>) {
        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("solve_quadratic.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/solve_quadratic.wgsl"))),

                    }),
                    Some("Solve quadratic ComputeObject"),
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

                            // @group(0) @binding(5) var<storage,read_write>  fmm_data: array<TempData>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    // None,
                    Some(vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::COMPUTE,
                        range: 0..4,
                    }]),
        );
        let bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &prefix_params.as_entire_binding(),
                        &fmm_params.as_entire_binding(),
                        &fmm_blocks.as_entire_binding(),
                        &temp_prefix_sum.as_entire_binding(),
                        &filtered_blocks.as_entire_binding(),
                        &fmm_data.as_entire_binding(),
                        &fmm_counter.as_entire_binding(),
                    ],
                ]
        );
        (compute_object, bind_groups)
    }

    fn create_solve_quadratic2_co(device: &wgpu::Device,
                                  prefix_params: &wgpu::Buffer,
                                  fmm_params: &wgpu::Buffer,
                                  fmm_blocks: &wgpu::Buffer,
                                  temp_prefix_sum: &wgpu::Buffer,
                                  filtered_blocks: &wgpu::Buffer,
                                  fmm_data: &wgpu::Buffer,
                                  fmm_counter: &wgpu::Buffer) -> (ComputeObject, Vec<wgpu::BindGroup>) {
        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("solve_quadratic2.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/solve_quadratic2.wgsl"))),

                    }),
                    Some("Solve quadratic2 ComputeObject"),
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

                            // @group(0) @binding(5) var<storage,read_write>  fmm_data: array<TempData>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    // None,
                    Some(vec![wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::COMPUTE,
                        range: 0..4,
                    }]),
        );
        let bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &prefix_params.as_entire_binding(),
                        &fmm_params.as_entire_binding(),
                        &fmm_blocks.as_entire_binding(),
                        &temp_prefix_sum.as_entire_binding(),
                        &filtered_blocks.as_entire_binding(),
                        &fmm_data.as_entire_binding(),
                        &fmm_counter.as_entire_binding(),
                    ],
                ]
        );
        (compute_object, bind_groups)
    }

    fn create_prefix_sum_1(device: &wgpu::Device,
                           prefix_params: &wgpu::Buffer,
                           fmm_params: &wgpu::Buffer,
                           fmm_blocks: &wgpu::Buffer,
                           temp_prefix_sum: &wgpu::Buffer,
                           filtered_blocks: &wgpu::Buffer,
                           fmm_data: &wgpu::Buffer,
                           fmm_counter: &wgpu::Buffer) -> (ComputeObject, Vec<wgpu::BindGroup>) {

        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("filter_active_blocks.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/filter_active_blocks.wgsl"))),

                    }),
                    Some("Filter active blocks part 1 ComputeObject"),
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

                            // @group(0) @binding(5) var<storage,read_write>  fmm_data: array<TempData>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None,
        );
        let bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &prefix_params.as_entire_binding(),
                        &fmm_params.as_entire_binding(),
                        &fmm_blocks.as_entire_binding(),
                        &temp_prefix_sum.as_entire_binding(),
                        &filtered_blocks.as_entire_binding(),
                        &fmm_data.as_entire_binding(),
                        &fmm_counter.as_entire_binding(),
                    ],
                ]
        );
        (compute_object, bind_groups)
    }

    fn create_prefix_sum_2(device: &wgpu::Device,
                           prefix_params: &wgpu::Buffer,
                           fmm_params: &wgpu::Buffer,
                           fmm_blocks: &wgpu::Buffer,
                           temp_prefix_sum: &wgpu::Buffer,
                           filtered_blocks: &wgpu::Buffer,
                           fmm_data: &wgpu::Buffer,
                           fmm_counter: &wgpu::Buffer) -> (ComputeObject, Vec<wgpu::BindGroup>) {

        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("filter_active_blocks2.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/filter_active_blocks2.wgsl"))),

                    }),
                    Some("Filter active blocks part 2 ComputeObject"),
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

                            // @group(0) @binding(5) var<storage,read_write>  fmm_data: array<TempData>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None,
        );
        let bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &prefix_params.as_entire_binding(),
                        &fmm_params.as_entire_binding(),
                        &fmm_blocks.as_entire_binding(),
                        &temp_prefix_sum.as_entire_binding(),
                        &filtered_blocks.as_entire_binding(),
                        &fmm_data.as_entire_binding(),
                        &fmm_counter.as_entire_binding(),
                    ],
                ]
        );
        (compute_object, bind_groups)
    }

    fn create_prefix_gather(device: &wgpu::Device,
                           prefix_params: &wgpu::Buffer,
                           fmm_params: &wgpu::Buffer,
                           fmm_blocks: &wgpu::Buffer,
                           temp_prefix_sum: &wgpu::Buffer,
                           filtered_blocks: &wgpu::Buffer,
                           fmm_data: &wgpu::Buffer,
                           fmm_counter: &wgpu::Buffer) -> (ComputeObject, Vec<wgpu::BindGroup>) {

        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("filter_active_blocks_gather.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/filter_active_blocks_gather.wgsl"))),

                    }),
                    Some("Filter active blocks gather ComputeObject"),
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

                            // @group(0) @binding(5) var<storage,read_write>  fmm_data: array<TempData>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None,
        );
        let bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &prefix_params.as_entire_binding(),
                        &fmm_params.as_entire_binding(),
                        &fmm_blocks.as_entire_binding(),
                        &temp_prefix_sum.as_entire_binding(),
                        &filtered_blocks.as_entire_binding(),
                        &fmm_data.as_entire_binding(),
                        &fmm_counter.as_entire_binding(),
                    ],
                ]
        );
        (compute_object, bind_groups)
    }

    fn create_prefix_sum_aux(device: &wgpu::Device,
                           prefix_params: &wgpu::Buffer,
                           fmm_params: &wgpu::Buffer,
                           fmm_blocks: &wgpu::Buffer,
                           temp_prefix_sum: &wgpu::Buffer,
                           filtered_blocks: &wgpu::Buffer,
                           fmm_data: &wgpu::Buffer,
                           fmm_counter: &wgpu::Buffer) -> (ComputeObject, Vec<wgpu::BindGroup>) {

        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("filter_active_blocks_sum_aux.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/filter_active_blocks_sum_aux.wgsl"))),

                    }),
                    Some("Filter active blocks sum aux ComputeObject"),
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

                            // @group(0) @binding(5) var<storage,read_write>  fmm_data: array<TempData>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None,
        );
        let bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &prefix_params.as_entire_binding(),
                        &fmm_params.as_entire_binding(),
                        &fmm_blocks.as_entire_binding(),
                        &temp_prefix_sum.as_entire_binding(),
                        &filtered_blocks.as_entire_binding(),
                        &fmm_data.as_entire_binding(),
                        &fmm_counter.as_entire_binding(),
                    ],
                ]
        );
        (compute_object, bind_groups)
    }

    fn create_reduce(device: &wgpu::Device,
                           prefix_params: &wgpu::Buffer,
                           fmm_params: &wgpu::Buffer,
                           fmm_blocks: &wgpu::Buffer,
                           temp_prefix_sum: &wgpu::Buffer,
                           filtered_blocks: &wgpu::Buffer,
                           fmm_data: &wgpu::Buffer,
                           fmm_counter: &wgpu::Buffer) -> (ComputeObject, Vec<wgpu::BindGroup>) {

        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("reduce.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/reduce.wgsl"))),

                    }),
                    Some("Reduce ComputeObject"),
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

                            // @group(0) @binding(5) var<storage,read_write>  fmm_data: array<TempData>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None,
        );
        let bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &prefix_params.as_entire_binding(),
                        &fmm_params.as_entire_binding(),
                        &fmm_blocks.as_entire_binding(),
                        &temp_prefix_sum.as_entire_binding(),
                        &filtered_blocks.as_entire_binding(),
                        &fmm_data.as_entire_binding(),
                        &fmm_counter.as_entire_binding(),
                    ],
                ]
        );
        (compute_object, bind_groups)
    }

    fn create_find_neighbors(device: &wgpu::Device,
                             prefix_params: &wgpu::Buffer,
                             fmm_params: &wgpu::Buffer,
                             fmm_blocks: &wgpu::Buffer,
                             temp_prefix_sum: &wgpu::Buffer,
                             filtered_blocks: &wgpu::Buffer,
                             fmm_data: &wgpu::Buffer,
                             fmm_counter: &wgpu::Buffer) -> (ComputeObject, Vec<wgpu::BindGroup>) {

        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("find_neighbors.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/find_neighbors.wgsl"))),

                    }),
                    Some("Find neighbors ComputeObject"),
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

                            // @group(0) @binding(5) var<storage,read_write>  fmm_data: array<TempData>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None,
        );
        let bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &prefix_params.as_entire_binding(),
                        &fmm_params.as_entire_binding(),
                        &fmm_blocks.as_entire_binding(),
                        &temp_prefix_sum.as_entire_binding(),
                        &filtered_blocks.as_entire_binding(),
                        &fmm_data.as_entire_binding(),
                        &fmm_counter.as_entire_binding(),
                    ],
                ]
        );
        (compute_object, bind_groups)
    }

    fn create_fmm_alg_visualizer(device: &wgpu::Device,
                                 prefix_params: &wgpu::Buffer,
                                 fmm_params: &wgpu::Buffer,
                                 fmm_blocks: &wgpu::Buffer,
                                 temp_prefix_sum: &wgpu::Buffer,
                                 filtered_blocks: &wgpu::Buffer,
                                 fmm_data: &wgpu::Buffer,
                                 fmm_counter: &wgpu::Buffer) -> ComputeObject {

        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("fmm_alg_visualizer.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/fmm_alg_visualizer.wgsl"))),

                    }),
                    Some("Fmm alg visualizer ComputeObject"),
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

                            // @group(0) @binding(5) var<storage,read_write>  fmm_data: array<TempData>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> fmm_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
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
                    None,
        );
        compute_object
    }
}
