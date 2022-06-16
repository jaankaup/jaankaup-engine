use std::mem::size_of;
use crate::fmm_things::{FmmPrefixParams, FmmCellPc, FmmBlock};
use crate::common_functions::udiv_up_safe32;
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
    point_to_interface_bind_groups: Option<Vec<wgpu::BindGroup>>,

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

    count_cells: ComputeObject,
    _count_cells_bg: Vec<wgpu::BindGroup>,
    create_init_band: ComputeObject,
    _create_init_band_gb: Vec<wgpu::BindGroup>,
    solve_quadratic: ComputeObject,
    prefix1: ComputeObject,
    prefix2: ComputeObject,
    reduce: ComputeObject,
    find_neighbors: ComputeObject,
    fmm_alg_visualizer: ComputeObject, 
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

        // Buffer hash_map. DO we need this?
        let buffers: HashMap<String, wgpu::Buffer> = HashMap::new();

        // Fmm histogram.
        let fmm_histogram = Histogram::init(&device, &vec![0; 5]);

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

        let mut fmm_data = vec![FmmCellPc { tag: 0, value: 100000.0, color: 0, } ; number_of_fmm_cells + 1];

        // The outside value.
        fmm_data[number_of_fmm_cells] = FmmCellPc { tag: 4, value: 100000.0, color: 0, };
                                                
        // Fast marching method cell data.
        let fmm_data = buffer_from_data::<FmmCellPc>(
                &device,
                &fmm_data,
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                None
        );

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

        // Create temp data buffer.
        let temporary_fmm_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fmm temporary data buffer"),
            size: 131072 * size_of::<FmmBlock>() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Prefix temp array.
        let prefix_temp_array = buffer_from_data::<u32>(
                &device,
                &vec![0 ; number_of_fmm_cells + 2048 as usize],
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                None
        );

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

        // Update band counts shader.

        let update_band_counts_compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
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

        let (count_cells, count_cells_bg) = Self::create_gather_known_cells_co(
                                        &device,
                                        &fmm_prefix_params,
                                        &fmm_params_buffer.get_buffer(),
                                        &fmm_blocks,
                                        &prefix_temp_array,
                                        &temporary_fmm_data,
                                        &fmm_data,
                                        &fmm_histogram.get_histogram_buffer(),
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
            point_to_interface_bind_groups: None,
            fmm_value_fixer: fmm_value_fixer,
            fmm_prefix_params: fmm_prefix_params,
            prefix_temp_array: prefix_temp_array,
            update_band_counts_compute_object: update_band_counts_compute_object,
            update_band_counts_compute_object_bind_groups: update_band_counts_compute_object_bind_groups,
            fmm_histogram: fmm_histogram,
            count_cells: count_cells,
            _count_cells_bg: count_cells_bg,
            create_init_band: create_init_band,
            _create_init_band_gb: create_init_band_gb,
            solve_quadratic: solve_quadratic,
            prefix1: prefix1,
            prefix2: prefix2,
            reduce: reduce,
            find_neighbors: find_neighbors,
            fmm_alg_visualizer: fmm_alg_visualizer,
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

    pub fn fmm_iteration(&self, encoder: &mut wgpu::CommandEncoder, gpu_timer: &mut GpuTimer) {

        let number_of_dispatches = udiv_up_safe32(self.calculate_cell_count(), 1024);
        let number_of_dispatches_prefix = udiv_up_safe32((self.global_dimension[0] * self.global_dimension[1] * self.global_dimension[2]) as u32, 1024 * 2);
        // let number_of_dispatches_prefix = udiv_up_safe32(self.calculate_cell_count(), 1024 * 2);
        // pass.set_pipeline(&self.count_cells.pipeline);

        let mut pass = encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor { label: Some("Fmm compute pass.")}
        );
        // pass.set_pipeline(&self.compute_object.pipeline);
        pass.set_pipeline(&self.count_cells.pipeline);
        for (e, bgs) in self.compute_object_bind_groups.iter().enumerate() {
            pass.set_bind_group(e as u32, &bgs, &[]);
        }

        // Collect all known cells.
        //timer gpu_timer.start_pass(&mut pass);
        pass.set_push_constants(0, bytemuck::cast_slice(&[3]));
        pass.dispatch_workgroups(number_of_dispatches * 8, 1, 1); // TODO: create dispatch_indirect
        //timer gpu_timer.end_pass(&mut pass);

        // Create initial band.
        //timer gpu_timer.start_pass(&mut pass);
        pass.set_pipeline(&self.create_init_band.pipeline);
        pass.dispatch_workgroups(number_of_dispatches * 8, 1, 1); // TODO: dispatch_indirect
        //timer gpu_timer.end_pass(&mut pass);

        // Collect all band cells.
        //timer gpu_timer.start_pass(&mut pass);
        pass.set_pipeline(&self.count_cells.pipeline);
        pass.set_push_constants(0, bytemuck::cast_slice(&[2])); // TODO: create dispatch_indirect
        pass.dispatch_workgroups(number_of_dispatches * 8, 1, 1);
        //timer gpu_timer.end_pass(&mut pass);

        // Solve quadratic on band cells.
        //timer gpu_timer.start_pass(&mut pass);
        pass.set_pipeline(&self.solve_quadratic.pipeline);
        pass.set_push_constants(0, bytemuck::cast_slice(&[3]));
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
        //timer gpu_timer.end_pass(&mut pass);
        //
        // pass.set_pipeline(&self.fmm_alg_visualizer.pipeline);
        // pass.dispatch_workgroups(1, 1, 1);

        // Recude and add new known point.
        //timer gpu_timer.start_pass(&mut pass);
        pass.set_pipeline(&self.reduce.pipeline);
        pass.dispatch_workgroups(self.calculate_block_count(), 1, 1);
        //timer gpu_timer.end_pass(&mut pass);

        for i in 0..10 {

            // Collect all band cells.
            //timer gpu_timer.start_pass(&mut pass);
            pass.set_pipeline(&self.find_neighbors.pipeline);
            pass.dispatch_workgroups(number_of_dispatches * 8, 1, 1);
            //timer gpu_timer.end_pass(&mut pass);

            //++ // Solve quadratic on band cells.
            //++ //timer gpu_timer.start_pass(&mut pass);
            pass.set_pipeline(&self.solve_quadratic.pipeline);
            pass.set_push_constants(0, bytemuck::cast_slice(&[4]));
            pass.dispatch_workgroups(number_of_dispatches, 1, 1); // TODO: dispatch_indirect!
            //++ //timer gpu_timer.end_pass(&mut pass);

            //++ // Scan active blocks part 1.
            //++ //timer gpu_timer.start_pass(&mut pass);
            pass.set_pipeline(&self.prefix1.pipeline);
            pass.dispatch_workgroups(number_of_dispatches_prefix, 1, 1);
            //++ //timer gpu_timer.end_pass(&mut pass);

            //++ // Scan active blocks part 2.
            //++ //timer gpu_timer.start_pass(&mut pass);
            pass.set_pipeline(&self.prefix2.pipeline);
            pass.dispatch_workgroups(1, 1, 1);
            //++ //timer gpu_timer.end_pass(&mut pass);

            //++ // Recude and add new known point.
            //++ //timer gpu_timer.start_pass(&mut pass);
            pass.set_pipeline(&self.reduce.pipeline);
            pass.dispatch_workgroups(self.calculate_block_count(), 1, 1);
            //++ //timer gpu_timer.end_pass(&mut pass);

        }

        drop(pass);
        //timer gpu_timer.resolve_timestamps(encoder);

    }

    pub fn print_fmm_histogram(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        println!("{:?}", self.fmm_histogram.get_values(&device, queue));
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

    fn calculate_cell_count(&self) -> u32 {
        self.global_dimension[0] * self.global_dimension[1] * self.global_dimension[2] * self.local_dimension[0]  * self.local_dimension[1]  * self.local_dimension[2] 
    }

    fn calculate_block_count(&self) -> u32 {
        self.global_dimension[0] * self.global_dimension[1] * self.global_dimension[2]
    }


    /// Collect all known cells to temp_data array. The known cell count is saved to fmm_histogram[0].
    // pub fn switch_pipeline1<'a>(&'a self, pass: &'a mut wgpu::ComputePass<'a>) {
    pub fn collect_known_cells<'a>(&'a self, pass: &'a mut wgpu::ComputePass<'a>) {

        let number_of_dispatches = udiv_up_safe32(self.calculate_cell_count(), 1024);
        pass.set_pipeline(&self.count_cells.pipeline);
        pass.set_push_constants(0, bytemuck::cast_slice(&[2]));
        pass.dispatch_workgroups(number_of_dispatches, 1, 1);
        //pass.dispatch_workgroups(udiv_up_safe32(self.calculate_cell_count(), 1024), 1, 1);
    }

    /// Collect all band cells to temp_data array. The known cell count is saved to fmm_histogram[0].
    pub fn collect_band_cells(&self, pass: &mut wgpu::ComputePass) {

        pass.set_push_constants(0, bytemuck::cast_slice(&[4]));
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Fmm for all selected band cells.
    pub fn fmm(&self, pass: &mut wgpu::ComputePass) {
        pass.set_push_constants(0, bytemuck::cast_slice(&[5]));
        pass.dispatch_workgroups(udiv_up_safe32(self.calculate_cell_count(), 1024), 1, 1);
    }

    pub fn filter_active_blocks(&self, pass: &mut wgpu::ComputePass) {
                
        let number_of_dispatches = udiv_up_safe32((self.global_dimension[0] * self.global_dimension[1] * self.global_dimension[2]) as u32, 1024 * 2);

        pass.set_push_constants(0, bytemuck::cast_slice(&[0]));
        pass.dispatch_workgroups(number_of_dispatches, 1, 1);

        pass.set_push_constants(0, bytemuck::cast_slice(&[1]));
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Update all band point counts from global computational domain. 
    pub fn update_band_point_counts(&self, pass: &mut wgpu::ComputePass) {

        let number_of_dispatches = udiv_up_safe32(self.calculate_cell_count() , 64);

        pass.set_push_constants(0, bytemuck::cast_slice(&[6]));
        pass.dispatch_workgroups(number_of_dispatches, 1, 1);
    }

    /// Gather the indexes from knwon points to fmm_temp array. 
    pub fn gather_all_known_points(&self, pass: &mut wgpu::ComputePass) {
        pass.set_push_constants(0, bytemuck::cast_slice(&[2]));
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// 
    pub fn create_initial_band(&self, pass: &mut wgpu::ComputePass) {

        pass.set_push_constants(0, bytemuck::cast_slice(&[3]));
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub fn visualize_active_blocs(&self, pass: &mut wgpu::ComputePass) {

        pass.set_push_constants(0, bytemuck::cast_slice(&[15]));
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub fn get_pipeline1(&self) -> &wgpu::ComputePipeline {
        &self.pc_to_interface_compute_object.pipeline
    }

    pub fn get_pipeline2(&self) -> &wgpu::ComputePipeline {
        &self.compute_object.pipeline
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

    pub fn get_fmm_temp_buffer(&self) -> &wgpu::Buffer {
        &self.temporary_fmm_data
    }

    pub fn get_fmm_prefix_temp_buffer(&self) -> &wgpu::Buffer {
        &self.prefix_temp_array
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
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("collect_cells.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_shaders/collect_cells.wgsl"))),

                    }),
                    Some("Collect cells ComputeObject"),
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
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
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
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
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
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
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
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
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
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
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
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
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
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
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

