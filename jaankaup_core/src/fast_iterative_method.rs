use std::mem::size_of;
use crate::fmm_things::{FmmPrefixParams, FmmCellPc};
use crate::common_functions::{udiv_up_safe32, encode_rgba_u32};
use crate::fmm_things::FmmValueFixer;
use crate::fmm_things::PointCloudParamsBuffer;
use std::convert::TryInto;
use crate::buffer::buffer_from_data;
use std::borrow::Cow;
use std::collections::HashMap;
use crate::render_object::{ComputeObject, create_bind_groups};
use crate::common_functions::{create_uniform_bindgroup_layout, create_buffer_bindgroup_layout};
use crate::fmm_things::{FmmParamsBuffer, PointCloud};
use crate::gpu_debugger::GpuDebugger;
use crate::histogram::Histogram;
use crate::gpu_timer::GpuTimer;

/// Tag value for a unknow cell.
const OTHER: u32  = 0;

// /// Tag value for a remedy cell.
// const REMEDY: u32 = 1;
// 
// /// Tag value for active cell.
// const ACTIVE: u32 = 2;
// 
// /// Tag value for Source cell.
// const SOURCE: u32 = 3;

/// Tag value for outside cell.
const OUTSIDE: u32 = 4;

// #[allow(dead_code)]
// #[derive(PartialEq, Debug, Clone, Copy)]
// pub enum FmmState {
//     FindNeighbors,
//     SolveQuadratic,
//     FilterActiveBlocks,
//     Reduce,
// }

// #[repr(C)]
// #[derive(Debug, Clone, Copy, Pod, Zeroable)]
// struct FmmCellPointCloud {
//     tag: u32,
//     value: f32,
//     color: u32,
//     update: u32,
// }

// We use the same name convention as FMM.

/// Max number of arrows for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_ARROWS:     usize = 262144;

/// Max number of aabbs for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_AABBS:      usize = 262144;

/// Max number of box frames for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_AABB_WIRES: usize =  40960;

/// Max number of renderable char elements (f32, vec3, vec4, ...) for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_CHARS:      usize = 262144;

/// Max number of vvvvnnnn vertices reserved for gpu draw buffer.
#[allow(dead_code)]
const MAX_NUMBER_OF_VVVVNNNN: usize =  2000000;

/// Struct for parallel fast marching method. 
pub struct FastIterativeMethod {

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
    fim_data: wgpu::Buffer,

    // /// Fim active list buffer.
    // #[allow(dead_code)]
    // active_list: wgpu::Buffer,

    // /// Temporary data for prefix sum.
    // #[allow(dead_code)]
    // remedy_list: wgpu::Buffer,

    /// Point data to fmm interface compute object.
    #[allow(dead_code)]
    pc_to_interface_compute_object: ComputeObject,

    /// Point data to fmm interface bing groups.
    #[allow(dead_code)]
    pc_to_interface_bind_groups: Option<Vec<wgpu::BindGroup>>,

    sample_data: wgpu::Buffer,

    /// Preprocessor for fmm values.
    #[allow(dead_code)]
    fmm_value_fixer: FmmValueFixer,

    /// Prefix params.
    #[allow(dead_code)]
    fmm_prefix_params: wgpu::Buffer,

    /// Temporary data for prefix sum.
    #[allow(dead_code)]
    prefix_temp_array: wgpu::Buffer,

    /// A histogram for fmm purposes.
    #[allow(dead_code)]
    fim_histogram: Histogram,

    first_time: bool,
    // fmm_state: FmmState,

    initial_active_cells: ComputeObject,
    update_phase: ComputeObject,
    pre_remedy: ComputeObject,
    remedy_phase: ComputeObject,
}

impl FastIterativeMethod {

    /// Creates an instance of FastIterativeMethod.
    #[allow(dead_code)]
    pub fn init(device: &wgpu::Device,
                global_dimension: [u32; 3],
                local_dimension: [u32; 3],
                gpu_debugger: &Option<GpuDebugger>,
                // gpu_debugger: &Option<&GpuDebugger>,
                ) -> Self {

        // TODO: assertions for local and global dimension.

        // let iteration_counter = 0;
        // let fmm_state = FmmState::Reduce;

        // Buffer hash_map. DO we need this?
        let buffers: HashMap<String, wgpu::Buffer> = HashMap::new();

        // Fmm histogram.
        let fim_histogram = Histogram::init(&device, &vec![0; 5]);

        let cell_count = global_dimension[0] * global_dimension[1] * global_dimension[2] * local_dimension[0]  * local_dimension[1]  * local_dimension[2];

        print!("Creating sample_data buffer.");
        // Prefix temp array.
        let sample_data = buffer_from_data::<f32>(
                &device,
                &vec![0.0 ; (cell_count * 3).try_into().unwrap()],
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                None
        );
        println!("OK");

        // Create fim shader template. Only for binding resources.
        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("dummy_fim.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            if cfg!(not(target_arch = "wasm32")) {
                                Cow::Borrowed(include_str!("../../assets/shaders/fim_shaders/dummy_fim.wgsl"))
                            }
                            else {
                                Cow::Borrowed(include_str!("../../assets/shaders/fim_shaders/wasm/dummy_fim.wgsl"))
                            }
                        ),
                    }),
                    Some("FastIterativeMethod dummy ComputeObject"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> prefix_params: PrefixParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<uniform> fmm_params: FmmParams;
                            create_uniform_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(2) var<storage, read_write> active_list: array<FmmBlock>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),
                              
                            // @group(0) @binding(6) var<storage,read_write>  fim_data: array<TempData>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(7) var<storage,read_write> fmm_counter: array<atomic<u32>>;
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
                    None,
        );

        let fmm_params_buffer = FmmParamsBuffer::create(device, global_dimension, local_dimension);

        let number_of_active_list: usize = (global_dimension[0] * global_dimension[1] * global_dimension[2]).try_into().unwrap();
        let number_of_fmm_cells: usize = (number_of_active_list as u32 * local_dimension[0] * local_dimension[1] * local_dimension[2]).try_into().unwrap();

        let the_color = encode_rgba_u32(0, 0, 0, 255);
        let mut fim_data = vec![FmmCellPc { tag: OTHER, value: 100000.0, color: the_color, } ; number_of_fmm_cells + 1];

        // The outside value.
        fim_data[number_of_fmm_cells] = FmmCellPc { tag: OUTSIDE, value: 100000.0, color: the_color, };
                                                
        // Fast marching method cell data.
        print!("Creating fim_data buffer.  ");
        let fim_data = buffer_from_data::<FmmCellPc>(
                &device,
                &fim_data,
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                None
        );
        println!("OK");

        print!("Creating active list buffer.  ");
        let active_list = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fim active list buffer"),
            size: (number_of_fmm_cells * size_of::<FmmCellPc>()) as u64 * 2,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        println!("OK");

        print!("Creating prefix_temp_array.  ");
        // TODO: remove. Only fmm_value_fixer uses this binding. Remove this binding from
        // fmm_value_fixer.
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
        // println!("fmm_prefix_params = {:?}", FmmPrefixParams { data_start_index: 0,
        //                                                        data_end_index: (number_of_fmm_cells - 1) as u32,
        //                                                        exclusive_parts_start_index: number_of_fmm_cells as u32,
        //                                                        exclusive_parts_end_index: number_of_fmm_cells as u32 + 2048,});

        #[cfg(feature = "gpu_debug")]
        let compute_object_bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &fmm_prefix_params.as_entire_binding(),
                        &fmm_params_buffer.get_buffer().as_entire_binding(),
                        &active_list.as_entire_binding(),
                        // &prefix_temp_array.as_entire_binding(),
                        &fim_data.as_entire_binding(),
                        &fim_histogram.get_histogram_buffer().as_entire_binding(),
                    ],
                    vec![
                        &gpu_debugger.as_ref().unwrap().get_element_counter_buffer().as_entire_binding(),
                        &gpu_debugger.as_ref().unwrap().get_output_chars_buffer().as_entire_binding(),
                        &gpu_debugger.as_ref().unwrap().get_output_arrows_buffer().as_entire_binding(),
                        &gpu_debugger.as_ref().unwrap().get_output_aabbs_buffer().as_entire_binding(),
                        &gpu_debugger.as_ref().unwrap().get_output_aabb_wires_buffer().as_entire_binding(),
                    ],
                ]
        );

        // TODO: create a shader without gpu_debugger bindings.
        #[cfg(not(feature = "gpu_debug"))]
        let compute_object_bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &fmm_prefix_params.as_entire_binding(),
                        &fmm_params_buffer.get_buffer().as_entire_binding(),
                        &active_list.as_entire_binding(),
                        &fim_data.as_entire_binding(),
                        &fim_histogram.get_histogram_buffer().as_entire_binding(),
                    ],
                    vec![
                        &fim_data.as_entire_binding(), // TODO: shader without gpu_debug bindings.
                        &fim_data.as_entire_binding(), // TODO: shader without gpu_debug bindings.
                        &fim_data.as_entire_binding(), // TODO: shader without gpu_debug bindings.
                        &fim_data.as_entire_binding(), // TODO: shader without gpu_debug bindings.
                        &fim_data.as_entire_binding(), // TODO: shader without gpu_debug bindings.
                    ],
                ]
        );

        let pc_to_interface_compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("pc_to_interface_fmm.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            if cfg!(not(target_arch = "wasm32")) {
                                Cow::Borrowed(include_str!("../../assets/shaders/pc_to_interface_fmm.wgsl"))
                            }
                            else {
                                Cow::Borrowed(include_str!("../../assets/shaders/pc_to_interface_fmm_wasm.wgsl"))
                            }
                        ),
                    }),
                    Some("Cloud data to fmm Compute object fmm"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> fmm_params: FmmParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),
 
                            // @group(0) @binding(1) var<uniform> point_cloud_params: PointCloudParams;
                            create_uniform_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),
 
                            // @group(0) @binding(2) var<storage, read_write> fim_data: array<FmmCellPc>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),
 
                            // @group(0) @binding(3) var<storage, read_write> point_data: array<VVVC>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(4) var<storage, read_write> sample_data: array<SamplerCell>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
 
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
                                                  &fim_data,
                                                  &prefix_temp_array,
        );

        // Create FIM compute objects. 
        let initial_active_cells = Self::create_initial_active_cells( &device);
        let update_phase = Self::create_update_phase( &device);
        let pre_remedy = Self::create_pre_remedy( &device);
        let remedy_phase = Self::create_remedy_phase( &device);

        Self {
            compute_object: compute_object,
            compute_object_bind_groups: compute_object_bind_groups,
            buffers: buffers,
            fmm_params_buffer: fmm_params_buffer, 
            global_dimension: global_dimension,
            local_dimension: local_dimension,
            fim_data: fim_data,
            pc_to_interface_compute_object: pc_to_interface_compute_object,
            pc_to_interface_bind_groups: None,
            sample_data: sample_data,
            fmm_value_fixer: fmm_value_fixer,
            fmm_prefix_params: fmm_prefix_params,
            prefix_temp_array: prefix_temp_array,
            fim_histogram: fim_histogram,
            first_time: true,
            // fmm_state: fmm_state,
            //count_source_cells: count_source_cells,
            initial_active_cells: initial_active_cells,
            update_phase: update_phase,
            pre_remedy: pre_remedy,
            remedy_phase: remedy_phase,
        }
    }

    /// Create the initial interface using point cloud data.
    /// TODO: point sampling method.
    #[allow(dead_code)]
    pub fn initialize_interface_pc(&self, encoder: &mut wgpu::CommandEncoder, _pc: &PointCloud) {

        assert!(!self.pc_to_interface_bind_groups.is_none(), "Consider calling add_point_cloud_data before this function."); 

         self.pc_to_interface_compute_object.dispatch(
             &self.pc_to_interface_bind_groups.as_ref().unwrap(),
             encoder,
             1, 1, 1, Some("Point data to interface dispatch")
         );

         if cfg!(not(target_arch = "wasm32")) {
             self.fmm_value_fixer.dispatch(encoder, [udiv_up_safe32(self.calculate_cell_count(), 1024), 1, 1]);
         }
         else {
             self.fmm_value_fixer.dispatch(encoder, [udiv_up_safe32(self.calculate_cell_count(), 256), 1, 1]);
         }
    }

    pub fn fim_iteration(&mut self, encoder: &mut wgpu::CommandEncoder, gpu_timer: &mut Option<GpuTimer>) {

        // let number_of_dispatches = udiv_up_safe32(self.calculate_cell_count(), 1024);
        // let number_of_dispatches_64 = udiv_up_safe32(self.calculate_cell_count(), 64);
        // let number_of_dispatches_128 = udiv_up_safe32(self.calculate_cell_count(), 128);
        let number_of_dispatches_256 = udiv_up_safe32(self.calculate_cell_count(), 256);
        // let number_of_dispatches_1024 = udiv_up_safe32(self.calculate_cell_count(), 256);
        //let number_of_dispatches_2048 = udiv_up_safe32(self.calculate_cell_count(), 2048);
        // println!("number_of_dispatches == {}", number_of_dispatches);
        // println!("number_of_dispatches_64 == {}", number_of_dispatches_64);
        // println!("number_of_dispatches_128 == {}", number_of_dispatches_128);
        // println!("number_of_dispatches_256 == {}", number_of_dispatches_256);
        // println!("self.calculate_block_count() == {}", self.calculate_block_count());
        // let number_of_dispatches_prefix = udiv_up_safe32((self.global_dimension[0] * self.global_dimension[1] * self.global_dimension[2]) as u32, 1024 * 2);

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

            pass.set_pipeline(&self.initial_active_cells.pipeline);
            pass.dispatch_workgroups(number_of_dispatches_256, 1, 1); // TODO: dispatch_indirect

            pass.set_pipeline(&self.update_phase.pipeline);
            pass.dispatch_workgroups(1, 1, 1); // TODO: dispatch_indirect

            pass.set_pipeline(&self.pre_remedy.pipeline);
            pass.dispatch_workgroups(number_of_dispatches_256, 1, 1); // TODO: dispatch_indirect

            pass.set_pipeline(&self.remedy_phase.pipeline);
            pass.dispatch_workgroups(1, 1, 1); // TODO: dispatch_indirect
        }

        // println!("FIM: number_of_dispatches_256 == {}", number_of_dispatches_256);

        if !gpu_timer.is_none() {
            gpu_timer.as_mut().unwrap().end_pass(&mut pass);
        }
        drop(pass);
        self.first_time = false;
          
        if !gpu_timer.is_none() {
            gpu_timer.as_mut().unwrap().resolve_timestamps(encoder);
        }
    }

    pub fn print_fim_histogram(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        print!("{:?} ", self.fim_histogram.get_values(&device, queue));
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
                    &self.fim_data.as_entire_binding(),
                    &pc_data.as_entire_binding(),
                    &self.sample_data.as_entire_binding(),
                    ],
                ]
        );
        self.pc_to_interface_bind_groups = Some(pc_to_interface_bind_groups);
    }

    fn calculate_cell_count(&self) -> u32 {
        self.global_dimension[0] * self.global_dimension[1] * self.global_dimension[2] * self.local_dimension[0]  * self.local_dimension[1]  * self.local_dimension[2] 
    }

    // fn calculate_block_count(&self) -> u32 {
    //     self.global_dimension[0] * self.global_dimension[1] * self.global_dimension[2]
    // }

    pub fn get_fim_data_buffer(&self) -> &wgpu::Buffer {
        &self.fim_data
    }

    pub fn get_fmm_prefix_temp_buffer(&self) -> &wgpu::Buffer {
        &self.prefix_temp_array
    }

    pub fn get_fmm_params_buffer(&self) -> &wgpu::Buffer {
        &self.fmm_params_buffer.get_buffer()
    }

    fn create_initial_active_cells(device: &wgpu::Device) -> ComputeObject {
        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("create_initial_active_cells.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            if cfg!(not(target_arch = "wasm32")) {
                                Cow::Borrowed(include_str!("../../assets/shaders/fim_shaders/create_initial_active_cells.wgsl"))
                            }
                            else {
                                Cow::Borrowed(include_str!("../../assets/shaders/fim_shaders/wasm/create_initial_active_cells.wgsl"))
                            }
                        ),
                    }),
                    Some("Create initial active cells ComputeObject"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> prefix_params: PrefixParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<uniform> fmm_params: FmmParams;
                            create_uniform_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(2) var<storage, read_write> active_list: array<FmmBlock>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),
                              
                            // @group(0) @binding(6) var<storage,read_write>  fim_data: array<TempData>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(7) var<storage,read_write> fim_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None,
        );
        compute_object
    }

    fn create_update_phase(device: &wgpu::Device) -> ComputeObject {

        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("update_phase.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            if cfg!(not(target_arch = "wasm32")) {
                                Cow::Borrowed(include_str!("../../assets/shaders/fim_shaders/update_phase.wgsl"))
                            }
                            else {
                                Cow::Borrowed(include_str!("../../assets/shaders/fim_shaders/wasm/update_phase.wgsl"))
                            }
                        ),
                    }),
                    Some("Update phase ComputeObject"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> prefix_params: PrefixParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<uniform> fmm_params: FmmParams;
                            create_uniform_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(2) var<storage, read_write> active_list: array<FmmBlock>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),
                              
                            // @group(0) @binding(6) var<storage,read_write>  fim_data: array<TempData>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(7) var<storage,read_write> fim_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None,
        );
        compute_object
    }

    fn create_pre_remedy(device: &wgpu::Device) -> ComputeObject {
        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("pre_remedy.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            if cfg!(not(target_arch = "wasm32")) {
                                Cow::Borrowed(include_str!("../../assets/shaders/fim_shaders/pre_remedy.wgsl"))
                            }
                            else {
                                Cow::Borrowed(include_str!("../../assets/shaders/fim_shaders/wasm/pre_remedy.wgsl"))
                            }
                        ),
                    }),
                    Some("Pre-remedy ComputeObject"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> prefix_params: PrefixParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<uniform> fmm_params: FmmParams;
                            create_uniform_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(2) var<storage, read_write> active_list: array<FmmBlock>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),
                              
                            // @group(0) @binding(3) var<storage, read_write> temp_prefix_sum: array<u32>;
                            // create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),
                              
                            // @group(0) @binding(6) var<storage,read_write>  fim_data: array<TempData>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(7) var<storage,read_write> fim_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None,
        );
        compute_object
    }

    fn create_remedy_phase(device: &wgpu::Device) -> ComputeObject {

        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("remedy_phase.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            if cfg!(not(target_arch = "wasm32")) {
                                Cow::Borrowed(include_str!("../../assets/shaders/fim_shaders/remedy_phase.wgsl"))
                            }
                            else {
                                Cow::Borrowed(include_str!("../../assets/shaders/fim_shaders/wasm/remedy_phase.wgsl"))
                            }
                        ),
                    }),
                    Some("Remedy phase ComputeObject"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> prefix_params: PrefixParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<uniform> fmm_params: FmmParams;
                            create_uniform_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(2) var<storage, read_write> active_list: array<FmmBlock>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),
                              
                            // @group(0) @binding(3) var<storage, read_write> temp_prefix_sum: array<u32>;
                            // create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),
                              
                            // @group(0) @binding(6) var<storage,read_write>  fim_data: array<TempData>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(7) var<storage,read_write> fim_counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None,
        );
        compute_object
    }
}
