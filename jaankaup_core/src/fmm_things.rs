use crate::pc_parser::{read_pc_data, VVVC};
use crate::render_object::create_bind_groups;
use crate::common_functions::{
    udiv_up_safe32,
};
use std::borrow::Cow;
use crate::render_object::ComputeObject;
use crate::common_functions::{create_uniform_bindgroup_layout, create_buffer_bindgroup_layout};
use crate::gpu_debugger::GpuDebugger;
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
struct FmmParams {
    global_dimension: [u32; 3],
    padding: u32,
    local_dimension: [u32; 3],
    padding2: u32,
}

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
pub struct ComputationalDomain {
    global_dimension: [u32; 3],
    aabb_size: f32,
    local_dimension:  [u32; 3],
    font_size: f32,
    domain_iterator: [u32; 3],
    show_numbers: u32,
    // permutation_index: u32,
}

impl_convert!{FMMCell}
impl_convert!{ComputationalDomain}

/// A struct for FmmParams.
pub struct FmmParamsBuffer {
    params: FmmParams,
    params_buffer: wgpu::Buffer,
}

impl FmmParamsBuffer {
    pub fn create(device: &wgpu::Device,
                  global_dimension: [u32; 3],
                  local_dimension: [u32; 3])  -> Self {

        let params =  FmmParams {
            global_dimension: global_dimension,
            padding: 0,
            local_dimension: local_dimension,
            padding2: 0,
        };

        let buf = buffer_from_data::<FmmParams>(
                  &device,
                  &vec![params],
                  wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                  Some("FmmParams buffer.")
        );

        Self {
            params: params,
            params_buffer: buf,
        }
    }

    pub fn get_buffer(&self) -> &wgpu::Buffer {
        &self.params_buffer
    }

    pub fn get_global_dimension(&self) -> [u32; 3] { self.params.global_dimension }
    pub fn get_local_dimension(&self) -> [u32; 3] { self.params.local_dimension }
}

/// A struct for Computational domain data, operations and buffer.
pub struct ComputationalDomainBuffer {
    computational_domain: ComputationalDomain,
    buffer: wgpu::Buffer,
}

impl ComputationalDomainBuffer {
    
    /// Initialize and create ComputationalDomainBuffer object.
    pub fn create(device: &wgpu::Device,
                  global_dimension: [u32; 3],
                  local_dimension: [u32; 3],
                  aabb_size: f32,
                  font_size: f32,
                  show_numbers: bool,
                  domain_iterator: [u32; 3])  -> Self {

        // TODO: asserts

        let domain = ComputationalDomain {
            global_dimension: global_dimension,
            aabb_size: aabb_size,
            local_dimension: local_dimension,
            font_size: font_size,
            // permutation_index: permutation_index,
            domain_iterator: domain_iterator,
            show_numbers: if show_numbers { 1 } else { 0 },
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

    pub fn update_domain_iterator(&mut self, queue: &wgpu::Queue, domain_iterator: [u32; 3]) {
        // TODO: asserts
        self.computational_domain.domain_iterator = domain_iterator;
        self.update(queue);
    }

    pub fn get_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    pub fn get_computational_domain(&self) -> ComputationalDomain {
        self.computational_domain
    }

}

pub struct DomainTester {
    computational_domain_buffer: ComputationalDomainBuffer,
    compute_object: ComputeObject,
    bind_groups: Vec<wgpu::BindGroup>,
    domain_iterator: [u32; 3],
}

impl DomainTester {

    pub fn init(device: &wgpu::Device,
                gpu_debugger: &GpuDebugger,
                global_dimension: [u32; 3],
                local_dimension: [u32; 3],
                aabb_size: f32,
                font_size: f32,
                show_numbers: bool,
                domain_iterator: [u32 ; 3]
                ) -> Self {

        let shader = &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                      label: Some("domain_tester.wgsl"),
                      source: wgpu::ShaderSource::Wgsl(
                          Cow::Borrowed(include_str!("../../assets/shaders/domain_test.wgsl"))),
                      }
        );

        let computational_domain_buffer = ComputationalDomainBuffer::create(
                  device,
                  global_dimension,
                  local_dimension,
                  aabb_size,
                  font_size,
                  show_numbers,
                  domain_iterator
        );

        let compute_object =
                ComputeObject::init(
                    &device,
                    &shader,
                    Some("DomainTester Compute object"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> computational_domain: ComputationalDomain;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // // @group(0) @binding(1) var<storage,read_write> permutations: array<Permutation>;
                            // create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(2) var<storage,read_write> counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(3) var<storage,read_write> output_char: array<Char>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(4) var<storage,read_write> output_arrow: array<Arrow>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(5) var<storage,read_write> output_aabb: array<AABB>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> output_aabb_wire: array<AABB>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string()
        );

        let bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                    &computational_domain_buffer.get_buffer().as_entire_binding(),
                    &gpu_debugger.get_element_counter_buffer().as_entire_binding(),
                    &gpu_debugger.get_output_chars_buffer().as_entire_binding(),
                    &gpu_debugger.get_output_arrows_buffer().as_entire_binding(),
                    &gpu_debugger.get_output_aabbs_buffer().as_entire_binding(),
                    &gpu_debugger.get_output_aabb_wires_buffer().as_entire_binding(),
                    ],
                ]
        );

        Self {
            computational_domain_buffer,
            compute_object: compute_object, 
            bind_groups: bind_groups,
            domain_iterator: domain_iterator,
        }
    }

    pub fn update_font_size(&mut self, queue: &wgpu::Queue, font_size: f32) {
        self.computational_domain_buffer.computational_domain.font_size = font_size; 
        self.computational_domain_buffer.update(queue);
    }

    pub fn update_aabb_size(&mut self, queue: &wgpu::Queue, aabb_size: f32) {
        self.computational_domain_buffer.computational_domain.aabb_size = aabb_size; 
        self.computational_domain_buffer.update(queue);
    }

    pub fn update_domain_iterator(&mut self, queue: &wgpu::Queue, domain_iterator: [u32; 3]) {
        self.computational_domain_buffer.computational_domain.domain_iterator = domain_iterator; 
        self.computational_domain_buffer.update(queue);
    }

    pub fn update_show_numbers(&mut self, queue: &wgpu::Queue, show_numbers: bool) {
        self.computational_domain_buffer.computational_domain.show_numbers = if show_numbers { 1 } else { 0 }; 
        self.computational_domain_buffer.update(queue);
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {

        let global_dimension = self.computational_domain_buffer.get_computational_domain().global_dimension;
        let local_dimension = self.computational_domain_buffer.get_computational_domain().local_dimension;

        let total_grid_count =
                        global_dimension[0] *
                        global_dimension[1] *
                        global_dimension[2] *
                        local_dimension[0] *
                        local_dimension[1] *
                        local_dimension[2];

        self.compute_object.dispatch(
            &self.bind_groups,
            encoder,
            udiv_up_safe32(total_grid_count, 1024), 1, 1, Some("Domain tester dispatch")
        );
    }
}

fn load_pc_data(device: &wgpu::Device, src_file: &String) -> (u32, [f32; 3], [f32; 3], wgpu::Buffer) {

    let (aabb_min, aabb_max, pc_data) = read_pc_data(src_file);

    (pc_data.len() as u32,
     aabb_min,
     aabb_max,
     buffer_from_data::<VVVC>(
         &device,
         &pc_data,
         wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
         Some("Cloud data buffer")
         )
    )
}

/// Sample point for point cloud data.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FmmCellPc {
    pub tag: u32,
    pub value: u32,
    pub color: u32,
    // pub padding: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PointCloudParams {
    min_point: [f32; 3],
    point_count: u32,
    max_point: [f32; 3],
    scale_factor: f32,
    thread_group_number: u32,
    show_numbers: u32,
    padding: [u32; 2],
}

/// A Struct for PointCloudParams.
pub struct PointCloudParamsBuffer {
    pc_params_buffer: wgpu::Buffer,
    pc_params: PointCloudParams,
}

impl PointCloudParamsBuffer {
    pub fn create(device: &wgpu::Device, point_count: u32, aabb_min: [f32; 3], aabb_max: [f32; 3], scale_factor: f32, thread_group_number: u32, show_numbers: bool) -> Self {

        let params = PointCloudParams {
            min_point: aabb_min,
            point_count: point_count,
            max_point: aabb_max,
            scale_factor: scale_factor,
            thread_group_number: thread_group_number,
            show_numbers: if show_numbers { 1 } else { 0 }, 
            padding: [0, 0],
        };

        let buf = buffer_from_data::<PointCloudParams>(
                  &device,
                  &vec![params],
                  wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                  Some("PointCloudParams buffer.")
        );

        Self {
            pc_params_buffer: buf,
            pc_params: params,
        }
    }

    pub fn get_buffer(&self) -> &wgpu::Buffer {
        &self.pc_params_buffer
    }

    pub fn get_point_count(&self) -> u32 {
        self.pc_params.point_count
    }
}

/// Struct that generates and owns point data buffer.
pub struct PointCloud {
    point_cloud_buffer: wgpu::Buffer,
    point_count: u32,
    min_coord: [f32; 3],
    max_coord: [f32; 3],
}

impl PointCloud {

    /// Creates PointCloud structure from given v v v c c c data.
    /// TODO: Check if file_location exists.
    pub fn init(device: &wgpu::Device, file_location: &String) -> Self {

        let (point_count, aabb_min, aabb_max, buffer) = load_pc_data(device, file_location);

        Self {
            point_cloud_buffer: buffer,
            point_count: point_count,
            min_coord: aabb_min,
            max_coord: aabb_max,
        }
    }

    /// Get the minimum coordinate from point cloud data.
    pub fn get_min_coord(&self) -> [f32; 3] { self.min_coord }

    /// Get the maximum coordinate from point cloud data.
    pub fn get_max_coord(&self) -> [f32; 3] { self.max_coord }

    /// Get the data buffer.
    pub fn get_buffer(&self) -> &wgpu::Buffer { &self.point_cloud_buffer }

    /// Get point count.
    pub fn get_point_count(&self) -> u32 { self.point_count }
}

/// A single cell for point cloud data. Store the euclidin distance, color and fmm tag. 
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GridDataPc {
    distance: f32,
    color: u32,
    tag: u32,
}

/// A struct that offers some functionality for PointCloud data manipulation.
pub struct PointCloudHandler {
    compute_object_point_to_interface: ComputeObject,
    point_to_interface_bind_groups: Vec<wgpu::BindGroup>,
    fmm_params_buffer: FmmParamsBuffer, // TODO: from parameter, remove this 
    point_cloud_params_buffer: PointCloudParamsBuffer,
}

impl PointCloudHandler {

    pub fn init(device: &wgpu::Device,
                global_dimension: [u32; 3],
                local_dimension: [u32; 3],
                point_count: u32,
                aabb_min: [f32 ; 3],
                aabb_max: [f32 ; 3],
                scale_factor: f32,
                thread_group_number: u32,
                show_numbers: bool,
                fmm_data: &wgpu::Buffer,
                point_data: &wgpu::Buffer,
                gpu_debugger: &GpuDebugger) -> Self {

        // From parameter.
        let fmm_params_buffer = FmmParamsBuffer::create(&device, global_dimension, local_dimension);
        let point_cloud_params_buffer = PointCloudParamsBuffer::create(&device, point_count, aabb_min, aabb_max, scale_factor, thread_group_number, show_numbers);
        
        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("point_cloud_to_interface.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/point_cloud_to_interface.wgsl"))),

                    }),
                    Some("Cloud data to fmm Compute object"),
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
 
                            // @group(0) @binding(4) var<storage,read_write> counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
 
                            // @group(0) @binding(5) var<storage,read_write> output_char: array<Char>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),
 
                            // @group(0) @binding(6) var<storage,read_write> output_arrow: array<Arrow>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
 
                            // @group(0) @binding(7) var<storage,read_write> output_aabb: array<AABB>;
                            create_buffer_bindgroup_layout(7, wgpu::ShaderStages::COMPUTE, false),
 
                            // @group(0) @binding(8) var<storage,read_write> output_aabb_wire: array<AABB>;
                            create_buffer_bindgroup_layout(8, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string()
        );
 
        let point_to_interface_bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                    &fmm_params_buffer.get_buffer().as_entire_binding(),
                    &point_cloud_params_buffer.get_buffer().as_entire_binding(),
                    &fmm_data.as_entire_binding(),
                    &point_data.as_entire_binding(),
                    &gpu_debugger.get_element_counter_buffer().as_entire_binding(),
                    &gpu_debugger.get_output_chars_buffer().as_entire_binding(),
                    &gpu_debugger.get_output_arrows_buffer().as_entire_binding(),
                    &gpu_debugger.get_output_aabbs_buffer().as_entire_binding(),
                    &gpu_debugger.get_output_aabb_wires_buffer().as_entire_binding(),
                    ],
                ]
        );
 
        Self {
             compute_object_point_to_interface: compute_object,
             point_to_interface_bind_groups: point_to_interface_bind_groups,
             fmm_params_buffer: fmm_params_buffer,
             point_cloud_params_buffer: point_cloud_params_buffer,
        }
    }

    pub fn update_thread_group_number(&mut self, queue: &wgpu::Queue, work_group_number: u32) {
   
        self.point_cloud_params_buffer.pc_params.thread_group_number = work_group_number;
        self.update(queue);
    }

    /// Update buffer.
    fn update(&self, queue: &wgpu::Queue) {
        
        queue.write_buffer(
            &self.point_cloud_params_buffer.get_buffer(),
            0,
            bytemuck::cast_slice(&[self.point_cloud_params_buffer.pc_params])
        );
    }

     pub fn point_data_to_interface(&self, encoder: &mut wgpu::CommandEncoder) {

         let global_dimension = self.fmm_params_buffer.get_global_dimension();
         let local_dimension = self.fmm_params_buffer.get_local_dimension();

         let total_grid_count =
                         global_dimension[0] *
                         global_dimension[1] *
                         global_dimension[2] *
                         local_dimension[0] *
                         local_dimension[1] *
                         local_dimension[2];

         self.compute_object_point_to_interface.dispatch(
             &self.point_to_interface_bind_groups,
             encoder,
             1, 1, 1, Some("Point data to interface dispatch")
             // udiv_up_safe32(self.point_cloud_params_buffer.pc_params.point_count, 1024), 1, 1, Some("Point data to interface dispatch")
         );
     }
}

pub struct FmmValueFixer {
    compute_object: ComputeObject,
    bind_groups: Vec<wgpu::BindGroup>,
}

impl FmmValueFixer {

    pub fn init(device: &wgpu::Device,
                fmm_params_buffer: &wgpu::Buffer,
                fmm_data: &wgpu::Buffer,
                ) -> Self {

        let compute_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("fmm_value_fixer.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_value_fixer.wgsl"))),

                    }),
                    Some("FmmValueFixer ComputeObject"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> fmm_params: FmmParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<storage,read_write> fmm_data: array<FmmCellPc>;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string()
        );

        let bind_groups = create_bind_groups(
                &device,
                &compute_object.bind_group_layout_entries,
                &compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &fmm_params_buffer.as_entire_binding(),
                        &fmm_data.as_entire_binding(),
                    ],
                ]
        );

        Self {
            compute_object: compute_object, 
            bind_groups: bind_groups,
        }
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, dispatch_dimensions: [u32; 3]) {

        self.compute_object.dispatch(
            &self.bind_groups,
            encoder,
            dispatch_dimensions[0], dispatch_dimensions[1], dispatch_dimensions[2], Some("Domain tester dispatch")
        );
    }
}
