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

// /// parameters for permutations.
// #[repr(C)]
// #[derive(Debug, Clone, Copy, Pod, Zeroable)]
// pub struct Permutation {
//     pub modulo: u32,
//     pub x_factor: u32,  
//     pub y_factor: u32,  
//     pub z_factor: u32,  
// }

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
    padding: u32,
    // permutation_index: u32,
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
                  local_dimension: [u32; 3],
                  aabb_size: f32,
                  font_size: f32,
                  domain_iterator: [u32; 3])  -> Self {

        // TODO: asserts

        let domain = ComputationalDomain {
            global_dimension: global_dimension,
            aabb_size: aabb_size,
            local_dimension: local_dimension,
            font_size: font_size,
            // permutation_index: permutation_index,
            domain_iterator: domain_iterator,
            padding: 0,
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

    // pub fn update_permutation_index(&mut self, queue: &wgpu::Queue, permutation_index: u32) {
    //     // TODO: asserts
    //     self.computational_domain.permutation_index = permutation_index;
    //     self.update(queue);
    // }

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
    // permutations: Vec<Permutation>,
    // permutations_buffer: wgpu::Buffer,
}

impl DomainTester {

    pub fn init(device: &wgpu::Device,
                gpu_debugger: &GpuDebugger,
                global_dimension: [u32; 3],
                local_dimension: [u32; 3],
                aabb_size: f32,
                font_size: f32,
                domain_iterator: [u32 ; 3]
                //permutations: &Vec<Permutation>,
                //permutation_index: u32,
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
                  domain_iterator,
                  //permutations: permutations,
                  //permutation_index
        );

        // let permutations_buffer = buffer_from_data::<Permutation>(
        //           &device,
        //           permutations,
        //           wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        //           Some("Permutations wgpu::buffer.")
        // );

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
                    // &permutations_buffer.as_entire_binding(),
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
            // permutations_buffer: permutations_buffer,
            // permutations: permutations.to_vec(),
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

/// Struct that generates and own point data buffer.
pub struct PointCloud {
    point_cloud_buffer: wgpu::Buffer,
    point_count: u32,
    min_coord: [f32; 3],
    max_coord: [f32; 3],
    // compute_object: ComputeObject,
    // bind_groups: Vec<wgpu::BindGroup>,
}

impl PointCloud {

    pub fn init(device: &wgpu::Device, file_location: &String) -> Self {

        let (point_count, aabb_min, aabb_max, buffer) = load_pc_data(device, file_location);

        // let result = read_pc_data(src_file);

        Self {
            point_cloud_buffer: buffer,
            point_count: point_count,
            min_coord: aabb_min,
            max_coord: aabb_max,
        }
    }

    pub fn get_min_coord(&self) -> [f32; 3] { self.min_coord }

    pub fn get_max_coord(&self) -> [f32; 3] { self.max_coord }

    pub fn get_buffer(&self) -> &wgpu::Buffer { &self.point_cloud_buffer }
}

