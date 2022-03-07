use crate::render_object::{RenderObject, ComputeObject, create_bind_groups,draw};
use std::borrow::Cow;
use crate::misc::Convert2Vec;
use crate::impl_convert;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use crate::common_functions::{
    udiv_up_safe32,
    create_uniform_bindgroup_layout,
    create_buffer_bindgroup_layout
};
use crate::camera::Camera;
use crate::histogram::Histogram;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Vertex {
    v: [f32; 4],
    n: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Triangle {
    a: Vertex,
    b: Vertex,
    c: Vertex,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AABB {
    min: [f32; 4],
    max: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Arrow {
    start_pos: [f32 ; 4],
    end_pos: [f32 ; 4],
    color: u32,
    size: f32,
    _padding: [u32; 2]
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Char {
    start_pos: [f32 ; 4],
    value: [f32 ; 4],
    font_size: f32,
    vec_dim_count: u32, // 1 => f32, 2 => vec3<f32>, 3 => vec3<f32>, 4 => vec4<f32>
    color: u32,
    z_offset: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ArrowAabbParams{
    max_number_of_vertices: u32,
    iterator_start_index: u32,
    iterator_end_index: u32,
    element_type: u32, // 0 :: array, 1 :: aabb, 2 :: aabb wire
}

impl_convert!{Arrow}
impl_convert!{Char}

pub struct GpuDebugger {
    pub compute_object_char: ComputeObject, 
    pub compute_bind_groups_char: Vec<wgpu::BindGroup>,
    pub compute_object_arrow: ComputeObject, 
    pub compute_bind_groups_arrow: Vec<wgpu::BindGroup>,
    pub buffers: HashMap<String, wgpu::Buffer>,
}

//++ const MAX_NUMBER_OF_VVVVNNNN: usize = 2000000;
//++ const MAX_NUMBER_OF_VVVC: usize = MAX_NUMBER_OF_VVVVNNNN * 2;
//++ 
//++ /// The size of draw buffer in bytes;
//++ const VERTEX_BUFFER_SIZE: usize = MAX_NUMBER_OF_VVVVNNNN * size_of::<Vertex>();

impl GpuDebugger {

    pub fn Init(device: &wgpu::Device, camera: &mut Camera, vertex_buffer_size: u32) -> Self {

        let mut buffers: HashMap<String, wgpu::Buffer> = HashMap::new();

        let histogram_draw_counts = Histogram::init(&device, &vec![0; 2]);

        let arrow_aabb_params = ArrowAabbParams {
            max_number_of_vertices: 123 as u32,
            // max_number_of_vertices: VERTEX_BUFFER_SIZE as u32,
            iterator_start_index: 0,
            iterator_end_index: 0,
            element_type: 0,
        };

        ////////////////////////////////////////////////////
        ////                 Compute char               ////
        ////////////////////////////////////////////////////

        let compute_object_char =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("numbers.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/numbers.wgsl"))),
                    
                    }),
                    Some("Visualizer Compute object"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> camerauniform: Camera;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<uniform> arrow_aabb_params: VisualizationParams;
                            create_uniform_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(2) var<storage, read_write> counter: Counter;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(3) var<storage, read> counter: array<Arrow>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(4) var<storage,read_write> output: array<VertexBuffer>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE),
                        ],
                    ]
        );

        let compute_bind_groups_char = create_bind_groups(
                                      &device,
                                      &compute_object_char.bind_group_layout_entries,
                                      &compute_object_char.bind_group_layouts,
                                      &vec![
                                          vec![
                                               &camera.get_camera_uniform(&device).as_entire_binding(),
                                               &buffers.get(&"arrow_aabb_params".to_string()).unwrap().as_entire_binding(),
                                               &histogram_draw_counts.get_histogram_buffer().as_entire_binding(),
                                               &buffers.get(&"output_chars".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output_render".to_string()).unwrap().as_entire_binding()
                                          ]
                                      ]
        );

        ////////////////////////////////////////////////////
        ////               Compute arrow/aabb           ////
        ////////////////////////////////////////////////////

        let compute_object_arrow =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("arrow_aabb.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/arrow_aabb.wgsl"))),
                    
                    }),
                    Some("Arrow_aabb Compute object"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> arrow_aabb_params: VisualizationParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<storage, read_write> counter: Counter;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(2) var<storage, read> counter: array<Arrow>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(3) var<storage, read_write> aabbs: array<AABB>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(4) var<storage, read_write> aabb_wires: array<AABB>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(5) var<storage,read_write> output: array<Triangle>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE),
                        ],
                    ]
        );

        let compute_bind_groups_arrow = create_bind_groups(
                                      &device,
                                      &compute_object_arrow.bind_group_layout_entries,
                                      &compute_object_arrow.bind_group_layouts,
                                      &vec![
                                          vec![
                                               &buffers.get(&"arrow_aabb_params".to_string()).unwrap().as_entire_binding(),
                                               &histogram_draw_counts.get_histogram_buffer().as_entire_binding(),
                                               &buffers.get(&"output_arrows".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output_aabbs".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output_aabb_wires".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output_render".to_string()).unwrap().as_entire_binding()
                                          ]
                                      ]
        );

        Self {
            compute_object_char: compute_object_char,
            compute_bind_groups_char: compute_bind_groups_char,
            compute_object_arrow: compute_object_arrow, 
            compute_bind_groups_arrow: compute_bind_groups_arrow,
            buffers: buffers,
            // arrow_aabb_params: arrow_aabb_params,
        }
    }
}
