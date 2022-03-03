use std::mem::size_of;
use std::collections::HashMap;
use std::borrow::Cow;
use jaankaup_core::template::{
        WGPUFeatures,
        WGPUConfiguration,
        Application,
        BasicLoop,
};
use jaankaup_core::render_object::{RenderObject, ComputeObject, create_bind_groups,draw};
use jaankaup_core::input::*;
use jaankaup_core::model_loader::*;
use jaankaup_core::aabb::Triangle_vvvvnnnn;
use jaankaup_core::camera::Camera;
use jaankaup_core::buffer::{buffer_from_data};
use jaankaup_core::wgpu;
use jaankaup_core::winit;
use jaankaup_core::log;
use jaankaup_core::screen::ScreenTexture;
use jaankaup_core::texture::Texture;
use jaankaup_core::misc::Convert2Vec;
use jaankaup_core::impl_convert;
use jaankaup_core::common_functions::{encode_rgba_u32, udiv_up_safe32};
use jaankaup_algorithms::histogram::Histogram;
use bytemuck::{Pod, Zeroable};
use jaankaup_core::cgmath::Vector4 as Vec4;
// use cgmath::{prelude::*, Vector3, Vector4};

use winit::event as ev;
pub use ev::VirtualKeyCode as Key;

// TODO: add to fmm params.
const MAX_NUMBERS_OF_ARROWS:     usize = 40960;
const MAX_NUMBERS_OF_AABBS:      usize = 262144;
const MAX_NUMBERS_OF_AABB_WIRES: usize = 40960;
const MAX_NUMBERS_OF_CHARS:      usize = 40960;

// FMM global dimensions.
const FMM_GLOBAL_X: usize = 16; 
const FMM_GLOBAL_Y: usize = 16; 
const FMM_GLOBAL_Z: usize = 16; 

// FMM inner dimensions.
const FMM_INNER_X: usize = 4; 
const FMM_INNER_Y: usize = 4; 
const FMM_INNER_Z: usize = 4; 

const MAX_NUMBER_OF_VVVVNNNN: usize = 2000000;
const MAX_NUMBER_OF_VVVC: usize = MAX_NUMBER_OF_VVVVNNNN * 2;

/// The size of draw buffer in bytes;
const VERTEX_BUFFER_SIZE: usize = MAX_NUMBER_OF_VVVVNNNN * size_of::<Vertex>();
//const VERTEX_BUFFER_SIZE: usize = MAX_NUMBER_OF_VVVVNNNN * size_of::<Vertex>() / 4;

const THREAD_COUNT: u32 = 64;

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

// #[repr(C)]
// #[derive(Debug, Clone, Copy, Pod, Zeroable)]
// struct VisualizationParams{
//     max_number_of_vertices: u32,
//     iterator_start_index: u32,
//     iterator_end_index: u32,
//     arrow_size: f32, // 0 :: array, 1 :: aabb, 2 :: aabb wire
// }

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ArrowAabbParams{
    max_number_of_vertices: u32,
    iterator_start_index: u32,
    iterator_end_index: u32,
    element_type: u32, // 0 :: array, 1 :: aabb, 2 :: aabb wire
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FmmCell {
    tag: u32,
    value: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FmmParams {
    fmm_global_dimension: [u32; 3],
    visualize: u32,
    fmm_inner_dimension: [u32; 3],
    triangle_count: u32,
}

impl_convert!{Arrow}
impl_convert!{Char}
impl_convert!{FmmCell}
impl_convert!{FmmParams}
impl_convert!{ArrowAabbParams}

struct FmmFeatures {}

impl WGPUFeatures for FmmFeatures {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::empty()
    }
    fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }
    fn required_limits() -> wgpu::Limits {
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 8;
        limits
    }
}

// State for this application.
struct Fmm {
    pub screen: ScreenTexture, 
    pub render_object_vvvvnnnn: RenderObject, 
    pub render_bind_groups_vvvvnnnn: Vec<wgpu::BindGroup>,
    pub render_object_vvvc: RenderObject,
    pub render_bind_groups_vvvc: Vec<wgpu::BindGroup>,
    pub compute_object_char: ComputeObject, 
    pub compute_bind_groups_char: Vec<wgpu::BindGroup>,
    pub compute_object_arrow: ComputeObject, 
    pub compute_bind_groups_arrow: Vec<wgpu::BindGroup>,
    pub compute_object_fmm: ComputeObject, 
    pub compute_bind_groups_fmm: Vec<wgpu::BindGroup>,
    pub compute_object_fmm_triangle: ComputeObject, 
    pub compute_bind_groups_fmm_triangle: Vec<wgpu::BindGroup>,
    // pub _textures: HashMap<String, Texture>,
    pub buffers: HashMap<String, wgpu::Buffer>,
    pub camera: Camera,
    pub histogram_draw_counts: Histogram,
    pub histogram_fmm: Histogram,
    pub draw_count_points: u32,
    pub draw_count_triangles: u32,
    pub arrow_aabb_params: ArrowAabbParams,
    pub keys: KeyboardManager,
    pub block64mode: bool, // TODO: remove
    pub triangle_mesh_draw_count: u32,
}

impl Fmm {

}

//#[allow(unused_variables)]
impl Application for Fmm {

    fn init(configuration: &WGPUConfiguration) -> Self {

        log::info!("Adapter limits are: ");

        // let adapter_limits = configuration.adapter.limits(); 

        // log::info!("max_compute_workgroup_storage_size: {:?}", adapter_limits.max_compute_workgroup_storage_size);
        // log::info!("max_compute_invocations_per_workgroup: {:?}", adapter_limits.max_compute_invocations_per_workgroup);
        // log::info!("max_compute_workgroup_size_x: {:?}", adapter_limits.max_compute_workgroup_size_x);
        // log::info!("max_compute_workgroup_size_y: {:?}", adapter_limits.max_compute_workgroup_size_y);
        // log::info!("max_compute_workgroup_size_z: {:?}", adapter_limits.max_compute_workgroup_size_z);
        // log::info!("max_compute_workgroups_per_dimension: {:?}", adapter_limits.max_compute_workgroups_per_dimension);

        let mut buffers: HashMap<String, wgpu::Buffer> = HashMap::new();

        let mut keys = KeyboardManager::init();
        // keys.register_key(Key::L, 20.0);

        // Camera.
        let mut camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (0.0, 0.0, 10.0), -89.0, 0.0);
        camera.set_rotation_sensitivity(0.4);
        camera.set_movement_sensitivity(0.02);

        // vvvvnnnn
        let render_object_vvvvnnnn =
                RenderObject::init(
                    &configuration.device,
                    &configuration.sc_desc,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("renderer_v4n4_debug_visualizator.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/renderer_v4n4_debug_visualizator.wgsl"))),
                    
                    }),
                    &vec![wgpu::VertexFormat::Float32x4, wgpu::VertexFormat::Float32x4],
                    &vec![
                        // Group 0
                        vec![wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                    },
                                count: None,
                            },
                        ],
                    ],
                    Some("Debug visualizator vvvvnnnn renderer with camera."),
                    true,
                    wgpu::PrimitiveTopology::TriangleList
        );
        let render_bind_groups_vvvvnnnn = create_bind_groups(
                                     &configuration.device,
                                     &render_object_vvvvnnnn.bind_group_layout_entries,
                                     &render_object_vvvvnnnn.bind_group_layouts,
                                     &vec![
                                         vec![&wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                                 buffer: &camera.get_camera_uniform(&configuration.device),
                                                 offset: 0,
                                                 size: None,
                                         })],
                                     ]
        );

        // vvvc
        let render_object_vvvc =
                RenderObject::init(
                    &configuration.device,
                    &configuration.sc_desc,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("renderer_v3c1.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/renderer_v3c1.wgsl"))),
                    
                    }),
                    &vec![wgpu::VertexFormat::Float32x3, wgpu::VertexFormat::Uint32],
                    &vec![
                        // Group 0
                        vec![wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                    },
                                count: None,
                            },
                        ],
                    ],
                    Some("Debug visualizator vvvc renderer with camera."),
                    true,
                    wgpu::PrimitiveTopology::PointList
        );
        let render_bind_groups_vvvc = create_bind_groups(
                                     &configuration.device,
                                     &render_object_vvvc.bind_group_layout_entries,
                                     &render_object_vvvc.bind_group_layouts,
                                     &vec![
                                         vec![&wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                                 buffer: &camera.get_camera_uniform(&configuration.device),
                                                 offset: 0,
                                                 size: None,
                                         })],
                                     ]
        );

        println!("Creating compute object.");

        let histogram_draw_counts = Histogram::init(&configuration.device, &vec![0; 2]);
        let histogram_fmm = Histogram::init(&configuration.device, &vec![0; 4]);

        let arrow_aabb_params = ArrowAabbParams {
            max_number_of_vertices: VERTEX_BUFFER_SIZE as u32,
            iterator_start_index: 0,
            iterator_end_index: 0,
            element_type: 0,
        };
        
        ////////////////////////////////////////////////////
        ////                 BUFFERS                    ////
        ////////////////////////////////////////////////////


        // Load model. Tower.
        let (_, mut triangle_mesh_wood, _) = load_triangles_from_obj(
            "assets/models/wood.obj",
            2.0,
            [5.0, -2.0, 5.0],
            None)
            .unwrap(); // -> Option<(Vec<Triangle>, Vec<Triangle_vvvvnnnn>, BBox)> {
        let triangle_mesh_draw_count = triangle_mesh_wood.len() as u32; 
        println!("triangle_mesh_draw_count == {}", triangle_mesh_draw_count);

        let color = encode_rgba_u32(255, 0, 0, 255) as f32;

        // Apply color information to the fourth component (triangle position).
        for tr in triangle_mesh_wood.iter_mut() {
            tr.a.w = color;
            tr.b.w = color;
            tr.c.w = color;
        }

        buffers.insert(
            "triangle_mesh".to_string(),
            buffer_from_data::<Triangle_vvvvnnnn>(
            &configuration.device,
            &triangle_mesh_wood,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            //wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            None)
        );

        // buffers.insert(
        //     "triangle_mesh".to_string(),
        //     buffer_from_data::<Triangle>(
        //     &configuration.device,
        //     &vec![Triangle { a: Vertex {v: [1.0, 1.0, 1.0, 1.0], n: [1.0, 1.0, 1.0, 1.0]},
        //                      b: Vertex {v: [1.0, 1.0, 1.0, 1.0], n: [1.0, 1.0, 1.0, 1.0]},
        //                      c: Vertex {v: [1.0, 1.0, 1.0, 1.0], n: [1.0, 1.0, 1.0, 1.0]},
        //                    }],
        //     wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        //     None)
        // );

        buffers.insert(
            "fmm_params".to_string(),
            buffer_from_data::<FmmParams>(
            &configuration.device,
            &vec![FmmParams {
                     fmm_global_dimension: [16, 16, 16],
                     visualize: 0,
                     fmm_inner_dimension: [4, 4, 4],
                     triangle_count: triangle_mesh_draw_count,
            }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            None)
        );

        buffers.insert(
            "fmm_data".to_string(),
            buffer_from_data::<FmmCell>(
            &configuration.device,
            &vec![FmmCell { tag: 0, value: 1000000.0, } ; FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z * FMM_INNER_X * FMM_INNER_Y * FMM_INNER_Z],
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            Some("fmm data buffer.")
            //configuration.device.create_buffer(&wgpu::BufferDescriptor{
            //    label: Some("output_arrays buffer"),
            //    size: (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z * FMM_INNER_X * FMM_INNER_Y * FMM_INNER_Z * std::mem::size_of::<FmmCell>()) as u64,
            //    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            //    mapped_at_creation: false,
            //    }
            )
        );

        buffers.insert(
            "output_arrows".to_string(),
            configuration.device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("output_arrays buffer"),
                size: (MAX_NUMBERS_OF_ARROWS * std::mem::size_of::<Arrow>()) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
            )
        );
        buffers.insert(
            "output_chars".to_string(),
            configuration.device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("output_chars"),
                size: (MAX_NUMBERS_OF_CHARS * std::mem::size_of::<Char>()) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
            )
        );

        buffers.insert(
            "output_aabbs".to_string(),
            configuration.device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("output_aabbs"),
                size: (MAX_NUMBERS_OF_AABBS * std::mem::size_of::<AABB>()) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
            )
        );

        buffers.insert(
            "output_aabb_wires".to_string(),
            configuration.device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("output_aabbs"),
                size: (MAX_NUMBERS_OF_AABB_WIRES * std::mem::size_of::<AABB>()) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
            )
        );

        buffers.insert(
            "output_render".to_string(),
            configuration.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("draw buffer"),
                size: VERTEX_BUFFER_SIZE as u64, 
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        );

        buffers.insert(
            "arrow_aabb_params".to_string(),
            buffer_from_data::<ArrowAabbParams>(
            &configuration.device,
            &vec![arrow_aabb_params],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            None)
        );

        ////////////////////////////////////////////////////
        ////                 Compute char               ////
        ////////////////////////////////////////////////////

        let compute_object_char =
                ComputeObject::init(
                    &configuration.device,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("numbers.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/numbers.wgsl"))),
                    
                    }),
                    Some("Visualizer Compute object"),
                    &vec![
                        vec![
                            // @group(0)
                            // @binding(0)
                            // var<uniform> camerauniform: Camera;
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(1)
                            // var<uniform> arrow_aabb_params: VisualizationParams;
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(2)
                            // var<storage, read_write> counter: Counter;
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(3)
                            // var<storage, read> counter: array<Arrow>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(4)
                            // var<storage,read_write> output: array<VertexBuffer>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    ]
        );

        println!("Creating compute bind groups.");

        // println!("{:?}", buffers.get(&"debug_arrays".to_string()).unwrap().as_entire_binding());

        let compute_bind_groups_char = create_bind_groups(
                                      &configuration.device,
                                      &compute_object_char.bind_group_layout_entries,
                                      &compute_object_char.bind_group_layouts,
                                      &vec![
                                          vec![
                                               &camera.get_camera_uniform(&configuration.device).as_entire_binding(),
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
                    &configuration.device,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("arrow_aabb.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/arrow_aabb.wgsl"))),
                    
                    }),
                    Some("Visualizer Compute object"),
                    &vec![
                        vec![
                            // @group(0)
                            // @binding(0)
                            // var<uniform> arrow_aabb_params: VisualizationParams;
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(1)
                            // var<storage, read_write> counter: Counter;
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(2)
                            // var<storage, read> counter: array<Arrow>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(3)
                            // var<storage, read_write> aabbs: array<AABB>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(4)
                            // var<storage, read_write> aabb_wires: array<AABB>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(5)
                            // var<storage,read_write> output: array<Triangle>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 5,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    ]
        );

        let compute_bind_groups_arrow = create_bind_groups(
                                      &configuration.device,
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

        ////////////////////////////////////////////////////
        ////               Compute fmm                  ////
        ////////////////////////////////////////////////////

        let compute_object_fmm =
                ComputeObject::init(
                    &configuration.device,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("fmm.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm.wgsl"))),
                    
                    }),
                    Some("FMM Compute object"),
                    &vec![
                        vec![
                            // @group(0)
                            // @binding(0)
                            // var<uniform> fmm_params: FmmParams;
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(1)
                            // var<storage, read_write> fmm_data: array<FmmCell>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(2)
                            // var<storage, read_write> counter: array<atomic<u32>>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(3)
                            // var<storage,read_write> output_char: array<Char>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(4)
                            // var<storage,read_write> output_arrow: array<Arrow>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(5)
                            // var<storage,read_write> output_aabb: array<AABB>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 5,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(6)
                            // var<storage,read_write> output_aabb_wire: array<AABB>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 6,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    ]
        );

        let compute_bind_groups_fmm = create_bind_groups(
                                      &configuration.device,
                                      &compute_object_fmm.bind_group_layout_entries,
                                      &compute_object_fmm.bind_group_layouts,
                                      &vec![
                                          vec![
                                               &buffers.get(&"fmm_params".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"fmm_data".to_string()).unwrap().as_entire_binding(),
                                               &histogram_fmm.get_histogram_buffer().as_entire_binding(),
                                               &buffers.get(&"output_chars".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output_arrows".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output_aabbs".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output_aabb_wires".to_string()).unwrap().as_entire_binding()
                                          ]
                                      ]
        );

        ////////////////////////////////////////////////////
        ////               Compute fmm_triangle         ////
        ////////////////////////////////////////////////////

        let compute_object_fmm_triangle =
                ComputeObject::init(
                    &configuration.device,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("triangle_mest_to_interface.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/triangle_mesh_to_interface.wgsl"))),
                    
                    }),
                    Some("FMM triangle Compute object"),
                    &vec![
                        vec![
                            // @group(0)
                            // @binding(0)
                            // var<uniform> fmm_params: FmmParams;
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(1)
                            // var<storage, read_write> fmm_data: array<FmmCell>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(2)
                            // var<storage, read_write> triangle_mesh_in: array<Triangle>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(3)
                            // var<storage, read_write> counter: array<atomic<u32>>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(4)
                            // var<storage,read_write> output_char: array<Char>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(5)
                            // var<storage,read_write> output_arrow: array<Arrow>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 5,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(6)
                            // var<storage,read_write> output_aabb: array<AABB>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 6,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            // @group(0)
                            // @binding(7)
                            // var<storage,read_write> output_aabb_wire: array<AABB>;
                            wgpu::BindGroupLayoutEntry {
                                binding: 7,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    ]
        );

        let compute_bind_groups_fmm_triangle =
            create_bind_groups(
                &configuration.device,
                &compute_object_fmm_triangle.bind_group_layout_entries,
                &compute_object_fmm_triangle.bind_group_layouts,
                &vec![
                    vec![
                         &buffers.get(&"fmm_params".to_string()).unwrap().as_entire_binding(),
                         &buffers.get(&"fmm_data".to_string()).unwrap().as_entire_binding(),
                         &buffers.get(&"triangle_mesh".to_string()).unwrap().as_entire_binding(),
                         &histogram_fmm.get_histogram_buffer().as_entire_binding(),
                         &buffers.get(&"output_chars".to_string()).unwrap().as_entire_binding(),
                         &buffers.get(&"output_arrows".to_string()).unwrap().as_entire_binding(),
                         &buffers.get(&"output_aabbs".to_string()).unwrap().as_entire_binding(),
                         &buffers.get(&"output_aabb_wires".to_string()).unwrap().as_entire_binding()
                    ]
                ]
        );


        println!("Creating render bind groups.");
 
        Self {
            screen: ScreenTexture::init(&configuration.device, &configuration.sc_desc, true),
            render_object_vvvvnnnn: render_object_vvvvnnnn,
            render_bind_groups_vvvvnnnn: render_bind_groups_vvvvnnnn,
            render_object_vvvc: render_object_vvvc,
            render_bind_groups_vvvc: render_bind_groups_vvvc,
            compute_object_char: compute_object_char,
            compute_bind_groups_char: compute_bind_groups_char,
            compute_object_arrow: compute_object_arrow, 
            compute_bind_groups_arrow: compute_bind_groups_arrow,
            compute_object_fmm: compute_object_fmm, 
            compute_bind_groups_fmm: compute_bind_groups_fmm,
            compute_object_fmm_triangle: compute_object_fmm_triangle, 
            compute_bind_groups_fmm_triangle: compute_bind_groups_fmm_triangle,
            // _textures: textures,
            buffers: buffers,
            camera: camera,
            histogram_draw_counts: histogram_draw_counts,
            histogram_fmm: histogram_fmm,
            draw_count_points: 0,
            draw_count_triangles: 0,
            arrow_aabb_params: arrow_aabb_params,
            keys: keys,
            block64mode: false,
            triangle_mesh_draw_count: triangle_mesh_draw_count, 
        }
    }

    fn render(&mut self,
              device: &wgpu::Device,
              queue: &mut wgpu::Queue,
              surface: &wgpu::Surface,
              sc_desc: &wgpu::SurfaceConfiguration) {

        // println!("acquiring screen texture.");

        self.screen.acquire_screen_texture(
            &device,
            &sc_desc,
            &surface
        );

        let thread_count = 64;

        let view = self.screen.surface_texture.as_ref().unwrap().texture.create_view(&wgpu::TextureViewDescriptor::default());

        //// EXECUTE FMM ////
        let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Fmm command encoder") });

        queue.write_buffer(
            &self.buffers.get(&"fmm_params".to_string()).unwrap(),
            0,
            bytemuck::cast_slice(&[FmmParams {
                                      fmm_global_dimension: [16, 16, 16],
                                      visualize: 0,
                                      fmm_inner_dimension: [4, 4, 4],
                                      triangle_count: 1 }
                                      //triangle_count: 2036 }
            ]));

        //self.compute_object_fmm.dispatch(
        //    &self.compute_bind_groups_fmm,
        //    &mut encoder_command,
        //    1, 1, 1, Some("fmm dispatch")
        //);

        // Compute interface.
        self.compute_object_fmm_triangle.dispatch(
            &self.compute_bind_groups_fmm_triangle,
            &mut encoder_command,
            udiv_up_safe32(2036, thread_count), 1, 1,
            // (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z) as u32, 1, 1,
            Some("fmm triangle dispatch")
        );

        queue.submit(Some(encoder_command.finish()));

       //  let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Fmm command encoder") });

       //  queue.write_buffer(
       //      &self.buffers.get(&"fmm_params".to_string()).unwrap(),
       //      0,
       //      bytemuck::cast_slice(&[FmmParams {
       //                                fmm_global_dimension: [16, 16, 16],
       //                                visualize: 1,
       //                                fmm_inner_dimension: [4, 4, 4],
       //                                triangle_count: 2036 }
       //      ]));

       //  //     buffer_from_data::<FmmParams>(
       //  //     &configuration.device,
       //  //     &vec![FmmParams {
       //  //              fmm_global_dimension: [16, 16, 16],
       //  //              visualize: 0,
       //  //              fmm_inner_dimension: [4, 4, 4],
       //  //              padding2: 123,
       //  //     }],

       //  // Visualize.
       //  self.compute_object_fmm_triangle.dispatch(
       //      &self.compute_bind_groups_fmm_triangle,
       //      &mut encoder_command,
       //      (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z) as u32, 1, 1,
       //      Some("fmm triangle dispatch")
       //  );

       //  queue.submit(Some(encoder_command.finish()));

        let fmm_counter = self.histogram_fmm.get_values(device, queue);

        // Get the total number of elements.
        let number_of_chars = fmm_counter[0];
        let total_number_of_arrows = fmm_counter[1];
        let total_number_of_aabbs = fmm_counter[2];
        let total_number_of_aabb_wires = fmm_counter[3];
        
        let vertices_per_element_arrow = 72;
        let vertices_per_element_aabb = 36;
        let vertices_per_element_aabb_wire = 432;

        // The number of vertices created with one dispatch.
        let vertices_per_dispatch_arrow = thread_count * vertices_per_element_arrow;
        let vertices_per_dispatch_aabb = thread_count * vertices_per_element_aabb;
        let vertices_per_dispatch_aabb_wire = thread_count * vertices_per_element_aabb_wire;

        // [(element_type, total number of elements, number of vercies per dispatch, vertices_per_element)]
        let draw_params = [(0, total_number_of_arrows,     vertices_per_dispatch_arrow, vertices_per_element_arrow),
                           (1, total_number_of_aabbs,      vertices_per_dispatch_aabb, vertices_per_element_aabb), // !!!
                           (2, total_number_of_aabb_wires, vertices_per_dispatch_aabb_wire, vertices_per_element_aabb_wire)]; 

        // Clear the previous screen.
        let mut clear = true;

        // For each element type, create triangle meshes and render with respect of draw buffer size.
        for (e_type, e_size, v_per_dispatch, vertices_per_elem) in draw_params.iter() {
            println!("*****************");

            // The number of safe dispathes. This ensures the draw buffer doesn't over flow.
            let safe_number_of_dispatches = MAX_NUMBER_OF_VVVVNNNN as u32 / v_per_dispatch;

            // // The number of remaining dispatches to complete the triangle mesh creation and
            // // rendering.
            // let mut total_number_of_dispatches = udiv_up_safe32(*e_size, thread_count);

            // The number of items to create and draw.
            let mut items_to_process = *e_size;

            // Nothing to process.
            if *e_size == 0 { continue; }

            // Create the initial params.
            self.arrow_aabb_params.iterator_start_index = 0;
            self.arrow_aabb_params.iterator_end_index = std::cmp::min(*e_size, safe_number_of_dispatches * v_per_dispatch);
            self.arrow_aabb_params.element_type = *e_type;

            queue.write_buffer(
                &self.buffers.get(&"arrow_aabb_params".to_string()).unwrap(),
                0,
                bytemuck::cast_slice(&[self.arrow_aabb_params])
            );

            // Continue process until all element are rendered.
            while items_to_process > 0 {

                // The number of remaining dispatches to complete the triangle mesh creation and
                // rendering.
                let total_number_of_dispatches = udiv_up_safe32(items_to_process, thread_count);

                // Calculate the number of dispatches for this run. 
                let local_dispatch = std::cmp::min(total_number_of_dispatches, safe_number_of_dispatches);

                // Then number of elements that are going to be rendered. 
                let number_of_elements = std::cmp::min(local_dispatch * thread_count, items_to_process);

                let mut encoder_arrow_aabb = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("arrow_aabb ... ") });

                self.arrow_aabb_params.iterator_end_index = self.arrow_aabb_params.iterator_start_index + std::cmp::min(number_of_elements, safe_number_of_dispatches * v_per_dispatch);

                // println!("self.arrow_aabb_params.iterator_start_index == {}", self.arrow_aabb_params.iterator_start_index);
                // println!("self.arrow_aabb_params.iterator_end_index == {}", self.arrow_aabb_params.iterator_end_index);

                queue.write_buffer(
                    &self.buffers.get(&"arrow_aabb_params".to_string()).unwrap(),
                    0,
                    bytemuck::cast_slice(&[self.arrow_aabb_params])
                );

                self.compute_object_arrow.dispatch(
                    &self.compute_bind_groups_arrow,
                    &mut encoder_arrow_aabb,
                    local_dispatch, 1, 1, Some("arrow local dispatch")
                );

                queue.submit(Some(encoder_arrow_aabb.finish()));

                let counter = self.histogram_draw_counts.get_values(device, queue);

                // self?
                // let draw_count = counter[0] * 3;
                let draw_count = number_of_elements * vertices_per_elem;

                println!("local_dispatch == {}", local_dispatch);
                println!("draw_count == {}", draw_count);

                let mut encoder_arrow_rendering = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("arrow rendering ... ") });

                draw(&mut encoder_arrow_rendering,
                     &view,
                     self.screen.depth_texture.as_ref().unwrap(),
                     &self.render_bind_groups_vvvvnnnn,
                     &self.render_object_vvvvnnnn.pipeline,
                     &self.buffers.get("output_render").unwrap(),
                     0..draw_count,
                     clear
                );
               
                if clear { clear = false; }

                // Decrease the total count of elements.
                items_to_process = items_to_process - number_of_elements; 

                queue.submit(Some(encoder_arrow_rendering.finish()));
                self.histogram_draw_counts.reset_all_cpu_version(queue, 0);

                self.arrow_aabb_params.iterator_start_index = self.arrow_aabb_params.iterator_end_index; // + items_to_process;
                //++ queue.write_buffer(
                //++     &self.buffers.get(&"arrow_aabb_params".to_string()).unwrap(),
                //++     0,
                //++     bytemuck::cast_slice(&[self.arrow_aabb_params])
                //++ );
            } // for number_of_loop
        }

        //// DRAW CHARS

        // Update Visualization params for arrows.
          
        self.arrow_aabb_params.iterator_start_index = 0;
        self.arrow_aabb_params.iterator_end_index = number_of_chars;
        self.arrow_aabb_params.element_type = 0;
        // TODO: a better heurestic. This doesn't work as expected.
        self.arrow_aabb_params.max_number_of_vertices = std::cmp::max(MAX_NUMBER_OF_VVVC as i32 - 500000, 0) as u32;
        //self.arrow_aabb_params.max_number_of_vertices = MAX_NUMBER_OF_VVVC as u32;

        queue.write_buffer(
            &self.buffers.get(&"arrow_aabb_params".to_string()).unwrap(),
            0,
            bytemuck::cast_slice(&[self.arrow_aabb_params])
        );

        self.histogram_draw_counts.reset_all_cpu_version(queue, 0);

        let mut current_char_index = 0;

        let mut ccc = -1;
        while true { 
            ccc = ccc + 1;

            let mut encoder_char = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("numbers encoder") });

            self.compute_object_char.dispatch(
                &self.compute_bind_groups_char,
                &mut encoder_char,
                // 1, 1, 1, Some("numbers dispatch")
                number_of_chars, 1, 1, Some("numbers dispatch")
            );

            queue.submit(Some(encoder_char.finish()));

            let counter = self.histogram_draw_counts.get_values(device, queue);
            self.draw_count_triangles = counter[0];

            current_char_index = counter[1];

            let mut encoder_char_rendering = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("numbers encoder") });

            draw(&mut encoder_char_rendering,
                 &view,
                 self.screen.depth_texture.as_ref().unwrap(),
                 &self.render_bind_groups_vvvc,
                 &self.render_object_vvvc.pipeline,
                 &self.buffers.get("output_render").unwrap(),
                 0..self.draw_count_triangles.min(MAX_NUMBER_OF_VVVC as u32),
                 clear
            );
            queue.submit(Some(encoder_char_rendering.finish()));

            self.histogram_draw_counts.reset_all_cpu_version(queue, 0);

            // There are still chars left.
            if current_char_index != 0 && current_char_index != self.arrow_aabb_params.iterator_end_index - 1 {
                self.arrow_aabb_params.iterator_start_index = current_char_index;
                self.arrow_aabb_params.iterator_end_index = number_of_chars;
                self.arrow_aabb_params.element_type = 0;
                self.arrow_aabb_params.max_number_of_vertices = std::cmp::max(MAX_NUMBER_OF_VVVC as i32 - 500000, 0) as u32; //MAX_NUMBER_OF_VVVC as u32;

                queue.write_buffer(
                    &self.buffers.get(&"arrow_aabb_params".to_string()).unwrap(),
                    0,
                    bytemuck::cast_slice(&[self.arrow_aabb_params])
                );

            }
            else { break; }
        }

        let mut model_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("model encoder") });

        draw(&mut model_encoder,
             &view,
             self.screen.depth_texture.as_ref().unwrap(),
             &self.render_bind_groups_vvvvnnnn,
             &self.render_object_vvvvnnnn.pipeline,
             &self.buffers.get("triangle_mesh").unwrap(),
             0..3,
             //0..self.triangle_mesh_draw_count * 3,
             clear
        );
        queue.submit(Some(model_encoder.finish()));

        // Update screen.
        self.screen.prepare_for_rendering();

        // Reset counter.
        self.histogram_fmm.reset_all_cpu_version(queue, 0);
    }

    #[allow(unused)]
    fn input(&mut self, queue: &wgpu::Queue, input_cache: &InputCache) {
    }

    fn resize(&mut self, device: &wgpu::Device, sc_desc: &wgpu::SurfaceConfiguration, _new_size: winit::dpi::PhysicalSize<u32>) {
        self.screen.depth_texture = Some(Texture::create_depth_texture(&device, &sc_desc, Some("depth-texture")));
        self.camera.resize(sc_desc.width as f32, sc_desc.height as f32);
    }

    #[allow(unused)]
    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, input: &InputCache) {
        self.camera.update_from_input(&queue, &input);

        // if self.keys.test_key(&Key::L, input) { 
        //     self.arrow_aabb_params.arrow_size = self.arrow_aabb_params.arrow_size + 0.005;  
        //     self.temp_arrow_aabb_params.arrow_size = self.arrow_aabb_params.arrow_size + 0.005;  
        // }
        // if self.keys.test_key(&Key::K, input) { 
        //     self.arrow_aabb_params.arrow_size = (self.arrow_aabb_params.arrow_size - 0.005).max(0.01);  
        //     self.temp_arrow_aabb_params.arrow_size = (self.temp_arrow_aabb_params.arrow_size - 0.005).max(0.01);  
        // }
        // if self.keys.test_key(&Key::Key1, input) { 
        //     self.arrow_aabb_params.max_local_vertex_capacity = 1;  
        //     self.temp_arrow_aabb_params.max_local_vertex_capacity = 1;  
        // }
        // if self.keys.test_key(&Key::Key2, input) { 
        //     self.arrow_aabb_params.max_local_vertex_capacity = 2;
        //     self.temp_arrow_aabb_params.max_local_vertex_capacity = 2;  
        // }
        // if self.keys.test_key(&Key::Key3, input) { 
        //     self.arrow_aabb_params.max_local_vertex_capacity = 3;  
        //     self.temp_arrow_aabb_params.max_local_vertex_capacity = 3;  
        // }
        // if self.keys.test_key(&Key::Key4, input) { 
        //     self.arrow_aabb_params.max_local_vertex_capacity = 4;  
        //     self.temp_arrow_aabb_params.max_local_vertex_capacity = 4;  
        // }
        // if self.keys.test_key(&Key::Key9, input) { 
        //     self.block64mode = true;  
        // }
        // if self.keys.test_key(&Key::Key0, input) { 
        //     self.block64mode = false;  
        // }
        // if self.keys.test_key(&Key::NumpadSubtract, input) { 
        // //if self.keys.test_key(&Key::T, input) { 
        //     let si = self.temp_arrow_aabb_params.iterator_start_index as i32;
        //     if si >= THREAD_COUNT as i32 {
        //         self.temp_arrow_aabb_params.iterator_start_index = self.temp_arrow_aabb_params.iterator_start_index - THREAD_COUNT;
        //         self.temp_arrow_aabb_params.iterator_end_index = self.temp_arrow_aabb_params.iterator_end_index - THREAD_COUNT;
        //     }
        // }
        // if self.keys.test_key(&Key::NumpadAdd, input) { 
        //     let ei = self.temp_arrow_aabb_params.iterator_end_index;
        //     if ei <= 4096 - THREAD_COUNT {
        //         self.temp_arrow_aabb_params.iterator_start_index = self.temp_arrow_aabb_params.iterator_start_index + THREAD_COUNT;
        //         self.temp_arrow_aabb_params.iterator_end_index = self.temp_arrow_aabb_params.iterator_end_index + THREAD_COUNT;
        //     }
        // }

        // if self.block64mode {
        //     queue.write_buffer(
        //         &self.buffers.get(&"arrow_aabb_params".to_string()).unwrap(),
        //         0,
        //         bytemuck::cast_slice(&[self.temp_arrow_aabb_params])
        //     );
        // }
        // else {
        //     queue.write_buffer(
        //         &self.buffers.get(&"arrow_aabb_params".to_string()).unwrap(),
        //         0,
        //         bytemuck::cast_slice(&[self.arrow_aabb_params])
        //     );
        // }
    }
}

fn main() {
    
    jaankaup_core::template::run_loop::<Fmm, BasicLoop, FmmFeatures>(); 
    println!("Finished...");
}
