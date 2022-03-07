use std::convert::TryInto;
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
use jaankaup_core::gpu_debugger::GpuDebugger;
use jaankaup_core::texture::Texture;
use jaankaup_core::misc::Convert2Vec;
use jaankaup_core::histogram::Histogram;
use jaankaup_core::impl_convert;
use jaankaup_core::common_functions::{
    encode_rgba_u32,
    udiv_up_safe32,
    create_uniform_bindgroup_layout,
    create_buffer_bindgroup_layout
};
// use jaankaup_algorithms::histogram::Histogram;
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

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FmmBlock {
    index: u32,
    band_points_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FmmPrefixParams {
    blaah: u32,
}

impl_convert!{Arrow}
impl_convert!{Char}
impl_convert!{FmmCell}
impl_convert!{FmmParams}
impl_convert!{FmmBlock}
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
    pub gpu_debugger: GpuDebugger,
    pub render_object_vvvvnnnn: RenderObject, 
    pub render_bind_groups_vvvvnnnn: Vec<wgpu::BindGroup>,
    pub render_object_vvvc: RenderObject,
    pub render_bind_groups_vvvc: Vec<wgpu::BindGroup>,
    //++ pub compute_object_char: ComputeObject, 
    //++ pub compute_bind_groups_char: Vec<wgpu::BindGroup>,
    //++ pub compute_object_arrow: ComputeObject, 
    //++ pub compute_bind_groups_arrow: Vec<wgpu::BindGroup>,
    pub compute_object_fmm: ComputeObject, 
    pub compute_bind_groups_fmm: Vec<wgpu::BindGroup>,
    pub compute_object_fmm_triangle: ComputeObject, 
    pub compute_bind_groups_fmm_triangle: Vec<wgpu::BindGroup>,
    pub compute_object_fmm_prefix_scan: ComputeObject,
    pub compute_bind_groups_fmm_prefix_scan: Vec<wgpu::BindGroup>,
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
    pub draw_triangle_mesh: bool,
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

        // Camera.
        let mut camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (0.0, 0.0, 10.0), -89.0, 0.0);
        camera.set_rotation_sensitivity(0.4);
        camera.set_movement_sensitivity(0.02);

        // gpu debugger.
        let gpu_debugger = GpuDebugger::Init(
                &configuration.device,
                &configuration.sc_desc,
                &camera.get_camera_uniform(&configuration.device),
                MAX_NUMBER_OF_VVVVNNNN.try_into().unwrap(),
                MAX_NUMBERS_OF_CHARS.try_into().unwrap(),
                MAX_NUMBERS_OF_ARROWS.try_into().unwrap(),
                MAX_NUMBERS_OF_AABBS.try_into().unwrap(),
                MAX_NUMBERS_OF_AABB_WIRES.try_into().unwrap(),
                64,
        );

        let mut keys = KeyboardManager::init();
        keys.register_key(Key::P, 200.0);

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
            None)
        );

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
            "fmm_prefix_params".to_string(),
            buffer_from_data::<FmmPrefixParams>(
            &configuration.device,
            &vec![FmmPrefixParams {
                blaah: 123,
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
            )
        );

        buffers.insert(
            "temp_prefix_sum".to_string(),
            configuration.device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("termp_prefix_sum buffer"),
                size: (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
            )
        );

        let mut fmm_blocks: Vec<FmmBlock> = Vec::with_capacity(FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z);

        // Create FMM blocks.
        for i in 0..FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z {
           fmm_blocks.push(FmmBlock{index: i as u32, band_points_count: 0, });
        }

        buffers.insert(
            "fmm_blocks".to_string(),
            buffer_from_data::<FmmBlock>(
                &configuration.device,
                &fmm_blocks,
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("fmm blocks buffer")
            )
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
                            // @group(0) @binding(0) var<uniform> fmm_params: FmmParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<storage, read_write> fmm_data: array<FmmCell>;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(2) var<storage, read_write> counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(3) var<storage,read_write> output_char: array<Char>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(4) var<storage,read_write> output_arrow: array<Arrow>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(5) var<storage,read_write> output_aabb: array<AABB>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(6) var<storage,read_write> output_aabb_wire: array<AABB>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE),
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
                                               &gpu_debugger.get_element_counter_buffer().as_entire_binding(),
                                               &gpu_debugger.get_output_chars_buffer().as_entire_binding(),
                                               &gpu_debugger.get_output_arrows_buffer().as_entire_binding(),
                                               &gpu_debugger.get_output_aabbs_buffer().as_entire_binding(),
                                               &gpu_debugger.get_output_aabb_wires_buffer().as_entire_binding(),
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
                            // @group(0) @binding(0) var<uniform> fmm_params: FmmParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<storage, read_write> fmm_data: array<FmmCell>;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(2) var<storage, read_write> triangle_mesh_in: array<Triangle>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(3) var<storage, read_write> counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(4) var<storage,read_write> output_char: array<Char>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(5) var<storage,read_write> output_arrow: array<Arrow>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(6) var<storage,read_write> output_aabb: array<AABB>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(7) var<storage,read_write> output_aabb_wire: array<AABB>;
                            create_buffer_bindgroup_layout(7, wgpu::ShaderStages::COMPUTE),
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
                         &gpu_debugger.get_element_counter_buffer().as_entire_binding(),
                         &gpu_debugger.get_output_chars_buffer().as_entire_binding(),
                         &gpu_debugger.get_output_arrows_buffer().as_entire_binding(),
                         &gpu_debugger.get_output_aabbs_buffer().as_entire_binding(),
                         &gpu_debugger.get_output_aabb_wires_buffer().as_entire_binding(),
                    ]
                ]
        );

        ////////////////////////////////////////////////////
        ////               Prefix scan FmmBlock         ////
        ////////////////////////////////////////////////////

        let compute_object_fmm_prefix_scan =
                ComputeObject::init(
                    &configuration.device,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("prefix_scan.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/prefix_scan.wgsl"))),
                    
                    }),
                    Some("FMM prefix scan compute object"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> fmm_params: PrefixParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<storage, read_write> fmm_blocks: array<FmmBlock>;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(2) var<storage, read_write> temp_prefix_sum: array<u32>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE),
                        ],
                    ]
        );

        let compute_bind_groups_fmm_prefix_scan =
            create_bind_groups(
                &configuration.device,
                &compute_object_fmm_prefix_scan.bind_group_layout_entries,
                &compute_object_fmm_prefix_scan.bind_group_layouts,
                &vec![
                    vec![
                         &buffers.get(&"fmm_prefix_params".to_string()).unwrap().as_entire_binding(),
                         &buffers.get(&"fmm_blocks".to_string()).unwrap().as_entire_binding(),
                         &buffers.get(&"temp_prefix_sum".to_string()).unwrap().as_entire_binding(),
                    ]
                ]
        );

        println!("Creating render bind groups.");
 
        Self {
            screen: ScreenTexture::init(&configuration.device, &configuration.sc_desc, true),
            gpu_debugger: gpu_debugger,
            render_object_vvvvnnnn: render_object_vvvvnnnn,
            render_bind_groups_vvvvnnnn: render_bind_groups_vvvvnnnn,
            render_object_vvvc: render_object_vvvc,
            render_bind_groups_vvvc: render_bind_groups_vvvc,
            compute_object_fmm: compute_object_fmm, 
            compute_bind_groups_fmm: compute_bind_groups_fmm,
            compute_object_fmm_triangle: compute_object_fmm_triangle, 
            compute_bind_groups_fmm_triangle: compute_bind_groups_fmm_triangle,
            compute_object_fmm_prefix_scan: compute_object_fmm_prefix_scan,
            compute_bind_groups_fmm_prefix_scan: compute_bind_groups_fmm_prefix_scan,
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
            draw_triangle_mesh: true,
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

        //++ let thread_count = 64;

        let mut clear = true;

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
                                      //triangle_count: 1 }
                                      triangle_count: 2036 }
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
            2036, 1, 1,
            //udiv_up_safe32(1, thread_count), 1, 1,
            //udiv_up_safe32(2036, thread_count), 1, 1,
            // (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z) as u32, 1, 1,
            Some("fmm triangle dispatch")
        );

        queue.submit(Some(encoder_command.finish()));

       let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Fmm triangle encoder") });

       queue.write_buffer(
           &self.buffers.get(&"fmm_params".to_string()).unwrap(),
           0,
           bytemuck::cast_slice(&[FmmParams {
                                     fmm_global_dimension: [16, 16, 16],
                                     visualize: 1,
                                     fmm_inner_dimension: [4, 4, 4],
                                     triangle_count: 2036 }
           ]));


       // Visualize.
       self.compute_object_fmm_triangle.dispatch(
           &self.compute_bind_groups_fmm_triangle,
           &mut encoder_command,
           (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z) as u32, 1, 1,
           Some("fmm triangle dispatch")
       );

       queue.submit(Some(encoder_command.finish()));

       self.gpu_debugger.render(
                  &device,
                  &queue,
                  &view,
                  self.screen.depth_texture.as_ref().unwrap(),
                  &mut clear
        );
        self.gpu_debugger.reset_element_counters(&queue);

        if self.draw_triangle_mesh {

            let mut model_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("model encoder") });

            draw(&mut model_encoder,
                 &view,
                 self.screen.depth_texture.as_ref().unwrap(),
                 &self.render_bind_groups_vvvvnnnn,
                 &self.render_object_vvvvnnnn.pipeline,
                 &self.buffers.get("triangle_mesh").unwrap(),
                 0..2036 * 3,
                 //0..self.triangle_mesh_draw_count * 3,
                 clear
            );
            queue.submit(Some(model_encoder.finish()));
        }

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

        if self.keys.test_key(&Key::P, input) { 
            println!("Space pressed.");
            self.draw_triangle_mesh = !self.draw_triangle_mesh;
        }

        // Prefix sum.
        let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Fmm prefix scan command encoder") });

        // queue.write_buffer(
        //     &self.buffers.get(&"fmm_params".to_string()).unwrap(),
        //     0,
        //     bytemuck::cast_slice(&[FmmParams {
        //                               fmm_global_dimension: [16, 16, 16],
        //                               visualize: 0,
        //                               fmm_inner_dimension: [4, 4, 4],
        //                               //triangle_count: 1 }
        //                               triangle_count: 2036 }
        //     ]));

        // Compute interface.
        self.compute_object_fmm_prefix_scan.dispatch(
            &self.compute_bind_groups_fmm_prefix_scan,
            &mut encoder_command,
            1, 1, 1,
            //udiv_up_safe32(1, thread_count), 1, 1,
            //udiv_up_safe32(2036, thread_count), 1, 1,
            // (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z) as u32, 1, 1,
            Some("fmm prefix scan dispatch")
        );

        queue.submit(Some(encoder_command.finish()));
    }
}

fn main() {
    
    jaankaup_core::template::run_loop::<Fmm, BasicLoop, FmmFeatures>(); 
    println!("Finished...");
}
