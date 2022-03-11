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
use jaankaup_core::buffer::{buffer_from_data,to_vec};
use jaankaup_core::wgpu;
use jaankaup_core::winit;
use jaankaup_core::log;
use jaankaup_core::screen::ScreenTexture;
use jaankaup_core::gpu_debugger::GpuDebugger;
use jaankaup_core::texture::Texture;
use jaankaup_core::misc::Convert2Vec;
use jaankaup_core::histogram::Histogram;
use jaankaup_core::impl_convert;
use jaankaup_core::gpu_timer::GpuTimer;
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
// const VERTEX_BUFFER_SIZE: usize = MAX_NUMBER_OF_VVVVNNNN * size_of::<Vertex>();
//const VERTEX_BUFFER_SIZE: usize = MAX_NUMBER_OF_VVVVNNNN * size_of::<Vertex>() / 4;

const THREAD_COUNT: u32 = 64;
//const PREFIX_THREAD_COUNT: u32 = 256;
const PREFIX_THREAD_COUNT: u32 = 1024;

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
    data_start_index: u32,
    data_end_index: u32,
    exclusive_parts_start_index: u32,
    exclusive_parts_end_index: u32,
    temp_prefix_data_start_index: u32,
    temp_prefix_data_end_index: u32,
    stage: u32,
}

impl_convert!{FmmCell}
impl_convert!{FmmParams}
impl_convert!{FmmBlock}

struct FmmFeatures {}

impl WGPUFeatures for FmmFeatures {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::TIMESTAMP_QUERY
        // wgpu::Features::empty()
    }
    fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }
    fn required_limits() -> wgpu::Limits {
        let mut limits = wgpu::Limits::default();
        limits.max_compute_invocations_per_workgroup = 1024;
        limits.max_compute_workgroup_size_x = 1024;
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
    pub compute_object_fmm: ComputeObject, 
    pub compute_bind_groups_fmm: Vec<wgpu::BindGroup>,
    pub compute_object_fmm_triangle: ComputeObject, 
    pub compute_bind_groups_fmm_triangle: Vec<wgpu::BindGroup>,
    pub compute_object_fmm_prefix_scan: ComputeObject,
    pub compute_bind_groups_fmm_prefix_scan: Vec<wgpu::BindGroup>,
    pub buffers: HashMap<String, wgpu::Buffer>,
    pub camera: Camera,
    pub keys: KeyboardManager,
    pub triangle_mesh_draw_count: u32,
    pub draw_triangle_mesh: bool,
    pub once: bool,
    pub fmm_prefix_params: FmmPrefixParams, 
    pub gpu_timer: Option<GpuTimer>,
}

impl Fmm {

}

//#[allow(unused_variables)]
impl Application for Fmm {

    fn init(configuration: &WGPUConfiguration) -> Self {

        log::info!("Adapter limits are: ");

        let once = true;

        let gpu_timer = GpuTimer::init(
            &configuration.device,
            &configuration.queue,
            8,
            Some("gpu timer")
        );

        let adapter_limits = configuration.adapter.limits(); 

        log::info!("max_compute_workgroup_storage_size: {:?}", adapter_limits.max_compute_workgroup_storage_size);
        log::info!("max_compute_invocations_per_workgroup: {:?}", adapter_limits.max_compute_invocations_per_workgroup);
        log::info!("max_compute_workgroup_size_x: {:?}", adapter_limits.max_compute_workgroup_size_x);
        log::info!("max_compute_workgroup_size_y: {:?}", adapter_limits.max_compute_workgroup_size_y);
        log::info!("max_compute_workgroup_size_z: {:?}", adapter_limits.max_compute_workgroup_size_z);
        log::info!("max_compute_workgroups_per_dimension: {:?}", adapter_limits.max_compute_workgroups_per_dimension);

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

        let temp_prefix_start = 0;
        let temp_prefix_end = (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z) as u32;
        let exclusive_start = temp_prefix_end;
        let exclusive_end   = exclusive_start + (PREFIX_THREAD_COUNT * 2) as u32;

        println!("temp_prefix_start == {}", temp_prefix_start);
        println!("temp_prefix_end == {}", temp_prefix_end);
        println!("exclusive_start == {}", exclusive_start);
        println!("exclusive_end == {}", exclusive_end);

        let fmm_prefix_params = FmmPrefixParams {
                data_start_index: 0,
                data_end_index: (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z) as u32,
                exclusive_parts_start_index: exclusive_start,
                exclusive_parts_end_index: exclusive_end,
                temp_prefix_data_start_index: temp_prefix_start,
                temp_prefix_data_end_index: temp_prefix_end,
                stage: 1,
        };

        buffers.insert(
            "fmm_prefix_params".to_string(),
            buffer_from_data::<FmmPrefixParams>(
            &configuration.device,
            &[fmm_prefix_params],
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
                label: Some("temp_prefix_sum buffer"),
                size: (exclusive_end as usize * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
            )
        );

        let mut fmm_blocks: Vec<FmmBlock> = Vec::with_capacity(FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z);

        let magic_block_numbers = vec![2,3,4, 127, 128, 1023, 1024, 1025, 2055, 2047, 2048, 2049, 2051];

        // Create FMM blocks.
        for i in 0..FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z {
            let contains = magic_block_numbers.iter().any(|x| *x == i);  
            //let fmm_block = FmmBlock{index: i as u32, band_points_count: 0 , };
            let fmm_block = FmmBlock{index: i as u32, band_points_count: if contains { i as u32 } else {0} , };
            fmm_blocks.push(fmm_block);
            println!("{:?}", fmm_block); 
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

        buffers.insert(
            "filtered_blocks".to_string(),
            configuration.device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("filtered_blokcs buffer"),
                size: ((FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z) as usize * std::mem::size_of::<FmmBlock>()) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
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
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(2) var<storage, read_write> counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(3) var<storage,read_write> output_char: array<Char>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(4) var<storage,read_write> output_arrow: array<Arrow>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(5) var<storage,read_write> output_aabb: array<AABB>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> output_aabb_wire: array<AABB>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),
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
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(2) var<storage, read_write> triangle_mesh_in: array<Triangle>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(3) var<storage, read_write> counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(4) var<storage,read_write> output_char: array<Char>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(5) var<storage,read_write> output_arrow: array<Arrow>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(6) var<storage,read_write> output_aabb: array<AABB>;
                            create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(7) var<storage,read_write> output_aabb_wire: array<AABB>;
                            create_buffer_bindgroup_layout(7, wgpu::ShaderStages::COMPUTE, false),
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
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(2) var<storage, read_write> temp_prefix_sum: array<u32>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(3) var<storage,read_write> filtered_blocks: array<FmmBlock>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            //debug // @group(0) @binding(4) var<storage, read_write> counter: array<atomic<u32>>;
                            //debug create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE),

                            //debug // @group(0) @binding(5) var<storage,read_write> output_char: array<Char>;
                            //debug create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE),

                            //debug // @group(0) @binding(6) var<storage,read_write> output_arrow: array<Arrow>;
                            //debug create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE),

                            //debug // @group(0) @binding(7) var<storage,read_write> output_aabb: array<AABB>;
                            //debug create_buffer_bindgroup_layout(6, wgpu::ShaderStages::COMPUTE),

                            //debug // @group(0) @binding(8) var<storage,read_write> output_aabb_wire: array<AABB>;
                            //debug create_buffer_bindgroup_layout(7, wgpu::ShaderStages::COMPUTE),

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
                         &buffers.get(&"filtered_blocks".to_string()).unwrap().as_entire_binding(),
                         //debug &gpu_debugger.get_element_counter_buffer().as_entire_binding(),
                         //debug &gpu_debugger.get_output_chars_buffer().as_entire_binding(),
                         //debug &gpu_debugger.get_output_arrows_buffer().as_entire_binding(),
                         //debug &gpu_debugger.get_output_aabbs_buffer().as_entire_binding(),
                         //debug &gpu_debugger.get_output_aabb_wires_buffer().as_entire_binding(),
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
            keys: keys,
            triangle_mesh_draw_count: triangle_mesh_draw_count, 
            draw_triangle_mesh: false,
            once: once,
            fmm_prefix_params: fmm_prefix_params,
            gpu_timer: gpu_timer,
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

       //++ let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Fmm triangle encoder") });

       //++ queue.write_buffer(
       //++     &self.buffers.get(&"fmm_params".to_string()).unwrap(),
       //++     0,
       //++     bytemuck::cast_slice(&[FmmParams {
       //++                               fmm_global_dimension: [16, 16, 16],
       //++                               visualize: 1,
       //++                               fmm_inner_dimension: [4, 4, 4],
       //++                               triangle_count: 2036 }
       //++     ]));


       //++ // Visualize.
       //++ self.compute_object_fmm_triangle.dispatch(
       //++     &self.compute_bind_groups_fmm_triangle,
       //++     &mut encoder_command,
       //++     (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z) as u32, 1, 1,
       //++     Some("fmm triangle dispatch")
       //++ );

       //++ queue.submit(Some(encoder_command.finish()));

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
            self.draw_triangle_mesh = !self.draw_triangle_mesh;
        }

        self.fmm_prefix_params.stage = 1;

        queue.write_buffer(
            &self.buffers.get(&"fmm_prefix_params".to_string()).unwrap(),
            0,
            bytemuck::cast_slice(&[self.fmm_prefix_params])
        );

        // How many dispatches.
        let number_of_dispathces = (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z) as u32 / (PREFIX_THREAD_COUNT * 2); 
        println!("number_of_dispathces == {}", number_of_dispathces);

        let wgpu_timer_unwrapped = self.gpu_timer.as_mut().unwrap();

        // Prefix sum.
        let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Fmm prefix scan command encoder") });

        wgpu_timer_unwrapped.start(&mut encoder_command);

        // Compute interface.
        self.compute_object_fmm_prefix_scan.dispatch(
            &self.compute_bind_groups_fmm_prefix_scan,
            &mut encoder_command,
            number_of_dispathces, 1, 1,
            Some("fmm prefix scan dispatch")
        );

        wgpu_timer_unwrapped.end(&mut encoder_command);
        // wgpu_timer_unwrapped.start(&mut encoder_command);
        // wgpu_timer_unwrapped.end(&mut encoder_command);
        // wgpu_timer_unwrapped.start(&mut encoder_command);
        // wgpu_timer_unwrapped.end(&mut encoder_command);

        wgpu_timer_unwrapped.resolve_timestamps(&mut encoder_command);

        queue.submit(Some(encoder_command.finish()));

        self.fmm_prefix_params.stage = 2;


        queue.write_buffer(
            &self.buffers.get(&"fmm_prefix_params".to_string()).unwrap(),
            0,
            bytemuck::cast_slice(&[self.fmm_prefix_params])
        );


        // Prefix sum.
        let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Fmm prefix scan command encoder") });

        wgpu_timer_unwrapped.start(&mut encoder_command);

        // Compute interface.
        self.compute_object_fmm_prefix_scan.dispatch(
            &self.compute_bind_groups_fmm_prefix_scan,
            &mut encoder_command,
            1, 1, 1,
            Some("fmm prefix scan dispatch 2")
        );

        wgpu_timer_unwrapped.end(&mut encoder_command);
        wgpu_timer_unwrapped.resolve_timestamps(&mut encoder_command);

        queue.submit(Some(encoder_command.finish()));

        wgpu_timer_unwrapped.create_timestamp_data(&device, &queue);

        wgpu_timer_unwrapped.print_data();

        wgpu_timer_unwrapped.reset();
        // let gpu_timer_result = wgpu_timer_unwrapped.get_data(); 

        // println!("{:?}", gpu_timer_result);
        //++ self.fmm_prefix_params.stage = 3;

        //++ queue.write_buffer(
        //++     &self.buffers.get(&"fmm_prefix_params".to_string()).unwrap(),
        //++     0,
        //++     bytemuck::cast_slice(&[self.fmm_prefix_params])
        //++ );

        //++ let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Fmm prefix scan command encoder") });

        //++ // Compute interface.
        //++ self.compute_object_fmm_prefix_scan.dispatch(
        //++     &self.compute_bind_groups_fmm_prefix_scan,
        //++     &mut encoder_command,
        //++     1, 1, 1,
        //++     Some("fmm prefix scan dispatch 2")
        //++ );

        //++ queue.submit(Some(encoder_command.finish()));

        let result =  to_vec::<FmmBlock>(
            &device,
            &queue,
            &self.buffers.get(&"filtered_blocks".to_string()).unwrap(),
            0,
            (size_of::<FmmBlock>() * 128) as wgpu::BufferAddress
        );

        // let result =  to_vec::<u32>(
        //     &device,
        //     &queue,
        //     &self.buffers.get(&"temp_prefix_sum".to_string()).unwrap(),
        //     0,
        //     (size_of::<u32>() * 4608) as wgpu::BufferAddress
        // );

        if self.once {
            for i in 0..128 {
                println!("{:?} == {:?}", i, result[i]);
            }
            self.once = !self.once;
        }
        //println!("{:?}", result);
    }
}

fn main() {
    
    jaankaup_core::template::run_loop::<Fmm, BasicLoop, FmmFeatures>(); 
    println!("Finished...");
}
