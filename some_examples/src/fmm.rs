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
use jaankaup_core::camera::Camera;
use jaankaup_core::buffer::{buffer_from_data};
use jaankaup_core::wgpu;
use jaankaup_core::winit;
use jaankaup_core::log;
use jaankaup_core::screen::ScreenTexture;
use jaankaup_core::texture::Texture;
use jaankaup_core::misc::Convert2Vec;
use jaankaup_core::impl_convert;
use jaankaup_core::common_functions::encode_rgba_u32;
use jaankaup_algorithms::histogram::Histogram;
use bytemuck::{Pod, Zeroable};

use winit::event as ev;
pub use ev::VirtualKeyCode as Key;

const MAX_NUMBERS_OF_ARROWS: usize = 4096; 
const MAX_NUMBERS_OF_CHARS:  usize = 4096; 

// FMM dimensions.
const FMM_X: usize = 16; 
const FMM_Y: usize = 16; 
const FMM_Z: usize = 16; 

/// The number of vertices per chunk.
const MAX_VERTEX_CAPACITY: usize = 128 * 64 * 64; 

/// The size of draw buffer;
const VERTEX_BUFFER_SIZE: usize = 16 * MAX_VERTEX_CAPACITY * size_of::<f32>(); // VVVC

const THREAD_COUNT: u32 = 64;

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
struct VisualizationParams{
    max_number_of_vertices: u32,
    iterator_start_index: u32,
    iterator_end_index: u32,
    arrow_size: f32,
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
    blah: f32,
}

impl_convert!{Arrow}
impl_convert!{Char}
impl_convert!{FmmCell}
impl_convert!{FmmParams}

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
    pub compute_object: ComputeObject, 
    pub compute_bind_groups: Vec<wgpu::BindGroup>,
    pub compute_object_arrow: ComputeObject, 
    pub compute_bind_groups_arrow: Vec<wgpu::BindGroup>,
    // pub _textures: HashMap<String, Texture>,
    pub buffers: HashMap<String, wgpu::Buffer>,
    pub camera: Camera,
    pub histogram: Histogram,
    pub draw_count_points: u32,
    pub draw_count_triangles: u32,
    pub visualization_params: VisualizationParams,
    pub temp_visualization_params: VisualizationParams,
    pub keys: KeyboardManager,
    pub block64mode: bool,
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
        let mut camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (0.0, 0.0, 40.0), -90.0, 0.0);
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

        let histogram = Histogram::init(&configuration.device, &vec![0; 2]);

        let params = VisualizationParams {
            max_number_of_vertices: 1,
            iterator_start_index: 0,
            iterator_end_index: 4096,
            //iterator_end_index: 32768,
            arrow_size: 0.3,
        };

        let temp_params = VisualizationParams {
            max_number_of_vertices: 1,
            iterator_start_index: 0,
            iterator_end_index: THREAD_COUNT,
            arrow_size: 0.3,
        };

        buffers.insert(
            "fmm_params".to_string(),
            buffer_from_data::<FmmParams>(
            &configuration.device,
            &vec![FmmParams { blah: 234.0, }],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            None)
        );

        buffers.insert(
            "fmm_cells".to_string(),
            configuration.device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("output_arrays buffer"),
                size: (FMM_X * FMM_Y * FMM_Z * std::mem::size_of::<FmmCell>()) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
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
        // buffers.insert(
        //     "debug_arrays".to_string(),
        //     buffer_from_data::<Arrow>(
        //         &configuration.device,
        //         &vec![Arrow { start_pos: [0.0, 0.0, 0.0, 1.0],
        //                       end_pos:   [4.0, 3.0, 0.0, 1.0],
        //                       color: encode_rgba_u32(0, 0, 255, 255),
        //                       size: 0.2,
        //                       _padding: [0, 0]}],
        //         wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        //         Some("Debug array buffer")
        //     )
        // );

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
            "visualization_params".to_string(),
            buffer_from_data::<VisualizationParams>(
            &configuration.device,
            &vec![params],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            None)
        );


        let compute_object =
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
                            // var<uniform> visualization_params: VisualizationParams;
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

        let compute_bind_groups = create_bind_groups(
                                      &configuration.device,
                                      &compute_object.bind_group_layout_entries,
                                      &compute_object.bind_group_layouts,
                                      &vec![
                                          vec![
                                               &camera.get_camera_uniform(&configuration.device).as_entire_binding(),
                                               &buffers.get(&"visualization_params".to_string()).unwrap().as_entire_binding(),
                                               &histogram.get_histogram_buffer().as_entire_binding(),
                                               &buffers.get(&"output_chars".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output_render".to_string()).unwrap().as_entire_binding()
                                          ]
                                      ]
        );

        let compute_object_arrow =
                ComputeObject::init(
                    &configuration.device,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("compute_visuzlizer_wgsl array"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/visualizer.wgsl"))),
                    
                    }),
                    Some("Visualizer Compute object"),
                    &vec![
                        vec![
                            // @group(0)
                            // @binding(0)
                            // var<uniform> visualization_params: VisualizationParams;
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
                            // var<storage,read_write> output: array<VertexBuffer>;
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
                        ],
                    ]
        );

        let compute_bind_groups_arrow = create_bind_groups(
                                      &configuration.device,
                                      &compute_object_arrow.bind_group_layout_entries,
                                      &compute_object_arrow.bind_group_layouts,
                                      &vec![
                                          vec![
                                               &buffers.get(&"visualization_params".to_string()).unwrap().as_entire_binding(),
                                               &histogram.get_histogram_buffer().as_entire_binding(),
                                               &buffers.get(&"output_arrows".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output_render".to_string()).unwrap().as_entire_binding()
                                          ]
                                      ]
        );

        let compute_object_fmm =
                ComputeObject::init(
                    &configuration.device,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("compute_visuzlizer_wgsl array"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/visualizer.wgsl"))),
                    
                    }),
                    Some("Visualizer Compute object"),
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
                        ],
                    ]
        );

        let compute_bind_groups_fmm = create_bind_groups(
                                      &configuration.device,
                                      &compute_object_arrow.bind_group_layout_entries,
                                      &compute_object_arrow.bind_group_layouts,
                                      &vec![
                                          vec![
                                               &buffers.get(&"fmm_params".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"fmm_cells".to_string()).unwrap().as_entire_binding(),
                                               &histogram.get_histogram_buffer().as_entire_binding(), // create a histogram for fmm.
                                               &buffers.get(&"output_arrows".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output_chars".to_string()).unwrap().as_entire_binding()
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
            compute_object: compute_object,
            compute_bind_groups: compute_bind_groups,
            compute_object_arrow: compute_object_arrow, 
            compute_bind_groups_arrow: compute_bind_groups_arrow,
            // _textures: textures,
            buffers: buffers,
            camera: camera,
            histogram: histogram,
            draw_count_points: 0,
            draw_count_triangles: 0,
            visualization_params: params,
            temp_visualization_params: temp_params,
            keys: keys,
            block64mode: false,
        }
    }

    fn render(&mut self,
              device: &wgpu::Device,
              queue: &mut wgpu::Queue,
              surface: &wgpu::Surface,
              sc_desc: &wgpu::SurfaceConfiguration) {

        self.screen.acquire_screen_texture(
            &device,
            &sc_desc,
            &surface
        );

        let view = self.screen.surface_texture.as_ref().unwrap().texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut items_available: i32 = 1; 
        let dispatch_x = 1;
        let mut clear = true;
        let mut i = dispatch_x * 64;

        while items_available > 0 {

            let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Visualiztion (AABB)") });

            // Take 128 * 64 = 8192 items.
            self.compute_object.dispatch(
                &self.compute_bind_groups,
                &mut encoder_command,
                dispatch_x, 1, 1, Some("font visualizer dispatch")
            );


            // Submit compute.
            queue.submit(Some(encoder_command.finish()));

            let counter = self.histogram.get_values(device, queue);
            self.draw_count_points = counter[0];
            
            let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Visualiztion (AABB)") });

            self.compute_object_arrow.dispatch(
                &self.compute_bind_groups_arrow,
                &mut encoder_command,
                dispatch_x, 1, 1, Some("font visualizer dispatch")
            );

            queue.submit(Some(encoder_command.finish()));

            let counter2 = self.histogram.get_values(device, queue);
            self.draw_count_triangles = counter2[0];

            //println!("self.draw_count  == {}", self.draw_count); 

            let mut encoder_render = device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
            });

            items_available = items_available - dispatch_x as i32 * 64; 

            // Draw the cube.
            draw(&mut encoder_render,
                 &view,
                 self.screen.depth_texture.as_ref().unwrap(),
                 &self.render_bind_groups_vvvc,
                 &self.render_object_vvvc.pipeline,
                 &self.buffers.get("output_render").unwrap(),
                 0..self.draw_count_points, // TODO: Cube 
                 clear
            );

            clear = false;

            draw(&mut encoder_render,
                 &view,
                 self.screen.depth_texture.as_ref().unwrap(),
                 &self.render_bind_groups_vvvvnnnn,
                 &self.render_object_vvvvnnnn.pipeline,
                 &self.buffers.get("output_render").unwrap(),
                 self.draw_count_points..self.draw_count_triangles, // TODO: Cube 
                 clear
            );
            queue.submit(Some(encoder_render.finish()));
            clear = items_available <= 0;
            self.histogram.set_values_cpu_version(queue, &vec![0, i]);
            i = i + dispatch_x*64;
        }

        self.screen.prepare_for_rendering();

        // Reset counter.
        self.histogram.reset_all_cpu_version(queue, 0); // TODO: fix histogram.reset_cpu_version        
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
        //     self.visualization_params.arrow_size = self.visualization_params.arrow_size + 0.005;  
        //     self.temp_visualization_params.arrow_size = self.visualization_params.arrow_size + 0.005;  
        // }
        // if self.keys.test_key(&Key::K, input) { 
        //     self.visualization_params.arrow_size = (self.visualization_params.arrow_size - 0.005).max(0.01);  
        //     self.temp_visualization_params.arrow_size = (self.temp_visualization_params.arrow_size - 0.005).max(0.01);  
        // }
        // if self.keys.test_key(&Key::Key1, input) { 
        //     self.visualization_params.max_local_vertex_capacity = 1;  
        //     self.temp_visualization_params.max_local_vertex_capacity = 1;  
        // }
        // if self.keys.test_key(&Key::Key2, input) { 
        //     self.visualization_params.max_local_vertex_capacity = 2;
        //     self.temp_visualization_params.max_local_vertex_capacity = 2;  
        // }
        // if self.keys.test_key(&Key::Key3, input) { 
        //     self.visualization_params.max_local_vertex_capacity = 3;  
        //     self.temp_visualization_params.max_local_vertex_capacity = 3;  
        // }
        // if self.keys.test_key(&Key::Key4, input) { 
        //     self.visualization_params.max_local_vertex_capacity = 4;  
        //     self.temp_visualization_params.max_local_vertex_capacity = 4;  
        // }
        // if self.keys.test_key(&Key::Key9, input) { 
        //     self.block64mode = true;  
        // }
        // if self.keys.test_key(&Key::Key0, input) { 
        //     self.block64mode = false;  
        // }
        // if self.keys.test_key(&Key::NumpadSubtract, input) { 
        // //if self.keys.test_key(&Key::T, input) { 
        //     let si = self.temp_visualization_params.iterator_start_index as i32;
        //     if si >= THREAD_COUNT as i32 {
        //         self.temp_visualization_params.iterator_start_index = self.temp_visualization_params.iterator_start_index - THREAD_COUNT;
        //         self.temp_visualization_params.iterator_end_index = self.temp_visualization_params.iterator_end_index - THREAD_COUNT;
        //     }
        // }
        // if self.keys.test_key(&Key::NumpadAdd, input) { 
        //     let ei = self.temp_visualization_params.iterator_end_index;
        //     if ei <= 4096 - THREAD_COUNT {
        //         self.temp_visualization_params.iterator_start_index = self.temp_visualization_params.iterator_start_index + THREAD_COUNT;
        //         self.temp_visualization_params.iterator_end_index = self.temp_visualization_params.iterator_end_index + THREAD_COUNT;
        //     }
        // }

        // if self.block64mode {
        //     queue.write_buffer(
        //         &self.buffers.get(&"visualization_params".to_string()).unwrap(),
        //         0,
        //         bytemuck::cast_slice(&[self.temp_visualization_params])
        //     );
        // }
        // else {
        //     queue.write_buffer(
        //         &self.buffers.get(&"visualization_params".to_string()).unwrap(),
        //         0,
        //         bytemuck::cast_slice(&[self.visualization_params])
        //     );
        // }
    }
}

fn main() {
    
    jaankaup_core::template::run_loop::<Fmm, BasicLoop, FmmFeatures>(); 
    println!("Finished...");
}
