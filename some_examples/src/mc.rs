use std::mem::size_of;
use std::collections::HashMap;
use std::borrow::Cow;
use jaankaup_core::template::{
        WGPUFeatures,
        WGPUConfiguration,
        Application,
        BasicLoop,
};
use jaankaup_core::render_object::{RenderObject, ComputeObject, create_bind_groups, draw, draw_indirect, DrawIndirect};
use jaankaup_core::input::*;
use jaankaup_core::camera::Camera;
use jaankaup_core::buffer::{buffer_from_data}; //, to_vec};
use jaankaup_core::wgpu;
use jaankaup_core::winit;
use jaankaup_core::log;
use jaankaup_core::screen::ScreenTexture;
use jaankaup_core::texture::Texture;
use jaankaup_core::misc::Convert2Vec;
use jaankaup_core::impl_convert;
use jaankaup_core::common_functions::{
    encode_rgba_u32, 
    create_uniform_bindgroup_layout,
    create_buffer_bindgroup_layout
};
use jaankaup_core::histogram::Histogram;
use jaankaup_algorithms::mc::{McParams, MarchingCubes};
use bytemuck::{Pod, Zeroable};

use winit::event as ev;
pub use ev::VirtualKeyCode as Key;

const GLOBAL_NOISE_X_DIMENSION: u32 = 1; 
const GLOBAL_NOISE_Y_DIMENSION: u32 = 1; 
const GLOBAL_NOISE_Z_DIMENSION: u32 = 1; 
const LOCAL_NOISE_X_DIMENSION:  u32 = 128; 
const LOCAL_NOISE_Y_DIMENSION:  u32 = 128; 
const LOCAL_NOISE_Z_DIMENSION:  u32 = 128; 

const NOISE_BUFFER_SIZE: u32 = GLOBAL_NOISE_X_DIMENSION *
                               GLOBAL_NOISE_Y_DIMENSION *
                               GLOBAL_NOISE_Z_DIMENSION *
                               LOCAL_NOISE_X_DIMENSION *
                               LOCAL_NOISE_Y_DIMENSION *
                               LOCAL_NOISE_Z_DIMENSION *
                               size_of::<f32>() as u32;

const MC_OUTPUT_BUFFER_SIZE: u32 = NOISE_BUFFER_SIZE * 8;

/// The number of vertices per chunk.
const MAX_VERTEX_CAPACITY: usize = 128 * 64 * 64; // 128 * 64 * 36 = 262144 verticex. 

/// The size of draw buffer;
const VERTEX_BUFFER_SIZE: usize = 8 * MAX_VERTEX_CAPACITY * size_of::<f32>();

const THREAD_COUNT: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct NoiseParams {
    global_dim: [u32; 3],
    time: f32,
    local_dim: [u32; 3],
    value: f32,
}

struct McAppFeatures {}

impl WGPUFeatures for McAppFeatures {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::empty()
    }
    fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }
    fn required_limits() -> wgpu::Limits {
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 8;
        // limits.max_compute_workgroup_size_x = 512;
        limits
    }
}

// State for this application.
struct McApp {
    pub screen: ScreenTexture, 
    pub render_object: RenderObject, 
    pub render_bind_groups: Vec<wgpu::BindGroup>,
    pub noise_compute_object: ComputeObject, 
    pub noise_compute_bind_groups: Vec<wgpu::BindGroup>,
    pub textures: HashMap<String, Texture>,
    pub buffers: HashMap<String, wgpu::Buffer>,
    pub camera: Camera,
    //pub histogram: Histogram,
    // pub draw_count: u32,
    // pub visualization_params: VisualizationParams,
    pub keys: KeyboardManager,
    pub marching_cubes: MarchingCubes,
    pub update: bool,
}

impl McApp {

        fn create_textures(configuration: &WGPUConfiguration, textures: &mut HashMap<String, Texture>) {
        log::info!("Creating textures.");
        let grass_texture = Texture::create_from_bytes(
            &configuration.queue,
            &configuration.device,
            &configuration.sc_desc,
            1,
            &include_bytes!("../../assets/textures/slime.png")[..],
            None);
        let rock_texture = Texture::create_from_bytes(
            &configuration.queue,
            &configuration.device,
            &configuration.sc_desc,
            1,
            &include_bytes!("../../assets/textures/slime2.png")[..],
            None);
        let slime_texture = Texture::create_from_bytes(
            &configuration.queue,
            &configuration.device,
            &configuration.sc_desc,
            1,
            &include_bytes!("../../assets/textures/xXqQP0.png")[..],
            //&include_bytes!("../../assets/textures/slime.png")[..],
            None);
        let slime_texture2 = Texture::create_from_bytes(
            &configuration.queue,
            &configuration.device,
            &configuration.sc_desc,
            1,
            //&include_bytes!("../../assets/textures/slime2.png")[..],
            //&include_bytes!("../../assets/textures/xXqQP0.png")[..],
            &include_bytes!("../../assets/textures/luava.png")[..],
            None);
        log::info!("Textures created OK.");

        textures.insert("grass".to_string(), grass_texture);
        textures.insert("rock".to_string(), rock_texture);
        textures.insert("slime".to_string(), slime_texture);
        textures.insert("slime2".to_string(), slime_texture2);
    }
}

impl Application for McApp {

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
        let mut textures: HashMap<String, Texture> = HashMap::new();

        let mut keys = KeyboardManager::init();

        // Camera.
        let mut camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (50.0, 60.0, 150.0), -90.0, 0.0);
        camera.set_rotation_sensitivity(0.4);
        camera.set_movement_sensitivity(0.02);

        let render_object =
                RenderObject::init(
                    &configuration.device,
                    &configuration.sc_desc,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("renderer_v4n4_debug_visualizator_wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            //Cow::Borrowed(include_str!("../../assets/shaders/renderer_v4n4.wgsl"))),
                            Cow::Borrowed(include_str!("../../assets/shaders/renderer_v4n4_wasm.wgsl"))),
                    
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
                        // Group 1
                        vec![wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Texture {
                                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                    multisampled: false,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::FRAGMENT,
                                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                                count: None,
                            }
                        ]
                    ],
                    Some("vvvvnnnn renderer with camera."),
                    false,
                    wgpu::PrimitiveTopology::TriangleList
        );

        McApp::create_textures(&configuration, &mut textures);

        let render_bind_groups = create_bind_groups(
                                     &configuration.device,
                                     &render_object.bind_group_layout_entries,
                                     &render_object.bind_group_layouts,
                                     &vec![
                                         vec![&wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                                 buffer: &camera.get_camera_uniform(&configuration.device),
                                                 offset: 0,
                                                 size: None,
                                         })],
                                        vec![&wgpu::BindingResource::TextureView(&textures.get("slime").unwrap().view),
                                             &wgpu::BindingResource::Sampler(&textures.get("slime").unwrap().sampler),
                                             &wgpu::BindingResource::TextureView(&textures.get("slime2").unwrap().view),
                                             &wgpu::BindingResource::Sampler(&textures.get("slime2").unwrap().sampler)
                                        ]
                                     ]
        );

        println!("Creating compute object.");

        let noise_params = NoiseParams {
            global_dim: [GLOBAL_NOISE_X_DIMENSION, GLOBAL_NOISE_Y_DIMENSION, GLOBAL_NOISE_Z_DIMENSION],
            time: 0.0,
            local_dim: [LOCAL_NOISE_X_DIMENSION, LOCAL_NOISE_Y_DIMENSION, LOCAL_NOISE_Z_DIMENSION],
            value: 0.0,
        };

        buffers.insert(
            "noise_params".to_string(),
            buffer_from_data::<NoiseParams>(
            &configuration.device,
            &vec![noise_params],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            None)
        );

        buffers.insert(
            "noise_output".to_string(),
            configuration.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Noise output buffer"),
                size: NOISE_BUFFER_SIZE as u64, 
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        );

        buffers.insert(
            "mc_output".to_string(),
            configuration.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Mc output buffer"),
                size: MC_OUTPUT_BUFFER_SIZE as u64, 
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        );
        

        ///// The mc struct. /////

        let mc_params = McParams {
                base_position: [0.0, 0.0, 0.0, 1.0],
                isovalue: 0.0,
                cube_length: 1.0,
                future_usage1: 0.0,
                future_usage2: 0.0,
                noise_global_dimension: [GLOBAL_NOISE_X_DIMENSION,
                                         GLOBAL_NOISE_Y_DIMENSION,
                                         GLOBAL_NOISE_Z_DIMENSION,
                                         0
                ],
                noise_local_dimension: [LOCAL_NOISE_X_DIMENSION,
                                         LOCAL_NOISE_Y_DIMENSION,
                                         LOCAL_NOISE_Z_DIMENSION,
                                         0
                ],
        };

        println!("compiling mc shader");
        let mc_shader = &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("mc compute shader"),
                        source: wgpu::ShaderSource::Wgsl(
                            //Cow::Borrowed(include_str!("../../assets/shaders/mc_with_3d_texture.wgsl"))),
                            Cow::Borrowed(include_str!("../../assets/shaders/mc_with_3d_texture_wasm.wgsl"))),
                        }
        );

        let mc_instance = MarchingCubes::init_with_noise_buffer(
            &configuration.device,
            &mc_params,
            &mc_shader,
            &buffers.get(&"noise_output".to_string()).unwrap(),
            &buffers.get(&"mc_output".to_string()).unwrap(),
        );

        println!("creating noise compute object");

        // let histogram = Histogram::init(&configuration.device, &vec![0; 1]);

        let noise_compute_object =
                ComputeObject::init(
                    &configuration.device,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("noise compute object"),
                        source: wgpu::ShaderSource::Wgsl(
                            //Cow::Borrowed(include_str!("../../assets/shaders/noise_to_buffer.wgsl"))),
                            Cow::Borrowed(include_str!("../../assets/shaders/noise_to_buffer_wasm.wgsl"))),
                    
                    }),
                    Some("Noise compute object"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> noise_params: NoiseParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<storage, read_write> counter: Counter;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ]
        );

        let noise_compute_bind_groups = create_bind_groups(
                                           &configuration.device,
                                           &noise_compute_object.bind_group_layout_entries,
                                           &noise_compute_object.bind_group_layouts,
                                           &vec![
                                               vec![
                                                    &buffers.get(&"noise_params".to_string()).unwrap().as_entire_binding(),
                                                    &buffers.get(&"noise_output".to_string()).unwrap().as_entire_binding()
                                               ]
                                           ]
        );

        println!("Done!.");

        Self {
            screen: ScreenTexture::init(&configuration.device, &configuration.sc_desc, true),
            render_object: render_object,
            render_bind_groups: render_bind_groups,
            noise_compute_object: noise_compute_object,
            noise_compute_bind_groups: noise_compute_bind_groups,
            textures: textures,
            buffers: buffers,
            camera: camera,
            // histogram: histogram,
            // draw_count: 0,
            // visualization_params: params,
            keys: keys,
            marching_cubes: mc_instance,
            update: true,
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

        let clear = true;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Visualiztion (AABB)") });

        // draw(&mut encoder,
        //      &view,
        //      self.screen.depth_texture.as_ref().unwrap(),
        //      &self.render_bind_groups,
        //      &self.render_object.pipeline,
        //      &self.buffers.get("mc_output").unwrap(),
        //      0..self.draw_count, 
        //      clear
        // );
        //
        // let result =  to_vec::<DrawIndirect>(
        //     &device,
        //     &queue,
        //     self.marching_cubes.get_draw_indirect_buffer(),
        //     0,
        //     (size_of::<DrawIndirect>()) as wgpu::BufferAddress
        // );

        // println!("{:?}", result);
        draw_indirect(
             &mut encoder,
             &view,
             self.screen.depth_texture.as_ref().unwrap(),
             &self.render_bind_groups,
             &self.render_object.pipeline,
             &self.buffers.get("mc_output").unwrap(),
             self.marching_cubes.get_draw_indirect_buffer(),
             clear
        );
        queue.submit(Some(encoder.finish()));
        // self.histogram.set_values_cpu_version(queue, &vec![0]);

        self.screen.prepare_for_rendering();

        // Reset counter.
        self.marching_cubes.reset_counter_value(device, queue);
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

        // if !self.update { return; }

        let val = (((input.get_time() / 5000000) as f32) * 0.0015).sin() * 5.0;
        let val2 = (((input.get_time() / 5000000) as f32) * 0.0015).cos() * 0.35;

        let noise_params = NoiseParams {
            global_dim: [GLOBAL_NOISE_X_DIMENSION, GLOBAL_NOISE_Y_DIMENSION, GLOBAL_NOISE_Z_DIMENSION],
            time: (input.get_time() / 50000000) as f32,
            local_dim: [LOCAL_NOISE_X_DIMENSION, LOCAL_NOISE_Y_DIMENSION, LOCAL_NOISE_Z_DIMENSION],
            value: val2,
        };

        queue.write_buffer(
            &self.buffers.get(&("noise_params".to_string())).unwrap(),
            0,
            bytemuck::cast_slice(&[noise_params])
        );
    
        self.marching_cubes.update_mc_params(queue, val);    

        let total_grid_count = GLOBAL_NOISE_X_DIMENSION *
                               GLOBAL_NOISE_Y_DIMENSION * 
                               GLOBAL_NOISE_Z_DIMENSION * 
                               LOCAL_NOISE_X_DIMENSION *  
                               LOCAL_NOISE_Y_DIMENSION * 
                               LOCAL_NOISE_Z_DIMENSION; 

        let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Noise & Mc encoder.") });

        self.noise_compute_object.dispatch(
            &self.noise_compute_bind_groups,
            &mut encoder_command,
            total_grid_count / 1024, 1, 1, Some("noise dispatch")
        );

        self.marching_cubes.dispatch(&mut encoder_command, total_grid_count / 256, 1, 1);

        // Submit compute.
        queue.submit(Some(encoder_command.finish()));

        // self.draw_count = self.marching_cubes.get_counter_value(device, queue);

        // self.update = false;

        // let counter = self.histogram.get_values(device, queue);
        // self.draw_count = counter[0] * 3;
    }
}

fn main() {
    
    jaankaup_core::template::run_loop::<McApp, BasicLoop, McAppFeatures>(); 
    println!("Finished...");
}
