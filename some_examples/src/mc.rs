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

const GLOBAL_NOISE_X_DIMENSION: u32 = 1; 
const GLOBAL_NOISE_Y_DIMENSION: u32 = 1; 
const GLOBAL_NOISE_Z_DIMENSION: u32 = 1; 
const LOCAL_NOISE_X_DIMENSION:  u32 = 256; 
const LOCAL_NOISE_Y_DIMENSION:  u32 = 256; 
const LOCAL_NOISE_Z_DIMENSION:  u32 = 256; 

const NOISE_BUFFER_SIZE: u32 = GLOBAL_NOISE_X_DIMENSION *
                               GLOBAL_NOISE_Y_DIMENSION *
                               GLOBAL_NOISE_Z_DIMENSION *
                               LOCAL_NOISE_X_DIMENSION *
                               LOCAL_NOISE_Y_DIMENSION *
                               LOCAL_NOISE_Z_DIMENSION *
                               size_of::<f32>() as u32;

/// The number of vertices per chunk.
const MAX_VERTEX_CAPACITY: usize = 128 * 64 * 64; // 128 * 64 * 36 = 262144 verticex. 

/// The size of draw buffer;
const VERTEX_BUFFER_SIZE: usize = 8 * MAX_VERTEX_CAPACITY * size_of::<f32>();

const THREAD_COUNT: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct NoiseParams {
    global_dim: [u32; 4],
    local_dim: [u32; 4],
    time: u32,
    _pad: [u32; 3],
}

struct McFeatures {}

impl WGPUFeatures for McFeatures {
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
struct Mc {
    pub screen: ScreenTexture, 
    pub render_object: RenderObject, 
    pub render_bind_groups: Vec<wgpu::BindGroup>,
    pub noise_compute_object: ComputeObject, 
    pub noise_compute_bind_groups: Vec<wgpu::BindGroup>,
    pub textures: HashMap<String, Texture>,
    pub buffers: HashMap<String, wgpu::Buffer>,
    pub camera: Camera,
    pub histogram: Histogram,
    pub draw_count: u32,
    // pub visualization_params: VisualizationParams,
    pub keys: KeyboardManager,
}

impl Mc {

        fn create_textures(configuration: &WGPUConfiguration, textures: &mut HashMap<String, Texture>) {
        log::info!("Creating textures.");
        let grass_texture = Texture::create_from_bytes(
            &configuration.queue,
            &configuration.device,
            &configuration.sc_desc,
            1,
            &include_bytes!("../../assets/textures/grass2.png")[..],
            None);
        let rock_texture = Texture::create_from_bytes(
            &configuration.queue,
            &configuration.device,
            &configuration.sc_desc,
            1,
            &include_bytes!("../../assets/textures/rock.png")[..],
            None);
        let slime_texture = Texture::create_from_bytes(
            &configuration.queue,
            &configuration.device,
            &configuration.sc_desc,
            1,
            &include_bytes!("../../assets/textures/lava.png")[..],
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

impl Application for Mc {


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
        // keys.register_key(Key::L, 20.0);
        // keys.register_key(Key::K, 20.0);
        // keys.register_key(Key::Key1, 10.0);
        // keys.register_key(Key::Key2, 10.0);
        // keys.register_key(Key::Key3, 10.0);
        // keys.register_key(Key::Key4, 10.0);
        // keys.register_key(Key::Key9, 10.0);
        // keys.register_key(Key::Key0, 10.0);
        // keys.register_key(Key::NumpadSubtract, 50.0);
        // keys.register_key(Key::NumpadAdd, 50.0);

        // Camera.
        let mut camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32);
        camera.set_rotation_sensitivity(0.4);
        camera.set_movement_sensitivity(0.02);

        let render_object =
                RenderObject::init(
                    &configuration.device,
                    &configuration.sc_desc,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("renderer_v4n4_debug_visualizator_wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/renderer_v4n4.wgsl"))),
                    
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
                    Some("Debug visualizator vvvvnnnn renderer with camera.")
        );

        Mc::create_textures(&configuration, &mut textures);

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

        let histogram = Histogram::init(&configuration.device, &vec![0; 1]);

        //++ let params = VisualizationParams {
        //++     curve_number: 1, // curver!!!  MAX_VERTEX_CAPACITY as u32,
        //++     iterator_start_index: 0,
        //++     iterator_end_index: 4096,
        //++     //iterator_end_index: 32768,
        //++     arrow_size: 0.3,
        //++     thread_mode: 0,
        //++     thread_mode_start_index: 0,
        //++     thread_mode_end_index: 0,
        //++     _padding: 0,
        //++ };

        //++ let temp_params = VisualizationParams {
        //++     curve_number: 1, // curver!!!  MAX_VERTEX_CAPACITY as u32,
        //++     iterator_start_index: 0,
        //++     iterator_end_index: THREAD_COUNT,
        //++     //iterator_end_index: 32768,
        //++     arrow_size: 0.3,
        //++     thread_mode: 1,
        //++     thread_mode_start_index: 0,
        //++     thread_mode_end_index: 0,
        //++     _padding: 0,
        //++ };

        //++ buffers.insert(
        //++     "visualization_params".to_string(),
        //++     buffer_from_data::<VisualizationParams>(
        //++     &configuration.device,
        //++     &vec![params],
        //++     wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        //++     None)
        //++ );

        //++ buffers.insert(
        //++     "debug_arrays".to_string(),
        //++     buffer_from_data::<Arrow>(
        //++         &configuration.device,
        //++         &vec![Arrow { start_pos: [0.0, 0.0, 0.0, 1.0],
        //++                       end_pos:   [4.0, 3.0, 0.0, 1.0],
        //++                       color: encode_rgba_u32(0, 0, 255, 255),
        //++                       size: 0.2,
        //++                       _padding: [0, 0]}],
        //++         wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        //++         Some("Debug array buffer")
        //++     )
        //++ );

        //++ buffers.insert(
        //++     "output".to_string(),
        //++     configuration.device.create_buffer(&wgpu::BufferDescriptor {
        //++         label: Some("draw buffer"),
        //++         size: VERTEX_BUFFER_SIZE as u64, 
        //++         usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        //++         mapped_at_creation: false,
        //++     })
        //++ );
        //
        
        let noise_params = NoiseParams {
            global_dim: [GLOBAL_NOISE_X_DIMENSION, GLOBAL_NOISE_Y_DIMENSION, GLOBAL_NOISE_Z_DIMENSION, 0],
            local_dim: [LOCAL_NOISE_X_DIMENSION, LOCAL_NOISE_Y_DIMENSION, LOCAL_NOISE_Z_DIMENSION, 0],
            time: 123,
            _pad: [0,0,0],
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

        let noise_compute_object =
                ComputeObject::init(
                    &configuration.device,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("noise compute object"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/noise_to_buffer.wgsl"))),
                    
                    }),
                    Some("Visualizer Compute object"),
                    &vec![
                        vec![
                            // @group(0)
                            // @binding(0)
                            // var<uniform> noise_params: NoiseParams;
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
                        ],
                    ]
        );

        //++ println!("Creating compute bind groups.");

        //++ println!("{:?}", buffers.get(&"debug_arrays".to_string()).unwrap().as_entire_binding());

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

        //++ println!("Creating render bind groups.");
 
        Self {
            screen: ScreenTexture::init(&configuration.device, &configuration.sc_desc, true),
            render_object: render_object,
            render_bind_groups: render_bind_groups,
            noise_compute_object: noise_compute_object,
            noise_compute_bind_groups: noise_compute_bind_groups,
            textures: textures,
            buffers: buffers,
            camera: camera,
            histogram: histogram,
            draw_count: 0,
            // visualization_params: params,
            keys: keys,
        }
    }

    fn render(&mut self,
              device: &wgpu::Device,
              queue: &mut wgpu::Queue,
              surface: &wgpu::Surface,
              sc_desc: &wgpu::SurfaceConfiguration) {

        // self.screen.acquire_screen_texture(
        //     &device,
        //     &sc_desc,
        //     &surface
        // );

        // let view = self.screen.surface_texture.as_ref().unwrap().texture.create_view(&wgpu::TextureViewDescriptor::default());

        // let clear = true;

            // draw(&mut encoder_render,
            //      &view,
            //      self.screen.depth_texture.as_ref().unwrap(),
            //      &self.render_bind_groups,
            //      &self.render_object.pipeline,
            //      &self.buffers.get("output").unwrap(),
            //      0..self.draw_count, 
            //      clear
            // );
            // queue.submit(Some(encoder_render.finish()));
            // self.histogram.set_values_cpu_version(queue, &vec![0]);

        // self.screen.prepare_for_rendering();

        // Reset counter.
        // self.histogram.reset_all_cpu_version(queue, 0); // TODO: fix histogram.reset_cpu_version        
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
    }
}

fn main() {
    
    jaankaup_core::template::run_loop::<Mc, BasicLoop, McFeatures>(); 
    println!("Finished...");
}
