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
use jaankaup_core::buffer::{buffer_from_data, to_vec};
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

/// The number of vertices per chunk.
const MAX_VERTEX_CAPACITY: usize = 128 * 64 * 64; // 128 * 64 * 36 = 262144 verticex. 

/// The size of draw buffer;
const VERTEX_BUFFER_SIZE: usize = 8 * MAX_VERTEX_CAPACITY * size_of::<f32>();

// 
//  Arrow  
//
//  +----------------+
//  | start: [3;f32] |
//  | end: [3;f32]   |
//  | color: u32     |
//  | size: f32      |
//  +----------------+
//

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
struct VisualizationParams{
    max_local_vertex_capacity: u32,
    iterator_start_index: u32,
    iterator_end_index: u32,
    arrow_size: f32,
    // current_iterator_index: u32,
}

impl_convert!{Arrow}

struct DebugVisualizatorFeatures {}

impl WGPUFeatures for DebugVisualizatorFeatures {
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

struct KeyboardManager {
    keys: HashMap<Key, (f64, f64)>,
}

impl KeyboardManager {
    pub fn init() -> Self {
        Self {
            keys: HashMap::<Key, (f64, f64)>::new(),
        }
    }

    pub fn register_key(&mut self, key: Key, threshold: f64) {
        self.keys.insert(key, (0.0, threshold)); 
    }

    pub fn test_key(&mut self, key: &Key, input: &InputCache) -> bool {
        
        let state_key = input.key_state(key);
        let mut result = false;

        match state_key {
            Some(InputState::Down(s, e)) => {
                let delta = (e - s) as f64 / 1000000.0;
                if let Some(v) = self.keys.get_mut(key) {
                    // println!("size ....{:}", *v );
                    v.0 = v.0 + delta; 
                    if v.0 > v.1 {
                        println!("updating ....{:}", delta);
                        //self.visualization_params.arrow_size = self.visualization_params.arrow_size + 0.005;  
                        v.0 = 0.0;
                        result = true;
                    }
                }
            },
            _ => { }
        }

        return result;
    }
}

// State for this application.
struct DebugVisualizator {
    pub screen: ScreenTexture, 
    pub render_object: RenderObject, 
    pub render_bind_groups: Vec<wgpu::BindGroup>,
    pub compute_object: ComputeObject, 
    pub compute_bind_groups: Vec<wgpu::BindGroup>,
    // pub _textures: HashMap<String, Texture>,
    pub buffers: HashMap<String, wgpu::Buffer>,
    pub camera: Camera,
    pub histogram: Histogram,
    pub draw_count: u32,
    pub update: bool,
    pub visualization_params: VisualizationParams,
    pub keys: KeyboardManager,
}

impl DebugVisualizator {

}

//#[allow(unused_variables)]
impl Application for DebugVisualizator {

    fn init(configuration: &WGPUConfiguration) -> Self {

        log::info!("Adapter limits are: ");

        let adapter_limits = configuration.adapter.limits(); 

        log::info!("max_compute_workgroup_storage_size: {:?}", adapter_limits.max_compute_workgroup_storage_size);
        log::info!("max_compute_invocations_per_workgroup: {:?}", adapter_limits.max_compute_invocations_per_workgroup);
        log::info!("max_compute_workgroup_size_x: {:?}", adapter_limits.max_compute_workgroup_size_x);
        log::info!("max_compute_workgroup_size_y: {:?}", adapter_limits.max_compute_workgroup_size_y);
        log::info!("max_compute_workgroup_size_z: {:?}", adapter_limits.max_compute_workgroup_size_z);
        log::info!("max_compute_workgroups_per_dimension: {:?}", adapter_limits.max_compute_workgroups_per_dimension);

        let mut buffers: HashMap<String, wgpu::Buffer> = HashMap::new();
        // buffers.insert("cube".to_string(), create_cube(&configuration.device, false));

        let mut keys = KeyboardManager::init();
        keys.register_key(Key::L, 1000.0);
        keys.register_key(Key::K, 1000.0);
        keys.register_key(Key::Key1, 1000.0);
        keys.register_key(Key::Key2, 1000.0);
        keys.register_key(Key::Key3, 1000.0);
        keys.register_key(Key::Key4, 1000.0);

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
                    Some("Debug visualizator vvvvnnnn renderer with camera.")
        );
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
                                     ]
        );

        println!("Creating compute object.");

        // let histogram = Histogram::init(&configuration.device, &vec![0; 2]);
        let histogram = Histogram::init(&configuration.device, &vec![0; 2]);

        let params = VisualizationParams {
            max_local_vertex_capacity: 1, // curver!!!  MAX_VERTEX_CAPACITY as u32,
            iterator_start_index: 0,
            iterator_end_index: 4096,
            //iterator_end_index: 32768,
            arrow_size: 0.3,
        };

        buffers.insert(
            "visualization_params".to_string(),
            buffer_from_data::<VisualizationParams>(
            &configuration.device,
            &vec![params],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            None)
        );

        buffers.insert(
            "debug_arrays".to_string(),
            buffer_from_data::<Arrow>(
                &configuration.device,
                &vec![Arrow { start_pos: [0.0, 0.0, 0.0, 1.0],
                              end_pos:   [4.0, 3.0, 0.0, 1.0],
                              color: encode_rgba_u32(0, 0, 255, 255),
                              size: 0.2,
                              _padding: [0, 0]}],
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Debug array buffer")
            )
        );

        buffers.insert(
            "output".to_string(),
            configuration.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("draw buffer"),
                size: VERTEX_BUFFER_SIZE as u64, 
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        );


        let compute_object =
                ComputeObject::init(
                    &configuration.device,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("compute_visuzlizer_wgsl"),
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

        println!("Creating compute bind groups.");

        println!("{:?}", buffers.get(&"debug_arrays".to_string()).unwrap().as_entire_binding());

        let compute_bind_groups = create_bind_groups(
                                      &configuration.device,
                                      &compute_object.bind_group_layout_entries,
                                      &compute_object.bind_group_layouts,
                                      &vec![
                                          vec![
                                               &buffers.get(&"visualization_params".to_string()).unwrap().as_entire_binding(),
                                               &histogram.get_histogram_buffer().as_entire_binding(),
                                               &buffers.get(&"debug_arrays".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output".to_string()).unwrap().as_entire_binding()

                                               // vec![&wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                               //         buffer: &buffers.get(&"visualization_params".to_string()).unwrap(),
                                               //         offset: 0,
                                               //         size: Some(core::num::NonZeroU64::new(40)),
                                               // }),
                                               // &wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                               //         buffer: &histogram.get_histogram_buffer(),
                                               //         offset: Some(core::num::NonZeroU64::new(8)),
                                               //         size: None,
                                               // }),
                                               // &wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                               //         buffer: &buffers.get(&"debug_arrays".to_string()).unwrap(),
                                               //         offset: 0,
                                               //         size: None,
                                               // }),
                                               // &wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                               //         buffer: &buffers.get(&"output".to_string()).unwrap(),
                                               //         offset: 0,
                                               //         size: None,
                                               // }),
                                          ]
                                      ]
        );

        println!("Creating render bind groups.");

        // Create bind groups for basic render pipeline and grass/rock textures.
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
                                    ]
        );
 
        Self {
            screen: ScreenTexture::init(&configuration.device, &configuration.sc_desc, true),
            render_object: render_object,
            render_bind_groups: render_bind_groups,
            compute_object: compute_object,
            compute_bind_groups: compute_bind_groups,
            // _textures: textures,
            buffers: buffers,
            camera: camera,
            histogram: histogram,
            draw_count: 0,
            update: true,
            visualization_params: params,
            keys: keys,
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

        // let mut items_available: i32 = 32768; 
        let mut items_available: i32 = 4096; 
        let mut dispatch_x = 64;
        let mut clear = true;
        let mut i = dispatch_x * 64;

        while items_available > 0 {

            let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Visualiztion (AABB)") });

            // Take 128 * 64 = 8192 items.
            self.compute_object.dispatch(
                &self.compute_bind_groups,
                &mut encoder_command,
                dispatch_x, 1, 1, Some("aabb dispatch")
            );

            // Submit compute.
            queue.submit(Some(encoder_command.finish()));

            let counter = self.histogram.get_values(device, queue);
            self.draw_count = counter[0];
            // println!("draw_count == {}", self.draw_count);
            // println!("items_available == {}", items_available);

            // set_values_cpu_version(&self, queue, !vec[0, counter[1]);

            let mut encoder_render = device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
            });

            items_available = items_available - dispatch_x as i32 * 64; 
            // println!("items_available == {}", items_available);

            // Draw the cube.
            draw(&mut encoder_render,
                 &view,
                 self.screen.depth_texture.as_ref().unwrap(),
                 &self.render_bind_groups,
                 &self.render_object.pipeline,
                 &self.buffers.get("output").unwrap(),
                 0..self.draw_count, // TODO: Cube 
                 clear
                 //true
            );
            queue.submit(Some(encoder_render.finish()));
            clear = items_available <= 0;
            self.histogram.set_values_cpu_version(queue, &vec![0, i]);
            i = i + dispatch_x*64;
            // if (items_available <= 0) { break; } 
        }

        self.screen.prepare_for_rendering();

        // Reset counter.
        self.histogram.reset_all_cpu_version(queue, 0); // TODO: fix histogram.reset_cpu_version        
    }

    fn input(&mut self, queue: &wgpu::Queue, input_cache: &InputCache) {
    }

    fn resize(&mut self, device: &wgpu::Device, sc_desc: &wgpu::SurfaceConfiguration, _new_size: winit::dpi::PhysicalSize<u32>) {
        self.screen.depth_texture = Some(Texture::create_depth_texture(&device, &sc_desc, Some("depth-texture")));
        self.camera.resize(sc_desc.width as f32, sc_desc.height as f32);
    }

    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, input: &InputCache) {
        self.camera.update_from_input(&queue, &input);

        if self.update {
            // let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Visualiztion (AABB)") });
            // self.compute_object.dispatch(
            //     &self.compute_bind_groups,
            //     &mut encoder,
            //     512, 1, 1, Some("aabb dispatch")
            // );
            // queue.submit(Some(encoder.finish()));
            // let counter = self.histogram.get_values(device, queue);
            // self.draw_count = counter[0];
            // println!("{:?}", counter);
            // self.histogram.reset_all_cpu_version(queue, 0); // TODO: fix histogram.reset_cpu_version        
            // self.update = false;
        }

        if self.keys.test_key(&Key::L, input) { 
            self.visualization_params.arrow_size = self.visualization_params.arrow_size + 0.005;  
        }
        if self.keys.test_key(&Key::K, input) { 
            self.visualization_params.arrow_size = (self.visualization_params.arrow_size - 0.005).max(0.01);  
        }
        if self.keys.test_key(&Key::Key1, input) { 
            self.visualization_params.max_local_vertex_capacity = 1;  
        }
        if self.keys.test_key(&Key::Key2, input) { 
            self.visualization_params.max_local_vertex_capacity = 2;
        }
        if self.keys.test_key(&Key::Key3, input) { 
            self.visualization_params.max_local_vertex_capacity = 3;  
        }
        if self.keys.test_key(&Key::Key4, input) { 
            self.visualization_params.max_local_vertex_capacity = 4;  
        }
        // let state_k = input.key_state(&Key::K);
        // let mut change = false;
        // 
        // match state_l {
        //     Some(InputState::Down(s, e)) => {
        //         let delta = (e - s) as f64 / 1000000.0;
        //         if let Some(v) = self.keys.get_mut(&Key::L) {
        //             println!("size ....{:}", *v );
        //             *v = *v + delta; 
        //             if *v > 1000.0 {
        //                 println!("updating ....{:}", delta);
        //                 self.visualization_params.arrow_size = self.visualization_params.arrow_size + 0.005;  
        //                 *v = 0.0;
        //             }
        //         }
        //     },
        //     _ => { }
        // }

        // println!("{:?}", state_l);
        queue.write_buffer(
            &self.buffers.get(&"visualization_params".to_string()).unwrap(),
            0,
            bytemuck::cast_slice(&[self.visualization_params])
        );
    }
}

fn main() {
    
    jaankaup_core::template::run_loop::<DebugVisualizator, BasicLoop, DebugVisualizatorFeatures>(); 
    println!("Finished...");
}
