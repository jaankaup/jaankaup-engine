use std::mem::size_of;
use std::collections::HashMap;
use std::borrow::Cow;
use jaankaup_core::template::{
        WGPUFeatures,
        WGPUConfiguration,
        Application,
        BasicLoop,
        Spawner,
};
use jaankaup_core::render_things::LightBuffer;
use jaankaup_core::render_object::{RenderObject, ComputeObject, create_bind_groups,draw};
use jaankaup_core::input::{KeyboardManager, InputCache};
use jaankaup_core::camera::Camera;
use jaankaup_core::buffer::{buffer_from_data};
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
    create_buffer_bindgroup_layout,
};
use jaankaup_core::histogram::Histogram;
use bytemuck::{Pod, Zeroable};

use winit::event as ev;
pub use ev::VirtualKeyCode as Key;

/// The number of vertices per chunk.
const MAX_VERTEX_CAPACITY: usize = 128 * 64 * 64; // 128 * 64 * 36 = 262144 verticex.

/// The size of draw buffer;
const VERTEX_BUFFER_SIZE: usize = 8 * MAX_VERTEX_CAPACITY * size_of::<f32>();

const THREAD_COUNT: u32 = 512;
// const THREAD_COUNT: u32 = 64;

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
    curve_number: u32,
    iterator_start_index: u32,
    iterator_end_index: u32,
    arrow_size: f32,
    thread_mode: u32,
    thread_mode_start_index: u32,
    thread_mode_end_index: u32,
    _padding: u32,
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
        limits.max_compute_workgroup_size_x = 1024;
        limits
    }
}

// struct KeyboardManager {
//     keys: HashMap<Key, (f64, f64)>,
// }
//
// impl KeyboardManager {
//     pub fn init() -> Self {
//         Self {
//             keys: HashMap::<Key, (f64, f64)>::new(),
//         }
//     }
//
//     pub fn register_key(&mut self, key: Key, threshold: f64) {
//         self.keys.insert(key, (0.0, threshold));
//     }
//
//     pub fn test_key(&mut self, key: &Key, input: &InputCache) -> bool {
//
//         let state_key = input.key_state(key);
//         let mut result = false;
//
//         if let Some(v) = self.keys.get_mut(key) {
//
//             match state_key {
//                 Some(InputState::Pressed(_)) => {
//                     let delta = (input.get_time_delta() / 1000000) as f64;
//                     v.0 = delta;
//                 }
//                 Some(InputState::Down(_, _)) => {
//                     let delta = (input.get_time_delta() / 1000000) as f64;
//                     v.0 = v.0 + delta;
//                     if v.0 > v.1 {
//                         v.0 = v.0 - v.1;
//                         result = true;
//                     }
//                 },
//                 Some(InputState::Released(_, _)) => {
//                     v.0 = 0.0;
//                 }
//                 _ => { }
//             }
//         }
//
//         return result;
//     }
// }
//
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct OtherRenderParams {
    scale_factor: f32,
}

// State for this application.
struct DebugVisualizator {
    screen: ScreenTexture,
    render_object: RenderObject,
    render_bind_groups: Vec<wgpu::BindGroup>,
    compute_object: ComputeObject,
    compute_bind_groups: Vec<wgpu::BindGroup>,
    buffers: HashMap<String, wgpu::Buffer>,
    camera: Camera,
    histogram: Histogram,
    draw_count: u32,
    visualization_params: VisualizationParams,
    temp_visualization_params: VisualizationParams,
    keys: KeyboardManager,
    block64mode: bool,
    _light: LightBuffer,
}

impl DebugVisualizator {

}

//#[allow(unused_variables)]
impl Application for DebugVisualizator {

    fn init(configuration: &WGPUConfiguration) -> Self {

        log::info!("Adapter limits are: ");

        let adapter_limits = configuration.adapter.limits();

        log::info!("\n");
        log::info!("max_compute_workgroup_storage_size: {:?}", adapter_limits.max_compute_workgroup_storage_size);
        log::info!("max_compute_invocations_per_workgroup: {:?}", adapter_limits.max_compute_invocations_per_workgroup);
        log::info!("max_compute_workgroup_size_x: {:?}", adapter_limits.max_compute_workgroup_size_x);
        log::info!("max_compute_workgroup_size_y: {:?}", adapter_limits.max_compute_workgroup_size_y);
        log::info!("max_compute_workgroup_size_z: {:?}", adapter_limits.max_compute_workgroup_size_z);
        log::info!("max_compute_workgroups_per_dimension: {:?}", adapter_limits.max_compute_workgroups_per_dimension);

        let mut buffers: HashMap<String, wgpu::Buffer> = HashMap::new();

        let mut keys = KeyboardManager::init();
        keys.register_key(Key::L, 20.0);
        keys.register_key(Key::K, 20.0);
        keys.register_key(Key::Key1, 10.0);
        keys.register_key(Key::Key2, 10.0);
        keys.register_key(Key::Key3, 10.0);
        keys.register_key(Key::Key4, 10.0);
        keys.register_key(Key::Key9, 10.0);
        keys.register_key(Key::Key0, 10.0);
        keys.register_key(Key::NumpadSubtract, 50.0);
        keys.register_key(Key::NumpadAdd, 50.0);

        // Camera.
        let mut camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (40.0,40.0,120.0), -90.0, 0.0);
        camera.set_rotation_sensitivity(0.4);
        camera.set_movement_sensitivity(0.02);

        let light = LightBuffer::create(
                      &configuration.device,
                      [10.0, 20.0, 10.0], // pos
                      [25, 25, 130],  // spec
                      [200,200,200], // light 
                      15.0,
                      0.15,
                      0.00013
        );

        let other_render_params = OtherRenderParams {
            scale_factor: 1.0,
        };

        buffers.insert(
            "other_render_params".to_string(),
            buffer_from_data::<OtherRenderParams>(
            &configuration.device,
            &[other_render_params],
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            None)
        );

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
                        vec![
                            // @group(0) @binding(0) var<uniform> camerauniform: Camera;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),

                            // @group(0) @binding(1) var<uniform> light: Light;
                            create_uniform_bindgroup_layout(1, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),

                            // @group(0) @binding(2) var<uniform> other_params: OtherParams;
                            create_uniform_bindgroup_layout(2, wgpu::ShaderStages::VERTEX)
                        ],
                    ],
                    Some("Debug visualizator vvvvnnnn renderer with camera."),
                    true,
                    wgpu::PrimitiveTopology::TriangleList
        );
        let render_bind_groups = create_bind_groups(
                                     &configuration.device,
                                     &render_object.bind_group_layout_entries,
                                     &render_object.bind_group_layouts,
                                     &vec![
                                         vec![
                                             &camera.get_camera_uniform(&configuration.device).as_entire_binding(),
                                             &light.get_buffer().as_entire_binding(),
                                             &buffers.get(&"other_render_params".to_string()).unwrap().as_entire_binding()
                                         ],
                                     ]
        );

        println!("Creating compute object.");

        let histogram = Histogram::init(&configuration.device, &vec![0; 2]);

        let params = VisualizationParams {
            curve_number: 1, // curver!!!  MAX_VERTEX_CAPACITY as u32,
            iterator_start_index: 0,
            iterator_end_index: 4096,
            //iterator_end_index: 32768,
            arrow_size: 0.3,
            thread_mode: 0,
            thread_mode_start_index: 0,
            thread_mode_end_index: 0,
            _padding: 0,
        };

        let temp_params = VisualizationParams {
            curve_number: 1, // curver!!!  MAX_VERTEX_CAPACITY as u32,
            iterator_start_index: 0,
            iterator_end_index: THREAD_COUNT,
            //iterator_end_index: 32768,
            arrow_size: 0.3,
            thread_mode: 1,
            thread_mode_start_index: 0,
            thread_mode_end_index: 0,
            _padding: 0,
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
                        label: Some("curves_wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/curves.wgsl"))),

                    }),
                    Some("Curves Compute object"),
                    &vec![
                        vec![
                            // @group(0) @binding(0)
                            // var<uniform> visualization_params: VisualizationParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1)
                            // var<storage, read_write> counter: Counter;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(2)
                            // var<storage, read> counter: array<Arrow>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(3)
                            // var<storage,read_write> output: array<VertexBuffer>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None
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
                                          ]
                                      ]
        );

        println!("Creating render bind groups.");

        Self {
            screen: ScreenTexture::init(&configuration.device, &configuration.sc_desc, true),
            render_object: render_object,
            render_bind_groups: render_bind_groups,
            compute_object: compute_object,
            compute_bind_groups: compute_bind_groups,
            buffers: buffers,
            camera: camera,
            histogram: histogram,
            draw_count: 0,
            visualization_params: params,
            temp_visualization_params: temp_params,
            keys: keys,
            block64mode: false,
            _light: light,
        }
    }

    fn render(&mut self,
              device: &wgpu::Device,
              queue: &mut wgpu::Queue,
              surface: &wgpu::Surface,
              sc_desc: &wgpu::SurfaceConfiguration,
              spawner: &Spawner) {

        self.screen.acquire_screen_texture(
            &device,
            &sc_desc,
            &surface
        );

        let view = self.screen.surface_texture.as_ref().unwrap().texture.create_view(&wgpu::TextureViewDescriptor::default());

        // @workgroup_size(512,1,1)
        let thread_count: u32 = 256;

        // let mut items_available: i32 = 32768;
        let mut items_available: i32 = // self.visualization_params.iterator_end_index as i32; //4096;
            if !self.block64mode { self.visualization_params.iterator_end_index as i32 }
            else { THREAD_COUNT as i32 };
        let dispatch_x: u32 = if !self.block64mode { items_available as u32 / thread_count }
                              else {
                                  if THREAD_COUNT == thread_count { 1 }
                                  else { std::cmp::max(1, THREAD_COUNT / thread_count) }
                              };

        let mut clear = true;
        let mut actual_index = if !self.block64mode { dispatch_x * thread_count }
                               else { self.temp_visualization_params.iterator_start_index };

        if self.block64mode { self.histogram.set_values_cpu_version(queue, &vec![0, actual_index]); }

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
            self.draw_count = counter[0] * 3;

            let mut encoder_render = device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
            });

            items_available = items_available - (dispatch_x * thread_count ) as i32;

            // Draw the cube.
            draw(&mut encoder_render,
                 &view,
                 self.screen.depth_texture.as_ref().unwrap(),
                 &self.render_bind_groups,
                 &self.render_object.pipeline,
                 &self.buffers.get("output").unwrap(),
                 0..self.draw_count,
                 clear
            );
            queue.submit(Some(encoder_render.finish()));
            clear = items_available <= 0;
            self.histogram.set_values_cpu_version(queue, &vec![0, actual_index]);
            actual_index = actual_index + dispatch_x*thread_count;
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
    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, input: &InputCache, spawner: &Spawner) {
        self.camera.update_from_input(&queue, &input);

        if self.keys.test_key(&Key::L, input) {
            self.visualization_params.arrow_size = self.visualization_params.arrow_size + 0.005;
            self.temp_visualization_params.arrow_size = self.visualization_params.arrow_size + 0.005;
        }
        if self.keys.test_key(&Key::K, input) {
            self.visualization_params.arrow_size = (self.visualization_params.arrow_size - 0.005).max(0.01);
            self.temp_visualization_params.arrow_size = (self.temp_visualization_params.arrow_size - 0.005).max(0.01);
        }
        if self.keys.test_key(&Key::Key1, input) {
            self.visualization_params.curve_number = 1;
            self.temp_visualization_params.curve_number = 1;
        }
        if self.keys.test_key(&Key::Key2, input) {
            self.visualization_params.curve_number = 2;
            self.temp_visualization_params.curve_number = 2;
        }
        if self.keys.test_key(&Key::Key3, input) {
            self.visualization_params.curve_number = 3;
            self.temp_visualization_params.curve_number = 3;
        }
        if self.keys.test_key(&Key::Key4, input) {
            self.visualization_params.curve_number = 4;
            self.temp_visualization_params.curve_number = 4;
        }
        if self.keys.test_key(&Key::Key9, input) {
            self.block64mode = true;
        }
        if self.keys.test_key(&Key::Key0, input) {
            self.block64mode = false;
        }
        if self.keys.test_key(&Key::NumpadSubtract, input) {
        //if self.keys.test_key(&Key::T, input) {
            let si = self.temp_visualization_params.iterator_start_index as i32;
            if si >= THREAD_COUNT as i32 {
                self.temp_visualization_params.iterator_start_index = self.temp_visualization_params.iterator_start_index - THREAD_COUNT;
                self.temp_visualization_params.iterator_end_index = self.temp_visualization_params.iterator_end_index - THREAD_COUNT;
            }
        }
        if self.keys.test_key(&Key::NumpadAdd, input) {
            let ei = self.temp_visualization_params.iterator_end_index;
            if ei <= 4096 - THREAD_COUNT {
                self.temp_visualization_params.iterator_start_index = self.temp_visualization_params.iterator_start_index + THREAD_COUNT;
                self.temp_visualization_params.iterator_end_index = self.temp_visualization_params.iterator_end_index + THREAD_COUNT;
            }
        }

        if self.block64mode {
            queue.write_buffer(
                &self.buffers.get(&"visualization_params".to_string()).unwrap(),
                0,
                bytemuck::cast_slice(&[self.temp_visualization_params])
            );
        }
        else {
            queue.write_buffer(
                &self.buffers.get(&"visualization_params".to_string()).unwrap(),
                0,
                bytemuck::cast_slice(&[self.visualization_params])
            );
        }
    }
}

fn main() {

    jaankaup_core::template::run_loop::<DebugVisualizator, BasicLoop, DebugVisualizatorFeatures>();
    println!("Finished...");
}
