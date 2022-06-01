use rand::Rng;
use jaankaup_core::radix::KeyMemoryIndex;
use std::mem::size_of;
use std::borrow::Cow;
use std::collections::HashMap;
use std::convert::TryInto;
use jaankaup_core::input::*;
use jaankaup_algorithms::mc::{McParams, MarchingCubes};
use jaankaup_core::template::{
        WGPUFeatures,
        WGPUConfiguration,
        Application,
        BasicLoop,
        Spawner,
};
use jaankaup_core::{wgpu, log};
use jaankaup_core::winit;
use jaankaup_core::buffer::{buffer_from_data, to_vec};
use jaankaup_core::model_loader::{load_triangles_from_obj, TriangleMesh, create_from_bytes};
use jaankaup_core::camera::Camera;
use jaankaup_core::gpu_debugger::GpuDebugger;
use jaankaup_core::gpu_timer::GpuTimer;
use jaankaup_core::screen::ScreenTexture;
use jaankaup_core::shaders::{Render_VVVVNNNN_camera};
use jaankaup_core::render_things::{LightBuffer, RenderParamBuffer};
use jaankaup_core::texture::Texture;
use jaankaup_core::aabb::Triangle_vvvvnnnn;
use jaankaup_core::common_functions::encode_rgba_u32;
use jaankaup_core::render_object::{draw, draw_indirect};
use jaankaup_core::radix::{RadixSort};

/// Max number of arrows for gpu debugger.
const MAX_NUMBER_OF_ARROWS:     usize = 40960;

/// Max number of aabbs for gpu debugger.
const MAX_NUMBER_OF_AABBS:      usize = 262144;

/// Max number of box frames for gpu debugger.
const MAX_NUMBER_OF_AABB_WIRES: usize = 40960;

/// Max number of renderable char elements (f32, vec3, vec4, ...) for gpu debugger.
const MAX_NUMBER_OF_CHARS:      usize = 262144;

/// Max number of vvvvnnnn vertices reserved for gpu draw buffer.
const MAX_NUMBER_OF_VVVVNNNN: usize = 2000000;

/// Name for the fire tower mesh (assets/models/wood.obj).
const FIRE_TOWER_MESH: &'static str = "FIRE_TOWER";

/// Mc global dimensions. 
const FMM_GLOBAL_X: usize = 32; 
const FMM_GLOBAL_Y: usize = 8; 
const FMM_GLOBAL_Z: usize = 32; 

/// Mc inner dimensions.
const FMM_INNER_X: usize = 4; 
const FMM_INNER_Y: usize = 4; 
const FMM_INNER_Z: usize = 4; 

// const MC_OUTPUT_BUFFER_SIZE: u32 = (FMM_GLOBAL_X *
//                                     FMM_GLOBAL_Y *
//                                     FMM_GLOBAL_Z *
//                                     FMM_INNER_X *
//                                     FMM_INNER_Y *
//                                     FMM_INNER_Z *
//                                     size_of::<f32>()) as u32 * 16;

/// Features and limits for FastMarchingMethod application.
struct FastMarchingMethodFeatures {}

impl WGPUFeatures for FastMarchingMethodFeatures {

    fn optional_features() -> wgpu::Features {
        wgpu::Features::TIMESTAMP_QUERY
    }

    fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }

    fn required_limits() -> wgpu::Limits {
        let mut limits = wgpu::Limits::default();
        limits.max_compute_invocations_per_workgroup = 1024;
        limits.max_compute_workgroup_size_x = 1024;
        limits.max_storage_buffers_per_shader_stage = 10;
        limits
    }
}

/// FastMarchingMethod solver. A Fast marching method (GPU) for solving the eikonal equation.
struct FastMarchingMethod {
    camera: Camera,
    gpu_debugger: GpuDebugger,
    gpu_timer: Option<GpuTimer>,
    keyboard_manager: KeyboardManager,
    screen: ScreenTexture, 
    light: LightBuffer,
    render_params: RenderParamBuffer,
    triangle_mesh_renderer: Render_VVVVNNNN_camera,
    triangle_mesh_bindgroups: Vec<wgpu::BindGroup>,
    buffers: HashMap<String, wgpu::Buffer>,
    radix_sort: RadixSort,
}

impl Application for FastMarchingMethod {

    fn init(configuration: &WGPUConfiguration) -> Self {

        // Log adapter info.
        log_adapter_info(&configuration.adapter);

        // Camera.
        let mut camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (0.0, 30.0, 10.0), -89.0, 0.0);
        camera.set_rotation_sensitivity(0.4);
        camera.set_movement_sensitivity(0.02);

        // Gpu debugger.
        let gpu_debugger = create_gpu_debugger( &configuration.device, &configuration.sc_desc, &mut camera);

        // Gpu timer.
        let gpu_timer = GpuTimer::init(&configuration.device, &configuration.queue, 8, Some("gpu timer"));

        // Keyboard manager. Keep tract of keys which has been pressed, and for how long time.
        let mut keyboard_manager = create_keyboard_manager();

        // Light source for triangle meshes.
        let light = LightBuffer::create(
                      &configuration.device,
                      [25.0, 55.0, 25.0], // pos
                      // [25, 25, 130],  // spec
                      [25, 25, 130],  // spec
                      [255,200,255], // light 
                      55.0,
                      0.35,
                      0.000013
        );

        // Scale_factor for triangle meshes.
        let render_params = RenderParamBuffer::create(
                    &configuration.device,
                    1.0
        );

        // RenderObject for basic triangle mesh rendering.
        let triangle_mesh_renderer = Render_VVVVNNNN_camera::init(&configuration.device, &configuration.sc_desc);

        // Create bindgroups for triangle_mesh_renderer.
        let triangle_mesh_bindgroups = 
                triangle_mesh_renderer.create_bingroups(
                    &configuration.device, &mut camera, &light, &render_params
                );

        // Buffer hash_map.
        let mut buffers: HashMap<String, wgpu::Buffer> = HashMap::new();

        let key_count: u32 = 30000;

        let mut rng = rand::thread_rng();
        let mut radix_test_data: Vec<KeyMemoryIndex> = Vec::with_capacity(key_count as usize);

        for i in 0..key_count {
            radix_test_data.push(KeyMemoryIndex { key: rng.gen_range(0..256), memory_location: 55, }); 
        }

        let radix_input_buffer = buffer_from_data::<KeyMemoryIndex>(
            &configuration.device,
            &radix_test_data,
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            None
        );

        // Radix sort
        let radix_sort = RadixSort::init(&configuration.device, &radix_input_buffer, key_count, key_count);

        // Store radix input buffer. TODO: create a better sample buffer.
        buffers.insert("radix_buffer".to_string(), radix_input_buffer);
        
        let mut encoder = configuration.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Radix sort counting sort encoder"),
        });

        radix_sort.initial_counting_sort(&mut encoder);

        configuration.queue.submit(Some(encoder.finish())); 

        let radix_histogram = to_vec::<u32>(
                &configuration.device,
                &configuration.queue,
                radix_sort.get_global_histogram(),
                0,
                (size_of::<u32>()) as wgpu::BufferAddress * (256 as u64)
            );

        for i in 0..256 {
            println!("{} :: {}", i, radix_histogram[i]);
        }

        println!("{}", radix_histogram.iter().sum::<u32>());

        Self {
            camera: camera,
            gpu_debugger: gpu_debugger,
            gpu_timer: gpu_timer,
            keyboard_manager: keyboard_manager,
            screen: ScreenTexture::init(&configuration.device, &configuration.sc_desc, true),
            light: light,
            render_params: render_params,
            triangle_mesh_renderer: triangle_mesh_renderer,
            triangle_mesh_bindgroups: triangle_mesh_bindgroups,
            buffers: buffers,
            radix_sort: radix_sort,
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

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
        });

        let view = self.screen.surface_texture.as_ref().unwrap().texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut clear = true;

        // draw(&mut encoder,
        //      &view,
        //      self.screen.depth_texture.as_ref().unwrap(),
        //      if self.texture_setup { &self.triangle_mesh_bindgroups_tex2 } else { &self.triangle_mesh_bindgroups_tex2b },
        //      &self.triangle_mesh_renderer_tex2.get_render_object().pipeline,
        //      &fire_tower.get_buffer(),
        //      0..fire_tower.get_triangle_count() * 3, 
        //      clear
        // );
        // 
        // clear = false;

        queue.submit(Some(encoder.finish())); 

        self.screen.prepare_for_rendering();
    }

    #[allow(unused)]
    fn input(&mut self, queue: &wgpu::Queue, input: &InputCache) {
        self.camera.update_from_input(&queue, &input);
    }

    fn resize(&mut self, device: &wgpu::Device, sc_desc: &wgpu::SurfaceConfiguration, _new_size: winit::dpi::PhysicalSize<u32>) {

        // TODO: add this functionality to the Screen.
        self.screen.depth_texture = Some(Texture::create_depth_texture(&device, &sc_desc, Some("depth-texture")));
        self.camera.resize(sc_desc.width as f32, sc_desc.height as f32);
    }

    #[allow(unused)]
    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, input: &InputCache, spawner: &Spawner) {

        let total_grid_count = FMM_GLOBAL_X *
                               FMM_GLOBAL_Y *
                               FMM_GLOBAL_Z *
                               FMM_INNER_X *
                               FMM_INNER_Y *
                               FMM_INNER_Z;

        //++ let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Noise & Mc encoder.") });

        //++ queue.submit(Some(encoder_command.finish()));

    }
}

fn main() {
    
    jaankaup_core::template::run_loop::<FastMarchingMethod, BasicLoop, FastMarchingMethodFeatures>(); 
    println!("Finished...");
}

/// A helper function for logging adapter information.
fn log_adapter_info(adapter: &wgpu::Adapter) {

        let adapter_limits = adapter.limits(); 

        log::info!("Adapter limits.");
        log::info!("max_compute_workgroup_storage_size: {:?}", adapter_limits.max_compute_workgroup_storage_size);
        log::info!("max_compute_invocations_per_workgroup: {:?}", adapter_limits.max_compute_invocations_per_workgroup);
        log::info!("max_compute_workgroup_size_x: {:?}", adapter_limits.max_compute_workgroup_size_x);
        log::info!("max_compute_workgroup_size_y: {:?}", adapter_limits.max_compute_workgroup_size_y);
        log::info!("max_compute_workgroup_size_z: {:?}", adapter_limits.max_compute_workgroup_size_z);
        log::info!("max_compute_workgroups_per_dimension: {:?}", adapter_limits.max_compute_workgroups_per_dimension);
}

/// Initialize and create GpuDebugger for this project. TODO: add MAX_... to function parameters.
fn create_gpu_debugger(device: &wgpu::Device,
                       sc_desc: &wgpu::SurfaceConfiguration,
                       camera: &mut Camera) -> GpuDebugger {

        GpuDebugger::init(
                &device,
                &sc_desc,
                &camera.get_camera_uniform(&device),
                MAX_NUMBER_OF_VVVVNNNN.try_into().unwrap(),
                MAX_NUMBER_OF_CHARS.try_into().unwrap(),
                MAX_NUMBER_OF_ARROWS.try_into().unwrap(),
                MAX_NUMBER_OF_AABBS.try_into().unwrap(),
                MAX_NUMBER_OF_AABB_WIRES.try_into().unwrap(),
                64,
                1.0,
        )
}

/// Initialize and create KeyboardManager. Register all the keys that are used in the application.
/// Registered keys: P, N, Key1, Key2, Key3, Key4, Key0
fn create_keyboard_manager() -> KeyboardManager {

        let mut keys = KeyboardManager::init();

        keys.register_key(Key::Up, 5.0);
        keys.register_key(Key::Down, 5.0);
        keys.register_key(Key::P, 5.0);
        keys.register_key(Key::O, 5.0);
        keys.register_key(Key::Key1, 20.0);
        keys.register_key(Key::Key2, 20.0);
        keys.register_key(Key::Key3, 20.0);
        keys.register_key(Key::Key4, 20.0);
        keys.register_key(Key::Key5, 20.0);
        keys.register_key(Key::Key0, 20.0);
        keys.register_key(Key::N, 200.0);
        keys.register_key(Key::Space, 50.0);
        
        keys
}

/// Load a wavefront mesh and store it to hash_map. Drop texture coordinates.
fn load_vvvnnn_mesh(device: &wgpu::Device,
                    data: String,
                    buffer_name: &'static str,
                    scale_factor: f32,
                    transition: [f32;3],
                    color: [u32;3]) -> TriangleMesh {

        // Load model. Tower.
        let (_, mut triangle_mesh_wood, _) = create_from_bytes(
            data,
            scale_factor,
            transition,
            None)
            .unwrap();

        let triangle_mesh_draw_count = triangle_mesh_wood.len() as u32; 

        let col = unsafe {
            std::mem::transmute::<u32, f32>(encode_rgba_u32(color[0], color[1], color[2], 255))
        };

        // Apply color information to the fourth component (triangle position).
        for tr in triangle_mesh_wood.iter_mut() {
            tr.a.w = col;
            tr.b.w = col;
            tr.c.w = col;
        }

        TriangleMesh::create_from_data(&device,
                                       &triangle_mesh_wood,
                                       buffer_name,
                                       triangle_mesh_draw_count)
}
