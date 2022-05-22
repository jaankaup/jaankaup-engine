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
use jaankaup_core::render_object::{draw, draw_indirect, RenderObject, ComputeObject, create_bind_groups};
use jaankaup_core::{wgpu, log};
use jaankaup_core::winit;
use jaankaup_core::buffer::{buffer_from_data};
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
use jaankaup_core::fmm_things::{DomainTester, Permutation};

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

/// Global dimensions. 
const FMM_GLOBAL_X: usize = 32; 
const FMM_GLOBAL_Y: usize = 32; 
const FMM_GLOBAL_Z: usize = 32; 

/// Inner dimensions.
const FMM_INNER_X: usize = 4; 
const FMM_INNER_Y: usize = 4; 
const FMM_INNER_Z: usize = 4; 

const OUTPUT_BUFFER_SIZE: u32 = (FMM_GLOBAL_X *
                                 FMM_GLOBAL_Y *
                                 FMM_GLOBAL_Z *
                                 FMM_INNER_X *
                                 FMM_INNER_Y *
                                 FMM_INNER_Z *
                                 size_of::<f32>()) as u32 * 16;

/// Features and limits for TestProject application.
struct TestProjectFeatures {}

impl WGPUFeatures for TestProjectFeatures {

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

/// The purpose of this application is to test modules and functions.
struct TestProject {
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
    triangle_meshes: HashMap<String, TriangleMesh>,
    domain_tester: DomainTester,
}

impl Application for TestProject {

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

        // Permutations.
        let mut permutations: Vec<Permutation> = Vec::new();

        permutations.push(
            Permutation { modulo: 3, x_factor: 2,  y_factor: 13,  z_factor: 17, }
        );

        // let permutation_number = (2u * position_u32_temp.x +  13u * position_u32_temp.y + 17u * position_u32_temp.z) % 4u; 
        // let permutation_number = (2u * position_u32_temp.x +  3u * position_u32_temp.y + 5u * position_u32_temp.z) & 3u; 
        // let permutation_number = (3u * position_u32_temp.x +  5u * position_u32_temp.y + 7u * position_u32_temp.z) & 2u; 
        
        // The DomainTester.
        let domain_tester = DomainTester::init(
            &configuration.device,
            &gpu_debugger,
            [16, 16, 16],
            [4, 4, 4],
            &permutations
            );

        // Container for triangle meshes.
        let mut triangle_meshes: HashMap<String, TriangleMesh> = HashMap::new();

        // Load fire tower mesh.
        let fire_tower_mesh = load_vvvnnn_mesh(
                         &configuration.device,
                         include_str!("../../assets/models/wood.obj")[..].to_string(),
                         FIRE_TOWER_MESH,
                         2.0,
                         [50.0, 25.0, 50.0],
                         [11, 0, 155]
        );

        triangle_meshes.insert(
            FIRE_TOWER_MESH.to_string(),
            fire_tower_mesh);

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
            triangle_meshes: triangle_meshes,
            domain_tester: domain_tester,
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

        let buf = buffer_from_data::<f32>(
                  &device,
                  &vec![1.0,2.0,3.0,2.0,4.0],
                  wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                  Some("Computational domain wgpu::buffer.")
        );

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
        });

        let view = self.screen.surface_texture.as_ref().unwrap().texture.create_view(&wgpu::TextureViewDescriptor::default());

        let fire_tower = self.triangle_meshes.get(&FIRE_TOWER_MESH.to_string()).unwrap();

        let mut clear = true;

        draw(&mut encoder,
             &view,
             self.screen.depth_texture.as_ref().unwrap(),
             &self.triangle_mesh_bindgroups,
             &self.triangle_mesh_renderer.get_render_object().pipeline,
             &fire_tower.get_buffer(),
             0..fire_tower.get_triangle_count() * 3, 
             clear
        );
        
        clear = false;

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
    
    jaankaup_core::template::run_loop::<TestProject, BasicLoop, TestProjectFeatures>(); 
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

        GpuDebugger::Init(
                &device,
                &sc_desc,
                &camera.get_camera_uniform(&device),
                MAX_NUMBER_OF_VVVVNNNN.try_into().unwrap(),
                MAX_NUMBER_OF_CHARS.try_into().unwrap(),
                MAX_NUMBER_OF_ARROWS.try_into().unwrap(),
                MAX_NUMBER_OF_AABBS.try_into().unwrap(),
                MAX_NUMBER_OF_AABB_WIRES.try_into().unwrap(),
                64,
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
