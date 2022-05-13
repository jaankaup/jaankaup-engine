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
use jaankaup_core::buffer::{buffer_from_data};
use jaankaup_core::model_loader::{load_triangles_from_obj, TriangleMesh, create_from_bytes};
use jaankaup_core::camera::Camera;
use jaankaup_core::gpu_debugger::GpuDebugger;
use jaankaup_core::gpu_timer::GpuTimer;
use jaankaup_core::screen::ScreenTexture;
use jaankaup_core::shaders::{Render_VVVVNNNN_camera, Render_VVVVNNNN_camera_textures2, NoiseMaker};
use jaankaup_core::render_things::{LightBuffer, RenderParamBuffer};
use jaankaup_core::texture::Texture;
use jaankaup_core::aabb::Triangle_vvvvnnnn;
use jaankaup_core::common_functions::encode_rgba_u32;
use jaankaup_core::render_object::{draw, draw_indirect};

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
const FMM_GLOBAL_Y: usize = 32; 
const FMM_GLOBAL_Z: usize = 32; 

/// Mc inner dimensions.
const FMM_INNER_X: usize = 4; 
const FMM_INNER_Y: usize = 4; 
const FMM_INNER_Z: usize = 4; 

const MC_OUTPUT_BUFFER_SIZE: u32 = (FMM_GLOBAL_X *
                                    FMM_GLOBAL_Y *
                                    FMM_GLOBAL_Z *
                                    FMM_INNER_X *
                                    FMM_INNER_Y *
                                    FMM_INNER_Z *
                                    size_of::<f32>()) as u32 * 16;

/// Features and limits for Eikonal application.
struct EikonalFeatures {}

impl WGPUFeatures for EikonalFeatures {

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

/// Eikonal solver. A Fast marching method (GPU) for solving the eikonal equation.
struct Eikonal {
    camera: Camera,
    gpu_debugger: GpuDebugger,
    gpu_timer: Option<GpuTimer>,
    keyboard_manager: KeyboardManager,
    screen: ScreenTexture, 
    light: LightBuffer,
    render_params: RenderParamBuffer,
    triangle_mesh_renderer: Render_VVVVNNNN_camera,
    triangle_mesh_renderer_tex2: Render_VVVVNNNN_camera_textures2,
    triangle_mesh_bindgroups: Vec<wgpu::BindGroup>,
    triangle_mesh_bindgroups_tex2: Vec<wgpu::BindGroup>,
    buffers: HashMap<String, wgpu::Buffer>,
    triangle_meshes: HashMap<String, TriangleMesh>,
    noise_maker: NoiseMaker,
    marching_cubes: MarchingCubes,
    selected_noise: usize,
    noise_params: [f32; 5],
    textures: HashMap<String, Texture>,
    y_coord: f32,
//++    pub render_object_vvvvnnnn: RenderObject, 
//++    pub render_bind_groups_vvvvnnnn: Vec<wgpu::BindGroup>,
//++    pub render_object_vvvc: RenderObject,
//++    pub render_bind_groups_vvvc: Vec<wgpu::BindGroup>,
//++    pub compute_object_fmm: ComputeObject, 
//++    pub compute_bind_groups_fmm: Vec<wgpu::BindGroup>,
//++    pub compute_object_fmm_triangle: ComputeObject, 
//++    pub compute_bind_groups_fmm_triangle: Vec<wgpu::BindGroup>,
//++    pub compute_object_fmm_prefix_scan: ComputeObject,
//++    pub compute_bind_groups_fmm_prefix_scan: Vec<wgpu::BindGroup>,
//++    pub compute_object_reduce: ComputeObject,
//++    pub compute_bind_groups_reduce: Vec<wgpu::BindGroup>,
//++    pub compute_object_fmm_visualizer: ComputeObject,
//++    pub compute_bind_groups_fmm_visualizer: Vec<wgpu::BindGroup>,
//++    pub compute_object_initial_band_points: ComputeObject,
//++    pub compute_bind_groups_initial_band_points: Vec<wgpu::BindGroup>,
//++    pub buffers: HashMap<String, wgpu::Buffer>,
//++    pub camera: Camera,
//++    pub triangle_mesh_draw_count: u32,
//++    pub draw_triangle_mesh: bool,
//++    pub once: bool,
//++    pub fmm_prefix_params: FmmPrefixParams, 
//++    pub gpu_timer: Option<GpuTimer>,
//++    pub fmm_visualization_params: FmmVisualizationParams,
//++    pub histogram_fmm: Histogram,
//++    pub compute_object_calculate_all_band_values: ComputeObject,
//++    pub light: LightBuffer,
//++    pub other_render_params: OtherRenderParams,
}

impl Application for Eikonal {

    fn init(configuration: &WGPUConfiguration) -> Self {

        let selected_noise = 0;
        let noise_params = [1.0, 1.0, 1.0, 1.0, 1.0];
        let y_coord = 0.0;

        // Log adapter info.
        log_adapter_info(&configuration.adapter);

        // Camera.
        let mut camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (0.0, 0.0, 10.0), -89.0, 0.0);
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
                      [10.0, 20.0, 10.0], // pos
                      // [25, 25, 130],  // spec
                      [25, 25, 130],  // spec
                      [255,200,255], // light 
                      55.0,
                      0.15,
                      0.000013
        );

        // Scale_factor for triangle meshes.
        let render_params = RenderParamBuffer::create(
                    &configuration.device,
                    4.0
        );

        // RenderObject for basic triangle mesh rendering.
        let triangle_mesh_renderer = Render_VVVVNNNN_camera::init(&configuration.device, &configuration.sc_desc);

        // RenderObject for basic triangle mesh rendering with 2 textures.
        let triangle_mesh_renderer_tex2 = Render_VVVVNNNN_camera_textures2::init(&configuration.device, &configuration.sc_desc);

        // Create bindgroups for triangle_mesh_renderer.
        let triangle_mesh_bindgroups = 
                triangle_mesh_renderer.create_bingroups(
                    &configuration.device, &mut camera, &light, &render_params
                );

        // Textures hash map.
        let mut textures: HashMap<String, Texture> = HashMap::new();

        // Create textures for this application.
        create_textures(&configuration, &mut textures);

        // Create bindgroups for triangle_mesh_renderer.
        let triangle_mesh_bindgroups_tex2 = 
                triangle_mesh_renderer_tex2.create_bingroups(
                    &configuration.device,
                    &mut camera,
                    &light,
                    &render_params,
                    textures.get("rock").unwrap(),
                    textures.get("grass").unwrap(),
                );

        // Buffer hash_map.
        let mut buffers: HashMap<String, wgpu::Buffer> = HashMap::new();

        // Container for triangle meshes.
        let mut triangle_meshes: HashMap<String, TriangleMesh> = HashMap::new();

        // Load fire tower mesh.
        let fire_tower_mesh = load_vvvnnn_mesh(
                         &configuration.device,
                         include_str!("../../assets/models/wood.obj")[..].to_string(),
                         FIRE_TOWER_MESH,
                         2.0,
                         [25.0, -2.0, 25.0],
                         [11, 0, 155]
        );

        triangle_meshes.insert(
            FIRE_TOWER_MESH.to_string(),
            fire_tower_mesh);

        let noise_maker = NoiseMaker::init(
                &configuration.device,
                &"land_scape".to_string(),
                [32, 32, 32],
                [4, 4, 4],
                [0.0, 0.0, 0.0],
                0.0,
                0.0,
                noise_params[0],
                noise_params[1],
                noise_params[2],
                noise_params[3],
                noise_params[4],
        );

        let mc_params = McParams {
                base_position: [0.0, 0.0, 0.0, 1.0],
                isovalue: 0.5,
                cube_length: 1.0,
                future_usage1: 0.0,
                future_usage2: 0.0,
                noise_global_dimension: [FMM_GLOBAL_X.try_into().unwrap(),
                                         FMM_GLOBAL_Y.try_into().unwrap(),
                                         FMM_GLOBAL_Z.try_into().unwrap(),
                                         0
                ],
                noise_local_dimension: [FMM_INNER_X.try_into().unwrap(),
                                        FMM_INNER_Y.try_into().unwrap(),
                                        FMM_INNER_Z.try_into().unwrap(),
                                        0
                ],
        };

        buffers.insert(
            "mc_output".to_string(),
            configuration.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Mc output buffer"),
                size: MC_OUTPUT_BUFFER_SIZE as u64, 
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        );

        let mc_shader = &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("mc compute shader"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/mc_with_3d_texture.wgsl"))),
                        }
        );

        let mc_instance = MarchingCubes::init_with_noise_buffer(
            &configuration.device,
            &mc_params,
            &mc_shader,
            noise_maker.get_buffer(),
            &buffers.get(&"mc_output".to_string()).unwrap(),
        );

        Self {
            camera: camera,
            gpu_debugger: gpu_debugger,
            gpu_timer: gpu_timer,
            keyboard_manager: keyboard_manager,
            screen: ScreenTexture::init(&configuration.device, &configuration.sc_desc, true),
            light: light,
            render_params: render_params,
            triangle_mesh_renderer: triangle_mesh_renderer,
            triangle_mesh_renderer_tex2,
            triangle_mesh_bindgroups: triangle_mesh_bindgroups,
            triangle_mesh_bindgroups_tex2: triangle_mesh_bindgroups_tex2,
            buffers: buffers,
            triangle_meshes: triangle_meshes,
            noise_maker: noise_maker,
            marching_cubes: mc_instance,
            selected_noise: selected_noise,
            noise_params: noise_params,
            textures: textures,
            y_coord: y_coord,
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

        draw_indirect(
             &mut encoder,
             &view,
             self.screen.depth_texture.as_ref().unwrap(),
             &self.triangle_mesh_bindgroups_tex2,
             &self.triangle_mesh_renderer_tex2.get_render_object().pipeline,
             &self.buffers.get("mc_output").unwrap(),
             self.marching_cubes.get_draw_indirect_buffer(),
             0,
             clear
        );

        queue.submit(Some(encoder.finish())); 

        self.screen.prepare_for_rendering();

        // Reset counter.
        self.marching_cubes.reset_counter_value(device, queue);
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

        // TODO: something better
        if self.keyboard_manager.test_key(&Key::Key1, input) { self.selected_noise = 0; }
        if self.keyboard_manager.test_key(&Key::Key2, input) { self.selected_noise = 1; }
        if self.keyboard_manager.test_key(&Key::Key3, input) { self.selected_noise = 2; }
        if self.keyboard_manager.test_key(&Key::Key4, input) { self.selected_noise = 3; }
        if self.keyboard_manager.test_key(&Key::Key5, input) { self.selected_noise = 4; }
        if self.keyboard_manager.test_key(&Key::Up, input)   { self.y_coord = self.y_coord + 0.1; }
        if self.keyboard_manager.test_key(&Key::Down, input) { self.y_coord = self.y_coord - 0.1; }
        // if self.keyboard_manager.test_key(&Key::LEFT, input) { self.selected_noise = 4; }
        // if self.keyboard_manager.test_key(&Key::RIGHT, input){ self.noise_params[self.selected_noise] = self.noise_params[self.selected_noise] + 0.01; }
        if self.keyboard_manager.test_key(&Key::O, input) { self.noise_params[self.selected_noise] = self.noise_params[self.selected_noise] - 0.01;;}
        if self.keyboard_manager.test_key(&Key::P, input) { self.noise_params[self.selected_noise] = self.noise_params[self.selected_noise] + 0.01;;}

        match self.selected_noise {
            0 => { self.noise_maker.update_value2(queue, self.noise_params[0]); },
            1 => { self.noise_maker.update_value3(queue, self.noise_params[1]); },
            2 => { self.noise_maker.update_value4(queue, self.noise_params[2]); },
            3 => { self.noise_maker.update_value5(queue, self.noise_params[3]); },
            4 => { self.noise_maker.update_value6(queue, self.noise_params[4]); },
            _ => {}
        }

        let val = (((input.get_time() / 5000000) as f32) * 0.0015).sin() * 5.0;
        let value = (((input.get_time() / 5000000) as f32) * 0.0015).cos() * 0.35;

        let time = (input.get_time() as f64 / 50000000.0) as f32;

        self.noise_maker.update_time(&queue, time);
        self.noise_maker.update_value(&queue, value);


        let total_grid_count = FMM_GLOBAL_X *
                               FMM_GLOBAL_Y *
                               FMM_GLOBAL_Z *
                               FMM_INNER_X *
                               FMM_INNER_Y *
                               FMM_INNER_Z;

        let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Noise & Mc encoder.") });

        self.noise_maker.dispatch(&mut encoder_command);
        self.marching_cubes.dispatch(&mut encoder_command, total_grid_count as u32 / 256, 1, 1);

        // Submit compute.
        queue.submit(Some(encoder_command.finish()));

        let mut position = self.noise_maker.get_position();
        position[1] = self.y_coord;
        self.noise_maker.update_position(&queue, position);

        //++ let mut mc_params = self.marching_cubes.get_mc_params();
        //++ let camera_pos = self.camera.get_position();
        //++ mc_params.base_position = [20.0,
        //++                            0.0,
        //++                            20.0,
        //++                            0.0];
        //++ self.marching_cubes.update_mc_params(queue, mc_params);

        //++ let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Noise & Mc encoder.") });

        //++ self.noise_maker.dispatch(&mut encoder_command);
        //++ self.marching_cubes.dispatch(&mut encoder_command, total_grid_count as u32 / 256, 1, 1);

        //++ // Submit compute.
        //++ queue.submit(Some(encoder_command.finish()));

        //++ self.noise_maker.update_position(&queue, [0.0, 0.0, 0.0]);
        //++ let mut mc_params = self.marching_cubes.get_mc_params();
        //++ let camera_pos = self.camera.get_position();
        //++ mc_params.base_position = [0.0,
        //++                            0.0,
        //++                            0.0,
        //++                            0.0];
        //++ self.marching_cubes.update_mc_params(queue, mc_params);
    }
}

fn main() {
    
    jaankaup_core::template::run_loop::<Eikonal, BasicLoop, EikonalFeatures>(); 
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

fn create_textures(configuration: &WGPUConfiguration, textures: &mut HashMap<String, Texture>) {
    log::info!("Creating textures.");
    let grass_texture = Texture::create_from_bytes(
        &configuration.queue,
        &configuration.device,
        &configuration.sc_desc,
        1,
        &include_bytes!("../../assets/textures/grass1.png")[..],
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
