use jaankaup_core::pc_parser::VVVC;
use jaankaup_core::pc_parser::read_pc_data;
use jaankaup_core::common_functions::create_uniform_bindgroup_layout;
use itertools::Itertools;
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
    const MAX_NUMBER_OF_AABBS:      usize = TOTAL_INDICES;

    /// Max number of box frames for gpu debugger.
    const MAX_NUMBER_OF_AABB_WIRES: usize = 40960;

    /// Max number of renderable char elements (f32, vec3, vec4, ...) for gpu debugger.
    const MAX_NUMBER_OF_CHARS:      usize = 262144;

    /// Max number of vvvvnnnn vertices reserved for gpu draw buffer.
    const MAX_NUMBER_OF_VVVVNNNN: usize = 2000000;

    /// Name for the fire tower mesh (assets/models/wood.obj).
    const FIRE_TOWER_MESH: &'static str = "FIRE_TOWER";

    const CLOUD_DATA: &'static str = "CLOUD_DATA";

    /// Global dimensions. 
    const FMM_GLOBAL_X: usize = 16; 
    const FMM_GLOBAL_Y: usize = 16; 
    const FMM_GLOBAL_Z: usize = 16; 

    /// Inner dimensions.
    const FMM_INNER_X: usize = 4; 
    const FMM_INNER_Y: usize = 4; 
    const FMM_INNER_Z: usize = 4; 

    const TOTAL_INDICES: usize = FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z * FMM_INNER_X * FMM_INNER_Y * FMM_INNER_Z; 

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
        aabb_size: f32,
        font_size: f32,
        render_object_vvvc: RenderObject,
        render_bind_groups_vvvc: Vec<wgpu::BindGroup>,
        point_count: u32,
        start_index: u32,
        step_size: u32,
    }

    impl Application for TestProject {

        fn init(configuration: &WGPUConfiguration) -> Self {

            let aabb_size: f32 = 0.15;
            let font_size: f32 = 0.016;

            let start_index = 0;
            let step_size = 1024;

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

            // Create different permutations such that (a*x + b*y + c*) % d where
            //
            // a, b and c are prime numbers, a != b != c, and a % d != , b % d != 0, c % d != 0.
            //

            // 2 	3 	5 	7 	11 	13 	17 	19 	23 	29 	31 	37 	41 	43 	47 	53 	59 	61 	67 	71 
            // 73 	79 	83 	89 	97 	101 	103 	107 	109 	113 	127 	131 	137 	139 	149 	151 	157 	163 	167 	173

            let primes = vec![2, 3,	5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]; 

            let perms = (0..5).permutations(3).collect_vec();
            //++ let mut blah: Vec<[i32; 3]> = Vec::new();
            //++ for mut elem in perms.clone() {
            //++     elem.sort();
            //++     //blah.append([elem[0] as i32, elem[1] as i32, elem[2] as i32]);
            //++     blah.append([1,2,3]);
            //++     // blah.append(elem);
            //++ }
            // println!("{:?}", perms);

            permutations.push(Permutation { modulo: 3, x_factor: 2,  y_factor: 13,  z_factor: 17, });
            permutations.push(Permutation { modulo: 3, x_factor: 5,  y_factor: 13,  z_factor: 17, });

            // let permutation_number = (2u * position_u32_temp.x +  13u * position_u32_temp.y + 17u * position_u32_temp.z) % 4u; 
            // let permutation_number = (2u * position_u32_temp.x +  3u * position_u32_temp.y + 5u * position_u32_temp.z) & 3u; 
            // let permutation_number = (3u * position_u32_temp.x +  5u * position_u32_temp.y + 7u * position_u32_temp.z) & 2u; 
            
            // The DomainTester.
            let domain_tester = DomainTester::init(
                &configuration.device,
                &gpu_debugger,
                [FMM_GLOBAL_X as u32, FMM_GLOBAL_Y as u32, FMM_GLOBAL_Z as u32],
                [FMM_INNER_X as u32, FMM_INNER_Y as u32, FMM_INNER_Z as u32],
                0.15,
                0.016,
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


            let (point_count, buf) = load_pc_data(&configuration.device, &"../../cloud_data.asc".to_string());

            println!("point_count == {}", point_count);

            buffers.insert("pc_data".to_string(), buf);

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
                            vec![
                                create_uniform_bindgroup_layout(0, wgpu::ShaderStages::VERTEX),
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
                                              vec![
                                                  &camera.get_camera_uniform(&configuration.device).as_entire_binding(),
                                             ]
                                         ]
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
                triangle_mesh_bindgroups: triangle_mesh_bindgroups,
                buffers: buffers,
                triangle_meshes: triangle_meshes,
                domain_tester: domain_tester,
                aabb_size: aabb_size,
                font_size: font_size,
                render_object_vvvc: render_object_vvvc,
                render_bind_groups_vvvc: render_bind_groups_vvvc,
                point_count: point_count,
                start_index: start_index,
                step_size: step_size,
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

        // let buf = buffer_from_data::<f32>(
        //           &device,
        //           &vec![1.0,2.0,3.0,2.0,4.0],
        //           wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        //           Some("Computational domain wgpu::buffer.")
        // );

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
        });

        let view = self.screen.surface_texture.as_ref().unwrap().texture.create_view(&wgpu::TextureViewDescriptor::default());

        // let fire_tower = self.triangle_meshes.get(&FIRE_TOWER_MESH.to_string()).unwrap();

        let mut clear = true;

        // draw(&mut encoder,
        //      &view,
        //      self.screen.depth_texture.as_ref().unwrap(),
        //      &self.triangle_mesh_bindgroups,
        //      &self.triangle_mesh_renderer.get_render_object().pipeline,
        //      &fire_tower.get_buffer(),
        //      0..fire_tower.get_triangle_count() * 3, 
        //      clear
        // );

        // println!("{}", self.point_count);

        // if self.start_index + 2 * self.step_size < self.point_count {
        //     self.start_index = self.start_index + self.step_size;
        // }
        // else { self.start_index = 0; }

        draw(&mut encoder,
             &view,
             &self.screen.depth_texture.as_ref().unwrap(),
             &self.render_bind_groups_vvvc,
             &self.render_object_vvvc.pipeline,
             &self.buffers.get("pc_data").unwrap(),
             // self.start_index..self.start_index+self.step_size,
             0..self.point_count,
             clear
        );
        
        clear = false;

        queue.submit(Some(encoder.finish())); 

        // self.gpu_debugger.render(
        //           &device,
        //           &queue,
        //           &view,
        //           self.screen.depth_texture.as_ref().unwrap(),
        //           &mut clear,
        //           spawner
        // );

        // self.gpu_debugger.reset_element_counters(&queue);

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

        if self.keyboard_manager.test_key(&Key::NumpadSubtract, input) { 
            if self.font_size - 0.0005 > 0.0 && self.aabb_size - 0.01 > 0.0 {
                self.font_size = self.font_size - 0.0005;
                self.aabb_size = self.aabb_size - 0.01;
                self.domain_tester.update_font_size(queue, self.font_size);
                self.domain_tester.update_aabb_size(queue, self.aabb_size);
            }
            println!("aabb_size == {:?}", self.aabb_size);
            println!("font_size == {:?}", self.font_size);
        }

        if self.keyboard_manager.test_key(&Key::NumpadAdd, input) {
              self.font_size = self.font_size + 0.0005;
              self.aabb_size = self.aabb_size + 0.01;
              self.domain_tester.update_font_size(queue, self.font_size);
              self.domain_tester.update_aabb_size(queue, self.aabb_size);
              println!("aabb_size == {:?}", self.aabb_size);
              println!("font_size == {:?}", self.font_size);
        }

        let total_grid_count = FMM_GLOBAL_X *
                               FMM_GLOBAL_Y *
                               FMM_GLOBAL_Z *
                               FMM_INNER_X *
                               FMM_INNER_Y *
                               FMM_INNER_Z;

        // let mut encoder = device.create_command_encoder(
        //     &wgpu::CommandEncoderDescriptor {
        //         label: Some("Domain tester encoder"),
        // });

        // self.domain_tester.dispatch(&mut encoder);

        // queue.submit(Some(encoder.finish())); 
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

fn load_pc_data(device: &wgpu::Device,
                src_file: &String) -> (u32, wgpu::Buffer) {

    //let result = read_pc_data(&"../../cloud_data.asc".to_string());
    let result = read_pc_data(src_file);

    (result.len() as u32,
     buffer_from_data::<VVVC>(
        &device,
        &result,
        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        Some("Cloud data buffer")
    ))
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

/// Initialize and create KeyboardManager. Register all the keys that are used in the application.
fn create_keyboard_manager() -> KeyboardManager {

        let mut keys = KeyboardManager::init();

        keys.register_key(Key::NumpadSubtract, 50.0);
        keys.register_key(Key::NumpadAdd, 50.0);
        
        keys
}
