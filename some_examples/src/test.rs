use jaankaup_core::common_functions::udiv_up_safe32;
use jaankaup_algorithms::mc::{McParams, MarchingCubes};
use jaankaup_core::common_functions::{
    create_buffer_bindgroup_layout,
    create_uniform_bindgroup_layout,
    set_bit_to,
    get_bit,
};
use jaankaup_core::pc_parser::VVVC;
use jaankaup_core::pc_parser::read_pc_data;
use itertools::Itertools;
use std::mem::size_of;
use std::borrow::Cow;
use std::collections::HashMap;
use std::convert::TryInto;
use jaankaup_core::input::*;
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
use jaankaup_core::fmm_things::{DomainTester, PointCloud, FmmCellPc, PointCloudHandler, FmmValueFixer};
use bytemuck::{Pod, Zeroable};

/// Max number of arrows for gpu debugger.
const MAX_NUMBER_OF_ARROWS:     usize = 262144;

/// Max number of aabbs for gpu debugger.
const MAX_NUMBER_OF_AABBS:      usize =  TOTAL_INDICES;

/// Max number of box frames for gpu debugger.
const MAX_NUMBER_OF_AABB_WIRES: usize =  40960;

/// Max number of renderable char elements (f32, vec3, vec4, ...) for gpu debugger.
const MAX_NUMBER_OF_CHARS:      usize =  TOTAL_INDICES;

/// Max number of vvvvnnnn vertices reserved for gpu draw buffer.
const MAX_NUMBER_OF_VVVVNNNN: usize =  2000000;

/// Name for the fire tower mesh (assets/models/wood.obj).
const FIRE_TOWER_MESH: &'static str = "FIRE_TOWER";

const CLOUD_DATA: &'static str = "CLOUD_DATA";

/// Global dimensions. 
const FMM_GLOBAL_X: usize = 32; 
const FMM_GLOBAL_Y: usize = 8; 
const FMM_GLOBAL_Z: usize = 32; 

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
                                 size_of::<f32>()) as u32 * 8;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FmmVisualizationParams {
    fmm_global_dimension: [u32; 3],
    visualization_method: u32, // ???.
    fmm_inner_dimension: [u32; 3],
    future_usage: u32,
}

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
    permutation_index: u32,
    point_cloud: PointCloud,
    cell_iterator: [i32; 3],
    camera_mode: bool,
    point_cloud_handler: PointCloudHandler,
    render_params_point_cloud: RenderParamBuffer,
    draw_point_cloud: bool,
    point_cloud_draw_iterator: u32,
    show_domain_tester: bool,
    show_numbers: bool,
    compute_bind_groups_fmm_visualizer: Vec<wgpu::BindGroup>,
    compute_object_fmm_visualizer: ComputeObject,
    fmm_value_fixer: FmmValueFixer,
    marching_cubes: MarchingCubes,
    render_vvvvnnnn: Render_VVVVNNNN_camera,
    render_vvvvnnnn_bg: Vec<wgpu::BindGroup>,
    show_marching_cubes: bool,
    fmm_visualization_params: FmmVisualizationParams,
    once: bool,
}

impl Application for TestProject {

    fn init(configuration: &WGPUConfiguration) -> Self {

        let once = true;
        let show_marching_cubes = false;
        let show_numbers = false;
        let show_domain_tester = false;
        let point_cloud_draw_iterator = 0;
        let draw_point_cloud = true;
        let cell_iterator: [i32; 3] = [0, 0, 0];
        let camera_mode = true;

        let permutation_index = 0;

        let aabb_size: f32 = 0.15;
        let font_size: f32 = 0.016;

        let start_index = 0;
        let step_size = 1024;

        // Log adapter info.
        log_adapter_info(&configuration.adapter);

        // Camera.
        let mut camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (0.0, 30.0, 10.0), -89.0, 0.0);
        camera.set_rotation_sensitivity(0.4);
        camera.set_movement_sensitivity(0.1);

        // Gpu debugger.
        let gpu_debugger = create_gpu_debugger( &configuration.device, &configuration.sc_desc, &mut camera);

        // Gpu timer.
        let gpu_timer = GpuTimer::init(&configuration.device, &configuration.queue, 8, Some("gpu timer"));

        // Keyboard manager. Keep tract of keys which has been pressed, and for how long time.
        let mut keyboard_manager = create_keyboard_manager();

        // Light source for triangle meshes.
        let light = LightBuffer::create(
                      &configuration.device,
                      [25.0, 75.0, 25.0], // pos
                      [25, 25, 130],  // spec
                      [255,200,255], // light 
                      55.0,
                      0.35,
                      0.000013
        );

        // Scale_factor for triangle meshes.
        let render_params = RenderParamBuffer::create(
                    &configuration.device,
                    4.0
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

        // Generate the point cloud.
        let point_cloud = PointCloud::init(&configuration.device, &"../../cloud_data.asc".to_string());
        println!("Point cloud generated.");

        println!("Generate fmm sample data.");
        let pc_sample_data = 
            buffers.insert(
                "pc_sample_data".to_string(),
                buffer_from_data::<FmmCellPc>(
                &configuration.device,
                &vec![FmmCellPc {
                    tag: 0,
                    value: 10000000,
                    color: 0,
                    // padding: 0,
                } ; FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z * FMM_INNER_X * FMM_INNER_Y * FMM_INNER_Z],
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                None)
            );

        let pc_min_coord = point_cloud.get_min_coord();
        let pc_max_coord = point_cloud.get_max_coord();

        println!("pc_min_coord == {:?}", pc_min_coord);
        println!("pc_max_coord == {:?}", pc_max_coord);

        let point_cloud_scale_factor_x = (FMM_GLOBAL_X * FMM_INNER_X) as f32 / pc_max_coord[0];
        let point_cloud_scale_factor_y = (FMM_GLOBAL_Y * FMM_INNER_Y) as f32 / pc_max_coord[1];
        let point_cloud_scale_factor_z = (FMM_GLOBAL_Z * FMM_INNER_Z) as f32 / pc_max_coord[2];

        println!("FMM_GLOBAL_X * FMM_INNER_X == {:?}", (FMM_GLOBAL_X * FMM_INNER_X) as f32);
        println!("FMM_GLOBAL_Y * FMM_INNER_X == {:?}", (FMM_GLOBAL_Y * FMM_INNER_Z) as f32);
        println!("FMM_GLOBAL_Z * FMM_INNER_X == {:?}", (FMM_GLOBAL_Y * FMM_INNER_Z) as f32);
        
        println!("point_cloud_scale_factor_x == {}", point_cloud_scale_factor_x);
        println!("point_cloud_scale_factor_y == {}", point_cloud_scale_factor_y);
        println!("point_cloud_scale_factor_z == {}", point_cloud_scale_factor_z);

        let pc_scale_factor = point_cloud_scale_factor_x.min(point_cloud_scale_factor_y).min(point_cloud_scale_factor_z);

        println!("pc_scale_factor == {}", pc_scale_factor);

        let render_params_point_cloud = RenderParamBuffer::create(
                    &configuration.device,
                    pc_scale_factor
        );

        let point_cloud_handler = PointCloudHandler::init(
                &configuration.device,
                [FMM_GLOBAL_X as u32, FMM_GLOBAL_Y as u32, FMM_GLOBAL_Z as u32],
                [FMM_INNER_X as u32, FMM_INNER_Y as u32, FMM_INNER_Z as u32],
                point_cloud.get_point_count(),
                pc_min_coord,
                pc_max_coord,
                pc_scale_factor, // scale_factor
                point_cloud_draw_iterator,
                show_numbers,
                &buffers.get("pc_sample_data").unwrap(),
                point_cloud.get_buffer(),
                &gpu_debugger
        );

        // The DomainTester.
        let domain_tester = DomainTester::init(
            &configuration.device,
            &gpu_debugger,
            [FMM_GLOBAL_X as u32, FMM_GLOBAL_Y as u32, FMM_GLOBAL_Z as u32],
            [FMM_INNER_X as u32, FMM_INNER_Y as u32, FMM_INNER_Z as u32],
            0.15,
            0.016,
            show_numbers,
            [cell_iterator[0] as u32, cell_iterator[1] as u32, cell_iterator[2] as u32],
            );

        // Fmm scene visualizer parasm
        let fmm_visualization_params = 
                 FmmVisualizationParams {
                     fmm_global_dimension: [FMM_GLOBAL_X as u32, FMM_GLOBAL_Y as u32, FMM_GLOBAL_Z as u32],
                     visualization_method: 4, // ???.
                     //visualization_method: 1 | 2 | 4, // ???.
                     fmm_inner_dimension: [FMM_INNER_X as u32, FMM_INNER_Y as u32, FMM_INNER_Z as u32],
                     future_usage: 0,
        };

        buffers.insert(
            "fmm_visualization_params".to_string(),
            buffer_from_data::<FmmVisualizationParams>(
            &configuration.device,
            &vec![fmm_visualization_params],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            None)
        );

        let fmm_value_fixer = FmmValueFixer::init(&configuration.device,
                                                  &buffers.get(&"fmm_visualization_params".to_string()).unwrap(),
                                                  &buffers.get(&"pc_sample_data".to_string()).unwrap()
        );

        // The fmm scene visualizer.
        let compute_object_fmm_visualizer =
                ComputeObject::init(
                    &configuration.device,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("fmm_data_visualizer.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/fmm_data_visualizer.wgsl"))),
                    
                    }),
                    Some("Fmm data visualizer compute object"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> fmm_params: FmmVisualizationParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<storage, read_write> fmm_data: array<FmmCellPc>;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // // @group(0) @binding(2) var<storage, read_write> fmm_blocks: array<FmmBlock>;
                            // create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                            // // @group(0) @binding(3) var<storage, read_write> isotropic_data: array<f32>;
                            // create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

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
                    ],
                    &"main".to_string()
        );

        let compute_bind_groups_fmm_visualizer =
            create_bind_groups(
                &configuration.device,
                &compute_object_fmm_visualizer.bind_group_layout_entries,
                &compute_object_fmm_visualizer.bind_group_layouts,
                &vec![
                    vec![
                         &buffers.get(&"fmm_visualization_params".to_string()).unwrap().as_entire_binding(),
                         &buffers.get(&"pc_sample_data".to_string()).unwrap().as_entire_binding(),
                         // &buffers.get(&"fmm_blocks".to_string()).unwrap().as_entire_binding(),
                         // &buffers.get(&"isotropic_data".to_string()).unwrap().as_entire_binding(),
                         &gpu_debugger.get_element_counter_buffer().as_entire_binding(),
                         &gpu_debugger.get_output_chars_buffer().as_entire_binding(),
                         &gpu_debugger.get_output_arrows_buffer().as_entire_binding(),
                         &gpu_debugger.get_output_aabbs_buffer().as_entire_binding(),
                         &gpu_debugger.get_output_aabb_wires_buffer().as_entire_binding(),
                    ]
                ]
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

        let point_count = 0;

        // vvvc
        let render_object_vvvc =
                RenderObject::init(
                    &configuration.device,
                    &configuration.sc_desc,
                    &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("renderer_v3c1_x4.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/renderer_v3c1_x4.wgsl"))),

                    }),
                    &vec![wgpu::VertexFormat::Float32x3, wgpu::VertexFormat::Uint32],
                    &vec![
                        vec![
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::VERTEX),
                            create_uniform_bindgroup_layout(1, wgpu::ShaderStages::VERTEX),
                        ],
                    ],
                    Some("Debug visualizator vvvc x4 renderer with camera."),
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
                                              &render_params_point_cloud.get_buffer().as_entire_binding(),
                                         ]
                                     ]
        );


        // MC Cubes stuff.
          
         let mc_params = McParams {
                 base_position: [0.0, 0.0, 0.0, 1.0],
                 isovalue: 0.0,
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
                 size: OUTPUT_BUFFER_SIZE as u64, 
                 usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                 mapped_at_creation: false,
             })
         );

         let mc_shader = &configuration.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                         label: Some("mc compute shader"),
                         source: wgpu::ShaderSource::Wgsl(
                             Cow::Borrowed(include_str!("../../assets/shaders/mc_pc_data.wgsl"))),
                         }
         );

         let mc_instance = MarchingCubes::init_with_noise_buffer(
             &configuration.device,
             &mc_params,
             &mc_shader,
             &buffers.get(&"pc_sample_data".to_string()).unwrap(),
             &buffers.get(&"mc_output".to_string()).unwrap(),
         );

        // vvvvnnnn renderer.
        let render_vvvvnnnn = Render_VVVVNNNN_camera::init(&configuration.device, &configuration.sc_desc);
        let render_vvvvnnnn_bg = render_vvvvnnnn.create_bingroups(&configuration.device, &mut camera, &light, &render_params);

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
            //permutation_count: permutation_count as u32,
            point_count: point_count,
            permutation_index: permutation_index,
            point_cloud: point_cloud,
            cell_iterator: cell_iterator,
            camera_mode: camera_mode,
            point_cloud_handler: point_cloud_handler,
            render_params_point_cloud: render_params_point_cloud,
            draw_point_cloud: draw_point_cloud,
            point_cloud_draw_iterator: point_cloud_draw_iterator,
            show_domain_tester: show_domain_tester,
            show_numbers: show_numbers,
            compute_bind_groups_fmm_visualizer: compute_bind_groups_fmm_visualizer,
            compute_object_fmm_visualizer: compute_object_fmm_visualizer,
            fmm_value_fixer: fmm_value_fixer,
            marching_cubes: mc_instance,
            render_vvvvnnnn: render_vvvvnnnn,
            render_vvvvnnnn_bg: render_vvvvnnnn_bg,
            show_marching_cubes: show_marching_cubes,
            fmm_visualization_params: fmm_visualization_params,
            once: once,
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
    
        let fire_tower = self.triangle_meshes.get(&FIRE_TOWER_MESH.to_string()).unwrap();
    
        let mut clear = true;
    
        draw(&mut encoder,
             &view,
             self.screen.depth_texture.as_ref().unwrap(),
             &self.triangle_mesh_bindgroups,
             &self.triangle_mesh_renderer.get_render_object().pipeline,
             &fire_tower.get_buffer(),
             0..3,
             //0..fire_tower.get_triangle_count() * 3, 
             clear
        );
    
        //++ clear = false;
        // println!("{}", self.point_count);
    
        if self.draw_point_cloud {
    
            draw(&mut encoder,
                 &view,
                 &self.screen.depth_texture.as_ref().unwrap(),
                 &self.render_bind_groups_vvvc,
                 &self.render_object_vvvc.pipeline,
                 &self.point_cloud.get_buffer(),
                 0..self.point_cloud.get_point_count(),
                 clear
            );
    
            clear = false;
        }

         if self.show_marching_cubes {
             draw_indirect(
                  &mut encoder,
                  &view,
                  self.screen.depth_texture.as_ref().unwrap(),
                  &self.render_vvvvnnnn_bg,
                  &self.render_vvvvnnnn.get_render_object().pipeline,
                  &self.buffers.get("mc_output").unwrap(),
                  self.marching_cubes.get_draw_indirect_buffer(),
                  0,
                  clear
             );

             clear = false;
         }
    
        queue.submit(Some(encoder.finish())); 
    
    
        self.gpu_debugger.render(
                  &device,
                  &queue,
                  &view,
                  self.screen.depth_texture.as_ref().unwrap(),
                  &mut clear,
                  spawner
        );
    
        self.gpu_debugger.reset_element_counters(&queue);
    
        self.screen.prepare_for_rendering();

        // Reset counter.
        // self.marching_cubes.reset_counter_value(device, queue);
    }
    
    #[allow(unused)]
    fn input(&mut self, queue: &wgpu::Queue, input: &InputCache) {
        if self.camera_mode { self.camera.update_from_input(&queue, &input); } 
        else {
            let mut temp_coordinate = self.cell_iterator; 
            if self.keyboard_manager.test_key(&Key::A, input) { temp_coordinate[0] = temp_coordinate[0] - 1; }
            if self.keyboard_manager.test_key(&Key::D, input) { temp_coordinate[0] = temp_coordinate[0] + 1; }
            if self.keyboard_manager.test_key(&Key::W, input) { temp_coordinate[2] = temp_coordinate[2] - 1; }
            if self.keyboard_manager.test_key(&Key::S, input) { temp_coordinate[2] = temp_coordinate[2] + 1; }
            if self.keyboard_manager.test_key(&Key::E, input) { temp_coordinate[1] = temp_coordinate[1] + 1; }
            if self.keyboard_manager.test_key(&Key::C, input) { temp_coordinate[1] = temp_coordinate[1] - 1; }
            let dimension = [FMM_GLOBAL_X as i32 * FMM_INNER_X as i32, 
                             FMM_GLOBAL_Y as i32 * FMM_INNER_Y as i32, 
                             FMM_GLOBAL_Z as i32 * FMM_INNER_Z as i32]; 
            if temp_coordinate[0] >= 0 && temp_coordinate[1] >= 0 && temp_coordinate[2] >= 0 &&
               temp_coordinate[0] < dimension[0] && temp_coordinate[1] < dimension[1] && temp_coordinate[2] < dimension[2] {
                   if self.cell_iterator != temp_coordinate { println!("cell_iterator == {:?}", temp_coordinate); }
                   self.cell_iterator = temp_coordinate; 
               }
        }


        // Fmm visualizer.  

        // Far.
        if self.keyboard_manager.test_key(&Key::Key1, input) {
            let bit = if get_bit(self.fmm_visualization_params.visualization_method, 0) == 0 { true } else { false };
            self.fmm_visualization_params.visualization_method = set_bit_to(self.fmm_visualization_params.visualization_method, 0, bit);
        }

        // Band.
        if self.keyboard_manager.test_key(&Key::Key2, input) {
            let bit = if get_bit(self.fmm_visualization_params.visualization_method, 1) == 0 { true } else { false };
            self.fmm_visualization_params.visualization_method = set_bit_to(self.fmm_visualization_params.visualization_method, 1, bit);
        }

        // Known.
        if self.keyboard_manager.test_key(&Key::Key3, input) {
            let bit = if get_bit(self.fmm_visualization_params.visualization_method, 2) == 0 { true } else { false };
            self.fmm_visualization_params.visualization_method = set_bit_to(self.fmm_visualization_params.visualization_method, 2, bit);
        }
        
        // Numbers.
        if self.keyboard_manager.test_key(&Key::N, input) {
            let bit = if get_bit(self.fmm_visualization_params.visualization_method, 6) == 0 { true } else { false };
            self.fmm_visualization_params.visualization_method = set_bit_to(self.fmm_visualization_params.visualization_method, 6, bit);
        }

        queue.write_buffer(
            &self.buffers.get(&"fmm_visualization_params".to_string()).unwrap(),
            0,
            bytemuck::cast_slice(&[self.fmm_visualization_params])
        );
        // if self.keys.test_key(&Key::Key4, input) {
        //     let bit = if get_bit(self.fmm_visualization_params.visualization_method, 3) == 0 { true } else { false };
        //     self.fmm_visualization_params.visualization_method = set_bit_to(self.fmm_visualization_params.visualization_method, 3, bit);
        // }
    
        // if self.keyboard_manager.test_key(&Key::Key0, input) {
        //     if self.point_cloud_draw_iterator * 1024 + 1024 < self.point_cloud.get_point_count() {
        //         self.point_cloud_draw_iterator = self.point_cloud_draw_iterator + 1;
        //         self.point_cloud_handler.update_thread_group_number(queue, self.point_cloud_draw_iterator);
        //         println!("Updating fmm interface from range ({:?} .. {:?} < {:?}",
        //                       self.point_cloud_draw_iterator * 1024,
        //                       self.point_cloud_draw_iterator * 1024 + 1024,
        //                       self.point_cloud.get_point_count()
        //         );
        //     }
        // }
    
        // if self.keyboard_manager.test_key(&Key::Key9, input) {
        //         if self.point_cloud_draw_iterator != 0 {
        //             self.point_cloud_draw_iterator = self.point_cloud_draw_iterator - 1;
        //             self.point_cloud_handler.update_thread_group_number(queue, self.point_cloud_draw_iterator);
        //         }
        // }
    
        if self.keyboard_manager.test_key(&Key::Return, input) {
            self.show_domain_tester = !self.show_domain_tester;
        }
    
        if self.keyboard_manager.test_key(&Key::N, input) {
            self.show_numbers = !self.show_numbers;
            self.domain_tester.update_show_numbers(queue, self.show_numbers);
        }
         if self.keyboard_manager.test_key(&Key::M, input) {
             self.show_marching_cubes = !self.show_marching_cubes;
         }
    }
    
    fn resize(&mut self, device: &wgpu::Device, sc_desc: &wgpu::SurfaceConfiguration, _new_size: winit::dpi::PhysicalSize<u32>) {
    
        // TODO: add this functionality to the Screen.
        self.screen.depth_texture = Some(Texture::create_depth_texture(&device, &sc_desc, Some("depth-texture")));
        self.camera.resize(sc_desc.width as f32, sc_desc.height as f32);
    }
    
    #[allow(unused)]
    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, input: &InputCache, spawner: &Spawner) {
    
        self.domain_tester.update_domain_iterator(queue, [self.cell_iterator[0] as u32, self.cell_iterator[1] as u32, self.cell_iterator[2] as u32] );
    
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
    
        if self.keyboard_manager.test_key(&Key::P, input) {
              self.camera_mode = true;
              println!("camera mode == {:?}", self.camera_mode);
              // log::info!("camera mode == {:?}", self.camera_mode);
        }
    
        if self.keyboard_manager.test_key(&Key::O, input) {
              self.camera_mode = false;
              println!("camera mode == {:?}", self.camera_mode);
              //log::info!("camera mode == {:?}", self.camera_mode);
        }
    
        if self.keyboard_manager.test_key(&Key::Space, input) {
              self.draw_point_cloud = !self.draw_point_cloud;
              println!("draw point cloud == {:?}", self.draw_point_cloud);
        }
    
        let total_grid_count = FMM_GLOBAL_X *
                               FMM_GLOBAL_Y *
                               FMM_GLOBAL_Z *
                               FMM_INNER_X *
                               FMM_INNER_Y *
                               FMM_INNER_Z;
    
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Domain tester encoder"),
        });
    
        if self.show_domain_tester {
            self.domain_tester.dispatch(&mut encoder);
        }

        self.point_cloud_handler.point_data_to_interface(&mut encoder);
        self.fmm_value_fixer.dispatch(&mut encoder,
                                      [udiv_up_safe32(self.point_cloud.get_point_count(), 1024), 1, 1]);
        // println!("udiv_up_safe32({}, 1024, 1, 1) == {}", self.point_cloud.get_point_count(),  udiv_up_safe32(self.point_cloud.get_point_count(), 1024));

        self.compute_object_fmm_visualizer.dispatch(
            &self.compute_bind_groups_fmm_visualizer,
            &mut encoder,
            (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z) as u32, 1, 1,
            Some("fmm visualizer dispatch")
        );

         if self.once {
             self.marching_cubes.dispatch(&mut encoder, total_grid_count as u32 / 256, 1, 1);
             self.once = !self.once;
         }
    
        queue.submit(Some(encoder.finish())); 
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

// fn load_pc_data(device: &wgpu::Device,
//                 src_file: &String) -> (u32, wgpu::Buffer) {
// 
//     //let result = read_pc_data(&"../../cloud_data.asc".to_string());
//     let result = read_pc_data(src_file);
// 
//     (result.len() as u32,
//      buffer_from_data::<VVVC>(
//         &device,
//         &result,
//         wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
//         Some("Cloud data buffer")
//     ))
// }

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

        keys.register_key(Key::NumpadSubtract, 10.0);
        keys.register_key(Key::NumpadAdd, 10.0);
        keys.register_key(Key::O, 50.0);
        keys.register_key(Key::P, 50.0);
        keys.register_key(Key::A, 50.0);
        keys.register_key(Key::S, 50.0);
        keys.register_key(Key::D, 50.0);
        keys.register_key(Key::W, 50.0);
        keys.register_key(Key::C, 50.0);
        keys.register_key(Key::E, 50.0);
        keys.register_key(Key::Space, 100.0);
        keys.register_key(Key::Key9, 5.0);
        keys.register_key(Key::Key0, 5.0);
        keys.register_key(Key::Return, 50.0);
        keys.register_key(Key::Key1, 50.0);
        keys.register_key(Key::Key2, 50.0);
        keys.register_key(Key::Key3, 50.0);
        keys.register_key(Key::Key4, 50.0);
        keys.register_key(Key::M, 50.0);
        keys.register_key(Key::N, 50.0);
        
        keys
}
