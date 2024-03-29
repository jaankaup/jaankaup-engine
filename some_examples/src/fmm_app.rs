use std::env;
use std::num::NonZeroU32;
use jaankaup_core::two_triangles::TwoTriangles;
// use jaankaup_core::fmm_things::FmmBlock;
use jaankaup_core::fmm_things::PointCloudParamsBuffer;
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
use jaankaup_core::common_functions::{
    udiv_up_safe32,
    create_uniform_bindgroup_layout,
    encode_rgba_u32,
};

#[cfg(feature = "gpu_debug")]
use jaankaup_core::common_functions::{
    set_bit_to,
    get_bit,
};

#[cfg(feature = "gpu_debug")]
use jaankaup_core::common_functions::{
    create_buffer_bindgroup_layout,
};

#[derive(PartialEq)]
enum CameraMode {
    Camera,
    RayCamera,
}

use jaankaup_core::{wgpu, log};
use jaankaup_core::winit;
use jaankaup_core::buffer::{buffer_from_data};
use jaankaup_core::model_loader::{TriangleMesh, create_from_bytes};
use jaankaup_core::camera::Camera;
#[cfg(feature = "gpu_debug")]
use jaankaup_core::gpu_debugger::GpuDebugger;
use jaankaup_core::gpu_timer::GpuTimer;
use jaankaup_core::screen::ScreenTexture;
use jaankaup_core::shaders::{RenderVvvvnnnnCamera};
use jaankaup_core::render_things::{LightBuffer, RenderParamBuffer};
use jaankaup_core::render_object::{draw, RenderObject, create_bind_groups};
#[cfg(feature = "gpu_debug")]
use jaankaup_core::render_object::ComputeObject;
use jaankaup_core::texture::Texture;
use jaankaup_core::fmm_things::PointCloud;
// use jaankaup_core::fast_marching_method::{FastMarchingMethod, FmmState};
use jaankaup_core::fast_iterative_method::{FastIterativeMethod};
use jaankaup_core::sphere_tracer::{SphereTracer, SphereTracerParams};
use bytemuck::{Pod, Zeroable};

/// Max number of arrows for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_ARROWS:     usize = 262144 * 16;

/// Max number of aabbs for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_AABBS:      usize = TOTAL_INDICES;

/// Max number of box frames for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_AABB_WIRES: usize = 262144;

/// Max number of renderable char elements (f32, vec3, vec4, ...) for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_CHARS:      usize = TOTAL_INDICES;

/// Max number of vvvvnnnn vertices reserved for gpu draw buffer.
#[allow(dead_code)]
const MAX_NUMBER_OF_VVVVNNNN: usize = 4000000;

#[allow(dead_code)]
//const TOTAL_INDICES: usize = 64*16*54*4*4*4; // FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z * FMM_INNER_X * FMM_INNER_Y * FMM_INNER_Z; 
const TOTAL_INDICES: usize = 32*16*32*4*4*4; // FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z * FMM_INNER_X * FMM_INNER_Y * FMM_INNER_Z; 

/// Name for the fire tower mesh (assets/models/wood.obj).
//const FIRE_TOWER_MESH: &'static str = "FIRE_TOWER";

// const FMM_GLOBAL_X: usize = 32;
// const FMM_GLOBAL_Y: usize = 8;
// const FMM_GLOBAL_Z: usize = 27;
// const FMM_GLOBAL_X: usize = 62;
// const FMM_GLOBAL_Y: usize = 16;
// const FMM_GLOBAL_Z: usize = 54;
const FMM_GLOBAL_X: usize = 70;
const FMM_GLOBAL_Y: usize = 18;
const FMM_GLOBAL_Z: usize = 60;
// const FMM_GLOBAL_X: usize = 82;
// const FMM_GLOBAL_Y: usize = 17;
// const FMM_GLOBAL_Z: usize = 58;
// const FMM_GLOBAL_X: usize = 100;
// const FMM_GLOBAL_Y: usize = 22;
// const FMM_GLOBAL_Z: usize = 76;
// const FMM_GLOBAL_X: usize = 116;
// const FMM_GLOBAL_Y: usize = 25;
// const FMM_GLOBAL_Z: usize = 89;

const FMM_INNER_X: usize = 4;
const FMM_INNER_Y: usize = 4;
const FMM_INNER_Z: usize = 4;

struct AppRenderParams {
    draw_point_cloud: bool,
    visualization_method: u32,
    _step: bool,
    _show_numbers: bool,
    update: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct FmmVisualizationParams {
    fmm_global_dimension: [u32; 3],
    visualization_method: u32, // ???.
    fmm_inner_dimension: [u32; 3],
    future_usage: u32,
}

/// Features and limits for FmmApp application.
struct FmmAppFeatures {}

impl WGPUFeatures for FmmAppFeatures {

    fn optional_features() -> wgpu::Features {
        wgpu::Features::TIMESTAMP_QUERY
    }

    fn required_features() -> wgpu::Features {

        // #[cfg(not(target_arch = "wasm32"))] {
        if cfg!(not(target_arch = "wasm32")) {
            wgpu::Features::PUSH_CONSTANTS |
            wgpu::Features::WRITE_TIMESTAMP_INSIDE_PASSES
        }
        else {
        // #[cfg(target_arch = "wasm32"))] {
            wgpu::Features::empty()
        }
    }

    fn required_limits() -> wgpu::Limits {
        let mut limits = wgpu::Limits::default();

        #[cfg(not(target_arch = "wasm32"))]
        {
        limits.max_compute_invocations_per_workgroup = 1024;
        }

        #[cfg(not(target_arch = "wasm32"))] 
        {
        limits.max_compute_workgroup_size_x = 1024;
        }

        limits.max_storage_buffers_per_shader_stage = 10;

        #[cfg(not(target_arch = "wasm32"))] 
        {
        limits.max_push_constant_size = 4; // Now useless.
        }

        #[cfg(not(target_arch = "wasm32"))] 
        {
        limits.max_push_constant_size = 4; // Now useless.
        }

        //limits.max_compute_workgroup_size_x = 65536 * 2;
        //limits.max_storage_buffer_binding_size = 436101120; 
        //limits.max_storage_buffer_binding_size = 256819200; 
        limits.max_storage_buffer_binding_size = 396441600; 
        //limits.max_storage_buffer_binding_size = 154275840; 
        // limits.max_compute_workgroups_per_dimension = 65536 * 2;
        // println!("limits.max_storage_buffer_binding_size == {}", limits.max_storage_buffer_binding_size);
        limits
    }
}

/// FmmApp solver. A Fast marching method (GPU) for solving the eikonal equation.
struct FmmApp {
    camera: Camera,
    ray_camera: Camera,
    camera_mode: CameraMode,
    #[cfg(feature = "gpu_debug")]
    gpu_debugger: Option<GpuDebugger>,
    gpu_timer: Option<GpuTimer>,
    keyboard_manager: KeyboardManager,
    screen: ScreenTexture, 
    _light: LightBuffer,
    _render_params: RenderParamBuffer,
    _triangle_mesh_renderer: RenderVvvvnnnnCamera,
    _triangle_mesh_bindgroups: Vec<wgpu::BindGroup>,
    buffers: HashMap<String, wgpu::Buffer>,
    point_cloud: PointCloud,
    #[cfg(feature = "gpu_debug")]
    compute_bind_groups_fmm_visualizer: Option<Vec<wgpu::BindGroup>>,
    #[cfg(feature = "gpu_debug")]
    compute_object_fmm_visualizer: ComputeObject,
    render_vvvvnnnn: RenderVvvvnnnnCamera,
    render_vvvvnnnn_bg: Vec<wgpu::BindGroup>,
    render_object_vvvc: RenderObject,
    render_bind_groups_vvvc: Vec<wgpu::BindGroup>,
    app_render_params: AppRenderParams,
    // fmm: FastMarchingMethod,
    _pc_params: PointCloudParamsBuffer,
    once: bool,
    sphere_tracer: SphereTracer,
    sphere_tracer_renderer: TwoTriangles,
    sphere_tracer_params: SphereTracerParams,
    draw_two_triangles: bool,
    fim: FastIterativeMethod,
}

impl Application for FmmApp {

    fn init(configuration: &WGPUConfiguration) -> Self {

        // Log adapter info.
        // log_adapter_info(&configuration.adapter);

        let args: Vec<String> = env::args().collect();
        if args.len() < 2 { panic!("You must pass the point cloud data file as argument!"); }
        else {
            log::info!("The point cloud data file: {:?}", &args[1]); 
        }

        let once = true;
        // Buffer hash_map.
        let mut buffers: HashMap<String, wgpu::Buffer> = HashMap::new();

        // Camera.
        let mut camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (180.0, 130.0, 480.0), -89.0, -4.0);
        camera.set_rotation_sensitivity(0.4);
        camera.set_movement_sensitivity(0.2);

        let swap_camera_x = -1.0;
        let swap_camera_z = -1.0;

        //let cube_dim = [100.0, 50.0, 100.0];
        let cube_dim = [50.0, 20.0, 50.0];
        let cube_min = [120.0, 40.0, 180.0];
        // let cube_min = [220.0, 40.0, 280.0];
        let cube_max = [cube_min[0] + cube_dim[0],
                        cube_dim[1] + cube_min[1],
                        cube_dim[2] + cube_min[2]];

        let center   = [cube_min[0] + cube_dim[0] * 0.5,
                        cube_min[1] + cube_dim[1] * 0.5,
                        cube_min[2] + cube_dim[2] * 0.5];


        //let corner_lookat = [cube_min[0], (cube_min[1] + cube_dim[1]) * 0.75, cube_min[2] + cube_dim[2]]; 
        //let corner_lookat = [cube_min[0] + cube_dim[0], (cube_min[1] + cube_dim[1]) * 0.75, cube_min[2] + cube_dim[2]]; 
        let corner_lookat = [center[0], center[1], cube_min[2] + cube_dim[2]]; 


        let mut ray_camera = Camera::new(1024.0,
                                         1536.0,
                                         (corner_lookat[0],
                                          corner_lookat[1],
                                          corner_lookat[2]),
                                         -49.0, -18.0);
        //let mut ray_camera = Camera::new(configuration.size.width as f32,
        //                                 configuration.size.height as f32,
        //                                 (corner_lookat[0],
        //                                  corner_lookat[1],
        //                                  corner_lookat[2]),
        //                                 -49.0, -18.0);

        //let mut ray_camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (185.0 + movement[0], 76.0 + movement[1], 315.0 + movement[2]), -49.0, -18.0);
        //let mut ray_camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (180.0, 120.0, 500.0), -89.0, -20.0);
        //++let mut ray_camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (440.0, 110.0, 165.0), -171.0, -23.0);
        ray_camera.set_rotation_sensitivity(0.4);
        ray_camera.set_movement_sensitivity(0.05);
        ray_camera.set_lookat([center[0],
                               center[1],
                               center[2]], &configuration.queue);
        // ray_camera.set_lookat([center[0],
        //                        center[1],
        //                        center[2]], &configuration.queue);

        //ray_camera.move_forward(-1.0, &configuration.queue);

        println!("GRADU coord == ({:?},{:?},{:?})",
                  // camera.get_view()[0] * 1.0 + camera.get_position()[0],
                  // camera.get_view()[1] * 1.0 + camera.get_position()[1],
                  // camera.get_view()[2] * 1.0 + camera.get_position()[2]
                  ray_camera.get_position()[0],
                  ray_camera.get_position()[1],
                  ray_camera.get_position()[2]
        );

        let camera_mode = CameraMode::RayCamera;

        ray_camera.set_focal_distance(6.0, &configuration.queue);

        let app_render_params = AppRenderParams {
             draw_point_cloud: true,
             visualization_method: 0,
             _step: false,
             _show_numbers: false,
             update: false,
        };

        #[cfg(feature = "gpu_debug")]
        let gpu_debugger: Option<GpuDebugger> = create_gpu_debugger(&configuration.device, &configuration.sc_desc, &mut camera).into();

        #[cfg(not(feature = "gpu_debug"))]
        let gpu_debugger = None;

        let mut gpu_timer = GpuTimer::init(&configuration.device, &configuration.queue, 8, Some("gpu timer"));

        // Keyboard manager. Keep tract of keys which has been pressed, and for how long time.
        let keyboard_manager = create_keyboard_manager();

        let scene_x = (FMM_GLOBAL_X * FMM_INNER_X) as f32;
        let scene_y = (FMM_GLOBAL_Y * FMM_INNER_Y) as f32;
        let scene_z = (FMM_GLOBAL_Z * FMM_INNER_Z) as f32;

        // Light source for triangle meshes.
        let light = LightBuffer::create(
                      &configuration.device,
                      [scene_x * 0.5, scene_y * 1.75, scene_z * 0.5], // pos
                      [25, 25, 130],  // spec
                      [255,200,255], // light 
                      255.0,
                      0.35,
                      0.00013
        );
        // let light = LightBuffer::create(
        //               &configuration.device,
        //               [0.0, 87.0* 1.25, -50.0], // pos
        //               [25, 25, 130],  // spec
        //               [255,200,255], // light 
        //               255.0,
        //               0.35,
        //               0.000013
        // );

        // Scale_factor for triangle meshes.
        let render_params = RenderParamBuffer::create(
                    &configuration.device,
                    1.0
        );

        // A dummy buffer for rendering (nothing). 
        let dummy_buffer = buffer_from_data::<f32>(
            &configuration.device,
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            wgpu::BufferUsages::VERTEX,
            None
        );
        buffers.insert("dummy_buffer".to_string(), dummy_buffer);

        // vvvvnnnn renderer.
        let render_vvvvnnnn = RenderVvvvnnnnCamera::init(&configuration.device, &configuration.sc_desc);
        let render_vvvvnnnn_bg = render_vvvvnnnn.create_bingroups(&configuration.device, &mut camera, &light, &render_params);

        // vvvc
        let render_object_vvvc =
                RenderObject::init(
                    &configuration.device,
                    &configuration.sc_desc,
                    &configuration.device.create_shader_module(wgpu::ShaderModuleDescriptor {
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

        let global_dimension = [FMM_GLOBAL_X as u32, FMM_GLOBAL_Y as u32, FMM_GLOBAL_Z as u32];
        let local_dimension = [FMM_INNER_X as u32, FMM_INNER_Y as u32, FMM_INNER_Z as u32];
        let render_params = RenderParamBuffer::create(
                    &configuration.device,
                    4.0
        );

        // RenderObject for basic triangle mesh rendering.
        let triangle_mesh_renderer = RenderVvvvnnnnCamera::init(&configuration.device, &configuration.sc_desc);

        // Create bindgroups for triangle_mesh_renderer.
        let triangle_mesh_bindgroups = 
                triangle_mesh_renderer.create_bingroups(
                    &configuration.device, &mut camera, &light, &render_params
                );


        // Generate the point cloud.

        #[cfg(target_arch = "wasm32")]
        let point_cloud = PointCloud::init(&configuration.device, &"../../cloud_data.asc".to_string(), scene_x, scene_y, scene_z);

        #[cfg(not(target_arch = "wasm32"))]
        let point_cloud = PointCloud::init(&configuration.device, &args[1], scene_x, scene_y, scene_z);

        let pc_max_coord = point_cloud.get_max_coord();

        // println!("point_cloud.get_max_coord() == {:?}. point_cloud.get_min_coord() == {:?}", point_cloud.get_max_coord(), point_cloud.get_min_coord());

        // Create scale factor for point data so it fits under computation domain ranges.
        let point_cloud_scale_factor_x = (FMM_GLOBAL_X * FMM_INNER_X) as f32 / pc_max_coord[0];
        let point_cloud_scale_factor_y = (FMM_GLOBAL_Y * FMM_INNER_Y) as f32 / pc_max_coord[1];
        let point_cloud_scale_factor_z = (FMM_GLOBAL_Z * FMM_INNER_Z) as f32 / pc_max_coord[2];

        // let point_cloud_scale_factor_x = pc_max_coord[0] / (FMM_GLOBAL_X * FMM_INNER_X) as f32;
        // let point_cloud_scale_factor_y = pc_max_coord[1] / (FMM_GLOBAL_Y * FMM_INNER_Y) as f32;
        // let point_cloud_scale_factor_z = pc_max_coord[2] / (FMM_GLOBAL_Z * FMM_INNER_Z) as f32;

        // println!("point_cloud_scale_factor_x == {}", point_cloud_scale_factor_x);
        // println!("point_cloud_scale_factor_y == {}", point_cloud_scale_factor_y);
        // println!("point_cloud_scale_factor_z == {}", point_cloud_scale_factor_z);
        // println!("point_cloud.get_max_coord() == {:?}", point_cloud.get_max_coord());

        // #[cfg(feature = "gpu_debug")]
        // let pc_scale_factor = point_cloud_scale_factor_x.min(point_cloud_scale_factor_y).min(point_cloud_scale_factor_z);

        // println!("pc_scale_factor == {}", pc_scale_factor);

        let pc_params = PointCloudParamsBuffer::create(
            &configuration.device,
            point_cloud.get_point_count(),
            point_cloud.get_min_coord(),
            point_cloud.get_max_coord(),
            //[1.0, 1.0, 1.0],
            [point_cloud_scale_factor_x, point_cloud_scale_factor_y, point_cloud_scale_factor_z],
            //pc_scale_factor,
            123, // useless
            false); // useless

        let render_params_point_cloud = RenderParamBuffer::create(
                    &configuration.device,
                    1.0, //pc_scale_factor
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

        // The Fast marching method.
        //
        // let mut fmm = FastMarchingMethod::init(&configuration.device,
        //                                    global_dimension,
        //                                    local_dimension,
        //                                    &Some(&gpu_debugger),
        // );

        // #[cfg(feature = "gpu_debug")]
        let mut fim = FastIterativeMethod::init(&configuration.device,
                                               global_dimension,
                                               local_dimension,
                                               &gpu_debugger
        );

        //++ fmm.add_point_cloud_data(&configuration.device,
        //++                          &point_cloud.get_buffer(),
        //++                          &pc_params);
        fim.add_point_cloud_data(&configuration.device,
                                 &point_cloud.get_buffer(),
                                 &pc_params,
                                 &gpu_debugger);


        // The fmm scene visualizer. TODO: fim scene visualizer.
        
        #[cfg(feature = "gpu_debug")]
        let compute_object_fmm_visualizer =
                ComputeObject::init(
                    &configuration.device,
                    &configuration.device.create_shader_module(wgpu::ShaderModuleDescriptor {
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
                    &"main".to_string(),
                    None
        );

        // Fmm scene visualizer parasm
        let fmm_visualization_params = 
                 FmmVisualizationParams {
                     fmm_global_dimension: [FMM_GLOBAL_X as u32, FMM_GLOBAL_Y as u32, FMM_GLOBAL_Z as u32],
                     visualization_method: 0, // ???.
                     //visualization_method: 4, // ???.
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

        #[cfg(feature = "gpu_debug")]
        let compute_bind_groups_fmm_visualizer =
            Some(create_bind_groups(
                &configuration.device,
                &compute_object_fmm_visualizer.bind_group_layout_entries,
                &compute_object_fmm_visualizer.bind_group_layouts,
                &vec![
                    vec![
                         &buffers.get(&"fmm_visualization_params".to_string()).unwrap().as_entire_binding(),
                         &fim.get_fim_data_buffer().as_entire_binding(),
                         //&fmm.get_fmm_data_buffer().as_entire_binding(),
                         &gpu_debugger.as_ref().unwrap().get_element_counter_buffer().as_entire_binding(),
                         &gpu_debugger.as_ref().unwrap().get_output_chars_buffer().as_entire_binding(),
                         &gpu_debugger.as_ref().unwrap().get_output_arrows_buffer().as_entire_binding(),
                         &gpu_debugger.as_ref().unwrap().get_output_aabbs_buffer().as_entire_binding(),
                         &gpu_debugger.as_ref().unwrap().get_output_aabb_wires_buffer().as_entire_binding(),
                    ]
                ]
        ));

        // #[cfg(not(feature = "gpu_debug"))]
        // let compute_bind_groups_fmm_visualizer = None;

        let mut encoder = configuration.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Fmm initial interface encoder"),
        });

        fim.initialize_interface_pc(&mut encoder, &point_cloud);
        //++ fmm.initialize_interface_pc(&mut encoder, &point_cloud);
        // fmm.update_band_point_counts(&mut encoder);
        // fmm.filter_active_blocks(&mut encoder);

        configuration.queue.submit(Some(encoder.finish())); 

        // let filtered_blocks = to_vec::<FmmBlock>(
        //     &configuration.device,
        //     &configuration.queue,
        //     fmm.get_fmm_temp_buffer(),
        //     0,
        //     (size_of::<FmmBlock>()) as wgpu::BufferAddress * 1024
        // );

        // for (i, elem) in filtered_blocks.iter().enumerate() {
        //     println!("{:?} :: {:?}", i, elem);
        // }
        //
        let sphere_tracer_renderer = TwoTriangles::init(
                //&configuration.device, &configuration.sc_desc, [1024,1024]);
                //&configuration.device, &configuration.sc_desc, [1024,1024]);
                &configuration.device, &configuration.sc_desc, [1024,1536]);
                //&configuration.device, &configuration.sc_desc, [1536,2560]);
                //

        let sphere_tracer_params = SphereTracerParams {
                inner_dim: [8, 8],
                outer_dim: [128, 192],
                render_rays: 1,
                render_samplers: 0,
                isovalue: 0.015,
                draw_circles: 0,
                color_mode: 0,
                padding: 0,
        };

        let sphere_tracer = SphereTracer::init(
                &configuration.device,
                sphere_tracer_params.inner_dim,
                sphere_tracer_params.outer_dim,
                if sphere_tracer_params.render_rays == 0 {false} else {true},
                if sphere_tracer_params.render_samplers == 0 {false} else {true},
                sphere_tracer_params.isovalue,
                sphere_tracer_params.draw_circles,
                0, // TODO: enum
                //[192, 320],
                fim.get_fmm_params_buffer(),
                fim.get_fim_data_buffer(),
                //fmm.get_fmm_data_buffer(),
                &ray_camera.get_ray_camera_uniform(&configuration.device),
                &gpu_debugger
                //&Some(&gpu_debugger)
        );

        let mut encoder = configuration.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Fmm encoder.") });

        fim.fim_iteration(&mut encoder, &mut gpu_timer);
        //sphere_tracer.dispatch(&mut encoder); // SPHERE TRACER DISPATCH

        configuration.queue.submit(Some(encoder.finish())); 


        Self {
            camera: camera,
            ray_camera: ray_camera,
            camera_mode: camera_mode,
            #[cfg(feature = "gpu_debug")]
            gpu_debugger: gpu_debugger,
            gpu_timer: gpu_timer,
            keyboard_manager: keyboard_manager,
            screen: ScreenTexture::init(&configuration.device, &configuration.sc_desc, true),
            _light: light,
            _render_params: render_params,
            _triangle_mesh_renderer: triangle_mesh_renderer,
            _triangle_mesh_bindgroups: triangle_mesh_bindgroups,
            buffers: buffers,
            //radix radix_sort: radix_sort,
            point_cloud: point_cloud,
            #[cfg(feature = "gpu_debug")]
            compute_bind_groups_fmm_visualizer: compute_bind_groups_fmm_visualizer,
            #[cfg(feature = "gpu_debug")]
            compute_object_fmm_visualizer: compute_object_fmm_visualizer,
            render_vvvvnnnn: render_vvvvnnnn,
            render_vvvvnnnn_bg: render_vvvvnnnn_bg,
            render_object_vvvc: render_object_vvvc,
            render_bind_groups_vvvc: render_bind_groups_vvvc,
            app_render_params: app_render_params,
            // fmm: fmm,
            _pc_params: pc_params,
            once: once,
            sphere_tracer: sphere_tracer,
            sphere_tracer_renderer: sphere_tracer_renderer,
            sphere_tracer_params: sphere_tracer_params,
            draw_two_triangles: true,
            fim: fim,
         }
    }

    fn render(&mut self,
              device: &wgpu::Device,
              queue: &mut wgpu::Queue,
              surface: &wgpu::Surface,
              sc_desc: &wgpu::SurfaceConfiguration,
              _spawner: &Spawner) {

        self.screen.acquire_screen_texture(
            &device,
            &sc_desc,
            &surface
        );

        let view = self.screen.surface_texture.as_ref().unwrap().texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
        });

        let mut clear = true;

        // Always, render something :).
        draw(&mut encoder,
             &view,
             self.screen.depth_texture.as_ref().unwrap(),
             &self.render_vvvvnnnn_bg, 
             &self.render_vvvvnnnn.get_render_object().pipeline,
             &self.buffers.get(&"dummy_buffer".to_string()).unwrap(),
             0..1, 
             clear
        );

        clear = false;
        if self.draw_two_triangles {
           draw(&mut encoder,
                &view,
                self.screen.depth_texture.as_ref().unwrap(),
                &self.sphere_tracer_renderer.get_bind_groups(), 
                &self.sphere_tracer_renderer.get_render_object().pipeline,
                &self.sphere_tracer_renderer.get_draw_buffer(),
                0..6,
                clear
           );
        }

        // clear = false;

        if self.app_render_params.draw_point_cloud {
    
            draw(&mut encoder,
                 &view,
                 &self.screen.depth_texture.as_ref().unwrap(),
                 &self.render_bind_groups_vvvc,
                 &self.render_object_vvvc.pipeline,
                 &self.point_cloud.get_buffer(),
                 0..self.point_cloud.get_point_count(),
                 clear
            );
    
            // clear = false;
        }

        queue.submit(Some(encoder.finish())); 

        #[cfg(feature = "gpu_debug")]
        if !self.draw_two_triangles {
            self.gpu_debugger.as_mut().unwrap().render(
                      &device,
                      &queue,
                      &view,
                      self.screen.depth_texture.as_ref().unwrap(),
                      &mut clear,
                      _spawner
            );
        }
    
        //self.gpu_debugger.reset_element_counters(&queue);
        #[cfg(feature = "gpu_debug")] {
        self.gpu_debugger.as_mut().unwrap().reset_chars(&device, &queue);
        self.gpu_debugger.as_mut().unwrap().reset_arrows(&device, &queue);
        self.gpu_debugger.as_mut().unwrap().reset_aabbs(&device, &queue);
        self.gpu_debugger.as_mut().unwrap().reset_aabb_wires(&device, &queue);
        }

        self.screen.prepare_for_rendering();
    }

    #[allow(unused)]
    fn input(&mut self, queue: &wgpu::Queue, input: &InputCache) {
        match self.camera_mode {
            CameraMode::Camera => { self.camera.update_from_input(&queue, &input) },
            CameraMode::RayCamera => self.ray_camera.update_from_input(&queue, &input),
        }

        // Far.
        #[cfg(feature = "gpu_debug")]
        if self.keyboard_manager.test_key(&Key::Key1, input) {
            let bit = if get_bit(self.app_render_params.visualization_method, 0) == 0 { true } else { false };
            self.app_render_params.visualization_method = set_bit_to(self.app_render_params.visualization_method, 0, bit);
            self.app_render_params.update = true;
        }

        // Band.
        #[cfg(feature = "gpu_debug")]
        if self.keyboard_manager.test_key(&Key::Key2, input) {
            let bit = if get_bit(self.app_render_params.visualization_method, 1) == 0 { true } else { false };
            self.app_render_params.visualization_method = set_bit_to(self.app_render_params.visualization_method, 1, bit);
            self.app_render_params.update = true;
        }

        // Known.
        #[cfg(feature = "gpu_debug")]
        if self.keyboard_manager.test_key(&Key::Key3, input) {
            let bit = if get_bit(self.app_render_params.visualization_method, 2) == 0 { true } else { false };
            self.app_render_params.visualization_method = set_bit_to(self.app_render_params.visualization_method, 2, bit);
            self.app_render_params.update = true;
        }

        if self.keyboard_manager.test_key(&Key::Key0, input) {
            self.app_render_params.draw_point_cloud = !self.app_render_params.draw_point_cloud;
            if self.app_render_params.draw_point_cloud {
                println!("Show point cloud."); 
            }
            else {
                println!("Hide point cloud."); 
            }
        }
        
        // Numbers.
        #[cfg(feature = "gpu_debug")]
        if self.keyboard_manager.test_key(&Key::N, input) {
            let bit = if get_bit(self.app_render_params.visualization_method, 6) == 0 { true } else { false };
            self.app_render_params.visualization_method = set_bit_to(self.app_render_params.visualization_method, 6, bit);
            self.app_render_params.update = true;
        }
        // Focal distance
        if self.keyboard_manager.test_key(&Key::O, input) {
            self.ray_camera.set_focal_distance(self.ray_camera.get_focal_distance() - 0.01, &queue);
            println!("Ray camera: focal distance == {:?}.", self.ray_camera.get_focal_distance()); 
        }
        // Focal distance
        if self.keyboard_manager.test_key(&Key::P, input) {
            self.ray_camera.set_focal_distance(self.ray_camera.get_focal_distance() + 0.01, &queue);
            println!("Ray camera: focal distance == {:?}.", self.ray_camera.get_focal_distance()); 
        }
        // Switch camera mode to ray camera.
        if self.keyboard_manager.test_key(&Key::I, input) {
            if self.camera_mode != CameraMode::RayCamera { println!("Ray camera selected."); }
            self.camera_mode = CameraMode::RayCamera; 
        }
        // Switch camera mode to camera.
        if self.keyboard_manager.test_key(&Key::U, input) {
            if self.camera_mode != CameraMode::Camera { println!("Debug camera selected."); }
            self.camera_mode = CameraMode::Camera; 
        }
        // Switch camera mode to camera.
        if self.keyboard_manager.test_key(&Key::Space, input) {
            self.draw_two_triangles = !self.draw_two_triangles; 
            if self.draw_two_triangles {
                println!("Sphere tracer mode."); 
            }
            else {
                println!("Debug mode."); 
            }
        }

        let inc_value = 0.02;

        // Increase isovalue.
        if self.keyboard_manager.test_key(&Key::NumpadAdd, input) {
            self.sphere_tracer_params.isovalue = self.sphere_tracer_params.isovalue + inc_value;
            println!("sphere_tracer_params.isovalue == {}", self.sphere_tracer_params.isovalue);
            self.sphere_tracer.change_isovalue(queue, self.sphere_tracer_params.isovalue);
        }

        // Decrease isovalue.
        if self.keyboard_manager.test_key(&Key::NumpadSubtract, input) {
            let old_val = self.sphere_tracer_params.isovalue;
            if old_val - inc_value > 0.0 { 
                self.sphere_tracer_params.isovalue = self.sphere_tracer_params.isovalue - inc_value;
                println!("sphere_tracer_params.isovalue == {}", self.sphere_tracer_params.isovalue);
                self.sphere_tracer.change_isovalue(queue, self.sphere_tracer_params.isovalue);
            }
        }

        // Draw spheres.
        #[cfg(feature = "gpu_debug")]
        if self.keyboard_manager.test_key(&Key::Key9, input) {
                self.sphere_tracer_params.draw_circles = 1;
                self.sphere_tracer.draw_circles(queue, self.sphere_tracer_params.draw_circles);
        }
        // Hide spheres.
        #[cfg(feature = "gpu_debug")]
        if self.keyboard_manager.test_key(&Key::Key8, input) {
                self.sphere_tracer_params.draw_circles = 0;
                self.sphere_tracer.draw_circles(queue, self.sphere_tracer_params.draw_circles);
        }
        // Render samplers.
        #[cfg(feature = "gpu_debug")]
        if self.keyboard_manager.test_key(&Key::Key7, input) {
                self.sphere_tracer_params.render_samplers = 1;
                self.sphere_tracer.render_samplers(queue, self.sphere_tracer_params.render_samplers);
        }
        // Hide samplers.
        #[cfg(feature = "gpu_debug")]
        if self.keyboard_manager.test_key(&Key::Key6, input) {
                self.sphere_tracer_params.render_samplers = 0;
                self.sphere_tracer.render_samplers(queue, self.sphere_tracer_params.render_samplers);
        }

        if self.app_render_params.update {

            let fmm_visualization_params = 
                     FmmVisualizationParams {
                         fmm_global_dimension: [FMM_GLOBAL_X as u32, FMM_GLOBAL_Y as u32, FMM_GLOBAL_Z as u32],
                         visualization_method: self.app_render_params.visualization_method,
                         fmm_inner_dimension: [FMM_INNER_X as u32, FMM_INNER_Y as u32, FMM_INNER_Z as u32],
                         future_usage: 0,
            };

            queue.write_buffer(
                &self.buffers.get(&"fmm_visualization_params".to_string()).unwrap(),
                0,
                bytemuck::cast_slice(&[fmm_visualization_params])
            );
        }
    }

    fn resize(&mut self, device: &wgpu::Device, sc_desc: &wgpu::SurfaceConfiguration, _new_size: winit::dpi::PhysicalSize<u32>) {

        // TODO: add this functionality to the Screen.
        self.screen.depth_texture = Some(Texture::create_depth_texture(&device, &sc_desc, Some("depth-texture")));
        self.camera.resize(sc_desc.width as f32, sc_desc.height as f32);
    }

    #[allow(unused)]
    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, input: &InputCache, _spawner: &Spawner) {

        // Step fmm.
        // if self.keyboard_manager.test_key(&Key::B, input) {
        //     self.once = true;
        //     if self.fmm.get_fmm_state() == FmmState::FilterActiveBlocks {
        //         self.gpu_debugger.reset_aabb_wires(&device, &queue);
        //         // self.gpu_debugger.reset_chars(&device, &queue);
        //     }
        // }

        let total_grid_count = FMM_GLOBAL_X *
                               FMM_GLOBAL_Y *
                               FMM_GLOBAL_Z *
                               FMM_INNER_X *
                               FMM_INNER_Y *
                               FMM_INNER_Z;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Fmm encoder.") });

        // Fast marching method.
        if self.once {
            // self.fmm.fmm_iteration(&mut encoder, &mut self.gpu_timer);
            // self.fim.fim_iteration(&mut encoder, &mut self.gpu_timer);
            // self.sphere_tracer.dispatch(&mut encoder); // SPHERE TRACER DISPATCH
            // let mut pass = self.fmm.create_compute_pass_fmm(&mut encoder);

            // self.gpu_timer.start_pass(&mut pass);
            // self.fmm.collect_known_cells(&mut pass);
            // self.gpu_timer.end_pass(&mut pass);

            // self.gpu_timer.start_pass(&mut pass);
            // self.fmm.create_initial_band(&mut pass);
            // self.gpu_timer.end_pass(&mut pass);

            // self.gpu_timer.start_pass(&mut pass);
            // self.fmm.fmm(&mut pass);
            // self.gpu_timer.end_pass(&mut pass);

            // self.gpu_timer.start_pass(&mut pass);
            // self.fmm.filter_active_blocks(&mut pass);
            // self.gpu_timer.end_pass(&mut pass);

            // self.gpu_timer.start_pass(&mut pass);
            // pass.set_pipeline(&self.fmm.get_pipeline1());
            // self.gpu_timer.end_pass(&mut pass);

            // self.gpu_timer.start_pass(&mut pass);
            // pass.set_pipeline(&self.fmm.get_pipeline2());
            // self.gpu_timer.end_pass(&mut pass);

            // self.gpu_timer.start_pass(&mut pass);
            // self.fmm.switch_pipeline2(&mut pass);
            // self.gpu_timer.end_pass(&mut pass);

            // self.fmm.visualize_active_blocs(&mut pass);
            //drop(pass);
        }

        let total_block_count = (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z) as u32;
        let disp_y = udiv_up_safe32(total_block_count.try_into().unwrap(), 65535);  
        let disp_x = total_block_count / disp_y; 
        let dispatch_x = udiv_up_safe32(total_grid_count.try_into().unwrap(), 128);  

        // println!("dispatch_x == {}", dispatch_x);

        // println!("total_block_count == {}", total_block_count);
        // println!("disp_x == {}, disp_y == {}", disp_x, disp_y);

        // Cell visualizer.

        #[cfg(feature = "gpu_debug")]
        if self.app_render_params.visualization_method != 0 {
            self.compute_object_fmm_visualizer.dispatch(
                &self.compute_bind_groups_fmm_visualizer.as_ref().unwrap(),
                &mut encoder,
                // (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z) as u32, 1, 1,
                dispatch_x, 1, 1,
                //disp_x, disp_y, 1,
                Some("fmm visualizer dispatch")
            );
            self.app_render_params.update = false;
        }

        // self.gpu_timer.resolve_timestamps(&mut encoder);
        //if self.once {
        self.sphere_tracer.dispatch(&mut encoder); // SPHERE TRACER DISPATCH
        //}

        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: self.sphere_tracer.get_color_buffer(),
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(NonZeroU32::new(self.sphere_tracer.get_width() * 4).unwrap()), 
                    //bytes_per_row: Some(NonZeroU32::new(self.width * 4).unwrap()), 
                    rows_per_image: None,
                    // rows_per_image: Some(NonZeroU32::new(self.depth).unwrap()),
                },
            },
            wgpu::ImageCopyTexture {
                texture: self.sphere_tracer_renderer.get_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.sphere_tracer.get_width(),
                height: self.sphere_tracer.get_height(),
                depth_or_array_layers: 1,
            }
        );

        queue.submit(Some(encoder.finish())); 

        if self.once {

        // TODO: something better.
        if !self.gpu_timer.is_none() {
            self.gpu_timer.as_mut().unwrap().create_timestamp_data(&device);
            self.gpu_timer.as_mut().unwrap().print_data();
        }

            //let gpu_timer_result = self.gpu_timer.get_data(); 

            //self.fim.print_fim_histogram(&device, &queue);

            // // println!("{:?}", gpu_timer_result);
            // let filtered_blocks = to_vec::<FmmBlock>(
            //     &device,
            //     &queue,
            //     self.fmm.get_fmm_temp_buffer(),
            //     0,
            //     (size_of::<FmmBlock>()) as wgpu::BufferAddress * 39000
            // );

            // for (i, elem) in filtered_blocks.iter().enumerate() {
            //     println!("{:?} :: {:?}", i, elem);
            // }

            // let temp_prefix_data = to_vec::<u32>(
            //     &device,
            //     &queue,
            //     self.fmm.get_fmm_prefix_temp_buffer(),
            //     0,
            //     (size_of::<u32>()) as wgpu::BufferAddress * total_grid_count as u64
            // );

            // for (i, elem) in temp_prefix_data.iter().enumerate() {
            //     println!("{:?} :: {:?}", i, elem);
            // }
            self.once = false;
        }

    }
}

fn main() {
    
    jaankaup_core::template::run_loop::<FmmApp, BasicLoop, FmmAppFeatures>(); 
    println!("Finished...");
}

/// A helper function for logging adapter information.
#[allow(dead_code)]
fn log_adapter_info(adapter: &wgpu::Adapter) {

        let adapter_limits = adapter.limits(); 

        log::info!("Adapter limits.");
        log::info!("max_compute_workgroup_storage_size: {:?}", adapter_limits.max_compute_workgroup_storage_size);
        log::info!("max_compute_invocations_per_workgroup: {:?}", adapter_limits.max_compute_invocations_per_workgroup);
        log::info!("max_compute_workgroup_size_x: {:?}", adapter_limits.max_compute_workgroup_size_x);
        log::info!("max_compute_workgroup_size_y: {:?}", adapter_limits.max_compute_workgroup_size_y);
        log::info!("max_compute_workgroup_size_z: {:?}", adapter_limits.max_compute_workgroup_size_z);
        log::info!("max_compute_workgroups_per_dimension: {:?}", adapter_limits.max_compute_workgroups_per_dimension);
        log::info!("max_push_constant_size: {:?}", adapter_limits.max_push_constant_size);
}

/// Initialize and create GpuDebugger for this project. TODO: add MAX_... to function parameters.
#[cfg(feature = "gpu_debug")]
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
        keys.register_key(Key::I, 10.0);
        keys.register_key(Key::U, 10.0);
        keys.register_key(Key::Key1, 20.0);
        keys.register_key(Key::Key2, 20.0);
        keys.register_key(Key::Key3, 20.0);
        keys.register_key(Key::Key4, 20.0);
        keys.register_key(Key::Key5, 20.0);
        keys.register_key(Key::Key6, 20.0);
        keys.register_key(Key::Key7, 20.0);
        keys.register_key(Key::Key8, 20.0);
        keys.register_key(Key::Key9, 20.0);
        keys.register_key(Key::Key0, 150.0);
        keys.register_key(Key::N, 200.0);
        keys.register_key(Key::Space, 150.0);
        keys.register_key(Key::B, 50.0);
        keys.register_key(Key::NumpadAdd, 50.0);
        keys.register_key(Key::NumpadSubtract, 50.0);
        
        keys
}

/// Load a wavefront mesh and store it to hash_map. Drop texture coordinates.
#[allow(dead_code)]
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

//use rand::Rng;
//use jaankaup_core::radix::{KeyMemoryIndex, Bucket, LocalSortBlock, RadixSort};
// use std::mem::size_of;

//radix let key_count: u32 = 1000000;

//radix radix_sort: RadixSort,
  
//radix let mut rng = rand::thread_rng();
//radix let mut radix_test_data: Vec<KeyMemoryIndex> = Vec::with_capacity(key_count as usize);

//radix for i in 0..key_count {
//radix     radix_test_data.push(KeyMemoryIndex { key: rng.gen_range(0..256), memory_location: 55, }); 
//radix }

//radix let radix_input_buffer = buffer_from_data::<KeyMemoryIndex>(
//radix     &configuration.device,
//radix     &radix_test_data,
//radix     wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//radix     None
//radix );

//radix // Radix sort
//radix let radix_sort = RadixSort::init(&configuration.device, &radix_input_buffer, key_count, key_count);

//radix // Store radix input buffer. TODO: create a better sample buffer.
//radix buffers.insert("radix_buffer".to_string(), radix_input_buffer);
//radix 
//radix let mut encoder = configuration.device.create_command_encoder(
//radix     &wgpu::CommandEncoderDescriptor {
//radix         label: Some("Radix sort counting sort encoder"),
//radix });

//radix radix_sort.initial_counting_sort(&mut encoder);

//radix configuration.queue.submit(Some(encoder.finish())); 

//radix let radix_histogram = to_vec::<u32>(
//radix         &configuration.device,
//radix         &configuration.queue,
//radix         radix_sort.get_global_histogram(),
//radix         0,
//radix         (size_of::<u32>()) as wgpu::BufferAddress * (256 as u64)
//radix     );

//radix for i in 0..256 {
//radix     println!("{} :: {}", i, radix_histogram[i]);
//radix }

//radix let bitonic_result = to_vec::<KeyMemoryIndex>(
//radix         &configuration.device,
//radix         &configuration.queue,
//radix         &buffers.get(&"radix_buffer".to_string()).unwrap(),
//radix         0,
//radix         (size_of::<KeyMemoryIndex>()) as wgpu::BufferAddress * (3000 as u64)
//radix     );

//radix println!("{}", radix_histogram.iter().sum::<u32>());

//radix for i in 0..bitonic_result.len() {
//radix     println!("{} = {:?}", i, bitonic_result[i]);
//radix }

//radix // let exclusive_sum: Vec<u32> = vec![vec![0 as u32], radix_histogram]
//radix //     .into_iter()
//radix //     .flatten()
//radix //     .into_iter()
//radix //     .scan(0, |state, x| { *state = *state + x; Some(state) })
//radix //     .collect();

//radix let mut exclusive_scan: Vec<u32> = vec![0; 257];

//radix println!("len radix_histogram == {}", radix_histogram.len());
//radix println!("len exclusive_scan == {}", exclusive_scan.len());
//radix 
//radix for i in 1..257 {
//radix     exclusive_scan[i] = exclusive_scan[i-1] + radix_histogram[i-1];
//radix }

//radix println!("{:?}", exclusive_scan);

//radix // let temp_sum = 0; 

//radix // for i in 1..radix_histogram.len() {
//radix //     let previous = temp_sum + radix_histogram[i-1];
//radix //     exclusive_sum[    
//radix // }

//radix let mut sub_buckets: Vec<Bucket> = Vec::new();
//radix let mut local_blocks: Vec<LocalSortBlock> = Vec::new();
//radix let mut bucket_id_counter = 1;

//radix for i in 0..radix_histogram.len() {
//radix     if radix_histogram[i] >= 3072 {
//radix         sub_buckets.push(Bucket {
//radix             bucket_id: bucket_id_counter,
//radix             rank: 1,
//radix             bucket_offset: exclusive_scan[i],
//radix             size: radix_histogram[i],
//radix             }
//radix         );
//radix         bucket_id_counter = bucket_id_counter + 1; 
//radix     }
//radix }

//radix let mut keys_so_far = 0;
//radix // let mut offset = 0;
//radix let mut temp_bucket_offset = 0;
//radix // let mut temp_local_block: Option<LocalBlock> = None;

//radix for i in 0..radix_histogram.len() {

//radix     let temp_sum = radix_histogram[i] + keys_so_far;

//radix     // Found a sub bucket and there was no local bucket.
//radix     if keys_so_far == 0 && radix_histogram[i] >= 3072 {
//radix         continue;
//radix     }

//radix     // Start a new local block.
//radix     else if radix_histogram[i] < 3072 && keys_so_far == 0 {
//radix         keys_so_far = radix_histogram[i];
//radix         temp_bucket_offset = exclusive_scan[i];
//radix     }

//radix     else if keys_so_far > 0 && temp_sum < 3072 {
//radix         keys_so_far = temp_sum;
//radix     }

//radix     // Save local block.
//radix     else if keys_so_far > 0 && temp_sum >= 3072 {

//radix         local_blocks.push(
//radix             LocalSortBlock {
//radix                 local_sort_id: 0,
//radix                 local_offset: temp_bucket_offset,
//radix                 local_key_count: keys_so_far,
//radix                 is_merged: 0,
//radix             }
//radix         );

//radix         keys_so_far = 0;
//radix         // temp_bucket_offset = 0;
//radix     }
//radix     else if i == radix_histogram.len() - 1 && temp_sum > 0 {

//radix         local_blocks.push(
//radix             LocalSortBlock {
//radix                 local_sort_id: 0,
//radix                 local_offset: temp_bucket_offset,
//radix                 local_key_count: temp_sum,
//radix                 is_merged: 0,
//radix             }
//radix         );
//radix     }
//radix }
//radix println!("keys_so_far = {}", keys_so_far);

//radix println!("sub buckets");
//radix for i in sub_buckets.iter() {
//radix     println!("{:?}", i);
//radix }

//radix println!("local blocks");
//radix for i in local_blocks.iter() {
//radix     println!("{:?}", i);
//radix }
