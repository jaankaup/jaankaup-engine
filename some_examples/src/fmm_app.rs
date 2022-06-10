//use rand::Rng;
//use jaankaup_core::radix::{KeyMemoryIndex, Bucket, LocalSortBlock, RadixSort};
// use std::mem::size_of;
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
    create_buffer_bindgroup_layout,
    create_uniform_bindgroup_layout,
    set_bit_to,
    get_bit,
    encode_rgba_u32,
};
use jaankaup_core::{wgpu, log};
use jaankaup_core::winit;
use jaankaup_core::buffer::{buffer_from_data}; //, to_vec};
use jaankaup_core::model_loader::{TriangleMesh, create_from_bytes};
use jaankaup_core::camera::Camera;
use jaankaup_core::gpu_debugger::GpuDebugger;
use jaankaup_core::gpu_timer::GpuTimer;
use jaankaup_core::screen::ScreenTexture;
use jaankaup_core::shaders::{RenderVvvvnnnnCamera};
use jaankaup_core::render_things::{LightBuffer, RenderParamBuffer};
use jaankaup_core::render_object::{draw, RenderObject, ComputeObject, create_bind_groups};
use jaankaup_core::texture::Texture;
// use jaankaup_core::aabb::Triangle_vvvvnnnn;
// use jaankaup_core::common_functions::encode_rgba_u32;
use jaankaup_core::fmm_things::{PointCloud, FmmCellPc, PointCloudHandler, FmmValueFixer};
use jaankaup_core::fast_marching_method::FastMarchingMethod;
use bytemuck::{Pod, Zeroable};

/// Max number of arrows for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_ARROWS:     usize = 40960;

/// Max number of aabbs for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_AABBS:      usize = TOTAL_INDICES;

/// Max number of box frames for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_AABB_WIRES: usize = 40960;

/// Max number of renderable char elements (f32, vec3, vec4, ...) for gpu debugger.
#[allow(dead_code)]
const MAX_NUMBER_OF_CHARS:      usize = TOTAL_INDICES;

/// Max number of vvvvnnnn vertices reserved for gpu draw buffer.
#[allow(dead_code)]
const MAX_NUMBER_OF_VVVVNNNN: usize = 2000000;

#[allow(dead_code)]
const TOTAL_INDICES: usize = FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z * FMM_INNER_X * FMM_INNER_Y * FMM_INNER_Z; 

/// Name for the fire tower mesh (assets/models/wood.obj).
//const FIRE_TOWER_MESH: &'static str = "FIRE_TOWER";

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
//


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
        wgpu::Features::PUSH_CONSTANTS
        // wgpu::Features::empty()
    }

    fn required_limits() -> wgpu::Limits {
        let mut limits = wgpu::Limits::default();
        limits.max_compute_invocations_per_workgroup = 1024;
        limits.max_compute_workgroup_size_x = 1024;
        limits.max_storage_buffers_per_shader_stage = 10;
        limits.max_push_constant_size = 4;
        limits
    }
}

/// FmmApp solver. A Fast marching method (GPU) for solving the eikonal equation.
struct FmmApp {
    camera: Camera,
    gpu_debugger: GpuDebugger,
    _gpu_timer: Option<GpuTimer>,
    keyboard_manager: KeyboardManager,
    screen: ScreenTexture, 
    _light: LightBuffer,
    _render_params: RenderParamBuffer,
    _triangle_mesh_renderer: RenderVvvvnnnnCamera,
    _triangle_mesh_bindgroups: Vec<wgpu::BindGroup>,
    buffers: HashMap<String, wgpu::Buffer>,
    //radix radix_sort: RadixSort,
    point_cloud: PointCloud,
    compute_bind_groups_fmm_visualizer: Vec<wgpu::BindGroup>,
    compute_object_fmm_visualizer: ComputeObject,
    render_vvvvnnnn: RenderVvvvnnnnCamera,
    render_vvvvnnnn_bg: Vec<wgpu::BindGroup>,
    render_object_vvvc: RenderObject,
    render_bind_groups_vvvc: Vec<wgpu::BindGroup>,
    app_render_params: AppRenderParams,
    _render_params_point_cloud: RenderParamBuffer,
    _fmm_value_fixer: FmmValueFixer,
    fmm: FastMarchingMethod,
}

impl Application for FmmApp {

    fn init(configuration: &WGPUConfiguration) -> Self {

        // Log adapter info.
        // log_adapter_info(&configuration.adapter);

        // Buffer hash_map.
        let mut buffers: HashMap<String, wgpu::Buffer> = HashMap::new();

        // Camera.
        let mut camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (180.0, 130.0, 480.0), -89.0, 0.0);
        camera.set_rotation_sensitivity(0.4);
        camera.set_movement_sensitivity(0.1);


        let app_render_params = AppRenderParams {
             draw_point_cloud: false,
             visualization_method: 0,
             _step: false,
             _show_numbers: false,
             update: false,
        };

        // Gpu debugger.
        let gpu_debugger = create_gpu_debugger( &configuration.device, &configuration.sc_desc, &mut camera);

        // Gpu timer.
        let gpu_timer = GpuTimer::init(&configuration.device, &configuration.queue, 8, Some("gpu timer"));

        // Keyboard manager. Keep tract of keys which has been pressed, and for how long time.
        let keyboard_manager = create_keyboard_manager();

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

        let global_dimension = [FMM_GLOBAL_X as u32, FMM_GLOBAL_Y as u32, FMM_GLOBAL_Z as u32];
        let local_dimension = [FMM_INNER_X as u32, FMM_INNER_Y as u32, FMM_INNER_Z as u32];
        let total_cell_count = FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z * FMM_INNER_X * FMM_INNER_Y * FMM_INNER_Z;
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
        let point_cloud = PointCloud::init(&configuration.device, &"../../cloud_data.asc".to_string());

        // Store point data.
        let _pc_sample_data = 
            buffers.insert(
                "pc_sample_data".to_string(),
                buffer_from_data::<FmmCellPc>(
                &configuration.device,
                &vec![FmmCellPc {
                    tag: 0,
                    value: 10000000,
                    color: 0,
                    // padding: 0,
                } ; total_cell_count],
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                None)
            );

        let pc_min_coord = point_cloud.get_min_coord();
        let pc_max_coord = point_cloud.get_max_coord();

        // Create scale factor for point data so it fits under computation domain ranges.
        let point_cloud_scale_factor_x = (FMM_GLOBAL_X * FMM_INNER_X) as f32 / pc_max_coord[0];
        let point_cloud_scale_factor_y = (FMM_GLOBAL_Y * FMM_INNER_Y) as f32 / pc_max_coord[1];
        let point_cloud_scale_factor_z = (FMM_GLOBAL_Z * FMM_INNER_Z) as f32 / pc_max_coord[2];

        let pc_scale_factor = point_cloud_scale_factor_x.min(point_cloud_scale_factor_y).min(point_cloud_scale_factor_z);

        let render_params_point_cloud = RenderParamBuffer::create(
                    &configuration.device,
                    pc_scale_factor
        );


        let point_cloud_handler = PointCloudHandler::init(
                &configuration.device,
                global_dimension,
                local_dimension,
                point_cloud.get_point_count(),
                pc_min_coord,
                pc_max_coord,
                pc_scale_factor, // scale_factor
                0, //point_cloud_draw_iterator,
                false, //show_numbers,
                &buffers.get("pc_sample_data").unwrap(),
                point_cloud.get_buffer(),
                &gpu_debugger
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
        let fmm = FastMarchingMethod::init(&configuration.device,
                                           global_dimension,
                                           local_dimension,
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

        let fmm_value_fixer = FmmValueFixer::init(&configuration.device,
                                                  &buffers.get(&"fmm_visualization_params".to_string()).unwrap(),
                                                  &buffers.get(&"pc_sample_data".to_string()).unwrap()
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

        let mut encoder = configuration.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Domain tester encoder"),
        });

        point_cloud_handler.point_data_to_interface(&mut encoder);
        fmm_value_fixer.dispatch(&mut encoder, [udiv_up_safe32(point_cloud.get_point_count(), 1024) + 1, 1, 1]);

        configuration.queue.submit(Some(encoder.finish())); 

        //radix let key_count: u32 = 1000000;

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

        Self {
            camera: camera,
            gpu_debugger: gpu_debugger,
            _gpu_timer: gpu_timer,
            keyboard_manager: keyboard_manager,
            screen: ScreenTexture::init(&configuration.device, &configuration.sc_desc, true),
            _light: light,
            _render_params: render_params,
            _triangle_mesh_renderer: triangle_mesh_renderer,
            _triangle_mesh_bindgroups: triangle_mesh_bindgroups,
            buffers: buffers,
            //radix radix_sort: radix_sort,
            point_cloud: point_cloud,
            compute_bind_groups_fmm_visualizer: compute_bind_groups_fmm_visualizer,
            compute_object_fmm_visualizer: compute_object_fmm_visualizer,
            render_vvvvnnnn: render_vvvvnnnn,
            render_vvvvnnnn_bg: render_vvvvnnnn_bg,
            render_object_vvvc: render_object_vvvc,
            render_bind_groups_vvvc: render_bind_groups_vvvc,
            app_render_params: app_render_params,
            _render_params_point_cloud: render_params_point_cloud,
            _fmm_value_fixer: fmm_value_fixer,
            fmm: fmm,
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
    }

    #[allow(unused)]
    fn input(&mut self, queue: &wgpu::Queue, input: &InputCache) {
        self.camera.update_from_input(&queue, &input);

        // Far.
        if self.keyboard_manager.test_key(&Key::Key1, input) {
            let bit = if get_bit(self.app_render_params.visualization_method, 0) == 0 { true } else { false };
            self.app_render_params.visualization_method = set_bit_to(self.app_render_params.visualization_method, 0, bit);
            self.app_render_params.update = true;
        }

        // Band.
        if self.keyboard_manager.test_key(&Key::Key2, input) {
            let bit = if get_bit(self.app_render_params.visualization_method, 1) == 0 { true } else { false };
            self.app_render_params.visualization_method = set_bit_to(self.app_render_params.visualization_method, 1, bit);
            self.app_render_params.update = true;
        }

        // Known.
        if self.keyboard_manager.test_key(&Key::Key3, input) {
            let bit = if get_bit(self.app_render_params.visualization_method, 2) == 0 { true } else { false };
            self.app_render_params.visualization_method = set_bit_to(self.app_render_params.visualization_method, 2, bit);
            self.app_render_params.update = true;
        }
        
        // Numbers.
        if self.keyboard_manager.test_key(&Key::N, input) {
            let bit = if get_bit(self.app_render_params.visualization_method, 6) == 0 { true } else { false };
            self.app_render_params.visualization_method = set_bit_to(self.app_render_params.visualization_method, 6, bit);
            self.app_render_params.update = true;
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
    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, input: &InputCache, spawner: &Spawner) {

        let total_grid_count = FMM_GLOBAL_X *
                               FMM_GLOBAL_Y *
                               FMM_GLOBAL_Z *
                               FMM_INNER_X *
                               FMM_INNER_Y *
                               FMM_INNER_Z;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Fmm visualizer encoder.") });

        //++ queue.submit(Some(encoder_command.finish()));

        if self.app_render_params.visualization_method != 0 {
            self.compute_object_fmm_visualizer.dispatch(
                &self.compute_bind_groups_fmm_visualizer,
                &mut encoder,
                (FMM_GLOBAL_X * FMM_GLOBAL_Y * FMM_GLOBAL_Z) as u32, 1, 1,
                Some("fmm visualizer dispatch")
            );
            self.app_render_params.update = false;
        }

        queue.submit(Some(encoder.finish())); 
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