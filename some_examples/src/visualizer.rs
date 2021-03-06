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
use jaankaup_core::common_functions::{
    encode_rgba_u32, 
    create_uniform_bindgroup_layout,
    create_buffer_bindgroup_layout
};
use jaankaup_core::histogram::Histogram;
use bytemuck::{Pod, Zeroable};

use winit::event as ev;
pub use ev::VirtualKeyCode as Key;

/// The number of vertices per chunk.
const MAX_VERTEX_CAPACITY: usize = 128 * 64 * 64; 

/// The size of draw buffer;
const VERTEX_BUFFER_SIZE: usize = 16 * MAX_VERTEX_CAPACITY * size_of::<f32>(); // VVVC

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
struct Char {
    start_pos: [f32 ; 4],
    value: [f32 ; 4],
    font_size: f32,
    vec_dim_count: u32, // 1 => f32, 2 => vec3<f32>, 3 => vec3<f32>, 4 => vec4<f32>
    color: u32,
    z_offset: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VisualizationParams{
    max_number_of_vertices: u32,
    iterator_start_index: u32,
    iterator_end_index: u32,
    arrow_size: f32,
    // current_iterator_index: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct OtherRenderParams {
    scale_factor: f32,
}

impl_convert!{Arrow}
impl_convert!{Char}

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

// State for this application.
struct DebugVisualizator {
    screen: ScreenTexture, 
    render_object_vvvvnnnn: RenderObject, 
    render_bind_groups_vvvvnnnn: Vec<wgpu::BindGroup>,
    render_object_vvvc: RenderObject,
    render_bind_groups_vvvc: Vec<wgpu::BindGroup>,
    compute_object: ComputeObject, 
    compute_bind_groups: Vec<wgpu::BindGroup>,
    compute_object_arrow: ComputeObject, 
    compute_bind_groups_arrow: Vec<wgpu::BindGroup>,
    buffers: HashMap<String, wgpu::Buffer>,
    camera: Camera,
    histogram: Histogram, // [arrow counter, aabb counter, aabb wire counter, Char counter,  
    draw_count_points: u32,
    draw_count_triangles: u32,
    _visualization_params: VisualizationParams,
    _keys: KeyboardManager,
    _light: LightBuffer,
}

impl DebugVisualizator {

}

//#[allow(unused_variables)]
impl Application for DebugVisualizator {

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

        let keys = KeyboardManager::init();
        // keys.register_key(Key::L, 20.0);

        // Camera.
        let mut camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (0.0, 0.0, 40.0), -90.0, 0.0);
        camera.set_rotation_sensitivity(0.4);
        camera.set_movement_sensitivity(0.02);

        let light = LightBuffer::create(
                      &configuration.device,
                      [10.0, 250.0, 10.0], // pos
                      [25, 25, 130],  // spec
                      [25,100,25], // light 
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

        // vvvvnnnn
        let render_object_vvvvnnnn =
                RenderObject::init(
                    &configuration.device,
                    &configuration.sc_desc,
                    &configuration.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("renderer_v4n4_debug_visualizator_wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/renderer_v4n4_debug_visualizator.wgsl"))),
                    
                    }),
                    &vec![wgpu::VertexFormat::Float32x4, wgpu::VertexFormat::Float32x4],
                    &vec![
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
        let render_bind_groups_vvvvnnnn = create_bind_groups(
                                     &configuration.device,
                                     &render_object_vvvvnnnn.bind_group_layout_entries,
                                     &render_object_vvvvnnnn.bind_group_layouts,
                                     &vec![
                                         vec![
                                             &camera.get_camera_uniform(&configuration.device).as_entire_binding(),
                                             &light.get_buffer().as_entire_binding(),
                                             &buffers.get(&"other_render_params".to_string()).unwrap().as_entire_binding()
                                         ],
                                     ]
        );

        // vvvc
        let render_object_vvvc =
                RenderObject::init(
                    &configuration.device,
                    &configuration.sc_desc,
                    &configuration.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("renderer_v4c1.wgsl"),
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
                                         ],
                                     ]
        );

        println!("Creating compute object.");

        let histogram = Histogram::init(&configuration.device, &vec![0; 2]);

        let params = VisualizationParams {
            max_number_of_vertices: 1,
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
                    &configuration.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("font_visualizer_wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/font_visualizer.wgsl"))),
                    
                    }),
                    Some("Font visualizer Compute object"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> camerauniform: Camera;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<uniform> visualization_params: VisualizationParams;
                            create_uniform_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(2) var<storage, read_write> counter: Counter;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(3) var<storage, read> counter: array<Arrow>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(4) var<storage,read_write> output: array<VertexBuffer>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
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
                                               &camera.get_camera_uniform(&configuration.device).as_entire_binding(),
                                               &buffers.get(&"visualization_params".to_string()).unwrap().as_entire_binding(),
                                               &histogram.get_histogram_buffer().as_entire_binding(),
                                               &buffers.get(&"debug_arrays".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output".to_string()).unwrap().as_entire_binding()
                                          ]
                                      ]
        );

        let compute_object_arrow =
                ComputeObject::init(
                    &configuration.device,
                    &configuration.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("compute_visualizer_wgsl array"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/visualizer.wgsl"))),
                    
                    }),
                    Some("Visualizer Compute object"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> visualization_params: VisualizationParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<storage, read_write> counter: Counter;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(2) var<storage, read> counter: array<Arrow>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(3) var<storage,read_write> output: array<VertexBuffer>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ],
                    &"main".to_string(),
                    None
        );

        let compute_bind_groups_arrow = create_bind_groups(
                                      &configuration.device,
                                      &compute_object_arrow.bind_group_layout_entries,
                                      &compute_object_arrow.bind_group_layouts,
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
            render_object_vvvvnnnn: render_object_vvvvnnnn,
            render_bind_groups_vvvvnnnn: render_bind_groups_vvvvnnnn,
            render_object_vvvc: render_object_vvvc,
            render_bind_groups_vvvc: render_bind_groups_vvvc,
            compute_object: compute_object,
            compute_bind_groups: compute_bind_groups,
            compute_object_arrow: compute_object_arrow, 
            compute_bind_groups_arrow: compute_bind_groups_arrow,
            // _textures: textures,
            buffers: buffers,
            camera: camera,
            histogram: histogram,
            draw_count_points: 0,
            draw_count_triangles: 0,
            _visualization_params: params,
            _keys: keys,
            _light: light,
        }
    }

    fn render(&mut self,
              device: &wgpu::Device,
              queue: &mut wgpu::Queue,
              surface: &wgpu::Surface,
              sc_desc: &wgpu::SurfaceConfiguration,
              spawner: &Spawner) {

        // Prepare screen for rendering.
        self.screen.acquire_screen_texture(
            &device,
            &sc_desc,
            &surface
        );

        // The viev.
        let view = self.screen.surface_texture.as_ref().unwrap().texture.create_view(&wgpu::TextureViewDescriptor::default());

        // First, render arrows and cubes.
        let mut items_available: i32 = 1; // Total number of arrows. 
        let dispatch_x = 1;
        let mut clear = true;
        let mut i = dispatch_x * 64;

        while items_available > 0 {

            let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Visualiztion (AABB)") });

            // Take 128 * 64 = 8192 items.
            self.compute_object.dispatch(
                &self.compute_bind_groups,
                &mut encoder_command,
                dispatch_x, 1, 1, Some("font visualizer dispatch")
            );


            // Submit compute.
            queue.submit(Some(encoder_command.finish()));

            let counter = self.histogram.get_values(device, queue);
            self.draw_count_points = counter[0];
            
            let mut encoder_command = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Visualiztion (AABB)") });

            self.compute_object_arrow.dispatch(
                &self.compute_bind_groups_arrow,
                &mut encoder_command,
                dispatch_x, 1, 1, Some("font visualizer dispatch")
            );

            queue.submit(Some(encoder_command.finish()));

            let counter2 = self.histogram.get_values(device, queue);
            self.draw_count_triangles = counter2[0];

            //println!("self.draw_count  == {}", self.draw_count); 

            let mut encoder_render = device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
            });

            items_available = items_available - dispatch_x as i32 * 64; 

            // Draw the cube.
            draw(&mut encoder_render,
                 &view,
                 self.screen.depth_texture.as_ref().unwrap(),
                 &self.render_bind_groups_vvvc,
                 &self.render_object_vvvc.pipeline,
                 &self.buffers.get("output").unwrap(),
                 0..self.draw_count_points, // TODO: Cube 
                 clear
            );

            clear = false;

            draw(&mut encoder_render,
                 &view,
                 self.screen.depth_texture.as_ref().unwrap(),
                 &self.render_bind_groups_vvvvnnnn,
                 &self.render_object_vvvvnnnn.pipeline,
                 &self.buffers.get("output").unwrap(),
                 self.draw_count_points..self.draw_count_triangles, // TODO: Cube 
                 clear
            );
            queue.submit(Some(encoder_render.finish()));
            clear = items_available <= 0;
            self.histogram.set_values_cpu_version(queue, &vec![0, i]);
            i = i + dispatch_x*64;
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
    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, input: &InputCache, _spawner: &Spawner) {
        self.camera.update_from_input(&queue, &input);

    }
}

fn main() {
    
    jaankaup_core::template::run_loop::<DebugVisualizator, BasicLoop, DebugVisualizatorFeatures>(); 
    println!("Finished...");
}
