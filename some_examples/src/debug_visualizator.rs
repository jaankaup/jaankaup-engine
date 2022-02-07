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
use jaankaup_core::wgpu;
use jaankaup_core::winit;
use jaankaup_core::log;
use jaankaup_core::screen::ScreenTexture;
use jaankaup_core::texture::Texture;
use jaankaup_core::misc::Convert2Vec;
use jaankaup_core::impl_convert;
use bytemuck::{Pod, Zeroable};

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
struct VisualizationParams{
    triangle_start: u32,
    triangle_end: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DebugArray {
    start: [f32; 3],
    end:   [f32; 3],
    color: u32,
    size: f32,
}

impl_convert!{DebugArray}

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
    pub screen: ScreenTexture, 
    pub render_object: RenderObject, 
    pub bind_groups: Vec<wgpu::BindGroup>,
    // pub _textures: HashMap<String, Texture>,
    pub buffers: HashMap<String, wgpu::Buffer>,
    pub camera: Camera,
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
                            // var<storage,read_write> output: array<VertexBuffer>;
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
                        ],
                    ]
        );

        // Camera.
        let mut camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32);
        camera.set_rotation_sensitivity(0.2);

        // Create bind groups for basic render pipeline and grass/rock textures.
        let bind_groups = create_bind_groups(
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
 
        DebugVisualizator {
            screen: ScreenTexture::init(&configuration.device, &configuration.sc_desc, true),
            render_object: render_object,
            bind_groups,
            // _textures: textures,
            buffers: buffers,
            camera: camera,
        }
    }

    fn render(&mut self,
              device: &wgpu::Device,
              queue: &mut wgpu::Queue,
              surface: &wgpu::Surface,
              sc_desc: &wgpu::SurfaceConfiguration) {

        // self.screen.acquire_screen_texture(
        //     &device,
        //     &sc_desc,
        //     &surface
        // );

        // let mut encoder = device.create_command_encoder(
        //     &wgpu::CommandEncoderDescriptor {
        //         label: Some("Render Encoder"),
        // });

        // // Ownership?
        // let view = self.screen.surface_texture.as_ref().unwrap().texture.create_view(&wgpu::TextureViewDescriptor::default());

        // // Draw the cube.
        // draw(&mut encoder,
        //      &view,
        //      self.screen.depth_texture.as_ref().unwrap(),
        //      &self.bind_groups,
        //      &self.render_object.pipeline,
        //      &self.buffers.get("cube").unwrap(),
        //      0..36, // TODO: Cube 
        //      true
        // );

        // queue.submit(Some(encoder.finish())); 

        // self.screen.prepare_for_rendering();
    }

    fn input(&mut self, queue: &wgpu::Queue, input_cache: &InputCache) {
    }

    fn resize(&mut self, device: &wgpu::Device, sc_desc: &wgpu::SurfaceConfiguration, _new_size: winit::dpi::PhysicalSize<u32>) {
        self.screen.depth_texture = Some(Texture::create_depth_texture(&device, &sc_desc, Some("depth-texture")));
        self.camera.resize(sc_desc.width as f32, sc_desc.height as f32);
    }

    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, input: &InputCache) {
        self.camera.update_from_input(&queue, &input);
    }
}

fn main() {
    
    jaankaup_core::template::run_loop::<DebugVisualizator, BasicLoop, DebugVisualizatorFeatures>(); 
    println!("Finished...");
}
