use jaankaup_core::input::*;
use jaankaup_core::template::{
        WGPUFeatures,
        WGPUConfiguration,
        Application,
        BasicLoop,
        Spawner,
};
use jaankaup_core::{wgpu, log};
use jaankaup_core::winit;
use jaankaup_core::camera::Camera;

// TODO: add to fmm params.
const MAX_NUMBER_OF_ARROWS:     usize = 40960;
const MAX_NUMBER_OF_AABBS:      usize = 262144;
const MAX_NUMBER_OF_AABB_WIRES: usize = 40960;
const MAX_NUMBER_OF_CHARS:      usize = 262144;

// FMM global dimensions.
const FMM_GLOBAL_X: usize = 16; 
const FMM_GLOBAL_Y: usize = 16; 
const FMM_GLOBAL_Z: usize = 16; 

// FMM inner dimensions.
const FMM_INNER_X: usize = 4; 
const FMM_INNER_Y: usize = 4; 
const FMM_INNER_Z: usize = 4; 

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

struct Eikonal {
    camera: Camera,
//++    pub screen: ScreenTexture, 
//++    pub gpu_debugger: GpuDebugger,
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
//++    pub keys: KeyboardManager,
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

        log_adapter_info(&configuration.adapter);

        // Camera.
        let mut camera = Camera::new(configuration.size.width as f32, configuration.size.height as f32, (0.0, 0.0, 10.0), -89.0, 0.0);
        camera.set_rotation_sensitivity(0.4);
        camera.set_movement_sensitivity(0.02);

        Self {
            camera: camera,
        }
    }

    fn render(&mut self,
              device: &wgpu::Device,
              queue: &mut wgpu::Queue,
              surface: &wgpu::Surface,
              sc_desc: &wgpu::SurfaceConfiguration,
              spawner: &Spawner) {

    }

    #[allow(unused)]
    fn input(&mut self, queue: &wgpu::Queue, input: &InputCache) {
        self.camera.update_from_input(&queue, &input);
    }

    fn resize(&mut self, device: &wgpu::Device, sc_desc: &wgpu::SurfaceConfiguration, _new_size: winit::dpi::PhysicalSize<u32>) {
        // self.screen.depth_texture = Some(Texture::create_depth_texture(&device, &sc_desc, Some("depth-texture")));
        // self.camera.resize(sc_desc.width as f32, sc_desc.height as f32);
    }

    #[allow(unused)]
    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, input: &InputCache, spawner: &Spawner) {

    }
}

fn main() {
    
    jaankaup_core::template::run_loop::<Eikonal, BasicLoop, EikonalFeatures>(); 
    println!("Finished...");
}

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
