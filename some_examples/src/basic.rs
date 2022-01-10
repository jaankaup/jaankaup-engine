use jaankaup_core::template::{
        WGPUFeatures,
        WGPUConfiguration,
        Application,
        BasicLoop,
};
use jaankaup_core::misc::Convert2Vec;
use jaankaup_core::impl_convert;
use jaankaup_core::input::*;
use jaankaup_core::wgpu;
use jaankaup_core::winit;
use jaankaup_core::log;
use jaankaup_core::screen::ScreenTexture;
use bytemuck::{Pod,Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TestData {
    pub pos: [f32; 3],
    pub something_else: u32,
}

impl_convert!{TestData}

struct BasicFeatures {}
impl WGPUFeatures for BasicFeatures { 
    fn optional_features() -> wgpu::Features {
        wgpu::Features::empty()
    }
    fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
        // wgpu::Features::SPIRV_SHADER_PASSTHROUGH
    }
    fn required_limits() -> wgpu::Limits {
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 8;
        limits
    }
}

// State for this application.
struct BasicApp {
    pub screen: ScreenTexture, 
}

#[allow(unused_variables)]
impl Application for BasicApp {

    fn init(configuration: &WGPUConfiguration) -> Self {

        log::info!("Adapter limits are: ");

        let adapter_limits = configuration.adapter.limits(); 

        log::info!("max_compute_workgroup_storage_size: {:?}", adapter_limits.max_compute_workgroup_storage_size);
        log::info!("max_compute_invocations_per_workgroup: {:?}", adapter_limits.max_compute_invocations_per_workgroup);
        log::info!("max_compute_workgroup_size_x: {:?}", adapter_limits.max_compute_workgroup_size_x);
        log::info!("max_compute_workgroup_size_y: {:?}", adapter_limits.max_compute_workgroup_size_y);
        log::info!("max_compute_workgroup_size_z: {:?}", adapter_limits.max_compute_workgroup_size_z);
        log::info!("max_compute_workgroups_per_dimension: {:?}", adapter_limits.max_compute_workgroups_per_dimension);
        
        BasicApp {
            screen: ScreenTexture::init(&configuration.device, &configuration.sc_desc, true)
        }
    }

    fn render(&mut self,
              device: &wgpu::Device,
              queue: &mut wgpu::Queue,
              surface: &wgpu::Surface,
              sc_desc: &wgpu::SurfaceConfiguration) {

        self.screen.acquire_screen_texture(
            &device,
            &sc_desc,
            &surface
        );



        self.screen.prepare_for_rendering();

        // queue.submit(Some(encoder.finish()));
        // frame.present();
    }

    fn input(&mut self, queue: &wgpu::Queue, input_cache: &InputCache) {
    }

    fn resize(&mut self, device: &wgpu::Device, sc_desc: &wgpu::SurfaceConfiguration, _new_size: winit::dpi::PhysicalSize<u32>) {
    }

    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, input: &InputCache) {
    }
}

fn main() {
    
    jaankaup_core::template::run_loop::<BasicApp, BasicLoop, BasicFeatures>(); 
    println!("Finished...");
}
