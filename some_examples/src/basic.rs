use jaankaup_core::template::{
        WGPUFeatures,
        WGPUConfiguration,
        Application,
        BasicLoop,
};
use jaankaup_core::input::*;
use jaankaup_core::wgpu;
use jaankaup_core::winit;

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
}

#[allow(unused_variables)]
impl Application for BasicApp {

    fn init(configuration: &WGPUConfiguration) -> Self {

        BasicApp {
        }
    }

    fn render(&mut self,
              device: &wgpu::Device,
              queue: &mut wgpu::Queue,
              surface: &wgpu::Surface,
              sc_desc: &wgpu::SurfaceConfiguration) {

        // let frame = match surface.get_current_texture() {
        //     Ok(frame) => { frame },
        //     Err(_) => {
        //         surface.configure(&device, &sc_desc);
        //         surface.get_current_texture().expect("Failed to acquire next texture")
        //     },
        // };

        // let mut encoder = device.create_command_encoder(
        //     &wgpu::CommandEncoderDescriptor {
        //         label: Some("Render Encoder"),
        // });

        // let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

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
