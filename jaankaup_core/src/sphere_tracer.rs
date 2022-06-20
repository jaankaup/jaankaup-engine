use crate::common_functions::{create_uniform_bindgroup_layout, create_buffer_bindgroup_layout};
use crate::render_object::{ComputeObject, create_bind_groups};
use crate::gpu_debugger::GpuDebugger;
use std::borrow::Cow;

#[allow(dead_code)]
#[derive(PartialEq, Debug, Clone, Copy)]
struct SphereTracerParams {
    inner_dim: [u32; 3],
    padding: u32,
    outer_dim: [u32; 3],
    padding2: u32,
}

struct SphereTracer {

    /// Sphere tracer compute object. 
    st_object: ComputeObject,
}

impl SphereTracer {

    pub fn init(device: &wgpu::Device,
                inner_dimension: [u32; 3],
                outer_dimension: [u32; 3],
                gpu_debugger: &Option<&GpuDebugger>,
                ) -> Self {

        let st_object =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("sphere_tracer.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/sphere_tracer.wgsl"))),

                    }),
                    Some("Sphere tracer ComputeObject"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> fmm_params: FmmParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<uniform> sphere_tracer_params: SphereTracerParams;
                            create_uniform_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(2) var<uniform> camera: RayCamera;
                            create_uniform_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE),
                              
                            // @group(0) @binding(3) var<storage,read_write> fmm_data: array<FmmCellPc>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),
                              
                            // @group(0) @binding(4) var<storage,read_write> screen_output: array<RayOutput>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),

                        ],
                        // vec![

                        //     // @group(1) @binding(0) var<storage, read_write> counter: array<atomic<u32>>;
                        //     create_buffer_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE, false),

                        //     // @group(1) @binding(1) var<storage,read_write>  output_char: array<Char>;
                        //     create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                        //     // @group(1) @binding(2) var<storage,read_write>  output_arrow: array<Arrow>;
                        //     create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                        //     // @group(1) @binding(3) var<storage,read_write>  output_aabb: array<AABB>;
                        //     create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                        //     // @group(1) @binding(4) var<storage,read_write>  output_aabb_wire: array<AABB>;
                        //     create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
                        // ],

                    ],
                    &"main".to_string(),
                    None,
        );

        Self {
            st_object: st_object,
        }
    }
}
