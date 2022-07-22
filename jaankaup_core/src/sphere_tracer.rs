use std::mem::size_of;
use bytemuck::{Pod, Zeroable};
use crate::buffer::buffer_from_data;
use crate::common_functions::{create_uniform_bindgroup_layout, create_buffer_bindgroup_layout};
use crate::render_object::{ComputeObject, create_bind_groups};
use crate::gpu_debugger::GpuDebugger;
use std::borrow::Cow;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SphereTracerParams {
    inner_dim: [u32; 2],
    outer_dim: [u32; 2],
    render_rays: u32,
    render_samplers: u32,
    padding: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RayOutput {
    origin: [f32; 3],
    visibility: f32,
    intersection_point: [f32; 3],
    opacity: f32,
    normal: [f32; 3],
    diffuce_color: u32,
}

pub struct SphereTracer {

    /// Sphere tracer compute object. 
    st_compute_object: ComputeObject,
    st_bind_groups: Vec<wgpu::BindGroup>,
    // output_buffer: wgpu::Buffer,
    output_buffer_color: wgpu::Buffer,
    sphere_tracer_params: SphereTracerParams,
    sphere_tracer_buffer: wgpu::Buffer,
}

impl SphereTracer {

    pub fn init(device: &wgpu::Device,
                inner_dimension: [u32; 2],
                outer_dimension: [u32; 2],
                render_rays: bool,
                render_samplers: bool,
                fmm_params: &wgpu::Buffer,
                fmm_data: &wgpu::Buffer,
                camera_buffer: &wgpu::Buffer,
                gpu_debugger: &Option<&GpuDebugger>,
                ) -> Self {

        let st_compute_object =
                ComputeObject::init(
                    &device,
                    //&device.create_shader_module(wgpu::ShaderModuleDescriptor {
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
                              
                            // // @group(0) @binding(4) var<storage,read_write> screen_output: array<RayOutput>;
                            // create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(5) var<storage,read_write> screen_output_color: array<u32>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
                        ],
                        vec![

                            // @group(1) @binding(0) var<storage, read_write> counter: array<atomic<u32>>;
                            create_buffer_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE, false),

                            // @group(1) @binding(1) var<storage,read_write>  output_char: array<Char>;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // @group(1) @binding(2) var<storage,read_write>  output_arrow: array<Arrow>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                            // @group(1) @binding(3) var<storage,read_write>  output_aabb: array<AABB>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            // @group(1) @binding(4) var<storage,read_write>  output_aabb_wire: array<AABB>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
                        ],

                    ],
                    &"main".to_string(),
                    None,
        );

        let sphere_tracer_params = SphereTracerParams {
            inner_dim: inner_dimension,
            outer_dim: outer_dimension,
            render_rays: if render_rays { 1 } else { 0 },
            render_samplers: if render_samplers { 1 } else { 0 },
            padding: [0, 0],
        };

        let sphere_tracer_buffer = buffer_from_data::<SphereTracerParams>(
                &device,
                &vec![sphere_tracer_params],
                wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                None
        );

        // let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        //     label: Some("Sphere tracer output"),
        //     size: (inner_dimension[0] * inner_dimension[1] * outer_dimension[0] * outer_dimension[1]) as u64 * size_of::<RayOutput>() as u64,
        //     usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        //     mapped_at_creation: false,
        // });

        let output_buffer_color = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sphere tracer output color"),
            size: (inner_dimension[0] * inner_dimension[1] * outer_dimension[0] * outer_dimension[1]) as u64 * size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let st_bind_groups = create_bind_groups(
                &device,
                &st_compute_object.bind_group_layout_entries,
                &st_compute_object.bind_group_layouts,
                &vec![
                    vec![
                        &fmm_params.as_entire_binding(),
                        &sphere_tracer_buffer.as_entire_binding(),
                        &camera_buffer.as_entire_binding(),
                        &fmm_data.as_entire_binding(),
                        // &output_buffer.as_entire_binding(),
                        &output_buffer_color.as_entire_binding(),
                    ],
                    vec![
                        &gpu_debugger.unwrap().get_element_counter_buffer().as_entire_binding(),
                        &gpu_debugger.unwrap().get_output_chars_buffer().as_entire_binding(),
                        &gpu_debugger.unwrap().get_output_arrows_buffer().as_entire_binding(),
                        &gpu_debugger.unwrap().get_output_aabbs_buffer().as_entire_binding(),
                        &gpu_debugger.unwrap().get_output_aabb_wires_buffer().as_entire_binding(),
                    ],
                ]
        );

        Self {
            st_compute_object: st_compute_object,
            st_bind_groups: st_bind_groups, 
            // output_buffer: output_buffer,
            output_buffer_color: output_buffer_color,
            sphere_tracer_params: sphere_tracer_params,
            sphere_tracer_buffer: sphere_tracer_buffer,
        }
    }

    pub fn show_rays(&mut self, queue: &wgpu::Queue, value: bool) {

        self.sphere_tracer_params.render_rays = if value { 1 } else { 0 };

        queue.write_buffer(
            &self.sphere_tracer_buffer,
            0,
            bytemuck::cast_slice(&[self.sphere_tracer_params])
        );
    }

    pub fn get_color_buffer(&self) -> &wgpu::Buffer {
        &self.output_buffer_color
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let number_of_dispatches = self.sphere_tracer_params.outer_dim[0] * self.sphere_tracer_params.outer_dim[1];

        let mut pass = encoder.begin_compute_pass(
            &wgpu::ComputePassDescriptor { label: Some("Sphere tracer dispatch.")}
        );

        pass.set_pipeline(&self.st_compute_object.pipeline);

        for (e, bgs) in self.st_bind_groups.iter().enumerate() {
            pass.set_bind_group(e as u32, &bgs, &[]);
        }

        pass.dispatch_workgroups(number_of_dispatches, 1, 1);

        drop(pass);
    }
    pub fn get_width(&self) -> u32 {
        self.sphere_tracer_params.inner_dim[0] * self.sphere_tracer_params.outer_dim[0]
    }

    pub fn get_height(&self) -> u32 {
        self.sphere_tracer_params.inner_dim[1] * self.sphere_tracer_params.outer_dim[1]
    }
}
