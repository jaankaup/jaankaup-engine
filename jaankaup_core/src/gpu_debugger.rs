use crate::texture::Texture;
use std::mem::size_of;
use crate::render_object::{RenderObject, ComputeObject, create_bind_groups,draw, draw_indirect, DrawIndirect};
use std::borrow::Cow;
use crate::misc::Convert2Vec;
use crate::impl_convert;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use crate::common_functions::{
    udiv_up_safe32,
    create_uniform_bindgroup_layout,
    create_buffer_bindgroup_layout
};
use crate::buffer::{buffer_from_data};
use crate::camera::Camera;

use crate::histogram::Histogram;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Vertex {
    v: [f32; 4],
    n: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Triangle {
    a: Vertex,
    b: Vertex,
    c: Vertex,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AABB {
    min: [f32; 4],
    max: [f32; 4],
}

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
    draw_index: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ArrowAabbParams{
    max_number_of_vertices: u32,
    iterator_start_index: u32,
    iterator_end_index: u32,
    element_type: u32, // 0 :: array, 1 :: aabb, 2 :: aabb wire
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CharParams{
    iterator_start: u32,
    iterator_end: u32,
    number_of_threads: u32,
    draw_index: u32, 
    max_points_per_char: u32,
    max_number_of_vertices: u32,
}

impl_convert!{Arrow}
impl_convert!{Char}
impl_convert!{CharParams}

pub struct GpuDebugger {
    render_object_vvvvnnnn: RenderObject, 
    render_bind_groups_vvvvnnnn: Vec<wgpu::BindGroup>,
    render_object_vvvc: RenderObject,
    render_bind_groups_vvvc: Vec<wgpu::BindGroup>,
    compute_object_char: ComputeObject, 
    compute_bind_groups_char: Vec<wgpu::BindGroup>,
    compute_object_arrow: ComputeObject, 
    compute_bind_groups_arrow: Vec<wgpu::BindGroup>,
    compute_object_char_preprocessor: ComputeObject, 
    compute_bind_groups_char_preprocessor: Vec<wgpu::BindGroup>,
    buffers: HashMap<String, wgpu::Buffer>,
    arrow_aabb_params: ArrowAabbParams,
    histogram_draw_counts: Histogram,
    histogram_element_counter: Histogram,
    max_number_of_vertices: u32,
    max_number_of_chars: u32,
    max_number_of_arrows: u32,
    max_number_of_aabbs: u32,
    max_number_of_aabb_wires: u32,
    thread_count: u32,
}

impl GpuDebugger {

    pub fn get_output_chars_buffer(&self) -> &wgpu::Buffer {
        self.buffers.get(&"output_chars".to_string()).unwrap()
    }
    pub fn get_output_arrows_buffer(&self) -> &wgpu::Buffer {
        self.buffers.get(&"output_arrows".to_string()).unwrap()
    }
    pub fn get_output_aabbs_buffer(&self) -> &wgpu::Buffer {
        self.buffers.get(&"output_aabbs".to_string()).unwrap()
    }
    pub fn get_output_aabb_wires_buffer(&self) -> &wgpu::Buffer {
        self.buffers.get(&"output_aabb_wires".to_string()).unwrap()
    }

    pub fn Init(device: &wgpu::Device,
                sc_desc: &wgpu::SurfaceConfiguration,
                camera_buffer: &wgpu::Buffer,
                max_number_of_vertices: u32,
                max_number_of_chars: u32,
                max_number_of_arrows: u32,
                max_number_of_aabbs: u32,
                max_number_of_aabb_wires: u32,
                thread_count: u32
                ) -> Self {

        let mut buffers: HashMap<String, wgpu::Buffer> = HashMap::new();

        let histogram_draw_counts = Histogram::init(&device, &vec![0; 2]);

        // This must be given to the shaders that uses GpuDebugger.
        let histogram_element_counter = Histogram::init(&device, &vec![0; 4]);

        let arrow_aabb_params = ArrowAabbParams {
            max_number_of_vertices: 123 as u32,
            // max_number_of_vertices: VERTEX_BUFFER_SIZE as u32,
            iterator_start_index: 0,
            iterator_end_index: 0,
            element_type: 0,
        };

        ////////////////////////////////////////////////////
        ////                 BUFFERS                    ////
        ////////////////////////////////////////////////////

        buffers.insert(
            "indirect_buffer".to_string(),
                buffer_from_data::<DrawIndirect>(
                    &device,
                    &vec![DrawIndirect{ vertex_count: 0, instance_count: 1, base_vertex: 0, base_instance: 0, } ; 64],
                    wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDIRECT,
                    None
                )
        );

        buffers.insert(
            "char_params".to_string(),
                buffer_from_data::<CharParams>(
                    &device,
                    &vec![CharParams{ iterator_start: 0,
                                      iterator_end: 0,
                                      number_of_threads: 256,
                                      draw_index: 0,
                                      max_points_per_char: 5000,
                                      max_number_of_vertices: 123}],
                    wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    None
                )
        );

        buffers.insert(
            "output_arrows".to_string(),
            device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("output_arrays buffer"),
                size: (max_number_of_arrows * std::mem::size_of::<Arrow>() as u32) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
            )
        );

        buffers.insert(
            "output_chars".to_string(),
            device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("output_chars"),
                size: (max_number_of_chars * std::mem::size_of::<Char>() as u32) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
            )
        );

        buffers.insert(
            "output_aabbs".to_string(),
            device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("output_aabbs"),
                size: (max_number_of_aabbs * std::mem::size_of::<AABB>() as u32) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
            )
        );

        buffers.insert(
            "output_aabb_wires".to_string(),
            device.create_buffer(&wgpu::BufferDescriptor{
                label: Some("output_aabbs"),
                size: (max_number_of_aabb_wires * std::mem::size_of::<AABB>() as u32) as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
                }
            )
        );

        println!("max_number_of_vertices == {}", max_number_of_vertices);

        buffers.insert(
            "output_render".to_string(),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_debug draw buffer"),
                size: (max_number_of_vertices * size_of::<Vertex>() as u32) as u64, 
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        );

        buffers.insert(
            "arrow_aabb_params".to_string(),
            buffer_from_data::<ArrowAabbParams>(
            &device,
            &vec![arrow_aabb_params],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            None)
        );

        ////////////////////////////////////////////////////
        ////               Render vvvvnnnn              ////
        ////////////////////////////////////////////////////

        let render_object_vvvvnnnn =
                RenderObject::init(
                    &device,
                    &sc_desc,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("renderer_v4n4_debug_visualizator.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/renderer_v4n4_debug_visualizator.wgsl"))),
                    
                    }),
                    &vec![wgpu::VertexFormat::Float32x4, wgpu::VertexFormat::Float32x4],
                    &vec![
                        vec![
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::VERTEX),
                        ],
                    ],
                    Some("Debug visualizator vvvvnnnn renderer with camera."),
                    true,
                    wgpu::PrimitiveTopology::TriangleList
        );
        let render_bind_groups_vvvvnnnn = create_bind_groups(
                                     &device,
                                     &render_object_vvvvnnnn.bind_group_layout_entries,
                                     &render_object_vvvvnnnn.bind_group_layouts,
                                     &vec![
                                          vec![
                                              &camera_buffer.as_entire_binding(),
                                         ]
                                     ]
        );

        // vvvc
        let render_object_vvvc =
                RenderObject::init(
                    &device,
                    &sc_desc,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("renderer_v3c1.wgsl"),
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
                                     &device,
                                     &render_object_vvvc.bind_group_layout_entries,
                                     &render_object_vvvc.bind_group_layouts,
                                     &vec![
                                          vec![
                                              &camera_buffer.as_entire_binding(),
                                         ]
                                     ]
        );

        ////////////////////////////////////////////////////
        ////                 Compute char               ////
        ////////////////////////////////////////////////////

        let compute_object_char =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("numbers.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/numbers.wgsl"))),
                    
                    }),
                    Some("Numbers Compute object"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> camerauniform: Camera;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<uniform> arrow_aabb_params: VisualizationParams;
                            create_uniform_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE),

                            // // @group(0) @binding(2) var<storage, read_write> counter: Counter;
                            // create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),
                            // @group(0) @binding(2) var<storage, read_write> indirec: array<DrawIndirect>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(3) var<storage, read> counter: array<Arrow>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(4) var<storage,read_write> output: array<VertexBuffer>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ]
        );

        let compute_bind_groups_char = create_bind_groups(
                                      &device,
                                      &compute_object_char.bind_group_layout_entries,
                                      &compute_object_char.bind_group_layouts,
                                      &vec![
                                          vec![
                                              &camera_buffer.as_entire_binding(),
                                              &buffers.get(&"arrow_aabb_params".to_string()).unwrap().as_entire_binding(),
                                              &buffers.get(&"indirect_buffer".to_string()).unwrap().as_entire_binding(),
                                              // &histogram_draw_counts.get_histogram_buffer().as_entire_binding(),
                                              &buffers.get(&"output_chars".to_string()).unwrap().as_entire_binding(),
                                              &buffers.get(&"output_render".to_string()).unwrap().as_entire_binding()
                                          ]
                                      ]
        );

        ////////////////////////////////////////////////////
        ////                 Char preprocessor          ////
        ////////////////////////////////////////////////////

        let compute_object_char_preprocessor =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("char_preprocessor.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/char_preprocessor.wgsl"))),
                    
                    }),
                    Some("Char preprocessor Compute object"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> camera: Camera;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<storage, read_write> char_params: array<CharParams>;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(2) var<storage, read_write> indirect: array<DrawIndirect>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(3) var<storage,read_write> chars_array: array<Char>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ]
        );

        let compute_bind_groups_char_preprocessor =
                create_bind_groups(
                    &device,
                    &compute_object_char_preprocessor.bind_group_layout_entries,
                    &compute_object_char_preprocessor.bind_group_layouts,
                    &vec![
                        vec![
                            &camera_buffer.as_entire_binding(),
                            &buffers.get(&"char_params".to_string()).unwrap().as_entire_binding(),
                            &buffers.get(&"indirect_buffer".to_string()).unwrap().as_entire_binding(),
                            &buffers.get(&"output_chars".to_string()).unwrap().as_entire_binding(),
                        ]
                    ]
        );


        ////////////////////////////////////////////////////
        ////               Compute arrow/aabb           ////
        ////////////////////////////////////////////////////

        let compute_object_arrow =
                ComputeObject::init(
                    &device,
                    &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
                        label: Some("arrow_aabb.wgsl"),
                        source: wgpu::ShaderSource::Wgsl(
                            Cow::Borrowed(include_str!("../../assets/shaders/arrow_aabb.wgsl"))),
                    
                    }),
                    Some("Arrow_aabb Compute object"),
                    &vec![
                        vec![
                            // @group(0) @binding(0) var<uniform> arrow_aabb_params: VisualizationParams;
                            create_uniform_bindgroup_layout(0, wgpu::ShaderStages::COMPUTE),

                            // @group(0) @binding(1) var<storage, read_write> counter: Counter;
                            create_buffer_bindgroup_layout(1, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(2) var<storage, read> counter: array<Arrow>;
                            create_buffer_bindgroup_layout(2, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(3) var<storage, read_write> aabbs: array<AABB>;
                            create_buffer_bindgroup_layout(3, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(4) var<storage, read_write> aabb_wires: array<AABB>;
                            create_buffer_bindgroup_layout(4, wgpu::ShaderStages::COMPUTE, false),

                            // @group(0) @binding(5) var<storage,read_write> output: array<Triangle>;
                            create_buffer_bindgroup_layout(5, wgpu::ShaderStages::COMPUTE, false),
                        ],
                    ]
        );

        let compute_bind_groups_arrow = create_bind_groups(
                                      &device,
                                      &compute_object_arrow.bind_group_layout_entries,
                                      &compute_object_arrow.bind_group_layouts,
                                      &vec![
                                          vec![
                                               &buffers.get(&"arrow_aabb_params".to_string()).unwrap().as_entire_binding(),
                                               &histogram_draw_counts.get_histogram_buffer().as_entire_binding(),
                                               &buffers.get(&"output_arrows".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output_aabbs".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output_aabb_wires".to_string()).unwrap().as_entire_binding(),
                                               &buffers.get(&"output_render".to_string()).unwrap().as_entire_binding()
                                          ]
                                      ]
        );

        Self {
            render_object_vvvvnnnn: render_object_vvvvnnnn, 
            render_bind_groups_vvvvnnnn: render_bind_groups_vvvvnnnn,
            render_object_vvvc: render_object_vvvc,
            render_bind_groups_vvvc: render_bind_groups_vvvc,
            compute_object_char: compute_object_char,
            compute_bind_groups_char: compute_bind_groups_char,
            compute_object_arrow: compute_object_arrow, 
            compute_bind_groups_arrow: compute_bind_groups_arrow,
            compute_object_char_preprocessor: compute_object_char_preprocessor, 
            compute_bind_groups_char_preprocessor: compute_bind_groups_char_preprocessor,
            buffers: buffers,
            arrow_aabb_params: arrow_aabb_params,
            histogram_draw_counts: histogram_draw_counts,
            histogram_element_counter: histogram_element_counter,
            max_number_of_vertices: max_number_of_vertices,
            max_number_of_chars: max_number_of_chars,
            max_number_of_arrows: max_number_of_arrows,
            max_number_of_aabbs: max_number_of_aabbs,
            max_number_of_aabb_wires: max_number_of_aabb_wires,
            thread_count: thread_count,
        }
    }

    pub fn reset_element_counters(&mut self, queue: &wgpu::Queue) {
        // Reset counter.
        self.histogram_element_counter.reset_all_cpu_version(queue, 0);
    }

    pub fn get_element_counter_buffer(&self) -> &wgpu::Buffer {
        &self.histogram_element_counter.get_histogram_buffer() 
    }

    // draw indirect! Reduce finish() calls.
    pub fn render(&mut self,
                  device: &wgpu::Device,
                  queue: &wgpu::Queue,
                  view: &wgpu::TextureView,
                  depth_texture: &Texture,
                  clear: &mut bool) {

        // Get the total number of elements.
        let elem_counter = self.histogram_element_counter.get_values(device, queue);

        let total_number_of_arrows = elem_counter[1];
        let total_number_of_aabbs = elem_counter[2];
        let total_number_of_aabb_wires = elem_counter[3];
        
        let vertices_per_element_arrow = 72;
        let vertices_per_element_aabb = 36;
        let vertices_per_element_aabb_wire = 432;

        // The number of vertices created with one dispatch.
        let vertices_per_dispatch_arrow = self.thread_count * vertices_per_element_arrow;
        let vertices_per_dispatch_aabb = self.thread_count * vertices_per_element_aabb;
        let vertices_per_dispatch_aabb_wire = self.thread_count * vertices_per_element_aabb_wire;

        // [(element_type, total number of elements, number of vercies per dispatch, vertices_per_element)]
        let draw_params = [(0, total_number_of_arrows,     vertices_per_dispatch_arrow, vertices_per_element_arrow),
                           (1, total_number_of_aabbs,      vertices_per_dispatch_aabb, vertices_per_element_aabb), // !!!
                           (2, total_number_of_aabb_wires, vertices_per_dispatch_aabb_wire, vertices_per_element_aabb_wire)]; 

        // For each element type, create triangle meshes and render with respect of draw buffer size.
        for (e_type, e_size, v_per_dispatch, vertices_per_elem) in draw_params.iter() {

            // The number of safe dispathes. This ensures the draw buffer doesn't over flow.
            let safe_number_of_dispatches = self.max_number_of_vertices as u32 / v_per_dispatch;

            // // The number of remaining dispatches to complete the triangle mesh creation and
            // // rendering.
            // let mut total_number_of_dispatches = udiv_up_safe32(*e_size, thread_count);

            // The number of items to create and draw.
            let mut items_to_process = *e_size;

            // Nothing to process.
            if *e_size == 0 { continue; }

            // Create the initial params.
            self.arrow_aabb_params.iterator_start_index = 0;
            self.arrow_aabb_params.iterator_end_index = std::cmp::min(*e_size, safe_number_of_dispatches * v_per_dispatch);
            self.arrow_aabb_params.element_type = *e_type;

            queue.write_buffer(
                &self.buffers.get(&"arrow_aabb_params".to_string()).unwrap(),
                0,
                bytemuck::cast_slice(&[self.arrow_aabb_params])
            );

            // Continue process until all element are rendered.
            while items_to_process > 0 {

                // The number of remaining dispatches to complete the triangle mesh creation and
                // rendering.
                let total_number_of_dispatches = udiv_up_safe32(items_to_process, self.thread_count);

                // Calculate the number of dispatches for this run. 
                let local_dispatch = std::cmp::min(total_number_of_dispatches, safe_number_of_dispatches);

                // Then number of elements that are going to be rendered. 
                let number_of_elements = std::cmp::min(local_dispatch * self.thread_count, items_to_process);

                let mut encoder_arrow_aabb = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("arrow_aabb ... ") });

                self.arrow_aabb_params.iterator_end_index = self.arrow_aabb_params.iterator_start_index + std::cmp::min(number_of_elements, safe_number_of_dispatches * v_per_dispatch);

                queue.write_buffer(
                    &self.buffers.get(&"arrow_aabb_params".to_string()).unwrap(),
                    0,
                    bytemuck::cast_slice(&[self.arrow_aabb_params])
                );

                self.compute_object_arrow.dispatch(
                    &self.compute_bind_groups_arrow,
                    &mut encoder_arrow_aabb,
                    local_dispatch, 1, 1, Some("arrow local dispatch")
                );

                queue.submit(Some(encoder_arrow_aabb.finish()));

                let counter = self.histogram_draw_counts.get_values(device, queue);

                let draw_count = number_of_elements * vertices_per_elem;

                // println!("local_dispatch == {}", local_dispatch);
                // println!("draw_count == {}", draw_count);

                let mut encoder_arrow_rendering = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("arrow rendering ... ") });

                draw(&mut encoder_arrow_rendering,
                     &view,
                     &depth_texture,
                     &self.render_bind_groups_vvvvnnnn,
                     &self.render_object_vvvvnnnn.pipeline,
                     &self.buffers.get("output_render").unwrap(),
                     0..draw_count,
                     *clear
                );
               
                if *clear { *clear = false; }

                // Decrease the total count of elements.
                items_to_process = items_to_process - number_of_elements; 

                queue.submit(Some(encoder_arrow_rendering.finish()));
                self.histogram_draw_counts.reset_all_cpu_version(queue, 0);

                self.arrow_aabb_params.iterator_start_index = self.arrow_aabb_params.iterator_end_index; // + items_to_process;
            } // for number_of_loop
        }

        //// DRAW CHARS

        // Update Visualization params for arrows.


        // TODO:
          
        let number_of_chars = elem_counter[0];

        self.arrow_aabb_params.iterator_start_index = 0;
        self.arrow_aabb_params.iterator_end_index = number_of_chars;
        self.arrow_aabb_params.element_type = 0;
        // TODO: a better heurestic. This doesn't work as expected.
        self.arrow_aabb_params.max_number_of_vertices = std::cmp::max(self.max_number_of_vertices as i32 - 500000, 0) as u32;
        //self.arrow_aabb_params.max_number_of_vertices = MAX_NUMBER_OF_VVVC as u32;

        queue.write_buffer(
            &self.buffers.get(&"arrow_aabb_params".to_string()).unwrap(),
            0,
            bytemuck::cast_slice(&[self.arrow_aabb_params])
        );

        self.histogram_draw_counts.reset_all_cpu_version(queue, 0);

        let mut current_char_index = 0;

        // let mut ccc = -1;
        loop { 
            // ccc = ccc + 1;

            let mut encoder_char = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("numbers encoder") });

            self.compute_object_char.dispatch(
                &self.compute_bind_groups_char,
                &mut encoder_char,
                // 1, 1, 1, Some("numbers dispatch")
                number_of_chars, 1, 1, Some("numbers dispatch")
            );

            queue.submit(Some(encoder_char.finish()));

            let counter = self.histogram_draw_counts.get_values(device, queue);
            // self.draw_count_triangles = counter[0];
            let draw_count_triangles = counter[0];

            current_char_index = counter[1];

            let mut encoder_char_rendering = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("numbers encoder") });

            draw(&mut encoder_char_rendering,
                 &view,
                 &depth_texture,
                 &self.render_bind_groups_vvvc,
                 &self.render_object_vvvc.pipeline,
                 &self.buffers.get("output_render").unwrap(),
                 0..draw_count_triangles.min(self.max_number_of_vertices * 2),
                 *clear
            );
            queue.submit(Some(encoder_char_rendering.finish()));

            self.histogram_draw_counts.reset_all_cpu_version(queue, 0);

            // There are still chars left.
            if current_char_index != 0 && current_char_index != self.arrow_aabb_params.iterator_end_index - 1 {
                self.arrow_aabb_params.iterator_start_index = current_char_index;
                self.arrow_aabb_params.iterator_end_index = number_of_chars;
                self.arrow_aabb_params.element_type = 0;
                self.arrow_aabb_params.max_number_of_vertices = std::cmp::max((self.max_number_of_vertices * 2) as i32 - 500000, 0) as u32;

                queue.write_buffer(
                    &self.buffers.get(&"arrow_aabb_params".to_string()).unwrap(),
                    0,
                    bytemuck::cast_slice(&[self.arrow_aabb_params])
                );

            }
            else { break; }
        }
    }
}
