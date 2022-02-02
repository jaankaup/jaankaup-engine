use crate::misc::Convert2Vec;
use bytemuck::Pod;
use wgpu::util::DeviceExt;

/// A struct that holds information for one draw call.
#[allow(dead_code)]
pub struct VertexBufferInfo {
    vertex_buffer_name: String,
    _index_buffer: Option<String>,
    start_index: u32,
    end_index: u32,
    instances: u32,
}

/// Create wgpu::buffer from data.
pub fn buffer_from_data<T: Pod>(
    device: &wgpu::Device,
    t: &[T],
    usage: wgpu::BufferUsages,
    _label: Option<String>)
    -> wgpu::Buffer {
        device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None, // TODO: label
                contents: bytemuck::cast_slice(&t),
                usage: usage,
            }
        )
}

/// Copy the content of the buffer into a vector. TODO: add range for reading buffer.
/// TODO: give res vector as parameter.
/// TODO: add _src_offset
pub fn to_vec<T: Convert2Vec>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    _src_offset: wgpu::BufferAddress,
    copy_size: wgpu::BufferAddress,
    ) -> Vec<T> {

    //log::info!("Creating staging buffer");
    // TODO: Recycle staging buffers.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: copy_size, 
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    // log::info!("Creating encoder");
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    
    // log::info!("Copy buffer to buffer");
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, copy_size);
    // log::info!("Submit");
    queue.submit(Some(encoder.finish()));

    let res: Vec<T>;
    
    let buffer_slice = staging_buffer.slice(..);
    //++let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
    let _ = buffer_slice.map_async(wgpu::MapMode::Read);
    device.poll(wgpu::Maintain::Wait);

    //++#[cfg(not(target_arch = "wasm32"))]
    //++{
    //++    pollster::block_on(buffer_future).unwrap(); 
    //++}

    // Not working.
    // #[cfg(target_arch = "wasm32")]
    // {
    //     //log::info!("Creating spawner");
    //     //let spawner = async_executor::LocalExecutor::new();
    //     //log::info!("Execute buffer_future");
    //     //spawner.run(buffer_future);
    //     wasm_bindgen_futures::spawn_local(async move {
    //         //log::info!("yeeeaaaah");
    //         if let Ok(()) = buffer_future.await { log::info!("Buffer future solved") }
    //     });
    // }

    let data = buffer_slice.get_mapped_range().to_vec();
    res = Convert2Vec::convert(&data);
    
    drop(data);
    staging_buffer.unmap();
    
    res
}
