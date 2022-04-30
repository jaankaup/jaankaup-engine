use crate::misc::Convert2Vec;
use bytemuck::Pod;
use wgpu::util::DeviceExt;
use crate::template::Spawner;
use std::{thread, time};

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
    label: wgpu::Label)
    -> wgpu::Buffer {
        device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: label,
                contents: bytemuck::cast_slice(&t),
                usage: usage,
            }
        )
}

/// Copy the content of the buffer into a vector.
pub fn to_vec<T: Convert2Vec + std::clone::Clone + bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    _src_offset: wgpu::BufferAddress,
    copy_size: wgpu::BufferAddress,
    _spawner: &Spawner,
    ) -> Vec<T> {

    // TODO: Recycle staging buffers.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: copy_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, copy_size);
    queue.submit(Some(encoder.finish()));

    let res: Vec<T>;

    let buffer_slice = staging_buffer.slice(..);
    let slice = buffer_slice.map_async(wgpu::MapMode::Read);
    device.poll(wgpu::Maintain::Wait);
    // let _ = buffer_slice.map_async(wgpu::MapMode::Read);

    _spawner.spawn_local(async {
        slice.await.unwrap()
    });

    device.map_buffer();

    //log::info!("{:?}", staging_buffer.map_state);
    log::info!("{:?}", buffer_slice);

    // thread::sleep(time::Duration::from_millis(10));

    // Wasm version crashes: DOMException.getMappedRange: Buffer not mapped.
    let data = buffer_slice.get_mapped_range().to_vec();
    res = Convert2Vec::convert(&data);
    drop(data);
    staging_buffer.unmap();

    res
}
