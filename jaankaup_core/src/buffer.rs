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
    //++ let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    //++ let _ = buffer_slice.map_async(wgpu::MapMode::Read, true);
    // let _ = buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    buffer_slice.map_async(wgpu::MapMode::Read, move |_| ());
    device.poll(wgpu::Maintain::Wait);

    // Wasm version crashes: DOMException.getMappedRange: Buffer not mapped.
    let data = buffer_slice.get_mapped_range().to_vec();
    res = Convert2Vec::convert(&data);
    drop(data);
    staging_buffer.unmap();

    res
}

//++ /// Copy the content of the buffer into a vector.
//++ pub fn to_vec<T: Convert2Vec + std::clone::Clone + bytemuck::Pod>(
//++     device: &wgpu::Device,
//++     queue: &wgpu::Queue,
//++     buffer: &wgpu::Buffer,
//++     _src_offset: wgpu::BufferAddress,
//++     copy_size: wgpu::BufferAddress,
//++     _spawner: &Spawner,
//++     ) -> Vec<T> {
//++ 
//++     log::info!("to_vec");
//++ 
//++     // TODO: Recycle staging buffers.
//++     let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
//++         label: None,
//++         size: copy_size,
//++         usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
//++         mapped_at_creation: false,
//++     });
//++ 
//++     let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
//++     encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, copy_size);
//++     queue.submit(Some(encoder.finish()));
//++ 
//++     // let res: Vec<T>;
//++ 
//++     log::info!("pah");
//++     let buffer_slice = staging_buffer.slice(..);
//++     log::info!("pah2");
//++     let slice = buffer_slice.map_async(wgpu::MapMode::Read);
//++     log::info!("pah3");
//++     device.poll(wgpu::Maintain::Wait);
//++     log::info!("pah4");
//++     //let _ = buffer_slice.map_async(wgpu::MapMode::Read);
//++     log::info!("pah5");
//++ 
//++     log::info!("before spawn_local");
//++     _spawner.spawn_local(async {
//++         log::info!("spawn_local");
//++         slice.await.unwrap()
//++     });
//++     log::info!("after spawn local");
//++ 
//++     // device.map_buffer();
//++ 
//++     //log::info!("{:?}", staging_buffer.map_state);
//++     //log::info!("{:?}", buffer_slice);
//++ 
//++     // thread::sleep(time::Duration::from_millis(10));
//++ 
//++     // Wasm version crashes: DOMException.getMappedRange: Buffer not mapped.
//++ 
//++     let res = temp_function(&buffer_slice);
//++     //++ let data = buffer_slice.get_mapped_range().to_vec();
//++     //++ res = Convert2Vec::convert(&data);
//++     //++ drop(data);
//++     log::info!("res created");
//++     staging_buffer.unmap();
//++     log::info!("staging_buffer unmapped");
//++ 
//++     res
//++ }
//++ 
//++ fn temp_function<T: Convert2Vec + std::clone::Clone + bytemuck::Pod>(bs: &wgpu::BufferSlice) -> Vec<T> {
//++     // let res: Vec<T>;
//++     log::info!("temp_function");
//++     let data = bs.get_mapped_range().to_vec();
//++     let result = Convert2Vec::convert(&data);
//++     drop(data);
//++     result
//++ }
