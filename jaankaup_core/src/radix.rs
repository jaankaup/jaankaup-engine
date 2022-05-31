use crate::buffer::{buffer_from_data};

struct KeyBlock {
	key_offset: u32,
	key_count: u32,
	bucket_id: u32,
	bucket_offset: u32,
}

struct LocalSortBlock {
	bucket_id: u32,
	bucket_offset: u32,
	is_merged: u32,
}

struct Radix {
    buffer: wgpu::Buffer, 
    buffer2: wgpu::Buffer, 
}

impl Radix {

    pub fn init(device: &wgpu::Device, buffer_size: u32) -> Self {

        let buffer= buffer_from_data::<u32>(
            &device,
            &vec![0 ; buffer_size as usize],
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            None
        );

        let buffer2 = buffer_from_data::<u32>(
            &device,
            &vec![0 ; buffer_size as usize],
            wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            None
        );

        Self {
            buffer: buffer,
            buffer2: buffer2,
        }
    }
}
