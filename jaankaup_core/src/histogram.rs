// use jaankaup_core::wgpu;
use crate::buffer::{buffer_from_data, to_vec};

/// Histogram struct for GPU purposes. 
pub struct Histogram {
    histogram: wgpu::Buffer,
    data: Vec<u32>,
}

impl Histogram {

    /// Create histogram with initial values.
    pub fn init(device: &wgpu::Device, initial_values: &Vec<u32>) -> Self {

        assert!(initial_values.len() > 0, "{}", format!("{} > 0", initial_values.len()));

        let histogram = buffer_from_data::<u32>(
            &device,
            &initial_values,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            None);

        Self {
            histogram: histogram,
            data: initial_values.to_vec(),
        }
    }

    /// TODO: implement wasm version! 
    pub fn get_values(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u32> {

        let result = to_vec::<u32>(&device,
                                   &queue,
                                   &self.histogram,
                                   0 as wgpu::BufferAddress,
                                   (std::mem::size_of::<u32>() * self.data.len()) as wgpu::BufferAddress);
        result
    }
    
    pub fn get_histogram_buffer(&self) -> &wgpu::Buffer {
        &self.histogram
    }

    pub fn set_values_cpu_version(&self, queue: &wgpu::Queue, value: &Vec<u32>)
    {
        // Make sure the updated values are the same size as old values.
        assert!(value.len() == self.data.len(), "{}", format!("{} > {}", self.data.len(), value.len()));

        queue.write_buffer(
            &self.histogram,
            0,
            bytemuck::cast_slice(&value)
        );
    }

    pub fn reset_all_cpu_version(&self, queue: &wgpu::Queue, value: u32) {
        queue.write_buffer(
            &self.histogram,
            0,
            bytemuck::cast_slice(&vec![value ; self.data.len() as usize])
        );
    }
}
