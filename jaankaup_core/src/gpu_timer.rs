use std::mem::size_of;
use crate::impl_convert;
use crate::misc::Convert2Vec;
use bytemuck::{Pod, Zeroable};
use std::convert::TryInto;
use crate::buffer::{to_vec}; 

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable,Debug)]
pub struct TimestampData {
    start: u64,
    end: u64,
}

impl_convert!{TimestampData}

// #[repr(C)]
// #[derive(Clone, Copy, Pod, Zeroable,Debug)]
// struct QueryData {
//     timestamps: [TimestampData; TIME_STAMP_COUNT as usize],
// }
// 
// impl_convert!{QueryData}

/// A timer for measuring gpu operations.
pub struct GpuTimer {
    timestamp: wgpu::QuerySet,
    timestamp_period: f32,
    query_buffer: wgpu::Buffer,
    max_number_of_time_stamps: u32,
    stamps_written: u32,
    data: Vec<TimestampData>,
}

impl GpuTimer {

    /// Create GpuTimer if wgpu::Features::TIMESTAMP_QUERY is set and supported.
    /// If the feature is not in use, return None.
    pub fn init(device: &wgpu::Device, queue: &wgpu::Queue, timestamp_count: u32, label: wgpu::Label) -> Option<Self> {

        // #[cfg(not(debug_assertions))]
        #[cfg(debug_assertions)]
        {
            debug_assert!(timestamp_count != 0, "timestamp_count must be greater than zero");
        }

        let data = vec![TimestampData {start: 0, end: 0, }; timestamp_count.try_into().unwrap()];


        // Create queries for time stamps.
        let result = if device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY) {

                let timestamp = device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: label,
                    count: timestamp_count * 2, // count * numb(begining/end)
                    ty: wgpu::QueryType::Timestamp,
                });

                let timestamp_period = queue.get_timestamp_period();

                //println!("mem::size_of::<QueryData>() == {}", mem::size_of::<QueryData>());
                // TODO: Try to implement buffer reading without 'wgpu::BufferUsages::COPY_SRC'
                let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Query buffer"),
                    size: (size_of::<TimestampData>() * timestamp_count as usize) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                Some(Self {
                    timestamp: timestamp,
                    timestamp_period: timestamp_period,
                    query_buffer: query_buffer,
                    max_number_of_time_stamps: timestamp_count,
                    stamps_written: 0,
                    data: data,
                })
            }
        else {
            None
        };
        result
    }

    pub fn start(&mut self, encoder: &mut wgpu::CommandEncoder, time_stamp_number: u32) {

        #[cfg(debug_assertions)]
        {
            debug_assert!(time_stamp_number < self.max_number_of_time_stamps , "{} < {}", time_stamp_number, self.max_number_of_time_stamps);
        }
        encoder.write_timestamp(&self.timestamp, time_stamp_number * 2);
    }

    pub fn end(&mut self, encoder: &mut wgpu::CommandEncoder, time_stamp_number: u32) {

        #[cfg(debug_assertions)]
        {
            debug_assert!(time_stamp_number < self.max_number_of_time_stamps , "{} < {}", time_stamp_number, self.max_number_of_time_stamps);
        }
        encoder.write_timestamp(&self.timestamp, time_stamp_number * 2 + 1);
    }

    pub fn resolve_timestamps(&self, encoder: &mut wgpu::CommandEncoder) {

        encoder.resolve_query_set(
            &self.timestamp,
            0..self.max_number_of_time_stamps * 2,
            &self.query_buffer,
            0,
            );
    }

    pub fn create_timestamp_data(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.data = 
            to_vec::<TimestampData>(&device,
                                &queue,
                                &self.query_buffer,
                                0 as wgpu::BufferAddress,
                                (size_of::<TimestampData>() * self.max_number_of_time_stamps as usize) as wgpu::BufferAddress
        );
    }

    pub fn get_data(&self) -> Vec<TimestampData> {
        self.data.clone()
    }

    pub fn print_data(&self) {

        for (i, elem) in self.data.iter().enumerate() {
            let nanoseconds =
                (elem.end - elem.start) as f32 * self.timestamp_period;
            let microseconds = nanoseconds / 1000.0;
            let milli = microseconds / 1000.0;
            println!("{:?} time is {:?} milli seconds.", i, milli);
        }
    }
}
