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

/// A timer for measuring gpu operations.
pub struct GpuTimer {
    timestamp: wgpu::QuerySet,
    timestamp_period: f32,
    query_buffer: wgpu::Buffer,
    max_number_of_time_stamps: u32,
    start_count: u32,
    end_count: u32,
    data: Vec<TimestampData>,
}

impl GpuTimer {

    /// Create GpuTimer if wgpu::Features::TIMESTAMP_QUERY is set and supported.
    pub fn init(device: &wgpu::Device, queue: &wgpu::Queue, timestamp_count: u32, label: wgpu::Label) -> Option<Self> {

        // #[cfg(not(debug_assertions))]
        #[cfg(debug_assertions)]
        {
            debug_assert!(timestamp_count != 0, "timestamp_count must be greater than zero");
        }

        // let mut query_hash_table: HashMap<u32, (bool, bool)> = HashMap::new(); 

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

                // TODO: Try to implement buffer reading without 'wgpu::BufferUsages::COPY_SRC'
                let query_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Query buffer"),
                    size: (size_of::<TimestampData>() * timestamp_count as usize) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                // for i in 0..timestamp_count {
                //     query_hash_table.insert(i, (false, false)); 
                // }

                Some(Self {
                    timestamp: timestamp,
                    timestamp_period: timestamp_period,
                    query_buffer: query_buffer,
                    max_number_of_time_stamps: timestamp_count,
                    start_count: 0,
                    end_count: 0,
                    data: data,
                })
            }
        else {
            None
        };
        result
    }

    /// Add start. For each start there must be end counter part.
    pub fn start_pass(&mut self, pass: &mut wgpu::ComputePass) {
        #[cfg(debug_assertions)]
        {
            debug_assert!(self.start_count < self.max_number_of_time_stamps, "{} < {}", self.start_count, self.max_number_of_time_stamps);
            debug_assert!(self.start_count - self.end_count < 2, "{} - {} < 2", self.start_count, self.end_count);
        }
        pass.write_timestamp(&self.timestamp, self.start_count * 2);
        self.start_count = self.start_count + 1;
    }

    /// Add end. For each end there must be end counter part.
    pub fn end_pass(&mut self, pass: &mut wgpu::ComputePass) {

        self.end_count = self.end_count + 1;
        #[cfg(debug_assertions)]
        {
            debug_assert!(self.start_count == self.end_count , "{} == {}", self.start_count, self.end_count);
            debug_assert!(self.end_count < self.max_number_of_time_stamps, "{} < {}", self.end_count, self.max_number_of_time_stamps);
        }
        pass.write_timestamp(&self.timestamp, (self.end_count - 1) * 2 + 1);
    }

    /// Add start. For each start there must be end counter part.
    pub fn start(&mut self, encoder: &mut wgpu::CommandEncoder) {
        #[cfg(debug_assertions)]
        {
            debug_assert!(self.start_count < self.max_number_of_time_stamps, "{} < {}", self.start_count, self.max_number_of_time_stamps);
            debug_assert!(self.start_count - self.end_count < 2, "{} - {} < 2", self.start_count, self.end_count);
        }
        encoder.write_timestamp(&self.timestamp, self.start_count * 2);
        self.start_count = self.start_count + 1;
    }

    /// Add end. For each end there must be end counter part.
    pub fn end(&mut self, encoder: &mut wgpu::CommandEncoder) {

        self.end_count = self.end_count + 1;
        #[cfg(debug_assertions)]
        {
            debug_assert!(self.start_count == self.end_count , "{} == {}", self.start_count, self.end_count);
            debug_assert!(self.end_count < self.max_number_of_time_stamps, "{} < {}", self.end_count, self.max_number_of_time_stamps);
        }
        encoder.write_timestamp(&self.timestamp, (self.end_count - 1) * 2 + 1);
    }

    /// Resolve the content of start and end counter parts.
    pub fn resolve_timestamps(&self, encoder: &mut wgpu::CommandEncoder) {

        #[cfg(debug_assertions)]
        {
            debug_assert!(self.start_count == self.end_count, "{} == {}", self.start_count, self.end_count);
            debug_assert!(self.start_count != 0, "{} != 0", self.start_count);
            debug_assert!(self.end_count != 0, "{} != 0", self.end_count);
        }

            encoder.resolve_query_set(
                &self.timestamp,
                0..self.start_count * 2,
                &self.query_buffer,
                0,
                );
    }

    /// Store the content of start and end counter parts.
    pub fn create_timestamp_data(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.data = 
            to_vec::<TimestampData>(&device,
                                &queue,
                                &self.query_buffer,
                                0 as wgpu::BufferAddress,
                                (size_of::<TimestampData>() * self.max_number_of_time_stamps as usize) as wgpu::BufferAddress
        );
    }

    /// Get the stored time stamp data.
    pub fn get_data(&self) -> Vec<TimestampData> {
        self.data.clone()
    }

    /// Print the timestamp data.
    pub fn print_data(&self) {

        for (i, elem) in self.data.iter().enumerate() {
            let nanoseconds =
                (elem.end - elem.start) as f32 * self.timestamp_period;
            let microseconds = nanoseconds / 1000.0;
            let milli = microseconds / 1000.0;
            println!("{:?} time is {:?} milli seconds.", i, milli);
            // println!("{:?} time is {:?} micro seconds.", i, microseconds);
        }
    }

    /// Reset the counters and data.
    pub fn reset(&mut self) {
        self.start_count = 0;
        self.end_count = 0;
        self.data = vec![TimestampData {start: 0, end: 0, }; self.max_number_of_time_stamps.try_into().unwrap()];
    }
}
