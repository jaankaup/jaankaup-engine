use std::ops::Add;
use std::mem::size_of;
// use std::collections::HashMap;
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
    // query_hash_table: HashMap<u32, (bool, bool)>,
    start_count: u32,
    end_count: u32,
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

                //println!("mem::size_of::<QueryData>() == {}", mem::size_of::<QueryData>());
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
                    // query_hash_table: query_hash_table,
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

    //pub fn start(&mut self, encoder: &mut wgpu::CommandEncoder, time_stamp_number: u32) {
    pub fn start(&mut self, encoder: &mut wgpu::CommandEncoder) {
        #[cfg(debug_assertions)]
        {
            debug_assert!(self.start_count < self.max_number_of_time_stamps, "{} < {}", self.start_count, self.max_number_of_time_stamps);
            debug_assert!(self.start_count - self.end_count < 2, "{} - {} < 2", self.start_count, self.end_count);
        }
        encoder.write_timestamp(&self.timestamp, self.start_count * 2);
        self.start_count = self.start_count + 1;
    }

    // pub fn end(&mut self, encoder: &mut wgpu::CommandEncoder, time_stamp_number: u32) {
    pub fn end(&mut self, encoder: &mut wgpu::CommandEncoder) {

        self.end_count = self.end_count + 1;
        #[cfg(debug_assertions)]
        {
            debug_assert!(self.start_count == self.end_count , "{} == {}", self.start_count, self.end_count);
            debug_assert!(self.end_count < self.max_number_of_time_stamps, "{} < {}", self.end_count, self.max_number_of_time_stamps);
        }
        encoder.write_timestamp(&self.timestamp, (self.end_count - 1) * 2 + 1);
    }

    pub fn resolve_timestamps(&self, encoder: &mut wgpu::CommandEncoder) {

        // build ranges for resolvin query set.
        // let mut ok_stamps: Vec<std::ops::Range<u32>> = Vec::with_capacity(self.max_number_of_time_stamps.try_into().unwrap()); 
        // let mut ok: Vec<u32> = Vec::with_capacity(self.max_number_of_time_stamps.try_into().unwrap()); 

        // let mut start = 0;
        // let mut end   = 0;

        // for i in 0..self.max_number_of_time_stamps {
        //     let (has_start, has_end) = self.query_hash_table.get(&i).unwrap();

        //     // The start and end are ok.
        //     if *has_start && *has_end {
        //         end = end + 1;
        //     }
        //     else {
        //         // Skip.
        //         if start == end {
        //             start = start + 1;
        //             end = end + 1;
        //         }
        //         // We found a new range.
        //         else {
        //            ok_stamps.push(std::ops::Range::<u32> { start: start, end: end, }); 
        //            end = end + 1;
        //            start = end;
        //         }
        //     }
        // }
        // // Finally, if start != end, we still have one range. 
        // if start != end {
        //     ok_stamps.push(std::ops::Range::<u32> { start: start, end: end + 1, }); 
        // }

        // println!("{:?}", ok_stamps);

        // for (s, e) in ok_stamsp.iter() {
        //     
        //     encoder.resolve_query_set(
        //         &self.timestamp,
        //         s..e,
        //         &self.query_buffer,
        //         s,
        //     );
        #[cfg(debug_assertions)]
        {
            debug_assert!(self.start_count == self.end_count, "{} == {}", self.start_count, self.end_count);
            debug_assert!(self.start_count != 0, "{} != 0", self.start_count);
            debug_assert!(self.end_count != 0, "{} != 0", self.end_count);
        }

        println!("start_count == {}", self.start_count);

            encoder.resolve_query_set(
                &self.timestamp,
                0..self.start_count * 2,
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

    pub fn reset(&mut self) {
        self.start_count = 0;
        self.end_count = 0;
    }
}


// https://www.rosettacode.org/wiki/Range_extraction#Rust
//struct RangeFinder<'a, T: 'a> {
//    index: usize,
//    length: usize,
//    arr: &'a [T],
//}
//
//impl<'a, T> Iterator for RangeFinder<'a, T> where T: PartialEq + Add<i8, Output=T> + Copy {
//    type Item = (T,  Option<T>);
//    fn next(&mut self) -> Option<Self::Item> {
//        if self.index == self.length {
//            return None;
//        }
//        let lo = self.index;
//        while self.index < self.length - 1 && self.arr[self.index + 1] == self.arr[self.index] + 1 {
//            self.index += 1
//        }
//        let hi = self.index;
//        self.index += 1;
//        if hi - lo > 1 {
//            Some((self.arr[lo], Some(self.arr[hi])))
//        } else {
//            if hi - lo == 1 {
//                self.index -= 1
//            }
//            Some((self.arr[lo], None))
//        }
//    }
//}
//
//impl<'a, T> RangeFinder<'a, T> {
//    fn new(a: &'a [T]) -> Self {
//        RangeFinder {
//            index: 0,
//            arr: a,
//            length: a.len(),
//        }
//    }
//}
