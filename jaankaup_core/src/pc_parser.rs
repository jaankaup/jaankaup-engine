use crate::common_functions::encode_rgba_u32;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use bytemuck::{Pod, Zeroable};
use crate::aabb::BBox64;
use cgmath::Vector3;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VVVC {
    pub position: [f32; 3],
    pub color: u32,
}

// #[repr(C)]
// #[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VVVC64 {
    position: [f64; 3],
    color: u32,
}

/// Read space separated pppccc data from file. Returns (aabb_min, aabb_max, VVVC data)
pub fn read_pc_data(file: &String, scene_x: f32, scene_y: f32, scene_z: f32) -> ([f32; 3], [f32; 3], Vec<VVVC>) {

    let mut result: Vec<VVVC64> = Vec::with_capacity(7000000);
    let mut result2: Vec<VVVC> = Vec::with_capacity(7000000);

    let mut aabb = BBox64 { min: Vector3::<f64>::new(0.0, 0.0, 0.0), max: Vector3::<f64>::new(0.0, 0.0, 0.0), };

    if let Ok(lines) = read_lines(file) {
        let mut first_time = true;
        for line in lines {
            if let Ok(li) = line {
                let sp = li.split(" ").collect::<Vec<&str>>();

                let x = sp[0].parse::<f64>().unwrap();
                let y = sp[2].parse::<f64>().unwrap();
                let z = sp[1].parse::<f64>().unwrap();

                let v = Vector3::<f64>::new(x, y, z);

                if first_time {
                    aabb = BBox64 { min: v, max: v, };
                    first_time = false;
                }

                aabb.expand(&v);

                result.push(VVVC64 {
                    position: [x, y, z],
                    color: encode_rgba_u32(sp[3].parse::<u32>().unwrap(), sp[4].parse::<u32>().unwrap(), sp[5].parse::<u32>().unwrap(), 255),
                });
            }
        }
    }

    for i in 0..result.len() {
        result2.push(VVVC {
            position: [(result[i].position[0] - aabb.min.x) as f32,
                       (result[i].position[1] - aabb.min.y) as f32,
                       (result[i].position[2] - aabb.min.z) as f32],
            color: result[i].color,
        });
    }
    // println!("{:?}", aabb);
    aabb.max = aabb.max - aabb.min;  
    aabb.min.x = 0.0;
    aabb.min.y = 0.0;
    aabb.min.z = 0.0;

    let point_cloud_scale_factor_x = scene_x / aabb.max[0] as f32;
    let point_cloud_scale_factor_y = scene_y / aabb.max[1] as f32;
    let point_cloud_scale_factor_z = scene_z / aabb.max[2] as f32;

    for i in 0..result2.len() {
        result2[i].position[0] = result2[i].position[0] * point_cloud_scale_factor_x;
        result2[i].position[1] = result2[i].position[1] * point_cloud_scale_factor_y;
        result2[i].position[2] = result2[i].position[2] * point_cloud_scale_factor_z;
    }

    ([aabb.min.x as f32, aabb.min.y as f32, aabb.min.z as f32], [aabb.max.x as f32, aabb.max.y as f32, aabb.max.z as f32], result2)
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where P: AsRef<Path>, {
        let file = File::open(filename)?;
        Ok(io::BufReader::new(file).lines())
}
