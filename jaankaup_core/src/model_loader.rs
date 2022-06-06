use std::io::Read;
use std::fs::File;
use wavefront_obj::obj::*;
use cgmath::{Vector3, Vector4};
use crate::aabb::{BBox, Triangle, Triangle_vvvvnnnn};
use crate::buffer::buffer_from_data;

pub struct TriangleMesh {
    name: String,
    buffer: wgpu::Buffer,
    triangle_count: u32,
}

impl TriangleMesh {

    pub fn create_from_data(device: &wgpu::Device,
                         data: &Vec<Triangle_vvvvnnnn>,
                         name: &'static str,
                         triangle_count: u32) -> Self {
        
        let buf = buffer_from_data::<Triangle_vvvvnnnn>(
                  &device,
                  data,
                  wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                  Some(&format!("Triangle mesh {:?}", &name[..]).to_string())
        );

        Self {
            name: name.to_string(),
            buffer: buf,
            triangle_count: triangle_count,
        }
    }

    pub fn get_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn get_triangle_count(&self) -> u32 {
        self.triangle_count
    }

    pub fn get_name(&self) -> String {
        self.name.clone()
    }
}

pub fn create_from_bytes(data: String,
                         scale_factor: f32,
                         transition: [f32;3],
                         take: Option<u32>) -> Option<(Vec<Triangle>, Vec<Triangle_vvvvnnnn>, BBox)> {

    let obj_set = parse(data).map_err(|e| format!("cannot parse: {:?}", e)).unwrap();
    let objects = obj_set.objects;

    let mut aabb = BBox { min: Vector3::<f32>::new(0.0, 0.0, 0.0), max: Vector3::<f32>::new(0.0, 0.0, 0.0), };
    let mut result: Vec<Triangle> = Vec::new();
    let mut result_vvvvnnnn: Vec<Triangle_vvvvnnnn> = Vec::with_capacity(50000); // Vec::new(); // with capacity from file?

    if objects.len() == 1 {
        for shape in &objects[0].geometry[0].shapes {
            match shape.primitive {
                Primitive::Triangle((ia, _, Some(na)), (ib, _, Some(nb)), (ic, _, Some(nc))) => {

                    let vertex_a = objects[0].vertices[ia];
                    let vertex_b = objects[0].vertices[ib];
                    let vertex_c = objects[0].vertices[ic];

                    let normal_a = objects[0].normals[na];
                    let normal_b = objects[0].normals[nb];
                    let normal_c = objects[0].normals[nc];

                    let mut vec_a = scale_factor * Vector4::<f32>::new(vertex_a.x as f32, vertex_a.y as f32, vertex_a.z as f32, 0.0); 
                    let mut vec_b = scale_factor * Vector4::<f32>::new(vertex_b.x as f32, vertex_b.y as f32, vertex_b.z as f32, 0.0); 
                    let mut vec_c = scale_factor * Vector4::<f32>::new(vertex_c.x as f32, vertex_c.y as f32, vertex_c.z as f32, 0.0); 
                    vec_a.w = 1.0;
                    vec_b.w = 1.0;
                    vec_c.w = 1.0;

                    vec_a = vec_a + Vector4::<f32>::new(transition[0], transition[1],transition[2], 0.0);
                    vec_b = vec_b + Vector4::<f32>::new(transition[0], transition[1],transition[2], 0.0);
                    vec_c = vec_c + Vector4::<f32>::new(transition[0], transition[1],transition[2], 0.0);

                    let vec_na = Vector4::<f32>::new(normal_a.x as f32, normal_a.y as f32, normal_a.z as f32, 0.0); 
                    let vec_nb = Vector4::<f32>::new(normal_b.x as f32, normal_b.y as f32, normal_b.z as f32, 0.0); 
                    let vec_nc = Vector4::<f32>::new(normal_c.x as f32, normal_c.y as f32, normal_c.z as f32, 0.0); 

                    aabb.expand(&Vector3::<f32>::new(vec_a.x, vec_a.y, vec_a.z));
                    aabb.expand(&Vector3::<f32>::new(vec_b.x, vec_b.y, vec_b.z));
                    aabb.expand(&Vector3::<f32>::new(vec_c.x, vec_c.y, vec_c.z));

                    let tr = Triangle_vvvvnnnn {
                        a: vec_a,
                        b: vec_b,
                        c: vec_c,
                        na: vec_na,
                        nb: vec_nb,
                        nc: vec_nc,
                    };

                    //println!("{:?}", tr);

                    result_vvvvnnnn.push(tr);
                }
                Primitive::Triangle((ia, _, _), (ib, _, _), (ic, _, _)) => {

                    let vertex_a = objects[0].vertices[ia];
                    let vertex_b = objects[0].vertices[ib];
                    let vertex_c = objects[0].vertices[ic];

                    let vec_a = Vector3::<f32>::new(vertex_a.x as f32, vertex_a.y as f32, vertex_a.z as f32) * 2.0; 
                    let vec_b = Vector3::<f32>::new(vertex_b.x as f32, vertex_b.y as f32, vertex_b.z as f32) * 2.0; 
                    let vec_c = Vector3::<f32>::new(vertex_c.x as f32, vertex_c.y as f32, vertex_c.z as f32) * 2.0; 

                    aabb.expand(&vec_a);
                    aabb.expand(&vec_b);
                    aabb.expand(&vec_c);

                    let tr = Triangle {
                        a: vec_a,
                        b: vec_b,
                        c: vec_c,
                    };

                    result.push(tr);
                }
                Primitive::Line(_, _) => { panic!("load_triangles_from_obj not supporting lines."); }
                Primitive::Point(_) => { panic!("load_triangles_from_obj not supporting points."); }
            }
        }
    }
    match take {
        // TODO: check bounds! Test!
        Some(amount) => Some((result, (&result_vvvvnnnn[0..std::cmp::min(amount as usize, result_vvvvnnnn.len())]).to_vec(), aabb)), 
        None => Some((result, result_vvvvnnnn, aabb)) 
    }
    // Some((result, result_vvvvnnnn, aabb))

}

pub fn load_triangles_from_obj(file_name: &'static str,
                               scale_factor: f32,
                               transition: [f32;3],
                               take: Option<u32>) -> Option<(Vec<Triangle>, Vec<Triangle_vvvvnnnn>, BBox)> {

    let file_content = {
            let mut file = File::open(file_name).map_err(|e| format!("cannot open file: {}", e)).unwrap();
            let mut content = String::new();
            file.read_to_string(&mut content).unwrap();
            content
    };

    create_from_bytes(file_content,
                      scale_factor,
                      transition,
                      take
    ) 
}
