use cgmath::{prelude::*, Vector3, Vector4};
use bytemuck::{Pod, Zeroable};

/**************************************************************************************/

/// A struct for aabb.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BBox4 {
    pub min: Vector4<f32>,
    pub max: Vector4<f32>,
}

unsafe impl bytemuck::Zeroable for BBox4 {}
unsafe impl bytemuck::Pod for BBox4 {}

/// A struct for aabb.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BBox {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

// TODO: convert Vector3 to [f32; 3] ?
unsafe impl bytemuck::Zeroable for BBox {}
unsafe impl bytemuck::Pod for BBox {}

impl BBox {

    pub fn convert_aabb_to_aabb4(&self) -> BBox4 {
        BBox4 {
            min: Vector4::<f32>::new(self.min.x, self.min.y, self.min.z, 2.5),
            max: Vector4::<f32>::new(self.max.x, self.max.y, self.max.z, 4.5),
        }
    }

    /// Create bounding box from two vectors.
    pub fn create_from_line(a: &Vector3<f32>, b: &Vector3<f32>) -> Self {

        let min = min_vec(a, b);
        let max = max_vec(a, b);

        Self {
            min: min,
            max: max,
        }
    }

    /// Create bounding box from three vectors.
    pub fn create_from_triangle(a: &Vector3<f32>, b: &Vector3<f32>, c: &Vector3<f32>) -> Self {
        let mut result = BBox::create_from_line(&a, &b);
        result.expand(&c);
        result
    }

    /// Expand BBox to include point p.
    pub fn expand(&mut self, p: &Vector3<f32>) {
        self.min = Vector3::<f32>::new(self.min.x.min(p.x), self.min.y.min(p.y), self.min.z.min(p.z));
        self.max = Vector3::<f32>::new(self.max.x.max(p.x), self.max.y.max(p.y), self.max.z.max(p.z));
    }

    pub fn combine(a: &BBox, b: &BBox) -> BBox {
        let mut result = a.clone();
        result.expand(&b.min);
        result.expand(&b.max);
        result
    }

    /// This should to be moved into FMM_Domain. // TODO
    pub fn expand_to_nearest_grids(&mut self, grid_resolution: f32) {
        println!("Original aabb: min ({}, {}, {}), max ({}, {}, {})", self.min.x, self.min.y, self.min.z, self.max.x, self.max.y, self.max.z);

        let expanded_min = Vector3::<f32>::new(
            (self.min.x / grid_resolution).floor() * grid_resolution,
            (self.min.y / grid_resolution).floor() * grid_resolution,
            (self.min.z / grid_resolution).floor() * grid_resolution,
        );
        let expanded_max = Vector3::<f32>::new(
            (self.max.x / grid_resolution).ceil() * grid_resolution,
            (self.max.y / grid_resolution).ceil() * grid_resolution,
            (self.max.z / grid_resolution).ceil() * grid_resolution,
        );
        self.min = expanded_min;
        self.max = expanded_max;
        // println!("Expanded aabb: min ({}, {}, {}), max ({}, {}, {})", self.min.x, self.min.y, self.min.z, self.max.x, self.max.y, self.max.z);
    }

    // Check if point p is inside this bounding box. Is on_boundary is true, also checks if point p is on boundary.
    pub fn includes_point(&self, p: &Vector3<f32>, on_boundary: bool) -> bool {
        if on_boundary && p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y && p.z >= self.min.z && p.z <= self.max.z {
            true
        }
        else if p.x > self.min.x && p.x < self.max.x && p.y > self.min.y && p.y < self.max.y && p.z > self.min.z && p.z < self.max.z {
            true
        }
        else {
            false 
        }
    }

    pub fn to_lines(&self) -> Vec<f32> {

        let dx = self.max.x - self.min.x;
        let dy = self.max.y - self.min.y;
        let dz = self.max.z - self.min.z;

        let p0 = self.min + Vector3::<f32>::new(0.0           , 0.0           , dz);
        let p1 = self.min + Vector3::<f32>::new(0.0           , dy            , dz);
        let p2 = self.min + Vector3::<f32>::new(dx            , dy            , dz);
        let p3 = self.min + Vector3::<f32>::new(dx            , 0.0           , dz);
        let p4 = self.min;
        let p5 = self.min + Vector3::<f32>::new(0.0           , dy            , 0.0);
        let p6 = self.min + Vector3::<f32>::new(dx            , dy            , 0.0);
        let p7 = self.min + Vector3::<f32>::new(dx            , 0.0           , 0.0);

        let mut result: Vec<f32> = Vec::new();

        result.push(p0.x); result.push(p0.y); result.push(p0.z); result.push(1.0); 
        result.push(p1.x); result.push(p1.y); result.push(p1.z); result.push(1.0); 

        result.push(p1.x); result.push(p1.y); result.push(p1.z); result.push(1.0); 
        result.push(p2.x); result.push(p2.y); result.push(p2.z); result.push(1.0); 

        result.push(p2.x); result.push(p2.y); result.push(p2.z); result.push(1.0); 
        result.push(p3.x); result.push(p3.y); result.push(p3.z); result.push(1.0); 

        result.push(p0.x); result.push(p0.y); result.push(p0.z); result.push(1.0); 
        result.push(p3.x); result.push(p3.y); result.push(p3.z); result.push(1.0); 

        result.push(p4.x); result.push(p4.y); result.push(p4.z); result.push(1.0); 
        result.push(p5.x); result.push(p5.y); result.push(p5.z); result.push(1.0); 

        result.push(p5.x); result.push(p5.y); result.push(p5.z); result.push(1.0); 
        result.push(p6.x); result.push(p6.y); result.push(p6.z); result.push(1.0); 

        result.push(p6.x); result.push(p6.y); result.push(p6.z); result.push(1.0); 
        result.push(p7.x); result.push(p7.y); result.push(p7.z); result.push(1.0); 

        result.push(p4.x); result.push(p4.y); result.push(p4.z); result.push(1.0); 
        result.push(p7.x); result.push(p7.y); result.push(p7.z); result.push(1.0); 

        result.push(p1.x); result.push(p1.y); result.push(p1.z); result.push(1.0); 
        result.push(p5.x); result.push(p5.y); result.push(p5.z); result.push(1.0); 

        result.push(p2.x); result.push(p2.y); result.push(p2.z); result.push(1.0); 
        result.push(p6.x); result.push(p6.y); result.push(p6.z); result.push(1.0); 

        result.push(p0.x); result.push(p0.y); result.push(p0.z); result.push(1.0); 
        result.push(p4.x); result.push(p4.y); result.push(p4.z); result.push(1.0); 

        result.push(p3.x); result.push(p3.y); result.push(p3.z); result.push(1.0); 
        result.push(p7.x); result.push(p7.y); result.push(p7.z); result.push(1.0); 

        result
    }
}

/**************************************************************************************/

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Triangle {
    pub a: Vector3<f32>,
    pub b: Vector3<f32>,
    pub c: Vector3<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Triangle_vvvvnnnn {
    pub a: Vector4<f32>,
    pub na: Vector4<f32>,
    pub b: Vector4<f32>,
    pub nb: Vector4<f32>,
    pub c: Vector4<f32>,
    pub nc: Vector4<f32>,
}

unsafe impl Pod for Triangle_vvvvnnnn {}
unsafe impl Zeroable for Triangle_vvvvnnnn {}

impl Triangle {

    pub fn closest_point_to_triangle(&self, p: &Vector3<f32>) -> Vector3<f32> {

        let a = self.a;
        let b = self.b;
        let c = self.c;


        let ab = b - a;
        let ac = c - a;
        let bc = c - b;

        // Surface normal ccw.
        let normal = (b-a).cross(c-a).normalize(); //ac.cross(ab).normalize();

        let snom = (p-a).dot(ab);
        let sdenom = (p-b).dot(a-b);

        let tnom = (p-a).dot(ac);
        let tdenom = (p-c).dot(a-c);

        if snom <= 0.0 && tnom <= 0.0 {
            let result  = a;
            // let minus = normal.dot(result.normalize());
            // if minus < 0.0 { return (result, false); }
            // else { return (result, true); }
            return result;
        }

        let unom = (p-b).dot(bc);
        let udenom = (p-c).dot(b-c);

        if sdenom <= 0.0 && unom <= 0.0 {
            let result = b;
            // let minus = normal.dot(result.normalize());
            // if minus < 0.0 { return (result, false); }
            // else { return (result, true); }
            return result;
        }
        if tdenom <= 0.0 && udenom <= 0.0 {
            let result = c;
            // let minus = normal.dot(result.normalize());
            // if minus < 0.0 { return (result, false); }
            // else { return (result, true); }
            return result;
        }

        let n = (b-a).cross(c-a);
        let vc = n.dot((a-p).cross(b-p));
        if vc <= 0.0 && snom >= 0.0 && sdenom >= 0.0 {
            let result = a + snom / (snom + sdenom) * ab;
            // let minus = normal.dot(result.normalize());
            // if minus < 0.0 { return (result, false); }
            // else { return (result, true); }
            return result;
        }

        let va = n.dot((b-p).cross(c-p));
        if va <= 0.0 && unom >= 0.0 && udenom >= 0.0 {
            let result = b + unom / (unom + udenom) * bc;
            // let minus = normal.dot(result.normalize());
            // if minus < 0.0 { return (result, false); }
            // else { return (result, true); }
            return result;
        }

        let vb = n.dot((c-p).cross(a-p));
        if vb <= 0.0 && tnom >= 0.0 && tdenom >= 0.0 {
            let result = a + tnom / (tnom + tdenom) * ac;
            // let minus = normal.dot(result.normalize());
            // if minus < 0.0 { return (result, false); }
            // else { return (result, true); }
            return result;
        }

        let u = va / (va + vb + vc);
        let v = vb / (va + vb + vc);
        let w = 1.0 - u - v;
        let result = u * a + v * b + w * c;
        // let minus = normal.dot(result.normalize());
        // if minus < 0.0 { return (result, false); }
        // else { return (result, true); }
        result
    }

    pub fn distance_to_triangle(&self, p: &Vector3<f32>) -> (f32, bool) { // (distance, is_positive)
        // TODO: solve sign of distance.
        // Surface normal ccw.
        //let normal = (self.b-self.a).cross(self.c-self.a).normalize(); //ac.cross(ab).normalize();
        let normal = (self.b-self.a).cross(self.c-self.a).normalize(); //ac.cross(ab).normalize();
        let point = self.closest_point_to_triangle(&p);
        let dot_p = normal.dot(point - p);
        let sign = { 
            if dot_p < 0.0 { true } // CHECK THESE 
            else { false }
        };
        //println!("SIGN == {}", sign);
        (point.distance(*p), sign)

        // let (result, sign) = self.closest_point_to_triangle(&p).distance(*p);
        // if sign == false { -result }
        // else { result }
    }

    pub fn divide_triangle_to_points(&self, max_n: u32) -> Vec<Vector3<f32>> {

        let epsilon: f32 = 0.3;

        let mut points: Vec<Vector3<f32>> = Vec::new();
        let ab = self.b - self.a; 
        let ac = self.c - self.a;
        let bc = self.c - self.b;
        // let ab = self.a - self.b; 
        // let ac = self.a - self.c;
        // let bc = self.b - self.c;

        let n = ab.cross(ac).normalize(); 
        let s: f32 = 0.5 as f32 * ab.cross(ac).magnitude(); 
        
        let mut n: u32 = (s/epsilon).sqrt().ceil() as u32;

        if n > max_n { n = max_n; } 

        let s1 = 1.0 / (n as f32) * ab;
        let s2 = 1.0 / (n as f32) * bc;
        let s3 = 1.0 / (n as f32) * ac;

        let mut ps = 1.0/3.0 * (self.a + ( self.a + s1) + (self.a + s3));
        points.push(ps.clone());
        let mut i = 2;

        while i <= n {
            ps = ps + s1;
            points.push(ps.clone());
            let mut p = ps.clone();
            let mut j = 2;

            while j <= i {
                p = p + s2;
                points.push(p.clone());
                j += 1;
            }
            i += 1;
        }

        println!("Points");
        for p in points.iter() {
            println!("Vector {{ x: {}, y: {}, z: {} }}", p.x, p.y, p.z);
        }

        points
    }

}

unsafe impl Pod for Triangle {}
unsafe impl Zeroable for Triangle {}

#[derive(Clone, Copy)]
pub struct Plane {
    n: Vector3<f32>,    
    d: f32,    
}

impl Plane {
    pub fn new(a: &Vector3<f32>, b: &Vector3<f32>, c: &Vector3<f32>) -> Self {
        let n = (b-a).cross(c-a).normalize();
        let d = n.dot(*a);
        Self {
            n: n,
            d: d,
        }
    }

    pub fn closest_point_to_plane(self, q: &Vector3<f32>) -> Vector3<f32> {
        let t = (self.n.dot(*q) - self.d) / self.n.dot(self.n);
        q - t * self.n
    }
}


/// Return min vector from a and b components.
fn min_vec(a: &Vector3<f32>, b: &Vector3<f32>) -> Vector3<f32> {
    let result = Vector3::<f32>::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z));
    result
}

/// Return max vector from a and b components.
fn max_vec(a: &Vector3<f32>, b: &Vector3<f32>) -> Vector3<f32> {
    let result = Vector3::<f32>::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z));
    result
}

pub fn barycentric_cooordinates(a: &Vector3<f32>, b: &Vector3<f32>, c: &Vector3<f32>, r: &Vector3<f32>) -> Vector3<f32> {
    let n = (b - a).cross(c - a);
    let rab = n.dot((a-r).cross(b-r));
    let rbc = n.dot((b-r).cross(c-r));
    let rca = n.dot((c-r).cross(a-r));
    let abc = rab + rbc + rca;
    let u = rbc/abc;
    let v = rca/abc;
    let w = rab/abc;
    let result = Vector3::<f32>::new(u,v,w);
    result
}

// /// Compute barycentric coordinates (u, v, w) for point p with respect to triangle (a, b, c)
// pub fn barycentric_cooordinates(a: &Vector3<f32>, b: &Vector3<f32>, c: &Vector3<f32>, p: &Vector3<f32>) -> Vector3<f32> {
//     let v0 = b - a;
//     let v1 = c - a;
//     let v2 = p - a;
//     let d00 = v0.dot(v0);
//     let d01 = v0.dot(v1);
//     let d11 = v1.dot(v1);
//     let d20 = v2.dot(v0);
//     let d21 = v2.dot(v1);
//     let denom = d00 * d11 - d01 * d01;
//     let v = (d11 * d20 - d01 * d21) / denom;
//     let w = (d00 * d21 - d01 * d20) / denom;
//     let u = 1.0 - v - w;
//     let result = Vector3::<f32>::new(u,v,w);
//     result
// }

// pub struct Ray {
//     origin: Vector3<f32>,
//     dir: Vector3<f32>,
// }
