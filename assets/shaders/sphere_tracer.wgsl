// Debugging.

struct AABB {
    min: vec4<f32>, 
    max: vec4<f32>, 
};

struct Char {
    start_pos: vec3<f32>,
    font_size: f32,
    value: vec4<f32>,
    vec_dim_count: u32,
    color: u32,
    decimal_count: u32,
    auxiliary_data: u32,
};

struct Arrow {
    start_pos: vec4<f32>,
    end_pos:   vec4<f32>,
    color: u32,
    size:  f32,
};

struct FmmCellPc {
    tag: atomic<u32>,
    value: f32,
    color: u32,
};

// struct FmmBlock {
//     index: u32,
//     number_of_band_points: atomic<u32>,
// };

struct FmmParams {
    global_dimension: vec3<u32>,
    future_usage: u32,
    local_dimension: vec3<u32>,
    future_usage2: u32,
};

// // Binding id from where to load data.
// struct PushConstants {
//     buffer_id: u32,    
// };

struct RayOutput {
    origin: vec3<f32>,
    visibility: f32,
    intersection_point: vec3<f32>,
    opacity: f32,
    normal: vec3<f32>,
    diffuse_color: u32,
};

struct Ray {
  origin: vec3<f32>,
  rMin: f32,
  direction: vec3<f32>,
  rMax: f32,
};

struct RayPayload {
  intersection_point: vec3<f32>,
  visibility: f32,
  normal: vec3<f32>,
  opacity: f32,
  color: u32,
};

struct RayCamera {
    pos: vec3<f32>,
    apertureRadius: f32,
    view: vec3<f32>,
    focalDistance: f32,
    up: vec3<f32>,
    fov: vec2<f32>,
};

struct SphereTracerParams {
    inner_dim: vec2<u32>,
    outer_dim: vec2<u32>,
};

// struct ModF {
//     fract: f32,
//     whole: f32,
// };

@group(0) @binding(0) var<uniform>            fmm_params: FmmParams;
@group(0) @binding(1) var<uniform>            sphere_tracer_params: SphereTracerParams;
@group(0) @binding(2) var<uniform>            camera: RayCamera;
@group(0) @binding(3) var<storage,read_write> fmm_data: array<FmmCellPc>;
// @group(0) @binding(4) var<storage,read_write> screen_output: array<RayOutput>;
@group(0) @binding(4) var<storage,read_write> screen_output_color: array<u32>;

@group(1) @binding(0) var<storage,read_write> counter: array<atomic<u32>>;
@group(1) @binding(1) var<storage,read_write> output_char: array<Char>;
@group(1) @binding(2) var<storage,read_write> output_arrow: array<Arrow>;
@group(1) @binding(3) var<storage,read_write> output_aabb: array<AABB>;
@group(1) @binding(4) var<storage,read_write> output_aabb_wire: array<AABB>;
// Push constants.
// var<push_constant> pc: PushConstants;

var<private> private_neighbors:     array<FmmCellPc, 8>;
var<private> private_neighbors_loc: array<u32, 8>;

fn total_cell_count() -> u32 {

    return fmm_params.global_dimension.x * 
           fmm_params.global_dimension.y * 
           fmm_params.global_dimension.z * 
           fmm_params.local_dimension.x * 
           fmm_params.local_dimension.y * 
           fmm_params.local_dimension.z; 
};

fn rgba_u32(r: u32, g: u32, b: u32, a: u32) -> u32 {
  return (r << 24u) | (g << 16u) | (b  << 8u) | a;
}

fn rgba_u32_tex(r: u32, g: u32, b: u32, a: u32) -> u32 {
  return (a << 24u) | (r << 16u) | (g  << 8u) | b;
}

fn rgba_u32_argb(c: u32) -> u32 {
  let b = (c & 0xffu);
  let g = (c & 0xff00u) >> 8u;
  let r = (c & 0xff0000u) >> 16u;
  let a = (c & 0xff000000u) >> 24u;
  return (r << 24u) | (g << 16u) | (b  << 8u) | a;
}

// fn my_modf(f: f32) -> ModF {
//     let iptr = trunc(f);
//     let fptr = f - iptr;
//     return ModF (
//         select(fptr, (-1.0)*fptr, f < 0.0),
//         iptr
//     );
// }

/// A function that checks if a given coordinate is within the global computational domain. 
fn isInside(coord: vec3<i32>) -> bool {
    return (coord.x >= 0 && coord.x < i32(fmm_params.local_dimension.x * fmm_params.global_dimension.x)) &&
           (coord.y >= 0 && coord.y < i32(fmm_params.local_dimension.y * fmm_params.global_dimension.y)) &&
           (coord.z >= 0 && coord.z < i32(fmm_params.local_dimension.z * fmm_params.global_dimension.z)); 
}

/// xy-plane indexing. (x,y,z) => index
fn index_to_uvec3(index: u32, dim_x: u32, dim_y: u32) -> vec3<u32> {
  var x  = index;
  let wh = dim_x * dim_y;
  let z  = x / wh;
  x  = x - z * wh; // check
  let y  = x / dim_x;
  x  = x - y * dim_x;
  return vec3<u32>(x, y, z);
}

fn encode3Dmorton32(x: u32, y: u32, z: u32) -> u32 {
    var x_temp = (x      | (x      << 16u)) & 0x030000FFu;
        x_temp = (x_temp | (x_temp <<  8u)) & 0x0300F00Fu;
        x_temp = (x_temp | (x_temp <<  4u)) & 0x030C30C3u;
        x_temp = (x_temp | (x_temp <<  2u)) & 0x09249249u;

    var y_temp = (y      | (y      << 16u)) & 0x030000FFu;
        y_temp = (y_temp | (y_temp <<  8u)) & 0x0300F00Fu;
        y_temp = (y_temp | (y_temp <<  4u)) & 0x030C30C3u;
        y_temp = (y_temp | (y_temp <<  2u)) & 0x09249249u;

    var z_temp = (z      | (z      << 16u)) & 0x030000FFu;
        z_temp = (z_temp | (z_temp <<  8u)) & 0x0300F00Fu;
        z_temp = (z_temp | (z_temp <<  4u)) & 0x030C30C3u;
        z_temp = (z_temp | (z_temp <<  2u)) & 0x09249249u;

    return x_temp | (y_temp << 1u) | (z_temp << 2u);
}

fn get_third_bits32(m: u32) -> u32 {
    var x = m & 0x9249249u;
    x = (x ^ (x >> 2u))  & 0x30c30c3u;
    x = (x ^ (x >> 4u))  & 0x0300f00fu;
    x = (x ^ (x >> 8u))  & 0x30000ffu;
    x = (x ^ (x >> 16u)) & 0x000003ffu;

    return x;
}

fn decode3Dmorton32(m: u32) -> vec3<u32> {
    return vec3<u32>(
        get_third_bits32(m),
        get_third_bits32(m >> 1u),
        get_third_bits32(m >> 2u)
   );
}

fn decode_color(c: u32) -> vec4<f32> {
  let a: f32 = f32(c & 0xffu) / 255.0;
  let b: f32 = f32((c & 0xff00u) >> 8u) / 255.0;
  let g: f32 = f32((c & 0xff0000u) >> 16u) / 255.0;
  let r: f32 = f32((c & 0xff000000u) >> 24u) / 255.0;
  return vec4<f32>(r,g,b,a);
}

fn vec4_to_rgba(v: vec4<f32>) -> u32 {
    let r = u32(v.x * 255.0);
    let g = u32(v.y * 255.0);
    let b = u32(v.z * 255.0);
    let a = u32(v.w * 255.0);
    return rgba_u32(r, g, b, a);
}

/// Get memory index from given cell coordinate.
fn get_cell_mem_location(v: vec3<u32>) -> u32 {

    let stride = fmm_params.local_dimension.x * fmm_params.local_dimension.y * fmm_params.local_dimension.z;

    let global_coordinate = vec3<u32>(v) / fmm_params.local_dimension;

    let global_index = (global_coordinate.x +
                        global_coordinate.y * fmm_params.global_dimension.x +
                        global_coordinate.z * fmm_params.global_dimension.x * fmm_params.global_dimension.y) * stride;

    let local_coordinate = vec3<u32>(v) - global_coordinate * fmm_params.local_dimension;

    let local_index = encode3Dmorton32(local_coordinate.x, local_coordinate.y, local_coordinate.z);

    return global_index + local_index;
}

// fn load_neighbors_6(coord: vec3<u32>) -> array<u32, 6> {
// 
//     var neighbors: array<vec3<i32>, 6> = array<vec3<i32>, 6>(
//         vec3<i32>(coord) + vec3<i32>(-1,  0,  0),
//         vec3<i32>(coord) + vec3<i32>(1,   0,  0),
//         vec3<i32>(coord) + vec3<i32>(0,   1,  0),
//         vec3<i32>(coord) + vec3<i32>(0,  -1,  0),
//         vec3<i32>(coord) + vec3<i32>(0,   0,  1),
//         vec3<i32>(coord) + vec3<i32>(0,   0, -1)
//     );
// 
//     let i0 = get_cell_mem_location(vec3<u32>(neighbors[0]));
//     let i1 = get_cell_mem_location(vec3<u32>(neighbors[1]));
//     let i2 = get_cell_mem_location(vec3<u32>(neighbors[2]));
//     let i3 = get_cell_mem_location(vec3<u32>(neighbors[3]));
//     let i4 = get_cell_mem_location(vec3<u32>(neighbors[4]));
//     let i5 = get_cell_mem_location(vec3<u32>(neighbors[5]));
// 
//     // The index of the "outside" cell.
//     let tcc = total_cell_count();
// 
//     return array<u32, 6>(
//         select(tcc ,i0, isInside(neighbors[0])),
//         select(tcc ,i1, isInside(neighbors[1])),
//         select(tcc ,i2, isInside(neighbors[2])),
//         select(tcc ,i3, isInside(neighbors[3])),
//         select(tcc ,i4, isInside(neighbors[4])),
//         select(tcc ,i5, isInside(neighbors[5])) 
//     );
// }

/// Get cell index based on domain dimension.
fn get_cell_index(global_index: u32) -> vec3<u32> {

    let stride = fmm_params.local_dimension.x * fmm_params.local_dimension.y * fmm_params.local_dimension.z;
    let block_index = global_index / stride;
    let block_position = index_to_uvec3(block_index, fmm_params.global_dimension.x, fmm_params.global_dimension.y) * fmm_params.local_dimension;

    // Calculate local position.
    let local_index = global_index - block_index * stride;

    let local_position = decode3Dmorton32(local_index);

    let cell_position = block_position + local_position;

    return cell_position; 
}

fn index_to_2d(index: u32, width: u32) -> vec2<u32> {
    var x = index;
    let y = x / width;
    x = x - y * width;
    return vec2<u32>(x, y);
}

fn d2_to_index(x: u32, y: u32, width: u32) -> u32 {
    return x + y * width;
}

fn inner_coord_d2(index: u32, dim_x: u32, dim_y: u32) -> vec2<u32> {
    var x = index & (dim_x - 1u);
    let y = (index / dim_x) & (dim_y - 1u);
    return vec2<u32>(x, y);
}

fn index_to_screen(index: u32, dim_x: u32, dim_y: u32, global_dim_x:u32) -> vec2<u32> {

        // Outer coords.
        let block_index = index / (dim_x*dim_y);
        let outer_coord = index_to_2d(block_index, global_dim_x);

        // inner coords.
        let x = index & (dim_x - 1u);
        let y = (index / dim_x) & (dim_y - 1u);

        // Global coordinate.
	return vec2<u32>(outer_coord[0] * dim_x + x, outer_coord[1] * dim_y + y);

        // let gx = outer_coord[0] * dim_x + x;
        // let gy = outer_coord[1] * dim_y + y;

        // [gx, gy]
}

// fn screen_to_index(v: vec2<u32>) -> u32 {
// 
//     let stride = sphere_tracer_params.inner_dim.x * sphere_tracer_params.inner_dim.y;
//     let global_coordinate = v / sphere_tracer_params.inner_dim;
//     let global_index = (global_coordinate.x + global_coordinate.y * sphere_tracer_params.outer_dim.x) * stride;
//     let local_coordinate = v - global_coordinate * sphere_tracer_params.inner_dim;
//     let local_index = local_coordinate.x + local_coordinate.y * sphere_tracer_params.inner_dim.x;
//     return global_index + local_index;
// }

fn screen_to_index(v: vec2<u32>) -> u32 {
    let stride = sphere_tracer_params.inner_dim.x *
                 sphere_tracer_params.outer_dim.x;
    let index = v.x + v.y * stride;
    return index;
}

// // Box (exact).
// fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
//   let q = abs(p) - b;
//   return length(max(q,vec3<f32>(0.0))) + min(max(q.x,max(q.y,q.z)),0.0);
//   //return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
// }
// 
// 
// fn sdSphere(p: vec3<f32>, s: f32) -> f32
// {
//   return length(p)-s;
// }

fn diffuse(ray: ptr<function, Ray>, payload: ptr<function, RayPayload>) {

    let light_pos = vec3<f32>(150.0,70.0,150.0);
    let lightColor = vec3<f32>(1.0,1.0,1.0);
    let lightPower = 1550.0;
    
    // Material properties
    let materialDiffuseColor = decode_color((*payload).color).xyz; //vec3<f32>(1.0, 0.0, 0.0);
    let materialAmbientColor = vec3<f32>(0.8,0.8,0.8) * materialDiffuseColor;
    let materialSpecularColor = vec3<f32>(0.5,0.5,0.5);
    
    // Distance to the light
    let distance = length(light_pos - (*payload).intersection_point);
    
    // Normal of the computed fragment, in camera space
    let n = (*payload).normal; //normalize( Normal_cameraspace );
    
    // Direction of the light (from the fragment to the light)
    let l = normalize(((*payload).intersection_point + light_pos));
    let cosTheta = clamp( dot( n,l ), 0.0,1.0 );
    
    // Eye vector (towards the camera)
    let e = normalize((*ray).origin - ((*payload).intersection_point));
    let r = reflect(-l,n);
    let cosAlpha = clamp( dot( e,r ), 0.0,1.0);

    let final_color =
    	// Ambient : simulates indirect lighting
    	materialAmbientColor +
    	// Diffuse : "color" of the object
    	materialDiffuseColor * lightColor * lightPower * cosTheta / (distance*distance) +
    	// Specular : reflective highlight, like a mirror
    	materialSpecularColor * lightColor * lightPower * pow(cosAlpha,7.0) / (distance*distance);
    
    (*payload).color = 
       rgba_u32_tex(
           min(u32(final_color.x * 255.0), 255u),
           min(u32(final_color.y * 255.0), 255u),
           min(u32(final_color.z * 255.0), 255u),
           // u32(final_color.x * 255.0),
           // u32(final_color.y * 255.0),
           // u32(final_color.z * 255.0),
           255u
       );
}

// Calculate the point in the Ray direction.
fn getPoint(parameter: f32, ray: ptr<function, Ray>) -> vec3<f32> { return (*ray).origin + parameter * (*ray).direction; }

fn load_trilinear_neighbors(coord: vec3<u32>) -> array<u32, 8> {

    var neighbors: array<vec3<i32>, 8> = array<vec3<i32>, 8>(
        vec3<i32>(coord) + vec3<i32>(0,  0,  0),
        vec3<i32>(coord) + vec3<i32>(1,  0,  0),
        vec3<i32>(coord) + vec3<i32>(0,  1,  0),
        vec3<i32>(coord) + vec3<i32>(1,  1,  0),
        vec3<i32>(coord) + vec3<i32>(0,  0,  1),
        vec3<i32>(coord) + vec3<i32>(1,  0,  1),
        vec3<i32>(coord) + vec3<i32>(0,  1,  1),
        vec3<i32>(coord) + vec3<i32>(1,  1,  1),
    );

    let i0 = get_cell_mem_location(vec3<u32>(neighbors[0]));
    let i1 = get_cell_mem_location(vec3<u32>(neighbors[1]));
    let i2 = get_cell_mem_location(vec3<u32>(neighbors[2]));
    let i3 = get_cell_mem_location(vec3<u32>(neighbors[3]));
    let i4 = get_cell_mem_location(vec3<u32>(neighbors[4]));
    let i5 = get_cell_mem_location(vec3<u32>(neighbors[5]));
    let i6 = get_cell_mem_location(vec3<u32>(neighbors[6]));
    let i7 = get_cell_mem_location(vec3<u32>(neighbors[7]));

    // The index of the "outside" cell.
    let tcc = total_cell_count();

    return array<u32, 8>(
        select(tcc ,i0, isInside(neighbors[0])),
        select(tcc ,i1, isInside(neighbors[1])),
        select(tcc ,i2, isInside(neighbors[2])),
        select(tcc ,i3, isInside(neighbors[3])),
        select(tcc ,i4, isInside(neighbors[4])),
        select(tcc ,i5, isInside(neighbors[5])),
        select(tcc ,i6, isInside(neighbors[5])),
        select(tcc ,i7, isInside(neighbors[5]))
    );
}

fn fmm_color(p: vec3<f32>) -> u32 {

   //let cell_value = fmm_data[get_cell_mem_location(vec3<u32>(p))];
   //++ let temp = decode_color(cell_value.color); 
   //++ return vec4_to_rgba(temp);

   // var memory_locations = load_trilinear_neighbors(vec3<u32>(p));

   var c000 = decode_color(private_neighbors[0].color);
   var c100 = decode_color(private_neighbors[1].color);
   var c010 = decode_color(private_neighbors[2].color);
   var c110 = decode_color(private_neighbors[3].color);
   var c001 = decode_color(private_neighbors[4].color);
   var c101 = decode_color(private_neighbors[5].color);
   var c011 = decode_color(private_neighbors[6].color);
   var c111 = decode_color(private_neighbors[7].color);


   var c000_factor = select(1.0, 0.0, c000.x == 0.0 && c000.y == 0.0 && c000.z == 0.0);
   var c100_factor = select(1.0, 0.0, c100.x == 0.0 && c100.y == 0.0 && c100.z == 0.0);
   var c010_factor = select(1.0, 0.0, c010.x == 0.0 && c010.y == 0.0 && c010.z == 0.0);
   var c110_factor = select(1.0, 0.0, c110.x == 0.0 && c110.y == 0.0 && c110.z == 0.0);
   var c001_factor = select(1.0, 0.0, c001.x == 0.0 && c001.y == 0.0 && c001.z == 0.0);
   var c101_factor = select(1.0, 0.0, c101.x == 0.0 && c101.y == 0.0 && c101.z == 0.0);
   var c011_factor = select(1.0, 0.0, c011.x == 0.0 && c011.y == 0.0 && c011.z == 0.0);
   var c111_factor = select(1.0, 0.0, c111.x == 0.0 && c111.y == 0.0 && c111.z == 0.0);

   let tx = fract(p.x);
   let ty = fract(p.y);
   let tz = fract(p.z);

   // let color = (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * c000 + 
   //        tx * (1.0 - ty) * (1.0 - tz) * c100 + 
   //        (1.0 - tx) * ty * (1.0 - tz) * c010 + 
   //        tx * ty * (1.0 - tz) * c110 + 
   //        (1.0 - tx) * (1.0 - ty) * tz * c001 + 
   //        tx * (1.0 - ty) * tz * c101 + 
   //        (1.0 - tx) * ty * tz * c011 + 
   //        tx * ty * tz * c111;
   var color = (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * c000 * c000_factor + 
          tx * (1.0 - ty) * (1.0 - tz) * c100 * c100_factor + 
          (1.0 - tx) * ty * (1.0 - tz) * c010 * c010_factor + 
          tx * ty * (1.0 - tz) * c110 * c110_factor + 
          (1.0 - tx) * (1.0 - ty) * tz * c001 * c001_factor + 
          tx * (1.0 - ty) * tz * c101 * c101_factor + 
          (1.0 - tx) * ty * tz * c011 * c011_factor + 
          tx * ty * tz * c111 * c111_factor;
   // var color = (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * decode_color(private_neighbors[0].color) * c000_factor + 
   //        tx * (1.0 - ty) * (1.0 - tz) * decode_color(private_neighbors[1].color) * c100_factor + 
   //        (1.0 - tx) * ty * (1.0 - tz) * decode_color(private_neighbors[2].color) * c010_factor + 
   //        tx * ty * (1.0 - tz) * decode_color(private_neighbors[3].color) * c110_factor + 
   //        (1.0 - tx) * (1.0 - ty) * tz * decode_color(private_neighbors[4].color) * c001_factor + 
   //        tx * (1.0 - ty) * tz * decode_color(private_neighbors[5].color) * c101_factor + 
   //        (1.0 - tx) * ty * tz * decode_color(private_neighbors[6].color) * c011_factor + 
   //        tx * ty * tz * decode_color(private_neighbors[7].color) * c111_factor;
   color.w = 1.0;
   return vec4_to_rgba(color);
}

fn load_neighbors_private(p: vec3<u32>) {

   private_neighbors_loc = load_trilinear_neighbors(p);

   private_neighbors[0] = fmm_data[private_neighbors_loc[0]];
   private_neighbors[1] = fmm_data[private_neighbors_loc[1]];
   private_neighbors[2] = fmm_data[private_neighbors_loc[2]];
   private_neighbors[3] = fmm_data[private_neighbors_loc[3]];
   private_neighbors[4] = fmm_data[private_neighbors_loc[4]];
   private_neighbors[5] = fmm_data[private_neighbors_loc[5]];
   private_neighbors[6] = fmm_data[private_neighbors_loc[6]];
   private_neighbors[7] = fmm_data[private_neighbors_loc[7]];
}

fn fmm_value(p: vec3<f32>) -> f32 {

   // let cell_value = fmm_data[get_cell_mem_location(vec3<u32>(p))];
   //let neighbors = load_neighbors_6(coord: vec3<u32>);

   // var memory_locations = load_trilinear_neighbors(vec3<u32>(p));

        // vec3<i32>(coord) + vec3<i32>(0,  0,  0),
        // vec3<i32>(coord) + vec3<i32>(1,  0,  0),
        // vec3<i32>(coord) + vec3<i32>(0,  1,  0),
        // vec3<i32>(coord) + vec3<i32>(1,  1,  0),
        // vec3<i32>(coord) + vec3<i32>(0,  0,  1),
        // vec3<i32>(coord) + vec3<i32>(1,  0,  1),
        // vec3<i32>(coord) + vec3<i32>(0,  1,  1),
        // vec3<i32>(coord) + vec3<i32>(1,  1,  1),
   // 
   //            n2                        n3
   //          +------------------------+
   //         /|                       /|
   //        / |                      / |
   //       /  |                     /  |
   //      /   |                    /   |
   //     /    |                   /    |
   // n6 +------------------------+ n7  |
   //    |     |                  |     |
   //    |     |                  |     |
   //    |     |                  |     |
   //    |  n0 +------------------|-----+ n1
   //    |    /                   |    /
   //    |   /       P            |   /
   //    |  /                     |  /
   //    | /                      | /
   //    |/                       |/
   //    +------------------------+
   //   n4                       n5
   // 

   // The point P should be inside the cube.
   // var c000 = fmm_data[memory_locations[0]].value;
   // var c100 = fmm_data[memory_locations[1]].value;
   // var c010 = fmm_data[memory_locations[2]].value;
   // var c110 = fmm_data[memory_locations[3]].value;
   // var c001 = fmm_data[memory_locations[4]].value;
   // var c101 = fmm_data[memory_locations[5]].value;
   // var c011 = fmm_data[memory_locations[6]].value;
   // var c111 = fmm_data[memory_locations[7]].value;

   // var c001 = fmm_data[memory_locations[0]].value;
   // var c101 = fmm_data[memory_locations[1]].value;
   // var c011 = fmm_data[memory_locations[2]].value;
   // var c111 = fmm_data[memory_locations[3]].value;
   // var c000 = fmm_data[memory_locations[4]].value;
   // var c100 = fmm_data[memory_locations[5]].value;
   // var c010 = fmm_data[memory_locations[6]].value;
   // var c110 = fmm_data[memory_locations[7]].value;

   load_neighbors_private(vec3<u32>(p));

   // fn my_modf(f: f32) -> ModF {
   let tx = fract(p.x);
   let ty = fract(p.y);
   let tz = fract(p.z);

   // let x_value = x_part * n0 + (1.0 - x_part) * n1;  
   // let y_value = y_part * n2 + (1.0 - y_part) * n1;  
   // let z_value = z_part * n0 + (1.0 - z_part) * n1;  

   // return (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * c000 + 
   //        tx * (1.0 - ty) * (1.0 - tz) * c100 + 
   //        (1.0 - tx) * ty * (1.0 - tz) * c010 + 
   //        tx * ty * (1.0 - tz) * c110 + 
   //        (1.0 - tx) * (1.0 - ty) * tz * c001 + 
   //        tx * (1.0 - ty) * tz * c101 + 
   //        (1.0 - tx) * ty * tz * c011 + 
   //        tx * ty * tz * c111; 

   return (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * private_neighbors[0].value + 
          tx * (1.0 - ty) * (1.0 - tz) * private_neighbors[1].value + 
          (1.0 - tx) * ty * (1.0 - tz) * private_neighbors[2].value + 
          tx * ty * (1.0 - tz) * private_neighbors[3].value + 
          (1.0 - tx) * (1.0 - ty) * tz *  private_neighbors[4].value + 
          tx * (1.0 - ty) * tz * private_neighbors[5].value + 
          (1.0 - tx) * ty * tz * private_neighbors[6].value + 
          tx * ty * tz * private_neighbors[7].value; 
}

fn hit(ray: ptr<function, Ray>, payload: ptr<function, RayPayload>) {

    (*payload).color = fmm_color((*payload).intersection_point); // rgba_u32(0u, 0u, 155u, 0u);

    var grad: vec3<f32>;

    var pos = (*payload).intersection_point;
    var offset = 0.01;
    // var right = sdBox(vec3(pos.x+offset, pos.y,pos.z), vec3<f32>(50.0, 50.0, 50.0));
    // var left = sdBox(vec3(pos.x-offset, pos.y,pos.z), vec3<f32>(50.0, 50.0, 50.0));
    // var up = sdBox(vec3(pos.x, pos.y+offset,pos.z), vec3<f32>(50.0, 50.0, 50.0));
    // var down = sdBox(vec3(pos.x, pos.y-offset,pos.z), vec3<f32>(50.0, 50.0, 50.0));
    // var z_minus = sdBox(vec3(pos.x, pos.y,pos.z-offset), vec3<f32>(50.0, 50.0, 50.0));
    // var z = sdBox(vec3(pos.x, pos.y,pos.z+offset), vec3<f32>(50.0, 50.0, 50.0));
    var right = fmm_value(vec3(pos.x+offset, pos.y,pos.z));
    var left = fmm_value(vec3(pos.x-offset, pos.y,pos.z));
    var up = fmm_value(vec3(pos.x, pos.y+offset,pos.z));
    var down = fmm_value(vec3(pos.x, pos.y-offset,pos.z));
    var z_minus = fmm_value(vec3(pos.x, pos.y,pos.z-offset));
    var z = fmm_value(vec3(pos.x, pos.y,pos.z+offset));
    // var right = sdSphere(vec3(pos.x+offset, pos.y,pos.z), 50.0);
    // var left = sdSphere(vec3(pos.x-offset, pos.y,pos.z), 50.0);
    // var up = sdSphere(vec3(pos.x, pos.y+offset,pos.z), 50.0);
    // var down = sdSphere(vec3(pos.x, pos.y-offset,pos.z), 50.0);
    // var z_minus = sdSphere(vec3(pos.x, pos.y,pos.z-offset), 50.0);
    // var z = sdSphere(vec3(pos.x, pos.y,pos.z+offset), 50.0);
    grad.x = right - left;
    grad.y = up - down;
    grad.z = z - z_minus;
    //grad.z = z_minus - z;
    var normal = normalize(grad);
    //(*payload).color = rgba_u32(255u, 0u, 0u, 255u);
    // (*payload).normal = normal;
    (*payload).visibility = 1.0;
    diffuse(ray, payload);
}

// fn hit(ray: ptr<function, Ray>, payload: ptr<function, RayPayload>) {
//    payload.intersection_point = ray. 
// }
// ray intersection aabb

fn fmm_is_outside_value(v: vec3<f32>) -> bool {
    if (v.x < 0.0 ||
        v.y < 0.0 ||
        v.z < 0.0 ||
        v.x >= f32(fmm_params.global_dimension.x * fmm_params.local_dimension.x) ||
        v.y >= f32(fmm_params.global_dimension.y * fmm_params.local_dimension.y) ||
        v.z >= f32(fmm_params.global_dimension.z * fmm_params.local_dimension.z)) {
            return true;
	}
     return false;
}

fn traceRay(ray: ptr<function, Ray>, payload: ptr<function, RayPayload>) {

  var dist = (*ray).rMin;
  var calc = 0u; 

  var p: vec3<f32>;
  var distance_to_interface: f32;

  while (dist < (*ray).rMax && calc < 800u) {
      p = getPoint(dist, ray);
      //distance_to_interface = sdSphere(p, 50.0);
      // distance_to_interface = sdBox(p, vec3<f32>(50.0, 50.0, 50.0));

      if (fmm_is_outside_value(p)) { return; }
      distance_to_interface = fmm_value(p) * 0.1;
      if (distance_to_interface < 0.03) {
	  (*payload).intersection_point = p;
          hit(ray, payload);
	  return;
      }
      calc = calc + 1u;
      dist = dist + distance_to_interface;
  } // while

  (*payload).intersection_point = p;
}

@compute
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    let screen_coord = vec2<f32>(index_to_screen(global_id.x,
                                       sphere_tracer_params.inner_dim.x,
				       sphere_tracer_params.inner_dim.y,
				       sphere_tracer_params.outer_dim.x));

    let screen_coord = index_to_screen(
                       global_id.x,
		       sphere_tracer_params.inner_dim.x,
		       sphere_tracer_params.inner_dim.y,
		       sphere_tracer_params.outer_dim.x
    );
    let screen_width = f32(sphere_tracer_params.inner_dim.x * sphere_tracer_params.outer_dim.x);
    let screen_height = f32(sphere_tracer_params.inner_dim.y * sphere_tracer_params.outer_dim.y);

    let right = normalize(cross(camera.view, camera.up.xyz));
    let y = normalize(cross(camera.view, right));

    let d = camera.focalDistance;

    let u = (d * tan(camera.fov.x*0.5)) * right;
    let v = (d * tan(camera.fov.y*0.5)) * y;

    let alpha = 2.0 * (f32(screen_coord.x) + 0.5) / screen_width - 1.0;
    let beta  = 1.0 - 2.0 * (f32(screen_coord.y) + 0.5) / screen_height;

    let point_on_plane = alpha * u + beta * v;

    var ray: Ray;
    ray.origin = point_on_plane + camera.pos.xyz * 0.25;
    ray.direction = normalize(point_on_plane + d*camera.view.xyz);
    ray.rMin = 0.0;
    ray.rMax = 200.0;

    var payload: RayPayload;
    payload.color = rgba_u32(0u, 0u, 0u, 0u);
    payload.visibility = 0.0;

    traceRay(&ray, &payload);

    var result: RayOutput;
    result.origin = ray.origin;
    result.visibility = 1.0;
    result.intersection_point = payload.intersection_point;
    result.opacity = 1.0;
    result.normal = payload.normal;
    result.diffuse_color = payload.color;

    //++ if (global_id.x == 0u) {
    //++     
    //++     let focal_point = camera.pos - camera.view * d; 

    //++     output_aabb[atomicAdd(&counter[2], 1u)] =  
    //++           AABB (
    //++               vec4<f32>(focal_point.x - 1.1,
    //++                         focal_point.y - 1.1,
    //++                         focal_point.z - 1.1,
    //++                         f32(rgba_u32(255u, 0u, 2550u, 255u))),
    //++               vec4<f32>(focal_point.x + 1.1,
    //++                         focal_point.y + 1.1,
    //++                         focal_point.z + 1.1,
    //++                         0.0),
    //++     );
    //++     output_arrow[atomicAdd(&counter[1], 1u)] =  
    //++           Arrow (
    //++               vec4<f32>(focal_point, 0.0),
    //++               vec4<f32>(focal_point + camera.view, 0.0),
    //++               //vec4<f32>(result.intersection_point, 0.0),
    //++               rgba_u32(255u, 0u, 2550u, 255u),
    //++               0.01
    //++     );
    //++     let renderable_element = Char (
    //++                     // element_position,
    //++                     vec3<f32>(focal_point.x - 1.11, focal_point.y + 1.11, focal_point.z + 1.11),
    //++                     0.2,
    //++                     vec4<f32>(0.25 * camera.pos.x, 0.25 * camera.pos.y, 0.25 * camera.pos.z, 0.0),
    //++                     //vec4<f32>(vec3<f32>(fmm_params.local_dimension), 6.0),
    //++                     4u,
    //++                     rgba_u32(255u, 255u, 255u, 255u),
    //++                     1u,
    //++                     0u
    //++     );
    //++     output_char[atomicAdd(&counter[0], 1u)] = renderable_element; 
    //++ }

    //++if (local_index == 0u) {
    //++//++ if (payload.visibility > 0.0) {
    //++     output_arrow[atomicAdd(&counter[1], 1u)] =  
    //++           Arrow (
    //++               vec4<f32>(ray.origin * 4.0, 0.0),
    //++               vec4<f32>(payload.intersection_point * 4.0, 0.0),
    //++               //vec4<f32>(result.intersection_point, 0.0),
    //++               rgba_u32_argb(payload.color),
    //++               0.1
    //++     );

    //++     // output_arrow[atomicAdd(&counter[1], 1u)] =  
    //++     //       Arrow (
    //++     //           vec4<f32>(payload.intersection_point, 0.0),
    //++     //           vec4<f32>(payload.intersection_point + payload.normal * 3.0, 0.0),
    //++     //           //vec4<f32>(result.intersection_point, 0.0),
    //++     //           rgba_u32(155u, 0u, 1550u, 255u),
    //++     //           0.1
    //++     // );
    //++ }

    //if (workgroup_id.x == 0u) {
        let buffer_index = screen_to_index(screen_coord); 
        // screen_output_color[buffer_index] = rgba_u32(255u, 0u, 0u, 255u);
        screen_output_color[buffer_index] = payload.color; // rgba_u32(u32(distance(result.origin, result.intersection_point) / 300.0), 0u, 0u, 255u);//  result.diffuse_color;
    //}
    // let buffer_index = screen_to_index(screen_coord); 
    // // let buffer_index = global_id.x; 
    // screen_output[buffer_index] = result;
    // // screen_output_color[buffer_index] = rgba_u32(0u, 255u, 0u, 255u);//  result.diffuse_color;
    // screen_output_color[buffer_index] = result.diffuse_color; // rgba_u32(u32(distance(result.origin, result.intersection_point) / 300.0), 0u, 0u, 255u);//  result.diffuse_color;
}
