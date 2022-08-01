// Debugging.

let FAR      = 0u;
let BAND_NEW = 1u;
let BAND     = 2u;
let KNOWN    = 3u;
let OUTSIDE  = 4u;

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
    render_rays: u32,
    render_samplers: u32,
};

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
var<private> this_aabb: AABB;
var<private> private_global_index: vec3<u32>;
var<private> private_workgroup_index: vec3<u32>;
var<private> private_local_index: vec3<u32>;
var<private> step_counter: u32;

fn total_cell_count() -> u32 {

    return fmm_params.global_dimension.x * 
           fmm_params.global_dimension.y * 
           fmm_params.global_dimension.z * 
           fmm_params.local_dimension.x * 
           fmm_params.local_dimension.y * 
           fmm_params.local_dimension.z; 
};

// fn mapRange(a1: f32, a2: f32, b1: f32, b2: f32, s: f32) -> f32 {
//     return b1 + (s - a1) * (b2 - b1) / (a2 - a1);
// }

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

/// A function that checks if a given coordinate is within the global computational domain. 
fn isInside(coord: vec3<i32>) -> bool {
    return (coord.x >= 0 && coord.x < i32(fmm_params.local_dimension.x * fmm_params.global_dimension.x)) &&
           (coord.y >= 0 && coord.y < i32(fmm_params.local_dimension.y * fmm_params.global_dimension.y)) &&
           (coord.z >= 0 && coord.z < i32(fmm_params.local_dimension.z * fmm_params.global_dimension.z));
}

/// A function that checks if a given coordinate is within the global computational domain. 
fn isInside_f32(coord: vec3<f32>) -> bool {
    return (coord.x >= 0.0 && coord.x < f32(fmm_params.local_dimension.x * fmm_params.global_dimension.x) - 1.0) &&
           (coord.y >= 0.0 && coord.y < f32(fmm_params.local_dimension.y * fmm_params.global_dimension.y) - 1.0) &&
           (coord.z >= 0.0 && coord.z < f32(fmm_params.local_dimension.z * fmm_params.global_dimension.z) - 1.0); 
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

fn screen_to_index(v: vec2<u32>) -> u32 {
    let stride = sphere_tracer_params.inner_dim.x *
                 sphere_tracer_params.outer_dim.x;
    let index = v.x + v.y * stride;
    return index;
}

fn diffuse(ray: ptr<function, Ray>, payload: ptr<function, RayPayload>) {


    let light_pos = vec3<f32>(
                        f32(fmm_params.global_dimension.x * fmm_params.local_dimension.x),
                        f32(fmm_params.global_dimension.y * fmm_params.local_dimension.y),
                        f32(fmm_params.global_dimension.z * fmm_params.local_dimension.z)) * 0.5 +
			vec3<f32>(0.0, f32(fmm_params.global_dimension.y * fmm_params.local_dimension.y), 0.0);


    // let light_pos = vec3<f32>(150.0,70.0,150.0);
    //let light_pos = camera.pos; // vec3<f32>(150.0,70.0,150.0);
    let lightColor = vec3<f32>(1.0,1.0,1.0);
    let lightPower = 1000.0;
    
    // Material properties
    let materialDiffuseColor = decode_color((*payload).color).xyz; //vec3<f32>(1.0, 0.0, 0.0);
    let materialAmbientColor = vec3<f32>(0.4,0.4,0.4) * materialDiffuseColor;
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

fn load_trilinear_neighbors(coord: vec3<u32>, render: bool) -> array<u32, 8> {

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

    if (sphere_tracer_params.render_samplers == 1u && render && (private_local_index.x == 0u) && (private_workgroup_index.x % 44u) == 0u) {
    //if (sphere_tracer_params.render_samplers == 1u && (private_global_index.x == total_cell_count() / 2u)) {
    // if (render && (((private_global_index.x + 32u) & 2047u) == 0u)) {
    //if (render && ((private_global_index.x & 127u) == 0u)) {
        output_aabb_wire[atomicAdd(&counter[3], 1u)] =  
              AABB (
                  vec4<f32>(vec3<f32>(neighbors[0]) * 4.0,
                            bitcast<f32>(rgba_u32(255u, 0u, 1550u, 255u))),
                  vec4<f32>(vec3<f32>(neighbors[7]) * 4.0,
                            0.1)
        );
    }

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
        select(tcc ,i6, isInside(neighbors[6])),
        select(tcc ,i7, isInside(neighbors[7]))
    );
}

fn load_neighbors_private(p: vec3<u32>, render: bool) {

   private_neighbors_loc = load_trilinear_neighbors(p, render);

   private_neighbors[0] = fmm_data[private_neighbors_loc[0]];
   private_neighbors[1] = fmm_data[private_neighbors_loc[1]];
   private_neighbors[2] = fmm_data[private_neighbors_loc[2]];
   private_neighbors[3] = fmm_data[private_neighbors_loc[3]];
   private_neighbors[4] = fmm_data[private_neighbors_loc[4]];
   private_neighbors[5] = fmm_data[private_neighbors_loc[5]];
   private_neighbors[6] = fmm_data[private_neighbors_loc[6]];
   private_neighbors[7] = fmm_data[private_neighbors_loc[7]];
}

fn fmm_color_nearest(p: vec3<f32>) -> u32 {
    return private_neighbors[0].color;
}

fn fmm_color(p: vec3<f32>) -> u32 {

   //++ var point_counter = 0u;
   //++ let base_point = vec3<i32>(floor(p));

   //++ let tcc = total_cell_count();
   //++ var weight = 0.0;
   //++ var val = vec4<f32>(0.0, 0.0, 0.0, 0.0);

   //++ //for (var i:u32 = 0u ; i < 64u ; i = i + 1u) {
   //++ for (var i:u32 = 0u ; i < 32u ; i = i + 1u) {
   //++     
   //++     let temp_point = vec3<i32>(index_to_uvec3(i, 3u, 3u)) - vec3<i32>(1, 1, 1) + base_point;
   //++     //let temp_point = vec3<i32>(index_to_uvec3(i, 4u, 4u)) - vec3<i32>(1, 1, 1) + base_point;
   //++     //let dist = pow(distance(vec3<f32>(temp_point), p), 2.0); 
   //++     let dist = distance(vec3<f32>(temp_point), p); 
   //++     if (dist < 1.41421356237 && isInside(temp_point)) {
   //++     //if (dist < 2.0 && isInside(temp_point)) {
   //++     //if (dist < 1.0) {
   //++     //if (dist < 2.0 && isInside(temp_point)) {
   //++         //let t_w = abs(dist - 1.0);

   //++         let t_w = pow(abs(dist - 1.41421356237), 2.0);
   //++         //let t_w = abs(dist - 2.0) / 2.0;
   //++         //let t_w = abs(dist - 1.0);
   //++         let mem = get_cell_mem_location(vec3<u32>(temp_point));
   //++         let fmm_cell= fmm_data[mem];
   //++         if (fmm_cell.color == 0u) { continue; }
   //++         val = val + t_w * decode_color(fmm_cell.color);
   //++         weight = weight + t_w;

   //++         if (private_global_index.x == 0u) {
   //++             output_arrow[atomicAdd(&counter[1], 1u)] =  
   //++                   Arrow (
   //++                       vec4<f32>(p * 4.0, 0.0),
   //++                       vec4<f32>(vec3<f32>(temp_point) * 4.0, 0.0),
   //++                       //vec4<f32>(tMins, 0.0),
   //++                       //vec4<f32>(tMaxes, 0.0),
   //++                       fmm_cell.color,
   //++                       0.1
   //++             );
   //++         }
   //++     }
   //++ }

   //++ var final_color = val / weight;

   //++ return vec4_to_rgba(vec4<f32>(final_color.xyz, 1.0));  //val / weight);

   //++++ var neighbors: array<vec3<i32>, 8> = array<vec3<i32>, 8>(
   //++++     vec3<i32>(floor(p)) + vec3<i32>(0,  0,  0),
   //++++     vec3<i32>(floor(p)) + vec3<i32>(1,  0,  0),
   //++++     vec3<i32>(floor(p)) + vec3<i32>(0,  1,  0),
   //++++     vec3<i32>(floor(p)) + vec3<i32>(1,  1,  0),
   //++++     vec3<i32>(floor(p)) + vec3<i32>(0,  0,  1),
   //++++     vec3<i32>(floor(p)) + vec3<i32>(1,  0,  1),
   //++++     vec3<i32>(floor(p)) + vec3<i32>(0,  1,  1),
   //++++     vec3<i32>(floor(p)) + vec3<i32>(1,  1,  1),
   //++++ );

   //++++ let w0 = 1.0 / distance(p, vec3<f32>(neighbors[0])); 
   //++++ let w1 = 1.0 / distance(p, vec3<f32>(neighbors[1])); 
   //++++ let w2 = 1.0 / distance(p, vec3<f32>(neighbors[2])); 
   //++++ let w3 = 1.0 / distance(p, vec3<f32>(neighbors[3])); 
   //++++ let w4 = 1.0 / distance(p, vec3<f32>(neighbors[4])); 
   //++++ let w5 = 1.0 / distance(p, vec3<f32>(neighbors[5])); 
   //++++ let w6 = 1.0 / distance(p, vec3<f32>(neighbors[6])); 
   //++++ let w7 = 1.0 / distance(p, vec3<f32>(neighbors[7])); 

   //++++ let factor0 = select(1.0, 0.0, private_neighbors[0].tag == OUTSIDE || private_neighbors[0].color == 0u);
   //++++ let factor1 = select(1.0, 0.0, private_neighbors[1].tag == OUTSIDE || private_neighbors[1].color == 0u);
   //++++ let factor2 = select(1.0, 0.0, private_neighbors[2].tag == OUTSIDE || private_neighbors[2].color == 0u);
   //++++ let factor3 = select(1.0, 0.0, private_neighbors[3].tag == OUTSIDE || private_neighbors[3].color == 0u);
   //++++ let factor4 = select(1.0, 0.0, private_neighbors[4].tag == OUTSIDE || private_neighbors[4].color == 0u);
   //++++ let factor5 = select(1.0, 0.0, private_neighbors[5].tag == OUTSIDE || private_neighbors[5].color == 0u);
   //++++ let factor6 = select(1.0, 0.0, private_neighbors[6].tag == OUTSIDE || private_neighbors[6].color == 0u);
   //++++ let factor7 = select(1.0, 0.0, private_neighbors[7].tag == OUTSIDE || private_neighbors[7].color == 0u);

   //++++ let c0 = decode_color(private_neighbors[0].color);
   //++++ let c1 = decode_color(private_neighbors[1].color); 
   //++++ let c2 = decode_color(private_neighbors[2].color);
   //++++ let c3 = decode_color(private_neighbors[3].color); 
   //++++ let c4 = decode_color(private_neighbors[4].color);
   //++++ let c5 = decode_color(private_neighbors[5].color); 
   //++++ let c6 = decode_color(private_neighbors[6].color); 
   //++++ let c7 = decode_color(private_neighbors[7].color); 

   //++++ var c = c0 * w0 * factor0 + // * factor0 +
   //++++         c1 * w1 * factor1 + // * factor1 +
   //++++         c2 * w2 * factor2 + // * factor2 +
   //++++         c3 * w3 * factor3 + // * factor3 +
   //++++         c4 * w4 * factor4 + // * factor4 +
   //++++         c5 * w5 * factor5 + // * factor5 +
   //++++         c7 * w6 * factor6 + // * factor6 +
   //++++         c7 * w7 * factor7;  // * factor7 ;

   //++++ c = c / (w0 * factor0 + w1 * factor1 + w2 * factor2 + w3 * factor3 + w4 * factor4 + w5 * factor5 + w6 * factor6 + w7 * factor7); // Zero? 

   //++++ // let x = 1.0/6.0 * f32(c0.x + c1.x + c2.x + c3.x + c4.x + c5.x); 
   //++++ // let y = 1.0/6.0 * f32(c0.y + c1.y + c2.y + c3.y + c4.y + c5.y); 
   //++++ // let z = 1.0/6.0 * f32(c0.z + c1.z + c2.z + c3.z + c4.z + c5.z); 
   //++++ c.w = 1.0;
   //++++ // var final_color = rgba_u32(u32(min(255.0, c.x)), u32(min(255.0, c.y)), u32(min(255.0, c.z)), 255u);
   //++++ return vec4_to_rgba(c);
   // var c001 = decode_color(private_neighbors[0].color);
   // var c101 = decode_color(private_neighbors[1].color);
   // var c011 = decode_color(private_neighbors[2].color);
   // var c111 = decode_color(private_neighbors[3].color);
   // var c000 = decode_color(private_neighbors[4].color);
   // var c100 = decode_color(private_neighbors[5].color);
   // var c010 = decode_color(private_neighbors[6].color);
   // var c110 = decode_color(private_neighbors[7].color);

   // let max_value = max(c000,  
   //                 max(c100, 
   //                 max(c010, 
   //                 max(c110, 
   //                 max(c001, 
   //                 max(c101, 
   //                 max(c011, 
   //                     c111))))))); 

   // c000 = select(c000, max_value, private_neighbors[0].tag == FAR || private_neighbors[0].color == 0u);
   // c100 = select(c100, max_value, private_neighbors[1].tag == FAR || private_neighbors[1].color == 0u);
   // c010 = select(c010, max_value, private_neighbors[2].tag == FAR || private_neighbors[2].color == 0u);
   // c110 = select(c110, max_value, private_neighbors[3].tag == FAR || private_neighbors[3].color == 0u);
   // c001 = select(c001, max_value, private_neighbors[4].tag == FAR || private_neighbors[4].color == 0u);
   // c101 = select(c101, max_value, private_neighbors[5].tag == FAR || private_neighbors[5].color == 0u);
   // c011 = select(c011, max_value, private_neighbors[6].tag == FAR || private_neighbors[6].color == 0u);
   // c111 = select(c111, max_value, private_neighbors[7].tag == FAR || private_neighbors[7].color == 0u);
   
   // c000 = select(c000, max_value, private_neighbors[4].tag == FAR || private_neighbors[4].color == 0u);
   // c100 = select(c100, max_value, private_neighbors[5].tag == FAR || private_neighbors[5].color == 0u);
   // c010 = select(c010, max_value, private_neighbors[6].tag == FAR || private_neighbors[6].color == 0u);
   // c110 = select(c110, max_value, private_neighbors[7].tag == FAR || private_neighbors[7].color == 0u);
   // c001 = select(c001, max_value, private_neighbors[0].tag == FAR || private_neighbors[0].color == 0u);
   // c101 = select(c101, max_value, private_neighbors[1].tag == FAR || private_neighbors[5].color == 0u);
   // c011 = select(c011, max_value, private_neighbors[2].tag == FAR || private_neighbors[6].color == 0u);
   // c111 = select(c111, max_value, private_neighbors[3].tag == FAR || private_neighbors[7].color == 0u);

   // var c000_factor = select(1.0, 0.0, c000.x == 0.0 && c000.y == 0.0 && c000.z == 0.0);
   // var c100_factor = select(1.0, 0.0, c100.x == 0.0 && c100.y == 0.0 && c100.z == 0.0);
   // var c010_factor = select(1.0, 0.0, c010.x == 0.0 && c010.y == 0.0 && c010.z == 0.0);
   // var c110_factor = select(1.0, 0.0, c110.x == 0.0 && c110.y == 0.0 && c110.z == 0.0);
   // var c001_factor = select(1.0, 0.0, c001.x == 0.0 && c001.y == 0.0 && c001.z == 0.0);
   // var c101_factor = select(1.0, 0.0, c101.x == 0.0 && c101.y == 0.0 && c101.z == 0.0);
   // var c011_factor = select(1.0, 0.0, c011.x == 0.0 && c011.y == 0.0 && c011.z == 0.0);
   // var c111_factor = select(1.0, 0.0, c111.x == 0.0 && c111.y == 0.0 && c111.z == 0.0);

   let tx = fract(p.x);
   let ty = fract(p.y);
   let tz = fract(p.z);

   let loc0 = get_cell_index(private_neighbors_loc[0]);
   let loc1 = get_cell_index(private_neighbors_loc[1]);
   let loc2 = get_cell_index(private_neighbors_loc[2]);
   let loc3 = get_cell_index(private_neighbors_loc[3]);
   let loc4 = get_cell_index(private_neighbors_loc[4]);
   let loc5 = get_cell_index(private_neighbors_loc[5]);
   let loc6 = get_cell_index(private_neighbors_loc[6]);
   let loc7 = get_cell_index(private_neighbors_loc[7]);

   let c000 = decode_color(private_neighbors[0].color);
   let c100 = decode_color(private_neighbors[1].color); 
   let c010 = decode_color(private_neighbors[2].color);
   let c110 = decode_color(private_neighbors[3].color); 
   let c001 = decode_color(private_neighbors[4].color);
   let c101 = decode_color(private_neighbors[5].color); 
   let c011 = decode_color(private_neighbors[6].color); 
   let c111 = decode_color(private_neighbors[7].color); 

   //        c010 : (0,1,0) : 2       c110 : (1,1,0) 3 
   //         +------------------------+
   //        /|                       /|
   //       / |                      / |
   //      /  |                     /  |
   //     /   |                    /   |
   //    /    |                   /    |
   //   +------------------------+c111 |
   //   |c011 (0, 1, 0) 6        | (1,1,1) 7
   //   |     |                  |     |
   //   |     |                  |     |
   //   |     +------------------|-----+ c100 : (1,0,0) 1 
   //   |    / c000 : (0,0,0) 0  |    /
   //   |   /                    |   /
   //   |  /                     |  /
   //   | /                      | /
   //   |/                       |/
   //   +------------------------+
   //  c001 : (0, 0, 1) 4       c101 : (1, 0, 1) 5 

   let gamma = 1.2;

   let c00 = pow(c000 * (1.0 - tx) + c100 * tx, vec4<f32>(1.0 / gamma)); 
   let c01 = pow(c010 * (1.0 - tx) + c110 * tx, vec4<f32>(1.0 / gamma)); 
   let c10 = pow(c001 * (1.0 - tx) + c101 * tx, vec4<f32>(1.0 / gamma)); 
   let c11 = pow(c011 * (1.0 - tx) + c111 * tx, vec4<f32>(1.0 / gamma)); 

   let c0 = pow(c00 * (1.0 - ty) + c01 * ty, vec4<f32>(1.0 / gamma));
   let c1 = pow(c10 * (1.0 - ty) + c11 * ty, vec4<f32>(1.0 / gamma));

   //let c = c1 * (1.0 - tz) + c0 * tz; 
   let c = pow(c0 * (1.0 - tz) + c1 * tz, vec4<f32>(1.0 / gamma)); 

   let size = 0.1;

   //if ((private_global_index.x & 2047u) == 0u) {
   //if (sphere_tracer_params.render_samplers == 1u && (private_global_index.x == total_cell_count() / 2u)) {
   if (sphere_tracer_params.render_samplers == 1u && (private_local_index.x == 0u) && (private_workgroup_index.x % 44u) == 0u) {
   //if (((private_global_index.x + 32u) & 2047u) == 0u) {

   output_arrow[atomicAdd(&counter[1], 1u)] =  
         Arrow (
             vec4<f32>(p * 4.0, 0.0),
             vec4<f32>(vec3<f32>(loc0) * 4.0, 0.0),
    	     vec4_to_rgba(c000),
             size
   );
   output_arrow[atomicAdd(&counter[1], 1u)] =  
         Arrow (
             vec4<f32>(p * 4.0, 0.0),
             vec4<f32>(vec3<f32>(loc1) * 4.0, 0.0),
    	     vec4_to_rgba(c100),
             size
   );
   output_arrow[atomicAdd(&counter[1], 1u)] =  
         Arrow (
             vec4<f32>(p * 4.0, 0.0),
             vec4<f32>(vec3<f32>(loc2) * 4.0, 0.0),
    	     vec4_to_rgba(c010),
             size
   );
   output_arrow[atomicAdd(&counter[1], 1u)] =  
         Arrow (
             vec4<f32>(p * 4.0, 0.0),
             vec4<f32>(vec3<f32>(loc3) * 4.0, 0.0),
    	     vec4_to_rgba(c110),
             size
   );
   output_arrow[atomicAdd(&counter[1], 1u)] =  
         Arrow (
             vec4<f32>(p * 4.0, 0.0),
             vec4<f32>(vec3<f32>(loc4) * 4.0, 0.0),
    	     vec4_to_rgba(c001),
             size
   );
   output_arrow[atomicAdd(&counter[1], 1u)] =  
         Arrow (
             vec4<f32>(p * 4.0, 0.0),
             vec4<f32>(vec3<f32>(loc5) * 4.0, 0.0),
    	     vec4_to_rgba(c101),
             size
   );
   output_arrow[atomicAdd(&counter[1], 1u)] =  
         Arrow (
             vec4<f32>(p * 4.0, 0.0),
             vec4<f32>(vec3<f32>(loc6) * 4.0, 0.0),
    	     vec4_to_rgba(c011),
             size
   );
   output_arrow[atomicAdd(&counter[1], 1u)] =  
         Arrow (
             vec4<f32>(p * 4.0, 0.0),
             vec4<f32>(vec3<f32>(loc7) * 4.0, 0.0),
    	     vec4_to_rgba(c111),
             size
   );
   output_aabb_wire[atomicAdd(&counter[3], 1u)] =  
         AABB (
             vec4<f32>(vec3<f32>(loc0) * 4.0,
                       bitcast<f32>(rgba_u32(255u, 0u, 1550u, 255u))),
             vec4<f32>(vec3<f32>(loc7) * 4.0,
                       0.1)
   );
   }

   //++++let c010 = decode_color(private_neighbors[0].color);
   //++++let c110 = decode_color(private_neighbors[1].color); 
   //++++let c011 = decode_color(private_neighbors[2].color);
   //++++let c111 = decode_color(private_neighbors[3].color); 
   //++++let c000 = decode_color(private_neighbors[4].color);
   //++++let c100 = decode_color(private_neighbors[5].color); 
   //++++let c001 = decode_color(private_neighbors[6].color); 
   //++++let c101 = decode_color(private_neighbors[7].color); 

   //++ var c = c0 * d0 * factor0 + // * factor0 +
   //++         c1 * d1 * factor1 + // * factor1 +
   //++         c2 * d2 * factor2 + // * factor2 +
   //++         c3 * d3 * factor3 + // * factor3 +
   //++         c4 * d4 * factor4 + // * factor4 +
   //++         c5 * d5 * factor5 + // * factor5 +
   //++         c7 * d6 * factor6 + // * factor6 +
   //++         c7 * d7 * factor7;  // * factor7 ;

   //++ c = c / (d0 * factor0 + d1 * factor1 + d2 * factor2 + d3 * factor3 + d4 * factor4 + d5 * factor5 + d6 * factor6 + d7 * factor7); // Zero? 

   //++ // let x = 1.0/6.0 * f32(c0.x + c1.x + c2.x + c3.x + c4.x + c5.x); 
   //++ // let y = 1.0/6.0 * f32(c0.y + c1.y + c2.y + c3.y + c4.y + c5.y); 
   //++ // let z = 1.0/6.0 * f32(c0.z + c1.z + c2.z + c3.z + c4.z + c5.z); 
   //++ c.w = 1.0;
   //++ // var final_color = rgba_u32(u32(min(255.0, c.x)), u32(min(255.0, c.y)), u32(min(255.0, c.z)), 255u);
   //++ return vec4_to_rgba(c);

   // var color = (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * c000 + 
   //        tx * (1.0 - ty) * (1.0 - tz) * c100 + 
   //        (1.0 - tx) * ty * (1.0 - tz) * c010 + 
   //        tx * ty * (1.0 - tz) * c110 + 
   //        (1.0 - tx) * (1.0 - ty) * tz * c001 + 
   //        tx * (1.0 - ty) * tz * c101 + 
   //        (1.0 - tx) * ty * tz * c011 + 
   //        tx * ty * tz * c111;


   // var color = (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * c000 + 
   //        tx * (1.0 - ty) * (1.0 - tz) * c100 + 
   //        (1.0 - tx) * ty * (1.0 - tz) * c010 + 
   //        tx * ty * (1.0 - tz) * c110 + 
   //        (1.0 - tx) * (1.0 - ty) * tz * c001 + 
   //        tx * (1.0 - ty) * tz * c101 + 
   //        (1.0 - tx) * ty * tz * c011 + 
   //        tx * ty * tz * c111;

   // var color = (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * c000 * c000_factor + 
   //        tx * (1.0 - ty) * (1.0 - tz) * c100 * c100_factor + 
   //        (1.0 - tx) * ty * (1.0 - tz) * c010 * c010_factor + 
   //        tx * ty * (1.0 - tz) * c110 * c110_factor + 
   //        (1.0 - tx) * (1.0 - ty) * tz * c001 * c001_factor + 
   //        tx * (1.0 - ty) * tz * c101 * c101_factor + 
   //        (1.0 - tx) * ty * tz * c011 * c011_factor + 
   //        tx * ty * tz * c111 * c111_factor;
   return vec4_to_rgba(vec4<f32>(c.x, c.y, c.z, 1.0));

   //++ // var color = (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * decode_color(private_neighbors[0].color) * c000_factor + 
   //++ //        tx * (1.0 - ty) * (1.0 - tz) * decode_color(private_neighbors[1].color) * c100_factor + 
   //++ //        (1.0 - tx) * ty * (1.0 - tz) * decode_color(private_neighbors[2].color) * c010_factor + 
   //++ //        tx * ty * (1.0 - tz) * decode_color(private_neighbors[3].color) * c110_factor + 
   //++ //        (1.0 - tx) * (1.0 - ty) * tz * decode_color(private_neighbors[4].color) * c001_factor + 
   //++ //        tx * (1.0 - ty) * tz * decode_color(private_neighbors[5].color) * c101_factor + 
   //++ //        (1.0 - tx) * ty * tz * decode_color(private_neighbors[6].color) * c011_factor + 
   //++ //        tx * ty * tz * decode_color(private_neighbors[7].color) * c111_factor;
}


// fn fmm_color_6(p: vec3<f32>) -> u32 {
// 
//     var neighbors: array<vec3<i32>, 6> = array<vec3<i32>, 6>(
//         vec3<i32>(p) + vec3<i32>(1,  0,  0),
//         vec3<i32>(p) - vec3<i32>(1,  0,  0),
//         vec3<i32>(p) + vec3<i32>(0,  1,  0),
//         vec3<i32>(p) - vec3<i32>(0,  1,  0),
//         vec3<i32>(p) + vec3<i32>(0,  0,  1),
//         vec3<i32>(p) - vec3<i32>(0,  0,  1),
//     );
// 
//    load_neighbors_private(vec3<u32>(neighbors[0]), false);
//    let c0 = decode_color(fmm_color(p + vec3<f32>(1.0, 0.0, 0.0))); 
// 
//    load_neighbors_private(vec3<u32>(neighbors[1]), false);
//    let c1 = decode_color(fmm_color(p + vec3<f32>(-1.0, 0.0, 0.0))); 
// 
//    load_neighbors_private(vec3<u32>(neighbors[2]), false);
//    let c2 = decode_color(fmm_color(p + vec3<f32>(0.0, 1.0, 0.0))); 
// 
//    load_neighbors_private(vec3<u32>(neighbors[3]), false);
//    let c3 = decode_color(fmm_color(p + vec3<f32>(0.0, -1.0, 0.0))); 
// 
//    load_neighbors_private(vec3<u32>(neighbors[4]), false);
//    let c4 = decode_color(fmm_color(p + vec3<f32>(0.0, 0.0, 1.0))); 
// 
//    load_neighbors_private(vec3<u32>(neighbors[5]), false);
//    let c5 = decode_color(fmm_color(p + vec3<f32>(0.0, 0.0, -1.0))); 
// 
//    let x = 1.0/6.0 * f32(c0.x + c1.x + c2.x + c3.x + c4.x + c5.x); 
//    let y = 1.0/6.0 * f32(c0.y + c1.y + c2.y + c3.y + c4.y + c5.y); 
//    let z = 1.0/6.0 * f32(c0.z + c1.z + c2.z + c3.z + c4.z + c5.z); 
// 
//    return rgba_u32(u32(min(255.0, x)),
//                    u32(min(255.0, y)),
// 		   u32(min(255.0, z)),
// 		   255u);
// }


fn fmm_value(p: vec3<f32>, render: bool) -> f32 {

   load_neighbors_private(vec3<u32>(floor(p)), render);

   let tx = fract(p.x);
   let ty = fract(p.y);
   let tz = fract(p.z);

   // let c000 = private_neighbors[0].value;
   // let c100 = private_neighbors[1].value; 
   // let c010 = private_neighbors[2].value;
   // let c110 = private_neighbors[3].value; 
   // let c001 = private_neighbors[4].value;
   // let c101 = private_neighbors[5].value; 
   // let c011 = private_neighbors[6].value; 
   // let c111 = private_neighbors[7].value; 

   // let c00 = c000 * (1.0 - tx) + c100 * tx; 
   // let c01 = c001 * (1.0 - tx) + c101 * tx; 
   // let c10 = c010 * (1.0 - tx) + c110 * tx; 
   // let c11 = c011 * (1.0 - tx) + c111 * tx; 

   // let c0 = c00 * (1.0 - ty) + c10 * ty;
   // let c1 = c01 * (1.0 - ty) + c11 * ty;

   // let c = c0 * (1.0 - tz) + c1 * tz; 

   // return c;

   let c000 = private_neighbors[0].value;
   let c100 = private_neighbors[1].value; 
   let c010 = private_neighbors[2].value;
   let c110 = private_neighbors[3].value; 
   let c001 = private_neighbors[4].value;
   let c101 = private_neighbors[5].value; 
   let c011 = private_neighbors[6].value; 
   let c111 = private_neighbors[7].value; 

   let c00 = c000 * (1.0 - tx) + c100 * tx; 
   let c01 = c010 * (1.0 - tx) + c110 * tx; 
   let c10 = c001 * (1.0 - tx) + c101 * tx; 
   let c11 = c011 * (1.0 - tx) + c111 * tx; 

   let c0 = c00 * (1.0 - ty) + c01 * ty;
   let c1 = c10 * (1.0 - ty) + c11 * ty;

   //let c = c1 * (1.0 - tz) + c0 * tz; 
   let c = c0 * (1.0 - tz) + c1 * tz; 

   return c;

   //++++let min_value = min(private_neighbors[0].value,  
   //++++                min(private_neighbors[1].value, 
   //++++                min(private_neighbors[2].value, 
   //++++                min(private_neighbors[3].value, 
   //++++                min(private_neighbors[4].value, 
   //++++                min(private_neighbors[5].value, 
   //++++                min(private_neighbors[6].value, 
   //++++                    private_neighbors[7].value))))))); 

   //++++var c000 = select(private_neighbors[0].value, min_value, private_neighbors[0].tag == FAR);
   //++++var c100 = select(private_neighbors[1].value, min_value, private_neighbors[1].tag == FAR);
   //++++var c010 = select(private_neighbors[2].value, min_value, private_neighbors[2].tag == FAR);
   //++++var c110 = select(private_neighbors[3].value, min_value, private_neighbors[3].tag == FAR);
   //++++var c001 = select(private_neighbors[4].value, min_value, private_neighbors[4].tag == FAR);
   //++++var c101 = select(private_neighbors[5].value, min_value, private_neighbors[5].tag == FAR);
   //++++var c011 = select(private_neighbors[6].value, min_value, private_neighbors[6].tag == FAR);
   //++++var c111 = select(private_neighbors[7].value, min_value, private_neighbors[7].tag == FAR);

   //++++// let c = mapRange(0.0, 
   //++++//                  64.0,
   //++++//     	     0.0,
   //++++//     	     255.0, 
   //++++//     	     f32(actual_index)
   //++++// );

   //++++let c = rgba_u32(min(step_counter, 255u),
   //++++                 0u,
   //++++                 255u - min(step_counter, 255u),
   //++++     	    255u);

   //++++// Draw the current point on ray and arrows to the neighbor cells. 
   //++++// if ((private_global_index.x & 1023u) == 0u) {
   //++++//     output_aabb[atomicAdd(&counter[2], 1u)] =  
   //++++//           AABB (
   //++++//               vec4<f32>(pah.x - 0.3,
   //++++//                         pah.y - 0.3,
   //++++//                         pah.z - 0.3,
   //++++//                         bitcast<f32>(c)),
   //++++//               vec4<f32>(pah.x + 0.3,
   //++++//                         pah.y + 0.3,
   //++++//                         pah.z + 0.3,
   //++++//                         0.0),
   //++++//     );

   //++++//     let known_color = rgba_u32(155u, 0u, 0u, 255u);
   //++++//     let outside_color = rgba_u32(0u, 0u, 255u, 255u);

   //++++//     for (var i: i32 =  0 ; i<8 ; i = i + 1) {
   //++++//         let col = select(known_color, outside_color, private_neighbors[i].tag == OUTSIDE); 
   //++++//         let size = select(0.1, 20.0, private_neighbors[i].tag == OUTSIDE); 
   //++++//         output_arrow[atomicAdd(&counter[1], 1u)] =  
   //++++//               Arrow (
   //++++//                   vec4<f32>(pah, 0.0),
   //++++//                   vec4<f32>(vec3<f32>(get_cell_index(private_neighbors_loc[i])) * 4.0, 0.0),
   //++++//          	     col,
   //++++//                   size
   //++++//         );
   //++++//     }
   //++++// }

   //++++// let x_value = x_part * n0 + (1.0 - x_part) * n1;  
   //++++// let y_value = y_part * n2 + (1.0 - y_part) * n1;  
   //++++// let z_value = z_part * n0 + (1.0 - z_part) * n1;  

   //++++return (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * c000 + 
   //++++       tx * (1.0 - ty) * (1.0 - tz) * c100 + 
   //++++       (1.0 - tx) * ty * (1.0 - tz) * c010 + 
   //++++       tx * ty * (1.0 - tz) * c110 + 
   //++++       (1.0 - tx) * (1.0 - ty) * tz * c001 + 
   //++++       tx * (1.0 - ty) * tz * c101 + 
   //++++       (1.0 - tx) * ty * tz * c011 + 
   //++++       tx * ty * tz * c111; 

 
   //++ var point_counter = 0u;
   //++ let base_point = vec3<i32>(floor(p));

   //++ let tcc = total_cell_count();
   //++ var weight = 0.0;
   //++ var val = 0.0;

   //++ // for (var i:u32 = 0u ; i < 64u ; i = i + 1u) {
   //++ for (var i:u32 = 0u ; i < 27u ; i = i + 1u) {
   //++     
   //++     let temp_point = vec3<i32>(index_to_uvec3(i, 3u, 3u)) - vec3<i32>(1, 1, 1) + base_point;
   //++     //let temp_point = vec3<i32>(index_to_uvec3(i, 4u, 4u)) - vec3<i32>(1, 1, 1) + base_point;
   //++     let dist = distance(vec3<f32>(temp_point), p); 
   //++     //if (dist < 1.41421356237 && isInside(temp_point)) {
   //++     if (dist < 1.0 && isInside(temp_point)) {
   //++     //if (dist < 2.0 && isInside(temp_point)) {
   //++         let t_w = abs(dist - 1.0);
   //++         //let t_w = abs(dist - 1.41421356237);
   //++         let mem = get_cell_mem_location(vec3<u32>(temp_point));
   //++         let fmm_cell= fmm_data[mem];
   //++         val = val + t_w * fmm_cell.value;
   //++         weight = weight + t_w;

   //++         // if (private_global_index.x == 0u) {
   //++         //     output_arrow[atomicAdd(&counter[1], 1u)] =  
   //++         //           Arrow (
   //++         //               vec4<f32>(p * 4.0, 0.0),
   //++         //               vec4<f32>(vec3<f32>(temp_point) * 4.0, 0.0),
   //++         //               //vec4<f32>(tMins, 0.0),
   //++         //               //vec4<f32>(tMaxes, 0.0),
   //++         //               rgba_u32(155u, 0u, 1550u, 255u),
   //++         //               0.1
   //++         //     );
   //++         // }
   //++     }
   //++ }

   //++ return val / weight;

   // var neighbors: array<vec3<i32>, 8> = array<vec3<i32>, 8>(
   //     vec3<i32>(floor(p)) + vec3<i32>(0,  0,  0),
   //     vec3<i32>(floor(p)) + vec3<i32>(1,  0,  0),
   //     vec3<i32>(floor(p)) + vec3<i32>(0,  1,  0),
   //     vec3<i32>(floor(p)) + vec3<i32>(1,  1,  0),
   //     vec3<i32>(floor(p)) + vec3<i32>(0,  0,  1),
   //     vec3<i32>(floor(p)) + vec3<i32>(1,  0,  1),
   //     vec3<i32>(floor(p)) + vec3<i32>(0,  1,  1),
   //     vec3<i32>(floor(p)) + vec3<i32>(1,  1,  1),
   // );

   // let w0 = distance(p, vec3<f32>(neighbors[0])); 
   // let w1 = distance(p, vec3<f32>(neighbors[1])); 
   // let w2 = distance(p, vec3<f32>(neighbors[2])); 
   // let w3 = distance(p, vec3<f32>(neighbors[3])); 
   // let w4 = distance(p, vec3<f32>(neighbors[4])); 
   // let w5 = distance(p, vec3<f32>(neighbors[5])); 
   // let w6 = distance(p, vec3<f32>(neighbors[6])); 
   // let w7 = distance(p, vec3<f32>(neighbors[7])); 

   // let factor0 = select(1.0, 0.0, private_neighbors[0].tag == OUTSIDE);
   // let factor1 = select(1.0, 0.0, private_neighbors[1].tag == OUTSIDE);
   // let factor2 = select(1.0, 0.0, private_neighbors[2].tag == OUTSIDE);
   // let factor3 = select(1.0, 0.0, private_neighbors[3].tag == OUTSIDE);
   // let factor4 = select(1.0, 0.0, private_neighbors[4].tag == OUTSIDE);
   // let factor5 = select(1.0, 0.0, private_neighbors[5].tag == OUTSIDE);
   // let factor6 = select(1.0, 0.0, private_neighbors[6].tag == OUTSIDE);
   // let factor7 = select(1.0, 0.0, private_neighbors[7].tag == OUTSIDE);

   //++ var w = private_neighbors[0].value * (sqrt(2.0) - w0) * factor0 + // * factor0 +
   //++         private_neighbors[1].value * (sqrt(2.0) - w1) * factor1 + // * factor1 +
   //++         private_neighbors[2].value * (sqrt(2.0) - w2) * factor2 + // * factor2 +
   //++         private_neighbors[3].value * (sqrt(2.0) - w3) * factor3 + // * factor3 +
   //++         private_neighbors[4].value * (sqrt(2.0) - w4) * factor4 + // * factor4 +
   //++         private_neighbors[5].value * (sqrt(2.0) - w5) * factor5 + // * factor5 +
   //++         private_neighbors[6].value * (sqrt(2.0) - w6) * factor6 + // * factor6 +
   //++         private_neighbors[7].value * (sqrt(2.0) - w7) * factor7;  // * factor7 ;

   //++ return w / (w0 * factor0 + w1 * factor1 + w2 * factor2 + w3 * factor3 + w4 * factor4 + w5 * factor5 + w6 * factor6 + w7 * factor7); // Zero? 

   //return (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * private_neighbors[0].value + // * c000_factor + 
   //       tx * (1.0 - ty) * (1.0 - tz) * private_neighbors[1].value + // * c100_factor+ 
   //       (1.0 - tx) * ty * (1.0 - tz) * private_neighbors[2].value + // * c010_factor+ 
   //       tx * ty * (1.0 - tz) * private_neighbors[3].value + // * c110_factor+ 
   //       (1.0 - tx) * (1.0 - ty) * tz *  private_neighbors[4].value + // * c001_factor+ 
   //       tx * (1.0 - ty) * tz * private_neighbors[5].value + // * c101_factor+ 
   //       (1.0 - tx) * ty * tz * private_neighbors[6].value + // * c011_factor+ 
   //       tx * ty * tz * private_neighbors[7].value; // * c111_factor; 
   // return (1.0 - tx) * (1.0 - ty) * (1.0 - tz) * private_neighbors[0].value * factor0 + // * c000_factor + 
   //        tx * (1.0 - ty) * (1.0 - tz) * private_neighbors[1].value * factor1 + // * c100_factor+ 
   //        (1.0 - tx) * ty * (1.0 - tz) * private_neighbors[2].value * factor2 + // * c010_factor+ 
   //        tx * ty * (1.0 - tz) * private_neighbors[3].value * factor3 + // * c110_factor+ 
   //        (1.0 - tx) * (1.0 - ty) * tz *  private_neighbors[4].value * factor4 + // * c001_factor+ 
   //        tx * (1.0 - ty) * tz * private_neighbors[5].value * factor5 + // * c101_factor+ 
   //        (1.0 - tx) * ty * tz * private_neighbors[6].value * factor6 + // * c011_factor+ 
   //        tx * ty * tz * private_neighbors[7].value * factor7; // * c111_factor; 
}

/// Calculate normal using fmm neighbors.
fn calculate_normal(payload: ptr<function, RayPayload>) -> vec3<f32> {

    var grad: vec3<f32>;

    var pos = (*payload).intersection_point;
    var offset = 0.01;
    // var right = sdBox(vec3(pos.x+offset, pos.y,pos.z), vec3<f32>(50.0, 50.0, 50.0));
    // var left = sdBox(vec3(pos.x-offset, pos.y,pos.z), vec3<f32>(50.0, 50.0, 50.0));
    // var up = sdBox(vec3(pos.x, pos.y+offset,pos.z), vec3<f32>(50.0, 50.0, 50.0));
    // var down = sdBox(vec3(pos.x, pos.y-offset,pos.z), vec3<f32>(50.0, 50.0, 50.0));
    // var z_minus = sdBox(vec3(pos.x, pos.y,pos.z-offset), vec3<f32>(50.0, 50.0, 50.0));
    // var z = sdBox(vec3(pos.x, pos.y,pos.z+offset), vec3<f32>(50.0, 50.0, 50.0));
    var right = fmm_value(vec3(pos.x+offset, pos.y,pos.z), false);
    var left = fmm_value(vec3(pos.x-offset, pos.y,pos.z), false);
    var up = fmm_value(vec3(pos.x, pos.y+offset,pos.z), false);
    var down = fmm_value(vec3(pos.x, pos.y-offset,pos.z), false);
    var z_minus = fmm_value(vec3(pos.x, pos.y,pos.z-offset), false);
    var z = fmm_value(vec3(pos.x, pos.y,pos.z+offset), false);
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
    return normalize(grad);
}

/// Function tha is called on ray hit. TODO: separate gradien calculation to another function.
fn hit(ray: ptr<function, Ray>, payload: ptr<function, RayPayload>) {

    fmm_value((*payload).intersection_point, true);
    //(*payload).color = fmm_color_6((*payload).intersection_point);
    (*payload).color = fmm_color((*payload).intersection_point);
    //(*payload).color = fmm_color_nearest((*payload).intersection_point);
    (*payload).normal = calculate_normal(payload);
    (*payload).visibility = 1.0;
    diffuse(ray, payload);
}

/// Checks if a given value is inside the fmm domain.
fn fmm_is_outside_value(v: vec3<f32>) -> bool {
    if (v.x < 0.0 ||
        v.y < 0.0 ||
        v.z < 0.0 ||
        v.x > f32(fmm_params.global_dimension.x * fmm_params.local_dimension.x - 1u) ||
        v.y > f32(fmm_params.global_dimension.y * fmm_params.local_dimension.y - 1u) ||
        v.z > f32(fmm_params.global_dimension.z * fmm_params.local_dimension.z - 1u)) {
            return true;
	}
     return false;
}

/// Ray aabb intersection. @p0 and @p1 are aabb box corners (p0 < p1).
/// @t_min and @t_max are the given interval where collision are allowed.
/// @t_near and @t_far are pointers to the intersection distances which are
/// computed in this function.
/// return true if ray hits the aabb, false othewise.
fn aabb_intersect(p0: ptr<function, vec3<f32>>,
                  p1: ptr<function, vec3<f32>>,
		  ray: ptr<function, Ray>,
		  t_min: ptr<function, vec3<f32>>, 
		  t_max: ptr<function, vec3<f32>>,
                  t_near: ptr<function, f32>,
		  t_far: ptr<function, f32>) -> bool {

    let invRayDir = vec3<f32>(1.0) / (*ray).direction;
    let t_lower = (*p0 - (*ray).origin) * invRayDir;
    let t_upper = (*p1 - (*ray).origin) * invRayDir;
    var t_mi =  min(t_lower, t_upper); 
    var t_ma = max(t_lower, t_upper); 
    var tMins =  max(t_mi, *t_min); 
    var tMaxes = min(t_ma, *t_max); 

    *t_near = select(t_mi.y, t_mi.x, t_mi.x > t_mi.y);
    *t_near = select(t_mi.z, *t_near, *t_near > t_mi.z);

    *t_far = select(t_ma.y, t_ma.x, t_ma.x < t_ma.y);
    *t_far = select(t_ma.z, *t_far, *t_far < t_ma.z);

    // Draw arrow between the intersection points.
    // output_arrow[atomicAdd(&counter[1], 1u)] =  
    //       Arrow (
    //           vec4<f32>(getPoint(t_near, ray) * 4.0, 0.0),
    //           vec4<f32>(getPoint(t_far, ray) * 4.0, 0.0),
    //           //vec4<f32>(tMins, 0.0),
    //           //vec4<f32>(tMaxes, 0.0),
    //           rgba_u32(155u, 0u, 1550u, 255u),
    //           1.0
    // );

     return tMins.x <= tMaxes.x && tMins.y <= tMaxes.y && tMins.z <= tMaxes.z;
}


/// Sphere tracing method.
fn traceRay(ray: ptr<function, Ray>, payload: ptr<function, RayPayload>) {

    var dist = (*ray).rMin;

    // Chech the intersection.
    var p0 = vec3<f32>(0.0, 0.0, 0.0);
    var p1 = vec3<f32>(fmm_params.global_dimension * fmm_params.local_dimension - 1u);
    var p_near = vec3<f32>(0.0, 0.0, 0.0);
    var p_far = vec3<f32>(10000.0, 10000.0, 10000.0);
    var t_near: f32;
    var t_far: f32;
    let intersects = aabb_intersect(&p0, &p1, ray, &p_near, &p_far, &t_near, &t_far);
    (*ray).rMin = t_near; // TODO: now useless.
    (*ray).rMax = t_far;
    // if ((private_global_index.x & 63u) == 0u && intersects) {
    //     // let result = boxIntersection(&ray, vec3<f32>(fmm_params.global_dimension * fmm_params.local_dimension));
    //     output_arrow[atomicAdd(&counter[1], 1u)] =  
    //           Arrow (
    //               vec4<f32>(getPoint(t_near, ray) * 4.0, 0.0),
    //               vec4<f32>(getPoint(t_far, ray) * 4.0, 0.0),
    //               //vec4<f32>(result.intersection_point, 0.0),
    //               rgba_u32(255u, 255u, 2550u, 255u),
    //               0.2
    //     );
    //     //var p0 = vec3<f32>(0.0);
    //     //var p1 = vec3<f32>(fmm_params.global_dimension * fmm_params.local_dimension);
    //     //var t_min = vec3<f32>(-1000.0, -1000.0, -1000.0);
    //     //var t_max = vec3<f32>(1000.0, 1000.0, 1000.0);

    //     //aabb_intersect(&p0, &p1, &ray, &t_min, &t_max);
    // }

    // TODO: this is not a bug. Check the fmm sampling code.
    if (!(isInside_f32((*ray).origin)) && intersects) {
        dist = t_near + 0.1;
        var temp_p = getPoint(dist, ray);
        if (fmm_is_outside_value(temp_p)) {
            dist = t_near - 0.1;
        }
    }

    // if (!intersects && private_global_index.x == 0u) {
    //     output_aabb[atomicAdd(&counter[2], 1u)] =  
    //           AABB (
    //               vec4<f32>(4.0 * p0.x - 23.0,
    //                         4.0 * p0.y - 23.0,
    //                         4.0 * p0.z - 23.0,
    //                         f32(rgba_u32(0u, 0u, 255u, 255u))),
    //               vec4<f32>(4.0 * p0.x + 23.0,
    //                         4.0 * p0.y + 23.0,
    //                         4.0 * p0.z + 23.0,
    //                         0.0));
    //     dist = 400.1;
    //     return;
    //     }

    var p: vec3<f32>;
    var distance_to_interface: f32;

    while (dist < (*ray).rMax && step_counter < 800u) {
        p = getPoint(dist, ray);

        if (fmm_is_outside_value(p)) {
        //     if ( private_global_index.x == 0u) {

        //         output_aabb[atomicAdd(&counter[2], 1u)] =  
        //               AABB (
        //                   vec4<f32>(4.0 * p.x - 23.0,
        //                             4.0 * p.y - 23.0,
        //                             4.0 * p.z - 23.0,
        //                             f32(rgba_u32(0u, 255u, 0u, 255u))),
        //                   vec4<f32>(4.0 * p.x + 23.0,
        //                             4.0 * p.y + 23.0,
        //                             4.0 * p.z + 23.0,
        //                             0.0),
        //         );
        //     }
             return;
        }
        distance_to_interface = fmm_value(p, false); // max(min(0.01 * dist, 0.2), 0.001);
        if (abs(distance_to_interface) < 0.03) {

	    // var temp_distance = dist;
	    //var temp_p = p;
            //var nearest_color = fmm_color_nearest(p);
            //var black = rgba_u32(0u, 0u, 0u, 255u);
	    //for (var b: i32 = 1; b < 10 ; b = b + 1) {
	    //    p = temp_p;
	    //    if (nearest_color != black) { break; } 
            //    load_neighbors_private(vec3<u32>(temp_p));
	    //    temp_distance = temp_distance + 0.01;
            //    temp_p = getPoint(temp_distance, ray);
            //    nearest_color = fmm_color_nearest(temp_p);
            //}
	    //dist = temp_distance;
	    p = getPoint(dist, ray);
            (*payload).intersection_point = p;
            hit(ray, payload);
            return;
        }
        step_counter = step_counter + 1u;
        dist = dist + distance_to_interface;
    } // while

    (*payload).intersection_point = p;
}

// fn box_intersect(aabb_min: vec<f32>, aabb_max: vec<f32>, result: ptr<function, f32>, canStartInBox: bool) -> bool {
//     
// }


// fn create_pinhole_ray(pixel: vec2<f32>) -> Ray {
//     let tan_half_angle = tan(camera.fov_angle / 2.0);
//     let screen_width = sphere_tracer_params.inner_dim.x * sphere_tracer_params.outer_dim.x; 
//     let screen_height = sphere_tracer_params.inner_dim.y * sphere_tracer_params.outer_dim.y; 
//     let aspect_scale = select(screen_height, screen_width, (camera_fov_direction == 0u)); 
//     let dir = normalize(vec3<f32>(vec2<f32>(pixel.x, -pixel.y) * tanHalfAngle / aspect_scale, -1.0);
//     return Ray(dir, 0.0, 
// }

@compute
@workgroup_size(64,1,1)
fn main(@builtin(local_invocation_id)    local_id: vec3<u32>,
        @builtin(local_invocation_index) local_index: u32,
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(global_invocation_id)   global_id: vec3<u32>) {

    // this_aabb = AABB(
    //     vec4<f32>(0.0, 0.0, 0.0, 1.0),
    //     vec4<f32>(fmm_params.global_dimension * fmm_params.inner_dimension, 1.0)

    // );

    private_global_index = global_id;
    private_workgroup_index = workgroup_id;
    private_local_index = local_id;
    step_counter = 0u;

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
    //ray.origin = point_on_plane + camera.pos.xyz * 0.25;
    ray.origin = point_on_plane + camera.pos.xyz;
    ray.direction = normalize(point_on_plane + d*camera.view.xyz);
    ray.rMin = 0.0;
    ray.rMax = 800.0;

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

    if (global_id.x == 0u) {
        
        let focal_point = camera.pos - camera.view * d; 

        output_aabb[atomicAdd(&counter[2], 1u)] =  
              AABB (
                  vec4<f32>(focal_point * 4.0 - vec3<f32>(2.2), f32(rgba_u32(255u, 0u, 2550u, 255u))),
                  vec4<f32>(focal_point * 4.0 + vec3<f32>(2.2), 0.0),
        );
        output_arrow[atomicAdd(&counter[1], 1u)] =  
              Arrow (
                  vec4<f32>(focal_point * 4.0, 0.0),
                  vec4<f32>(focal_point * 4.0 + camera.view * 19.0, 0.0),
                  //vec4<f32>(result.intersection_point, 0.0),
                  rgba_u32(255u, 0u, 2550u, 255u),
                  1.0
        );
        // let renderable_element = Char (
        //                 // element_position,
        //                 vec3<f32>(focal_point.x - 1.11, focal_point.y + 1.11, focal_point.z + 1.11),
        //                 0.2,
        //                 vec4<f32>(camera.pos * 4.0, 0.0),
        //                 //vec4<f32>(0.25 * camera.pos.x, 0.25 * camera.pos.y, 0.25 * camera.pos.z, 0.0),
        //                 //vec4<f32>(vec3<f32>(fmm_params.local_dimension), 6.0),
        //                 4u,
        //                 rgba_u32(255u, 255u, 255u, 255u),
        //                 1u,
        //                 0u
        // );
        // output_char[atomicAdd(&counter[0], 1u)] = renderable_element; 
    }

    //if ((global_id.x & 2047u) == 0u) {
    if (sphere_tracer_params.render_rays == 1u && (local_index == 0u) && (workgroup_id.x % 44u) == 0u) {
    //if (sphere_tracer_params.render_samplers == 1u && (private_global_index.x == total_cell_count() / 2u)) {
    //++ if (payload.visibility > 0.0) {
        output_arrow[atomicAdd(&counter[1], 1u)] =  
              Arrow (
                  vec4<f32>(ray.origin * 4.0, 0.0),
                  vec4<f32>(payload.intersection_point * 4.0, 0.0),
                  //vec4<f32>(result.intersection_point, 0.0),
                  rgba_u32_argb(payload.color),
                  0.1
        );

        // output_arrow[atomicAdd(&counter[1], 1u)] =  
        //       Arrow (
        //           vec4<f32>(payload.intersection_point, 0.0),
        //           vec4<f32>(payload.intersection_point + payload.normal * 3.0, 0.0),
        //           //vec4<f32>(result.intersection_point, 0.0),
        //           rgba_u32(155u, 0u, 1550u, 255u),
        //           0.1
        // );
    }
    if (global_id.x == 0u) {
	let aabbmin = vec3<f32>(0.0, 0.0, 0.0);
	let aabbmax = vec3<f32>(fmm_params.global_dimension * fmm_params.local_dimension);
        output_aabb_wire[atomicAdd(&counter[3], 1u)] =  
              AABB (
                  vec4<f32>(aabbmin * 4.0,
                            f32(rgba_u32(255u, 0u, 2550u, 255u))),
                  vec4<f32>(aabbmax * 4.0,
                            0.5)
        );
    }

    // if (global_id.x == 0u) {
    //     // let result = boxIntersection(&ray, vec3<f32>(fmm_params.global_dimension * fmm_params.local_dimension));
    //     let aabb_sizes = vec3<f32>(fmm_params.global_dimension * fmm_params.local_dimension) * 0.5;
    //     let result = boxIntersection(ray.origin - aabb_sizes, ray.direction, aabb_sizes);
    //     output_arrow[atomicAdd(&counter[1], 1u)] =  
    //           Arrow (
    //               vec4<f32>(getPoint(result[0], &ray) * 4.0, 0.0),
    //               vec4<f32>(getPoint(result[1], &ray) * 4.0, 0.0),
    //               //vec4<f32>(result.intersection_point, 0.0),
    //               rgba_u32(255u, 255u, 2550u, 255u),
    //               0.2
    //     );
    //     //var p0 = vec3<f32>(0.0);
    //     //var p1 = vec3<f32>(fmm_params.global_dimension * fmm_params.local_dimension);
    //     //var t_min = vec3<f32>(-1000.0, -1000.0, -1000.0);
    //     //var t_max = vec3<f32>(1000.0, 1000.0, 1000.0);

    //     //aabb_intersect(&p0, &p1, &ray, &t_min, &t_max);
    // }
        
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
