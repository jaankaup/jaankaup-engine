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

@group(0) @binding(0) var<uniform>            fmm_params: FmmParams;
@group(0) @binding(1) var<uniform>            sphere_tracer_params: SphereTracerParams;
@group(0) @binding(2) var<uniform>            camera: RayCamera;
@group(0) @binding(3) var<storage,read_write> fmm_data: array<FmmCellPc>;
@group(0) @binding(4) var<storage,read_write> screen_output: array<RayOutput>;
@group(0) @binding(5) var<storage,read_write> screen_output_color: array<u32>;

@group(1) @binding(0) var<storage,read_write> counter: array<atomic<u32>>;
@group(1) @binding(1) var<storage,read_write> output_char: array<Char>;
@group(1) @binding(2) var<storage,read_write> output_arrow: array<Arrow>;
@group(1) @binding(3) var<storage,read_write> output_aabb: array<AABB>;
@group(1) @binding(4) var<storage,read_write> output_aabb_wire: array<AABB>;
// Push constants.
// var<push_constant> pc: PushConstants;

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

fn load_neighbors_6(coord: vec3<u32>) -> array<u32, 6> {

    var neighbors: array<vec3<i32>, 6> = array<vec3<i32>, 6>(
        vec3<i32>(coord) + vec3<i32>(-1,  0,  0),
        vec3<i32>(coord) + vec3<i32>(1,   0,  0),
        vec3<i32>(coord) + vec3<i32>(0,   1,  0),
        vec3<i32>(coord) + vec3<i32>(0,  -1,  0),
        vec3<i32>(coord) + vec3<i32>(0,   0,  1),
        vec3<i32>(coord) + vec3<i32>(0,   0, -1)
    );

    let i0 = get_cell_mem_location(vec3<u32>(neighbors[0]));
    let i1 = get_cell_mem_location(vec3<u32>(neighbors[1]));
    let i2 = get_cell_mem_location(vec3<u32>(neighbors[2]));
    let i3 = get_cell_mem_location(vec3<u32>(neighbors[3]));
    let i4 = get_cell_mem_location(vec3<u32>(neighbors[4]));
    let i5 = get_cell_mem_location(vec3<u32>(neighbors[5]));

    // The index of the "outside" cell.
    let tcc = total_cell_count();

    return array<u32, 6>(
        select(tcc ,i0, isInside(neighbors[0])),
        select(tcc ,i1, isInside(neighbors[1])),
        select(tcc ,i2, isInside(neighbors[2])),
        select(tcc ,i3, isInside(neighbors[3])),
        select(tcc ,i4, isInside(neighbors[4])),
        select(tcc ,i5, isInside(neighbors[5])) 
    );
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

    let stride = sphere_tracer_params.inner_dim.x * sphere_tracer_params.inner_dim.y;
    let global_coordinate = v / sphere_tracer_params.inner_dim;
    let global_index = (global_coordinate.x + global_coordinate.y * sphere_tracer_params.outer_dim.x);
    let local_coordinate = v - global_coordinate * sphere_tracer_params.inner_dim;
    let local_index = local_coordinate.x + local_coordinate.y * sphere_tracer_params.inner_dim.x;
    return global_index + local_index;
}

// Box (exact).
fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
  let q = abs(p) - b;
  return length(max(q,vec3<f32>(0.0))) + min(max(q.x,max(q.y,q.z)),0.0);
  //return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

// Calculate the point in the Ray direction.
fn getPoint(parameter: f32, ray: ptr<function, Ray>) -> vec3<f32> { return (*ray).origin + parameter * (*ray).direction; }

fn hit(payload: ptr<function, RayPayload>) {

    var grad: vec3<f32>;

    var pos = (*payload).intersection_point;
    var offset = 0.1;
    var right = sdBox(vec3(pos.x+offset, pos.y,pos.z), vec3<f32>(15.0, 15.0, 15.0));
    var left = sdBox(vec3(pos.x-offset, pos.y,pos.z), vec3<f32>(15.0, 15.0, 15.0));
    var up = sdBox(vec3(pos.x, pos.y+offset,pos.z), vec3<f32>(15.0, 15.0, 15.0));
    var down = sdBox(vec3(pos.x, pos.y-offset,pos.z), vec3<f32>(15.0, 15.0, 15.0));
    var z_minus = sdBox(vec3(pos.x, pos.y,pos.z-offset), vec3<f32>(15.0, 15.0, 15.0));
    var z = sdBox(vec3(pos.x, pos.y,pos.z+offset), vec3<f32>(15.0, 15.0, 15.0));
    grad.x = right - left;
    grad.y = up - down;
    grad.z = z - z_minus;
    var normal = normalize(grad);
    (*payload).color = rgba_u32(0u, 255u, 0u, 255u);
}
// ray intersection aabb

fn traceRay(ray: ptr<function, Ray>, payload: ptr<function, RayPayload>) {

  var dist = (*ray).rMin;
  var maxi = (*ray).rMax;

  let temp_offset = 0.1;
  var p: vec3<f32>;
  var distance_to_interface: f32;

  while (dist < (*ray).rMax) {
      p = getPoint(dist, ray);
      distance_to_interface = sdBox(p, vec3<f32>(15.0, 15.0, 15.0));
      if (abs(distance_to_interface) < 0.1) {
	  (*payload).intersection_point = p;
          hit(payload);
	  return;
      }
      dist = dist + distance_to_interface;

  //            // Step backward the ray.
  // 	     float temp_distance = dist;
  //            dist -= STEP_SIZE;
  // 	     float value;
  // 	     vec3 p;

  //            // Calculate more accurate intersection point.
  // 	     while (dist < temp_distance) {
  // 	       p = getPoint(dist, ray);
  //              //value = trilinear_density(p); //calculate_density(p);
  //              value = calculate_density(p);
  // 	       if (value < 0.0) break;
////            if (temp_distance > 200.0) {
////                temp_distance = temp_distance + STEP_SIZE;  break;
////            }
  // 	       //temp_distance += temp_offset;
  // 	       dist += temp_offset;
  // 	     }

  //         // Jump back a litte.
  //         dist -= temp_offset;

  //         // Save intersection point.
  //         payload.intersection_point = vec4(getPoint(dist, ray) , 1.0);

  //         // Calculate normal and the actual value. payload.normal == vec3(normal, value);
  //     //payload.normal = vec4(calculate_normal2(payload.intersection_point.xyz).xyz, 0.0);
  //     payload.normal = vec4(calculate_normal(payload.intersection_point.xyz), 0.0);

  //     // Calculate the colo for intersection.
  //     //++hit(ray,payload);
  //     return;

  //   } // if
  //    dist += STEP_SIZE;
  } // while

  //  miss(ray,payload);
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
    ray.origin = point_on_plane + camera.pos.xyz;
    ray.direction = normalize(point_on_plane + d*camera.view.xyz);
    ray.rMin = 0.0;
    ray.rMax = 300.0;

    var payload: RayPayload;
    payload.color = rgba_u32(255u, 0u, 0u, 255u);
    payload.visibility = 1.0;

    // traceRay(&ray, &payload);

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
                  vec4<f32>(focal_point.x - 0.1,
                            focal_point.y - 0.1,
                            focal_point.z - 0.1,
                            f32(rgba_u32(255u, 0u, 2550u, 255u))),
                  vec4<f32>(focal_point.x + 0.1,
                            focal_point.y + 0.1,
                            focal_point.z + 0.1,
                            0.0),
        );
        output_arrow[atomicAdd(&counter[1], 1u)] =  
              Arrow (
                  vec4<f32>(focal_point, 0.0),
                  vec4<f32>(focal_point + camera.view * 1.0, 0.0),
                  //vec4<f32>(result.intersection_point, 0.0),
                  rgba_u32(255u, 0u, 2550u, 255u),
                  0.01
        );
    }

    //if (global_id.x % 256u == 0u) {
        output_arrow[atomicAdd(&counter[1], 1u)] =  
              Arrow (
                  vec4<f32>(ray.origin, 0.0),
                  vec4<f32>(getPoint(50.0, &ray), 0.0),
                  //vec4<f32>(result.intersection_point, 0.0),
                  payload.color,
                  0.1
        );
    //}

    let buffer_index = global_id.x; // screen_to_index(screen_coord);
    screen_output[buffer_index] = result;
    // screen_output_color[buffer_index] = rgba_u32(0u, 255u, 0u, 255u);//  result.diffuse_color;
    screen_output_color[buffer_index] = rgba_u32(u32(distance(result.origin, result.intersection_point) / 300.0), 0u, 0u, 255u);//  result.diffuse_color;
}
