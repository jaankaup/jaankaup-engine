struct Camera {
    u_view_proj: mat4x4<f32>;
    camera_pos: vec4<f32>;
};

@group(0)
@binding(0)
var<uniform> camerauniform: Camera;

fn decode_color(c: u32) -> vec4<f32> {
  let a: f32 = f32(c & 256u) / 255.0;
  let b: f32 = f32((c & 65280u) >> 8u) / 255.0;
  let g: f32 = f32((c & 16711680u) >> 16u) / 255.0;
  let r: f32 = f32((c & 4278190080u) >> 24u) / 255.0;
  return vec4<f32>(r,g,b,a);
}

// fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
//     let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
//     let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), vec4<f32>(step(c.b, c.g)));
//     let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), vec4<f32>(step(p.x, c.r)));
// 
//     let d: f32 = q.x - min(q.w, q.y);
//     let e: f32 = 1.0e-10;
//     return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
// }
// 
// fn hsv2rgb(c: vec3<f32>) -> vec3<f32>  {
//     let K: vec4<f32> = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
//     let p: vec3<f32> = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
//     return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.yyy);
// }

struct VertexOutput {
    @builtin(position) my_pos: vec4<f32>;
    @location(0) pos: vec4<f32>;
    @location(1) nor: vec4<f32>;
    @location(2) diff_coeffient: f32;
    @location(3) reflection_vector: vec3<f32>;
    @location(4) camera_dir: vec3<f32>;
    @location(5) @interpolate(flat) col: vec3<f32>;
};

// Ligth/material properties.
let light_pos: vec3<f32> = vec3<f32>(20.0, 20.0, 20.0);
let light_color: vec3<f32> = vec3<f32>(0.8, 0.3, 0.3);
let material_spec_color: vec3<f32> = vec3<f32>(0.5, 0.1, 0.1);
let material_shininess: f32 = 55.0;
let ambient_coeffience: f32 = 0.15;
let attentuation_factor: f32 = 0.000013;

@stage(vertex)
fn vs_main(@location(0) pos: vec4<f32>, @location(1) nor: vec4<f32>) -> VertexOutput {

    let color: vec3<f32> = decode_color(u32(pos.w)).xyz;

    var light_dir: vec3<f32> = normalize(light_pos - pos.xyz);
    var diff_coeffient: f32 = max(0.0, dot(nor.xyz, light_dir));
    var reflection_vector: vec3<f32> = reflect(-light_dir, nor.xyz);
    var camera_dir: vec3<f32> = normalize(camerauniform.camera_pos.xyz - pos.xyz);

    var out: VertexOutput;
    out.my_pos = camerauniform.u_view_proj * vec4<f32>(pos.xyz, 1.0);
    out.pos = pos;
    out.nor = nor;
    out.diff_coeffient = diff_coeffient;
    out.reflection_vector = reflection_vector;
    out.camera_dir = camera_dir;
    out.col = color;
    return out;
}

@stage(fragment)
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    var cosAngle: f32 = max(0.0, dot(in.camera_dir, in.reflection_vector));
    var specular_coeffient: f32 = 0.0;

    if (in.diff_coeffient > 0.0) {
        specular_coeffient = pow(cosAngle, material_shininess);
    }

    var surface_color: vec3<f32> = in.col; 

    var specular_component: vec3<f32> = specular_coeffient * material_spec_color * light_color;
    var ambient_component:  vec3<f32> = ambient_coeffience * light_color * surface_color.xyz;
    var diffuse_component:  vec3<f32> = in.diff_coeffient * light_color * surface_color.xyz;
    
    var distance_to_light: f32 = distance(in.pos.xyz, light_pos); 
    var attentuation: f32 = 1.0 / (1.0 + attentuation_factor * pow(distance_to_light,2.0));
    
    var final_color: vec4<f32> = vec4<f32>(ambient_component + attentuation * (diffuse_component + specular_component) , 1.0);

    return final_color;
}
