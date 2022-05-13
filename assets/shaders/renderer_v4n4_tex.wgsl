struct Camera {
    u_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

struct Light {
    light_pos: vec3<f32>,
    material_shininess: f32,
    material_spec_color: vec3<f32>,
    ambient_coeffience: f32,
    light_color: vec3<f32>,
    attentuation_factor: f32,
};

struct RenderParams {
    scale_factor: f32,
};

@group(0) @binding(0)
var<uniform> camerauniform: Camera;

@group(0) @binding(1)
var<uniform> light: Light;

@group(0) @binding(2)
var<uniform> other_params: RenderParams;

// Textures.

@group(1)
@binding(0)
var t_diffuse1: texture_2d<f32>;

@group(1)
@binding(1)
var s_diffuse1: sampler;

@group(1)
@binding(2)
var t_diffuse2: texture_2d<f32>;

@group(1)
@binding(3)
var s_diffuse2: sampler;

fn decode_color(c: u32) -> vec4<f32> {
  let a: f32 = f32(c & 256u) / 255.0;
  let b: f32 = f32((c & 65280u) >> 8u) / 255.0;
  let g: f32 = f32((c & 16711680u) >> 16u) / 255.0;
  let r: f32 = f32((c & 4278190080u) >> 24u) / 255.0;
  return vec4<f32>(r,g,b,a);
}

fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), vec4<f32>(step(c.b, c.g)));
    let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), vec4<f32>(step(p.x, c.r)));

    let d: f32 = q.x - min(q.w, q.y);
    let e: f32 = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv2rgb(c: vec3<f32>) -> vec3<f32>  {
    //vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let K: vec4<f32> = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    //vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    let p: vec3<f32> = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.yyy);
    //return vec3<f32>(1.0, 2.0, 3.0);
}

struct VertexOutput {
    @builtin(position) my_pos: vec4<f32>,
    @location(0) pos: vec4<f32>,
    @location(1) nor: vec4<f32>,
    @location(2) @interpolate(flat) col: vec3<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec4<f32>, @location(1) nor: vec4<f32>) -> VertexOutput {
    // var out_data: VertexOutput;
    // out_data.my_pos = camerauniform.u_view_proj * pos;
    // out_data.pos = pos;
    // out_data.nor = nor;
    // return out_data;
    let color: vec3<f32> = decode_color(bitcast<u32>(pos.w)).xyz;


    var out_next_stage: VertexOutput;
    out_next_stage.my_pos = camerauniform.u_view_proj * vec4<f32>(other_params.scale_factor * pos.xyz, 1.0);
    out_next_stage.pos = pos;
    out_next_stage.nor = nor;
    //++ out.diff_coeffient = diff_coeffient;
    //++ out.reflection_vector = reflection_vector;
    //++ out.camera_dir = camera_dir;
    out_next_stage.col = color;
    return out_next_stage;
}

// Ligth/material properties.
//++ let light_pos: vec3<f32> = vec3<f32>(50.0, 110.0, 53.0);
//++ let light_color: vec3<f32> = vec3<f32>(0.8, 0.3, 0.3);
//++ let material_spec_color: vec3<f32> = vec3<f32>(0.5, 0.1, 0.1);
//++ let material_shininess: f32 = 55.0;
//++ let ambient_coeffience: f32 = 0.15;
//++ let attentuation_factor: f32 = 0.0013;

// let light_pos: vec3<f32> = vec3<f32>(10.0, 10.0, 13.0);
// let light_color: vec3<f32> = vec3<f32>(0.8, 0.3, 0.3);
// let material_spec_color: vec3<f32> = vec3<f32>(0.5, 0.1, 0.1);
// let material_shininess: f32 = 55.0;
// let ambient_coeffience: f32 = 0.15;
// let attentuation_factor: f32 = 0.000013;

@fragment
fn fs_main(in_data: VertexOutput) -> @location(0) vec4<f32> {
    var light_dir: vec3<f32> = normalize(light.light_pos - in_data.pos.xyz);
    var normal: vec3<f32> = normalize(in_data.nor).xyz; // is this necessery? 
    var diff_coeffient: f32 = max(0.0, dot(normal, light_dir));
    var reflection_vector: vec3<f32> = reflect(-light_dir, normal);
    var camera_dir: vec3<f32> = normalize(camerauniform.camera_pos.xyz - in_data.pos.xyz);
    
    var cosAngle: f32 = max(0.0, dot(camera_dir, reflection_vector));
    var specular_coeffient: f32 = 0.0;

    if (diff_coeffient > 0.0) {
        specular_coeffient = pow(cosAngle, light.material_shininess);
    }

    var offset_factor: f32 = 0.5;
    
    var coord1: vec2<f32> = in_data.pos.xy*offset_factor * 2.1;
    var coord2: vec2<f32> = in_data.pos.xz*offset_factor * 2.1;
    var coord3: vec2<f32> = (in_data.pos.yz*offset_factor + in_data.pos.xz*offset_factor*offset_factor) * 2.1;
    
    var surfaceColor_grass: vec3<f32> = textureSample(t_diffuse1, s_diffuse1, offset_factor * (coord1 + coord3) / 59.0).xyz;
    var surfaceColor_rock:  vec3<f32>  = textureSample(t_diffuse2, s_diffuse2, 1.1 * (coord1 + coord2 - coord3) / 13.0).xyz;
    var surface_color: vec3<f32> = mix(
        surfaceColor_rock, surfaceColor_grass,
        vec3<f32>(clamp(0.4*in_data.nor.x + 0.6*in_data.nor.y, 0.0, 1.0)));

    var specular_component: vec3<f32> = specular_coeffient * light.material_spec_color * light.light_color;
    var ambient_component:  vec3<f32> = light.ambient_coeffience * light.light_color * surface_color.xyz;
    var diffuse_component:  vec3<f32> = diff_coeffient * light.light_color * surface_color.xyz;
    
    var distance_to_light: f32 = distance(in_data.pos.xyz, light.light_pos); 
    var attentuation: f32 = 1.0 / (1.0 + light.attentuation_factor * pow(distance_to_light,2.0));
    
    var final_color: vec4<f32> = vec4<f32>(ambient_component + attentuation * (diffuse_component + specular_component) , 1.0);

    return final_color;

    // var light_dir: vec3<f32> = normalize(light_pos - in_data.pos.xyz);
    // var normal: vec3<f32> = normalize(in_data.nor).xyz; // is this necessery? 
    // var diff_coeffient: f32 = max(0.0, dot(normal, light_dir));
    // var reflection_vector: vec3<f32> = reflect(-light_dir, normal);
    // var camera_dir: vec3<f32> = normalize(camerauniform.camera_pos.xyz - in_data.pos.xyz);
    // 
    // var cosAngle: f32 = max(0.0, dot(camera_dir, reflection_vector));
    // var specular_coeffient: f32 = 0.0;

    // if (diff_coeffient > 0.0) {
    //     specular_coeffient = pow(cosAngle, material_shininess);
    // }

    // var offset_factor: f32 = 0.5;
    // 
    // var coord1: vec2<f32> = in_data.pos.xy*offset_factor;
    // var coord2: vec2<f32> = in_data.pos.xz*offset_factor;
    // var coord3: vec2<f32> = in_data.pos.yz*offset_factor + in_data.pos.xz*offset_factor*offset_factor;
    // 
    // var surfaceColor_grass: vec3<f32> = textureSample(t_diffuse1, s_diffuse1, offset_factor * (coord1 + coord3) / 59.0).xyz;
    // var surfaceColor_rock:  vec3<f32>  = textureSample(t_diffuse2, s_diffuse2, 1.1 * (coord1 + coord2 - coord3) / 13.0).xyz;
    // var surface_color: vec3<f32> = mix(
    //     surfaceColor_rock, surfaceColor_grass,
    //     vec3<f32>(clamp(0.4*in_data.nor.x + 0.6*in_data.nor.y, 0.0, 1.0)));

    // var specular_component: vec3<f32> = specular_coeffient * material_spec_color * light_color;
    // var ambient_component:  vec3<f32> = ambient_coeffience * light_color * surface_color.xyz;
    // var diffuse_component:  vec3<f32> = diff_coeffient * light_color * surface_color.xyz;
    // 
    // var distance_to_light: f32 = distance(in_data.pos.xyz, light_pos); 
    // var attentuation: f32 = 1.0 / (1.0 + attentuation_factor * pow(distance_to_light,2.0));
    // 
    // var fin_dataal_color: vec4<f32> = vec4<f32>(ambient_component + attentuation * (diffuse_component + specular_component) , 1.0);

    // return fin_dataal_color;

    // // var the_color: vec3<f32> = rgb2hsv(final_color.xyz);

    // // var dist_to_frag: f32 = distance(camerauniform.camera_pos.xyz, in.pos.xyz);
    // // var blah: f32 = 1.0 / (1.0 + 0.0005 * pow(dist_to_frag,1.1));

    // // the_color = the_color + vec3<f32>(0.0, 0.2, 0.0);
    // // var the_color2: vec3<f32> = hsv2rgb(the_color);

    // // return vec4<f32>(mix(vec3<f32>(0.5, 0.0, 0.0), the_color2, vec3<f32>(blah)), 1.0);
}
